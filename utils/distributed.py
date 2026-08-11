from datetime import timedelta
from contextlib import contextmanager
from functools import partial
import os
import signal
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullStateDictConfig, FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy, StateDictType
from torch.distributed.fsdp.api import CPUOffload
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy, transformer_auto_wrap_policy


_graceful_stop_requested = False


def _request_graceful_stop(signum, frame):
    del signum, frame
    global _graceful_stop_requested
    if not _graceful_stop_requested:
        print("Graceful stop requested; finishing the current step...", flush=True)
    _graceful_stop_requested = True


def graceful_stop_and_save(trainer):
    """Synchronize a stop request, save on every rank, and end the train loop."""
    requested = torch.tensor(
        int(_graceful_stop_requested), device=trainer.device, dtype=torch.int32
    )
    dist.all_reduce(requested, op=dist.ReduceOp.MAX)
    if not requested.item():
        return False

    if trainer.is_main_process:
        print(f"Stopping gracefully at step {trainer.step}.", flush=True)
    if not trainer.config.no_save:
        # A manual stop should save promptly instead of starting checkpoint eval.
        if hasattr(trainer.config, "checkpoint_eval_enabled"):
            trainer.config.checkpoint_eval_enabled = False
        torch.cuda.empty_cache()
        trainer.save()
        torch.cuda.empty_cache()
    barrier()
    return True


def fsdp_state_dict(model):
    fsdp_fullstate_save_policy = FullStateDictConfig(
        offload_to_cpu=True, rank0_only=True
    )
    with FSDP.state_dict_type(
        model, StateDictType.FULL_STATE_DICT, fsdp_fullstate_save_policy
    ):
        checkpoint = model.state_dict()

    return checkpoint


def _materialize_meta_module(module: torch.nn.Module) -> None:
    if any(param.is_meta for param in module.parameters(recurse=False)) or any(
        buffer.is_meta for buffer in module.buffers(recurse=False)
    ):
        module.to_empty(device=torch.cuda.current_device(), recurse=False)


def fsdp_wrap(module, sharding_strategy="full", mixed_precision=False, wrap_strategy="size", min_num_params=int(5e7), transformer_module=None, cpu_offload=False, sync_module_states=False):
    if mixed_precision:
        mixed_precision_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
            cast_forward_inputs=False
        )
    else:
        mixed_precision_policy = None

    if wrap_strategy == "transformer":
        auto_wrap_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=transformer_module
        )
    elif wrap_strategy == "size":
        auto_wrap_policy = partial(
            size_based_auto_wrap_policy,
            min_num_params=min_num_params
        )
    else:
        raise ValueError(f"Invalid wrap strategy: {wrap_strategy}")

    os.environ["NCCL_CROSS_NIC"] = "1"

    sharding_strategy = {
        "full": ShardingStrategy.FULL_SHARD,
        "hybrid_full": ShardingStrategy.HYBRID_SHARD,
        "hybrid_zero2": ShardingStrategy._HYBRID_SHARD_ZERO2,
        "no_shard": ShardingStrategy.NO_SHARD,
    }[sharding_strategy]

    module = FSDP(
        module,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=sharding_strategy,
        mixed_precision=mixed_precision_policy,
        device_id=torch.cuda.current_device(),
        param_init_fn=_materialize_meta_module if any(p.is_meta for p in module.parameters()) else None,
        limit_all_gathers=True,
        use_orig_params=True,
        cpu_offload=CPUOffload(offload_params=cpu_offload),
        sync_module_states=sync_module_states,
    )
    return module


def barrier():
    if dist.is_initialized():
        dist.barrier()


def launch_distributed_job(backend: str = "nccl"):
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    host = os.environ["MASTER_ADDR"]
    port = int(os.environ["MASTER_PORT"])

    if ":" in host:  # IPv6
        init_method = f"tcp://[{host}]:{port}"
    else:  # IPv4
        init_method = f"tcp://{host}:{port}"
    timeout_minutes = float(os.environ.get("DIST_TIMEOUT_MINUTES", "5"))
    dist.init_process_group(
        rank=rank,
        world_size=world_size,
        backend=backend,
        init_method=init_method,
        timeout=timedelta(minutes=timeout_minutes),
    )
    torch.cuda.set_device(local_rank)
    signal.signal(signal.SIGINT, _request_graceful_stop)


class EMA_FSDP:
    def __init__(self, fsdp_module: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        self._init_shadow(fsdp_module)

    @torch.no_grad()
    def _init_shadow(self, fsdp_module):
        with FSDP.summon_full_params(fsdp_module, writeback=False):
            for n, p in fsdp_module.module.named_parameters():
                self.shadow[n] = p.detach().clone().float().cpu()

    @torch.no_grad()
    def update(self, fsdp_module):
        d = self.decay
        with FSDP.summon_full_params(fsdp_module, writeback=False):
            for n, p in fsdp_module.module.named_parameters():
                self.shadow[n].mul_(d).add_(p.detach().float().cpu(), alpha=1. - d)

    # Optional helpers ---------------------------------------------------
    def state_dict(self):
        return self.shadow            # picklable

    def load_state_dict(self, sd):
        self.shadow = {k: v.clone() for k, v in sd.items()}

    def copy_to(self, fsdp_module):
        with FSDP.summon_full_params(fsdp_module, writeback=True):
            for n, p in fsdp_module.module.named_parameters():
                if n in self.shadow:
                    p.data.copy_(self.shadow[n].to(dtype=p.dtype, device=p.device))

    @contextmanager
    def swap_with(self, fsdp_module):
        """Temporarily run an FSDP module with EMA weights, then restore it."""
        self._swap_with(fsdp_module)
        try:
            yield
        finally:
            self._swap_with(fsdp_module)

    @torch.no_grad()
    def _swap_with(self, fsdp_module):
        with FSDP.summon_full_params(fsdp_module, writeback=True):
            for name, param in fsdp_module.module.named_parameters():
                live = param.detach().float().cpu().clone()
                param.copy_(self.shadow[name].to(device=param.device, dtype=param.dtype))
                self.shadow[name] = live

    @staticmethod
    def _clean_param_name(name: str) -> str:
        for token in (
            "_fsdp_wrapped_module.",
            "_checkpoint_wrapped_module.",
            "_orig_mod.",
        ):
            name = name.replace(token, "")
        return name

    @torch.no_grad()
    def full_state_dict(self, fsdp_module):
        checkpoint = fsdp_state_dict(fsdp_module)
        if not checkpoint:
            return checkpoint

        shadow = {}
        for name, tensor in self.shadow.items():
            shadow[name] = tensor
            shadow[self._clean_param_name(name)] = tensor

        for name, tensor in list(checkpoint.items()):
            ema_tensor = shadow.get(name)
            if ema_tensor is None:
                ema_tensor = shadow.get(self._clean_param_name(name))
            if ema_tensor is None:
                continue
            if ema_tensor.shape != tensor.shape:
                raise RuntimeError(
                    f"EMA shape mismatch for {name}: "
                    f"ema={tuple(ema_tensor.shape)} checkpoint={tuple(tensor.shape)}"
                )
            checkpoint[name] = ema_tensor.to(dtype=tensor.dtype)

        return checkpoint


class ShardedEMA_FSDP:
    """EMA over local FSDP shards; never materializes the full model."""

    def __init__(self, fsdp_module: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {
            name: param.detach().float().cpu().clone()
            for name, param in fsdp_module.named_parameters()
        }

    @torch.no_grad()
    def update(self, fsdp_module):
        for name, param in fsdp_module.named_parameters():
            self.shadow[name].mul_(self.decay).add_(
                param.detach().float().cpu(), alpha=1.0 - self.decay
            )

    @torch.no_grad()
    def copy_to(self, fsdp_module):
        for name, param in fsdp_module.named_parameters():
            param.copy_(self.shadow[name].to(device=param.device, dtype=param.dtype))
