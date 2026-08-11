import argparse
import gc
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _resolve_eval_raw_steps(num_steps: int) -> list[int]:
    if not 1 <= num_steps <= 1000:
        raise ValueError("--num-steps must be between 1 and 1000")
    if num_steps == 1:
        return [1000]
    return [
        round(1000 - i * (1000 - 1000 / num_steps) / (num_steps - 1))
        for i in range(num_steps)
    ]


def _resolve_checkpoint_path(path: str) -> Path:
    checkpoint_path = Path(path).expanduser().resolve()
    if checkpoint_path.is_dir():
        checkpoint_path = checkpoint_path / "model.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    return checkpoint_path


def _load_config(args) -> OmegaConf:
    config = OmegaConf.merge(
        OmegaConf.load(ROOT / "configs/default_config.yaml"),
        OmegaConf.load(args.config_path),
    )
    config.disable_wandb = True
    config.no_visualize = True
    config.wandb_save_dir = str(args.output_root.resolve() / "wandb")
    config.logdir = str(args.output_root.resolve() / "eval_logs")
    config.checkpoint_eval_denoising_step_list = _resolve_eval_raw_steps(args.num_steps)
    return config


def _resolve_architecture(args, config) -> str:
    architecture = args.arch or getattr(config, "student_arch", "causal")
    if architecture not in {"causal", "bidirectional"}:
        raise ValueError(f"Unsupported --arch: {architecture}")
    return architecture


def _load_generator_checkpoint(generator, checkpoint_path: Path) -> None:
    if checkpoint_path.suffix == ".safetensors":
        from safetensors.torch import load_file as load_safetensors

        state_dict = {
            f"model.{key}": value
            for key, value in load_safetensors(str(checkpoint_path), device="cpu").items()
        }
        missing, unexpected = generator.load_state_dict(state_dict, strict=False)
        if dist.get_rank() == 0:
            print(
                "Generator safetensors load_state_dict: "
                f"missing={len(missing)} unexpected={len(unexpected)}",
                flush=True,
            )
            if missing:
                print(f"  Missing sample: {missing[:5]}", flush=True)
            if unexpected:
                print(f"  Unexpected sample: {unexpected[:5]}", flush=True)
        del state_dict
        gc.collect()
        return

    from trainer.egosim_distillation import Trainer

    state_dict = torch.load(checkpoint_path, map_location="cpu", mmap=True)
    state_dict = Trainer._extract_generator_state_dict(state_dict)
    generator.load_state_dict(state_dict, strict=True)
    del state_dict
    gc.collect()


def _resolve_master_addr(args) -> str:
    if args.master_addr:
        return args.master_addr
    worker_hosts = os.environ.get("VC_WORKER_HOSTS", "")
    if worker_hosts:
        return worker_hosts.split(",", 1)[0]
    if args.nnodes > 1:
        raise RuntimeError("Multi-node eval requires MASTER_ADDR or VC_WORKER_HOSTS.")
    return "127.0.0.1"


def _run_launcher(args) -> int:
    master_addr = _resolve_master_addr(args)
    log_dir = args.output_root.resolve() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"eval_{time.strftime('%Y%m%d_%H%M%S')}_node{args.node_rank}.log"
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
    ]
    if args.nnodes == 1:
        cmd.extend(["--standalone", "--nproc_per_node", str(args.nproc_per_node)])
    else:
        cmd.extend(
            [
                "--nnodes",
                str(args.nnodes),
                "--node_rank",
                str(args.node_rank),
                "--nproc_per_node",
                str(args.nproc_per_node),
                "--rdzv_id",
                args.rdzv_id,
                "--rdzv_backend=static",
                "--rdzv_endpoint",
                f"{master_addr}:{args.master_port}",
            ]
        )

    cmd.extend([
        str(Path(__file__).resolve()),
        "--distributed-worker",
        "--config-path", str(args.config_path),
        "--checkpoint-path", str(args.checkpoint_path),
        "--output-root", str(args.output_root),
        "--num-steps", str(args.num_steps),
        "--step", str(args.step),
    ])
    for sample_id in args.sample_id:
        cmd.extend(["--sample-id", sample_id])
    if args.arch is not None:
        cmd.extend(["--arch", args.arch])
    print(f"[EvalLauncher] log={log_path}", flush=True)
    print("[EvalLauncher] " + " ".join(cmd), flush=True)
    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write("[EvalLauncher] " + " ".join(cmd) + "\n")
        log_file.flush()
        process = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            env=os.environ.copy(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
        return process.wait()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate an EgoSim checkpoint on the eval cache and compute PSNR/SSIM/LPIPS."
    )
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--num-steps", type=int, default=12)
    parser.add_argument("--step", type=int, default=0, help="Iteration label for output folder.")
    parser.add_argument("--arch", choices=("causal", "bidirectional"), default=None)
    parser.add_argument("--sample-id", action="append", default=[], help="Evaluate only this sample ID; repeat for multiple samples.")
    parser.add_argument("--master-port", type=int, default=int(os.environ.get("MASTER_PORT", "6062")), help=argparse.SUPPRESS)
    parser.add_argument("--rdzv-id", default=os.environ.get("RDZV_ID", os.environ.get("MA_JOB_NAME", "egosim_eval")), help=argparse.SUPPRESS)
    parser.add_argument("--nproc-per-node", type=int, default=int(os.environ.get("NPROC_PER_NODE", os.environ.get("MA_NUM_GPUS", "8"))), help=argparse.SUPPRESS)
    parser.add_argument("--nnodes", type=int, default=int(os.environ.get("NNODES", os.environ.get("MA_NUM_HOSTS", os.environ.get("VC_WORKER_NUM", "1")))), help=argparse.SUPPRESS)
    parser.add_argument("--node-rank", type=int, default=int(os.environ.get("NODE_RANK", os.environ.get("VC_TASK_INDEX", "0"))), help=argparse.SUPPRESS)
    parser.add_argument("--master-addr", default=os.environ.get("MASTER_ADDR"), help=argparse.SUPPRESS)
    parser.add_argument("--distributed-worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if not args.distributed_worker and "RANK" not in os.environ:
        raise SystemExit(_run_launcher(args))

    from utils.distributed import fsdp_wrap, launch_distributed_job
    from utils.egosim_checkpoint_eval import EgoSimCheckpointEvalRunner
    from utils.egosim_dmd_wrapper import EgoSimBidirectionalDMDWrapper
    from utils.egosim_wrapper import EgoSimDiffusionWrapper

    launch_distributed_job()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.cuda.current_device()

    checkpoint_path = _resolve_checkpoint_path(args.checkpoint_path)
    config = _load_config(args)
    architecture = _resolve_architecture(args, config)
    dtype = torch.bfloat16 if config.mixed_precision else torch.float32

    if rank == 0:
        args.output_root.mkdir(parents=True, exist_ok=True)
        print(
            f"[EvalCLI] checkpoint={checkpoint_path} "
            f"arch={architecture} num_steps={args.num_steps} world_size={world_size} "
            f"output_root={args.output_root.resolve()}",
            flush=True,
        )
    dist.barrier()

    generator_class = (
        EgoSimDiffusionWrapper
        if architecture == "causal"
        else EgoSimBidirectionalDMDWrapper
    )
    generator = fsdp_wrap(
        generator_class(
            model_root=config.egosim_model_root,
            timestep_shift=getattr(config, "timestep_shift", 5.0),
            local_attn_size=getattr(config, "local_attn_size", -1),
            sink_size=getattr(config, "sink_size", 0),
            init_missing_weights=False,
            load_pretrained_weights=False,
            init_on_meta=True,
        ),
        sharding_strategy=config.sharding_strategy,
        mixed_precision=config.mixed_precision,
        wrap_strategy=config.generator_fsdp_wrap_strategy,
        cpu_offload=getattr(config, "cpu_offload", True),
    )
    _load_generator_checkpoint(generator, checkpoint_path)

    runner = EgoSimCheckpointEvalRunner(
        config,
        device=torch.device(f"cuda:{device}"),
        dtype=dtype,
        generator=generator,
        rank=rank,
        world_size=world_size,
        is_main_process=rank == 0,
        architecture=architecture,
        sample_ids=args.sample_id or None,
    )
    summary = runner.run(args.step)
    if rank == 0:
        if summary is None:
            raise RuntimeError("Eval produced no summary.")
        print(f"[EvalCLI] summary={summary}", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
