import os
import types
from contextlib import nullcontext
from typing import Optional

import torch
import torch.distributed as dist
from safetensors.torch import load_file as load_safetensors

from utils.scheduler import SchedulerInterface, FlowMatchScheduler
from wan.modules.causal_model import CausalWanModel
from wan.modules.model import rope_params


def _is_rank0() -> bool:
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


class EgoSimDiffusionWrapper(torch.nn.Module):
    def __init__(
        self,
        model_root: str,
        timestep_shift: float = 5.0,
        local_attn_size: int = -1,
        sink_size: int = 0,
        init_missing_weights: bool = True,
        load_pretrained_weights: bool = True,
        init_on_meta: bool = False,
    ):
        super().__init__()
        self.model_root = os.path.abspath(model_root)
        self._init_missing_weights = init_missing_weights
        self._load_pretrained_weights = load_pretrained_weights
        self.init_on_meta = init_on_meta
        self.model = self._build_model(local_attn_size=local_attn_size, sink_size=sink_size)
        self.model.eval()
        self.uniform_timestep = False
        self.scheduler = FlowMatchScheduler(
            shift=timestep_shift, sigma_min=0.0, extra_one_step=True
        )
        self.scheduler.set_timesteps(1000, training=True)
        self.seq_len = 32760  # [1, 21, 16, 60, 104]
        self.get_scheduler()

    def _build_model(self, local_attn_size: int, sink_size: int) -> CausalWanModel:
        original_init = CausalWanModel.init_weights
        CausalWanModel.init_weights = lambda self: None
        try:
            device_context = torch.device('meta') if self.init_on_meta else nullcontext()
            with device_context:
                model = CausalWanModel(
                    model_type='i2v',
                    patch_size=(1, 2, 2),
                    text_len=512,
                    in_dim=52,
                    dim=5120,
                    ffn_dim=13824,
                    freq_dim=256,
                    text_dim=4096,
                    out_dim=16,
                    num_heads=40,
                    num_layers=40,
                    local_attn_size=local_attn_size,
                    sink_size=sink_size,
                    qk_norm=True,
                    cross_attn_norm=True,
                    eps=1e-6,
                )
        finally:
            CausalWanModel.init_weights = original_init

        if self.init_on_meta:
            self._materialize_static_tensors(model)

        weight_path = os.path.join(self.model_root, 'diffusion_pytorch_model.safetensors')
        if not self._load_pretrained_weights:
            return model

        if not os.path.exists(weight_path):
            raise FileNotFoundError(f'EgoSim DiT weights not found: {weight_path}')

        state_dict = load_safetensors(weight_path)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if self._init_missing_weights:
            self._initialize_missing_parameters(model, missing)
        if _is_rank0():
            print(f'EgoSim load_state_dict: missing={len(missing)} unexpected={len(unexpected)}')
            if missing:
                print(f'  Missing sample: {missing[:5]}')
            if unexpected:
                print(f'  Unexpected sample: {unexpected[:5]}')
        return model

    @staticmethod
    def _materialize_static_tensors(model: CausalWanModel) -> None:
        d = model.dim // model.num_heads
        model.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
        ], dim=1)

    @staticmethod
    def _initialize_missing_parameters(model: torch.nn.Module, missing_keys: list[str]) -> None:
        named_params = dict(model.named_parameters())
        named_buffers = dict(model.named_buffers())
        for key in missing_keys:
            if key in named_params:
                param = named_params[key]
                with torch.no_grad():
                    if key.endswith('bias'):
                        param.zero_()
                    elif param.ndim >= 2:
                        torch.nn.init.xavier_uniform_(param)
                    elif 'norm' in key.lower():
                        param.fill_(1.0)
                    else:
                        param.zero_()
            elif key in named_buffers:
                buffer = named_buffers[key]
                with torch.no_grad():
                    buffer.zero_()

    def enable_gradient_checkpointing(self) -> None:
        self.model.enable_gradient_checkpointing()

    @staticmethod
    def _convert_flow_pred_to_x0(flow_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor, scheduler) -> torch.Tensor:
        original_dtype = flow_pred.dtype
        flow_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(flow_pred.device),
            [flow_pred, xt, scheduler.sigmas, scheduler.timesteps],
        )
        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1
        )
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        x0_pred = xt - sigma_t * flow_pred
        return x0_pred.to(original_dtype)

    def _build_condition_latents(self, conditional_dict: dict) -> torch.Tensor:
        mask_latent = conditional_dict['mask_latent']
        ego_prior_latent = conditional_dict['ego_prior_latent']
        hand_latent = conditional_dict['hand_latent']
        mask_weight = mask_latent[:, :, :1].expand_as(ego_prior_latent)
        masked_ego = ego_prior_latent * (1.0 - mask_weight)
        return torch.cat([mask_latent, masked_ego, hand_latent], dim=2)

    def forward(
        self,
        noisy_image_or_video: torch.Tensor,
        conditional_dict: dict,
        timestep: torch.Tensor,
        kv_cache: Optional[list[dict]] = None,
        crossattn_cache: Optional[list[dict]] = None,
        current_start: Optional[int] = None,
        classify_mode: Optional[bool] = False,
        concat_time_embeddings: Optional[bool] = False,
        clean_x: Optional[torch.Tensor] = None,
        aug_t: Optional[torch.Tensor] = None,
        cache_start: Optional[int] = None,
    ):
        del classify_mode, concat_time_embeddings
        prompt_embeds = conditional_dict['prompt_embeds']
        image_embeds = conditional_dict['image_embeds']
        condition_latents = self._build_condition_latents(conditional_dict)

        if kv_cache is not None:
            if crossattn_cache is None:
                raise ValueError('kv_cache requires crossattn_cache.')
            frame_seq_length = (
                noisy_image_or_video.shape[-2] * noisy_image_or_video.shape[-1]
                // (self.model.patch_size[1] * self.model.patch_size[2])
            )
            start_frame = (current_start or 0) // frame_seq_length
            condition_latents = condition_latents[
                :, start_frame:start_frame + noisy_image_or_video.shape[1]
            ]
            flow_pred = self.model(
                noisy_image_or_video.permute(0, 2, 1, 3, 4),
                t=timestep,
                context=prompt_embeds,
                seq_len=self.seq_len,
                clip_fea=image_embeds,
                y=condition_latents.permute(0, 2, 1, 3, 4),
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start=current_start or 0,
                cache_start=cache_start or 0,
            ).permute(0, 2, 1, 3, 4)
            pred_x0 = self._convert_flow_pred_to_x0(
                flow_pred=flow_pred.flatten(0, 1),
                xt=noisy_image_or_video.flatten(0, 1),
                timestep=timestep.flatten(0, 1),
                scheduler=self.scheduler,
            ).unflatten(0, flow_pred.shape[:2])
            return flow_pred, pred_x0
        if crossattn_cache is not None:
            raise ValueError('crossattn_cache requires kv_cache.')

        model_kwargs = {
            'x': noisy_image_or_video.permute(0, 2, 1, 3, 4),
            't': timestep,
            'context': prompt_embeds,
            'seq_len': self.seq_len,
            'clip_fea': image_embeds,
            'y': condition_latents.permute(0, 2, 1, 3, 4),
        }
        if clean_x is not None:
            clean_input = torch.cat([clean_x, condition_latents], dim=2)
            model_kwargs['clean_x'] = clean_input.permute(0, 2, 1, 3, 4)
            model_kwargs['aug_t'] = aug_t

        flow_pred = self.model(**model_kwargs).permute(0, 2, 1, 3, 4)
        pred_x0 = self._convert_flow_pred_to_x0(
            flow_pred=flow_pred.flatten(0, 1),
            xt=noisy_image_or_video.flatten(0, 1),
            timestep=timestep.flatten(0, 1),
            scheduler=self.scheduler,
        ).unflatten(0, flow_pred.shape[:2])
        return flow_pred, pred_x0

    def get_scheduler(self) -> SchedulerInterface:
        scheduler = self.scheduler
        scheduler.convert_x0_to_noise = types.MethodType(
            SchedulerInterface.convert_x0_to_noise, scheduler
        )
        scheduler.convert_noise_to_x0 = types.MethodType(
            SchedulerInterface.convert_noise_to_x0, scheduler
        )
        scheduler.convert_velocity_to_x0 = types.MethodType(
            SchedulerInterface.convert_velocity_to_x0, scheduler
        )
        self.scheduler = scheduler
        return scheduler
