from contextlib import nullcontext
from typing import Optional

import torch

from utils.egosim_wrapper import EgoSimDiffusionWrapper
from wan.modules.model import WanModel


class EgoSimCausalDMDWrapper(EgoSimDiffusionWrapper):
    """EgoSim causal wrapper with the KV-cache path required by DMD rollout."""

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
        if kv_cache is None:
            return super().forward(
                noisy_image_or_video=noisy_image_or_video,
                conditional_dict=conditional_dict,
                timestep=timestep,
                classify_mode=classify_mode,
                concat_time_embeddings=concat_time_embeddings,
                clean_x=clean_x,
                aug_t=aug_t,
            )

        del classify_mode, concat_time_embeddings, clean_x, aug_t
        frame_seq_length = (
            noisy_image_or_video.shape[-2]
            * noisy_image_or_video.shape[-1]
            // (self.model.patch_size[1] * self.model.patch_size[2])
        )
        start_frame = (current_start or 0) // frame_seq_length
        num_frames = noisy_image_or_video.shape[1]
        condition_latents = self._build_condition_latents(conditional_dict)
        condition_latents = condition_latents[:, start_frame:start_frame + num_frames]
        flow_pred = self.model(
            noisy_image_or_video.permute(0, 2, 1, 3, 4),
            t=timestep,
            context=conditional_dict['prompt_embeds'],
            seq_len=self.seq_len,
            clip_fea=conditional_dict['image_embeds'],
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


class EgoSimBidirectionalDMDWrapper(EgoSimDiffusionWrapper):
    """Original bidirectional EgoSim backbone used by real/fake scores."""

    def _build_model(self, local_attn_size: int, sink_size: int) -> WanModel:
        del local_attn_size, sink_size
        original_init = WanModel.init_weights
        WanModel.init_weights = lambda self: None
        try:
            device_context = torch.device('meta') if self.init_on_meta else nullcontext()
            with device_context:
                model = WanModel(
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
                    window_size=(-1, -1),
                    qk_norm=True,
                    cross_attn_norm=True,
                    eps=1e-6,
                )
        finally:
            WanModel.init_weights = original_init
        if self.init_on_meta:
            self._materialize_static_tensors(model)
        return model

    def forward(
        self,
        noisy_image_or_video: torch.Tensor,
        conditional_dict: dict,
        timestep: torch.Tensor,
        **kwargs,
    ):
        del kwargs
        condition_latents = self._build_condition_latents(conditional_dict)
        flow_pred = self.model(
            noisy_image_or_video.permute(0, 2, 1, 3, 4),
            t=timestep[:, 0],
            context=conditional_dict['prompt_embeds'],
            seq_len=self.seq_len,
            clip_fea=conditional_dict['image_embeds'],
            y=condition_latents.permute(0, 2, 1, 3, 4),
        ).permute(0, 2, 1, 3, 4)
        pred_x0 = self._convert_flow_pred_to_x0(
            flow_pred=flow_pred.flatten(0, 1),
            xt=noisy_image_or_video.flatten(0, 1),
            timestep=timestep.flatten(0, 1),
            scheduler=self.scheduler,
        ).unflatten(0, flow_pred.shape[:2])
        return flow_pred, pred_x0
