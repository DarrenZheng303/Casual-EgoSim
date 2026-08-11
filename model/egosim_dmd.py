import torch
import torch.nn.functional as F

from model.dmd import DMD
from pipeline import EgoSimSelfForcingTrainingPipeline
from pipeline.egosim_bidirectional_training import EgoSimBidirectionalTrainingPipeline
from utils.egosim_dmd_wrapper import EgoSimBidirectionalDMDWrapper
from utils.egosim_wrapper import EgoSimDiffusionWrapper


class EgoSimDMD(DMD):
    """EgoSim Stage-3 DMD with strict CoTracker-region gradient weighting."""

    _TRACK_MASK_RADIUS = 1
    _EPS = 1e-6

    def __init__(self, args, device):
        super().__init__(args, device)
        self._configure_object_dmd(args)

    def _configure_object_dmd(self, args) -> None:
        self.object_dmd_weight = float(getattr(args, 'object_dmd_weight', 1.0))
        self.background_dmd_weight = float(
            getattr(args, 'background_dmd_weight', 0.1)
        )
        if self.object_dmd_weight <= 0.0 or self.background_dmd_weight <= 0.0:
            raise ValueError('Object-DMD weights must be positive.')

    def _initialize_generator(self, args):
        self.student_arch = getattr(args, 'student_arch', 'causal')
        wrapper_kwargs = {
            'model_root': args.egosim_model_root,
            'timestep_shift': getattr(args, 'timestep_shift', 5.0),
            'load_pretrained_weights': False,
            'init_on_meta': True,
        }
        if self.student_arch == 'bidirectional':
            self.generator = EgoSimBidirectionalDMDWrapper(**wrapper_kwargs)
        elif self.student_arch == 'causal':
            self.generator = EgoSimDiffusionWrapper(
                **wrapper_kwargs,
                local_attn_size=getattr(args, 'local_attn_size', -1),
                sink_size=getattr(args, 'sink_size', 0),
            )
        else:
            raise ValueError(
                f'Unknown student_arch: {self.student_arch}. '
                "Expected 'causal' or 'bidirectional'."
            )
        return wrapper_kwargs

    def _initialize_models(self, args, device):
        wrapper_kwargs = self._initialize_generator(args)

        self.generator.model.requires_grad_(True)

        self.real_score = EgoSimBidirectionalDMDWrapper(**wrapper_kwargs)
        self.real_score.model.requires_grad_(False)
        self.fake_score = EgoSimBidirectionalDMDWrapper(**wrapper_kwargs)
        self.fake_score.model.requires_grad_(True)

        self.text_encoder = torch.nn.Identity().requires_grad_(False)
        self.vae = torch.nn.Identity().requires_grad_(False)
        self.scheduler = self.generator.get_scheduler()
        self.scheduler.timesteps = self.scheduler.timesteps.to(device)

    def _prepare_tracks(
        self,
        *,
        physics_tracks: torch.Tensor,
        physics_visibility: torch.Tensor,
        physics_object_track_ids: torch.Tensor,
        physics_hand_track_mask: torch.Tensor,
        num_latent_frames: int,
        raw_num_frames: torch.Tensor,
        raw_height: torch.Tensor,
        raw_width: torch.Tensor,
        train_height: torch.Tensor,
        train_width: torch.Tensor,
        spatial_height: int,
        spatial_width: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, _, max_track_frames, _ = physics_tracks.shape
        tracks = physics_tracks.to(device=device, dtype=torch.float32)
        visibility = physics_visibility.to(device=device, dtype=torch.bool)
        object_ids = physics_object_track_ids.to(device=device, dtype=torch.long)
        hand_tracks = physics_hand_track_mask.to(device=device, dtype=torch.bool)
        raw_num_frames = raw_num_frames.to(device=device, dtype=torch.long).flatten()
        raw_height = raw_height.to(device=device, dtype=torch.float32).flatten()
        raw_width = raw_width.to(device=device, dtype=torch.float32).flatten()
        train_height = train_height.to(device=device, dtype=torch.float32).flatten()
        train_width = train_width.to(device=device, dtype=torch.float32).flatten()

        frame_indices = []
        for index in range(batch_size):
            usable_frames = min(
                int(raw_num_frames[index].item()), max_track_frames
            )
            frame_indices.append(torch.tensor(
                [
                    0 if frame == 0 else min(frame * 4, usable_frames - 1)
                    for frame in range(num_latent_frames)
                ],
                device=device,
                dtype=torch.long,
            ))
        frame_indices = torch.stack(frame_indices)
        tracks = torch.gather(
            tracks,
            2,
            frame_indices[:, None, :, None].expand(
                batch_size, tracks.shape[1], num_latent_frames, 2
            ),
        )
        visibility = torch.gather(
            visibility,
            2,
            frame_indices[:, None, :].expand(
                batch_size, visibility.shape[1], num_latent_frames
            ),
        )

        raw_scale_x = (train_width - 1.0) / (raw_width - 1.0).clamp_min(1.0)
        raw_scale_y = (train_height - 1.0) / (raw_height - 1.0).clamp_min(1.0)
        latent_scale_x = (spatial_width - 1.0) / (train_width - 1.0).clamp_min(1.0)
        latent_scale_y = (spatial_height - 1.0) / (train_height - 1.0).clamp_min(1.0)
        tracks = tracks.clone()
        tracks[..., 0] = (
            tracks[..., 0]
            * raw_scale_x[:, None, None]
            * latent_scale_x[:, None, None]
        )
        tracks[..., 1] = (
            tracks[..., 1]
            * raw_scale_y[:, None, None]
            * latent_scale_y[:, None, None]
        )

        selected_tracks = physics_hand_track_mask.to(
            device=device, dtype=torch.bool
        ) | (object_ids >= 0)
        return tracks, visibility, selected_tracks

    def _build_object_dmd_weight(
        self,
        reference: torch.Tensor,
        tracks: torch.Tensor,
        visibility: torch.Tensor,
        selected_tracks: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        batch_size, _, num_frames, _ = tracks.shape
        height, width = reference.shape[-2:]
        flat_mask = torch.zeros(
            (batch_size * num_frames, height, width),
            device=reference.device,
            dtype=torch.float32,
        )
        visible = visibility & selected_tracks[:, :, None]
        rounded = tracks.round().to(dtype=torch.long)
        rounded[..., 0].clamp_(0, width - 1)
        rounded[..., 1].clamp_(0, height - 1)
        batch_index = torch.arange(
            batch_size, device=reference.device
        ).view(batch_size, 1, 1).expand_as(visible)
        frame_index = torch.arange(
            num_frames, device=reference.device
        ).view(1, 1, num_frames).expand_as(visible)
        flat_mask[
            batch_index[visible] * num_frames + frame_index[visible],
            rounded[..., 1][visible],
            rounded[..., 0][visible],
        ] = 1.0
        flat_mask = F.max_pool2d(
            flat_mask.unsqueeze(1),
            kernel_size=2 * self._TRACK_MASK_RADIUS + 1,
            stride=1,
            padding=self._TRACK_MASK_RADIUS,
        ).squeeze(1)
        track_mask = flat_mask.view(batch_size, num_frames, 1, height, width)

        empty_samples = track_mask.flatten(1).sum(dim=1) == 0
        if empty_samples.any():
            indices = empty_samples.nonzero(as_tuple=False).flatten().tolist()
            raise RuntimeError(
                'Object-DMD requires a non-empty CoTracker mask for every '
                f'batch sample; empty sample indices: {indices}'
            )

        weight = self.background_dmd_weight + (
            self.object_dmd_weight - self.background_dmd_weight
        ) * track_mask
        weight = weight.to(device=reference.device, dtype=reference.dtype).detach()
        weight_mean = weight.float().mean().detach()
        weight = weight / weight.float().mean(
            dim=(2, 3, 4), keepdim=True
        ).clamp_min(self._EPS).to(dtype=weight.dtype)
        return weight, {
            'object_dmd/track_mask_ratio': track_mask.mean().detach(),
            'object_dmd/weight_mean': weight_mean,
        }

    def generator_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        clean_latent: torch.Tensor,
        initial_latent: torch.Tensor = None,
        physics_tracks: torch.Tensor = None,
        physics_visibility: torch.Tensor = None,
        physics_object_track_ids: torch.Tensor = None,
        physics_hand_track_mask: torch.Tensor = None,
        physics_valid: torch.Tensor = None,
        raw_num_frames: torch.Tensor = None,
        raw_height: torch.Tensor = None,
        raw_width: torch.Tensor = None,
        train_height: torch.Tensor = None,
        train_width: torch.Tensor = None,
    ):
        del clean_latent
        physics_inputs = {
            'physics_tracks': physics_tracks,
            'physics_visibility': physics_visibility,
            'physics_object_track_ids': physics_object_track_ids,
            'physics_hand_track_mask': physics_hand_track_mask,
            'physics_valid': physics_valid,
            'raw_num_frames': raw_num_frames,
            'raw_height': raw_height,
            'raw_width': raw_width,
            'train_height': train_height,
            'train_width': train_width,
        }
        missing = [name for name, value in physics_inputs.items() if value is None]
        if missing:
            raise RuntimeError(
                'Object-DMD requires physics inputs; missing: ' + ', '.join(missing)
            )
        invalid = ~physics_valid.detach().bool().flatten()
        if invalid.any():
            indices = invalid.nonzero(as_tuple=False).flatten().tolist()
            raise RuntimeError(
                'Object-DMD requires valid physics data for every batch sample; '
                f'invalid sample indices: {indices}'
            )

        pred_image, gradient_mask, denoised_from, denoised_to = self._run_generator(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            initial_latent=initial_latent,
        )
        tracks, visibility, selected_tracks = self._prepare_tracks(
            physics_tracks=physics_tracks,
            physics_visibility=physics_visibility,
            physics_object_track_ids=physics_object_track_ids,
            physics_hand_track_mask=physics_hand_track_mask,
            num_latent_frames=pred_image.shape[1],
            raw_num_frames=raw_num_frames,
            raw_height=raw_height,
            raw_width=raw_width,
            train_height=train_height,
            train_width=train_width,
            spatial_height=pred_image.shape[-2],
            spatial_width=pred_image.shape[-1],
            device=pred_image.device,
        )
        weight, object_dmd_logs = self._build_object_dmd_weight(
            pred_image, tracks, visibility, selected_tracks
        )

        dmd_loss, log_dict = super().compute_distribution_matching_loss(
            image_or_video=pred_image,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            gradient_mask=gradient_mask,
            denoised_timestep_from=denoised_from,
            denoised_timestep_to=denoised_to,
            gradient_weight=weight,
        )

        log_dict.update(object_dmd_logs)
        log_dict['object_dmd/valid_sample_ratio'] = physics_valid.float().mean().detach()
        log_dict['dmd_loss'] = dmd_loss.detach()
        return dmd_loss, log_dict

    def _initialize_inference_pipeline(self):
        if self.student_arch == 'bidirectional':
            self.inference_pipeline = EgoSimBidirectionalTrainingPipeline(
                denoising_step_list=self.denoising_step_list,
                scheduler=self.scheduler,
                generator=self.generator,
                num_frame_per_block=self.num_frame_per_block,
                independent_first_frame=self.args.independent_first_frame,
                same_step_across_blocks=self.args.same_step_across_blocks,
                last_step_only=self.args.last_step_only,
                num_max_frames=self.num_training_frames,
                context_noise=self.args.context_noise,
            )
            return

        self.inference_pipeline = EgoSimSelfForcingTrainingPipeline(
            denoising_step_list=self.denoising_step_list,
            denoising_step_list_first_chunk=self.denoising_step_list_first_chunk,
            scheduler=self.scheduler,
            generator=self.generator,
            num_frame_per_block=self.num_frame_per_block,
            independent_first_frame=self.args.independent_first_frame,
            same_step_across_blocks=self.args.same_step_across_blocks,
            last_step_only=self.args.last_step_only,
            num_max_frames=self.num_training_frames,
            context_noise=self.args.context_noise,
            local_attn_size=getattr(self.args, 'local_attn_size', -1),
            kv_cache_cpu_offload=getattr(self.args, 'kv_cache_cpu_offload', False),
            kv_cache_cpu_offload_layers=getattr(
                self.args, 'kv_cache_cpu_offload_layers', None
            ),
        )
