from typing import List, Optional

import torch
import torch.distributed as dist

from utils.egosim_dmd_wrapper import EgoSimBidirectionalDMDWrapper
from utils.scheduler import SchedulerInterface


class EgoSimBidirectionalTrainingPipeline:
    """Minimal DMD2-style truncated denoising for EgoSim bidirectional students."""

    def __init__(
        self,
        denoising_step_list: List[int],
        scheduler: SchedulerInterface,
        generator: EgoSimBidirectionalDMDWrapper,
        num_frame_per_block: int = 1,
        independent_first_frame: bool = False,
        same_step_across_blocks: bool = True,
        last_step_only: bool = False,
        num_max_frames: int = 21,
        context_noise: int = 0,
        **kwargs,
    ):
        del num_frame_per_block, independent_first_frame, num_max_frames, context_noise, kwargs
        self.scheduler = scheduler
        self.generator = generator
        self.same_step_across_blocks = same_step_across_blocks
        self.last_step_only = last_step_only
        self.denoising_step_list = denoising_step_list
        if self.denoising_step_list[-1] == 0:
            self.denoising_step_list = self.denoising_step_list[:-1]

    def _sample_exit_step_index(self, device: torch.device) -> int:
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0:
            if self.last_step_only:
                exit_index = torch.tensor(
                    [len(self.denoising_step_list) - 1],
                    device=device,
                    dtype=torch.long,
                )
            else:
                exit_index = torch.randint(
                    low=0,
                    high=len(self.denoising_step_list),
                    size=(1,),
                    device=device,
                    dtype=torch.long,
                )
        else:
            exit_index = torch.empty((1,), device=device, dtype=torch.long)
        if dist.is_initialized():
            dist.broadcast(exit_index, src=0)
        return int(exit_index.item())

    def _to_training_timestep(self, timestep_value: torch.Tensor) -> int:
        scheduler_timesteps = self.scheduler.timesteps.to(timestep_value.device)
        timestep_index = torch.argmin(
            (scheduler_timesteps - timestep_value).abs(), dim=0
        ).item()
        return 1000 - timestep_index

    def inference_with_trajectory(
        self,
        noise: torch.Tensor,
        clean_image_or_video: torch.Tensor = None,
        initial_latent: Optional[torch.Tensor] = None,
        return_sim_step: bool = False,
        **conditional_dict,
    ):
        del clean_image_or_video, initial_latent
        if not self.same_step_across_blocks:
            raise NotImplementedError(
                'EgoSim bidirectional DMD currently requires same_step_across_blocks=true.'
            )

        batch_size, num_frames = noise.shape[:2]
        noisy_input = noise
        exit_index = self._sample_exit_step_index(device=noise.device)
        output = None

        for index, current_timestep in enumerate(self.denoising_step_list):
            timestep = torch.full(
                (batch_size, num_frames),
                float(current_timestep),
                device=noise.device,
                dtype=torch.int64,
            )
            if index < exit_index:
                with torch.no_grad():
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                    )
                    next_timestep = self.denoising_step_list[index + 1]
                    noisy_input = self.scheduler.add_noise(
                        denoised_pred.flatten(0, 1),
                        torch.randn_like(denoised_pred.flatten(0, 1)),
                        torch.full(
                            (batch_size * num_frames,),
                            float(next_timestep),
                            device=noise.device,
                            dtype=torch.long,
                        ),
                    ).unflatten(0, denoised_pred.shape[:2])
            else:
                _, output = self.generator(
                    noisy_image_or_video=noisy_input,
                    conditional_dict=conditional_dict,
                    timestep=timestep,
                )
                break

        if output is None:
            raise RuntimeError('Bidirectional training pipeline did not produce an output.')

        current_training_timestep = self._to_training_timestep(
            self.denoising_step_list[exit_index]
        )
        if exit_index == len(self.denoising_step_list) - 1:
            denoised_timestep_to = 0
        else:
            denoised_timestep_to = self._to_training_timestep(
                self.denoising_step_list[exit_index + 1]
            )
        denoised_timestep_from = current_training_timestep

        if return_sim_step:
            return output, denoised_timestep_from, denoised_timestep_to, exit_index + 1
        return output, denoised_timestep_from, denoised_timestep_to
