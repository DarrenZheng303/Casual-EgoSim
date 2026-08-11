import gc
import os
import time
from contextlib import contextmanager, nullcontext

import torch
import torch.distributed as dist
import wandb
from safetensors.torch import load_file as load_safetensors

from model import EgoSimDMD
from trainer.distillation import Trainer as _BaseTrainer
from utils.dataset import EgoSimCacheDataset
from utils.distributed import fsdp_state_dict
from utils.egosim_checkpoint_eval import EgoSimCheckpointEvalRunner
from utils.egosim_encoders import encode_prompt, load_text_encoder


class Trainer(_BaseTrainer):
    """Isolated Stage-3 trainer for EgoSim DMD."""

    def __init__(self, config):
        super().__init__(config)
        self.eval_runner = None

    def _build_model(self, config):
        return EgoSimDMD(config, device=self.device)

    def _wrap_conditioning_models(self, config):
        pass

    def _load_checkpoints(self, config):
        with self._loading_section(f'generator checkpoint from {config.generator_ckpt}'):
            self._load_generator_checkpoint(config.generator_ckpt)
        for name in ('real_score', 'fake_score'):
            checkpoint_path = getattr(config, f'{name}_ckpt')
            with self._loading_section(f'{name} checkpoint from {checkpoint_path}'):
                self._load_bidirectional_checkpoint(
                    getattr(self.model, name),
                    name,
                    checkpoint_path,
                )
        self._clear_state_dict_cache()

    def _build_dataset(self, config):
        return EgoSimCacheDataset(
            config.data_path,
            physics_track_mode=config.physics_track_mode,
            require_physics=True,
        )

    def train(self):
        if self.is_main_process:
            print('[Warmup] training fake_score for 5 updates', flush=True)
        for warmup_step in range(5):
            self.critic_optimizer.zero_grad(set_to_none=True)
            critic_log_dict = self.fwdbwd_one_step(
                next(self.dataloader), train_generator=False
            )
            self.critic_optimizer.step()
            if self.is_main_process:
                print(
                    f"[Warmup] fake_score {warmup_step + 1}/5 "
                    f"critic_loss={critic_log_dict['critic_loss'].item():.6f}",
                    flush=True,
                )
        return super().train()

    def save(self):
        if self.generator_ema is not None and self.config.ema_start_step < self.step:
            # After EMA starts, store EMA weights under the canonical generator key
            # so inference and visualization always load a single format.
            generator_state_dict = self.generator_ema.full_state_dict(self.model.generator)
        else:
            generator_state_dict = fsdp_state_dict(self.model.generator)
        state_dict = {
            'generator': generator_state_dict,
        }

        if self.is_main_process:
            checkpoint_dir = os.path.join(
                self.output_path, f'checkpoint_model_{self.step:06d}'
            )
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, 'model.pt')
            torch.save(state_dict, checkpoint_path)
            print(f'Model saved to {checkpoint_path}')

        del state_dict, generator_state_dict
        gc.collect()
        dist.barrier()
        if not getattr(self.config, 'checkpoint_eval_enabled', False):
            return
        if self.is_main_process:
            print(f'[Eval] start step={self.step}', flush=True)
        eval_start_time = time.perf_counter()

        if self.eval_runner is None:
            self.eval_runner = EgoSimCheckpointEvalRunner(
                self.config,
                device=torch.device(f'cuda:{self.device}'),
                dtype=self.dtype,
                generator=self.model.generator,
                rank=dist.get_rank(),
                world_size=self.world_size,
                is_main_process=self.is_main_process,
            )
        use_ema = (
            self.generator_ema is not None
            and self.config.ema_start_step < self.step
        )
        eval_weights = (
            self.generator_ema.swap_with(self.model.generator)
            if use_ema
            else nullcontext()
        )
        if self.is_main_process:
            print(
                f"[Eval] weights={'EMA checkpoint' if use_ema else 'current generator'}",
                flush=True,
            )
        with eval_weights:
            summary = self.eval_runner.run(self.step)
        if summary is not None and self.is_main_process and not self.disable_wandb:
            wandb.log(
                {f'eval/{key}': value for key, value in summary.items()},
                step=self.step,
            )
        if summary is not None and self.is_main_process:
            eval_duration = time.perf_counter() - eval_start_time
            message = (
                f"[Eval] done step={self.step} "
                f"psnr_teacher50step={summary['psnr_teacher50step']:.4f} "
                f"ssim_teacher50step={summary['ssim_teacher50step']:.4f} "
                f"psnr_gt={summary['psnr_gt']:.4f} "
                f"ssim_gt={summary['ssim_gt']:.4f} "
                f"lpips_teacher50step={summary['lpips_teacher50step']:.4f} "
                f"lpips_gt={summary['lpips_gt']:.4f} "
                f"num_samples={summary['num_samples']} "
            )
            message += f"using {eval_duration:.3f} s"
            print(message, flush=True)

    @staticmethod
    def _extract_generator_state_dict(state_dict):
        if 'generator' in state_dict:
            state_dict = state_dict['generator']
        elif 'model' in state_dict:
            state_dict = state_dict['model']
        elif 'generator_ema' in state_dict:
            state_dict = state_dict['generator_ema']
        return {
            key.replace('._fsdp_wrapped_module', '')
            .replace('._checkpoint_wrapped_module', '')
            .replace('._orig_mod', ''): value
            for key, value in state_dict.items()
        }

    @contextmanager
    def _loading_section(self, description: str):
        """Print rank0 start/finish logs around a checkpoint loading block."""
        if self.is_main_process:
            print(f'[Loading] {description} ...')
        yield
        if self.is_main_process:
            print(f'[Loaded]  {description}')

    def _load_generator_checkpoint(self, checkpoint_path):
        if checkpoint_path.endswith('.safetensors'):
            state_dict = self._load_prefixed_safetensors_state_dict(checkpoint_path)
            missing, unexpected = self.model.generator.load_state_dict(
                state_dict, strict=False)
            if self.is_main_process:
                print(
                    'Generator safetensors load_state_dict: '
                    f'missing={len(missing)} unexpected={len(unexpected)}')
                if missing:
                    print(f'  Missing sample: {missing[:5]}')
                if unexpected:
                    print(f'  Unexpected sample: {unexpected[:5]}')
        else:
            state_dict = torch.load(checkpoint_path, map_location='cpu', mmap=True)
            state_dict = self._extract_generator_state_dict(state_dict)
            self.model.generator.load_state_dict(state_dict, strict=True)
        del state_dict
        gc.collect()

    def _load_bidirectional_checkpoint(
        self,
        module,
        name,
        checkpoint_path,
    ):
        if checkpoint_path.endswith('.safetensors'):
            state_dict = self._load_prefixed_safetensors_state_dict(checkpoint_path)
        else:
            state_dict = torch.load(
                checkpoint_path, map_location='cpu', mmap=True
            )
            state_dict = self._extract_generator_state_dict(state_dict)
        module.load_state_dict(state_dict, strict=True)
        del state_dict
        gc.collect()

    def _load_prefixed_safetensors_state_dict(self, checkpoint_path):
        checkpoint_path = os.path.abspath(checkpoint_path)
        cache = getattr(self, '_prefixed_safetensors_state_dict_cache', None)
        if cache is None:
            cache = {}
            self._prefixed_safetensors_state_dict_cache = cache
        if checkpoint_path not in cache:
            raw_state_dict = load_safetensors(checkpoint_path, device='cpu')
            cache[checkpoint_path] = {
                f'model.{key}': value for key, value in raw_state_dict.items()
            }
        return cache[checkpoint_path]

    def _clear_state_dict_cache(self):
        cache = getattr(self, '_prefixed_safetensors_state_dict_cache', None)
        if cache is not None:
            cache.clear()
            del self._prefixed_safetensors_state_dict_cache
            gc.collect()

    def _global_log_mean(self, value):
        value = value.detach().float().mean().to(self.device)
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        return value / self.world_size

    def _global_log_scalars(self, log_dict):
        return {
            key: self._global_log_mean(value)
            if torch.is_tensor(value) and value.numel() == 1 else value
            for key, value in log_dict.items()
        }

    def _encode_egosim_negative_prompt_embedding(self) -> torch.Tensor:
        cached = getattr(self, '_egosim_negative_prompt_embedding', None)
        if cached is not None:
            return cached

        shape_tensor = torch.zeros((2,), device=self.device, dtype=torch.long)
        status = [None]
        negative_prompt_embedding = None
        if self.is_main_process:
            try:
                model_root = os.path.abspath(self.config.egosim_model_root)
                text_encoder = load_text_encoder(
                    model_root,
                    torch.device(f'cuda:{self.device}'),
                )
                with torch.no_grad():
                    negative_prompt_embedding = encode_prompt(
                        text_encoder,
                        self.config.negative_prompt,
                        torch.device(f'cuda:{self.device}'),
                    ).to(device=self.device, dtype=self.dtype)
                text_encoder.model.to('cpu')
                del text_encoder
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                shape_tensor = torch.tensor(
                    negative_prompt_embedding.shape,
                    device=self.device,
                    dtype=torch.long,
                )
                status[0] = 'ok'
            except Exception as exc:
                status[0] = (
                    'failed to build EgoSim negative prompt embedding for CFG. '
                    f'root_cause={type(exc).__name__}: {exc}'
                )

        dist.broadcast_object_list(status, src=0)
        if status[0] != 'ok':
            raise RuntimeError(
                '[CFG] EgoSim stage-3 requested true CFG, but negative prompt '
                'embedding could not be constructed. '
                f'{status[0]}'
            )
        dist.broadcast(shape_tensor, src=0)
        if negative_prompt_embedding is None:
            negative_prompt_embedding = torch.empty(
                tuple(int(v.item()) for v in shape_tensor),
                device=self.device,
                dtype=self.dtype,
            )
        dist.broadcast(negative_prompt_embedding, src=0)
        negative_prompt_embedding = negative_prompt_embedding.detach()
        self._egosim_negative_prompt_embedding = negative_prompt_embedding
        return negative_prompt_embedding

    def _build_egosim_unconditional_dict(self, conditional_dict: dict) -> dict:
        real_guidance_scale = float(getattr(self.model, 'real_guidance_scale', 0.0))
        fake_guidance_scale = float(getattr(self.model, 'fake_guidance_scale', 0.0))
        if real_guidance_scale == 0.0 and fake_guidance_scale == 0.0:
            return conditional_dict

        batch_size = int(conditional_dict['prompt_embeds'].shape[0])
        cache_key = (batch_size, self.dtype)
        cache = getattr(self, '_egosim_unconditional_prompt_cache', None)
        if cache is None:
            cache = {}
            self._egosim_unconditional_prompt_cache = cache

        prompt_embeds = cache.get(cache_key)
        if prompt_embeds is None:
            prompt_embeds = self._encode_egosim_negative_prompt_embedding().unsqueeze(0)
            prompt_embeds = prompt_embeds.expand(batch_size, -1, -1)
            cache[cache_key] = prompt_embeds

        unconditional_dict = dict(conditional_dict)
        unconditional_dict['prompt_embeds'] = prompt_embeds
        return unconditional_dict

    def fwdbwd_one_step(self, batch, train_generator, clean_latent=None):
        self.model.eval()
        if self.step % 20 == 0:
            torch.cuda.empty_cache()

        conditional_dict = {
            'prompt_embeds': batch['prompt_embeds'].to(self.device, self.dtype),
            'image_embeds': batch['image_embeds'].to(self.device, self.dtype),
            'ego_prior_latent': batch['ego_prior_latent'].to(self.device, self.dtype),
            'hand_latent': batch['hand_latent'].to(self.device, self.dtype),
            'mask_latent': batch['mask_latent'].to(self.device, self.dtype),
        }
        unconditional_dict = self._build_egosim_unconditional_dict(conditional_dict)
        image_or_video_shape = list(self.config.image_or_video_shape)
        image_or_video_shape[0] = batch['clean_latent'].shape[0]
        clean_latent = batch['clean_latent'].to(self.device, self.dtype)
        if train_generator:
            self.model.train_step = self.step
            generator_loss, log_dict = self.model.generator_loss(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                clean_latent=clean_latent,
                physics_tracks=batch.get('physics_tracks'),
                physics_visibility=batch.get('physics_visibility'),
                physics_object_track_ids=batch.get('physics_object_track_ids'),
                physics_hand_track_mask=batch.get('physics_hand_track_mask'),
                physics_valid=batch.get('physics_valid'),
                raw_num_frames=batch.get('raw_num_frames'),
                raw_height=batch.get('raw_height'),
                raw_width=batch.get('raw_width'),
                train_height=batch.get('train_height'),
                train_width=batch.get('train_width'),
            )
            generator_loss.backward()
            generator_grad_norm = self.model.generator.clip_grad_norm_(
                self.max_grad_norm_generator
            )
            log_dict.update({
                'generator_loss': generator_loss,
                'generator_grad_norm': generator_grad_norm,
            })
            return self._global_log_scalars(log_dict)

        critic_loss, log_dict = self.model.critic_loss(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            clean_latent=clean_latent,
        )
        critic_loss.backward()
        critic_grad_norm = self.model.fake_score.clip_grad_norm_(
            self.max_grad_norm_critic
        )
        log_dict.update({
            'critic_loss': critic_loss,
            'critic_grad_norm': critic_grad_norm,
        })
        return self._global_log_scalars(log_dict)
