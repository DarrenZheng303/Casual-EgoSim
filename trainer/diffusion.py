import gc
import logging

from model import CausalDiffusion
from utils.dataset import cycle, LatentLMDBDataset, EgoSimCacheDataset
from utils.misc import set_seed
import torch.distributed as dist
from omegaconf import OmegaConf
import torch
import wandb
import time
import os
import math
from utils.distributed import EMA_FSDP, barrier, fsdp_wrap, fsdp_state_dict, graceful_stop_and_save, launch_distributed_job
from utils.egosim_checkpoint_eval import EgoSimCheckpointEvalRunner
from pipeline import (
    CausalDiffusionInferencePipeline,
    CausalInferencePipeline,
)


def _extract_generator_state_dict(checkpoint):
    if "generator" in checkpoint:
        state_dict = checkpoint["generator"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "generator_ema" in checkpoint:
        state_dict = checkpoint["generator_ema"]
    else:
        state_dict = checkpoint
    return {
        key.replace("model._fsdp_wrapped_module.", "model.", 1)
        if key.startswith("model._fsdp_wrapped_module.")
        else key: value
        for key, value in state_dict.items()
    }

class Trainer:
    def __init__(self, config):
        self.config = config
        self.step = int(getattr(config, "resume_step", 0) or 0)
        if self.step < 0:
            raise ValueError("resume_step must be non-negative")

        # Step 1: Initialize the distributed training environment (rank, seed, dtype, logging etc.)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        launch_distributed_job()
        global_rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.device = torch.cuda.current_device()
        self.is_main_process = global_rank == 0
        self.causal = config.causal
        self.disable_wandb = config.disable_wandb
        self.model_type = getattr(config, "model_type", "wan")

        # use a random seed for the training
        if config.seed == 0:
            random_seed = torch.randint(0, 10000000, (1,), device=self.device)
            dist.broadcast(random_seed, src=0)
            config.seed = random_seed.item()

        set_seed(config.seed + global_rank)

        if self.is_main_process and not self.disable_wandb:
            wandb.login(host=config.wandb_host, key=config.wandb_key)
            wandb_kwargs = dict(
                config=OmegaConf.to_container(config, resolve=True),
                name=config.config_name,
                mode="online",
                entity=config.wandb_entity,
                project=config.wandb_project,
                dir=config.wandb_save_dir
            )
            wandb_run_id = getattr(config, "wandb_run_id", None)
            if wandb_run_id:
                wandb_kwargs["id"] = str(wandb_run_id)
                wandb_kwargs["resume"] = str(
                    getattr(config, "wandb_resume", "must")
                )
            wandb.init(**wandb_kwargs)

        self.output_path = config.logdir

        # Step 2: Initialize the model and optimizer
        self.model = CausalDiffusion(config, device=self.device)
        generator_ckpt = getattr(config, "generator_ckpt", False)
        if generator_ckpt and self.is_main_process:
            print(f"Loading full generator checkpoint on rank 0 from {generator_ckpt}")
            self.model.generator.model.to_empty(device="cpu")
            checkpoint = torch.load(
                generator_ckpt, map_location="cpu", mmap=True, weights_only=True
            )
            self.model.generator.load_state_dict(
                _extract_generator_state_dict(checkpoint), strict=True, assign=True
            )
            del checkpoint
            gc.collect()
        if generator_ckpt:
            dist.barrier()
        self.model.generator = fsdp_wrap(
            self.model.generator,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.generator_fsdp_wrap_strategy,
            sync_module_states=bool(generator_ckpt),
        )

        if self.model_type != "egosim":
            self.model.text_encoder = fsdp_wrap(
                self.model.text_encoder,
                sharding_strategy=config.sharding_strategy,
                mixed_precision=config.mixed_precision,
                wrap_strategy=config.text_encoder_fsdp_wrap_strategy
            )

        if not config.no_visualize or config.load_raw_video:
            self.model.vae = self.model.vae.to(
                device=self.device, dtype=torch.bfloat16 if config.mixed_precision else torch.float32)

        self.generator_optimizer = torch.optim.AdamW(
            [param for param in self.model.generator.parameters()
             if param.requires_grad],
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay
        )

        # Step 3: Initialize the dataloader
        if self.model_type == "egosim":
            dataset = EgoSimCacheDataset(
                config.data_path,
                max_pair=int(1e8),
                physics_track_mode=getattr(config, "physics_track_mode", "new"),
            )
        else:
            dataset = LatentLMDBDataset(config.data_path, max_pair=int(1e8))

        self.dataset = dataset
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=True, drop_last=True)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            num_workers=8)

        if dist.get_rank() == 0:
            print("DATASET SIZE %d" % len(dataset))
        self.dataloader = cycle(dataloader)

        ##############################################################################################################
        # 6. Set up EMA parameter containers
        rename_param = (
            lambda name: name.replace("_fsdp_wrapped_module.", "")
            .replace("_checkpoint_wrapped_module.", "")
            .replace("_orig_mod.", "")
        )
        self.name_to_trainable_params = {}
        for n, p in self.model.generator.named_parameters():
            if not p.requires_grad:
                continue

            renamed_n = rename_param(n)
            self.name_to_trainable_params[renamed_n] = p
        ema_weight = config.ema_weight
        self.generator_ema = None
        if (ema_weight is not None) and (ema_weight > 0.0):
            if self.is_main_process:
                print(f"Setting up EMA with weight {ema_weight}")
            self.generator_ema = EMA_FSDP(self.model.generator, decay=ema_weight)

        ##############################################################################################################
        # Let's delete EMA params for early steps to save some computes at training and inference
        if self.step < config.ema_start_step:
            self.generator_ema = None

        self.max_grad_norm = 10.0
        self.delta_mean = None
        self.rtf_ema_ratio = getattr(self.config, "rtf_ema_ratio", 0.9) 
        self.eval_interval = getattr(self.config, "eval_interval", 0)      # 0 => disable
        self.eval_frames = getattr(self.config, "eval_num_output_frames", 21)
        self.eval_init = getattr(self.config, "eval_num_init_frames", 3)
        self.rtf_single_gpu_batch = getattr(self.config, "rtf_single_gpu_batch", 1)
        self.given_first_chunk = getattr(self.config, "given_first_chunk", True)
        self.eval_runner = None
        if self.eval_interval:
            self.pipeline = CausalDiffusionInferencePipeline(config, device=self.device)
            self.pipeline.generator = self.model.generator
            self.pipeline.text_encoder = self.model.text_encoder
            
    def save(self):
        print("Start gathering distributed model states...")
        generator_state_dict = fsdp_state_dict(
            self.model.generator)

        if self.config.ema_start_step < self.step:
            state_dict = {
                "generator": generator_state_dict,
                "generator_ema": self.generator_ema.full_state_dict(self.model.generator),
            }
        else:
            state_dict = {
                "generator": generator_state_dict,
            }

        if self.is_main_process:
            os.makedirs(os.path.join(self.output_path,
                        f"checkpoint_model_{self.step:06d}"), exist_ok=True)
            torch.save(state_dict, os.path.join(self.output_path,
                       f"checkpoint_model_{self.step:06d}", "model.pt"))
            print("Model saved to", os.path.join(self.output_path,
                  f"checkpoint_model_{self.step:06d}", "model.pt"))

        if self.model_type != "egosim" or not getattr(
            self.config, "checkpoint_eval_enabled", False
        ):
            return
        dist.barrier()
        if self.is_main_process:
            print(f"[Eval] start step={self.step}", flush=True)
        if self.eval_runner is None:
            self.eval_runner = EgoSimCheckpointEvalRunner(
                self.config,
                device=torch.device(f"cuda:{self.device}"),
                dtype=self.dtype,
                generator=self.model.generator,
                rank=dist.get_rank(),
                world_size=self.world_size,
                is_main_process=self.is_main_process,
            )
        summary = self.eval_runner.run(self.step)
        if summary is not None and self.is_main_process and not self.disable_wandb:
            wandb.log(
                {f"eval/{key}": value for key, value in summary.items()
                 if key != "step" and isinstance(value, (int, float))},
                step=self.step,
            )

    def train_one_step(self, batch):
        self.log_iters = 1

        if self.step % 20 == 0:
            torch.cuda.empty_cache()

        # Step 1: Get the next batch of text prompts
        text_prompts = batch["prompts"]
        if self.model_type == "egosim":
            clean_latent = batch["clean_latent"].to(device=self.device, dtype=self.dtype)
            image_latent = None
            batch_size = clean_latent.shape[0]
            image_or_video_shape = list(clean_latent.shape)
            conditional_dict = {
                "prompt_embeds": batch["prompt_embeds"].to(device=self.device, dtype=self.dtype),
                "image_embeds": batch["image_embeds"].to(device=self.device, dtype=self.dtype),
                "ego_prior_latent": batch["ego_prior_latent"].to(device=self.device, dtype=self.dtype),
                "hand_latent": batch["hand_latent"].to(device=self.device, dtype=self.dtype),
                "mask_latent": batch["mask_latent"].to(device=self.device, dtype=self.dtype),
            }
            unconditional_dict = {}
        else:
            if not self.config.load_raw_video:  # precomputed latent
                clean_latent = batch["clean_latent"].to(
                    device=self.device, dtype=self.dtype)
            else:  # encode raw video to latent
                frames = batch["frames"].to(
                    device=self.device, dtype=self.dtype)

                with torch.no_grad():
                    clean_latent = self.model.vae.encode_to_latent(
                        frames).to(device=self.device, dtype=self.dtype)
            image_latent = clean_latent[:, 0:1, ]

            batch_size = len(text_prompts)
            image_or_video_shape = list(self.config.image_or_video_shape)
            image_or_video_shape[0] = batch_size

            # Step 2: Extract the conditional infos
            with torch.no_grad():
                conditional_dict = self.model.text_encoder(
                    text_prompts=text_prompts)
                if not getattr(self, "unconditional_dict", None):
                    unconditional_dict = self.model.text_encoder(
                        text_prompts=[self.config.negative_prompt] * batch_size)
                    unconditional_dict = {k: v.detach()
                                          for k, v in unconditional_dict.items()}
                    self.unconditional_dict = unconditional_dict  # cache the unconditional_dict
                else:
                    unconditional_dict = self.unconditional_dict

        # Step 3: Train the generator
        generator_loss, log_dict = self.model.generator_loss(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            clean_latent=clean_latent,
            initial_latent=image_latent,
        )
        self.generator_optimizer.zero_grad()
        generator_loss.backward()
        generator_grad_norm = self.model.generator.clip_grad_norm_(
            self.max_grad_norm)
        self.generator_optimizer.step()

        # Increment the step since we finished gradient update
        self.step += 1

        wandb_loss_dict = {
            "generator_loss": generator_loss.item(),
            "generator_grad_norm": generator_grad_norm.item(),
        }
        wandb_loss_dict.update({
            key: value.item()
            for key, value in log_dict.items()
            if isinstance(value, torch.Tensor) and value.ndim == 0
        })

        # Step 4: Logging
        if self.is_main_process:
            if not self.disable_wandb:
                wandb.log(wandb_loss_dict, step=self.step)

        if self.step % self.config.gc_interval == 0:
            if dist.get_rank() == 0:
                logging.info("DistGarbageCollector: Running GC.")
            gc.collect()


    def train(self):
        max_train_steps = int(getattr(self.config, "max_train_steps", 0) or 0)

        while True:
            step_start_time = time.perf_counter()
            batch = next(self.dataloader)
            self.train_one_step(batch)

            if graceful_stop_and_save(self):
                break
                
            if (not self.config.no_save) and self.step % self.config.log_iters == 0:
                torch.cuda.empty_cache()
                self.save()
                torch.cuda.empty_cache()

            barrier()
            if self.is_main_process and not self.disable_wandb:
                wandb.log(
                    {"per iteration time": time.perf_counter() - step_start_time},
                    step=self.step,
                )

            if max_train_steps > 0 and self.step >= max_train_steps:
                if self.is_main_process:
                    print(f"Reached max_train_steps={max_train_steps}; stopping training.")
                if not self.config.no_save and self.step % self.config.log_iters != 0:
                    torch.cuda.empty_cache()
                    self.save()
                    torch.cuda.empty_cache()
                break
