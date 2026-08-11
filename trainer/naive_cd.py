import gc
import logging
from utils.dataset import cycle
from utils.dataset import LatentLMDBDataset, EgoSimCacheDataset
from utils.distributed import ShardedEMA_FSDP, fsdp_wrap, fsdp_state_dict, graceful_stop_and_save, launch_distributed_job
from utils.misc import set_seed
from utils.egosim_checkpoint_eval import EgoSimCheckpointEvalRunner
import torch.distributed as dist
from omegaconf import OmegaConf
import torch
import wandb
import time
import os
from model import NaiveConsistency


def _extract_generator_state_dict(checkpoint):
    for key in ("generator", "model", "generator_ema"):
        if key in checkpoint:
            checkpoint = checkpoint[key]
            break
    return {
        key.replace("model._fsdp_wrapped_module.", "model.", 1)
        if key.startswith("model._fsdp_wrapped_module.")
        else key: value
        for key, value in checkpoint.items()
    }


class Trainer:
    def __init__(self, config):
        self.config = config
        self.step = int(getattr(config, "initial_step", 0) or 0)

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
            wandb.init(
                config=OmegaConf.to_container(config, resolve=True),
                name=config.config_name,
                mode="online",
                entity=config.wandb_entity,
                project=config.wandb_project,
                dir=config.wandb_save_dir
            )

        self.output_path = config.logdir

        # Step 2: Initialize the model and optimizer
        self.model = NaiveConsistency(config, device=self.device)
        generator_ckpt = getattr(config, "generator_ckpt", False)
        teacher_ckpt = getattr(config, "teacher_ckpt", generator_ckpt)
        sync_generator_ckpt = bool(generator_ckpt and self.model_type == "egosim")
        def load_rank0_state(path, role):
            if not (sync_generator_ckpt and self.is_main_process):
                return None
            print(f"Loading full {role} checkpoint on rank 0 from {path}")
            checkpoint = torch.load(
                path, map_location="cpu", mmap=True, weights_only=True
            )
            return _extract_generator_state_dict(checkpoint)

        def wrap_model(module, wrap_strategy, state_dict):
            if state_dict is not None:
                module.model.to_empty(device="cpu")
                module.load_state_dict(state_dict, strict=True, assign=True)
            if sync_generator_ckpt:
                dist.barrier()
            return fsdp_wrap(
                module,
                sharding_strategy=config.sharding_strategy,
                mixed_precision=config.mixed_precision,
                wrap_strategy=wrap_strategy,
                cpu_offload=True,
                sync_module_states=sync_generator_ckpt,
            )

        student_state = load_rank0_state(generator_ckpt, "student")
        self.model.generator = wrap_model(
            self.model.generator,
            config.generator_fsdp_wrap_strategy,
            student_state,
        )
        
        self.model.generator_ema = wrap_model(
            self.model.generator_ema,
            config.generator_fsdp_wrap_strategy,
            student_state,
        )
        del student_state
        gc.collect()

        teacher_state = load_rank0_state(teacher_ckpt, "teacher")
        self.model.teacher = wrap_model(
            self.model.teacher,
            config.real_score_fsdp_wrap_strategy,
            teacher_state,
        )
        del teacher_state
        gc.collect()

        if self.model_type != "egosim":
            self.model.text_encoder = fsdp_wrap(
                self.model.text_encoder,
                sharding_strategy=config.sharding_strategy,
                mixed_precision=config.mixed_precision,
                wrap_strategy=config.text_encoder_fsdp_wrap_strategy,
                cpu_offload=True
            )

        ##############################################################################################################
        # Non-EgoSim checkpoints retain the original loading path.
        if generator_ckpt and not sync_generator_ckpt:
            if self.is_main_process:
                print(f"Loading pretrained generator from {generator_ckpt}")
            state_dict = _extract_generator_state_dict(
                torch.load(generator_ckpt, map_location="cpu", mmap=True)
            )

            self.model.generator.load_state_dict(state_dict, strict=True)
            self.model.teacher.load_state_dict(state_dict, strict=True)
            self.model.generator_ema.load_state_dict(state_dict, strict=True)
            del state_dict
            gc.collect()

        ##############################################################################################################
        # 4. Initialize optimizer and EMA after checkpoint loading, so the EMA
        # shadow starts from the pretrained generator instead of random weights.
        self.generator_optimizer = torch.optim.AdamW(
            [param for param in self.model.generator.parameters()
             if param.requires_grad],
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay
        )
        
        ema_weight = config.ema_weight
        self.generator_ema = None
        if (ema_weight is not None) and (ema_weight > 0.0):
            if self.is_main_process:
                print(f"Setting up EMA with weight {ema_weight}")
            self.generator_ema = ShardedEMA_FSDP(
                self.model.generator, decay=ema_weight
            )

        # Step 5: Initialize the dataloader
 
        if self.model_type == "egosim":
            dataset = EgoSimCacheDataset(
                config.data_path,
                max_pair=int(1e8),
                physics_track_mode=getattr(config, "physics_track_mode", "new"),
            )
        else:
            dataset = LatentLMDBDataset(
                config.data_path, max_pair=int(1e8))
        
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=True, drop_last=True)
        num_workers = int(os.environ.get("DATALOADER_NUM_WORKERS", "8"))
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            num_workers=num_workers)

        if dist.get_rank() == 0:
            print("DATASET SIZE %d" % len(dataset))
        self.dataloader = cycle(dataloader)

        
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

        #############################################################################################################
        self.max_grad_norm_generator = getattr(config, "max_grad_norm_generator", 10.0)
        self.max_grad_norm_critic = getattr(config, "max_grad_norm_critic", 10.0)
        self.previous_time = None
        self.eval_runner = None
        
        

    def save(self):
        print("Start gathering distributed model states...")
        if self.generator_ema is not None and self.config.ema_start_step < self.step:
            print("Syncing EMA shadow to the EMA FSDP model...")
            self.generator_ema.copy_to(self.model.generator_ema)
            state_dict = {
                "generator_ema": fsdp_state_dict(self.model.generator_ema),
            }
        else:
            state_dict = {
                "generator": fsdp_state_dict(self.model.generator),
            }
        print("Finished gathering distributed model states.")

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
        eval_generator = (
            self.model.generator_ema
            if self.generator_ema is not None and self.config.ema_start_step < self.step
            else self.model.generator
        )
        if self.is_main_process:
            print(f"[Eval] start step={self.step}", flush=True)
        if self.eval_runner is None or self.eval_runner.generator is not eval_generator:
            self.eval_runner = EgoSimCheckpointEvalRunner(
                self.config,
                device=torch.device(f"cuda:{self.device}"),
                dtype=self.dtype,
                generator=eval_generator,
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

            
    def fwdbwd_one_step(self, batch, clean_latent=None):
        self.model.eval()

        if self.step % 20 == 0:
            torch.cuda.empty_cache()

        # Step 1: Get the next batch of text prompts
        text_prompts = batch["prompts"]
        batch_size = len(text_prompts)

        # Step 2: Extract the conditional infos
        if self.model_type == "egosim":
            conditional_dict = {
                "prompt_embeds": batch["prompt_embeds"].to(device=self.device, dtype=self.dtype),
                "image_embeds": batch["image_embeds"].to(device=self.device, dtype=self.dtype),
                "ego_prior_latent": batch["ego_prior_latent"].to(device=self.device, dtype=self.dtype),
                "hand_latent": batch["hand_latent"].to(device=self.device, dtype=self.dtype),
                "mask_latent": batch["mask_latent"].to(device=self.device, dtype=self.dtype),
            }
            unconditional_dict = conditional_dict
        else:
            image_or_video_shape = list(self.config.image_or_video_shape)
            image_or_video_shape[0] = batch_size
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

        # Step 3: Store gradients for the generator (if training the generator)
        generator_loss, generator_log_dict = self.model.generator_loss(
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            clean_latent=clean_latent,
            ema_model = self.generator_ema
        )
        generator_loss.backward()
        generator_grad_norm = self.model.generator.clip_grad_norm_(
            self.max_grad_norm_generator)

        generator_log_dict.update({"generator_loss": generator_loss,
                                    "generator_grad_norm": generator_grad_norm})

        return generator_log_dict
        

   

    def train(self):
        start_step = self.step
        max_train_steps = int(getattr(self.config, "max_train_steps", 0) or 0)

        while True:

            self.generator_optimizer.zero_grad(set_to_none=True)

            batch = next(self.dataloader)
            generator_log_dict = self.fwdbwd_one_step(batch, clean_latent=batch["clean_latent"])
            

            self.generator_optimizer.step()
            if self.generator_ema is not None:
                self.generator_ema.update(self.model.generator)
            
              

            # Increment the step since we finished gradient update
            self.step += 1

            if graceful_stop_and_save(self):
                break

           
            # Save the model
            if (not self.config.no_save) and (self.step - start_step) > 0 and self.step % self.config.log_iters == 0:
                torch.cuda.empty_cache()
                self.save()
                torch.cuda.empty_cache()

            # Logging
            if self.is_main_process:
                wandb_loss_dict = {}
                wandb_loss_dict.update(
                        {
                            "generator_loss": generator_log_dict["generator_loss"].mean().item(),
                            "generator_grad_norm": generator_log_dict["generator_grad_norm"].mean().item()
                        }
                    )

              

                if not self.disable_wandb:
                    wandb.log(wandb_loss_dict, step=self.step)

            if self.step % self.config.gc_interval == 0:
                if dist.get_rank() == 0:
                    logging.info("DistGarbageCollector: Running GC.")
                gc.collect()
                torch.cuda.empty_cache()

            if self.is_main_process:
                current_time = time.time()
                if self.previous_time is None:
                    self.previous_time = current_time
                else:
                    if not self.disable_wandb:
                        wandb.log({"per iteration time": current_time - self.previous_time}, step=self.step)
                    self.previous_time = current_time

            if max_train_steps > 0 and self.step >= max_train_steps:
                if self.is_main_process:
                    print(f"Reached max_train_steps={max_train_steps}; stopping training.")
                if not self.config.no_save and self.step % self.config.log_iters != 0:
                    torch.cuda.empty_cache()
                    self.save()
                    torch.cuda.empty_cache()
                break
