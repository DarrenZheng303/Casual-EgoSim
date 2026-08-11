import argparse
from copy import deepcopy
import os
from omegaconf import OmegaConf
import wandb

from trainer import (
    ConsistencyDistillationTrainer,
    DiffusionTrainer,
    EgoSimScoreDistillationTrainer,
    ODETrainer,
    ScoreDistillationTrainer,
)


def _mask_sensitive_config_values(value):
    if isinstance(value, dict):
        masked = {}
        for key, item in value.items():
            key_lower = str(key).lower()
            if any(token in key_lower for token in ("key", "token", "secret", "password")):
                masked[key] = "***"
            else:
                masked[key] = _mask_sensitive_config_values(item)
        return masked
    if isinstance(value, list):
        return [_mask_sensitive_config_values(item) for item in value]
    return value


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--no_visualize", action="store_true")
    parser.add_argument("--logdir", type=str, default="", help="Path to the directory to save logs")
    parser.add_argument("--wandb-save-dir", type=str, default="", help="Path to the directory to save wandb logs")
    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument("--tf", action="store_true")

    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)
    default_config = OmegaConf.load("configs/default_config.yaml")
    config = OmegaConf.merge(default_config, config)
    config.no_save = args.no_save
    config.no_visualize = args.no_visualize
    config.tf = args.tf 
    # get the filename of config_path
    config_name = os.path.basename(args.config_path).split(".")[0]
    config.config_name = config_name
    config.logdir = args.logdir
    config.wandb_save_dir = args.wandb_save_dir
    config.disable_wandb = args.disable_wandb

    egosim_model_root = os.environ.get("EGOSIM_MODEL_ROOT")
    if egosim_model_root:
        config.egosim_model_root = egosim_model_root

    rank = int(os.environ.get("RANK", "0"))
    if rank == 0:
        resolved_config = OmegaConf.to_container(config, resolve=True)
        printable_config = _mask_sensitive_config_values(deepcopy(resolved_config))
        print("\n========== Effective Config ==========")
        print(f"config_path: {args.config_path}")
        print(OmegaConf.to_yaml(OmegaConf.create(printable_config), resolve=True))
        print("======== End Effective Config ========\n", flush=True)

    if config.trainer == "diffusion":
        trainer = DiffusionTrainer(config)
    elif config.trainer == "ode":
        trainer = ODETrainer(config)
    elif config.trainer == "score_distillation":
        trainer = ScoreDistillationTrainer(config)
    elif config.trainer == "consistency_distillation":
        trainer = ConsistencyDistillationTrainer(config)
    elif config.trainer == "egosim_score_distillation":
        trainer = EgoSimScoreDistillationTrainer(config)
    else:
        raise ValueError(f"Unknown trainer: {config.trainer}")
    trainer.train()

    wandb.finish()


if __name__ == "__main__":
    main()
