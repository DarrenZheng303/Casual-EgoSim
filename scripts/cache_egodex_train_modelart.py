#!/usr/bin/env python3
"""One-shot Egodex cache builder with physics-mask generation for ModelArts/local runs.

This script is a non-watcher replacement for scripts/watch_egodex_cache_modelart.py.
It launches distributed cache jobs over the train and eval splits by default, while preserving:
- per-sample incremental saves
- skip-existing resume
- multi-node torchrun sharding

Extra outputs added on top of the original cache set:
- grounded_sam_tracks.pt
- grounded_sam_visibility.pt
- grounded_sam_object_union_mask.png
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import signal
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import imageio
import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.egosim_encoders import (
    encode_first_frame,
    encode_mask_to_latent,
    encode_prompt,
    encode_video,
    load_image_encoder,
    load_text_encoder,
    load_vae,
)


DEFAULT_SFS_ROOT = Path("/home/ma-user/work")
DEFAULT_USER_ROOT = DEFAULT_SFS_ROOT / "users" / "zhengshikang"
DEFAULT_MODEL_ROOT = DEFAULT_SFS_ROOT / "model" / "EgoSim-14B"
DEFAULT_MODEL_FILE_ROOT = DEFAULT_SFS_ROOT / "model"
DEFAULT_PYTHON_BIN = DEFAULT_USER_ROOT / "conda" / "envs" / "causal_forcing" / "bin" / "python"
DEFAULT_DATASET_BASE = DEFAULT_USER_ROOT / "datasets" / "luyitas" / "egosim_egodex_egovid"
DEFAULT_RAW_EGODEX_ROOT = (
    DEFAULT_USER_ROOT
    / "datasets"
    / "modelscope"
    / "datasets"
    / "luyitas"
    / "egosim_egodex_egovid_full"
    / "Egodex"
)
DEFAULT_DATASET_ROOT = DEFAULT_RAW_EGODEX_ROOT
DEFAULT_CACHEABLE_METADATA = "filtered_metadata_cacheable.csv"
DEFAULT_METADATA_PATH = DEFAULT_DATASET_BASE / "train" / DEFAULT_CACHEABLE_METADATA
DEFAULT_OUTPUT_ROOT = DEFAULT_DATASET_BASE / "cache"
DEFAULT_LOG_DIR = DEFAULT_DATASET_BASE / "cache_logs"
DEFAULT_COTRACKER_CKPT = DEFAULT_MODEL_FILE_ROOT / "scaled_offline.pth"
DEFAULT_GROUNDING_DINO_MODEL_ROOT = DEFAULT_MODEL_FILE_ROOT / "grounding-dino-base"
DEFAULT_SAM2_MODEL_ROOT = DEFAULT_MODEL_FILE_ROOT / "sam2.1-hiera-large"

DATASET_NAME = "egodex"
COMMON_OUTPUT_FILES = [
    "clean_latent.pt",
    "ego_prior_latent.pt",
    "hand_latent.pt",
    "mask_latent.pt",
    "prompt_embedding.pt",
    "image_embedding.pt",
    "meta.json",
]

GROUNDED_SAM_OUTPUT_FILES = [
    "grounded_sam_tracks.pt",
    "grounded_sam_visibility.pt",
    "grounded_sam_object_masks.pt",
    "grounded_sam_object_union_mask.png",
    "grounding_dino_boxes.png",
]

STOP_REQUESTED = False
NODE_LOCK_ACQUIRED = False


@dataclass(frozen=True)
class EgoDexSample:
    video_id: str
    output_id: str
    prompt: str
    short_prompt: str
    video: str
    ego_prior_video: str
    hand_keypoint_video: str
    first_frame: str


@dataclass(frozen=True)
class RuntimePaths:
    split: str
    sfs_root: Path
    user_root: Path
    model_root: Path
    python_bin: Path
    dataset_base: Path
    dataset_root: Path
    metadata_path: Path
    output_root: Path
    log_dir: Path
    cotracker_checkpoint: Path
    grounding_dino_model_root: Path
    sam2_model_root: Path


def request_stop(signum: int, _frame: Any) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build EgoSim cache for Egodex train/eval splits with extra physics masks."
    )
    parser.add_argument("--split", choices=["train", "eval", "all"], default=os.environ.get("SPLIT", "all"))
    parser.add_argument("--sfs_root", type=Path, default=Path(os.environ.get("SFS_ROOT", str(DEFAULT_SFS_ROOT))))
    parser.add_argument("--user_root", type=Path, default=Path(os.environ.get("USER_ROOT", str(DEFAULT_USER_ROOT))))
    parser.add_argument("--model_root", type=Path, default=Path(os.environ.get("MODEL_ROOT", str(DEFAULT_MODEL_ROOT))))
    parser.add_argument("--python_bin", type=Path, default=Path(os.environ.get("PYTHON_BIN", str(DEFAULT_PYTHON_BIN))))
    parser.add_argument("--dataset_base", type=Path, default=Path(os.environ.get("DATASET_BASE", str(DEFAULT_DATASET_BASE))))
    parser.add_argument("--dataset_root", type=Path, default=Path(os.environ.get("DATASET_ROOT", str(DEFAULT_DATASET_ROOT))))
    parser.add_argument("--metadata_path", type=Path, default=Path(os.environ.get("METADATA_PATH", str(DEFAULT_METADATA_PATH))))
    parser.add_argument("--output_root", type=Path, default=Path(os.environ.get("OUTPUT_ROOT", str(DEFAULT_OUTPUT_ROOT))))
    parser.add_argument("--log_dir", type=Path, default=Path(os.environ.get("LOG_DIR", str(DEFAULT_LOG_DIR))))
    parser.add_argument("--cotracker_checkpoint", type=Path, default=Path(os.environ.get("COTRACKER_CHECKPOINT", str(DEFAULT_COTRACKER_CKPT))))
    parser.add_argument("--grounding_dino_model_root", type=Path, default=Path(os.environ.get("GROUNDING_DINO_MODEL_ROOT", str(DEFAULT_GROUNDING_DINO_MODEL_ROOT))))
    parser.add_argument("--sam2_model_root", type=Path, default=Path(os.environ.get("SAM2_MODEL_ROOT", str(DEFAULT_SAM2_MODEL_ROOT))))
    parser.add_argument("--height", type=int, default=int(os.environ.get("HEIGHT", "480")))
    parser.add_argument("--width", type=int, default=int(os.environ.get("WIDTH", "832")))
    parser.add_argument("--num_frames", type=int, default=int(os.environ.get("NUM_FRAMES", "61")))
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--skip_existing", action="store_true", default=os.environ.get("SKIP_EXISTING", "1") == "1")
    parser.add_argument("--no_skip_existing", action="store_false", dest="skip_existing")
    parser.add_argument("--physics_grid_size", type=int, default=int(os.environ.get("PHYSICS_GRID_SIZE", "50")))
    parser.add_argument("--gdino_box_threshold", type=float, default=float(os.environ.get("GDINO_BOX_THRESHOLD", "0.35")))
    parser.add_argument("--gdino_text_threshold", type=float, default=float(os.environ.get("GDINO_TEXT_THRESHOLD", "0.25")))
    parser.add_argument("--master_port", type=int, default=int(os.environ.get("MASTER_PORT", "6069")))
    parser.add_argument("--rdzv_id", default=os.environ.get("RDZV_ID", os.environ.get("MA_JOB_NAME", "cache_egodex_train_modelart")))
    parser.add_argument("--nproc_per_node", type=int, default=int(os.environ.get("NPROC_PER_NODE", os.environ.get("MA_NUM_GPUS", "8"))))
    parser.add_argument("--nnodes", type=int, default=int(os.environ.get("NNODES", os.environ.get("MA_NUM_HOSTS", os.environ.get("VC_WORKER_NUM", "1")))))
    parser.add_argument("--node_rank", type=int, default=int(os.environ.get("NODE_RANK", os.environ.get("VC_TASK_INDEX", "0"))))
    parser.add_argument("--master_addr", default=os.environ.get("MASTER_ADDR"))
    parser.add_argument("--distributed_worker", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def resolve_runtime_paths(args: argparse.Namespace) -> RuntimePaths:
    default_split_root = args.dataset_base / args.split
    default_train_metadata = args.dataset_base / "train" / DEFAULT_CACHEABLE_METADATA
    default_split_metadata = default_split_root / DEFAULT_CACHEABLE_METADATA

    dataset_root = args.dataset_root
    metadata_path = args.metadata_path
    output_root = args.output_root
    log_dir = args.log_dir

    if metadata_path == default_train_metadata:
        metadata_path = default_split_metadata
    if output_root == DEFAULT_OUTPUT_ROOT:
        output_root = output_root / args.split
    if log_dir == args.dataset_base / "cache_logs":
        log_dir = log_dir / args.split

    return RuntimePaths(
        split=args.split,
        sfs_root=args.sfs_root,
        user_root=args.user_root,
        model_root=args.model_root,
        python_bin=args.python_bin,
        dataset_base=args.dataset_base,
        dataset_root=dataset_root,
        metadata_path=metadata_path,
        output_root=output_root,
        log_dir=log_dir,
        cotracker_checkpoint=args.cotracker_checkpoint,
        grounding_dino_model_root=args.grounding_dino_model_root,
        sam2_model_root=args.sam2_model_root,
    )


def ensure_required_paths(paths: RuntimePaths) -> None:
    required = {
        "sfs_root": paths.sfs_root,
        "model_root": paths.model_root,
        "python_bin": paths.python_bin,
        "dataset_root": paths.dataset_root,
        "metadata_path": paths.metadata_path,
        "cotracker_checkpoint": paths.cotracker_checkpoint,
        "grounding_dino_model_root": paths.grounding_dino_model_root,
        "sam2_model_root": paths.sam2_model_root,
    }
    missing = [f"{name}={value}" for name, value in required.items() if not value.exists()]
    if missing:
        raise FileNotFoundError("Missing required paths:\n" + "\n".join(missing))


def resolve_master_addr(args: argparse.Namespace) -> str:
    if args.nnodes == 1:
        return args.master_addr or "127.0.0.1"
    if args.master_addr:
        return args.master_addr
    worker_hosts = os.environ.get("VC_WORKER_HOSTS", "")
    if not worker_hosts:
        raise RuntimeError("Multi-node mode requires MASTER_ADDR or VC_WORKER_HOSTS.")
    return worker_hosts.split(",")[0]


def build_logger(log_dir: Path, node_rank: int, worker_mode: bool) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    role = "worker" if worker_mode else "launcher"
    rank_tag = os.environ.get("RANK", "na")
    local_rank_tag = os.environ.get("LOCAL_RANK", "na")
    pid_tag = os.getpid()
    logger_name = f"cache_egodex_train_modelart_node{node_rank}_{role}_rank{rank_tag}_local{local_rank_tag}_pid{pid_tag}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / (
        f"{role}_node{node_rank}_rank{rank_tag}_local{local_rank_tag}_pid{pid_tag}_{timestamp}.log"
    )
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.info("Logging to terminal and %s", log_path)
    return logger


def node_lock_path(log_dir: Path, node_rank: int) -> Path:
    return log_dir / f"cache_egodex_train_modelart_node{node_rank}.lock"


def pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def acquire_node_lock(log_dir: Path, node_rank: int) -> None:
    global NODE_LOCK_ACQUIRED
    lock_path = node_lock_path(log_dir, node_rank)
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        try:
            owner_text = lock_path.read_text(encoding="utf-8").strip()
            owner_pid = int(owner_text)
        except (OSError, ValueError):
            owner_pid = -1
        if not pid_is_alive(owner_pid):
            lock_path.unlink(missing_ok=True)
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        else:
            raise RuntimeError(f"Node lock already exists: {lock_path}") from exc
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(str(os.getpid()))
    NODE_LOCK_ACQUIRED = True


def release_node_lock(log_dir: Path, node_rank: int) -> None:
    global NODE_LOCK_ACQUIRED
    if not NODE_LOCK_ACQUIRED:
        return
    lock_path = node_lock_path(log_dir, node_rank)
    try:
        owner = lock_path.read_text(encoding="utf-8").strip()
    except OSError:
        owner = ""
    if owner and owner != str(os.getpid()):
        NODE_LOCK_ACQUIRED = False
        return
    lock_path.unlink(missing_ok=True)
    NODE_LOCK_ACQUIRED = False


def init_distributed() -> tuple[int, int, int, torch.device]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        dist.init_process_group("nccl")
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    return rank, world_size, local_rank, device


def load_samples(metadata_path: Path, max_samples: int | None) -> list[EgoDexSample]:
    samples: list[EgoDexSample] = []
    with metadata_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            video = str(row["video"])
            video_id = video.replace("/", "_").replace(".mp4", "")
            parts = video.split("/", 1)
            output_id = (parts[1] if len(parts) > 1 else video).replace("/", "_").replace(".mp4", "")
            samples.append(
                EgoDexSample(
                    video_id=video_id,
                    output_id=output_id,
                    prompt=str(row.get("prompt", "")),
                    short_prompt=str(row.get("short_prompt", "")),
                    video=video,
                    ego_prior_video=str(row.get("ego_prior_video", "")),
                    hand_keypoint_video=str(row.get("hand_keypoint_video", "")),
                    first_frame=str(row.get("first_frame", "")),
                )
            )
            if max_samples is not None and len(samples) >= max_samples:
                break
    return samples


def get_sample_paths(dataset_root: Path, sample: EgoDexSample) -> dict[str, Path]:
    target = dataset_root / sample.video
    ego_prior = dataset_root / sample.ego_prior_video
    hand = dataset_root / sample.hand_keypoint_video
    mask = ego_prior.parent / "pc_mask_video.mp4"
    hand_seg = ego_prior.parent / "hand_seg.png"
    first_frame = dataset_root / sample.first_frame if sample.first_frame else Path()
    return {
        "target": target,
        "ego_prior": ego_prior,
        "hand": hand,
        "mask": mask,
        "hand_seg": hand_seg,
        "first_frame": first_frame,
    }


def sample_output_dir(output_root: Path, sample: EgoDexSample) -> Path:
    return output_root / DATASET_NAME / sample.video_id


def expected_output_files() -> list[str]:
    return COMMON_OUTPUT_FILES + GROUNDED_SAM_OUTPUT_FILES


def is_complete(out_dir: Path) -> bool:
    return all((out_dir / name).exists() for name in expected_output_files())


def atomic_save_tensor(path: Path, tensor: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.rank{os.environ.get('RANK', '0')}.pid{os.getpid()}.tmp")
    torch.save(tensor.detach().cpu().contiguous(), tmp_path)
    tmp_path.replace(path)


def atomic_save_object(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.rank{os.environ.get('RANK', '0')}.pid{os.getpid()}.tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.rank{os.environ.get('RANK', '0')}.pid{os.getpid()}.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    tmp_path.replace(path)


def ensure_tensor_cache(
    path: Path,
    build_fn: Any,
    *,
    save_dtype: torch.dtype | None = torch.bfloat16,
    load_tensor: bool = False,
) -> torch.Tensor | None:
    if path.exists():
        return torch.load(path, map_location="cpu") if load_tensor else None

    tensor = build_fn()
    tensor_to_save = tensor.to(dtype=save_dtype) if save_dtype is not None else tensor
    atomic_save_tensor(path, tensor_to_save)
    if load_tensor:
        return tensor_to_save.detach().cpu().contiguous()
    return None


def physics_output_paths(out_dir: Path) -> dict[str, Path]:
    return {
        "track": out_dir / "grounded_sam_tracks.pt",
        "visibility": out_dir / "grounded_sam_visibility.pt",
        "object_masks": out_dir / "grounded_sam_object_masks.pt",
        "mask": out_dir / "grounded_sam_object_union_mask.png",
        "boxes_viz": out_dir / "grounding_dino_boxes.png",
    }


def build_physics_cache(
    sample: EgoDexSample,
    paths: dict[str, Path],
    out_dir: Path,
    physics_models: dict[str, Any],
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, Any]:
    output_paths = physics_output_paths(out_dir)
    need_outputs = any(not path.exists() for path in output_paths.values())
    physics_prompt = sample.short_prompt.strip()
    if not physics_prompt:
        raise ValueError(f"Missing short_prompt for sample {sample.video_id}")
    physics_debug: dict[str, Any] = {}

    if need_outputs:
        from physics_grounding_sam import (
            generate_grounded_sam_physics_tracks,
            save_grounding_dino_boxes,
            save_binary_mask as save_binary_mask_fn,
        )

        tracks_np, visibility_np, object_union_mask, sam_instances, first_frame, detections, physics_debug = generate_grounded_sam_physics_tracks(
            target_video_path=paths["target"],
            hand_seg_path=paths["hand_seg"],
            text_prompt=physics_prompt,
            models=physics_models,
            device=device,
            grid_size=args.physics_grid_size,
            box_threshold=args.gdino_box_threshold,
            text_threshold=args.gdino_text_threshold,
        )
        save_binary_mask_fn(object_union_mask, output_paths["mask"])
        save_grounding_dino_boxes(first_frame, detections, output_paths["boxes_viz"])
        atomic_save_object(
            output_paths["object_masks"],
            {
                "masks": torch.from_numpy(sam_instances["masks"]).to(torch.bool),
                "labels": list(sam_instances["labels"]),
                "scores": torch.from_numpy(sam_instances["scores"]),
                "boxes": torch.from_numpy(sam_instances["boxes"]),
                "track_object_ids": torch.from_numpy(sam_instances["track_object_ids"]).to(torch.int16),
                "height": int(sam_instances["height"]),
                "width": int(sam_instances["width"]),
            },
        )

        atomic_save_tensor(output_paths["track"], torch.from_numpy(tracks_np))
        atomic_save_tensor(output_paths["visibility"], torch.from_numpy(visibility_np))

    tracks = torch.load(output_paths["track"], map_location="cpu")
    visibility = torch.load(output_paths["visibility"], map_location="cpu")
    object_union_mask = load_binary_mask(output_paths["mask"]).astype(np.uint8)
    object_masks_path = output_paths.get("object_masks")
    num_object_masks = None
    if object_masks_path is not None and object_masks_path.exists():
        num_object_masks = int(torch.load(object_masks_path, map_location="cpu")["masks"].shape[0])
    return {
        "prompt": physics_prompt,
        "track_path": output_paths["track"],
        "visibility_path": output_paths["visibility"],
        "mask_path": output_paths["mask"],
        "object_masks_path": object_masks_path,
        "num_object_masks": num_object_masks,
        "track_shape": list(tracks.shape),
        "visibility_shape": list(visibility.shape),
        "mask_area": int(object_union_mask.sum()),
        "debug": physics_debug,
        "recomputed": need_outputs,
    }


def load_mask_video(mask_path: Path, target_frames: int, height: int, width: int) -> torch.Tensor:
    reader = imageio.get_reader(str(mask_path))
    frames = []
    for frame_data in reader:
        frame = Image.fromarray(frame_data).resize((width, height), Image.BILINEAR)
        frames.append(frame)
    reader.close()

    if len(frames) < target_frames:
        last = frames[-1] if frames else Image.new("RGB", (width, height))
        frames += [last] * (target_frames - len(frames))
    else:
        frames = frames[:target_frames]

    arr = np.stack([np.array(frame) for frame in frames], axis=0)
    tensor = torch.from_numpy(arr).float().permute(3, 0, 1, 2)
    tensor = tensor / 255.0 * 2.0 - 1.0
    tensor = tensor.unsqueeze(0)
    mask_video_raw = (tensor + 1.0) / 2.0
    mask_video_raw = mask_video_raw.clamp(0, 1)
    mask_video_raw = mask_video_raw[:, :1].squeeze(0)
    mask_video_raw[:, 0, :, :] = 0.0
    return mask_video_raw


def load_video_frames_uint8(video_path: Path) -> np.ndarray:
    reader = imageio.get_reader(str(video_path))
    frames = [np.array(frame) for frame in reader]
    reader.close()
    if not frames:
        raise RuntimeError(f"Empty video: {video_path}")
    return np.stack(frames, axis=0)


def load_binary_mask(mask_path: Path) -> np.ndarray:
    arr = np.array(Image.open(mask_path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr > 0


def ensure_first_frame_image(first_frame_path: Path, target_video_path: Path, out_dir: Path) -> Path:
    if first_frame_path.exists():
        return first_frame_path
    fallback = out_dir / "first_frame_fallback.png"
    if fallback.exists():
        return fallback
    frame = load_video_frames_uint8(target_video_path)[0]
    Image.fromarray(frame).save(fallback)
    return fallback


def build_physics_models(paths: RuntimePaths, device: torch.device) -> dict[str, Any]:
    from physics_grounding_sam import build_grounded_sam_cotracker_models

    return build_grounded_sam_cotracker_models(
        grounding_dino_model_root=paths.grounding_dino_model_root,
        sam2_model_root=paths.sam2_model_root,
        cotracker_checkpoint=paths.cotracker_checkpoint,
        device=device,
    )


def cache_one_sample(
    models: dict[str, Any],
    sample: EgoDexSample,
    paths: dict[str, Path],
    out_dir: Path,
    physics_models: dict[str, Any],
    device: torch.device,
    args: argparse.Namespace,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    clean_latent_path = out_dir / "clean_latent.pt"
    ego_prior_latent_path = out_dir / "ego_prior_latent.pt"
    hand_latent_path = out_dir / "hand_latent.pt"
    mask_latent_path = out_dir / "mask_latent.pt"
    prompt_embedding_path = out_dir / "prompt_embedding.pt"
    image_embedding_path = out_dir / "image_embedding.pt"
    meta_path = out_dir / "meta.json"
    need_meta = not meta_path.exists()

    clean_latent = ensure_tensor_cache(
        clean_latent_path,
        lambda: encode_video(
            models["vae"],
            str(paths["target"]),
            device,
            target_frames=args.num_frames,
            height=args.height,
            width=args.width,
        ),
        load_tensor=(not mask_latent_path.exists()) or need_meta,
    )
    ensure_tensor_cache(
        ego_prior_latent_path,
        lambda: encode_video(
            models["vae"],
            str(paths["ego_prior"]),
            device,
            target_frames=args.num_frames,
            height=args.height,
            width=args.width,
        ),
    )
    ensure_tensor_cache(
        hand_latent_path,
        lambda: encode_video(
            models["vae"],
            str(paths["hand"]),
            device,
            target_frames=args.num_frames,
            height=args.height,
            width=args.width,
        ),
    )
    ensure_tensor_cache(
        prompt_embedding_path,
        lambda: encode_prompt(models["text_encoder"], sample.prompt, device),
    )
    ensure_tensor_cache(
        image_embedding_path,
        lambda: encode_first_frame(
            models["image_encoder"],
            str(ensure_first_frame_image(paths["first_frame"], paths["target"], out_dir)),
            device,
            height=args.height,
            width=args.width,
        ),
    )
    if not mask_latent_path.exists():
        if clean_latent is None:
            clean_latent = torch.load(clean_latent_path, map_location="cpu")
        target_frames_pixel = clean_latent.shape[1] * 4 + 1
        if not paths["mask"].exists():
            raise FileNotFoundError(f"Missing mask for sample {sample.video_id}: {paths['mask']}")
        mask_video_raw = load_mask_video(
            paths["mask"],
            target_frames=target_frames_pixel,
            height=args.height,
            width=args.width,
        )

        ensure_tensor_cache(
            mask_latent_path,
            lambda: encode_mask_to_latent(
                mask_video_raw,
                (16, *clean_latent.shape[1:]),
            ),
        )

    physics_info = build_physics_cache(
        sample=sample,
        paths=paths,
        out_dir=out_dir,
        physics_models=physics_models,
        device=device,
        args=args,
    )

    if not (need_meta or physics_info["recomputed"]):
        return

    if clean_latent is None:
        clean_latent = torch.load(clean_latent_path, map_location="cpu")

    meta = {
        "dataset": DATASET_NAME,
        "split": args.split,
        "video_id": sample.video_id,
        "output_id": sample.output_id,
        "prompt": sample.prompt,
        "short_prompt": sample.short_prompt,
        "target_video": str(paths["target"]),
        "ego_prior_video": str(paths["ego_prior"]),
        "hand_video": str(paths["hand"]),
        "hand_seg": str(paths["hand_seg"]),
        "mask_video": str(paths["mask"]),
        "first_frame": str(ensure_first_frame_image(paths["first_frame"], paths["target"], out_dir)),
        "num_frames": args.num_frames,
        "height": args.height,
        "width": args.width,
        "latent_shape": list(clean_latent.shape),
        "physics_track_shape": physics_info["track_shape"],
        "physics_visibility_shape": physics_info["visibility_shape"],
        "physics_track_file": physics_info["track_path"].name,
        "physics_visibility_file": physics_info["visibility_path"].name,
        "physics_mask_file": physics_info["mask_path"].name,
        "physics_object_masks_file": (
            physics_info["object_masks_path"].name
            if physics_info.get("object_masks_path") is not None
            else None
        ),
        "physics_num_object_masks": physics_info.get("num_object_masks"),
        "physics_mode": "grounded_sam",
        "physics_grounding_prompt": physics_info["prompt"],
        "physics_score_rule": "grounding_dino_short_prompt + sam2 + hand_seg union + cotracker",
        "physics_grid_size": args.physics_grid_size,
        "physics_num_tracks": physics_info["track_shape"][0],
        "physics_selected_tracks": physics_info["track_shape"][0],
        "object_union_mask_area": physics_info["mask_area"],
    }
    debug = physics_info["debug"]
    if "selection_rule" in debug:
        meta["physics_selection_rule"] = debug["selection_rule"]
    if "num_tracks" in debug:
        meta["physics_num_tracks"] = debug["num_tracks"]
    if "selected_tracks" in debug:
        meta["physics_selected_tracks"] = debug["selected_tracks"]
    if "motion_mean" in debug:
        meta["physics_motion_mean"] = debug["motion_mean"]
    if "num_detections" in debug:
        meta["physics_num_detections"] = debug["num_detections"]
    if "detection_labels" in debug:
        meta["physics_detection_labels"] = debug["detection_labels"]
    if "object_mask_area" in debug:
        meta["physics_object_mask_area"] = debug["object_mask_area"]
    if "hand_mask_area" in debug:
        meta["physics_hand_mask_area"] = debug["hand_mask_area"]
    if "track_mask_area" in debug:
        meta["physics_track_mask_area"] = debug["track_mask_area"]
    atomic_write_json(meta_path, meta)


def build_launch_command(args: argparse.Namespace, paths: RuntimePaths, master_addr: str) -> list[str]:
    script_path = Path(__file__).resolve()
    base_cmd = [
        str(paths.python_bin),
        "-m",
        "torch.distributed.run",
    ]
    if args.nnodes == 1:
        base_cmd.extend(
            [
                "--standalone",
                "--nproc_per_node",
                str(args.nproc_per_node),
            ]
        )
    else:
        base_cmd.extend(
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

    worker_args = [
        str(script_path),
        "--distributed_worker",
        "--sfs_root",
        str(paths.sfs_root),
        "--split",
        paths.split,
        "--user_root",
        str(paths.user_root),
        "--model_root",
        str(paths.model_root),
        "--python_bin",
        str(paths.python_bin),
        "--dataset_base",
        str(paths.dataset_base),
        "--dataset_root",
        str(paths.dataset_root),
        "--metadata_path",
        str(paths.metadata_path),
        "--output_root",
        str(paths.output_root),
        "--log_dir",
        str(paths.log_dir),
        "--cotracker_checkpoint",
        str(paths.cotracker_checkpoint),
        "--grounding_dino_model_root",
        str(paths.grounding_dino_model_root),
        "--sam2_model_root",
        str(paths.sam2_model_root),
        "--height",
        str(args.height),
        "--width",
        str(args.width),
        "--num_frames",
        str(args.num_frames),
        "--physics_grid_size",
        str(args.physics_grid_size),
        "--gdino_box_threshold",
        str(args.gdino_box_threshold),
        "--gdino_text_threshold",
        str(args.gdino_text_threshold),
        "--master_port",
        str(args.master_port),
        "--rdzv_id",
        args.rdzv_id,
        "--nproc_per_node",
        str(args.nproc_per_node),
        "--nnodes",
        str(args.nnodes),
        "--node_rank",
        str(args.node_rank),
        "--master_addr",
        master_addr,
    ]
    if args.max_samples is not None:
        worker_args.extend(["--max_samples", str(args.max_samples)])
    if args.skip_existing:
        worker_args.append("--skip_existing")
    return base_cmd + worker_args


def run_launcher(args: argparse.Namespace, paths: RuntimePaths) -> int:
    ensure_required_paths(paths)
    master_addr = resolve_master_addr(args)
    logger = build_logger(paths.log_dir, args.node_rank, worker_mode=False)
    acquire_node_lock(paths.log_dir, args.node_rank)
    try:
        cmd = build_launch_command(args, paths, master_addr)
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{REPO_ROOT}:{env.get('PYTHONPATH', '')}".rstrip(":")
        env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
        env["NCCL_DEBUG"] = env.get("NCCL_DEBUG", "WARN")
        env["TORCH_DISTRIBUTED_DEBUG"] = env.get("TORCH_DISTRIBUTED_DEBUG", "OFF")
        env["PYTORCH_CUDA_ALLOC_CONF"] = env.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        env["NCCL_IB_TIMEOUT"] = env.get("NCCL_IB_TIMEOUT", "200")
        env["NCCL_IB_RETRY_CNT"] = env.get("NCCL_IB_RETRY_CNT", "15")

        logger.info("SFS_ROOT=%s", paths.sfs_root)
        logger.info("SPLIT=%s", paths.split)
        logger.info("DATASET_ROOT=%s", paths.dataset_root)
        logger.info("METADATA_PATH=%s", paths.metadata_path)
        logger.info("OUTPUT_ROOT=%s", paths.output_root)
        logger.info("MODEL_ROOT=%s", paths.model_root)
        logger.info("COTRACKER_CHECKPOINT=%s", paths.cotracker_checkpoint)
        logger.info("GROUNDING_DINO_MODEL_ROOT=%s", paths.grounding_dino_model_root)
        logger.info("SAM2_MODEL_ROOT=%s", paths.sam2_model_root)
        logger.info(
            "PHYSICS_MODE=grounded_sam GDINO_BOX_THRESHOLD=%s GDINO_TEXT_THRESHOLD=%s",
            args.gdino_box_threshold,
            args.gdino_text_threshold,
        )
        logger.info(
            "Distributed config: nnodes=%s nproc_per_node=%s node_rank=%s master_addr=%s master_port=%s",
            args.nnodes,
            args.nproc_per_node,
            args.node_rank,
            master_addr,
            args.master_port,
        )
        logger.info("Launching command: %s", " ".join(cmd))

        process = subprocess.Popen(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        try:
            assert process.stdout is not None
            for line in process.stdout:
                logger.info("[cache] %s", line.rstrip())
                if STOP_REQUESTED:
                    logger.warning("Stop requested, sending SIGINT to torchrun.")
                    process.send_signal(signal.SIGINT)
        finally:
            return process.wait()
    finally:
        release_node_lock(paths.log_dir, args.node_rank)


def run_worker(args: argparse.Namespace, paths: RuntimePaths) -> int:
    ensure_required_paths(paths)
    logger = build_logger(paths.log_dir, args.node_rank, worker_mode=True)
    rank, world_size, local_rank, device = init_distributed()
    logger.info(
        "Worker started: rank=%s world_size=%s local_rank=%s device=%s",
        rank,
        world_size,
        local_rank,
        device,
    )

    samples = load_samples(paths.metadata_path, args.max_samples)
    sharded_samples = [sample for idx, sample in enumerate(samples) if idx % world_size == rank]
    if rank == 0:
        logger.info("Dataset=%s total_samples=%s world_size=%s", DATASET_NAME, len(samples), world_size)
    logger.info("Rank %s handling %s samples", rank, len(sharded_samples))

    models = {
        "vae": load_vae(str(paths.model_root), device),
        "text_encoder": load_text_encoder(str(paths.model_root), device),
        "image_encoder": load_image_encoder(str(paths.model_root), device),
    }
    physics_models = build_physics_models(paths, device)
    output_root = paths.output_root

    iterator = tqdm(sharded_samples, desc=f"[Rank {rank}]", disable=rank != 0)
    success = 0
    skipped = 0
    failed = 0
    for sample in iterator:
        if STOP_REQUESTED:
            logger.warning("Stop requested, ending loop early on rank %s", rank)
            break
        out_dir = sample_output_dir(output_root, sample)
        if args.skip_existing and is_complete(out_dir):
            skipped += 1
            continue
        try:
            sample_paths = get_sample_paths(paths.dataset_root, sample)
            cache_one_sample(
                models=models,
                sample=sample,
                paths=sample_paths,
                out_dir=out_dir,
                physics_models=physics_models,
                device=device,
                args=args,
            )
            success += 1
        except Exception as exc:
            failed += 1
            logger.exception("Failed sample %s: %s", sample.video_id, exc)
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if world_size > 1:
        dist.barrier()
    logger.info("Rank %s done. success=%s skipped=%s failed=%s", rank, success, skipped, failed)
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
    return 0 if failed == 0 else 1


def main() -> int:
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    args = parse_args()
    if args.split == "all":
        if args.distributed_worker or ("RANK" in os.environ and "WORLD_SIZE" in os.environ):
            raise ValueError("--split all is only valid for the launcher, not distributed workers")
        base_rdzv_id = args.rdzv_id
        for split in ("train", "eval"):
            split_args = argparse.Namespace(**{**vars(args), "split": split, "rdzv_id": f"{base_rdzv_id}_{split}"})
            rc = run_launcher(split_args, resolve_runtime_paths(split_args))
            if rc != 0:
                return rc
        return 0
    paths = resolve_runtime_paths(args)
    if args.distributed_worker or ("RANK" in os.environ and "WORLD_SIZE" in os.environ):
        return run_worker(args, paths)
    return run_launcher(args, paths)


if __name__ == "__main__":
    raise SystemExit(main())
