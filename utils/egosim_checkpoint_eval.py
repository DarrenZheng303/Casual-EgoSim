import gc
import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from tqdm import tqdm
from torchvision.io import write_video

from utils.egosim_encoders import (
    encode_prompt,
    load_text_encoder,
    load_vae,
    move_vae,
)

DEFAULT_TEACHER_DIR = (
    "/home/ma-user/work/users/zhengshikang/Causal-Forcing/output/eval_teacher_50step"
)
# ponytail: keep one fixed evaluation budget for every stage; add a knob only if needed.
EVAL_SAMPLE_COUNT = 200
EVAL_RAW_STEPS = (1000, 750, 500, 250)


def _resolve_eval_raw_steps(config) -> list[int]:
    # Automatic checkpoint evaluation keeps the project-wide four-step default;
    # scripts/eval.py sets this field for an explicit manual step-count override.
    eval_steps = getattr(config, "checkpoint_eval_denoising_step_list", None)
    if eval_steps is None:
        return list(EVAL_RAW_STEPS)
    return [int(step) for step in eval_steps]


def _resolve_eval_data_path(train_data_path: str) -> str:
    if "/train/" not in train_data_path:
        raise ValueError(
            "Expected training cache path to contain '/train/' so eval cache can "
            f"be inferred, got: {train_data_path}"
        )
    return train_data_path.replace("/train/", "/eval/", 1)


def _infer_eval_category(sample_dir: Path, meta: dict) -> str:
    target_video = meta.get("target_video")
    if isinstance(target_video, str) and target_video:
        return Path(target_video).parent.name
    output_id = meta.get("output_id", sample_dir.name)
    parts = output_id.split("_")
    return "_".join(parts[:-3]) if len(parts) > 3 else output_id


def _pad_eval_samples(sample_infos: list[dict], multiple: int) -> list[dict]:
    pad_count = (-len(sample_infos)) % multiple
    return sample_infos + [
        {**sample_infos[index % len(sample_infos)], "_padding": True}
        for index in range(pad_count)
    ]


def _select_eval_samples(sample_infos: list[dict]) -> list[dict]:
    return sample_infos[:EVAL_SAMPLE_COUNT]


def _prepare_condition(sample: dict, device: torch.device, dtype: torch.dtype) -> dict:
    keys = (
        "prompt_embeds",
        "image_embeds",
        "ego_prior_latent",
        "hand_latent",
        "mask_latent",
    )
    return {
        key: sample[key].unsqueeze(0).to(device=device, dtype=dtype) for key in keys
    }


def _module_dtype(module: torch.nn.Module) -> torch.dtype:
    parameter = next(module.parameters(), None)
    if parameter is not None:
        return parameter.dtype
    buffer = next(module.buffers(), None)
    if buffer is not None:
        return buffer.dtype
    return torch.float32


def _to_uint8_video(video: torch.Tensor) -> torch.Tensor:
    return (video.mul(255.0).round().clamp(0, 255).to(torch.uint8).cpu())


def _save_video(path: str, video: torch.Tensor, fps: int) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_video(path, _to_uint8_video(video[0].permute(0, 2, 3, 1)), fps=fps)


def _read_video(path: str) -> np.ndarray:
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from {path}")
    return np.stack(frames)


def _metric_tensor(video: np.ndarray, device: torch.device) -> torch.Tensor:
    if video.ndim == 3:
        video = video[None]
    if video.ndim != 4 or video.shape[-1] != 3:
        raise ValueError(f"Expected video array [T,H,W,C], got {video.shape}")
    return torch.from_numpy(video).to(device=device, dtype=torch.float32).permute(0, 3, 1, 2)


@torch.no_grad()
def _psnr(a: np.ndarray, b: np.ndarray, device: torch.device) -> float:
    if a.shape != b.shape:
        raise ValueError(f"PSNR video shape mismatch: {a.shape} vs {b.shape}")
    x = _metric_tensor(a, device)
    y = _metric_tensor(b, device)
    mse = (x - y).square().flatten(1).mean(dim=1)
    value = torch.where(
        mse <= 1e-12,
        torch.full_like(mse, float("inf")),
        20.0 * torch.log10(torch.tensor(255.0, device=device)) - 10.0 * torch.log10(mse),
    )
    return float(value.mean().item())


def _ssim_kernel(channels: int, device: torch.device) -> torch.Tensor:
    coords = torch.arange(11, device=device, dtype=torch.float32) - 5
    kernel_1d = torch.exp(-(coords.square()) / (2 * 1.5**2))
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    kernel_2d = kernel_2d / kernel_2d.sum()
    return kernel_2d.expand(channels, 1, 11, 11)


def _blur(video: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    return F.conv2d(F.pad(video, (5, 5, 5, 5), mode="reflect"), kernel, groups=video.shape[1])


@torch.no_grad()
def _ssim(a: np.ndarray, b: np.ndarray, device: torch.device) -> float:
    if a.shape != b.shape:
        raise ValueError(f"SSIM video shape mismatch: {a.shape} vs {b.shape}")
    x = _metric_tensor(a, device)
    y = _metric_tensor(b, device)
    kernel = _ssim_kernel(x.shape[1], device)
    c1, c2 = (0.01 * 255.0) ** 2, (0.03 * 255.0) ** 2
    mux = _blur(x, kernel)
    muy = _blur(y, kernel)
    sigx = _blur(x * x, kernel) - mux * mux
    sigy = _blur(y * y, kernel) - muy * muy
    sigxy = _blur(x * y, kernel) - mux * muy
    numerator = (2 * mux * muy + c1) * (2 * sigxy + c2)
    denominator = (mux * mux + muy * muy + c1) * (sigx + sigy + c2) + 1e-12
    return float((numerator / denominator).mean().item())


def _build_lpips_metric(config, device: torch.device):
    torch_hub_dir = getattr(config, "checkpoint_eval_lpips_torch_hub_dir", None)
    if not torch_hub_dir:
        raise ValueError(
            "checkpoint evaluation requires LPIPS; set "
            "checkpoint_eval_lpips_torch_hub_dir."
        )

    try:
        import lpips  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "checkpoint_eval_lpips_torch_hub_dir is set, but lpips is not installed. "
            "Run: pip install lpips"
        ) from exc

    old_torch_hub_dir = torch.hub.get_dir()
    torch_hub_dir = Path(torch_hub_dir)
    if not torch_hub_dir.is_dir():
        raise FileNotFoundError(f"LPIPS torch hub dir not found: {torch_hub_dir}")
    alexnet_path = torch_hub_dir / "checkpoints" / "alexnet-owt-7be5be79.pth"
    if not alexnet_path.is_file():
        raise FileNotFoundError(f"AlexNet weight not found: {alexnet_path}")
    torch.hub.set_dir(str(torch_hub_dir))
    try:
        metric = lpips.LPIPS(net="alex").to(device).eval()
    finally:
        torch.hub.set_dir(old_torch_hub_dir)
    return metric


@torch.no_grad()
def _lpips_video(a: np.ndarray, b: np.ndarray, metric, device: torch.device) -> float:
    if a.shape != b.shape:
        raise ValueError(f"LPIPS video shape mismatch: {a.shape} vs {b.shape}")
    x = _metric_tensor(a, device) / 255.0
    y = _metric_tensor(b, device) / 255.0
    value = metric(x * 2.0 - 1.0, y * 2.0 - 1.0)
    return float(value.mean().item())


def _decode_latent(vae, latent: torch.Tensor, device: torch.device) -> torch.Tensor:
    with torch.no_grad():
        move_vae(vae, device)
        vae_dtype = _module_dtype(vae)
        latent = latent.to(device=device, dtype=vae_dtype)
        decoded = vae.decode(
            [latent[0].permute(1, 0, 2, 3).contiguous()], device=device
        )
        video = decoded.permute(0, 2, 1, 3, 4)
        move_vae(vae, "cpu")
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return (video * 0.5 + 0.5).clamp(0, 1)


def _build_warped_timesteps(
    scheduler_owner,
    raw_steps: list[int],
    warp: bool,
    device: torch.device,
) -> torch.Tensor:
    scheduler = scheduler_owner.get_scheduler()
    scheduler.set_timesteps(1000, training=True)
    scheduler.timesteps = scheduler.timesteps.to(device)
    scheduler.sigmas = scheduler.sigmas.to(device)
    steps = torch.tensor(raw_steps, dtype=torch.long, device=device)
    if not warp:
        return steps
    all_timesteps = torch.cat(
        [
            scheduler.timesteps,
            torch.zeros(1, device=device, dtype=scheduler.timesteps.dtype),
        ]
    )
    return all_timesteps[1000 - steps]


@torch.no_grad()
def _sample_video_full(
    model,
    scheduler_owner,
    conditional_dict: dict,
    unconditional_dict: dict | None,
    cfg_scale: float,
    noise: torch.Tensor,
    denoising_steps: torch.Tensor,
    transition_seed: int,
) -> torch.Tensor:
    model_dtype = next(model.parameters()).dtype
    scheduler = scheduler_owner.get_scheduler()
    latents = noise.clone().to(dtype=model_dtype)
    transition_generator = torch.Generator(device=noise.device)
    transition_generator.manual_seed(transition_seed)
    result = None

    for index, current_timestep in enumerate(denoising_steps):
        timestep = torch.full(
            latents.shape[:2],
            float(current_timestep.item()),
            device=latents.device,
            dtype=torch.float32,
        )
        _, pred_x0 = model(
            noisy_image_or_video=latents,
            conditional_dict=conditional_dict,
            timestep=timestep,
        )
        if unconditional_dict is not None:
            _, pred_x0_uncond = model(
                noisy_image_or_video=latents,
                conditional_dict=unconditional_dict,
                timestep=timestep,
            )
            pred_x0 = pred_x0_uncond + cfg_scale * (pred_x0 - pred_x0_uncond)
        is_final_step = index == len(denoising_steps) - 1
        if is_final_step:
            result = pred_x0
            break

        next_timestep = denoising_steps[index + 1]
        transition_noise = torch.randn(
            pred_x0.shape,
            generator=transition_generator,
            device=pred_x0.device,
            dtype=pred_x0.dtype,
        )
        next_timestep_batch = torch.full(
            (pred_x0.shape[0] * pred_x0.shape[1],),
            float(next_timestep.item()),
            device=pred_x0.device,
            dtype=torch.float32,
        )
        latents = scheduler.add_noise(
            pred_x0.flatten(0, 1),
            transition_noise.flatten(0, 1),
            next_timestep_batch,
        ).unflatten(0, pred_x0.shape[:2])

    if result is None:
        raise RuntimeError("Empty denoising schedule.")
    return result


def _unwrap_module(module):
    while hasattr(module, "module"):
        module = module.module
    return module


def _configure_causal_block_size(config, scheduler_owner) -> int:
    block_size = int(getattr(config, "num_frame_per_block", 1))
    if block_size <= 0:
        raise ValueError("num_frame_per_block must be positive for causal eval")
    causal_model = _unwrap_module(scheduler_owner.model)
    causal_model.num_frame_per_block = block_size
    return block_size


def _make_causal_caches(scheduler_owner, noise: torch.Tensor):
    causal_model = _unwrap_module(scheduler_owner.model)
    frame_seq_length = (
        noise.shape[-2] * noise.shape[-1]
        // (causal_model.patch_size[1] * causal_model.patch_size[2])
    )
    cache_size = (
        causal_model.local_attn_size * frame_seq_length
        if causal_model.local_attn_size != -1
        else noise.shape[1] * frame_seq_length
    )
    kv_cache = [
        {
            "k": torch.zeros(
                noise.shape[0], cache_size, causal_model.num_heads,
                causal_model.dim // causal_model.num_heads,
                device=noise.device, dtype=noise.dtype,
            ),
            "v": torch.zeros(
                noise.shape[0], cache_size, causal_model.num_heads,
                causal_model.dim // causal_model.num_heads,
                device=noise.device, dtype=noise.dtype,
            ),
            "global_end_index": torch.zeros(1, device=noise.device, dtype=torch.long),
            "local_end_index": torch.zeros(1, device=noise.device, dtype=torch.long),
        }
        for _ in causal_model.blocks
    ]
    return kv_cache, [None for _ in causal_model.blocks], frame_seq_length


@torch.no_grad()
def _sample_video_causal(
    model,
    scheduler_owner,
    conditional_dict: dict,
    unconditional_dict: dict | None,
    cfg_scale: float,
    noise: torch.Tensor,
    denoising_steps: torch.Tensor,
    transition_seed: int,
) -> torch.Tensor:
    model_dtype = next(model.parameters()).dtype
    scheduler = scheduler_owner.get_scheduler()
    kv_cache, crossattn_cache, frame_seq_length = _make_causal_caches(
        scheduler_owner, noise
    )
    uncond_caches = (
        _make_causal_caches(scheduler_owner, noise)[:2]
        if unconditional_dict is not None
        else None
    )
    output = torch.empty_like(noise, dtype=model_dtype)
    transition_generator = torch.Generator(device=noise.device)
    transition_generator.manual_seed(transition_seed)

    causal_model = _unwrap_module(scheduler_owner.model)
    block_size = int(getattr(causal_model, "num_frame_per_block", 1))
    if block_size <= 0 or noise.shape[1] % block_size != 0:
        raise ValueError(
            f"num_frames={noise.shape[1]} must be divisible by "
            f"num_frame_per_block={block_size}"
        )

    for block_start in range(0, noise.shape[1], block_size):
        block_end = block_start + block_size
        latents = noise[:, block_start:block_end].to(dtype=model_dtype)
        for index, current_timestep in enumerate(denoising_steps):
            timestep = torch.full(
                latents.shape[:2], float(current_timestep.item()),
                device=latents.device, dtype=torch.float32,
            )
            _, pred_x0 = model(
                noisy_image_or_video=latents,
                conditional_dict=conditional_dict,
                timestep=timestep,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start=block_start * frame_seq_length,
            )
            if unconditional_dict is not None:
                uncond_kv_cache, uncond_crossattn_cache = uncond_caches
                _, pred_x0_uncond = model(
                    noisy_image_or_video=latents,
                    conditional_dict=unconditional_dict,
                    timestep=timestep,
                    kv_cache=uncond_kv_cache,
                    crossattn_cache=uncond_crossattn_cache,
                    current_start=block_start * frame_seq_length,
                )
                pred_x0 = pred_x0_uncond + cfg_scale * (pred_x0 - pred_x0_uncond)
            if index == len(denoising_steps) - 1:
                break
            next_timestep = denoising_steps[index + 1]
            latents = scheduler.add_noise(
                pred_x0.flatten(0, 1),
                torch.randn(
                    pred_x0.shape, generator=transition_generator,
                    device=pred_x0.device, dtype=pred_x0.dtype,
                ).flatten(0, 1),
                torch.full(
                    (pred_x0.shape[0] * pred_x0.shape[1],),
                    float(next_timestep.item()), device=pred_x0.device,
                    dtype=torch.float32,
                ),
            ).unflatten(0, pred_x0.shape[:2])
        output[:, block_start:block_end] = pred_x0
        context_timestep = torch.zeros_like(timestep)
        model(
            noisy_image_or_video=pred_x0,
            conditional_dict=conditional_dict,
            timestep=context_timestep,
            kv_cache=kv_cache,
            crossattn_cache=crossattn_cache,
            current_start=block_start * frame_seq_length,
        )
        if unconditional_dict is not None:
            uncond_kv_cache, uncond_crossattn_cache = uncond_caches
            model(
                noisy_image_or_video=pred_x0,
                conditional_dict=unconditional_dict,
                timestep=context_timestep,
                kv_cache=uncond_kv_cache,
                crossattn_cache=uncond_crossattn_cache,
                current_start=block_start * frame_seq_length,
            )
    return output


class EgoSimCheckpointEvalRunner:
    def __init__(
        self,
        config,
        *,
        device: torch.device,
        dtype: torch.dtype,
        generator,
        rank: int,
        world_size: int,
        is_main_process: bool,
        architecture: str | None = None,
        sample_ids: list[str] | None = None,
    ):
        self.config = config
        self.device = device
        self.dtype = dtype
        self.rank = rank
        self.world_size = world_size
        self.is_main_process = is_main_process
        self.eval_data_path = _resolve_eval_data_path(config.data_path)
        self.teacher_dir = Path(
            getattr(config, "eval_teacher_dir", DEFAULT_TEACHER_DIR)
        )
        self.output_root = Path(config.wandb_save_dir).resolve().parent / "egosim_eval"
        self.fps = int(getattr(config, "eval_fps", 16))
        self.eval_seed = int(getattr(config, "eval_seed", 0))
        self.cfg_scale = float(getattr(config, "checkpoint_eval_cfg_scale", 1.0))
        if self.cfg_scale < 0.0:
            raise ValueError("checkpoint_eval_cfg_scale must be non-negative")
        self.generator = generator
        self.scheduler_owner = generator.module
        self.architecture = architecture or getattr(config, "student_arch", "causal")
        if self.architecture not in {"causal", "bidirectional"}:
            raise ValueError(f"Unsupported eval architecture: {self.architecture}")
        if self.architecture == "causal":
            _configure_causal_block_size(config, self.scheduler_owner)
        self.eval_data_root = Path(self.eval_data_path)
        self.requested_sample_ids = sample_ids
        self.sample_infos = []
        for path in sorted(self.eval_data_root.iterdir()):
            if not (
                path.is_dir()
                and (path / "clean_latent.pt").exists()
                and (path / "meta.json").exists()
            ):
                continue
            with open(path / "meta.json", "r", encoding="utf-8") as file:
                meta = json.load(file)
            self.sample_infos.append(
                {
                    "dir": path,
                    "sample_id": path.name,
                    "category": _infer_eval_category(path, meta),
                }
            )
        self.vae = None
        self.lpips_metric = None
        self._unconditional_prompt_embeds = None

    def _select_sample_infos(self) -> list[dict]:
        if self.requested_sample_ids is not None:
            by_id = {sample_info["sample_id"]: sample_info for sample_info in self.sample_infos}
            missing = [sample_id for sample_id in self.requested_sample_ids if sample_id not in by_id]
            if missing:
                raise FileNotFoundError(f"Eval cache missing requested samples: {missing}")
            return [by_id[sample_id] for sample_id in self.requested_sample_ids]
        return _select_eval_samples(self.sample_infos)

    def _clear_block_mask(self) -> None:
        model = getattr(self.scheduler_owner, "model", None)
        while hasattr(model, "module"):
            model = model.module
        if hasattr(model, "block_mask"):
            model.block_mask = None

    def _ensure_eval_modules(self) -> None:
        if self.vae is None:
            self.vae = load_vae(
                self.config.egosim_model_root,
                "cpu",
                torch.bfloat16,
            )
            gc.collect()

    def _build_unconditional_dict(self, conditional_dict: dict) -> dict | None:
        if self.cfg_scale == 1.0:
            return None
        if self._unconditional_prompt_embeds is None:
            text_encoder = load_text_encoder(
                self.config.egosim_model_root,
                self.device,
            )
            prompt_embeds = encode_prompt(
                text_encoder,
                self.config.negative_prompt,
                self.device,
            )
            self._unconditional_prompt_embeds = prompt_embeds.to(
                dtype=self.dtype
            ).unsqueeze(0)
            text_encoder.model.to('cpu')
            del text_encoder
            gc.collect()
            torch.cuda.empty_cache()
        unconditional_dict = dict(conditional_dict)
        unconditional_dict["prompt_embeds"] = self._unconditional_prompt_embeds.expand(
            conditional_dict["prompt_embeds"].shape[0], -1, -1
        )
        return unconditional_dict

    def run(self, step: int) -> dict | None:
        # Teacher-forcing training caches a different attention mask than AR inference.
        self._clear_block_mask()
        try:
            return self._run(step)
        finally:
            self._clear_block_mask()

    def _run(self, step: int) -> dict | None:
        if len(self.sample_infos) == 0:
            return None
        if not self.teacher_dir.exists():
            raise FileNotFoundError(f"Teacher eval dir not found: {self.teacher_dir}")

        self._ensure_eval_modules()
        if self.lpips_metric is None:
            self.lpips_metric = _build_lpips_metric(self.config, self.device)
        self.output_root.mkdir(parents=True, exist_ok=True)
        step_dir = self.output_root / f"iter_{step:09d}"
        step_dir.mkdir(parents=True, exist_ok=True)

        self.generator.eval()
        selected_sample_infos = self._select_sample_infos()
        padded_sample_infos = _pad_eval_samples(selected_sample_infos, self.world_size)
        local_sample_infos = padded_sample_infos[self.rank :: self.world_size]

        raw_steps = _resolve_eval_raw_steps(self.config)
        denoising_steps = _build_warped_timesteps(
            self.scheduler_owner,
            raw_steps,
            bool(self.config.warp_denoising_step),
            self.device,
        )

        local_metrics = []
        iterator = local_sample_infos
        if self.is_main_process:
            iterator = tqdm(iterator, desc=f"Eval step {step}", dynamic_ncols=True)
        for sample_info in iterator:
            sample_dir = sample_info["dir"]
            sample_id = sample_info["sample_id"]
            is_padding = sample_info.get("_padding", False)
            load_latent = lambda name: torch.load(
                sample_dir / name, map_location="cpu"
            ).permute(1, 0, 2, 3).contiguous().to(dtype=torch.float32)
            sample = {
                "clean_latent": load_latent("clean_latent.pt"),
                "prompt_embeds": torch.load(
                    sample_dir / "prompt_embedding.pt", map_location="cpu"
                ).to(dtype=torch.float32),
                "image_embeds": torch.load(
                    sample_dir / "image_embedding.pt", map_location="cpu"
                ).to(dtype=torch.float32),
                "ego_prior_latent": load_latent("ego_prior_latent.pt"),
                "hand_latent": load_latent("hand_latent.pt"),
                "mask_latent": load_latent("mask_latent.pt"),
            }

            seed = self.eval_seed + int(
                hashlib.sha1(sample_id.encode("utf-8")).hexdigest()[:8], 16
            )
            noise = torch.randn(
                sample["clean_latent"].unsqueeze(0).shape,
                generator=torch.Generator(device=self.device).manual_seed(seed),
                device=self.device,
                dtype=self.dtype,
            )
            transition_seed = seed + 100000
            condition = _prepare_condition(sample, device=self.device, dtype=self.dtype)
            unconditional_condition = self._build_unconditional_dict(condition)
            sample_video = (
                _sample_video_causal
                if self.architecture == "causal"
                else _sample_video_full
            )
            student_latent = sample_video(
                model=self.generator,
                scheduler_owner=self.scheduler_owner,
                conditional_dict=condition,
                unconditional_dict=unconditional_condition,
                cfg_scale=self.cfg_scale,
                noise=noise,
                denoising_steps=denoising_steps,
                transition_seed=transition_seed,
            )
            if is_padding:
                continue
            teacher_path = (
                self.teacher_dir
                / "egosim_eval"
                / "iter_000000000"
                / sample_id
                / "student_50step.mp4"
            )
            if not teacher_path.exists():
                raise FileNotFoundError(
                    f"Teacher video not found for {sample_id}: {teacher_path}"
                )

            eval_sample_dir = step_dir / sample_id
            eval_sample_dir.mkdir(parents=True, exist_ok=True)
            student_video = _decode_latent(self.vae, student_latent, self.device)
            gt_video = _decode_latent(
                self.vae,
                sample["clean_latent"].unsqueeze(0).to(device=self.device),
                self.device,
            )
            student_path = eval_sample_dir / f"student_{len(raw_steps)}step.mp4"
            _save_video(str(student_path), student_video, self.fps)

            teacher_frames = _read_video(str(teacher_path))
            student_frames = _to_uint8_video(
                student_video[0].permute(0, 2, 3, 1)
            ).numpy()
            gt_frames = _to_uint8_video(gt_video[0].permute(0, 2, 3, 1)).numpy()
            if teacher_frames.shape != student_frames.shape:
                raise ValueError(
                    f"Teacher/student video shape mismatch for {sample_id}: "
                    f"{teacher_frames.shape} vs {student_frames.shape}"
                )
            if gt_frames.shape != student_frames.shape:
                raise ValueError(
                    f"GT/student video shape mismatch for {sample_id}: "
                    f"{gt_frames.shape} vs {student_frames.shape}"
                )

            teacher_video = (
                torch.from_numpy(teacher_frames)
                .permute(0, 3, 1, 2)
                .unsqueeze(0)
                .to(dtype=torch.float32)
                .div_(255.0)
            )
            compare_video = torch.cat([teacher_video, student_video.cpu()], dim=4)
            compare_path = eval_sample_dir / "compare_teacher_student.mp4"
            _save_video(str(compare_path), compare_video, self.fps)

            metrics = {
                "sample_id": sample_id,
                "category": sample_info["category"],
                "psnr_teacher50step": _psnr(teacher_frames, student_frames, self.device),
                "ssim_teacher50step": _ssim(teacher_frames, student_frames, self.device),
                "psnr_gt": _psnr(gt_frames, student_frames, self.device),
                "ssim_gt": _ssim(gt_frames, student_frames, self.device),
                "teacher_path": str(teacher_path),
                "student_path": str(student_path),
                "compare_path": str(compare_path),
            }
            metrics.update(
                {
                    "lpips_teacher50step": _lpips_video(
                        teacher_frames, student_frames, self.lpips_metric, self.device
                    ),
                    "lpips_gt": _lpips_video(
                        gt_frames, student_frames, self.lpips_metric, self.device
                    ),
                }
            )
            with open(eval_sample_dir / "meta.json", "w", encoding="utf-8") as file:
                json.dump(metrics, file, indent=2)
            local_metrics.append(metrics)

        rank_metrics_path = step_dir / f"rank_{self.rank:03d}_metrics.json"
        with open(rank_metrics_path, "w", encoding="utf-8") as file:
            json.dump(local_metrics, file, indent=2)
        dist.barrier()

        summary = None
        if self.is_main_process:
            all_metrics = []
            for rank in range(self.world_size):
                with open(
                    step_dir / f"rank_{rank:03d}_metrics.json",
                    "r",
                    encoding="utf-8",
                ) as file:
                    all_metrics.extend(json.load(file))
            summary = {
                "step": step,
                "num_selected_samples": len(selected_sample_infos),
                "num_padded_samples": len(padded_sample_infos) - len(selected_sample_infos),
                "num_samples": len(all_metrics),
                "psnr_teacher50step": float(
                    np.mean([item["psnr_teacher50step"] for item in all_metrics])
                ),
                "ssim_teacher50step": float(
                    np.mean([item["ssim_teacher50step"] for item in all_metrics])
                ),
                "psnr_gt": float(np.mean([item["psnr_gt"] for item in all_metrics])),
                "ssim_gt": float(np.mean([item["ssim_gt"] for item in all_metrics])),
            }
            summary.update(
                {
                    "lpips_teacher50step": float(
                        np.mean([item["lpips_teacher50step"] for item in all_metrics])
                    ),
                    "lpips_gt": float(
                        np.mean([item["lpips_gt"] for item in all_metrics])
                    ),
                }
            )
            with open(step_dir / "summary.json", "w", encoding="utf-8") as file:
                json.dump(summary, file, indent=2)
            message = (
                f"[Eval] step={step} samples={summary['num_samples']} "
                f"psnr_teacher50step={summary['psnr_teacher50step']:.4f} "
                f"ssim_teacher50step={summary['ssim_teacher50step']:.4f} "
                f"psnr_gt={summary['psnr_gt']:.4f} "
                f"ssim_gt={summary['ssim_gt']:.4f} "
                f"lpips_teacher50step={summary['lpips_teacher50step']:.4f} "
                f"lpips_gt={summary['lpips_gt']:.4f}"
            )
            print(message, flush=True)
        dist.barrier()
        return summary
