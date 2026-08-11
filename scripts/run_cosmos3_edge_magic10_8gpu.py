#!/home/ma-user/work/users/zhengshikang/conda/envs/causal_forcing/bin/python
"""Run Cosmos3-Edge I2V on the 10 fixed EgoSim examples, one worker per GPU."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path("/home/ma-user/work/model/Cosmos3-Edge")
DEFAULT_SOURCE_REPO = Path("/home/ma-user/work/users/zhengshikang/Causal-Forcing")
DEFAULT_CACHE = Path(
    "/home/ma-user/work/users/zhengshikang/datasets/luyitas/"
    "egosim_egodex_egovid/cache/train/egodex"
)
DIFFUSERS_COMMIT = "90c0ffdc045902a3667d473d2fbfc03e8716dba9"
DEPS_DIR = REPO_ROOT / ".cosmos3_deps" / DIFFUSERS_COMMIT
DIFFUSERS_URL = f"https://codeload.github.com/huggingface/diffusers/tar.gz/{DIFFUSERS_COMMIT}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--source-repo", type=Path, default=DEFAULT_SOURCE_REPO)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_SOURCE_REPO / "output/cosmos3_edge_magic10",
    )
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--num-frames", type=int, default=61)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--fps", type=float, default=16.0)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=6.0)
    parser.add_argument("--flow-shift", type=float, default=12.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker-rank", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--world-size", type=int, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def load_samples(args: argparse.Namespace) -> list[dict[str, str]]:
    notes_path = args.source_repo / "AGENT_NOTES.md"
    text = notes_path.read_text(encoding="utf-8")
    try:
        section = text.split("## Fixed Eval Samples", 1)[1].split("\n## ", 1)[0]
    except IndexError as exc:
        raise RuntimeError(f"Fixed Eval Samples section not found in {notes_path}") from exc
    section = section.split("When the user asks", 1)[0]
    sample_ids = re.findall(r"^\d+\.\s+`([^`]+)`", section, flags=re.MULTILINE)
    if len(sample_ids) != 10 or len(set(sample_ids)) != 10:
        raise RuntimeError(f"Expected 10 unique fixed samples in {notes_path}, got {len(sample_ids)}")

    samples = []
    for sample_id in sample_ids:
        meta_path = args.cache / sample_id / "meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        prompt = str(meta.get("prompt", "")).strip()
        image_path = Path(str(meta.get("first_frame", "")))
        if not prompt:
            raise RuntimeError(f"Missing prompt: {meta_path}")
        if image_path.suffix.lower() != ".png" or not image_path.is_file():
            raise RuntimeError(f"Missing cache first-frame PNG for {sample_id}: {image_path}")
        samples.append(
            {
                "sample_id": sample_id,
                "prompt": prompt,
                "image_path": str(image_path),
                "meta_path": str(meta_path),
            }
        )
    return samples


def deps_env() -> dict[str, str]:
    env = os.environ.copy()
    old = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(DEPS_DIR) + (f":{old}" if old else "")
    return env


def ensure_diffusers() -> None:
    probe = [
        sys.executable,
        "-c",
        "from diffusers import Cosmos3OmniPipeline; print('Cosmos3 Diffusers OK')",
    ]
    if subprocess.run(probe, env=deps_env(), check=False).returncode == 0:
        return
    DEPS_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-deps",
            "--target",
            str(DEPS_DIR),
            DIFFUSERS_URL,
        ],
        check=True,
    )
    subprocess.run(probe, env=deps_env(), check=True)


def worker(args: argparse.Namespace, samples: list[dict[str, str]]) -> None:
    sys.path.insert(0, str(DEPS_DIR))
    import torch
    from diffusers import Cosmos3OmniPipeline
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
    from diffusers.utils import export_to_video, load_image

    rank = args.worker_rank
    assert rank is not None and args.world_size is not None
    assigned = samples[rank :: args.world_size]
    if not assigned:
        print(f"rank {rank}: no samples assigned", flush=True)
        return

    torch.cuda.set_device(0)
    print(f"rank {rank}: loading {args.model} on {torch.cuda.get_device_name(0)}", flush=True)
    pipe = Cosmos3OmniPipeline.from_pretrained(
        str(args.model),
        dtype=torch.bfloat16,
        enable_safety_checker=False,
        local_files_only=True,
    )
    pipe.to("cuda")
    pipe.scheduler = UniPCMultistepScheduler.from_config(
        pipe.scheduler.config,
        flow_shift=args.flow_shift,
        use_karras_sigmas=False,
    )

    negative_path = args.model / "assets/negative_prompt.json"
    negative_prompt = json.dumps(json.loads(negative_path.read_text(encoding="utf-8")))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for sample in assigned:
        sample_id = sample["sample_id"]
        output_path = args.output_dir / f"{sample_id}.mp4"
        if output_path.is_file() and output_path.stat().st_size > 0 and not args.overwrite:
            print(f"rank {rank}: skip existing {output_path}", flush=True)
            continue

        started = time.time()
        print(f"rank {rank}: generating {sample_id}", flush=True)
        result = pipe(
            prompt=sample["prompt"],
            negative_prompt=negative_prompt,
            image=load_image(sample["image_path"]),
            num_frames=args.num_frames,
            height=args.height,
            width=args.width,
            fps=args.fps,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            generator=torch.Generator(device="cuda").manual_seed(args.seed),
            add_resolution_template=False,
            add_duration_template=False,
        )
        partial_path = output_path.with_suffix(".partial.mp4")
        export_to_video(result.video, str(partial_path), fps=args.fps, macro_block_size=1)
        partial_path.replace(output_path)
        elapsed = time.time() - started
        sidecar = {
            **sample,
            "output_path": str(output_path),
            "model": str(args.model),
            "num_frames": args.num_frames,
            "height": args.height,
            "width": args.width,
            "fps": args.fps,
            "steps": args.steps,
            "guidance_scale": args.guidance_scale,
            "flow_shift": args.flow_shift,
            "seed": args.seed,
            "elapsed_seconds": elapsed,
        }
        output_path.with_suffix(".json").write_text(
            json.dumps(sidecar, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"rank {rank}: saved {output_path} ({elapsed:.1f}s)", flush=True)


def launcher(args: argparse.Namespace, samples: list[dict[str, str]]) -> None:
    gpu_ids = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if not gpu_ids or len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError(f"Invalid --gpus value: {args.gpus}")
    ensure_diffusers()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = args.output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    children: list[tuple[int, subprocess.Popen, object]] = []
    base_command = [item for item in sys.argv if item not in ("--dry-run",)]
    for rank, gpu_id in enumerate(gpu_ids):
        log_handle = (log_dir / f"gpu{gpu_id}.log").open("a", encoding="utf-8")
        env = deps_env()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            *base_command[1:],
            "--worker-rank",
            str(rank),
            "--world-size",
            str(len(gpu_ids)),
        ]
        print(f"launch rank {rank} on GPU {gpu_id}; log: {log_handle.name}")
        children.append((gpu_id, subprocess.Popen(command, env=env, stdout=log_handle, stderr=subprocess.STDOUT), log_handle))

    failed = []
    try:
        for gpu_id, process, log_handle in children:
            code = process.wait()
            log_handle.close()
            if code:
                failed.append((gpu_id, code))
    except KeyboardInterrupt:
        for _, process, _ in children:
            process.terminate()
        for _, process, log_handle in children:
            process.wait()
            log_handle.close()
        raise
    if failed:
        raise SystemExit(f"Workers failed: {failed}; see {log_dir}")
    print(f"Done: {args.output_dir}")


def main() -> None:
    args = parse_args()
    if not args.model.is_dir():
        raise FileNotFoundError(args.model)
    samples = load_samples(args)
    if args.dry_run:
        for index, sample in enumerate(samples):
            print(f"{index:02d}  {sample['sample_id']}  {sample['image_path']}")
        print(f"Validated {len(samples)} prompts and cache PNGs")
    elif args.worker_rank is not None:
        worker(args, samples)
    else:
        launcher(args, samples)


if __name__ == "__main__":
    main()
