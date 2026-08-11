from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any

import imageio
import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import (
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
    Sam2Model,
    Sam2Processor,
)

MAX_SAM_INSTANCE_AREA_RATIO = 0.10
TARGET_COTRACKER_TRACKS = 512


def load_video_frames_uint8(video_path: Path) -> np.ndarray:
    reader = imageio.get_reader(str(video_path))
    frames = [np.array(frame) for frame in reader]
    reader.close()
    if not frames:
        raise RuntimeError(f"Empty video: {video_path}")
    return np.stack(frames, axis=0)


def load_hand_mask(hand_seg_path: Path, size: tuple[int, int]) -> np.ndarray:
    if not hand_seg_path.exists():
        return np.zeros((size[1], size[0]), dtype=np.uint8)
    hand_mask = Image.open(hand_seg_path).convert("L")
    if hand_mask.size != size:
        hand_mask = hand_mask.resize(size, resample=Image.Resampling.NEAREST)
    return (np.array(hand_mask) > 0).astype(np.uint8)


def save_binary_mask(mask: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask > 0).astype(np.uint8) * 255, mode="L").save(output_path)


def save_grounding_dino_boxes(
    image: Image.Image,
    detections: list[dict[str, Any]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)

    for det in detections:
        x1, y1, x2, y2 = [float(v) for v in det["box"]]
        draw.rectangle([(x1, y1), (x2, y2)], outline=(255, 64, 64), width=3)
        label = f'{det["label"]} {det["score"]:.2f}'
        text_x = max(0.0, x1)
        text_y = max(0.0, y1 - 14.0)
        draw.text((text_x, text_y), label, fill=(255, 64, 64))

    canvas.save(output_path)


def normalize_grounding_prompt(text: str) -> str:
    text = " ".join(text.strip().split()).lower()
    if not text:
        return text
    phrases = []
    seen = set()
    for chunk in text.replace("\n", " ").split("."):
        phrase = " ".join(chunk.strip(" ,.").split())
        if not phrase or phrase in seen:
            continue
        seen.add(phrase)
        phrases.append(phrase)
    return ". ".join(phrases) + ("." if phrases else "")


def build_grounded_sam_cotracker_models(
    grounding_dino_model_root: Path,
    sam2_model_root: Path,
    cotracker_checkpoint: Path,
    device: torch.device,
) -> dict[str, Any]:
    from third_party.cotracker.predictor import CoTrackerPredictor

    grounding_processor = AutoProcessor.from_pretrained(str(grounding_dino_model_root))
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
        str(grounding_dino_model_root)
    ).to(device)

    sam2_processor = Sam2Processor.from_pretrained(str(sam2_model_root))
    sam2_model = Sam2Model.from_pretrained(str(sam2_model_root)).to(device)
    if device.type == "cuda":
        sam2_model = sam2_model.to(dtype=torch.bfloat16)

    cotracker = CoTrackerPredictor(
        checkpoint=str(cotracker_checkpoint),
        offline=True,
        window_len=60,
    ).to(device).eval()

    return {
        "grounding_processor": grounding_processor,
        "grounding_model": grounding_model.eval(),
        "sam2_processor": sam2_processor,
        "sam2_model": sam2_model.eval(),
        "cotracker": cotracker,
    }


def run_grounding_dino(
    image: Image.Image,
    prompt: str,
    models: dict[str, Any],
    device: torch.device,
    box_threshold: float,
    text_threshold: float,
) -> list[dict[str, Any]]:
    inputs = models["grounding_processor"](
        images=image,
        text=prompt,
        return_tensors="pt",
        truncation=True,
        max_length=256,
    ).to(device)

    with torch.no_grad():
        outputs = models["grounding_model"](**inputs)

    results = models["grounding_processor"].post_process_grounded_object_detection(
        outputs,
        input_ids=inputs.input_ids,
        threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[image.size[::-1]],
    )[0]

    labels = results.get("text_labels", results["labels"])
    detections = []
    for score, label, box in zip(results["scores"], labels, results["boxes"]):
        detections.append(
            {
                "score": float(score),
                "label": str(label),
                "box": [round(float(x), 2) for x in box],
            }
        )
    return detections


def empty_instances(height: int, width: int) -> dict[str, Any]:
    return {
        "masks": np.zeros((0, height, width), dtype=np.uint8),
        "labels": [],
        "scores": np.zeros((0,), dtype=np.float32),
        "boxes": np.zeros((0, 4), dtype=np.float32),
        "height": height,
        "width": width,
    }


def run_sam2_masks(
    image: Image.Image,
    detections: list[dict[str, Any]],
    models: dict[str, Any],
    device: torch.device,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    if not detections:
        return (
            np.zeros((image.size[1], image.size[0]), dtype=np.uint8),
            empty_instances(image.size[1], image.size[0]),
            {"kept_instances": [], "skipped_instances": []},
        )

    input_boxes = [[det["box"] for det in detections]]
    inputs = models["sam2_processor"](
        images=image,
        input_boxes=input_boxes,
        return_tensors="pt",
    ).to(device)

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )
    with torch.inference_mode(), autocast_ctx:
        outputs = models["sam2_model"](**inputs, multimask_output=False)

    masks = models["sam2_processor"].post_process_masks(
        outputs.pred_masks.float().cpu(),
        inputs["original_sizes"].cpu(),
    )[0]

    union_mask = np.zeros((image.size[1], image.size[0]), dtype=np.uint8)
    kept_instances = []
    skipped_instances = []
    kept_masks = []
    kept_labels = []
    kept_scores = []
    kept_boxes = []
    skipped_nonempty_masks: list[tuple[dict[str, Any], np.ndarray, str, float, list[float]]] = []
    frame_area = image.size[0] * image.size[1]
    for det, mask in zip(detections, masks):
        mask_np = mask.squeeze(0).cpu().numpy().astype(np.uint8)
        area = int(mask_np.sum())
        area_ratio = area / frame_area
        item = {
            "label": det["label"],
            "area": area,
            "area_ratio": area_ratio,
            "score": det["score"],
        }
        if area_ratio > MAX_SAM_INSTANCE_AREA_RATIO:
            skipped_instances.append(item)
            if area > 0:
                skipped_nonempty_masks.append(
                    (
                        item,
                        mask_np,
                        str(det["label"]),
                        float(det["score"]),
                        [float(v) for v in det["box"]],
                    )
                )
            continue
        kept_instances.append(item)
        kept_masks.append(mask_np)
        kept_labels.append(str(det["label"]))
        kept_scores.append(float(det["score"]))
        kept_boxes.append([float(v) for v in det["box"]])
        union_mask |= mask_np

    if not kept_masks and skipped_nonempty_masks:
        fallback_item, fallback_mask, fallback_label, fallback_score, fallback_box = min(
            skipped_nonempty_masks,
            key=lambda entry: (entry[0]["area"], -entry[0]["score"]),
        )
        fallback_item = dict(fallback_item)
        fallback_item["fallback_reason"] = "smallest_nonempty_mask_after_area_filter"
        kept_instances.append(fallback_item)
        kept_masks.append(fallback_mask)
        kept_labels.append(fallback_label)
        kept_scores.append(fallback_score)
        kept_boxes.append(fallback_box)
        union_mask |= fallback_mask

    instances = {
        "masks": (
            np.stack(kept_masks, axis=0).astype(np.uint8)
            if kept_masks
            else np.zeros((0, image.size[1], image.size[0]), dtype=np.uint8)
        ),
        "labels": kept_labels,
        "scores": np.asarray(kept_scores, dtype=np.float32),
        "boxes": (
            np.asarray(kept_boxes, dtype=np.float32)
            if kept_boxes
            else np.zeros((0, 4), dtype=np.float32)
        ),
        "height": image.size[1],
        "width": image.size[0],
    }
    return union_mask.astype(np.uint8), instances, {
        "kept_instances": kept_instances,
        "skipped_instances": skipped_instances,
    }


def sample_mask_queries(
    track_mask: np.ndarray,
    target_tracks: int = TARGET_COTRACKER_TRACKS,
) -> np.ndarray:
    ys, xs = np.nonzero(track_mask > 0)
    if xs.size == 0:
        raise RuntimeError("Empty tracking mask after SAM/hand union")
    if xs.size >= target_tracks:
        indices = np.linspace(0, xs.size, target_tracks, endpoint=False, dtype=np.int64)
    else:
        indices = np.arange(target_tracks, dtype=np.int64) % xs.size
    queries = np.stack(
        [
            np.zeros(target_tracks, dtype=np.float32),
            xs[indices].astype(np.float32),
            ys[indices].astype(np.float32),
        ],
        axis=1,
    )
    return queries


def run_cotracker_on_queries(
    video: torch.Tensor,
    queries: torch.Tensor,
    models: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        pred_tracks, pred_visibility = models["cotracker"](
            video,
            queries=queries,
            backward_tracking=False,
        )

    if pred_visibility.ndim == 4:
        vis_np = pred_visibility[0, :, :, 0].detach().cpu().numpy().astype(bool)
    else:
        vis_np = pred_visibility[0].detach().cpu().numpy().astype(bool)
    tracks_np = pred_tracks[0].detach().cpu().numpy()
    selected_tracks = np.transpose(tracks_np, (1, 0, 2)).astype(np.float32)
    selected_visibility = np.transpose(vis_np, (1, 0)).astype(bool)
    return selected_tracks, selected_visibility


def select_fixed_track_count(
    tracks: np.ndarray,
    visibility: np.ndarray,
    target_tracks: int = TARGET_COTRACKER_TRACKS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if tracks.shape[0] < target_tracks:
        raise RuntimeError(
            f"CoTracker returned {tracks.shape[0]} tracks, fewer than target {target_tracks}"
        )
    indices = np.linspace(0, tracks.shape[0], target_tracks, endpoint=False, dtype=np.int64)
    return tracks[indices], visibility[indices], indices


def assign_tracks_to_objects(
    tracks: np.ndarray,
    instance_masks: np.ndarray,
    height: int,
    width: int,
) -> np.ndarray:
    """Map each track to an object index by its frame-0 query point.

    Returns int16 array [num_tracks]; -1 means the point lies in no object mask
    (e.g. hand-only region). When masks overlap, the smallest-area (most
    specific) object wins.
    """
    num_tracks = int(tracks.shape[0])
    object_ids = np.full((num_tracks,), -1, dtype=np.int16)
    if instance_masks.shape[0] == 0 or num_tracks == 0:
        return object_ids
    query_x = np.clip(np.round(tracks[:, 0, 0]).astype(np.int64), 0, width - 1)
    query_y = np.clip(np.round(tracks[:, 0, 1]).astype(np.int64), 0, height - 1)
    areas = instance_masks.reshape(instance_masks.shape[0], -1).sum(axis=1)
    # Assign from largest to smallest so the smallest containing mask wins ties.
    for obj_idx in np.argsort(areas)[::-1]:
        hit = instance_masks[obj_idx, query_y, query_x] > 0
        object_ids[hit] = np.int16(obj_idx)
    return object_ids


def generate_grounded_sam_physics_tracks(
    target_video_path: Path,
    hand_seg_path: Path,
    text_prompt: str,
    models: dict[str, Any],
    device: torch.device,
    grid_size: int,
    box_threshold: float,
    text_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any], Image.Image, list[dict[str, Any]], dict[str, Any]]:
    prompt = normalize_grounding_prompt(text_prompt)
    if not prompt:
        raise ValueError("short_prompt is empty")

    frames = load_video_frames_uint8(target_video_path)
    height, width = frames.shape[1:3]
    image = Image.fromarray(frames[0])

    detections = run_grounding_dino(
        image=image,
        prompt=prompt,
        models=models,
        device=device,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )
    object_union_mask, sam_instances, sam_instance_debug = run_sam2_masks(
        image=image,
        detections=detections,
        models=models,
        device=device,
    )
    hand_mask = load_hand_mask(hand_seg_path, size=(width, height))
    track_mask = (object_union_mask | hand_mask).astype(np.uint8)
    if track_mask.sum() == 0:
        raise RuntimeError(
            f"Empty tracking mask for {target_video_path} with prompt: {prompt}"
        )

    video = torch.from_numpy(frames).permute(0, 3, 1, 2).unsqueeze(0).float().to(device)
    query_points = sample_mask_queries(track_mask)
    query_tensor = torch.from_numpy(query_points).unsqueeze(0).float().to(device)
    selected_tracks, selected_visibility = run_cotracker_on_queries(
        video=video,
        queries=query_tensor,
        models=models,
    )

    original_num_tracks = int(selected_tracks.shape[0])
    selected_tracks, selected_visibility, _ = select_fixed_track_count(
        selected_tracks,
        selected_visibility,
    )

    # One id array indexes the shared track axis, so it applies to both
    # selected_tracks and selected_visibility (visibility[ids == i] == object i).
    track_object_ids = assign_tracks_to_objects(
        selected_tracks, sam_instances["masks"], height, width
    )
    sam_instances["track_object_ids"] = track_object_ids

    debug = {
        "selection_rule": "grounding_dino_sam2_handseg_union",
        "grounding_prompt": prompt,
        "num_detections": len(detections),
        "detection_labels": [det["label"] for det in detections],
        "num_tracks": original_num_tracks,
        "selected_tracks": int(selected_tracks.shape[0]),
        "target_tracks": TARGET_COTRACKER_TRACKS,
        "base_grid_size": int(grid_size),
        "query_sampling_rule": "uniform_linspace_over_mask_pixels",
        "query_mask_pixels": int(track_mask.sum()),
        "track_subsample_rule": "uniform_linspace",
        "object_mask_area": int(object_union_mask.sum()),
        "hand_mask_area": int(hand_mask.sum()),
        "track_mask_area": int(track_mask.sum()),
        "track_mask_area_ratio": float(track_mask.sum()) / float(track_mask.size),
        "sam_instance_area_threshold": MAX_SAM_INSTANCE_AREA_RATIO,
        "sam_kept_instances": sam_instance_debug["kept_instances"],
        "sam_skipped_instances": sam_instance_debug["skipped_instances"],
        "tracks_per_object": [
            int((track_object_ids == i).sum()) for i in range(sam_instances["masks"].shape[0])
        ],
        "tracks_unassigned": int((track_object_ids == -1).sum()),
    }
    return selected_tracks, selected_visibility, object_union_mask, sam_instances, image, detections, debug
