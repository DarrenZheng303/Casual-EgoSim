import os

import imageio
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from utils.egosim_image_encoder import WanImageEncoder
from utils.egosim_vae import WanVideoVAE
from wan.modules.t5 import T5EncoderModel


def _model_path(model_root: str, relative_path: str) -> str:
    path = os.path.join(os.path.abspath(model_root), relative_path)
    if not os.path.exists(path):
        raise FileNotFoundError(f'EgoSim model file not found: {path}')
    return path


def load_text_encoder(
    model_root: str,
    device: torch.device | str,
    dtype: torch.dtype = torch.bfloat16,
) -> T5EncoderModel:
    return T5EncoderModel(
        text_len=512,
        dtype=dtype,
        device=device,
        checkpoint_path=_model_path(
            model_root, 'models_t5_umt5-xxl-enc-bf16.pth'
        ),
        tokenizer_path=_model_path(model_root, 'google/umt5-xxl'),
    )


@torch.no_grad()
def encode_prompt(
    text_encoder: T5EncoderModel,
    text: str,
    device: torch.device | str,
) -> torch.Tensor:
    ids, mask = text_encoder.tokenizer(
        text,
        return_mask=True,
        add_special_tokens=True,
    )
    ids = ids.to(device)
    mask = mask.to(device)
    sequence_lengths = mask.gt(0).sum(dim=1).long()
    embedding = text_encoder.model(ids, mask)
    for index, length in enumerate(sequence_lengths):
        embedding[index, length:] = 0
    return embedding.squeeze(0).to(dtype=torch.bfloat16)


def load_vae(
    model_root: str,
    device: torch.device | str,
    dtype: torch.dtype = torch.bfloat16,
) -> WanVideoVAE:
    state_dict = torch.load(
        _model_path(model_root, 'Wan2.1_VAE.pth'),
        map_location='cpu',
        mmap=True,
    )
    vae = WanVideoVAE().eval().requires_grad_(False)
    vae.load_state_dict(
        {f'model.{name}': value for name, value in state_dict.items()},
        assign=True,
    )
    return move_vae(vae, device, dtype)


def move_vae(
    vae: WanVideoVAE,
    device: torch.device | str,
    dtype: torch.dtype | None = None,
) -> WanVideoVAE:
    vae.to(device=device, dtype=dtype)
    vae.mean = vae.mean.to(device=device)
    vae.std = vae.std.to(device=device)
    vae.scale = [vae.mean, 1.0 / vae.std]
    return vae


def load_image_encoder(
    model_root: str,
    device: torch.device | str,
    dtype: torch.dtype = torch.bfloat16,
) -> WanImageEncoder:
    state_dict = torch.load(
        _model_path(
            model_root,
            'models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth',
        ),
        map_location='cpu',
        mmap=True,
    )
    image_encoder = WanImageEncoder().eval().requires_grad_(False)
    image_encoder.load_state_dict(
        {
            f'model.{name}': value
            for name, value in state_dict.items()
            if not name.startswith('textual.')
        },
        assign=True,
    )
    return image_encoder.to(device=device, dtype=dtype)


def _load_video(
    video_path: str,
    target_frames: int,
    height: int,
    width: int,
) -> torch.Tensor:
    reader = imageio.get_reader(video_path)
    frames = [frame for frame in reader]
    reader.close()
    if not frames:
        raise RuntimeError(f'Empty video: {video_path}')
    if len(frames) < target_frames:
        frames += [frames[-1]] * (target_frames - len(frames))
    else:
        frames = frames[:target_frames]
    frames = [
        np.asarray(
            Image.fromarray(frame).resize((width, height), Image.Resampling.LANCZOS)
            if frame.shape[:2] != (height, width)
            else Image.fromarray(frame)
        )
        for frame in frames
    ]
    video = torch.from_numpy(np.stack(frames).astype(np.float32))
    video = video.permute(3, 0, 1, 2)
    return video / 255.0 * 2.0 - 1.0


@torch.no_grad()
def encode_video(
    vae: WanVideoVAE,
    video_path: str,
    device: torch.device | str,
    target_frames: int = 61,
    height: int = 480,
    width: int = 832,
) -> torch.Tensor:
    video = _load_video(video_path, target_frames, height, width)
    dtype = next(vae.parameters()).dtype
    latent = vae.encode(
        [video.to(device=device, dtype=dtype)],
        device=device,
    )
    return latent.squeeze(0).to(dtype=torch.bfloat16)


@torch.no_grad()
def encode_first_frame(
    image_encoder: WanImageEncoder,
    image_path: str,
    device: torch.device | str,
    height: int = 480,
    width: int = 832,
) -> torch.Tensor:
    extension = image_path.lower().rsplit('.', 1)[-1]
    if extension in ('mp4', 'avi', 'mov', 'mkv'):
        reader = imageio.get_reader(image_path)
        image = Image.fromarray(next(iter(reader)))
        reader.close()
    else:
        image = Image.open(image_path).convert('RGB')
    if image.size != (width, height):
        image = image.resize((width, height), Image.Resampling.LANCZOS)
    tensor = torch.from_numpy(np.asarray(image).astype(np.float32))
    tensor = tensor.permute(2, 0, 1)
    tensor = tensor / 255.0 * 2.0 - 1.0
    dtype = next(image_encoder.parameters()).dtype
    embedding = image_encoder.encode_image(
        tensor.unsqueeze(0).to(device=device, dtype=dtype)
    )
    return embedding.squeeze(0).to(dtype=torch.bfloat16)


def encode_mask_to_latent(
    mask_video: torch.Tensor,
    target_shape: tuple[int, ...],
) -> torch.Tensor:
    _, target_frames, target_height, target_width = target_shape
    mask = torch.where(mask_video > 0.5, 1.0, 0.0)
    mask = F.interpolate(
        mask.unsqueeze(0),
        size=(target_frames, target_height * 2, target_width * 2),
        mode='nearest',
    ).squeeze(0).squeeze(0)
    mask = mask.view(target_frames, target_height, 2, target_width, 2)
    return mask.permute(2, 4, 0, 1, 3).reshape(
        4, target_frames, target_height, target_width
    )
