from utils.lmdb_ import get_array_shape_from_lmdb, retrieve_row_from_lmdb
from torch.utils.data import Dataset
import numpy as np
import torch
import lmdb
import json
import cv2
from pathlib import Path
from PIL import Image
import os
import uuid
from functools import lru_cache


@lru_cache(maxsize=8192)
def _probe_video_metadata(video_path: str) -> tuple[int, int, int]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Failed to open video for metadata: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count, height, width


def _resolve_cached_video_metadata(
    target_video: str,
    sample_id: str,
) -> tuple[int, int, int]:
    if not target_video:
        raise FileNotFoundError(
            f"Missing target_video in cache meta for sample {sample_id}"
        )
    return _probe_video_metadata(target_video)


class TextDataset(Dataset):
    def __init__(self, prompt_path, extended_prompt_path=None):
        with open(prompt_path, encoding="utf-8") as f:
            self.prompt_list = [line.rstrip() for line in f]

        if extended_prompt_path is not None:
            with open(extended_prompt_path, encoding="utf-8") as f:
                self.extended_prompt_list = [line.rstrip() for line in f]
            assert len(self.extended_prompt_list) == len(self.prompt_list)
        else:
            self.extended_prompt_list = None

    def __len__(self):
        return len(self.prompt_list)

    def __getitem__(self, idx):
        batch = {
            "prompts": self.prompt_list[idx],
            "idx": idx,
        }
        if self.extended_prompt_list is not None:
            batch["extended_prompts"] = self.extended_prompt_list[idx]
        return batch


class ODERegressionLMDBDataset(Dataset):
    def __init__(self, data_path: str, max_pair: int = int(1e8)):
        self.env = lmdb.open(data_path, readonly=True,
                             lock=False, readahead=False, meminit=False)

        self.latents_shape = get_array_shape_from_lmdb(self.env, 'latents')
        self.max_pair = max_pair

    def __len__(self):
        return min(self.latents_shape[0], self.max_pair)

    def __getitem__(self, idx):
        """
        Outputs:
            - prompts: List of Strings
            - latents: Tensor of shape (num_denoising_steps, num_frames, num_channels, height, width). It is ordered from pure noise to clean image.
        """
        latents = retrieve_row_from_lmdb(
            self.env,
            "latents", np.float16, idx, shape=self.latents_shape[1:]
        )

        if len(latents.shape) == 4:
            latents = latents[None, ...]

        prompts = retrieve_row_from_lmdb(
            self.env,
            "prompts", str, idx
        )
        return {
            "prompts": prompts,
            "ode_latent": torch.tensor(latents, dtype=torch.float32)
        }





class LatentLMDBDataset(Dataset):
    def __init__(self, data_path: str, max_pair: int = int(1e8)):
        self.env = lmdb.open(data_path, readonly=True,
                             lock=False, readahead=False, meminit=False)

        self.latents_shape = get_array_shape_from_lmdb(self.env, 'latents')
        self.max_pair = max_pair

    def __len__(self):
        return min(self.latents_shape[0], self.max_pair)

    def __getitem__(self, idx):
        """
        Outputs:
            - prompts: List of Strings
            - latents: Tensor of shape (num_denoising_steps, num_frames, num_channels, height, width). It is ordered from pure noise to clean image.
        """
        latents = retrieve_row_from_lmdb(
            self.env,
            "latents", np.float16, idx, shape=self.latents_shape[1:]
        )

        if len(latents.shape) == 4:
            latents = latents[None, ...]

        prompts = retrieve_row_from_lmdb(
            self.env,
            "prompts", str, idx
        )
        return {
            "prompts": prompts,
            "clean_latent": torch.tensor(latents, dtype=torch.float32)[-1]
        }


class EgoSimCacheDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        max_pair: int = int(1e8),
        physics_track_mode: str = 'old',
        require_physics: bool = False,
    ):
        self.data_path = Path(data_path)
        if not self.data_path.exists():
            raise FileNotFoundError(f"EgoSim cache root not found: {self.data_path}")
        if physics_track_mode not in {'old', 'new'}:
            raise ValueError(f"Unsupported physics_track_mode={physics_track_mode!r}, expected 'old' or 'new'")
        if require_physics and physics_track_mode != 'new':
            raise ValueError('Object-DMD requires physics_track_mode=new.')
        if physics_track_mode == 'new':
            self.physics_track_name = 'grounded_sam_tracks.pt'
            self.physics_visibility_name = 'grounded_sam_visibility.pt'
        else:
            self.physics_track_name = 'physics_tracks.pt'
            self.physics_visibility_name = 'physics_visibility.pt'
        self.sample_dirs = self._load_sample_dirs()
        self.max_pair = max_pair
        self.require_physics = require_physics

    def __len__(self):
        return min(len(self.sample_dirs), self.max_pair)

    def _load_sample_dirs(self) -> list[Path]:
        index_path = self.data_path / '.egosim_sample_dirs.txt'
        rebuild_index = os.environ.get('EGOSIM_CACHE_REBUILD_INDEX', '0') == '1'
        if index_path.exists() and not rebuild_index:
            sample_ids = [
                line.strip()
                for line in index_path.read_text(encoding='utf-8').splitlines()
                if line.strip()
            ]
            return [self.data_path / sample_id for sample_id in sample_ids]

        sample_dirs = [
            path for path in sorted(self.data_path.iterdir())
            if path.is_dir() and (path / 'clean_latent.pt').exists() and (path / 'meta.json').exists()
        ]
        tmp_index_path = index_path.with_suffix(
            f'{index_path.suffix}.{os.getpid()}.{uuid.uuid4().hex}.tmp'
        )
        tmp_index_path.write_text(
            ''.join(f'{path.name}\n' for path in sample_dirs),
            encoding='utf-8',
        )
        try:
            os.replace(tmp_index_path, index_path)
        except FileNotFoundError:
            if index_path.exists():
                sample_ids = [
                    line.strip()
                    for line in index_path.read_text(encoding='utf-8').splitlines()
                    if line.strip()
                ]
                return [self.data_path / sample_id for sample_id in sample_ids]
            raise
        return sample_dirs

    @staticmethod
    def _load_latent(sample_dir: Path, name: str) -> torch.Tensor:
        tensor = torch.load(sample_dir / name, map_location='cpu')
        if tensor.ndim != 4:
            raise ValueError(f"Expected 4D tensor for {sample_dir / name}, got {tuple(tensor.shape)}")
        return tensor.permute(1, 0, 2, 3).contiguous().to(dtype=torch.float32)

    @staticmethod
    def _build_hand_track_mask(
        hand_seg_path: str,
        physics_tracks: torch.Tensor,
        raw_height: int,
        raw_width: int,
    ) -> torch.Tensor:
        hand_mask = torch.zeros(
            (physics_tracks.shape[0],),
            dtype=torch.bool,
        )
        if not hand_seg_path:
            raise FileNotFoundError('Missing hand_seg path in cache metadata.')
        if not os.path.exists(hand_seg_path):
            raise FileNotFoundError(f'hand_seg not found: {hand_seg_path}')

        hand_image = Image.open(hand_seg_path).convert('L')
        if hand_image.size != (raw_width, raw_height):
            hand_image = hand_image.resize(
                (raw_width, raw_height),
                resample=Image.Resampling.NEAREST,
            )
        hand_array = np.array(hand_image) > 0
        if not hand_array.any():
            return hand_mask

        query_points = physics_tracks[:, 0].round().to(dtype=torch.long)
        query_points[:, 0].clamp_(0, raw_width - 1)
        query_points[:, 1].clamp_(0, raw_height - 1)
        hand_hits = hand_array[
            query_points[:, 1].cpu().numpy(),
            query_points[:, 0].cpu().numpy(),
        ]
        return torch.from_numpy(hand_hits).to(dtype=torch.bool)

    def __getitem__(self, idx):
        sample_dir = self.sample_dirs[idx]
        meta = json.loads((sample_dir / 'meta.json').read_text())
        sample = {
            'prompts': str(meta.get('prompt', '')),
            'sample_id': sample_dir.name,
            'clean_latent': self._load_latent(sample_dir, 'clean_latent.pt'),
            'ego_prior_latent': self._load_latent(sample_dir, 'ego_prior_latent.pt'),
            'hand_latent': self._load_latent(sample_dir, 'hand_latent.pt'),
            'mask_latent': self._load_latent(sample_dir, 'mask_latent.pt'),
            'prompt_embeds': torch.load(sample_dir / 'prompt_embedding.pt', map_location='cpu').to(dtype=torch.float32),
            'image_embeds': torch.load(sample_dir / 'image_embedding.pt', map_location='cpu').to(dtype=torch.float32),
        }
        if self.require_physics:
            target_video = str(meta.get('target_video', ''))
            raw_num_frames, raw_height, raw_width = _resolve_cached_video_metadata(
                target_video=target_video,
                sample_id=sample_dir.name,
            )
            physics_path = sample_dir / self.physics_track_name
            visibility_path = sample_dir / self.physics_visibility_name
            object_masks_name = meta.get('physics_object_masks_file')
            object_masks_path = sample_dir / object_masks_name if object_masks_name else None
            if not (
                physics_path.exists()
                and visibility_path.exists()
                and object_masks_path is not None
                and object_masks_path.exists()
            ):
                raise FileNotFoundError(
                    'Object-DMD requires tracks, visibility, and object-mask '
                    f'metadata for every sample: {sample_dir}'
                )
            physics_tracks = torch.load(
                physics_path, map_location='cpu'
            ).to(dtype=torch.float32)
            physics_visibility = torch.load(
                visibility_path, map_location='cpu'
            ).to(dtype=torch.bool)
            if physics_tracks.ndim != 3 or physics_tracks.shape[-1] != 2:
                raise ValueError(
                    f'Expected tracks [num_tracks, num_frames, 2], got '
                    f'{tuple(physics_tracks.shape)} in {physics_path}'
                )
            if tuple(physics_visibility.shape) != tuple(physics_tracks.shape[:2]):
                raise ValueError(
                    f'Visibility shape {tuple(physics_visibility.shape)} does not match '
                    f'track shape {tuple(physics_tracks.shape)} in {sample_dir}'
                )
            object_masks = torch.load(object_masks_path, map_location='cpu')
            if 'track_object_ids' not in object_masks:
                raise KeyError(f'track_object_ids missing from {object_masks_path}')
            object_track_ids = object_masks['track_object_ids'].to(dtype=torch.int16)
            if object_track_ids.shape[0] != physics_tracks.shape[0]:
                raise ValueError(
                    f'track_object_ids length {object_track_ids.shape[0]} does not match '
                    f'num_tracks {physics_tracks.shape[0]} in {object_masks_path}'
                )
            hand_track_mask = self._build_hand_track_mask(
                hand_seg_path=str(meta.get('hand_seg', '')),
                physics_tracks=physics_tracks,
                raw_height=raw_height,
                raw_width=raw_width,
            )
            sample.update({
                'physics_tracks': physics_tracks,
                'physics_visibility': physics_visibility,
                'physics_object_track_ids': object_track_ids,
                'physics_hand_track_mask': hand_track_mask,
                'physics_valid': torch.tensor(True, dtype=torch.bool),
            })
            sample.update({
                'raw_num_frames': torch.tensor(raw_num_frames, dtype=torch.long),
                'raw_height': torch.tensor(raw_height, dtype=torch.long),
                'raw_width': torch.tensor(raw_width, dtype=torch.long),
                'train_height': torch.tensor(int(meta.get('height', raw_height)), dtype=torch.long),
                'train_width': torch.tensor(int(meta.get('width', raw_width)), dtype=torch.long),
            })
        return sample


class ShardingLMDBDataset(Dataset):
    def __init__(self, data_path: str, max_pair: int = int(1e8)):
        self.envs = []
        self.index = []

        for fname in sorted(os.listdir(data_path)):
            path = os.path.join(data_path, fname)
            env = lmdb.open(path,
                            readonly=True,
                            lock=False,
                            readahead=False,
                            meminit=False)
            self.envs.append(env)

        self.latents_shape = [None] * len(self.envs)
        for shard_id, env in enumerate(self.envs):
            self.latents_shape[shard_id] = get_array_shape_from_lmdb(env, 'latents')
            for local_i in range(self.latents_shape[shard_id][0]):
                self.index.append((shard_id, local_i))

            # print("shard_id ", shard_id, " local_i ", local_i)

        self.max_pair = max_pair

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        """
            Outputs:
                - prompts: List of Strings
                - latents: Tensor of shape (num_denoising_steps, num_frames, num_channels, height, width). It is ordered from pure noise to clean image.
        """
        shard_id, local_idx = self.index[idx]

        latents = retrieve_row_from_lmdb(
            self.envs[shard_id],
            "latents", np.float16, local_idx,
            shape=self.latents_shape[shard_id][1:]
        )

        if len(latents.shape) == 4:
            latents = latents[None, ...]

        prompts = retrieve_row_from_lmdb(
            self.envs[shard_id],
            "prompts", str, local_idx
        )

        return {
            "prompts": prompts,
            "ode_latent": torch.tensor(latents, dtype=torch.float32)
        }



class TextImagePairDataset(Dataset):
    def __init__(
        self,
        data_dir,
        transform=None,
        eval_first_n=-1,
        pad_to_multiple_of=None
    ):
        """
        Args:
            data_dir (str): Path to the directory containing:
                - target_crop_info_*.json (metadata file)
                - */ (subdirectory containing images with matching aspect ratio)
            transform (callable, optional): Optional transform to be applied on the image
        """
        self.transform = transform
        data_dir = Path(data_dir)

        # Find the metadata JSON file
        metadata_files = list(data_dir.glob('target_crop_info_*.json'))
        if not metadata_files:
            raise FileNotFoundError(f"No metadata file found in {data_dir}")
        if len(metadata_files) > 1:
            raise ValueError(f"Multiple metadata files found in {data_dir}")

        metadata_path = metadata_files[0]
        # Extract aspect ratio from metadata filename (e.g. target_crop_info_26-15.json -> 26-15)
        aspect_ratio = metadata_path.stem.split('_')[-1]

        # Use aspect ratio subfolder for images
        self.image_dir = data_dir / aspect_ratio
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")

        # Load metadata
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        eval_first_n = eval_first_n if eval_first_n != -1 else len(self.metadata)
        self.metadata = self.metadata[:eval_first_n]

        # Verify all images exist
        for item in self.metadata:
            image_path = self.image_dir / item['file_name']
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")

        self.dummy_prompt = "DUMMY PROMPT"
        self.pre_pad_len = len(self.metadata)
        if pad_to_multiple_of is not None and len(self.metadata) % pad_to_multiple_of != 0:
            # Duplicate the last entry
            self.metadata += [self.metadata[-1]] * (
                pad_to_multiple_of - len(self.metadata) % pad_to_multiple_of
            )

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        """
        Returns:
            dict: A dictionary containing:
                - image: PIL Image
                - caption: str
                - target_bbox: list of int [x1, y1, x2, y2]
                - target_ratio: str
                - type: str
                - origin_size: tuple of int (width, height)
        """
        item = self.metadata[idx]

        # Load image
        image_path = self.image_dir / item['file_name']
        image = Image.open(image_path).convert('RGB')

        # Apply transform if specified
        if self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'prompts': item['caption'],
            'target_bbox': item['target_crop']['target_bbox'],
            'target_ratio': item['target_crop']['target_ratio'],
            'type': item['type'],
            'origin_size': (item['origin_width'], item['origin_height']),
            'idx': idx
        }



def cycle(dl):
    while True:
        for data in dl:
            yield data
