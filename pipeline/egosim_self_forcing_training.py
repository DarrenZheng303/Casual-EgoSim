import torch

from pipeline.self_forcing_training import SelfForcingTrainingPipeline


class EgoSimCPUKeyValueCache:
    """Opaque to FSDP input traversal while remaining indexable by model blocks."""

    def __init__(self, entries):
        self.entries = entries

    def __getitem__(self, index):
        return self.entries[index]

    def __len__(self):
        return len(self.entries)


class EgoSimSelfForcingTrainingPipeline(SelfForcingTrainingPipeline):
    """Self-forcing rollout with EgoSim-14B cache dimensions."""

    def __init__(self, *args, **kwargs):
        self.local_attn_size = kwargs.pop('local_attn_size', -1)
        self.kv_cache_cpu_offload = kwargs.pop('kv_cache_cpu_offload', False)
        self.kv_cache_cpu_offload_layers = kwargs.pop(
            'kv_cache_cpu_offload_layers', None)
        num_max_frames = kwargs.get('num_max_frames', 21)
        super().__init__(*args, **kwargs)
        self.num_transformer_blocks = 40
        self.num_attention_heads = 40
        self.head_dim = 128
        self.num_max_frames = num_max_frames
        if self.local_attn_size != -1:
            self.kv_cache_size = self.local_attn_size * self.frame_seq_length
        if self.kv_cache_cpu_offload_layers is None:
            self.kv_cache_cpu_offload_layers = (
                self.num_transformer_blocks if self.kv_cache_cpu_offload else 0)
        self.kv_cache_cpu_offload_layers = int(self.kv_cache_cpu_offload_layers)
        if not 0 <= self.kv_cache_cpu_offload_layers <= self.num_transformer_blocks:
            raise ValueError(
                'kv_cache_cpu_offload_layers must be between 0 and '
                f'{self.num_transformer_blocks}')

    def _initialize_kv_cache(self, batch_size, dtype, device):
        entries = []
        for block_index in range(self.num_transformer_blocks):
            offload = block_index < self.kv_cache_cpu_offload_layers
            cache_device = torch.device('cpu') if offload else device
            cache_factory = torch.empty if offload else torch.zeros
            cache_size = (
                self.num_max_frames * self.frame_seq_length
                if offload else self.kv_cache_size
            )
            entries.append({
                'k': cache_factory(
                    [batch_size, cache_size, self.num_attention_heads, self.head_dim],
                    dtype=dtype,
                    device=cache_device,
                ),
                'v': cache_factory(
                    [batch_size, cache_size, self.num_attention_heads, self.head_dim],
                    dtype=dtype,
                    device=cache_device,
                ),
                'global_end_index': torch.tensor([0], dtype=torch.long, device=cache_device),
                'local_end_index': torch.tensor([0], dtype=torch.long, device=cache_device),
            })
        self.kv_cache1 = (
            EgoSimCPUKeyValueCache(entries)
            if self.kv_cache_cpu_offload_layers > 0
            else entries
        )

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        # EgoSim uses I2V cross-attention, whose image/text branches do not use
        # the T2V cross-attention cache. Keep placeholders for the shared API.
        self.crossattn_cache = [None for _ in range(self.num_transformer_blocks)]
