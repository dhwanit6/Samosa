"""
Configuration for masked diffusion language modeling.
"""
from dataclasses import dataclass


@dataclass
class DiffusionLMConfig:
    # Vocabulary
    vocab_size: int = 32001
    mask_token_id: int = 32000
    pad_token_id: int = 0

    # Backbone
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    d_ff: int = 1536
    max_seq_len: int = 1024
    dropout: float = 0.0

    # Diffusion process
    timesteps: int = 32
    min_mask_rate: float = 0.05
    max_mask_rate: float = 0.95
    time_mode: str = "discrete"  # discrete | continuous
    timestep_sampling: str = "uniform"  # uniform | stratified
    masking_strategy: str = "token"  # token | span
    mean_span_length: float = 3.0

    # Sampling
    sample_steps: int = 16
    block_size: int = 64
    confidence_stop: float = 0.98

    # Initialization
    init_std: float = 0.02

    def __post_init__(self):
        if self.mask_token_id >= self.vocab_size:
            raise ValueError("mask_token_id must be < vocab_size")
        if self.pad_token_id >= self.vocab_size:
            raise ValueError("pad_token_id must be < vocab_size")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.timesteps < 2:
            raise ValueError("timesteps must be >= 2")
        if not (0.0 <= self.min_mask_rate < self.max_mask_rate <= 1.0):
            raise ValueError("mask rates must satisfy 0 <= min < max <= 1")
        if self.time_mode not in {"discrete", "continuous"}:
            raise ValueError("time_mode must be one of: discrete, continuous")
        if self.timestep_sampling not in {"uniform", "stratified"}:
            raise ValueError("timestep_sampling must be one of: uniform, stratified")
        if self.masking_strategy not in {"token", "span"}:
            raise ValueError("masking_strategy must be one of: token, span")
        if self.mean_span_length <= 0:
            raise ValueError("mean_span_length must be > 0")
        if self.block_size < 1:
            raise ValueError("block_size must be >= 1")
        if not (0.0 <= self.confidence_stop < 1.0):
            raise ValueError("confidence_stop must be in [0, 1)")
