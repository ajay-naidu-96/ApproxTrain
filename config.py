"""
Configuration file for transformer training with native and approximate multipliers.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """Transformer model hyperparameters."""
    vocab_size: int = 65  # Character-level vocabulary
    sequence_length: int = 128  # Context window
    d_model: int = 256  # Model dimension
    d_ff: int = 1024  # Feed-forward dimension (4x d_model)
    num_layers: int = 4  # Number of decoder layers
    num_heads: int = 4  # Number of attention heads
    d_k: int = 64  # Key dimension (d_model / num_heads)
    d_v: int = 64  # Value dimension (d_model / num_heads)
    dropout_rate: float = 0.4
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.d_model % self.num_heads == 0, \
            f"d_model ({self.d_model}) must be divisible by num_heads ({self.num_heads})"
        self.d_k = self.d_model // self.num_heads
        self.d_v = self.d_model // self.num_heads


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    batch_size: int = 64
    epochs: int = 50
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 4000
    gradient_clip_norm: float = 1.0
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_freq: int = 5  # Save every N epochs
    
    # Logging
    log_dir: str = "logs"
    log_freq: int = 100  # Log every N steps
    sample_freq: int = 500  # Generate text samples every N steps
    
    # Validation
    val_freq: int = 1  # Validate every N epochs
    val_steps: int = 100  # Number of validation steps
    max_steps_per_epoch: Optional[int] = None  # Limit steps per epoch


@dataclass
class DataConfig:
    """Data pipeline configuration."""
    data_dir: str = "data"
    dataset_name: str = "shakespeare"
    train_split: float = 0.9
    sequence_length: int = 128
    buffer_size: int = 10000  # For shuffling
    prefetch_size: int = 2  # Number of batches to prefetch


@dataclass
class MultiplierConfig:
    """Approximate multiplier configuration."""
    use_approximate: bool = False
    lut_file: Optional[str] = None  # Path to LUT file (e.g., "lut/MBM_7.bin")
    multiplier_name: str = "FP32"  # For logging purposes
    
    def __post_init__(self):
        """Validate multiplier configuration."""
        if self.use_approximate:
            assert self.lut_file is not None, \
                "lut_file must be specified when use_approximate=True"
            import os
            assert os.path.exists(self.lut_file), \
                f"LUT file not found: {self.lut_file}"


@dataclass
class Config:
    """Complete configuration for transformer training."""
    model: ModelConfig
    training: TrainingConfig
    data: DataConfig
    multiplier: MultiplierConfig
    
    @classmethod
    def create_fp32_config(cls):
        """Create configuration for FP32 baseline."""
        return cls(
            model=ModelConfig(),
            training=TrainingConfig(),
            data=DataConfig(),
            multiplier=MultiplierConfig(use_approximate=False)
        )
    
    @classmethod
    def create_approximate_config(cls, lut_file: str, multiplier_name: str):
        """Create configuration for approximate multiplier baseline."""
        return cls(
            model=ModelConfig(),
            training=TrainingConfig(),
            data=DataConfig(),
            multiplier=MultiplierConfig(
                use_approximate=True,
                lut_file=lut_file,
                multiplier_name=multiplier_name
            )
        )
