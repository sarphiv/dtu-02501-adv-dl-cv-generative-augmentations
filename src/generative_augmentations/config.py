from dataclasses import dataclass, field
from pathlib import Path

import tyro

from src.generative_augmentations.datasets.transforms import TransformTypes


@dataclass
class ArtifactConfig:
    project_name: str = "generative-augmentations"
    experiment_name: str | None = None
    wandb_entity: str = "metrics_logger"
    modeldir: str = ""

    log_image_every_n_epoch: int = 10
    check_val_every_n_epochs: int = 3
    checkpoint_save_n_best: int = 1
    checkpoint_save_every_n_steps: int = 10000

@dataclass
class VarientGenerationConfig: 
    subset_start: float = 0.0
    subset_end: float = 1.0
    num_variants: int = 3
    bbox_min_side_length: int = 75

@dataclass
class DataloaderConfig:
    processed_data_dir: str = "../scratch/coco" if Path("../scratch/coco").exists() else "data/processed"

    num_workers: int = 8
    batch_size: int = 32
    data_fraction: float = 1.0
    data_dir: str = "../scratch/coco" if Path("../scratch/coco").exists() else "data/processed"



@dataclass
class AugmentationConfig:
    augmentation_name: TransformTypes
    instance_prob: float | None = None
    diffusion_prob: float | None = None


augmentation_config_none = AugmentationConfig(
    augmentation_name="final transform",
    instance_prob=None,
    diffusion_prob=None
)

augmentation_config_simple = AugmentationConfig(
    augmentation_name="simple augmentation",
    instance_prob=None,
    diffusion_prob=None
)

augmentation_config_advanced = AugmentationConfig(
    augmentation_name="advanced augmentation",
    instance_prob=None,
    diffusion_prob=None
)

augmentation_config_instance = AugmentationConfig(
    augmentation_name="simple augmentation",
    instance_prob=0.3,
    diffusion_prob=None
)

augmentation_config_instance_advanced = AugmentationConfig(
    augmentation_name="advanced augmentation",
    instance_prob=0.3,
    diffusion_prob=None
)


@dataclass
class ModelConfig:
    max_epochs: int = 100

    learning_rate_max: float = 1e-2
    learning_rate_min: float = 1e-3
    learning_rate_half_period: int = 2000
    learning_rate_mult_period: int = 2
    learning_rate_warmup_max: float = 4e-2
    learning_rate_warmup_steps: int = 1000
    weight_decay: float = 1e-6
    pretrained_backbone: bool = True
    pretrained_head: bool = False




@dataclass
class Config:
    seed: int = 1337
    
    artifact: ArtifactConfig = field(default_factory=lambda: ArtifactConfig())
    dataloader: DataloaderConfig = field(default_factory=lambda: DataloaderConfig())
    model: ModelConfig = field(default_factory=lambda: ModelConfig())

    augmentation: AugmentationConfig = field(default_factory=lambda: augmentation_config_advanced)
    varient_generation: VarientGenerationConfig = field(default_factory=lambda: VarientGenerationConfig())
