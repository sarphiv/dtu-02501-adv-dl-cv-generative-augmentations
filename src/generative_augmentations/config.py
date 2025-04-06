from dataclasses import dataclass, field

import tyro



@dataclass
class ArtifactConfig:
    project_name: str = "generative-augmentations"
    experiment_name: str = "experiment"
    wandb_entity: str = "metrics_logger"

    checkpoint_save_n_best: int = 3
    checkpoint_save_every_n_steps: int = 1000


@dataclass
class DataloaderConfig:
    num_workers: int = 8
    batch_size: int = 20



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
