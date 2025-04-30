import sys
from pathlib import Path

import tyro
import torch as th
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.generative_augmentations.config import Config
from src.generative_augmentations.models.deeplab import DeepLabv3Lightning
from src.generative_augmentations.datasets.datamodule import COCODataModule

my_aug = A.Compose(
    [
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=45, border_mode=0, p=0.5),  
        A.RandomCropFromBorders(crop_bottom=0.3, crop_left=0.3, crop_right=0.3, crop_top=0.3),
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ]
)
my_trans = A.Compose(
    [
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ]
)


def main(config: Config) -> int:
    # Reproducibility
    seed_everything(config.seed)

    # Set up logging
    logger = WandbLogger(
        project=config.artifact.project_name,
        name=config.artifact.experiment_name,
        entity=config.artifact.wandb_entity,
    )

    logger.experiment.config.update(config)


    # Set up data
    datamodule = COCODataModule(num_workers=config.dataloader.num_workers,
                                batch_size=config.dataloader.batch_size,
                                transform=my_trans,
                                augmentations=my_aug, 
                                data_fraction=config.dataloader.data_fraction,
                                data_dir=Path(config.dataloader.data_dir))


    # Set up trainer
    th.set_float32_matmul_precision("medium")

    trainer = Trainer(
        accelerator="auto",
        max_epochs=config.model.max_epochs,
        logger=logger,
        callbacks=[
            LearningRateMonitor(logging_interval="step"),
            ModelCheckpoint(
                dirpath=f"{config.artifact.modeldir}/models/{logger.experiment.name}/",
                filename=f"{logger.experiment.id}" + ":top:{epoch:02d}:{step}:{val_loss:.3f}",
                every_n_train_steps=config.artifact.checkpoint_save_every_n_steps,
                save_top_k=config.artifact.checkpoint_save_n_best,
                mode="min",
                monitor="val_loss",
            ),
            # ModelCheckpoint(
            #     dirpath="models/{logger.experiment.name}:{logger.experiment.hash}/",
            #     filename=f"{logger.experiment.id}" + ":all:{epoch:02d}:{step}:{val_loss:.3f}",
            #     every_n_train_steps=config.artifact.checkpoint_save_every_n_steps,
            #     save_top_k=-1,
            # ),
        ],
    )


    # Set up model
    # model = MaskRCNNModel(
    #     learning_rate_max=config.model.learning_rate_max,
    #     learning_rate_min=config.model.learning_rate_min,
    #     learning_rate_half_period=config.model.learning_rate_half_period,
    #     learning_rate_mult_period=config.model.learning_rate_mult_period,
    #     learning_rate_warmup_max=config.model.learning_rate_warmup_max,
    #     learning_rate_warmup_steps=config.model.learning_rate_warmup_steps,
    #     weight_decay=config.model.weight_decay,
    #     pretrained_backbone=config.model.pretrained_backbone,
    #     pretrained_head=config.model.pretrained_head,
    # )
    model = DeepLabv3Lightning(
        learning_rate_max=config.model.learning_rate_max,
        learning_rate_min=config.model.learning_rate_min,
        learning_rate_half_period=config.model.learning_rate_half_period,
        learning_rate_mult_period=config.model.learning_rate_mult_period,
        learning_rate_warmup_max=config.model.learning_rate_warmup_max,
        learning_rate_warmup_steps=config.model.learning_rate_warmup_steps,
        log_image_every_n_epoch=config.artifact.log_image_every_n_epoch,
        weight_decay=config.model.weight_decay,
        pretrained_backbone=config.model.pretrained_backbone,
        num_classes=81
    )

    # Start training
    trainer.fit(model, datamodule=datamodule)


    # Return success
    return 0


if __name__ == "__main__":
    args = tyro.cli(Config)
    sys.exit(main(args))
