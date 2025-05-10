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
from src.generative_augmentations.datasets.transforms import transforms



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
                                transform_train=transforms[config.augmentation.augmentation_name], 
                                transform_val=transforms["final transform"],
                                augmentation_instance_prob=config.augmentation.instance_prob,
                                augmentation_diffusion_prob=config.augmentation.diffusion_prob, 
                                data_fraction=config.dataloader.data_fraction,
                                data_dir=Path(config.dataloader.data_dir), 
                                pin_images_mem=config.dataloader.pin_images_to_ram)


    # Set up trainer
    th.set_float32_matmul_precision("medium")

    trainer = Trainer(
        accelerator="auto",
        max_epochs=config.model.max_epochs,
        logger=logger,
        check_val_every_n_epoch=config.artifact.check_val_every_n_epochs,
        callbacks=[
            LearningRateMonitor(logging_interval="step"),
            ModelCheckpoint(
                dirpath=f"{config.artifact.modeldir}/models/{logger.experiment.name}/",
                filename=f"{logger.experiment.id}" + "_top_{epoch:02d}_{step}_{val_loss:.3f}",
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


    # Save the last model
    trainer.save_checkpoint(
        f"{config.artifact.modeldir}/models/{logger.experiment.name}/last.ckpt",
        weights_only=True,
    )


    # Return success
    return 0


if __name__ == "__main__":
    args = tyro.cli(Config)
    sys.exit(main(args))
