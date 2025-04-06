import sys

import tyro
import torch as th
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

from src.generative_augmentations.config import Config
from src.generative_augmentations.models.mask_rcnn import MaskRCNNModel



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
    datamodule = ...


    # Set up trainer
    th.set_float32_matmul_precision("medium")

    trainer = Trainer(
        accelerator="auto",
        max_epochs=config.model.max_epochs,
        logger=logger,
        callbacks=[
            LearningRateMonitor(logging_interval="step"),
            ModelCheckpoint(
                dirpath="models/{logger.experiment.name}:{logger.experiment.hash}/",
                filename=f"{logger.experiment.id}" + ":top:{epoch:02d}:{step}:{val_loss:.3f}",
                every_n_train_steps=config.artifact.checkpoint_save_every_n_steps,
                save_top_k=config.artifact.checkpoint_save_n_best,
                mode="min",
                monitor="val_loss",
            ),
            ModelCheckpoint(
                dirpath="models/{logger.experiment.name}:{logger.experiment.hash}/",
                filename=f"{logger.experiment.id}" + ":all:{epoch:02d}:{step}:{val_loss:.3f}",
                every_n_train_steps=config.artifact.checkpoint_save_every_n_steps,
                save_top_k=-1,
            ),
        ],
    )


    # Set up model
    model = MaskRCNNModel(
        learning_rate_max=config.model.learning_rate_max,
        learning_rate_min=config.model.learning_rate_min,
        learning_rate_half_period=config.model.learning_rate_half_period,
        learning_rate_mult_period=config.model.learning_rate_mult_period,
        learning_rate_warmup_max=config.model.learning_rate_warmup_max,
        learning_rate_warmup_steps=config.model.learning_rate_warmup_steps,
        weight_decay=config.model.weight_decay,
        pretrained_backbone=config.model.pretrained_backbone,
        pretrained_head=config.model.pretrained_head,
    )


    # Start training
    trainer.fit(model, datamodule)


    # Return success
    return 0


if __name__ == "__main__":
    sys.exit(tyro.cli(main))
