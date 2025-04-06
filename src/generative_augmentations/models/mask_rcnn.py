from typing import cast

import torch as th
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
from torchvision.models import ResNet50_Weights
from lightning import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from pytorch_optimizer import create_optimizer


class MaskRCNNModel(LightningModule):
    def __init__(
        self,
        learning_rate_max: float,
        learning_rate_min: float,
        learning_rate_half_period: int,
        learning_rate_mult_period: int,
        learning_rate_warmup_max: float,
        learning_rate_warmup_steps: int,
        weight_decay: float,
        pretrained_backbone: bool = True, 
        pretrained_head: bool = False
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.learning_rate_max = learning_rate_max
        self.learning_rate_min = learning_rate_min
        self.learning_rate_half_period = learning_rate_half_period
        self.learning_rate_mult_period = learning_rate_mult_period
        self.learning_rate_warmup_max = learning_rate_warmup_max
        self.learning_rate_warmup_steps = learning_rate_warmup_steps
        self.weight_decay = weight_decay

        self.num_classes = 80


        self.model = maskrcnn_resnet50_fpn_v2(
            num_classes=self.num_classes,
            weights=MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1 if pretrained_head else None,
            weights_backbone=ResNet50_Weights.IMAGENET1K_V2 if pretrained_backbone else None,
        )

        self.forward = self.model.forward



    def training_step(self, batch: th.Tensor, batch_idx: int) -> th.Tensor:
        
        # TODO: batch should be images and targets,
        #  check discord
        images, targets = batch
        
        
        loss_dict = cast(dict[str, th.Tensor], self.model.forward(images, targets))
        loss = sum(loss for loss in loss_dict.values())
        
        # TODO: Logging


        return loss


    def validation_step(self, batch: th.Tensor, batch_idx: int) -> th.Tensor:
        # TODO: batch should be images and targets,
        #  check discord
        images, targets = batch
        
        
        loss_dict = cast(dict[str, th.Tensor], self.model.forward(images, targets))
        loss = sum(loss for loss in loss_dict.values())


        # TODO: Logging


        return loss



    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = create_optimizer(
            self.model, # type: ignore
            "adan",
            lr=self.learning_rate_max,
            weight_decay=self.weight_decay,
            use_lookahead=True,
            use_gc=True,
            eps=1e-6
        )

        # NOTE: Must instantiate cosine scheduler first,
        #  because super scheduler mutates the initial learning rate.
        lr_cosine = th.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer=optimizer,
            T_0=self.learning_rate_half_period,
            T_mult=self.learning_rate_mult_period,
            eta_min=self.learning_rate_min
        )
        lr_super = th.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=self.learning_rate_warmup_max,
            total_steps=self.learning_rate_warmup_steps,
        )
        lr_scheduler = th.optim.lr_scheduler.SequentialLR(
            optimizer=optimizer,
            schedulers=[lr_super, lr_cosine], # type: ignore
            milestones=[self.learning_rate_warmup_steps],
        )

        return { # type: ignore
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1,
                "name": "lr"
            }
        }