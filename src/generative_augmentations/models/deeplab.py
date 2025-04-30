from typing import cast
import types
import matplotlib.pyplot as plt

import torch as th
import torchvision as tv 
from lightning import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from pytorch_optimizer import create_optimizer
import wandb

from src.generative_augmentations.utils.plotting import plot_segmentation
from src.generative_augmentations.utils.metrics import compute_metrics


class DeepLabv3Lightning(LightningModule):
    def __init__(
        self,
        learning_rate_max: float,
        learning_rate_min: float,
        learning_rate_half_period: int,
        learning_rate_mult_period: int,
        learning_rate_warmup_max: float,
        learning_rate_warmup_steps: int,
        weight_decay: float,
        log_image_every_n_epoch: int = 10,
        pretrained_backbone: bool = True, 
        num_classes: int = 81
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
        self.log_image_every_n_epoch = log_image_every_n_epoch
        
        self.model: th.nn.Module = tv.models.segmentation.deeplabv3_resnet50(num_classes=num_classes)
        self.forward = self.model.forward

        self.criterion = th.nn.CrossEntropyLoss()

        

    def training_step(self, batch: th.Tensor, batch_idx: int) -> th.Tensor:
        # TODO: figure out how many classes we actually have.   
        # TODO: batch should be images and targets,
        #  check discord
        images, targets = batch
        
        outputs = self.model.forward(images)
        pred_segmentations = outputs['out'] 
        gt_segmentations = th.stack([targets[i]['semantic_mask'] for i in range(len(targets))]).long()
        loss = self.criterion(pred_segmentations, gt_segmentations)

        # TODO: Logging
        self.log('train_loss', loss)

        return loss


    def validation_step(self, batch: th.Tensor, batch_idx: int) -> th.Tensor:
        # TODO: batch should be images and targets,
        #  check discord
        images, targets = batch

        outputs = self.model.forward(images)
        pred_segmentations = outputs['out'] 
        gt_segmentations = th.stack([targets[i]['semantic_mask'] for i in range(len(targets))]).long()
        loss = self.criterion(pred_segmentations, gt_segmentations)

        # Evaluate IoU, dice, precision, specificity, sensitivity and accuracy  
        iou, dice, precision, specificity, sensitivity, accuracy = compute_metrics(pred_segmentations, gt_segmentations)
    
        # TODO: Logging
        self.log('val_loss', loss)
        self.log('val_iou', iou)
        self.log('val_dice', dice)
        self.log('val_precision', precision)
        self.log('val_specificity', specificity)
        self.log('val_sensitivity', sensitivity)
        self.log('val_accuracy', accuracy)


        if batch_idx == 0 and self.current_epoch % self.log_image_every_n_epoch == 0: 
            log_images = []
            for i, image in enumerate(images): 
                fig = plot_segmentation(image=image, target=targets[i], detection=pred_segmentations[i])
                log_images.append(wandb.Image(fig))
                plt.close(fig) #TODO: Check that this works 
            
            self.logger.log_image(key="Instance Segmentations", images=log_images)
            # plt.close('all') 
        # # Average Precision 
        # mAP = average_precision(targets=targets, detections=detections)
        # self.log('val_map', mAP)


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