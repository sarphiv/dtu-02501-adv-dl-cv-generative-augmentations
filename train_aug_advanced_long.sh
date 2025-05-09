#!/bin/bash
#BSUB -J train_adlcv_advanced_aug_long
#BSUB -q gpuv100
#BSUB -W 24:00
#BSUB -R "rusage[mem=2GB]"
#BSUB -R "span[hosts=1]"
#BSUB -n 16

#BSUB -o outputfiles/advanced/%J_%I_advanced_aug.out 
#BSUB -e outputfiles/advanced/%J_%I_advanced_aug.err




uv run python src/generative_augmentations/train.py \
    --dataloader.data_fraction 1.0 \
    --model.max_epochs 385 \
    --artifact.check_val_every_n_epochs 10 \
    --artifact.modeldir "/work3/s204121/models" \
    --dataloader.data_dir "/work3/s204102/coco" \
    --augmentation.augmentation_name "advanced augmentation"
