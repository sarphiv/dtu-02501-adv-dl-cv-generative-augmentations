#!/bin/bash
#BSUB -J train_adlcv_instance_advanced_aug[1-4]
#BSUB -q gpua10
#BSUB -W 9:00
#BSUB -R "rusage[mem=8GB]"
#BSUB -R "span[hosts=1]"
#BSUB -n 4

#BSUB -o outputfiles/instance_advanced/%J_%I_instance_advanced_aug.out 
#BSUB -e outputfiles/instance_advanced/%J_%I_instance_advanced_aug.err



fractions=(1.0 0.5 0.25 0.125)
fraction=${fractions[$((LSB_JOBINDEX - 1))]}
reverse_fracs=(120 240 480 960) #(100 200 400 800)
reverse_frac=${reverse_fracs[$((LSB_JOBINDEX - 1))]}
n_e_between_images=(10 20 40 80)
n_e_between_image=${n_e_between_images[$((LSB_JOBINDEX - 1))]}

uv run python src/generative_augmentations/train.py \
    --dataloader.data_fraction "$fraction" \
    --model.max_epochs "$reverse_frac" \
    --artifact.check_val_every_n_epochs "$n_e_between_image" \
    --artifact.modeldir "/work3/s204102" \
    --augmentation.augmentation_name "advanced augmentation" \
    --augmentation.instance_prob 0.3
    