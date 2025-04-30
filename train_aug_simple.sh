#!/bin/bash
#BSUB -J train_adlcv_no_aug[1-4]
#BSUB -q gpua10
#BSUB -W 12:00
#BSUB -R "rusage[mem=6GB]"
#BSUB -R "span[hosts=1]"
#BSUB -n 4

#BSUB -o %J_%I_no_aug.out 
#BSUB -e %J_%I_no_aug.err



fractions=(1.0 0.5 0.25 0.125)
fraction=${fractions[$((LSB_JOBINDEX - 1))]}
reverse_fracs=(100 200 400 800)
reverse_frac=${reverse_fracs[$((LSB_JOBINDEX - 1))]}
n_e_between_images=(10 20 40 80)
n_e_between_image=${n_e_between_images[$((LSB_JOBINDEX - 1))]}

uv run python src/generative_augmentations/train.py \
    --dataloader.data_fraction "$fraction" \
    --model.max_epochs "$reverse_frac" \
    --artifact.log_image_every_n_epoch "$n_e_between_image" \
    --artifact.modeldir "/work3/s204102" \
    --dataloader.augmentations "no"
    