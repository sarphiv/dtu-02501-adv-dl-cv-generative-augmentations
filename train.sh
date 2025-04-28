#!/bin/bash
#BSUB -J train_adlcv[1-4]
#BSUB -q gpua40
#BSUB -W 24:00
#BSUB -R "rusage[mem=2GB]"
#BSUB -R "span[hosts=1]"
#BSUB -n 8

#BSUB -o %J_%I_out-train.out 
#BSUB -e %J_%I_out-train.err



fractions=(1.0 0.5 0.25 0.125)
fraction=${fractions[$((LSB_JOBINDEX - 1))]}
reverse_fracs=(100 200 400 800)
reverse_frac=${reverse_fracs[$((LSB_JOBINDEX - 1))]}
n_e_between_images=(10 20 40 80)
n_e_between_image=${n_e_between_images[$((LSB_JOBINDEX - 1))]}

uv run python src/generative_augmentations/train.py --dataloader.data_fraction "$fraction" --model.max_epochs "$reverse_frac" --artifact.log_image_every_n_epoch "$n_e_between_image" --artifact.modeldir "/work3/s204102"

# uv run python src/generative_augmentations/train.py
# uv run python src/generative_augmentations/train.py --dataloader.data_fraction 0.5
# uv run python src/generative_augmentations/train.py --dataloader.data_fraction 0.25
# uv run python src/generative_augmentations/train.py --dataloader.data_fraction 0.125
