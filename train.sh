#!/bin/bash
#BSUB -J train_adlcv[1-4]
#BSUB -q gpua100
#BSUB -W 24:00
#BSUB -R "rusage[mem=2GB]"
#BSUB -R "span[hosts=1]"
#BSUB -n 8

#BSUB -o %J_%I_out-train.out 
#BSUB -e %J_%I_out-train.err



fractions=(1.0 0.5 0.25 0.125)
fraction=${fractions[$((LSB_JOBINDEX - 1))]}

uv run python src/generative_augmentations/train.py --dataloader.data_fraction "$fraction"

# uv run python src/generative_augmentations/train.py
# uv run python src/generative_augmentations/train.py --dataloader.data_fraction 0.5
# uv run python src/generative_augmentations/train.py --dataloader.data_fraction 0.25
# uv run python src/generative_augmentations/train.py --dataloader.data_fraction 0.125