#!/bin/bash
#BSUB -J gen_aug_adlcv[1-64]
#BSUB -q gpua10
#BSUB -W 12:00
#BSUB -R "rusage[mem=5GB]"
#BSUB -R "span[hosts=1]"
#BSUB -n 4

#BSUB -o %J_%I_out-gen_aug.out 
#BSUB -e %J_%I_out-gen_aug.err



start_frac=$(echo "scale=6; ($LSB_JOBINDEX - 1) / 64" | bc)
end_frac=$(echo "scale=6; $LSB_JOBINDEX / 64" | bc)

export HF_HOME=/work3/s204102/huggingface_cache

uv run python src/generative_augmentations/datasets/variant_generation.py --varient_generation.subset_start "$start_frac" --varient_generation.subset_end "$end_frac"
