#!/bin/bash
#BSUB -J gen_aug_adlcv[1-16]
#BSUB -q gpua100
#BSUB -W 16:00
#BSUB -R "rusage[mem=6GB]"
#BSUB -R "span[hosts=1]"
#BSUB -n 4

#BSUB -o outputfiles/%J_%I_out-gen_aug.out 
#BSUB -e outputfiles/%J_%I_out-gen_aug.err



start_frac=$(echo "scale=6; ($LSB_JOBINDEX - 1) / 16" | bc)
end_frac=$(echo "scale=6; $LSB_JOBINDEX / 16" | bc)

export HF_HOME=/work3/s204102/huggingface_cache

uv run python src/generative_augmentations/datasets/variant_generation.py --varient_generation.subset_start "$start_frac" --varient_generation.subset_end "$end_frac"
