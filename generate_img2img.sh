#!/bin/bash
#BSUB -J gen_aug_adlcv[1-4]
#BSUB -q gpuv100
#BSUB -W 12:00
#BSUB -R "rusage[mem=6GB]"
#BSUB -R "span[hosts=1]"
#BSUB -n 3

#BSUB -o outputfiles/%J_%I_out-gen_aug.out 
#BSUB -e outputfiles/%J_%I_out-gen_aug.err



start_frac=$(echo "scale=6; ($LSB_JOBINDEX - 1) / 4" | bc)
end_frac=$(echo "scale=6; $LSB_JOBINDEX / 4" | bc)

export HF_HOME=/work3/s204121/huggingface

uv run python src/generative_augmentations/datasets/variant_generation.py \
    --variant_generation.input_dir "/work3/s204102/coco" \
    --variant_generation.output_dir "/work3/s204121/data/processed" \
    --variant_generation.full_pipeline False \
    --variant_generation.subset_start "$start_frac" \
    --variant_generation.subset_end "$end_frac"
