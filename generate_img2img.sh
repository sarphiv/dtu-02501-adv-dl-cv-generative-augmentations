#!/bin/bash
#BSUB -J gen_aug_adlcv[1-3]
#BSUB -q gpua40
#BSUB -W 16:00
#BSUB -R "rusage[mem=6GB]"
#BSUB -R "span[hosts=1]"
#BSUB -n 3

#BSUB -o outputfiles/%J_%I_out-gen_aug.out 
#BSUB -e outputfiles/%J_%I_out-gen_aug.err



start_frac=$(echo "scale=6; ($LSB_JOBINDEX - 1) / 3" | bc)
end_frac=$(echo "scale=6; $LSB_JOBINDEX / 3" | bc)

export HF_HOME=/work3/s204121/huggingface

uv run python src/generative_augmentations/datasets/variant_generation.py \
    --variant_generation.input_dir '/work3/s204102/coco' \
    --variant_generation.output_dir '/work3/s204121/data/processed' \
    --varient_generation.full_pipeline False \
    --varient_generation.subset_start "$start_frac" \
    --varient_generation.subset_end "$end_frac"
