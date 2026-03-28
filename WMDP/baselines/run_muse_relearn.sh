#!/bin/bash

# Exit immediately if any command fails
set -e

echo "Starting relearning experiments..."


# Define model paths and data files
WMDP_MODEL="./baselines/ckpt/wmdp/checkpoint-1020"
TOKENIZER="mistralai/Ministral-8B-Instruct-2410"
WMDP_DATA="../data/books/raw/forget.txt"

WMDP_MAX_STEPS=(100 125 150)

# Run experiments for WMDP
for steps in "${WMDP_MAX_STEPS[@]}"; do
    echo "Running: Books Relearning with max_steps=$steps..."
    python relearn.py \
        --model_dir "$WMDP_MODEL" \
        --tokenizer_dir "$TOKENIZER" \
        --data_file "$WMDP_DATA" \
        --max_len 2048 \
        --max_steps "$steps" \
        --lr 1e-5 \
        --per_device_batch_size 4
    echo "Finished: Books Relearning with max_steps=$steps"
done


echo "All relearning experiments completed successfully!"
