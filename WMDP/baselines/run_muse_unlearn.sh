#!/bin/bash

set -e

echo "Starting unlearning experiments..."

# NPO w/ PRISM on News
echo "Running: WMDP with PRISM..."
python unlearn.py \
    --algo prism_npo_gdr \
    --model_dir mistralai/Ministral-8B-Instruct-2410 \
    --tokenizer_dir mistralai/Ministral-8B-Instruct-2410 \
    --data_file ../data/forget.jsonl \
    --retain_data_file ../data/retain.jsonl \
    --out_dir ./ckpt/wmdp \
    --max_len 2048 \
    --epochs 10 \
    --lr 1e-5 \
    --per_device_batch_size 4 \
    --beta 0.1 \
    --coeff 1.0 \
    --npo_coeff 1.0 \
    --sam_rho 0.008  \
    --pretrained_probe_path ./WMDP/Probe/final_probe.pt \
    --adv_gamma 0.05 \
    --select_layer 28

echo "All experiments completed successfully!"
