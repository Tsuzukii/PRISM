#!/bin/bash

set -e

echo "Starting unlearning experiments..."

conda activate muse

# NPO w/ PRISM on News
echo "Running: News NPO with PRISM..."
python unlearn.py \
    --algo sam_npo_gdr \
    --model_dir muse-bench/MUSE-News_target \
    --tokenizer_dir meta-llama/Llama-2-7b-hf \
    --data_file ../data/news/raw/forget.txt \
    --retain_data_file ../data/news/raw/retain1.txt \
    --out_dir ./ckpt/news/prism \
    --max_len 2048 \
    --epochs 10 \
    --lr 1e-5 \
    --per_device_batch_size 4 \
    --beta 0.1 \
    --coeff 1.0 \
    --npo_coeff 1.0 \
    --sam_rho 0.011 \
    --pretrained_probe_path ..probe/probe_news.pt \
    --adv_gamma 0.085 \
    --select_layer 24

echo "Finished: News NPO with PRISM"

# NPO w/ PRISM on Books
echo "Running: Books NPO with PRISM..."
python unlearn.py \
    --algo sam_npo_gdr \
    --model_dir muse-bench/MUSE-Books_target \
    --tokenizer_dir meta-llama/Llama-2-7b-hf \
    --data_file ../data/books/raw/forget.txt \
    --retain_data_file ../data/books/raw/retain1.txt \
    --out_dir ./ckpt/books/prism \
    --max_len 2048 \
    --epochs 1 \
    --lr 1e-5 \
    --per_device_batch_size 4 \
    --beta 0.1 \
    --coeff 2.5 \
    --npo_coeff 1.0 \
    --sam_rho 0.008 \
    --pretrained_probe_path ..probe/probe_books.pt \
    --adv_gamma 0.05 \
    --select_layer 32

echo "Finished: Books NPO with PRISM"

echo "All experiments completed successfully!"
