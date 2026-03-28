#!/bin/bash

# Exit immediately if any command fails
set -e

conda activate muse

python eval.py \
  --model_dirs "./baselines/ckpt/books/prism/checkpoint-139" \
  --names "books" \
  --corpus books \
  --out_file "results/prism.csv"