#!/bin/bash

set -e

conda activate muse

MODEL_PATH="./baselines/"

python gen_response.py \
  --mode "general" \
  --model_path "$MODEL_PATH" \
  --eval_path "/Jailbreak/data/xstest_prompts.csv"

python gen_response.py \
  --mode "prefill" \
  --model_path "$MODEL_PATH" \
  --eval_path "/Jailbreak/data/llama_harmful-prefix.jsonl"

python gen_response.py \
  --mode "harmbench" \
  --model_path "$MODEL_PATH" \
  --eval_path "/Jailbreak/data/mistral_autodan.json"

python gen_response.py \
  --mode "multi_turn" \
  --model_path "$MODEL_PATH" \
  --eval_path "/Jailbreak/data/llama_multi.json"