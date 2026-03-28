#!/bin/bash

set -e

conda activate muse

BASE_DIR=""
# Model path for probe_mistral checkpoint-834
MODEL_PROBE=""

# 1. harmbench
python eval.py \
  --file "${BASE_DIR}/harmbench.json" \
  --output_prefix "harmbench/llama2" \
  --eval_method "harmbench" \
  --model_path "${MODEL_PROBE}"

# 2. multi-turn
python eval.py \
  --file "${BASE_DIR}/multi_turn.json" \
  --output_prefix "multi-turn/llama2" \
  --eval_method "multi-turn" \
  --model_path "${MODEL_PROBE}"

# 3. oversafe
python eval.py \
  --file "${BASE_DIR}/oversafe.json" \
  --output_prefix "oversafe/llama2" \
  --eval_method "oversafe" \
  --model_path "${MODEL_PROBE}"

# 4. prefill_15
python eval.py \
  --file "${BASE_DIR}/ft_prefill_15.json" \
  --output_prefix "prefill/llama2(1)" \
  --eval_method "prefill" \
  --model_path "${MODEL_PROBE}"

# 5. prefill_20
python eval.py \
  --file "${BASE_DIR}/ft_prefill_20.json" \
  --output_prefix "prefill/llama2(2)" \
  --eval_method "prefill" \
  --model_path "${MODEL_PROBE}"