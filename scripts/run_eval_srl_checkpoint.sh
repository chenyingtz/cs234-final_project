#!/bin/bash
# Evaluate SRL fine-tuned checkpoint on AIME24 (greedy + Avg@32)

set -e
cd "$(dirname "$0")/.."

CKPT="${1:-checkpoints/srl/step_100}"  # default checkpoint

echo "Evaluating SRL checkpoint: $CKPT"

# Greedy
python -m src.eval_aime24 \
  --model "$CKPT" \
  --model-path "$CKPT" \
  --backend hf \
  --mode greedy \
  --output-dir "./results/aime24_greedy_srl"

# Avg@32
python -m src.eval_aime24 \
  --model "$CKPT" \
  --model-path "$CKPT" \
  --backend hf \
  --mode avg32 \
  --output-dir "./results/aime24_avg32_srl"
