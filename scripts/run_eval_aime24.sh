#!/bin/bash
# Wrapper to run AIME24 evaluation. Uses Python module.

set -e
cd "$(dirname "$0")/.."

# Default: greedy on base Qwen2.5-7B
python -m src.eval_aime24 \
  --model Qwen/Qwen2.5-7B-Instruct \
  --backend hf \
  --mode greedy \
  --output-dir ./results/aime24_greedy_base
