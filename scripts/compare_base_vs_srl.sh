#!/bin/bash
# Compare base Qwen vs SRL fine-tuned model on AIME24.
# Usage: ./scripts/compare_base_vs_srl.sh [path-to-srl-checkpoint]
# Example: ./scripts/compare_base_vs_srl.sh checkpoints/srl/step_500

set -e
cd "$(dirname "$0")/.."

CKPT="${1:-checkpoints/srl/step_500}"
TIMESTAMP=$(date +%Y-%m-%dT%H-%M-%S)
BASE_DIR="./results/compare_${TIMESTAMP}"
mkdir -p "$BASE_DIR"

echo "========================================"
echo "AIME24: Base Qwen vs SRL ($CKPT)"
echo "========================================"

# --- Base Qwen (no SRL) ---
echo ""
echo "[1/4] Base Qwen - greedy..."
python -m src.eval_aime24 \
  --model Qwen/Qwen2.5-7B-Instruct \
  --backend hf \
  --mode greedy \
  --output-dir "$BASE_DIR/base_greedy"

echo ""
echo "[2/4] Base Qwen - Avg@32..."
python -m src.eval_aime24 \
  --model Qwen/Qwen2.5-7B-Instruct \
  --backend hf \
  --mode avg32 \
  --output-dir "$BASE_DIR/base_avg32"

# --- SRL fine-tuned ---
echo ""
echo "[3/4] SRL checkpoint - greedy..."
python -m src.eval_aime24 \
  --model "$CKPT" \
  --model-path "$CKPT" \
  --backend hf \
  --mode greedy \
  --output-dir "$BASE_DIR/srl_greedy"

echo ""
echo "[4/4] SRL checkpoint - Avg@32..."
python -m src.eval_aime24 \
  --model "$CKPT" \
  --model-path "$CKPT" \
  --backend hf \
  --mode avg32 \
  --output-dir "$BASE_DIR/srl_avg32"

# --- Summary ---
echo ""
echo "========================================"
echo "Results saved to: $BASE_DIR"
echo ""
echo "Check these folders for lm_eval output:"
echo "  Base (no SRL):  $BASE_DIR/base_greedy  $BASE_DIR/base_avg32"
echo "  SRL fine-tuned: $BASE_DIR/srl_greedy   $BASE_DIR/srl_avg32"
echo "========================================"
