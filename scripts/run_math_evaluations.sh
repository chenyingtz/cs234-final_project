#!/bin/bash
# Evaluate all models on all mathematical reasoning benchmarks
# Generates evaluation results and comparison graphs

set -e
cd "$(dirname "$0")/.."

echo "========================================"
echo "Mathematical Reasoning Benchmarks Evaluation"
echo "========================================"
echo "Benchmarks: AMC23, AIME24, AIME25, Minerva Math"
echo "Models: Base, SFT, SRL, SRL→RLVR"
echo "Modes: greedy, avg1, avg32"
echo "========================================"
echo ""

# Step 1: Run all evaluations
echo "Step 1: Running evaluations..."
#python -m src.eval_all_benchmarks \
#  --models base sft srl srl_rlvr \
#  --benchmarks amc23 aime24 aime25 minerva_math \
#  --modes greedy avg1 avg32 \
#  --max-gen-toks 4096 \
#  --config configs/models_config.json


DATE_TAG=${DATE_TAG:-$(date +%m%d)}

# Allow controlling models / benchmarks / modes via environment variables,
# with sensible defaults if not provided.
MODELS=${MODELS:-"srl"}
BENCHMARKS=${BENCHMARKS:-"aime24 aime25"}
MODES=${MODES:-"greedy avg1"}

for model in $MODELS; do
  for bench in $BENCHMARKS; do
    for mode in $MODES; do
      echo "Start run benchmark with model=${model}, benchmark=${bench}, mode=${mode}"
      RESULTS_DIR="results-${DATE_TAG}-${model}-${bench}_${mode}"
      CACHE_DIR="benchmark_cache_dir/persistent/lm_eval_cache_${bench}_${mode}"

      python -m src.eval_all_benchmarks \
        --models "${model}" \
        --benchmarks "${bench}" \
        --modes "${mode}" \
        --max-gen-toks 4096 \
        --results-dir "${RESULTS_DIR}" \
        --checkpoint-file "${RESULTS_DIR}/eval_checkpoint.json" \
        --cache-dir "${CACHE_DIR}" \
        --config configs/models_config.json
    done
  done
done


# test run
#python -m src.eval_all_benchmarks \
#  --models base \
#  --benchmarks aime24 aime25 \
#  --modes greedy \
#  --max-gen-toks 4096 \
#  --config configs/models_config.json \
#  --limit 1 \
#  --device mps

# Step 2: Find the most recent summary file
SUMMARY_FILE=$(ls -t results/evaluation_summary_*.json 2>/dev/null | head -1)

if [ -z "$SUMMARY_FILE" ]; then
    echo "Error: No evaluation summary file found"
    exit 1
fi

echo ""
echo "Step 2: Aggregating results..."
python -m src.aggregate_results "$SUMMARY_FILE" --format all

echo ""
echo "Step 3: Generating comparison graphs..."
python -m src.plot_comparison "$SUMMARY_FILE" --format all

echo ""
echo "========================================"
echo "Evaluation Complete!"
echo "========================================"
echo "Results summary: $SUMMARY_FILE"
echo "Plots saved to: results/plots/"
echo "========================================"