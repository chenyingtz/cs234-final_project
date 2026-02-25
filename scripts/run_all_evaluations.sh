#!/bin/bash
# Comprehensive evaluation script wrapper
# Evaluates all models on all benchmarks using paper parameters

set -e
cd "$(dirname "$0")/.."

# Default values
MODELS="base sft srl srl_rlvr"
BENCHMARKS="amc23 aime24 aime25 minerva_math"
MODES="greedy avg1 avg32"
DEVICE="cuda"
BACKEND="hf"
MAX_GEN_TOKS=4096
LIMIT=""
CONFIG=""
RESULTS_DIR="results"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --models)
            MODELS="$2"
            shift 2
            ;;
        --benchmarks)
            BENCHMARKS="$2"
            shift 2
            ;;
        --modes)
            MODES="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --backend)
            BACKEND="$2"
            shift 2
            ;;
        --max-gen-toks)
            MAX_GEN_TOKS="$2"
            shift 2
            ;;
        --limit)
            LIMIT="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --results-dir)
            RESULTS_DIR="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --models MODEL [MODEL ...]     Models to evaluate (default: base sft srl srl_rlvr)"
            echo "  --benchmarks BENCH [BENCH ...]  Benchmarks (default: aime24)"
            echo "  --modes MODE [MODE ...]        Evaluation modes (default: greedy avg1 avg32)"
            echo "  --device DEVICE                Device: cuda, mps, cpu (default: cuda)"
            echo "  --backend BACKEND              Backend: hf, vllm (default: hf)"
            echo "  --max-gen-toks N               Max tokens (default: 4096)"
            echo "  --limit N                      Limit samples (for testing)"
            echo "  --config PATH                  Custom model config JSON"
            echo "  --results-dir DIR              Results directory (default: results)"
            echo ""
            echo "Examples:"
            echo "  $0 --models base srl --benchmarks aime24 --modes greedy"
            echo "  $0 --device mps --limit 10"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Build command
CMD="python -m src.eval_all_benchmarks"
CMD="$CMD --models $MODELS"
CMD="$CMD --benchmarks $BENCHMARKS"
CMD="$CMD --modes $MODES"
CMD="$CMD --device $DEVICE"
CMD="$CMD --backend $BACKEND"
CMD="$CMD --max-gen-toks $MAX_GEN_TOKS"
CMD="$CMD --results-dir $RESULTS_DIR"

if [ -n "$LIMIT" ]; then
    CMD="$CMD --limit $LIMIT"
fi

if [ -n "$CONFIG" ]; then
    CMD="$CMD --config $CONFIG"
fi

echo "========================================"
echo "Running Comprehensive Evaluation"
echo "========================================"
echo "Command: $CMD"
echo "========================================"
echo ""

# Run evaluation
$CMD

# Find the most recent summary file
SUMMARY_FILE=$(ls -t ${RESULTS_DIR}/evaluation_summary_*.json 2>/dev/null | head -1)

if [ -n "$SUMMARY_FILE" ]; then
    echo ""
    echo "========================================"
    echo "Aggregating Results"
    echo "========================================"
    python -m src.aggregate_results "$SUMMARY_FILE" --format all
    echo ""
    echo "Results summary: $SUMMARY_FILE"
else
    echo ""
    echo "Warning: No summary file found. Results may not have been saved."
fi
