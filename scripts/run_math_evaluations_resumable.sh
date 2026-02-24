#!/bin/bash
# Resumable evaluation script for spot instances
# Automatically resumes from checkpoint if interrupted
# Handles GCP spot instance terminations gracefully

set -e  # Exit on error, but we'll handle interruptions
cd "$(dirname "$0")/.."

# Configuration
RESULTS_DIR="${RESULTS_DIR:-results}"
CHECKPOINT_FILE="${CHECKPOINT_FILE:-$RESULTS_DIR/evaluation_checkpoint.json}"
MAX_RETRIES="${MAX_RETRIES:-10}"
RETRY_DELAY="${RETRY_DELAY:-60}"  # seconds

# Function to handle interruptions
cleanup() {
    echo ""
    echo "========================================"
    echo "Interruption detected! Saving checkpoint..."
    echo "========================================"
    # The Python script should have already saved checkpoint
    # But we'll make sure it's saved
    if [ -f "$CHECKPOINT_FILE" ]; then
        echo "Checkpoint saved: $CHECKPOINT_FILE"
    fi
    exit 130  # Exit with SIGINT code
}

# Set up signal handlers for spot instance termination
trap cleanup SIGINT SIGTERM

# Function to check if evaluation is complete
is_complete() {
    local summary_file="$1"
    if [ ! -f "$summary_file" ]; thenu
        return 1
    fi
    
    # Check if all evaluations are successful
    local total=$(python3 -c "
import json
import sys
try:
    with open('$summary_file', 'r') as f:
        data = json.load(f)
    if isinstance(data, list):
        results = data
    else:
        results = data.get('results', [])
    config = data.get('config', {}) if isinstance(data, dict) else {}
    models = config.get('models', [])
    benchmarks = config.get('benchmarks', [])
    modes = config.get('modes', [])
    expected = len(models) * len(benchmarks) * len(modes)
    completed = len([r for r in results if r.get('success', False)])
    print(f'{completed}/{expected}')
    sys.exit(0 if completed >= expected else 1)
except:
    sys.exit(1)
" 2>/dev/null)
    
    return $?
}

# Function to run evaluation with retry logic
run_evaluation() {
    local attempt=1
    
    while [ $attempt -le $MAX_RETRIES ]; do
        echo ""
        echo "========================================"
        echo "Evaluation Attempt $attempt/$MAX_RETRIES"
        echo "========================================"
        
        # Run evaluation with resume flag
        if python -m src.eval_all_benchmarks \
            --models base sft srl srl_rlvr \
            --benchmarks amc23 aime24 aime25 minerva_math \
            --modes greedy avg1 avg32 \
            --max-gen-toks 4096 \
            --config configs/models_config.json \
            --results-dir "$RESULTS_DIR" \
            --checkpoint-file "$CHECKPOINT_FILE" \
            --resume \
            --skip-existing; then
            
            echo ""
            echo "Evaluation completed successfully!"
            return 0
        else
            local exit_code=$?
            echo ""
            echo "Evaluation interrupted or failed (exit code: $exit_code)"
            
            if [ $attempt -lt $MAX_RETRIES ]; then
                echo "Waiting ${RETRY_DELAY}s before retry..."
                sleep $RETRY_DELAY
                attempt=$((attempt + 1))
            else
                echo "Max retries reached. Checkpoint saved for manual resume."
                return 1
            fi
        fi
    done
}

# Main execution
echo "========================================"
echo "Resumable Mathematical Reasoning Benchmarks Evaluation"
echo "========================================"
echo "Benchmarks: AMC23, AIME24, AIME25, Minerva Math"
echo "Models: Base, SFT, SRL, SRL→RLVR"
echo "Modes: greedy, avg1, avg32"
echo "Checkpoint: $CHECKPOINT_FILE"
echo "Results: $RESULTS_DIR"
echo "========================================"
echo ""

# Check if we're resuming
if [ -f "$CHECKPOINT_FILE" ]; then
    echo "Found existing checkpoint. Will resume from last saved state."
    echo "To start fresh, delete: $CHECKPOINT_FILE"
    echo ""
fi

# Step 1: Run evaluations with resume support
echo "Step 1: Running evaluations (with auto-resume)..."
run_evaluation

# Step 2: Find the most recent summary file
SUMMARY_FILE=$(ls -t "$RESULTS_DIR"/evaluation_summary_*.json 2>/dev/null | head -1)

if [ -z "$SUMMARY_FILE" ]; then
    echo ""
    echo "Warning: No evaluation summary file found"
    echo "Evaluation may have been interrupted. Check checkpoint: $CHECKPOINT_FILE"
    exit 1
fi

# Step 3: Verify completion
if is_complete "$SUMMARY_FILE"; then
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
    echo "Plots saved to: $RESULTS_DIR/plots/"
    echo "Checkpoint: $CHECKPOINT_FILE (can be deleted now)"
    echo "========================================"
else
    echo ""
    echo "========================================"
    echo "Evaluation Incomplete"
    echo "========================================"
    echo "Some evaluations may still be pending."
    echo "Re-run this script to resume:"
    echo "  bash scripts/run_math_evaluations_resumable.sh"
    echo ""
    echo "Checkpoint: $CHECKPOINT_FILE"
    echo "Summary: $SUMMARY_FILE"
    echo "========================================"
fi
