#!/bin/bash
# Check if training pipeline is running, and if not, start it with logging

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PIPELINE_SCRIPT="$SCRIPT_DIR/run_training_pipeline.sh"

# Command to check for (what we're looking for)
CHECK_CMD="run_training_pipeline.sh --skip-sft --skip-srl"

# Command to execute if not running
EXECUTE_CMD="run_training_pipeline.sh --skip-sft --skip-rlvr"

# Log file with timestamp
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/pipeline_${TIMESTAMP}.log"

# Function to check if the process is running
is_running() {
    # Check for the exact command pattern
    # Using pgrep to find processes, then checking the full command line
    local found=0
    
    # Get all bash processes that might be running the pipeline
    while IFS= read -r pid; do
        if [ -n "$pid" ]; then
            # Check the full command line for the process
            local cmdline=$(ps -p "$pid" -o args= 2>/dev/null || echo "")
            if [[ "$cmdline" == *"run_training_pipeline.sh"* ]] && \
               [[ "$cmdline" == *"--skip-sft"* ]] && \
               [[ "$cmdline" == *"--skip-srl"* ]]; then
                found=1
                break
            fi
        fi
    done < <(pgrep -f "run_training_pipeline.sh" 2>/dev/null || true)
    
    return $found
}

# Check interval in seconds (default: 60 seconds)
CHECK_INTERVAL=${CHECK_INTERVAL:-60}

# Main logic
echo "========================================"
echo "Pipeline Monitor Script"
echo "========================================"
echo "Checking for: $CHECK_CMD"
echo "Will execute: $EXECUTE_CMD"
echo "Log file: $LOG_FILE"
echo "Check interval: ${CHECK_INTERVAL} seconds"
echo "========================================"
echo ""

# Use while loop to continuously check if pipeline is running
while true; do
    if is_running; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Pipeline is running (found process with --skip-sft --skip-srl)"
        echo "Waiting ${CHECK_INTERVAL} seconds before checking again..."
        sleep "$CHECK_INTERVAL"
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Pipeline is NOT running."
        echo "Starting pipeline: $EXECUTE_CMD"
        echo "Logging to: $LOG_FILE"
        echo ""
        
        # Change to project root directory
        cd "$PROJECT_ROOT"
        
        # Execute the command and log output
        # Use unbuffered output for real-time logging
        "$PIPELINE_SCRIPT" --skip-sft --skip-rlvr 2>&1 | tee "$LOG_FILE"
        
        EXIT_CODE=${PIPESTATUS[0]}
        
        echo ""
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Pipeline execution completed with exit code: $EXIT_CODE"
        echo "Log saved to: $LOG_FILE"
        
        # After execution, continue monitoring (loop back to check again)
        echo ""
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Resuming monitoring..."
        echo "Waiting ${CHECK_INTERVAL} seconds before next check..."
        sleep "$CHECK_INTERVAL"
    fi
done
