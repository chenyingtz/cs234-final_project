#!/bin/sh

# The specific process command and arguments to monitor
CHECK_CMD="run_training_pipeline.sh --skip-sft --skip-srl"
#CHECK_CMD="./run_training_pipeline.sh --skip-sft --skip-rlvr"

# The command to execute if the monitored process is missing
EXEC_CMD="./run_training_pipeline.sh --skip-sft --skip-rlvr"
#EXEC_CMD="run_training_pipeline.sh --skip-sft --skip-srl"

# Log file path
LOG_FILE="pipeline_monitor-0227-0124.log"

# Interval between checks (in seconds)
CHECK_INTERVAL=60

echo "$(date): Starting monitoring loop for '$CHECK_CMD'..." | tee -a "$LOG_FILE"

while true; do
    # pgrep -f matches the full command line. 
    # We exclude the monitoring script's own PID ($$) from the results.
    if pgrep -f "$CHECK_CMD" | grep -v "$$" > /dev/null; then
        # Process is running; do nothing and wait for next check
        echo "Process "$CHECK_CMD" is running; do nothing and wait for next check"
        : 
    else
        echo "$(date): '$CHECK_CMD' not found. Restarting with '$EXEC_CMD'..." >> "$LOG_FILE"
        
        # Execute in background, redirecting both stdout and stderr to the log
        # We use nohup to ensure it survives if the terminal closes
        $EXEC_CMD >> "$LOG_FILE" 2>&1 &
        
        echo "$(date): Process restarted in background." >> "$LOG_FILE"
    fi

    # Wait before checking again
    sleep "$CHECK_INTERVAL"
done

