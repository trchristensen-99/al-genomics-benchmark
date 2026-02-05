#!/bin/bash
# Monitor baseline experiment jobs

cd "$(dirname "$0")/.."

echo "=========================================="
echo "BASELINE EXPERIMENT JOB MONITOR"
echo "=========================================="
echo ""

# Check if launcher is running
LAUNCHER_PID=$(ps aux | grep "[l]aunch_baseline_experiments.py" | awk '{print $2}' | head -1)
if [ -n "$LAUNCHER_PID" ]; then
    echo "✓ Launcher running (PID: $LAUNCHER_PID)"
else
    echo "✗ Launcher not running"
fi

# Count running jobs
RUNNING=$(ps aux | grep "[0]1_baseline_subsets.py" | wc -l)
echo "✓ Running jobs: $RUNNING"

# Count completed jobs (check for results files)
COMPLETED=$(find results/ -name "results.json" -newer logs/baseline_launcher.log 2>/dev/null | wc -l)
echo "✓ Completed jobs: $COMPLETED"

# Count log files
TOTAL_LOGS=$(ls -1 logs/baseline_runs/*.log 2>/dev/null | wc -l)
echo "✓ Total job logs: $TOTAL_LOGS"

echo ""
echo "=========================================="
echo "GPU UTILIZATION"
echo "=========================================="
nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu --format=csv

echo ""
echo "=========================================="
echo "RECENT JOB ACTIVITY"
echo "=========================================="
echo "Last 10 log updates:"
ls -lt logs/baseline_runs/*.log 2>/dev/null | head -10 | awk '{print $NF}' | xargs -I {} basename {} | sed 's/^/  /'

echo ""
echo "=========================================="
echo "MONITORING COMMANDS"
echo "=========================================="
echo "Watch launcher log:  tail -f logs/baseline_launcher.log"
echo "Watch a job log:     tail -f logs/baseline_runs/<job_name>.log"
echo "Check all jobs:      ps aux | grep '01_baseline_subsets.py'"
echo "GPU usage live:      watch -n 5 nvidia-smi"
echo "Results summary:     python scripts/summarize_results.py"
