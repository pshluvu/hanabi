#!/bin/bash
# Auto-restart wrapper for population_training.py.
#
# The training crash we chased down turned out to be an intermittent,
# probabilistic memory-corruption bug (likely WSL2/virtualization-level,
# not anything in the training code itself) -- it doesn't happen every run,
# but it can happen on any given run. Rather than chase 100% reliability
# further, this script just restarts training automatically if it crashes,
# relying on population_training.py's checkpoint-resume support (added
# alongside this script) to pick back up close to where it left off instead
# of starting over from scratch each time.
#
# Usage:
#   chmod +x run_population_training.sh
#   nohup ./run_population_training.sh > population_training_supervisor.log 2>&1 &
#   tail -f population_training_log.txt

#export CUDA_VISIBLE_DEVICES=""
export PYTHONPATH=/home/percy/hanabi

MAX_RESTARTS=100
restart_count=0

while [ $restart_count -lt $MAX_RESTARTS ]; do
  echo "=== Starting population_training.py (attempt $((restart_count + 1))) ==="
  python -u population_training.py >> population_training_log.txt 2>&1
  exit_code=$?

  if [ $exit_code -eq 0 ]; then
    echo "=== Training finished successfully (exit code 0) ==="
    break
  fi

  restart_count=$((restart_count + 1))
  echo "=== Crashed with exit code $exit_code -- restarting in 5 seconds (attempt $restart_count/$MAX_RESTARTS) ===" \
    >> population_training_log.txt
  sleep 5
done

if [ $restart_count -ge $MAX_RESTARTS ]; then
  echo "=== Gave up after $MAX_RESTARTS restarts -- check population_training_log.txt ===" \
    >> population_training_log.txt
fi
