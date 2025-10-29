#!/bin/bash

# Script to run active learning experiments in parallel on different GPUs
# This will run experiments idx 2-6 on GPUs 1-5

set -e

echo "Starting parallel active learning experiments..."
echo "GPU 0: Running idx 1 (already started)"
echo "GPU 1: Starting idx 2"
echo "GPU 2: Starting idx 3" 
echo "GPU 3: Starting idx 4"
echo "GPU 4: Starting idx 5"
echo "GPU 5: Starting idx 6"

# Function to run experiment on specific GPU
run_experiment() {
    local idx=$1
    local gpu=$2
    local log_file="test_idx${idx}_fixed.log"
    
    echo "Starting experiment idx $idx on GPU $gpu..."
    
    CUDA_VISIBLE_DEVICES=$gpu ./scripts/gpu_wrapper.sh python scripts/run_active_learning.py \
        --config configs/active_learning_genomic_init.json \
        --run-index $idx \
        2>&1 | tee $log_file &
    
    echo "Experiment idx $idx started on GPU $gpu (PID: $!)"
}

# Start experiments on different GPUs
run_experiment 2 1
run_experiment 3 2
run_experiment 4 3
run_experiment 5 4
run_experiment 6 5

echo ""
echo "All experiments started!"
echo "Monitor progress with:"
echo "  watch -n 10 'nvidia-smi'"
echo "  tail -f test_idx*_fixed.log"
echo ""
echo "To check specific experiment:"
echo "  tail -f test_idx2_fixed.log  # for idx 2"
echo "  tail -f test_idx3_fixed.log  # for idx 3"
echo "  etc."

# Wait for all background processes
wait
echo "All experiments completed!"

