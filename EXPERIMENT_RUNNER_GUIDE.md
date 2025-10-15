# Extensible Experiment Runner Guide

## Overview

The experiment runner system provides a flexible, user-friendly way to run multiple active learning experiments in parallel with automatic GPU management and full reproducibility.

## Key Features

✅ **Flexible Configuration**: YAML or command-line interface  
✅ **Automatic GPU Management**: Auto-detect and assign available GPUs  
✅ **CUDA Compatibility**: Built-in GPU wrapper for cross-system compatibility  
✅ **Full Reproducibility**: Deterministic seeding and checkpoint support  
✅ **Parallel Execution**: Run multiple experiments simultaneously  
✅ **Easy Monitoring**: Centralized logging and status checking  

## Quick Start

### 1. Check GPU Availability

```bash
python scripts/run_experiments.py --show-gpus
```

### 2. Run Pre-configured Experiments

```bash
# Random/random replicates (3 runs on GPUs 1, 2, 3)
python scripts/run_experiments.py \
    --config experiments/random_random_replicates.yaml \
    --wait
```

### 3. Run Custom Experiments

```bash
# Auto-assign to available GPUs
python scripts/run_experiments.py \
    --base-config configs/active_learning_random_random.json \
    --run-indices 1 2 3 4 5 \
    --auto-gpu \
    --wait
```

## Usage Patterns

### Pattern 1: YAML Configuration (Recommended)

**Best for**: Complex experiment batches, reproducible research

Create a YAML file (`experiments/my_experiments.yaml`):

```yaml
base_config: configs/active_learning_random_random.json

experiments:
  - run_index: 1
    gpu: 1
    acquisition_function:
      type: random
      seed: 42
  
  - run_index: 2
    gpu: 2
    acquisition_function:
      type: uncertainty
      seed: 42
  
  - run_index: 3
    gpu: 3
    proposal_strategy:
      type: random
      seqsize: 249
      seed: 42
    acquisition_function:
      type: uncertainty
      seed: 42
```

Run it:
```bash
python scripts/run_experiments.py --config experiments/my_experiments.yaml --wait
```

### Pattern 2: Command-Line (Quick Testing)

**Best for**: Quick tests, simple replicates

```bash
# Specify exact GPUs
python scripts/run_experiments.py \
    --base-config configs/active_learning_random_random.json \
    --run-indices 1 2 3 \
    --gpus 1 2 3 \
    --wait

# Auto-assign GPUs
python scripts/run_experiments.py \
    --base-config configs/active_learning_random_random.json \
    --run-indices 1 2 3 4 5 6 7 8 \
    --auto-gpu \
    --wait
```

### Pattern 3: DVC Integration (Full Reproducibility)

**Best for**: Published research, long-term reproducibility

```bash
# Run via DVC stages
dvc repro active_learning_random_random_idx1
dvc repro active_learning_random_random_idx2
dvc repro active_learning_random_random_idx3
```

## Configuration Options

### Base Configuration

The base configuration file defines default settings:

```json
{
  "run_index": 0,
  "active_learning": {
    "n_cycles": 5,
    "n_candidates_per_cycle": 100000,
    "n_acquire_per_cycle": 20000,
    "training_strategy": "from_scratch"
  },
  "trainer": {
    "architecture": "deepstarr",
    "num_epochs": 100,
    "batch_size": 8192,
    "lr": 0.001
  },
  "proposal_strategy": {
    "type": "random",
    "seqsize": 249,
    "seed": 42
  },
  "acquisition_function": {
    "type": "random",
    "seed": 42
  },
  "round0": {
    "proposal_strategy": {"type": "random", "seqsize": 249, "seed": 42},
    "acquisition_function": {"type": "random", "seed": 42},
    "n_candidates": 100000,
    "n_acquire": 20000
  }
}
```

### YAML Overrides

Override any configuration parameter in the YAML file:

```yaml
base_config: configs/base.json

experiments:
  - run_index: 1
    gpu: 1
    # Override proposal strategy
    proposal_strategy:
      type: random
      seqsize: 249
      seed: 42
    
    # Override acquisition function
    acquisition_function:
      type: uncertainty
      seed: 42
    
    # Override round 0 configuration
    round0:
      n_candidates: 50000
      n_acquire: 10000
```

## GPU Management

### Auto-Detection

The system automatically detects available GPUs based on:
- Memory usage (must have >1GB free)
- Utilization (must be <10%)

```bash
python scripts/run_experiments.py \
    --base-config configs/base.json \
    --run-indices 1 2 3 4 5 \
    --auto-gpu
```

### Manual Assignment

Specify exact GPUs for each experiment:

```bash
python scripts/run_experiments.py \
    --base-config configs/base.json \
    --run-indices 1 2 3 \
    --gpus 1 2 3
```

### CUDA Compatibility

The GPU wrapper is automatically used to ensure CUDA/cuDNN compatibility:

```bash
# GPU wrapper enabled by default
python scripts/run_experiments.py --config experiments/my_experiments.yaml

# Disable if not needed
python scripts/run_experiments.py --config experiments/my_experiments.yaml --no-gpu-wrapper
```

## Monitoring and Debugging

### View GPU Status

```bash
# Show GPU info
python scripts/run_experiments.py --show-gpus

# Continuous monitoring
watch -n 1 nvidia-smi
```

### View Logs

```bash
# All experiments
tail -f logs/experiment_*.log

# Specific experiment
tail -f logs/experiment_idx1_gpu1_*.log

# Multiple experiments
tail -f logs/experiment_idx{1,2,3}_*.log
```

### Check Progress

```bash
# List running processes
ps aux | grep run_active_learning

# Check result directories
ls -la results/random/random/*/deepstarr/dream_rnn_ensemble/*/

# View checkpoints
ls -la results/random/random/*/deepstarr/dream_rnn_ensemble/*/idx1/round_*/
```

## Example Use Cases

### Use Case 1: Replicate Study

Run 5 replicates with identical configuration:

```yaml
base_config: configs/active_learning_random_random.json
experiments:
  - {run_index: 1, gpu: 1}
  - {run_index: 2, gpu: 2}
  - {run_index: 3, gpu: 3}
  - {run_index: 4, gpu: 4}
  - {run_index: 5, gpu: 5}
```

```bash
python scripts/run_experiments.py --config experiments/replicates.yaml --wait
```

### Use Case 2: Strategy Comparison

Compare different acquisition strategies:

```yaml
base_config: configs/base.json
experiments:
  - run_index: 1
    gpu: 1
    acquisition_function: {type: random, seed: 42}
  
  - run_index: 2
    gpu: 2
    acquisition_function: {type: uncertainty, seed: 42}
```

```bash
python scripts/run_experiments.py --config experiments/comparison.yaml --wait
```

### Use Case 3: Hyperparameter Sweep

Test different batch sizes:

```yaml
base_config: configs/base.json
experiments:
  - run_index: 1
    gpu: 1
    trainer: {batch_size: 4096}
  
  - run_index: 2
    gpu: 2
    trainer: {batch_size: 8192}
  
  - run_index: 3
    gpu: 3
    trainer: {batch_size: 16384}
```

### Use Case 4: Round 0 Initialization Study

Compare different initialization strategies:

```yaml
base_config: configs/base.json
experiments:
  # Genomic DNA initialization
  - run_index: 1
    gpu: 1
    data:
      initial_data_path: data/processed/train.txt
      n_initial: 20000
  
  # Random DNA initialization
  - run_index: 2
    gpu: 2
    round0:
      proposal_strategy: {type: random, seqsize: 249, seed: 42}
      acquisition_function: {type: random, seed: 42}
      n_candidates: 100000
      n_acquire: 20000
```

## Integration with Other Tools

### With DVC

```bash
# Option 1: Use experiment runner (more flexible)
python scripts/run_experiments.py --config experiments/my_experiments.yaml

# Option 2: Use DVC stages (tracked by version control)
dvc repro active_learning_random_random_idx1
```

### With Slurm/PBS

```bash
#!/bin/bash
#SBATCH --gres=gpu:3
#SBATCH --time=24:00:00

cd /path/to/project
python scripts/run_experiments.py \
    --config experiments/my_experiments.yaml \
    --wait
```

### With Screen/Tmux

```bash
screen -S experiments
python scripts/run_experiments.py --config experiments/large_batch.yaml --wait
# Detach: Ctrl+A, D
# Reattach: screen -r experiments
```

## Troubleshooting

### Problem: GPU out of memory

**Solution**: Reduce batch size in configuration
```json
{
  "trainer": {
    "batch_size": 4096  // Reduce from 8192
  }
}
```

### Problem: CUDA errors

**Solution**: Ensure GPU wrapper is enabled (default)
```bash
# Check CUDA setup
python scripts/setup_gpu.py

# Verify wrapper is working
./scripts/gpu_wrapper.sh python -c "import torch; print(torch.cuda.is_available())"
```

### Problem: Experiments not starting

**Solution**: Check GPU availability
```bash
python scripts/run_experiments.py --show-gpus
nvidia-smi
```

### Problem: Need to resume interrupted experiments

**Solution**: Simply rerun the same command - checkpoints will be detected automatically
```bash
python scripts/run_experiments.py --config experiments/my_experiments.yaml
```

## Best Practices

1. **Always check GPU availability first**
   ```bash
   python scripts/run_experiments.py --show-gpus
   ```

2. **Use YAML for complex experiments** - easier to version control and share

3. **Use `--wait` for important experiments** - ensures you're notified of failures

4. **Monitor logs during execution**
   ```bash
   tail -f logs/experiment_*.log
   ```

5. **Use descriptive run indices** - helps identify experiments later

6. **Version control your YAML files** - enables reproducibility

7. **Document your experiments** - add comments to YAML files

## For Package Users

If someone downloads this package, they can:

1. **Check GPU availability**:
   ```bash
   python scripts/run_experiments.py --show-gpus
   ```

2. **Run pre-configured experiments**:
   ```bash
   python scripts/run_experiments.py --config experiments/random_random_replicates.yaml --wait
   ```

3. **Create custom experiments**:
   - Copy an example YAML from `experiments/`
   - Modify as needed
   - Run with `python scripts/run_experiments.py --config my_experiments.yaml`

4. **Use auto-GPU for flexibility**:
   ```bash
   python scripts/run_experiments.py \
       --base-config configs/active_learning_random_random.json \
       --run-indices 1 2 3 \
       --auto-gpu \
       --wait
   ```

The system handles all GPU management, CUDA compatibility, and reproducibility automatically!

