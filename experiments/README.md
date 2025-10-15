# Experiment Configurations

This directory contains YAML configuration files for running batches of active learning experiments.

## Quick Start

### Run pre-configured experiments:
```bash
# Random/random replicates on GPUs 1, 2, 3
python scripts/run_experiments.py --config experiments/random_random_replicates.yaml --wait

# Mixed strategies comparison
python scripts/run_experiments.py --config experiments/mixed_strategies.yaml --wait
```

### Run custom experiments from command line:
```bash
# Auto-assign to available GPUs
python scripts/run_experiments.py \
    --base-config configs/active_learning_random_random.json \
    --run-indices 1 2 3 4 5 \
    --auto-gpu \
    --wait

# Specify exact GPUs
python scripts/run_experiments.py \
    --base-config configs/active_learning_random_random.json \
    --run-indices 1 2 3 \
    --gpus 1 2 3 \
    --wait
```

### Check GPU availability:
```bash
python scripts/run_experiments.py --show-gpus
```

## YAML Configuration Format

```yaml
# Base configuration file (required)
base_config: path/to/base_config.json

# List of experiments (required)
experiments:
  - run_index: 1              # Unique index (automatically determines seed: 42 + index * 1000)
    gpu: 1                    # GPU to use
    # Optional: override proposal strategy
    proposal_strategy:
      type: random
      seqsize: 249            # Sequence length (bp) - must match model expectations
    # Optional: override acquisition function
    acquisition_function:
      type: uncertainty
    # Optional: override round 0 configuration
    round0:
      proposal_strategy:
        type: random
        seqsize: 249
      acquisition_function:
        type: random
      n_candidates: 100000
      n_acquire: 20000
```

**Important Notes:**
- **Seeds are automatic**: Calculated as `seed = 42 + run_index * 1000`. Don't specify seeds in YAML.
- **Sequence size (`seqsize`)**: Must match your oracle and student model expectations
  - DeepSTARR: 249 bp (default)
  - Custom datasets: Specify the appropriate length
  - All models in the pipeline must support the same sequence length

## Available Strategies

### Proposal Strategies
- `random`: Generate random DNA sequences
  - Parameters: `seqsize` (sequence length in bp)
- `mixed`: Combine multiple proposal strategies

### Acquisition Functions
- `random`: Random selection from candidates
- `uncertainty`: Select sequences with highest uncertainty

## Sequence Size Configuration

The `seqsize` parameter determines the length of DNA sequences (in base pairs) used throughout the pipeline.

### Important Considerations

1. **Model Compatibility**: All models must support the same sequence length
   - Oracle model (DREAM-RNN ensemble)
   - Student model (DeepSTARR)
   - Proposal strategies

2. **Default Size**: DeepSTARR uses 249 bp sequences

3. **Custom Sizes**: To use different sequence lengths:
   - Train oracle models with the desired sequence length
   - Configure student model architecture for that length
   - Set `seqsize` in proposal strategies

### Example: Different Sequence Sizes

See `experiments/different_seqsize_example.yaml` for a complete example.

```yaml
experiments:
  # Standard 249 bp
  - run_index: 1
    gpu: 1
    proposal_strategy:
      type: random
      seqsize: 249
  
  # Longer 500 bp sequences
  - run_index: 2
    gpu: 2
    proposal_strategy:
      type: random
      seqsize: 500
    trainer:
      seqsize: 500
    oracle:
      seqsize: 500
```

**Note**: Using non-standard sequence sizes requires oracle and student models trained for those lengths.

## Example Configurations

### 1. Simple Replicates

Run 3 replicates with identical configuration but different seeds:

```yaml
base_config: configs/active_learning_random_random.json
experiments:
  - {run_index: 1, gpu: 1}
  - {run_index: 2, gpu: 2}
  - {run_index: 3, gpu: 3}
```

Seeds will be: 1042, 2042, 3042

### 2. Strategy Comparison

Compare different acquisition strategies:

```yaml
base_config: configs/active_learning_random_random.json
experiments:
  - run_index: 1
    gpu: 1
    acquisition_function: {type: random, seed: 42}
    
  - run_index: 2
    gpu: 2
    acquisition_function: {type: uncertainty, seed: 42}
```

### 3. Round 0 Variations

Test different initialization strategies:

```yaml
base_config: configs/active_learning_random_random.json
experiments:
  # Random initialization
  - run_index: 1
    gpu: 1
    round0:
      proposal_strategy: {type: random, seqsize: 249, seed: 42}
      acquisition_function: {type: random, seed: 42}
      n_candidates: 100000
      n_acquire: 20000
  
  # Uncertainty-based initialization
  - run_index: 2
    gpu: 2
    round0:
      proposal_strategy: {type: random, seqsize: 249, seed: 42}
      acquisition_function: {type: uncertainty, seed: 42}
      n_candidates: 100000
      n_acquire: 20000
```

## Command-Line Options

### Input Methods

**From YAML file:**
```bash
python scripts/run_experiments.py --config experiments/my_experiments.yaml
```

**From command line:**
```bash
python scripts/run_experiments.py \
    --base-config configs/base.json \
    --run-indices 1 2 3 \
    --gpus 1 2 3
```

**Auto-assign GPUs:**
```bash
python scripts/run_experiments.py \
    --base-config configs/base.json \
    --run-indices 1 2 3 4 5 \
    --auto-gpu
```

### Options

- `--wait`: Wait for all experiments to complete before exiting
- `--log-dir DIR`: Specify log directory (default: `logs`)
- `--no-gpu-wrapper`: Disable CUDA compatibility wrapper
- `--show-gpus`: Show GPU status and exit

## Monitoring

### View all experiment logs:
```bash
tail -f logs/experiment_*.log
```

### View specific experiment:
```bash
tail -f logs/experiment_idx1_gpu1_*.log
```

### Monitor GPU usage:
```bash
watch -n 1 nvidia-smi
```

### Check experiment status:
```bash
# List running experiments
ps aux | grep run_active_learning

# Check results directories
ls -la results/random/random/*/deepstarr/dream_rnn_ensemble/*/
```

## Results Organization

Results are automatically organized by configuration:

```
results/
  {proposal_strategy}/
    {acquisition_strategy}/
      {n_cand}cand_{n_acq}acq/
        {student_arch}/
          {oracle_arch}/
            {dataset}/
              idx{run_index}/
                round_000/
                round_001/
                ...
                config.json
                summary.json
```

Example:
```
results/random/random/100000cand_20000acq/deepstarr/dream_rnn_ensemble/random_random/
├── idx1/
├── idx2/
└── idx3/
```

## Tips

### 1. GPU Selection

Check available GPUs before running:
```bash
python scripts/run_experiments.py --show-gpus
```

Use `--auto-gpu` to automatically use available GPUs:
```bash
python scripts/run_experiments.py \
    --base-config configs/base.json \
    --run-indices 1 2 3 4 5 6 7 8 \
    --auto-gpu
```

### 2. Batch Processing

If you have more experiments than GPUs, run in batches:

```bash
# Batch 1
python scripts/run_experiments.py \
    --base-config configs/base.json \
    --run-indices 1 2 3 \
    --gpus 1 2 3 \
    --wait

# Batch 2
python scripts/run_experiments.py \
    --base-config configs/base.json \
    --run-indices 4 5 6 \
    --gpus 1 2 3 \
    --wait
```

### 3. Background Execution

For long-running experiments, use `screen` or `tmux`:

```bash
screen -S experiments
python scripts/run_experiments.py --config experiments/large_batch.yaml --wait
# Detach with Ctrl+A, D
# Reattach with: screen -r experiments
```

### 4. Resumption

Experiments automatically resume from checkpoints if interrupted:

```bash
# If interrupted, simply rerun the same command
python scripts/run_experiments.py --config experiments/my_experiments.yaml
```

## Troubleshooting

### GPU out of memory

Reduce batch size in the base configuration:
```json
{
  "trainer": {
    "batch_size": 4096  // Reduce from 8192
  }
}
```

### Experiments not starting

Check GPU availability:
```bash
python scripts/run_experiments.py --show-gpus
nvidia-smi
```

### CUDA errors

The GPU wrapper is enabled by default. If you still have issues:
```bash
# Check CUDA setup
python scripts/setup_gpu.py

# Manually set LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

## Integration with DVC

The experiment runner works alongside DVC:

```bash
# Option 1: Use experiment runner directly (recommended for flexibility)
python scripts/run_experiments.py --config experiments/my_experiments.yaml

# Option 2: Use DVC stages (for full reproducibility tracking)
dvc repro active_learning_random_random_idx1
```

Both methods produce identical results and are fully reproducible.

