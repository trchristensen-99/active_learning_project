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
    acquisition_function: {type: random}
    
  - run_index: 2
    gpu: 2
    acquisition_function: {type: uncertainty}
```

**Note:** Seeds are automatically calculated from `run_index`, so don't specify them explicitly.

### 3. Round 0 Variations

Test different initialization strategies:

```yaml
base_config: configs/active_learning_random_random.json
experiments:
  # Random initialization
  - run_index: 1
    gpu: 1
    round0:
      proposal_strategy: {type: random, seqsize: 249}
      acquisition_function: {type: random}
      n_candidates: 100000
      n_acquire: 20000
  
  # Uncertainty-based initialization
  - run_index: 2
    gpu: 2
    round0:
      proposal_strategy: {type: random, seqsize: 249}
      acquisition_function: {type: uncertainty}
      n_candidates: 100000
      n_acquire: 20000
```

**Results will be saved to:**
- Index 1: `results/.../deepstarr_train/init_random_random/idx1/`
- Index 2: `results/.../deepstarr_train/init_random_uncertainty/idx2/`

This allows direct comparison of initialization strategies.

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
ls -la results/random_proposal/random_acquisition/*/deepstarr/dream_rnn_ensemble/*/
```

## Results Organization

Results are automatically organized by configuration:

```
results/
  {proposal_strategy}_proposal/
    {acquisition_strategy}_acquisition/
      {n_cand}cand_{n_acq}acq/
        {student_arch}/
          {oracle_arch}/
            {dataset}/
              {round0_init}/
                idx{run_index}/
                  round_000/
                  round_001/
                  ...
                  config.json
                  summary.json
```

**Directory Structure Explained:**
- `{proposal_strategy}_proposal`: How sequences are generated (e.g., `random_proposal`)
- `{acquisition_strategy}_acquisition`: How sequences are selected (e.g., `uncertainty_acquisition`)
- `{dataset}`: Source dataset name (e.g., `deepstarr_train`)
- `{round0_init}`: Round 0 initialization method:
  - `init_genomic`: Pretrained on provided genomic sequences
  - `init_random_random`: Pretrained on random proposal + random acquisition
  - `init_random_uncertainty`: Pretrained on random proposal + uncertainty acquisition

Example:
```
results/random_proposal/random_acquisition/100000cand_20000acq/deepstarr/dream_rnn_ensemble/deepstarr_train/
├── init_genomic/
│   ├── genomic/          # Trained on genomic validation set
│   │   ├── idx1/
│   │   ├── idx2/
│   │   └── idx3/
│   └── low_shift/        # Trained on low_shift validation set
│       ├── idx1/
│       └── idx2/
└── init_random_random/
    ├── genomic/
    │   ├── idx1/
    │   ├── idx2/
    │   └── idx3/
```

## Multi-Test Dataset Evaluation

All experiments automatically evaluate models on **three test datasets** with different distribution shifts:

### Test Dataset Types

1. **no_shift**: Original genomic sequences (~82k sequences)
   - In-distribution performance
   - Direct comparison to DeepSTARR paper results

2. **low_shift**: Genomic sequences with 5% per-position mutations (~82k sequences)
   - Minor distribution shift
   - Tests robustness to sequence variations

3. **high_shift_low_activity**: Random DNA sequences (~82k sequences)
   - Major distribution shift
   - Tests out-of-distribution behavior

### Automatic Generation

Test/validation datasets are automatically generated on first run:

- **Location**: `data/test_val_sets/deepstarr/`
- **Generation Time**: ~5-10 minutes (one-time, reused across all experiments)
- **Progress Logging**: Shows 10% increments during oracle labeling
- **Oracle Labeling**: Non-genomic datasets labeled by oracle ensemble

### Validation Dataset Configuration

Specify which validation set to use for training:

```yaml
experiments:
  - run_index: 1
    gpu: 1
    # Use genomic validation set (default)
    validation_dataset: genomic
  
  - run_index: 2
    gpu: 2
    # Use low_shift validation set
    validation_dataset: low_shift
```

Or in JSON config:
```json
{
  "validation_dataset": "genomic"
}
```

### Evaluation Results

Each round's `metrics.json` includes results for all test sets:

```json
{
  "evaluation": {
    "no_shift": {
      "dev_mse": 1.234,
      "hk_mse": 0.987,
      "total_mse": 1.111,
      "dev_correlation": 0.85,
      "hk_correlation": 0.82,
      "avg_correlation": 0.835,
      "n_test_samples": 82372
    },
    "low_shift": {
      "dev_mse": 1.456,
      "hk_mse": 1.123,
      "total_mse": 1.290,
      "dev_correlation": 0.78,
      "hk_correlation": 0.75,
      "avg_correlation": 0.765,
      "n_test_samples": 82372
    },
    "high_shift_low_activity": {
      "dev_mse": 2.345,
      "hk_mse": 2.123,
      "total_mse": 2.234,
      "dev_correlation": 0.45,
      "hk_correlation": 0.42,
      "avg_correlation": 0.435,
      "n_test_samples": 82372
    }
  }
}
```

### Comparing Performance Across Shifts

```bash
# Extract correlations for all test sets from a specific round
jq '.round_3.evaluation | to_entries | map({type: .key, correlation: .value.avg_correlation})' \
  results/.../idx1/summary.json

# Compare no_shift vs high_shift performance
jq '[.round_3.evaluation.no_shift.avg_correlation, .round_3.evaluation.high_shift_low_activity.avg_correlation]' \
  results/.../idx1/summary.json
```

### Memory Management

Large test sets (>80k sequences) are handled automatically:

- **Batched Prediction**: 512 sequences per batch
- **Prevents OOM**: Works on 16GB GPUs
- **No Configuration Needed**: Automatic batching

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

