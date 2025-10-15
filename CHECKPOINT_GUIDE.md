# Active Learning Checkpoint and Reproducibility Guide

## Overview

The active learning framework now supports automatic checkpointing and resumption, making it easy to:
- Run reproducible experiments with deterministic seeding
- Resume interrupted training from the last completed round
- Organize results in a hierarchical directory structure
- Run multiple configurations in parallel

## Key Features

### 1. Configurable Round 0 (Pretraining)

Round 0 can be configured to either:
- **Use provided initial data** (e.g., genomic DNA sequences from a dataset)
- **Generate and acquire sequences** using proposal and acquisition strategies

This allows testing different initialization strategies, including:
- Random DNA sequences
- Sequences from specific genomic regions
- Sequences selected by uncertainty or diversity
- Any custom proposal/acquisition combination

#### Configuration Example

Using provided initial data:
```json
{
  "data": {
    "initial_data_path": "data/processed/train.txt",
    "n_initial": 20000,
    "dataset_name": "train"
  }
}
```

Generating sequences for round 0:
```json
{
  "data": {
    "dataset_name": "random_random"
  },
  "round0": {
    "proposal_strategy": {
      "type": "random",
      "seqsize": 249,
      "seed": 42
    },
    "acquisition_function": {
      "type": "random",
      "seed": 42
    },
    "n_candidates": 100000,
    "n_acquire": 20000
  }
}
```

### 2. Deterministic Seeding

Seeds are calculated from a run index:
```
seed = 42 + run_index * 1000
```

Examples:
- `run_index=0` → `seed=42`
- `run_index=1` → `seed=1042`
- `run_index=2` → `seed=2042`

### 2. Hierarchical Directory Structure

Results are organized by configuration parameters:

```
results/
  {proposal_strategy}/
    {acquisition_strategy}/
      {n_candidates}cand_{n_acquire}acq/
        {student_arch}/
          {oracle_arch}/
            {dataset}/
              idx{run_index}/
                config.json              # Full configuration snapshot
                round_000/               # Baseline training on initial data
                  model_best.pth
                  metrics.json
                  training_data.json
                round_001/               # After cycle 1
                  model_best.pth
                  metrics.json
                  training_data.json
                round_002/               # After cycle 2
                  ...
                summary.json             # Overall results
```

Example path:
```
results/random/uncertainty/100000cand_20000acq/deepstarr/dream_rnn_ensemble/train/idx0/
```

### 3. Automatic Checkpointing

After each round (including round 0 baseline), the system saves:
- **Model weights**: `model_best.pth`
- **Metrics**: Training and evaluation results
- **Training data**: All sequences and labels acquired so far

### 4. Automatic Resumption

When you run the script:
1. It checks for existing checkpoints in the output directory
2. If found, it loads the last completed round
3. Training continues from the next round
4. If all rounds are complete, it skips training entirely

## Usage

### Basic Usage

Run with default configuration (run_index=0):
```bash
python scripts/run_active_learning.py --config configs/active_learning_full_config_adaptive.json
```

### Specify Run Index

Run with a specific index for different seed:
```bash
python scripts/run_active_learning.py \
  --config configs/active_learning_full_config_adaptive.json \
  --run-index 1
```

### Multiple Runs in Parallel

Run multiple configurations simultaneously:
```bash
# Terminal 1
python scripts/run_active_learning.py --config config.json --run-index 0

# Terminal 2
python scripts/run_active_learning.py --config config.json --run-index 1

# Terminal 3
python scripts/run_active_learning.py --config config.json --run-index 2
```

Each will use a different seed and save to a different directory.

### Resume Interrupted Training

If training is interrupted, simply rerun the same command:
```bash
python scripts/run_active_learning.py \
  --config configs/active_learning_full_config_adaptive.json \
  --run-index 0
```

The script will automatically:
1. Detect the last completed round
2. Load the training data from that checkpoint
3. Continue from the next round

### Override Output Directory

To use a custom output directory instead of the hierarchical structure:
```bash
python scripts/run_active_learning.py \
  --config config.json \
  --output_dir custom/output/path
```

## Configuration File Requirements

Your configuration file must include these fields for proper organization:

```json
{
  "run_index": 0,
  "data": {
    "dataset_name": "train"
  },
  "oracle": {
    "architecture": "dream_rnn_ensemble"
  },
  "trainer": {
    "architecture": "deepstarr"
  }
}
```

## Round Numbering

- **Round 0**: Baseline training on initial data only
- **Round 1-N**: Active learning cycles 1-N

Example for 5 cycles:
- Round 0: Train on 20k initial sequences
- Round 1: After acquiring first 20k sequences (40k total)
- Round 2: After acquiring second 20k sequences (60k total)
- Round 3: After acquiring third 20k sequences (80k total)
- Round 4: After acquiring fourth 20k sequences (100k total)
- Round 5: After acquiring fifth 20k sequences (120k total)

## Checkpoint Contents

Each round checkpoint contains:

### metrics.json
```json
{
  "round": 0,
  "cycle": 0,
  "n_training_sequences": 20000,
  "training_results": {
    "model_path": "...",
    "best_val_loss": 2.5,
    "best_epoch": 15
  },
  "evaluation_results": {
    "avg_correlation": 0.85,
    "total_mse": 1.2
  }
}
```

### training_data.json
```json
{
  "sequences": ["ATCG...", "GCTA...", ...],
  "labels": [[1.2, 0.8], [0.5, 1.1], ...],
  "n_sequences": 20000,
  "round": 0
}
```

### model_best.pth
PyTorch model state dict with trained weights.

## Examples

### Example 1: Run 5 replicates with different seeds

```bash
for i in {0..4}; do
  python scripts/run_active_learning.py \
    --config configs/active_learning_full_config_adaptive.json \
    --run-index $i &
done
```

This creates:
- `results/.../idx0/` (seed=42)
- `results/.../idx1/` (seed=1042)
- `results/.../idx2/` (seed=2042)
- `results/.../idx3/` (seed=3042)
- `results/.../idx4/` (seed=4042)

### Example 2: Resume after interruption

```bash
# Start training
python scripts/run_active_learning.py --config config.json --run-index 0

# ... training interrupted after round 2 ...

# Resume (will start from round 3)
python scripts/run_active_learning.py --config config.json --run-index 0
```

Output:
```
*** Resuming from round 2 ***
Loaded 60000 training sequences from checkpoint
Starting from round 3 (0=baseline, 1-5=AL cycles)
```

### Example 3: Check if training is complete

```bash
python scripts/run_active_learning.py --config config.json --run-index 0
```

If all rounds are complete:
```
*** All 5 cycles already completed! ***
Results are in: results/.../idx0/
```

## Troubleshooting

### Issue: Training always starts from round 0

**Cause**: Checkpoint files are missing or incomplete.

**Solution**: Check that each round directory has all three files:
```bash
ls results/.../idx0/round_000/
# Should show: model_best.pth, metrics.json, training_data.json
```

### Issue: Different runs use the same seed

**Cause**: Not specifying `--run-index` or using the same index.

**Solution**: Use different run indices:
```bash
python scripts/run_active_learning.py --config config.json --run-index 0
python scripts/run_active_learning.py --config config.json --run-index 1
```

### Issue: Can't find results

**Cause**: Results are in hierarchical directory structure.

**Solution**: Check the configuration printout at the start:
```
=== Configuration ===
Output directory: results/random/uncertainty/100000cand_20000acq/deepstarr/dream_rnn_ensemble/train/idx0
```

Or use `find`:
```bash
find results -name "idx0" -type d
```

## Benefits

1. **Reproducibility**: Same run_index always produces same results
2. **Efficiency**: No wasted computation re-running completed rounds
3. **Organization**: Clear structure makes comparing configurations easy
4. **Flexibility**: Easy to run multiple configurations in parallel
5. **Robustness**: Automatic recovery from interruptions
6. **Traceability**: Full configuration saved with each run

## Advanced Usage

### Custom Configuration Sweeps

Create a script to sweep over configurations:

```python
import subprocess
import json

# Base configuration
with open('base_config.json') as f:
    base_config = json.load(f)

# Sweep parameters
acquisition_strategies = ['random', 'uncertainty']
n_candidates_list = [50000, 100000, 200000]

run_index = 0
for acq_strategy in acquisition_strategies:
    for n_candidates in n_candidates_list:
        config = base_config.copy()
        config['acquisition_function']['type'] = acq_strategy
        config['active_learning']['n_candidates_per_cycle'] = n_candidates
        config['run_index'] = run_index
        
        # Save temporary config
        config_path = f'temp_config_{run_index}.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Run
        subprocess.run([
            'python', 'scripts/run_active_learning.py',
            '--config', config_path
        ])
        
        run_index += 1
```

This automatically organizes results by configuration in the hierarchical structure.

