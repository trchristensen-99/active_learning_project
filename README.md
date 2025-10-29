# Active Learning for Genomics

A modular framework for active learning with genomic sequences, featuring automated experiment management, comprehensive evaluation, and reproducible results.

**📚 For detailed documentation, see [IMPLEMENTATION_DETAILS.md](IMPLEMENTATION_DETAILS.md)**

## Quick Start

### 1. Setup (One Command)

```bash
git clone <your-repo-url>
cd active_learning_project
make setup
```

This automatically installs dependencies, detects your GPU, and sets up the environment.

### 2. Run Experiments

```bash
# Run pre-configured experiments
python scripts/run_experiments.py \
    --config experiments/genomic_init_replicates.yaml \
    --wait

# Check GPU availability
python scripts/run_experiments.py --show-gpus
```

### 3. Train Oracle Ensemble

Train a DREAM-RNN ensemble to serve as the oracle:

```bash
# Standard training
dvc repro train_dream_rnn_ensemble

# Or manually with parallel training
python scripts/train_ensemble.py \
  --model_type dream_rnn \
  --train_data data/processed/train.txt \
  --val_data data/processed/val.txt \
  --test_data data/processed/test.txt \
  --output_dir models/oracles/deepstarr/dream_rnn \
  --n_models 5 \
  --epochs 80 \
  --batch_size 1024 \
  --lr 0.005 \
  --parallel

# With EvoAug augmentations
python scripts/train_ensemble.py \
  --model_type dream_rnn \
  --train_data data/processed/train.txt \
  --val_data data/processed/val.txt \
  --test_data data/processed/test.txt \
  --output_dir models/oracles/deepstarr/dream_rnn \
  --n_models 5 \
  --evoaug-config configs/evoaug_standard.json \
  --parallel
```

**Oracle Directory Structure:**
```
models/oracles/
  {dataset}/
    {architecture}/           # Standard models
      model_0/
      ...
    {architecture}/
      {evoaug_sig}/          # EvoAug-trained models
        model_0/
        ...
```

### 4. Generate Test/Validation Datasets

```bash
dvc repro generate_test_val_datasets_deepstarr
```

### 5. Run Active Learning Experiments

```bash
# Run specific experiments
dvc repro active_learning_genomic_init_20k_batch
dvc repro active_learning_genomic_init_10k_batch
```

## What This Framework Does

### Active Learning Pipeline

The framework implements an iterative active learning approach:

1. **Train** a student model on initial data
2. **Generate** candidate sequences using proposal strategies
3. **Select** informative sequences using acquisition functions
4. **Label** selected sequences with an oracle ensemble
5. **Retrain** the model with augmented data
6. **Repeat** for multiple cycles

### Key Features

- 🔄 **Automatic Checkpointing**: Resume interrupted experiments seamlessly
- 📊 **Multi-Test Evaluation**: Evaluate on multiple distribution shifts
- 🎯 **Reproducible**: Deterministic seeding (`seed = 42 + run_index * 1000`)
- ⚡ **GPU Optimized**: Automatic memory management and batch processing
- 🔧 **Modular Design**: Easy to extend with new strategies and datasets
- 📁 **Organized Results**: Hierarchical directory structure (v2)

## Results Organization (v2)

Results are automatically organized in a hierarchical structure:

```
results/
  {dataset}/                    # e.g., deepstarr, lentimpra
    {oracle_composition}/       # e.g., 5dreamrnn, 3deepstarr+5dreamrnn
      {student_composition}/    # e.g., 1deepstarr, 5deepstarr
        {proposal_strategy}/    # e.g., 100random_proposal
          {acquisition_strategy}/  # e.g., 100random_acquisition
            {pool_sizes}/       # e.g., 100000cand_20000acq
              {round0_init}/    # e.g., init_prop_genomic_acq_random_20k
                {validation}/   # e.g., val_genomic
                  idx{N}/       # e.g., idx1
                    config.json
                    metadata.json
                    round_000/
                    round_001/
                    ...
```

**Example:**
```
results/deepstarr/5dreamrnn/1deepstarr/100random_proposal/
  100random_acquisition/100000cand_20000acq/
  init_prop_genomic_acq_random_20k/train_scratch/val_genomic/ground_truth/idx1/

# Oracle-labeled experiments use:
results/deepstarr/5dreamrnn/1deepstarr/100random_proposal/
  100random_acquisition/100000cand_20000acq/
  init_prop_genomic_acq_random_20k/train_scratch/val_genomic/oracle_labels/idx1/
```

**Key improvements in v2:**
- Dataset first (most fundamental parameter)
- Explicit ensemble sizes (`5dreamrnn`, not `dream_rnn_ensemble`)
- Strategy percentages (`100random_proposal`, not `random_proposal`)
- Detailed initialization (`init_prop_genomic_acq_random_20k`, not `init_genomic`)
- Clear validation dataset (`val_genomic`, not `genomic`)
- Data source distinction (`ground_truth` vs `oracle_labels`)

## Configuration

Experiments are configured via JSON files:

```json
{
  "run_index": 1,
  "validation_dataset": "val_genomic",
  "data": {
    "dataset_name": "deepstarr",
    "initial_data_path": "data/processed/train.txt",
    "n_initial": 20000
  },
  "oracle": {
    "composition": [
      {"type": "dream_rnn", "count": 5, "model_dir": "models/oracles/deepstarr/dream_rnn"}
    ]
  },
  "active_learning": {
    "n_cycles": 5,
    "n_candidates_per_cycle": 100000,
    "n_acquire_per_cycle": 20000,
    "training_strategy": "from_scratch"
  }
}
```

### Fine-tuning Configuration

For continual learning experiments, use the `fine_tune` training strategy with detailed configuration:

```json
{
  "active_learning": {
    "training_strategy": "fine_tune",
    "finetune_config": {
      "learning_rate": {
        "type": "fixed",
        "value": 0.0001
      },
      "num_epochs": 50,
      "replay_strategy": {
        "type": "fixed_ratio",
        "old_new_ratio": 1.0
      },
      "optimizer": {
        "weight_decay": 1e-6
      },
      "early_stopping": {
        "enabled": true,
        "patience": 10
      }
    }
  }
}
```

**Replay Strategies:**
- `"all_data"`: Use all accumulated data (default behavior)
- `"fixed_ratio"`: Sample old data based on ratio to new data (e.g., 1.0 = equal old:new)
- `"percentage"`: Sample fixed % of old data (e.g., 20 = 20% of old data)

**Learning Rate Options:**
- `"fixed"`: Use constant learning rate for all epochs
- `"scheduled"`: Use learning rate scheduler (e.g., ReduceLROnPlateau)

**Directory Naming:**
Fine-tuning configurations are encoded in directory names:
- `train_scratch` - Train from scratch each cycle
- `finetune_lr1e4_50ep_alldata` - Fixed LR, all data replay
- `finetune_lr1e4_50ep_ratio1p0` - Fixed ratio replay (1:1 old:new)
- `finetune_lrsch_red_50ep_20pct` - Scheduled LR, 20% replay

## Common Commands

### Run Experiments

```bash
# Single experiment
python scripts/run_active_learning.py \
  --config configs/active_learning_genomic_init.json \
  --run-index 1

# Multiple replicates in parallel
python scripts/run_experiments.py \
  --config experiments/genomic_init_replicates.yaml \
  --wait
```

### Query Results

```bash
# Find all experiments for a dataset
find results/deepstarr -name "summary.json"

# Find experiments by oracle
find results/*/5dreamrnn -name "summary.json"

# Find experiments by strategy
find results -path "*/100uncertainty_acquisition/*" -name "summary.json"

# Find specific run index
find results -path "*/idx1/summary.json"
```

### Monitor Progress

```bash
# Watch experiment logs
tail -f logs/experiment_*.log

# Monitor GPU usage
watch -n 1 nvidia-smi
```

## Evaluation

Models are automatically evaluated on three test sets:

1. **No Shift**: Original genomic sequences (~82k)
2. **Low Shift**: 5% per-position mutations (~82k)
3. **High Shift**: Random DNA sequences (~82k)

This enables robust assessment of model generalization across distribution shifts.

## Results Analysis

The framework includes a comprehensive analysis system for comparing experiment configurations and generating publication-ready visualizations.

### Quick Analysis

```bash
# Compare acquisition sizes (10k vs 20k)
python scripts/analyze_results.py \
  --results-dir results/deepstarr \
  --compare-variable n_acquire_per_cycle \
  --compare-values 10000,20000 \
  --hold-constant oracle_composition=5dreamrnn,student_composition=1deepstarr

# Compare training strategies
python scripts/analyze_results.py \
  --results-dir results/deepstarr \
  --compare-variable training_mode \
  --compare-values train_scratch,finetune_lr1e4_50ep_ratio1p0 \
  --hold-constant n_acquire_per_cycle=20000
```

### Analysis Features

- **Flexible Comparisons**: Compare any variable while holding others constant
- **Statistical Rigor**: Proper aggregation with mean/std/sem/95% CI using t-distribution
- **Publication-Ready Plots**: Clean matplotlib plots with 95% confidence intervals
- **Data Export**: CSV files for custom analysis and summary tables
- **Reproducible**: Analysis configuration saved with outputs

### Output Structure

```
results_analysis/
  plots/
    n_acquire_per_cycle=5000_10000_20000/        # Compared variable + values
      all/                                         # Constants signature (or specific values)
        no_shift/
          avg_correlation/
            no_shift_avg_correlation.png
          total_mse/
            no_shift_total_mse.png
          dev_mse/
            no_shift_dev_mse.png
          ...
        low_shift/
          ...
        combined/
          avg_correlation/
            combined_avg_correlation.png
          ...
  data/
    n_acquire_per_cycle=5000_10000_20000/
      all/
        aggregated_metrics.csv   # Per-cycle statistics for all metrics
        summary_table.md        # Final performance summary
        raw_data.csv           # Complete raw data
  config.json                  # Analysis configuration
```

## EvoAug (Evolution-inspired Augmentations)

We support EvoAug-style training for the student (and optionally the oracle) with:
- Online augmentations during training (one-stage)
- Two-stage curriculum: pretrain with augmentations, then finetune on clean data

Enable via `trainer.evoaug` in config:
```
"evoaug": {
  "enabled": true,
  "stage": "both",  // pretrain|finetune|both
  "augmentations": {
    "mutation": {"enabled": true, "mutate_frac": 0.05},
    "translocation": {"enabled": true, "shift_min": 0, "shift_max": 20},
    "insertion": {"enabled": false},
    "deletion": {"enabled": true, "delete_min": 0, "delete_max": 30},
    "inversion": {"enabled": false},
    "reverse_complement": {"enabled": false},
    "noise": {"enabled": true, "noise_mean": 0, "noise_std": 0.3}
  },
  "max_augs_per_sequence": 2,
  "mode": "hard"
}
```

Directory naming reflects EvoAug usage:
- Oracle:
  - `5dreamrnn` (no EvoAug)
  - `5dreamrnn+evoaug_mut_trans_del3` (with EvoAug)
- Student:
  - `train_scratch` (no EvoAug)
  - `train_scratch+evoaug_pretrain(mut_del_noise2_hard)+finetune(clean)` (two-stage)

## Oracle-labeled Genomic Datasets

For consistency across shifts, we provide `no_shift_oracle` datasets (genomic sequences labeled by the oracle):
- Generated by `scripts/generate_test_val_datasets.py`
- Output under `data/test_val_sets/deepstarr/no_shift_oracle/{test.txt,val.txt}`

To use for validation, set `validation_dataset` to `val_genomic_oracle` (or `no_shift_oracle`).

Reproducibility is managed with DVC:
- Stage: `generate_test_val_datasets_deepstarr`
- Outs include `no_shift`, `no_shift_oracle`, `low_shift`, `high_shift_low_activity`

## Available Experiments

### Genomic Initialization

```bash
# 20k acquisition per cycle
dvc repro active_learning_genomic_init_20k_batch  # Runs idx 1,2,3

# 10k acquisition per cycle
dvc repro active_learning_genomic_init_10k_batch  # Runs idx 4,5,6
```

### Random Initialization

```bash
# Random proposal + random acquisition
dvc repro active_learning_random_random_idx1
dvc repro active_learning_random_random_idx2
dvc repro active_learning_random_random_idx3
```

## Documentation

- **[IMPLEMENTATION_DETAILS.md](IMPLEMENTATION_DETAILS.md)** - Comprehensive technical guide
  - Complete setup instructions
  - Framework architecture and components
  - Configuration reference
  - Directory structure details
  - Checkpointing and reproducibility
  - Multi-test evaluation system
  - Troubleshooting and optimization
  - API reference

- **[experiments/README.md](experiments/README.md)** - Experiment configuration guide

## Dataset

Uses the [DeepSTARR dataset](https://doi.org/10.5281/zenodo.5502060) from Zenodo:
- DNA sequences from Drosophila enhancers
- Developmental and housekeeping activity measurements
- Training, validation, and test sets

**Citation**: de Almeida, Bernardo P. (2021). DeepSTARR manuscript data. Zenodo. https://doi.org/10.5281/zenodo.5502060

## Key Concepts

### Deterministic Seeding
```python
seed = 42 + run_index * 1000
```
- `idx0` → seed 42
- `idx1` → seed 1042
- `idx2` → seed 2042

### Oracle Composition
Explicit tracking of ensemble size and types:
- `5dreamrnn` - 5 DREAM-RNN models
- `3deepstarr+5dreamrnn` - Mixed ensemble (alphabetical)

### Strategy Percentages
Explicit percentages for mixed strategies:
- `100random_proposal` - 100% random
- `50mixed_50random_proposal` - 50% mixed, 50% random (alphabetical)

### Round 0 Initialization
Detailed initialization information:
- `init_prop_genomic_acq_random_20k` - 20k genomic, random selection
- `init_prop_random_acq_uncertainty_10k` - 10k random, uncertainty selection

## System Requirements

- **GPU**: NVIDIA GPU with 16GB+ VRAM recommended
- **CUDA**: Compatible version (automatically detected)
- **Python**: 3.8+ (3.10+ recommended)
- **Memory**: 32GB+ RAM recommended
- **Storage**: 50GB+ free space

## Troubleshooting

### CUDA Out of Memory
Reduce batch sizes in configuration:
```json
{
  "oracle": {"batch_size": 8},
  "trainer": {"batch_size": 16}
}
```

### cuDNN Incompatibility
GPU wrapper is used automatically:
```bash
./scripts/gpu_wrapper.sh python your_script.py
```

### Import Errors
Ensure you're in the project root and environment is activated:
```bash
cd /path/to/active_learning_project
conda activate active-learning-genomics  # or source .venv/bin/activate
```

## License

This project is licensed under the MIT License.

The DeepSTARR dataset is licensed under [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).
