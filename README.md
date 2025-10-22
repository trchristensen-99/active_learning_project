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

### 3. Reproduce with DVC

```bash
# Complete pipeline from scratch
dvc repro download_data
dvc repro preprocess_deepstarr
dvc repro train_dream_rnn_ensemble
dvc repro generate_test_val_datasets_deepstarr

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
  init_prop_genomic_acq_random_20k/val_genomic/idx1/
```

**Key improvements in v2:**
- Dataset first (most fundamental parameter)
- Explicit ensemble sizes (`5dreamrnn`, not `dream_rnn_ensemble`)
- Strategy percentages (`100random_proposal`, not `random_proposal`)
- Detailed initialization (`init_prop_genomic_acq_random_20k`, not `init_genomic`)
- Clear validation dataset (`val_genomic`, not `genomic`)

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
      {"type": "dream_rnn", "count": 5, "model_dir": "models/dream_rnn_ensemble"}
    ]
  },
  "active_learning": {
    "n_cycles": 5,
    "n_candidates_per_cycle": 100000,
    "n_acquire_per_cycle": 20000
  }
}
```

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
