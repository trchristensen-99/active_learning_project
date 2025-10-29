# Active Learning for Genomics - Complete Implementation Guide

> **Quick Start**: For a high-level overview, see [README.md](README.md)

This is the comprehensive technical documentation for the active learning framework.

## Table of Contents

1. [Setup and Installation](#setup-and-installation)
2. [Framework Architecture](#framework-architecture)
3. [Configuration System](#configuration-system)
4. [Results Directory Structure](#results-directory-structure)
5. [Checkpointing and Reproducibility](#checkpointing-and-reproducibility)
6. [Running Experiments](#running-experiments)
7. [Multi-Test Evaluation System](#multi-test-evaluation-system)
8. [Results Analysis and Plotting](#results-analysis-and-plotting)
9. [Oracle and Student Ensembles](#oracle-and-student-ensembles)
10. [EvoAug Integration](#evoaug-integration)
11. [Oracle-labeled Genomic Datasets](#oracle-labeled-genomic-datasets)
12. [Extending the Framework](#extending-the-framework)
13. [Troubleshooting](#troubleshooting)
14. [API Reference](#api-reference)

---

## Setup and Installation

### One-Command Setup (Recommended)

```bash
git clone <your-repo-url>
cd active_learning_project
make setup
```

This automatically:
- Installs all dependencies
- Detects NVIDIA driver version
- Installs compatible PyTorch
- Tests GPU availability
- Creates GPU wrapper for cuDNN compatibility

### Manual Setup

#### Option 1: Conda

```bash
conda env create -f environment.yml
conda activate active-learning-genomics
python scripts/setup_gpu.py
```

#### Option 2: Pip + Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/setup_gpu.py
```

### GPU Compatibility

The `setup_gpu.py` script automatically handles GPU setup:

**Supported Driver Versions:**
- 525+: PyTorch 2.1.0 with CUDA 12.1
- 520-524: PyTorch 2.0.1 with CUDA 11.8
- 515-519: PyTorch 1.13.1 with CUDA 11.7
- 510-514: PyTorch 1.12.1 with CUDA 11.6
- 470-509: PyTorch 1.11.0 with CUDA 11.5
- 450-469: PyTorch 1.10.1 with CUDA 11.1

**cuDNN Compatibility:**
All training scripts automatically use `./scripts/gpu_wrapper.sh` to handle cuDNN version mismatches.

### Download Data and Train Oracle

```bash
# Complete pipeline
make full-pipeline

# Or step by step
make data      # Download and preprocess DeepSTARR data
make train     # Train 5-model DREAM-RNN ensemble
```

---

## Framework Architecture

### Overview

The framework implements iterative active learning following the PIONEER approach:

```
┌─────────────────────────────────────────────────────────────┐
│  Active Learning Cycle                                       │
│                                                              │
│  1. Student Model (DeepSTARR)                               │
│     └─> Trained on current dataset                          │
│                                                              │
│  2. Proposal Strategy (Random/Mixed/Guided)                 │
│     └─> Generate candidate sequences                        │
│                                                              │
│  3. Acquisition Function (Random/Uncertainty/Diversity)     │
│     └─> Select most informative sequences                   │
│                                                              │
│  4. Oracle Ensemble (5x DREAM-RNN)                          │
│     └─> Label selected sequences                            │
│                                                              │
│  5. Retraining                                              │
│     └─> Update student with augmented dataset               │
│                                                              │
│  Repeat for N cycles                                        │
└─────────────────────────────────────────────────────────────┘
```

### Core Components

#### 1. Oracle Ensemble (`code/active_learning/oracle.py`)

Provides high-quality labels for proposed sequences.

**Features:**
- Multi-model ensemble support (single type or mixed)
- Uncertainty quantification via ensemble variance
- Batched prediction for memory efficiency
- Composition tracking (e.g., `5dreamrnn`, `3deepstarr+5dreamrnn`)

**Example:**
```python
from code.active_learning import EnsembleOracle

# Single model type
oracle = EnsembleOracle(
    composition=[{
        "type": "dream_rnn",
        "count": 5,
        "model_dir": "models/dream_rnn_ensemble"
    }],
    device="auto"
)

# Mixed ensemble
oracle = EnsembleOracle(
    composition=[
        {"type": "deepstarr", "count": 3, "model_dir": "models/deepstarr_ensemble"},
        {"type": "dream_rnn", "count": 5, "model_dir": "models/dream_rnn_ensemble"}
    ]
)

# Predict and get uncertainty
predictions = oracle.predict(sequences)
uncertainties = oracle.get_uncertainty(sequences)
```

#### 2. Student Model (`code/active_learning/student.py`)

DeepSTARR CNN architecture trained via active learning.

**Features:**
- Exact implementation from DeepSTARR paper
- 249bp sequences with 4-channel one-hot encoding
- Two output tasks (Dev and Hk activity)
- Support for from-scratch training or fine-tuning

#### 3. Proposal Strategies (`code/active_learning/proposal.py`)

Generate candidate sequences for active learning cycles.

**Available Strategies:**
- **Random**: Completely random DNA sequences
- **PartialRandomMutagenesis**: Mutate existing sequences
- **Mixed**: Combine multiple strategies with percentages

**Naming Convention:**
- Single: `100random_proposal`
- Mixed: `50mixed_50random_proposal` (alphabetical)

#### 4. Acquisition Functions (`code/active_learning/acquisition.py`)

Select most informative sequences from candidates.

**Available Functions:**
- **Random**: Random selection
- **Uncertainty**: Select highest uncertainty (ensemble variance)
- **Diversity**: Maximize sequence diversity
- **LCMD**: Balance informativeness, diversity, representativeness

**Naming Convention:**
- Single: `100random_acquisition`
- Mixed: `30diversity_70uncertainty_acquisition` (alphabetical)

#### 5. Configuration Manager (`code/active_learning/config_manager.py`)

Manages hierarchical directory structure and seeding.

**Key Responsibilities:**
- Parse configuration parameters
- Generate hierarchical directory paths
- Calculate deterministic seeds: `seed = 42 + run_index * 1000`
- Track oracle/student compositions
- Format strategy names with percentages

#### 6. Checkpoint Manager (`code/active_learning/checkpoint.py`)

Handles automatic checkpointing and resumption.

**Saved Per Round:**
- `model_best.pth`: Model weights
- `metrics.json`: Training/evaluation metrics
- `training_data.json`: Sequences and labels

---

## Configuration System

### JSON Configuration Files

Experiments are configured via JSON files with the following structure:

```json
{
  "run_index": 1,
  "validation_dataset": "val_genomic",
  
  "active_learning": {
    "n_cycles": 5,
    "n_candidates_per_cycle": 100000,
    "n_acquire_per_cycle": 20000,
    "training_strategy": "from_scratch"
  },
  
  "data": {
    "dataset_name": "deepstarr",
    "initial_data_path": "data/processed/train.txt",
    "n_initial": 20000,
    "genomic_test_path": "data/processed/test.txt",
    "genomic_val_path": "data/processed/val.txt"
  },
  
  "oracle": {
    "composition": [
      {
        "type": "dream_rnn",
        "count": 5,
        "model_dir": "models/dream_rnn_ensemble"
      }
    ],
    "device": "auto",
    "batch_size": 1024
  },
  
  "trainer": {
    "architecture": "deepstarr",
    "num_epochs": 100,
    "batch_size": 8192,
    "lr": 0.001,
    "early_stopping": true,
    "early_stopping_patience": 10
  },
  
  "proposal_strategy": {
    "type": "random",
    "seqsize": 249
  },
  
  "acquisition_function": {
    "type": "random"
  }
}
```

### Configuration Fields Reference

#### `run_index` (required)
- Determines random seed: `seed = 42 + run_index * 1000`
- Creates separate directory: `idx{run_index}/`
- Use different indices for replicates

#### `validation_dataset`
- Which validation set to use for training
- Options: `"val_genomic"`, `"val_low_shift"`, `"val_high_shift_low_activity"`
- Mixed: `{"mix": {"no_shift": 34, "low_shift": 33, "high_shift_low_activity": 33}}`

#### `active_learning`
- `n_cycles`: Number of active learning cycles
- `n_candidates_per_cycle`: Candidates to generate per cycle
- `n_acquire_per_cycle`: Sequences to acquire per cycle
- `training_strategy`: `"from_scratch"` or `"fine_tune"`

#### `data`
- `dataset_name`: Dataset identifier (e.g., `"deepstarr"`, `"lentimpra"`)
- `initial_data_path`: Path to genomic sequences for round 0 (optional)
- `n_initial`: Number of initial sequences
- `genomic_test_path`: Path to genomic test set
- `genomic_val_path`: Path to genomic validation set

#### `oracle.composition`
- List of model types and counts
- Each entry: `{"type": str, "count": int, "model_dir": str}`
- Generates composition string (e.g., `5dreamrnn`, `3deepstarr+5dreamrnn`)

#### `trainer`
- `architecture`: Student model architecture
- `composition`: For student ensembles (optional)
- `hyperparameters`: Non-default hyperparameters (optional)
- Standard training parameters: epochs, batch_size, lr, etc.

#### `round0` (optional)
Generate sequences for round 0 instead of using genomic data:

```json
{
  "round0": {
    "proposal_strategy": {"type": "random", "seqsize": 249},
    "acquisition_function": {"type": "random"},
    "n_candidates": 100000,
    "n_acquire": 20000
  }
}
```

### YAML Batch Configuration

For running multiple experiments, use YAML configuration:

```yaml
base_config: configs/active_learning_genomic_init.json

experiments:
  - run_index: 1
    gpu: 1
  
  - run_index: 2
    gpu: 2
    active_learning:
      n_acquire_per_cycle: 10000
  
  - run_index: 3
    gpu: 3
    proposal_strategy:
      type: mixed
    acquisition_function:
      type: uncertainty
```

**Override Rules:**
- Nested fields use dot notation: `active_learning.n_acquire_per_cycle`
- Special fields extracted: `run_index`, `gpu`, `proposal_strategy`, `acquisition_function`, `round0`
- All other fields treated as overrides

---

## Results Directory Structure

### Hierarchy (v2)

Results are organized in a 10-level hierarchy:

```
results/
  {dataset}/                              # 1. Dataset name
    {oracle_composition}/                 # 2. Oracle ensemble
      {student_composition}/              # 3. Student model(s)
        {proposal_strategy}/              # 4. Proposal method
          {acquisition_strategy}/         # 5. Acquisition method
            {n_cand}cand_{n_acq}acq/      # 6. Pool sizes
              {round0_init}/              # 7. Initialization
                {training_mode}/           # 8. Training mode
                  {validation_dataset}/   # 9. Validation set
                    idx{run_index}/       # 10. Run index
                      config.json
                      metadata.json
                      summary.json
                      round_000/
                        model_best.pth
                        metrics.json
                        training_data.json
                      round_001/
                        ...
```

### Level Details

#### Level 1: Dataset (`{dataset}`)
**Format:** Lowercase, no underscores  
**Examples:** `deepstarr`, `lentimpra`, `synthetic`

**Configuration:**
```json
{"data": {"dataset_name": "deepstarr"}}
```

#### Level 2: Oracle Composition (`{oracle_composition}`)
**Format:** `{n}{modeltype}[+{n}{modeltype}]*` (lowercase, alphabetical)  
**Examples:**
- `5dreamrnn` - 5 DREAM-RNN models
- `3deepstarr+5dreamrnn` - Mixed (alphabetical order)

**Configuration:**
```json
{
  "oracle": {
    "composition": [
      {"type": "dream_rnn", "count": 5, "model_dir": "models/dream_rnn_ensemble"}
    ]
  }
}
```

#### Level 3: Student Composition (`{student_composition}`)
**Format:** `{n}{modeltype}[+{n}{modeltype}]*[_hyperparam_suffix]`  
**Examples:**
- `1deepstarr` - Single model
- `5deepstarr` - Ensemble of 5
- `1deepstarr_e200_lr0005` - With custom hyperparameters

**Hyperparameter Suffix:**
- Only for non-default configs
- Alphabetical: `{param1}{value1}_{param2}{value2}`
- Abbreviations: `e`=epochs, `bs`=batch_size, `lr`=learning_rate

#### Level 4: Proposal Strategy (`{proposal_strategy}`)
**Format:** `{pct1}{strategy1}[_{pct2}{strategy2}]*_proposal`  
**Examples:**
- `100random_proposal`
- `50mixed_50random_proposal` (alphabetical)

#### Level 5: Acquisition Strategy (`{acquisition_strategy}`)
**Format:** `{pct1}{strategy1}[_{pct2}{strategy2}]*_acquisition`  
**Examples:**
- `100random_acquisition`
- `30diversity_70uncertainty_acquisition` (alphabetical)

#### Level 6: Pool Sizes (`{n_candidates}cand_{n_acquire}acq`)
**Format:** `{n_candidates}cand_{n_acquire}acq`  
**Examples:** `100000cand_20000acq`, `50000cand_10000acq`

#### Level 7: Round 0 Initialization (`{round0_init}`)
**Format:** `init_prop_{prop}_acq_{acq}_{size}[_{sources}]`  
**Examples:**
- `init_prop_genomic_acq_random_20k`
- `init_prop_random_acq_uncertainty_10k`
- `init_prop_genomic_random_acq_random_genomic10k_random10k`

#### Level 8: Training Mode (`{training_mode}`)
**Format:** `train_{mode}` or `finetune_{params}`  
**Examples:**
- `train_scratch` - Train from scratch each cycle
- `finetune_lr1e4_50ep_alldata` - Fixed LR, all data replay
- `finetune_lr1e4_50ep_ratio1p0` - Fixed ratio replay (1:1 old:new)
- `finetune_lrsch_red_50ep_20pct` - Scheduled LR, 20% replay

**Configuration:**
```json
{
  "active_learning": {
    "training_strategy": "fine_tune",
    "finetune_config": {
      "learning_rate": {"type": "fixed", "value": 0.0001},
      "num_epochs": 50,
      "replay_strategy": {"type": "fixed_ratio", "old_new_ratio": 1.0},
      "optimizer": {"weight_decay": 1e-6},
      "early_stopping": {"enabled": true, "patience": 10}
    }
  }
}
```

**Parameter Encoding:**
- `lr{value}` - Fixed learning rate (e.g., `lr1e4` for 0.0001)
- `lrsch_{scheduler}` - Scheduled learning rate (e.g., `lrsch_red` for ReduceLROnPlateau)
- `{epochs}ep` - Number of epochs (e.g., `50ep`)
- `alldata` - Use all accumulated data
- `ratio{ratio}` - Fixed ratio replay (e.g., `ratio1p0` for 1.0)
- `{percentage}pct` - Percentage replay (e.g., `20pct` for 20%)
- `wd{value}` - Weight decay (if non-default, e.g., `wd1e5` for 1e-5)

#### Level 9: Validation Dataset (`{validation_dataset}`)
**Format:** `val_{name}` or `val_{pct1}{type1}_{pct2}{type2}*`  
**Examples:**
- `val_genomic`
- `val_33highshiftlowactivity_33lowshift_34noshift` (alphabetical)

#### Level 10: Run Index (`idx{run_index}`)
**Format:** `idx{N}`  
**Seed:** `42 + N * 1000`  
**Examples:** `idx0` (seed=42), `idx1` (seed=1042), `idx2` (seed=2042)

### Complete Example

```
results/deepstarr/5dreamrnn/1deepstarr/100random_proposal/
  100random_acquisition/100000cand_20000acq/
  init_prop_genomic_acq_random_20k/
    train_scratch/val_genomic/ground_truth/idx1/
    finetune_lr1e4_50ep_ratio1p0/val_genomic/ground_truth/idx2/

# Oracle-labeled experiments:
results/deepstarr/5dreamrnn/1deepstarr/100random_proposal/
  100random_acquisition/100000cand_20000acq/
  init_prop_genomic_acq_random_20k/
    train_scratch/val_genomic/oracle_labels/idx1/
```

**Interpretation:**
- Dataset: DeepSTARR
- Oracle: 5 DREAM-RNN models
- Student: Single DeepSTARR model
- Proposal: 100% random sequences
- Acquisition: 100% random selection
- Pool: 100k candidates → 20k acquired per cycle
- Round 0: 20k genomic sequences, randomly selected
- Training: `train_scratch` (from scratch) vs `finetune_lr1e4_50ep_ratio1p0` (fine-tune with 1:1 replay)
- Validation: Genomic validation set
- Run: Index 1 (seed = 1042) vs Index 2 (seed = 2042)

### Oracle-Labeled Experiments

Oracle-labeled experiments use labels generated by the oracle ensemble instead of ground truth experimental data. This allows for:

1. **Consistency**: All test sets (no_shift, low_shift, high_shift) use oracle labels
2. **Scalability**: Generate labels for large datasets without experimental costs
3. **Comparison**: Fair comparison between different oracle compositions

**Configuration:**
```json
{
  "active_learning": {
    "use_oracle_labels": true,
    "oracle_labeled_data_dir": "data/oracle_labels/deepstarr/5dreamrnn/no_shift"
  }
}
```

**Directory Structure:**
Oracle-labeled experiments are saved in separate directories with `oracle_labels` instead of `ground_truth`:

```
results/deepstarr/5dreamrnn/1deepstarr/100random_proposal/
  100random_acquisition/100000cand_20000acq/
  init_prop_genomic_acq_random_20k/
    train_scratch/val_genomic/oracle_labels/idx1/
```

This prevents conflicts with ground truth experiments and allows direct comparison.

**Data Generation:**
```bash
# Generate oracle labels for all datasets
python scripts/generate_oracle_labels.py \
  --config configs/oracle_labeled_20k.json \
  --output-root data/oracle_labels
```

### Querying Results

```bash
# All experiments for a dataset
find results/deepstarr -name "summary.json"

# All experiments with specific oracle
find results/*/5dreamrnn -name "summary.json"

# All uncertainty acquisition experiments
find results -path "*/100uncertainty_acquisition/*" -name "summary.json"

# All genomic initialization
find results -path "*/init_prop_genomic_acq_random_*/*" -name "summary.json"

# Specific run index
find results -path "*/idx1/summary.json"

# Compare configurations
diff results/deepstarr/5dreamrnn/1deepstarr/100random_proposal/100random_acquisition/100000cand_20000acq/init_prop_genomic_acq_random_20k/val_genomic/ground_truth/idx0/summary.json \
     results/deepstarr/5dreamrnn/1deepstarr/100random_proposal/100random_acquisition/100000cand_20000acq/init_prop_random_acq_random_20k/val_genomic/ground_truth/idx0/summary.json
```

---

## Checkpointing and Reproducibility

### Deterministic Seeding

All experiments use deterministic seeding:

```python
seed = 42 + run_index * 1000
```

**Examples:**
- `run_index=0` → `seed=42`
- `run_index=1` → `seed=1042`
- `run_index=2` → `seed=2042`

**Applied to:**
- NumPy random number generator
- PyTorch global seed
- CUDA seeds (if GPU available)
- Proposal strategies
- Acquisition functions

### Automatic Checkpointing

After each round (including round 0), the system saves:

**Per Round Directory (`round_XXX/`):**
- `model_best.pth`: Best model weights
- `metrics.json`: Training and evaluation metrics
- `training_data.json`: Sequences and labels used

**Root Run Directory (`idx{N}/`):**
- `config.json`: Complete configuration snapshot
- `metadata.json`: Git hash, timestamp, hardware info
- `summary.json`: Aggregate results across all rounds

### Automatic Resumption

When running an experiment:

1. Checks for existing checkpoints in output directory
2. Finds last completed round with all required files
3. Loads training data and continues from next round
4. If all rounds complete, skips training entirely

**Example Output:**
```
*** Resuming from round 2 ***
Loaded 60000 training sequences from checkpoint
Starting from round 3 (0=baseline, 1-5=AL cycles)
```

### Metadata Tracking

Each experiment saves `metadata.json`:

```json
{
  "timestamp": "2024-10-22T14:30:00",
  "hostname": "server01",
  "run_index": 1,
  "seed": 1042,
  "git_commit": "a1b2c3d4",
  "cuda_version": "12.1",
  "pytorch_version": "2.1.0",
  "gpu": {
    "count": 4,
    "devices": ["NVIDIA A100", "NVIDIA A100", ...]
  },
  "directory_structure": {
    "dataset": "deepstarr",
    "oracle_composition": "5dreamrnn",
    "student_composition": "1deepstarr",
    ...
  }
}
```

### Usage Examples

**Basic:**
```bash
python scripts/run_active_learning.py \
  --config configs/active_learning_genomic_init.json \
  --run-index 1
```

**Multiple Replicates:**
```bash
for i in {1..5}; do
  python scripts/run_active_learning.py \
    --config configs/active_learning_genomic_init.json \
    --run-index $i &
done
```

**Resume Interrupted:**
```bash
# Simply rerun the same command
python scripts/run_active_learning.py \
  --config configs/active_learning_genomic_init.json \
  --run-index 1
```

---

## Running Experiments

### Method 1: Single Experiment

```bash
python scripts/run_active_learning.py \
  --config configs/active_learning_genomic_init.json \
  --run-index 1
```

### Method 2: YAML Batch Configuration

**Create YAML file:**
```yaml
base_config: configs/active_learning_genomic_init.json

experiments:
  - run_index: 1
    gpu: 1
  - run_index: 2
    gpu: 2
  - run_index: 3
    gpu: 3
```

**Run:**
```bash
python scripts/run_experiments.py \
  --config experiments/genomic_init_replicates.yaml \
  --wait
```

**Options:**
- `--wait`: Wait for all experiments to complete
- `--no-wait`: Start experiments and return immediately (default)

### Method 3: DVC Pipeline

**Individual experiments:**
   ```bash
dvc repro active_learning_genomic_init_idx1
dvc repro active_learning_genomic_init_idx2
dvc repro active_learning_genomic_init_idx3
```

**Batch experiments:**
   ```bash
dvc repro active_learning_genomic_init_20k_batch  # Runs idx 1,2,3
dvc repro active_learning_genomic_init_10k_batch  # Runs idx 4,5,6
   ```

### GPU Management

**Check available GPUs:**
   ```bash
python scripts/run_experiments.py --show-gpus
   ```

**Auto-assign GPUs:**
   ```bash
python scripts/run_experiments.py \
  --base-config configs/active_learning_genomic_init.json \
  --run-indices 1 2 3 4 5 \
  --auto-gpu \
  --wait
```

**Specify GPUs:**
```bash
python scripts/run_experiments.py \
  --base-config configs/active_learning_genomic_init.json \
  --run-indices 1 2 3 \
  --gpus 1 2 3 \
  --wait
```

### Monitoring

**Watch logs:**
```bash
tail -f logs/experiment_idx1_gpu1_*.log
```

**Monitor GPU usage:**
```bash
watch -n 1 nvidia-smi
```

**Check status:**
```bash
# See running processes
ps aux | grep run_active_learning

# Check GPU processes
nvidia-smi
```

---

## Multi-Test Evaluation System

### Overview

Models are automatically evaluated on three test datasets with different distribution shifts:

1. **No Shift (Genomic)**: Original DeepSTARR test sequences (~82k)
2. **Low Shift**: 5% per-position mutations (~82k)
3. **High Shift Low Activity**: Random DNA sequences (~82k)

### Automatic Dataset Generation

Test/validation datasets are generated automatically on first run:

```bash
# Automatic (happens during first experiment)
python scripts/run_active_learning.py --config config.json --run-index 1

# Manual generation
python scripts/generate_test_val_datasets.py \
  --dataset deepstarr \
  --genomic-test data/processed/test.txt \
  --genomic-val data/processed/val.txt \
  --oracle-dir models/dream_rnn_ensemble \
  --output-dir data/test_val_sets \
  --seed 42
```

**Generated Files:**
```
data/test_val_sets/deepstarr/
  no_shift/
    test.txt          # Original genomic sequences
    val.txt
  no_shift_oracle/
    test.txt          # Genomic sequences labeled by oracle
    val.txt
  low_shift/
    test.txt          # 5% mutated sequences
    val.txt
  high_shift_low_activity/
    test.txt          # Random sequences
    val.txt
```

### Dataset Types

#### No Shift (Genomic)
- Source: Original DeepSTARR test/validation sequences
- No modifications
- Represents in-distribution performance

#### Low Shift
- Source: Genomic sequences
- Modification: 5% per-position mutation rate
- Represents minor distribution shift
- Oracle-labeled for ground truth

#### High Shift Low Activity
- Source: Completely random DNA sequences
- Represents major distribution shift (out-of-distribution)
- Oracle-labeled for ground truth

### Validation Dataset Selection

Configure which validation set to use for training:

```json
{
  "validation_dataset": "val_genomic"
}
```

**Options:**
- `"val_genomic"`: No shift validation
- `"val_genomic_oracle"` or `"no_shift_oracle"`: No-shift genomic labeled by oracle
- `"val_low_shift"`: Low shift validation
- `"val_high_shift_low_activity"`: High shift validation
- Mixed: `{"mix": {"no_shift": 0.33, "low_shift": 0.33, "high_shift_low_activity": 0.34}}`

**Important:** All models are evaluated on all test sets regardless of which validation set was used for training.

---

## Results Analysis and Plotting

The framework includes a comprehensive analysis system (`code/plotting/`) for comparing experiment configurations and generating publication-ready visualizations.

### Output Paths

- Plots are saved to hierarchical paths:
  - `results_analysis/plots/{compare_var=values}/{constants_sig}/{test_set}/{metric}/`
- CSV exports are organized by comparison:
  - `results_analysis/data/{compare_var=values}/{constants_sig}/`

### Architecture

The analysis system consists of four main components:

1. **`ResultsParser`** (`results_parser.py`): Parses the 11-level directory hierarchy and loads `all_cycle_results.json` files
2. **`MetricsAggregator`** (`results_aggregator.py`): Aggregates metrics across replicates with proper statistics
3. **`ResultsPlotter`** (`plotter.py`): Generates publication-ready matplotlib plots with 95% CI shaded regions
4. **`ResultsExporter`** (`exporter.py`): Exports CSV data and summary tables

### Usage

#### Basic Analysis

```bash
python scripts/analyze_results.py \
  --results-dir results/deepstarr \
  --compare-variable n_acquire_per_cycle \
  --compare-values 10000,20000 \
  --hold-constant oracle_composition=5dreamrnn,student_composition=1deepstarr
```

#### Advanced Options

   ```bash
python scripts/analyze_results.py \
  --results-dir results/deepstarr \
  --compare-variable training_mode \
  --compare-values train_scratch,finetune_lr1e4_50ep_ratio1p0 \
  --hold-constant n_acquire_per_cycle=20000 \
  --metrics avg_correlation,total_mse,dev_correlation,dev_mse,hk_mse \
  --test-sets no_shift,low_shift \
  --figure-size 10,8 \
  --font-size 14 \
  --plot-format pdf
```

### Command Line Arguments

#### Required Arguments
- `--results-dir`: Root results directory (e.g., `results/deepstarr`)
- `--compare-variable`: Variable to compare (e.g., `n_acquire_per_cycle`, `training_mode`)
- `--compare-values`: Values to compare (comma-separated, e.g., `10000,20000`)

#### Optional Arguments
- `--hold-constant`: Variables to hold constant (format: `var1=val1,var2=val2`)
- `--metrics`: Metrics to plot (comma-separated, default: `avg_correlation,total_mse,dev_mse,hk_mse,dev_correlation,hk_correlation,avg_mse`)
- `--test-sets`: Test sets to include (comma-separated, default: all available)
- `--output-dir`: Output directory for plots/CSVs (default: `results_analysis/`)
- `--plot-format`: Plot format (`png`, `pdf`, `svg`, default: `png`)
- `--figure-size`: Figure size as width,height (default: `8,6`)
- `--font-size`: Font size (default: `12`)
- `--no-plots`: Skip plot generation
- `--no-csv`: Skip CSV export

### Data Structure

#### Input Data
The system automatically parses the hierarchical directory structure:
```
results/{dataset}/{oracle_composition}/{student_composition}/
  {proposal_strategy}/{acquisition_strategy}/{pool_sizes}/
  {round0_init}/{training_mode}/{validation_dataset}/idx{N}/
    all_cycle_results.json
```

#### Output Structure
```
results_analysis/
  plots/
    {compare_var=values}/           # Compared variable + values
      {constants_sig}/              # Held-constant values or 'all'
        {test_set}/
          {metric}/
            {test_set}_{metric}.png # Individual test set plots
        combined/
          {metric}/
            combined_{metric}.png   # Combined test set plots
  data/
    {compare_var=values}/
      {constants_sig}/
        aggregated_metrics.csv      # Per-cycle statistics for all metrics
        summary_table.md           # Final performance summary
        raw_data.csv              # Complete raw data
  config.json                     # Analysis configuration
```

### Statistical Methods

#### Aggregation
- **Mean**: Arithmetic mean across replicates
- **Standard Deviation**: Sample standard deviation (ddof=1)
- **Standard Error**: SEM = std / sqrt(n)
- **95% Confidence Interval**: Using t-distribution with n-1 degrees of freedom
- **Min/Max**: Range across replicates

#### Plot Features
- **Line plots**: Mean values with markers
- **Confidence intervals**: 95% CI as shaded regions
- **Color scheme**: Colorblind-friendly palette
- **Grid**: Subtle grid for readability
- **Legends**: Clear variable labeling

### Available Variables

The system can compare any variable from the directory hierarchy:

#### Numeric Variables
- `n_candidates_per_cycle`: Candidate pool size
- `n_acquire_per_cycle`: Acquisition size
- `run_index`: Replicate number

#### Categorical Variables
- `dataset`: Dataset name (e.g., `deepstarr`)
- `oracle_composition`: Oracle ensemble (e.g., `5dreamrnn`)
- `student_composition`: Student model (e.g., `1deepstarr`)
- `proposal_strategy`: Proposal method (e.g., `100random_proposal`)
- `acquisition_strategy`: Acquisition method (e.g., `100random_acquisition`)
- `round0_init`: Initialization strategy
- `training_mode`: Training strategy (e.g., `train_scratch`, `finetune_lr1e4_50ep_ratio1p0`)
- `validation_dataset`: Validation set (e.g., `val_genomic`)

### Available Metrics

Common metrics available for analysis:
- `avg_correlation`: Average correlation across output heads
- `dev_correlation`: Developmental activity correlation
- `hk_correlation`: Housekeeping activity correlation
- `total_mse`: Total mean squared error
- `dev_mse`: Developmental activity MSE
- `hk_mse`: Housekeeping activity MSE
- `n_test_samples`: Number of test samples

### Example Analyses

#### 1. Acquisition Size Comparison
```bash
python scripts/analyze_results.py \
  --results-dir results/deepstarr \
  --compare-variable n_acquire_per_cycle \
  --compare-values 10000,20000 \
  --hold-constant oracle_composition=5dreamrnn,student_composition=1deepstarr \
  --output-dir analysis_output/acquisition_comparison
```

#### 2. Training Strategy Comparison
   ```bash
python scripts/analyze_results.py \
  --results-dir results/deepstarr \
  --compare-variable training_mode \
  --compare-values train_scratch,finetune_lr1e4_50ep_ratio1p0 \
  --hold-constant n_acquire_per_cycle=20000 \
  --output-dir analysis_output/training_comparison
```

#### 3. Oracle Composition Comparison
   ```bash
python scripts/analyze_results.py \
  --results-dir results/deepstarr \
  --compare-variable oracle_composition \
  --compare-values 5dreamrnn,3deepstarr+5dreamrnn \
  --hold-constant n_acquire_per_cycle=20000,training_mode=train_scratch \
  --output-dir analysis_output/oracle_comparison
```

### Customization

#### Plot Styling
The `ResultsPlotter` class supports customization:
```python
plotter = ResultsPlotter(
    figure_size=(10, 8),    # Width, height in inches
    font_size=14,           # Font size
    dpi=300                 # Resolution
)
```

#### Statistical Parameters
The `MetricsAggregator` supports different confidence levels:
```python
aggregator = MetricsAggregator(confidence_level=0.95)  # 95% CI
```

### Error Handling

The system gracefully handles:
- **Missing experiments**: Warns about missing values
- **Incomplete cycles**: Plots available data only
- **Missing metrics**: Skips unavailable metrics
- **Invalid configurations**: Clear error messages

### Performance

- **Automatic detection**: Finds all experiments without manual specification
- **Efficient parsing**: Loads only required data
- **Memory management**: Processes experiments in batches
- **Parallel plotting**: Generates multiple plots efficiently

---

## Oracle and Student Ensembles

### Oracle Ensemble Composition

**Single Type:**
```json
{
  "oracle": {
    "composition": [
      {
        "type": "dream_rnn",
        "count": 5,
        "model_dir": "models/dream_rnn_ensemble"
      }
    ]
  }
}
```
Result: `5dreamrnn`

**Mixed Ensemble:**
```json
{
  "oracle": {
    "composition": [
      {
        "type": "deepstarr",
        "count": 3,
        "model_dir": "models/deepstarr_ensemble"
      },
      {
        "type": "dream_rnn",
        "count": 5,
        "model_dir": "models/dream_rnn_ensemble"
      }
    ]
  }
}
```
Result: `3deepstarr+5dreamrnn` (alphabetical)

### Student Ensemble Composition

**Single Model (Default):**
```json
{
  "trainer": {
    "architecture": "deepstarr"
  }
}
```
Result: `1deepstarr`

**Ensemble:**
```json
{
  "trainer": {
    "composition": [
      {
        "type": "deepstarr",
        "count": 5
      }
    ]
  }
}
```
Result: `5deepstarr`

**With Custom Hyperparameters:**
```json
{
  "trainer": {
    "architecture": "deepstarr",
    "hyperparameters": {
      "num_epochs": 200,
      "lr": 0.0005
    }
  }
}
```
Result: `1deepstarr_e200_lr0005`

### Oracle Model Directory Structure

Oracle models are organized hierarchically for better organization and reproducibility:

```
models/
  oracles/
    {dataset}/
      {architecture}/
        model_0/
        model_1/
        ...
        training_config.json
        training_results.json
      {architecture}/
        {evoaug_signature}/
          model_0/
          model_1/
          ...
          training_config.json
          training_results.json
```

**Examples:**
- `models/oracles/deepstarr/dream_rnn/` - Standard 5-model DREAM-RNN ensemble
- `models/oracles/deepstarr/dream_rnn/evoaug_del0p05_mut0p1_trans0p1_2_hard/` - EvoAug-trained ensemble
- `models/oracles/lentimpra/deepstarr/` - DeepSTARR models trained on LentiMPRA data

### Training DREAM-RNN Oracle

**Standard Training:**
```bash
# Using DVC
dvc repro train_dream_rnn_ensemble

# Manual
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
```

**With EvoAug Augmentations:**
```bash
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
  --evoaug-config configs/evoaug_standard.json \
  --parallel
```

The EvoAug signature will be automatically appended as a subdirectory based on the configuration.

**Model Files:**
```
models/oracles/deepstarr/dream_rnn/
  model_0/model_best_MSE.pth
  model_1/model_best_MSE.pth
  ...
  model_4/model_best_MSE.pth
  training_config.json
  training_results.json
```

---

## EvoAug Integration

### Config Schema

```
"trainer": {
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
}
```

### Implementation Details

- `code/augmentations/evoaug.py` implements mutation, translocation (roll), insertion, deletion, inversion, reverse-complement (optional), and noise for one-hot sequences.
- `DeepSTARRTrainer` applies online augmentations in `_train_epoch` when `evoaug.enabled=true`.
- Two-stage training in `DeepSTARRActiveLearningTrainer`: if `evoaug.stage` includes `pretrain`, runs a pretrain phase with EvoAug, then finetunes on clean data by default (or with EvoAug if `stage=finetune`).

### Directory Naming

- Oracle: `5dreamrnn+evoaug_mut_trans_del3` when EvoAug is enabled for oracle training.
- Student: `train_scratch+evoaug_pretrain(mut_ins2_hard)+finetune(clean)` for two-stage.

---

## Oracle-labeled Genomic Datasets

- `scripts/generate_test_val_datasets.py` supports `no_shift_oracle`:
  - Same genomic sequences as `no_shift`, labeled with current oracle.
  - Outputs: `data/test_val_sets/{dataset}/no_shift_oracle/{test.txt,val.txt}`
- `scripts/run_active_learning.py` maps `validation_dataset` values `val_genomic_oracle` or `no_shift_oracle` to these files.
- DVC stage `generate_test_val_datasets_deepstarr` tracks explicit outs per dataset type for reproducibility.

---

## Extending the Framework

### Adding New Proposal Strategies

**1. Create Strategy Class** in `code/active_learning/proposal.py`:

```python
class MyProposalStrategy(BaseProposalStrategy):
    def __init__(self, seqsize: int = 249, seed: Optional[int] = None):
        self.seqsize = seqsize
        if seed:
            random.seed(seed)
    
    def propose_sequences(self, n_sequences: int, **kwargs) -> List[str]:
        # Your implementation here
        sequences = []
        # ... generate sequences ...
        return sequences
    
    def get_strategy_name(self, percentage: int = 100) -> str:
        return f"{percentage}mystrategy"
```

**2. Use in Configuration:**

```json
{
  "proposal_strategy": {
    "type": "myproposal",
    "seqsize": 249
  }
}
```

### Adding New Acquisition Functions

**1. Create Acquisition Class** in `code/active_learning/acquisition.py`:

```python
class MyAcquisition(BaseAcquisitionFunction):
    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
    
    def select_sequences(
        self,
        candidate_sequences: List[str],
        n_select: int,
        oracle_model: Optional[object] = None,
        **kwargs
    ) -> List[str]:
        # Your implementation here
        # ... select sequences ...
        return selected_sequences
    
    def get_strategy_name(self, percentage: int = 100) -> str:
        return f"{percentage}myacquisition"
```

**2. Use in Configuration:**

```json
{
  "acquisition_function": {
    "type": "myacquisition"
  }
}
```

---

## Troubleshooting

### CUDA Out of Memory

**Symptoms:**
```
RuntimeError: CUDA out of memory. Tried to allocate 29.68 GiB
```

**Solutions:**

1. **Reduce Batch Sizes:**
```json
{
  "oracle": {"batch_size": 8},
  "trainer": {"batch_size": 16}
}
```

2. **Reduce Candidate Pool:**
```json
{
  "active_learning": {
    "n_candidates_per_cycle": 50000
  }
}
```

3. **Use Smaller Model:**
```json
{
  "trainer": {
    "architecture": "deepstarr_small"
  }
}
```

### cuDNN Version Incompatibility

**Symptoms:**
```
Error: cuDNN version incompatibility detected
```

**Solution:**
GPU wrapper is used automatically by all scripts. If manual run needed:
```bash
./scripts/gpu_wrapper.sh python your_script.py
```

### Import Errors

**Symptoms:**
```
ImportError: cannot import name 'RandomProposalStrategy'
```

**Solutions:**

1. **Check Working Directory:**
```bash
cd /path/to/active_learning_project
python scripts/run_active_learning.py ...
```

2. **Verify Environment:**
```bash
# Conda
conda activate active-learning-genomics

# Pip
source .venv/bin/activate
```

3. **Reinstall Dependencies:**
```bash
pip install -r requirements.txt --force-reinstall
```

### Missing Oracle Models

**Symptoms:**
```
ValueError: No trained models found in models/dream_rnn_ensemble
```

**Solution:**
Train oracle ensemble:
```bash
make train  # or dvc repro train_dream_rnn_ensemble
```

### Checkpoint Not Found

**Symptoms:**
Training always starts from round 0 despite previous runs.

**Check:**
```bash
ls results/.../idx0/round_000/
# Should show: model_best.pth, metrics.json, training_data.json
```

**Solution:**
If files missing, previous run didn't complete. Rerun to completion.

### Different Runs Same Seed

**Cause:**
Not specifying `--run-index` or using same index.

**Solution:**
```bash
python scripts/run_active_learning.py --config config.json --run-index 0
python scripts/run_active_learning.py --config config.json --run-index 1
python scripts/run_active_learning.py --config config.json --run-index 2
```

### Performance Optimization

**Monitor GPU:**
```bash
watch -n 1 nvidia-smi
```

**Optimize Batch Sizes:**
- Start small (8-16) and increase until GPU memory ~80% full
- Monitor with `nvidia-smi`

**Parallel Data Loading:**
```json
{
  "trainer": {
    "n_workers": 8
  }
}
```

**Disk I/O:**
- Use SSD for data/models directories
- Consider tmpfs/ramfs for temporary files

---

## API Reference

### ConfigurationManager

```python
from code.active_learning import ConfigurationManager

config_manager = ConfigurationManager(config_dict)

# Get run directory
run_dir = config_manager.get_run_directory()

# Get seed
seed = config_manager.seed  # 42 + run_index * 1000

# Find last completed round
last_round = config_manager.find_last_completed_round(run_dir, n_cycles)

# Save configuration
config_manager.save_config(run_dir)
config_manager.save_metadata(run_dir)
```

### EnsembleOracle

```python
from code.active_learning import EnsembleOracle

oracle = EnsembleOracle(
    composition=[
        {"type": "dream_rnn", "count": 5, "model_dir": "models/dream_rnn_ensemble"}
    ],
    device="auto",
    seqsize=249,
    batch_size=1024
)

# Predict
predictions = oracle.predict(sequences)  # shape: (n_sequences, 2)

# Get uncertainty
uncertainties = oracle.get_uncertainty(sequences)  # shape: (n_sequences,)

# Both at once
preds, uncerts = oracle.predict_with_uncertainty(sequences)

# Composition string
comp_str = oracle.get_composition_string()  # "5dreamrnn"
```

### ActiveLearningCycle

```python
from code.active_learning import ActiveLearningCycle

al_cycle = ActiveLearningCycle(
    oracle=oracle,
    trainer=trainer,
    proposal_strategy=proposal_strategy,
    acquisition_function=acquisition_function,
    output_dir=output_dir,
    initial_sequences=initial_sequences,
    initial_labels=initial_labels,
    n_cycles=5,
    n_candidates_per_cycle=100000,
    n_acquire_per_cycle=20000,
    training_strategy="from_scratch",
    seed=1042,
    config_manager=config_manager,
    checkpoint_manager=checkpoint_manager,
    test_datasets=test_datasets
)

# Run all cycles
results = al_cycle.run_all_cycles()
```

### CheckpointManager

```python
from code.active_learning import CheckpointManager

checkpoint_mgr = CheckpointManager()

# Save checkpoint
checkpoint_mgr.save_round_checkpoint(
    round_dir=Path("results/.../idx0/round_001"),
    model_path="path/to/model_best.pth",
    metrics=metrics_dict,
    training_data={"sequences": seqs, "labels": labels}
)

# Load checkpoint
checkpoint_data = checkpoint_mgr.load_round_checkpoint(
    round_dir=Path("results/.../idx0/round_001")
)

# Check if exists
exists = checkpoint_mgr.checkpoint_exists(
    round_dir=Path("results/.../idx0/round_001")
)
```

---

## Dataset Information

### DeepSTARR Dataset

**Source:** [Zenodo DOI: 10.5281/zenodo.5502060](https://doi.org/10.5281/zenodo.5502060)

**Files:**
- `Sequences_Train.fa`: Training sequences (116.9 MB)
- `Sequences_Val.fa`: Validation sequences (11.8 MB)
- `Sequences_Test.fa`: Test sequences (12.0 MB)
- `Sequences_activity_Train.txt`: Training labels (43.7 MB)
- `Sequences_activity_Val.txt`: Validation labels (4.4 MB)
- `Sequences_activity_Test.txt`: Test labels (4.5 MB)
- `DeepSTARR.model.h5`: Pre-trained model (2.6 MB)
- `DeepSTARR.model.json`: Model architecture (11.6 kB)

**Sizes:**
- Training: ~473k sequences
- Validation: ~47k sequences
- Test: ~82k sequences
- Sequence length: 249 bp

**Citation:**
de Almeida, Bernardo P. (2021). DeepSTARR manuscript data. Zenodo. https://doi.org/10.5281/zenodo.5502060

### Data Processing Pipeline

```bash
# 1. Download raw data
dvc repro download_data

# 2. Preprocess to TSV format
dvc repro preprocess_deepstarr

# 3. Generate test/val datasets
dvc repro generate_test_val_datasets_deepstarr
```

**Processed Format (TSV):**
```
Sequence	Dev_log2_enrichment	Hk_log2_enrichment
ATCG...	1.234	0.567
GCTA...	0.890	1.234
```

---

## System Requirements

**Minimum:**
- GPU: NVIDIA GPU with 8GB+ VRAM
- CUDA: Compatible version (automatically detected)
- Python: 3.8+
- Memory: 16GB RAM
- Storage: 20GB free space

**Recommended:**
- GPU: NVIDIA GPU with 16GB+ VRAM (A100, V100, RTX 3090, etc.)
- CUDA: 11.7+ or 12.1+
- Python: 3.10+
- Memory: 32GB+ RAM
- Storage: 50GB+ free space (SSD preferred)

**Supported Operating Systems:**
- Linux (Ubuntu 20.04+, CentOS 8+)
- macOS (CPU only, limited testing)
- Windows (via WSL2)

---

## License

This project is licensed under the MIT License.

The DeepSTARR dataset is licensed under [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).

---

## Changelog

### v2.0 - Directory Structure Reorganization (2024-10-22)

**Major Changes:**
- Reorganized results hierarchy with dataset as top level
- Explicit oracle/student ensemble compositions
- Strategy percentages in naming
- Detailed round 0 initialization tracking
- Clear validation dataset naming with `val_` prefix
- Added metadata.json with git hash and hardware info

**Breaking Changes:**
- New directory structure incompatible with v1
- Configuration requires `oracle.composition` instead of `oracle.architecture`
- `validation_dataset` must include `val_` prefix

**Migration:**
Existing results will need to be moved or regenerated. No automatic migration tool provided.

### v1.0 - Initial Release

- Hierarchical directory structure
- Automatic checkpointing and resumption
- Multi-test dataset evaluation
- DVC pipeline integration
- Deterministic seeding
