# Active Learning for ML in Genomics

A research project focused on active learning strategies for machine learning in genomics, using the DeepSTARR dataset as a foundation.

## Dataset

This project uses the [DeepSTARR manuscript data](https://doi.org/10.5281/zenodo.5502060) from Zenodo, which includes:

- **Sequences**: FASTA files with DNA sequences from train/validation/test sets
- **Activity data**: Developmental and housekeeping activity measurements
- **Pre-trained model**: DeepSTARR Keras model files

**Citation**: de Almeida, Bernardo P. (2021). DeepSTARR manuscript data. Zenodo. https://doi.org/10.5281/zenodo.5502060

## Quick Start

### Option 1: One-Command Setup (Recommended)

```bash
git clone <your-repo-url>
cd active_learning_project
make setup
```

This automatically:
- Installs all dependencies
- Detects your NVIDIA driver version
- Installs compatible PyTorch
- Tests GPU availability
- Sets up the complete environment

### Active Learning Pipeline

After setup, you can run the active learning pipeline:

#### Quick Start - Run Experiments

```bash
# Run pre-configured experiments (recommended)
python scripts/run_experiments.py \
    --config experiments/random_random_replicates.yaml \
    --wait

# Run custom experiments with auto-GPU assignment
python scripts/run_experiments.py \
    --base-config configs/active_learning_random_random.json \
    --run-indices 1 2 3 \
    --auto-gpu \
    --wait

# Check GPU availability
python scripts/run_experiments.py --show-gpus
```

**See `experiments/README.md` for detailed documentation.**

#### Using Make Commands

```bash
# Test the active learning framework
make test-active-learning

# Run full active learning pipeline (from scratch training)
make active-learning

# Run active learning with fine-tuning and replay
make active-learning-finetune
```

#### Using DVC

```bash
# Run specific experiments via DVC
dvc repro active_learning_random_random_idx1
dvc repro active_learning_random_random_idx2
dvc repro active_learning_random_random_idx3
```

### GPU Compatibility and Reproducibility

This project includes automatic GPU compatibility fixes to ensure reproducible training across different systems:

#### cuDNN Version Compatibility
- **Issue**: PyTorch may be compiled against a different cuDNN version than system libraries
- **Solution**: Automatic detection and fix via `scripts/gpu_wrapper.sh`
- **Usage**: All GPU commands automatically use the wrapper script

#### Reproducible Setup
The setup process automatically:
1. Detects NVIDIA driver version
2. Installs compatible PyTorch version
3. Tests GPU availability
4. Creates GPU wrapper script for cuDNN compatibility
5. Verifies all components work together

#### Manual GPU Setup (if needed)
```bash
# Run GPU setup script
python scripts/setup_gpu.py

# Use GPU wrapper for any training command
./scripts/gpu_wrapper.sh python your_script.py
```

#### DVC Integration
All training stages in the DVC pipeline automatically use the GPU wrapper:
```bash
# These commands automatically handle GPU compatibility
dvc repro train_dream_rnn_ensemble
dvc repro test_active_learning
dvc repro active_learning
```

### Option 2: Manual Setup

#### Step 1: Clone Repository
```bash
git clone <your-repo-url>
cd active_learning_project
```

#### Step 2: Choose Environment Method

**Conda (Recommended):**
```bash
make conda-setup
# OR manually:
conda env create -f environment.yml
conda activate active-learning-genomics
conda run -n active-learning-genomics python scripts/setup_gpu.py
```

**Pip Virtual Environment:**
```bash
make pip-setup
# OR manually:
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/setup_gpu.py
```

### GPU Setup Details

The `setup_gpu.py` script automatically:
- Detects your NVIDIA driver version
- Installs the compatible PyTorch version
- Tests GPU availability
- Provides clear error messages if setup fails

**Supported Driver Versions:**
- 525+: PyTorch 2.1.0 with CUDA 12.1
- 520-524: PyTorch 2.0.1 with CUDA 11.8
- 515-519: PyTorch 1.13.1 with CUDA 11.7
- 510-514: PyTorch 1.12.1 with CUDA 11.6
- 470-509: PyTorch 1.11.0 with CUDA 11.5
- 450-469: PyTorch 1.10.1 with CUDA 11.1

### 3. Download Data and Train Models

```bash
# Complete pipeline (recommended)
make full-pipeline

# OR step by step:
make data      # Download and preprocess data
make train     # Train DREAM-RNN ensemble
make predict   # Generate predictions

# OR using DVC directly:
dvc repro download_data
dvc repro preprocess_deepstarr  
dvc repro train_dream_rnn_ensemble
```

The dataset will be downloaded to `data/raw/` with the following files:
- `DeepSTARR.model.h5` (2.6 MB) - Pre-trained model weights
- `DeepSTARR.model.json` (11.6 kB) - Model architecture
- `Sequences_activity_Test.txt` (4.5 MB) - Test set activity data
- `Sequences_activity_Train.txt` (43.7 MB) - Training set activity data
- `Sequences_activity_Val.txt` (4.4 MB) - Validation set activity data
- `Sequences_Test.fa` (12.0 MB) - Test set DNA sequences
- `Sequences_Train.fa` (116.9 MB) - Training set DNA sequences
- `Sequences_Val.fa` (11.8 MB) - Validation set DNA sequences

After preprocessing, the data will be available in `data/processed/` as TSV files:
- `train.txt` - Training data with sequences and activity measurements
- `val.txt` - Validation data
- `test.txt` - Test data

### 4. Train DREAM-RNN Models (Optional)

To train DREAM-RNN ensemble models:

```bash
# Train ensemble of 5 models (this will take several hours)
dvc repro train_dream_rnn_ensemble

# Or train with custom parameters
python scripts/train_ensemble.py \
  --model_type dream_rnn \
  --train_data data/processed/train.txt \
  --val_data data/processed/val.txt \
  --output_dir models/dream_rnn_ensemble \
  --n_models 5 \
  --epochs 80 \
  --batch_size 32 \
  --lr 0.005
```

### 5. Generate Predictions

```bash
# Generate predictions using trained ensemble
python scripts/predict_ensemble.py \
  --checkpoint_dir models/dream_rnn_ensemble \
  --input_data data/processed/test.txt \
  --output results/test_predictions.csv
```

### 6. Configure DVC Remote (Optional)

To share data with collaborators or backup to cloud storage:

```bash
# Example for S3
dvc remote add -d storage s3://your-bucket/active_learning_project

# Push data to remote
dvc push
```

## Project Structure

```
active_learning_project/
├── README.md                 # This file
├── .gitignore               # Git ignore rules for Python/ML projects
├── requirements.txt         # Python dependencies (pip)
├── environment.yml          # Conda environment specification
├── dvc.yaml                 # DVC pipeline definition
├── dvc.lock                 # DVC lock file for reproducibility
├── params.yaml              # DVC parameters for training
├── code/                    # Core package
│   ├── __init__.py
│   ├── config.py           # Configuration management
│   ├── utils.py            # Utility functions
│   ├── prixfixe.py         # Streamlined Prix Fixe framework
│   └── models/             # Model architectures
│       ├── __init__.py
│       ├── config.py       # Model registry
│       └── dream_rnn.py    # DREAM-RNN implementation
├── data/
│   ├── README.md           # Data documentation
│   ├── manifest.csv        # Dataset URLs and checksums
│   ├── .gitkeep           # Keeps data/ directory in Git
│   ├── raw/               # Downloaded datasets (ignored by Git)
│   └── processed/         # Processed data (ignored by Git)
├── models/                 # Trained models (ignored by Git)
├── results/               # Experiment results (ignored by Git)
└── scripts/
    ├── download_data.py    # Data download and verification script
    ├── preprocess_deepstarr.py  # Data preprocessing
    ├── train_ensemble.py   # Model training
    └── predict_ensemble.py # Model prediction
```

## Active Learning Framework

This project implements a modular active learning framework for genomic sequence design, following the PIONEER approach. The framework enables iterative improvement of deep learning models through strategic sequence proposal and acquisition.

### Framework Components

#### 1. Oracle Ensemble
- **Purpose**: Provides high-quality labels for proposed sequences
- **Implementation**: Uses the trained 5-model DREAM-RNN ensemble
- **Capabilities**: Predicts regulatory activity and provides uncertainty estimates

#### 2. Student Model
- **Architecture**: DeepSTARR CNN model (exact implementation from the paper)
- **Input**: 249bp DNA sequences with 4-channel one-hot encoding
- **Output**: Developmental and housekeeping activity predictions
- **Training**: Configurable (from scratch or fine-tuning with replay)

#### 3. Sequence Proposal Strategies
- **Random**: Generates completely random DNA sequences
- **Partial Mutagenesis**: Introduces mutations to existing sequences
- **Uncertainty-Guided**: Uses model uncertainty to guide mutations
- **Mixed**: Combines multiple strategies

#### 4. Acquisition Functions
- **Random**: Randomly selects sequences from candidates
- **Uncertainty**: Selects sequences with highest predictive uncertainty
- **LCMD**: Balances informativeness, diversity, and representativeness
- **Diversity**: Maximizes sequence diversity in selected batch

#### 5. Training Strategies
- **From Scratch**: Retrain model completely each cycle
- **Fine-tuning**: Continual learning with optional replay buffer

### Active Learning Cycle

Each cycle follows the PIONEER approach:

1. **Generation**: Propose candidate sequences using selected strategy
2. **Acquisition**: Select most informative sequences using acquisition function
3. **Labeling**: Label selected sequences using oracle ensemble
4. **Retraining**: Update student model with new labeled data

### Configuration

Active learning parameters can be configured in JSON files:

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
    "enable_replay": false,
    "replay_buffer_size": 1000
  },
  "oracle": {
    "architecture": "dream_rnn_ensemble"
  },
  "data": {
    "dataset_name": "train"
  }
}
```

### Checkpointing and Reproducibility

The framework includes comprehensive checkpointing for reproducible experiments:

#### Deterministic Seeding
- Seeds are calculated from run index: `seed = 42 + run_index * 1000`
- Same run index always produces identical results
- All random number generators (NumPy, PyTorch) use the calculated seed

#### Hierarchical Result Organization
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
                round_000/  # Baseline
                round_001/  # After cycle 1
                ...
```

Example: `results/random/uncertainty/100000cand_20000acq/deepstarr/dream_rnn_ensemble/train/idx0/`

#### Automatic Resumption
- Each round saves: model weights, metrics, training data state
- Interrupted training automatically resumes from last completed round
- No wasted computation re-running completed rounds

#### Usage Examples

Run with default seed (42):
```bash
python scripts/run_active_learning.py \
  --config configs/active_learning_full_config_adaptive.json
```

Run with specific index for different seed:
```bash
python scripts/run_active_learning.py \
  --config configs/active_learning_full_config_adaptive.json \
  --run-index 5  # Uses seed = 5042
```

Run multiple replicates in parallel:
```bash
for i in {0..4}; do
  python scripts/run_active_learning.py \
    --config config.json \
    --run-index $i &
done
```

Resume interrupted training (automatically detects checkpoints):
```bash
# Simply rerun the same command
python scripts/run_active_learning.py \
  --config config.json \
  --run-index 0
```

**See `CHECKPOINT_GUIDE.md` for detailed documentation.**

### Expected Performance

Based on testing, the active learning framework achieves:

- **Oracle Performance**: 5-model DREAM-RNN ensemble with uncertainty quantification
- **Student Training**: DeepSTARR model training in ~2-3 minutes per epoch (GPU)
- **Sequence Generation**: 100k random sequences in ~1-2 seconds
- **Acquisition**: Uncertainty-based selection of 20k sequences in ~30 seconds
- **Oracle Labeling**: 20k sequences labeled in ~2-3 minutes (with batch_size=8 for memory efficiency)
- **Total Cycle Time**: ~15-20 minutes per active learning cycle

### Memory Management

The framework includes automatic memory management for large-scale active learning:

- **Oracle Batch Size**: Reduced to 8 for processing large candidate sets (100k sequences)
- **GPU Memory**: Automatically handles CUDA memory allocation and cleanup
- **Chunked Processing**: Large candidate sets are processed in manageable chunks
- **Memory Monitoring**: Built-in memory usage tracking and optimization

### Results Structure

Active learning results are saved in `results/active_learning_full/`:
```
results/active_learning_full/
├── active_learning_config.json     # Configuration used
├── all_cycle_results.json         # Detailed results for all cycles
├── active_learning_summary.json   # Summary statistics
├── final_training_data.json       # Final training dataset
└── student_model/                 # Student model weights
    └── model_best_MSE.pth         # Best model checkpoint
```

### Troubleshooting

#### Common Issues and Solutions

**1. CUDA Out of Memory Error**
```
Error: CUDA out of memory. Tried to allocate 29.68 GiB
```
**Solution**: Reduce batch sizes in configuration:
- Set `oracle.batch_size` to 8 or lower
- Set `trainer.batch_size` to 16 or lower
- Reduce `n_candidates_per_cycle` if needed

**2. cuDNN Version Incompatibility**
```
Error: cuDNN version incompatibility
```
**Solution**: Use the GPU wrapper script (automatic):
```bash
./scripts/gpu_wrapper.sh python your_script.py
```

**3. Missing Oracle Models**
```
Error: No models could be loaded
```
**Solution**: Ensure DREAM-RNN ensemble is trained first:
```bash
make train  # or dvc repro train_dream_rnn_ensemble
```

**4. Import Errors**
```
ImportError: cannot import name 'MixedProposalStrategy'
```
**Solution**: Update the active learning module imports:
```bash
# The imports are automatically handled in the latest version
```

#### Performance Optimization

- **GPU Memory**: Monitor with `nvidia-smi` during training
- **Batch Sizes**: Start with small values (8-16) and increase if memory allows
- **Parallel Processing**: Use multiple GPUs by setting `--device auto --parallel`
- **Data Loading**: Increase `n_workers` for faster data loading (4-8 workers)

### Reproducibility

This project is designed for complete reproducibility across different systems:

#### Environment Reproducibility
- **DVC Pipeline**: All data processing and training steps are versioned
- **Dependency Management**: Exact package versions specified in `requirements.txt` and `environment.yml`
- **GPU Compatibility**: Automatic detection and setup of compatible PyTorch versions
- **Random Seeds**: All components use fixed seeds for reproducible results

#### Data Reproducibility
- **DVC Tracking**: All datasets and models are tracked with checksums
- **Manifest Files**: Exact URLs and checksums for all downloaded data
- **Lock Files**: `dvc.lock` captures exact pipeline state
- **Configuration**: All parameters saved in JSON configuration files

#### Model Reproducibility
- **Fixed Seeds**: Random number generators seeded consistently
- **Architecture**: Exact model architectures from published papers
- **Training**: Deterministic training with fixed hyperparameters
- **Checkpoints**: Model weights saved with metadata

#### Reproducing Results

To reproduce the exact results:

1. **Clone and Setup**:
   ```bash
   git clone <repository>
   cd active_learning_project
   make setup  # or make conda-setup
   ```

2. **Run Complete Pipeline**:
   ```bash
   make full-pipeline  # Downloads data, trains oracle, runs active learning
   ```

3. **Or Run Individual Steps**:
   ```bash
   make data      # Download and preprocess data
   make train     # Train oracle ensemble
   make active-learning-full  # Run full active learning
   ```

4. **Verify Results**:
   ```bash
   # Check DVC pipeline status
   dvc status
   
   # Verify data integrity
   dvc repro --dry-run
   ```

#### System Requirements

- **GPU**: NVIDIA GPU with 8GB+ VRAM (16GB+ recommended for full pipeline)
- **CUDA**: Compatible CUDA version (automatically detected)
- **Python**: 3.8+ (3.10+ recommended)
- **Memory**: 32GB+ RAM recommended for large datasets
- **Storage**: 50GB+ free space for datasets and models

## DREAM-RNN Training

This project implements DREAM-RNN models using the Prix Fixe framework for genomic sequence analysis.

### Model Architecture

The DREAM-RNN model implements the winning architecture from the Random Promoter DREAM Challenge 2022:

- **First Block**: Convolutional layers with kernel sizes [9, 15] and dropout (0.2)
- **Core Block**: Bidirectional LSTM (320 hidden channels) + convolutional layers with dropout (0.5)
- **Final Block**: Global average pooling + fully connected layers for two tasks (Dev, Hk activity)

**Key Features:**
- Input: 5-channel encoding (A,C,G,T,reverse_complement_indicator)
- Sequence length: 249 bp (standardized for DeepSTARR data)
- Total parameters: ~4.3M (efficient design)
- Based on Prix Fixe framework for modular architecture

### Training Pipeline

1. **Data Preprocessing**: Merge FASTA sequences with activity measurements
2. **Model Training**: Train ensemble of 5 models with different random seeds
3. **Prediction**: Generate predictions with uncertainty quantification

### Training Configuration

Key hyperparameters can be adjusted in `params.yaml`:

```yaml
train_ensemble:
  n_models: 5          # Number of models in ensemble
  epochs: 80           # Training epochs (from DREAM Challenge)
  batch_size: 32       # Batch size
  lr: 0.005           # Learning rate (from DREAM Challenge)
  out_channels: 320    # Model width (from DREAM Challenge)
  lstm_hidden_channels: 320  # LSTM hidden size (from DREAM Challenge)
  kernel_sizes: [9, 15] # Convolutional kernel sizes (from DREAM Challenge)
  dropout1: 0.2        # Dropout for first/core blocks
  dropout2: 0.5        # Dropout for core block
```

**Note**: All hyperparameters are based on the winning DREAM Challenge solution and have been validated on millions of random promoter sequences.

### Model Files

After training, the ensemble will be saved in `models/dream_rnn_ensemble/`:

```
models/dream_rnn_ensemble/
├── model_0/           # Individual model directories
│   └── model_best_MSE.pth
├── model_1/
│   └── model_best_MSE.pth
├── ...
├── training_config.json    # Training configuration
└── training_results.json   # Training results
```

## Data Management

### Data Versioning with DVC

This project uses [DVC (Data Version Control)](https://dvc.org/) to manage large datasets:

- **Large files** (datasets, models) are stored outside Git
- **Small metadata files** (`.dvc` files, `dvc.lock`) are tracked in Git
- **Reproducible downloads** with checksum verification
- **Pipeline automation** for data processing workflows

### Data Manifest

The `data/manifest.csv` file contains:
- **URLs**: Direct download links to dataset files
- **Checksums**: MD5 hashes for integrity verification
- **Filenames**: Standardized output names

### Adding New Datasets

1. Add entries to `data/manifest.csv`:
   ```csv
   url,md5,sha256,filename
   https://example.org/data/new_dataset.fa,abc123...,,new_dataset.fa
   ```

2. Re-run the pipeline:
   ```bash
   dvc repro
   ```

## Reproducibility

### Environment Reproducibility

- **Conda**: Use `environment.yml` for complete environment specification
- **Pip**: Use `requirements.txt` with pinned versions
- **Docker**: Consider adding a `Dockerfile` for full containerization

### Data Reproducibility

- **DVC lock file**: `dvc.lock` ensures exact same data versions
- **Checksum verification**: All downloads are verified against provided hashes
- **Pipeline tracking**: DVC tracks dependencies and automatically re-runs when inputs change

### Reproducing Results

```bash
# Clone the repository
git clone <your-repo-url>
cd active_learning_project

# Setup environment
conda env create -f environment.yml
conda activate active-learning-genomics

# Download data (will use exact versions from dvc.lock)
dvc repro

# Your data is now ready for analysis
```

## Development Workflow

### Making Changes

1. **Modify pipeline**: Edit `dvc.yaml` to add new stages
2. **Update dependencies**: Modify `data/manifest.csv` for new datasets
3. **Test changes**: Run `dvc repro --dry` to see what would change
4. **Commit changes**: Include both code and `dvc.lock` in commits

### Collaboration

- **Share code**: Standard Git workflow
- **Share data**: Use `dvc push` to upload to shared remote storage
- **Sync data**: Collaborators run `dvc pull` to download latest data

## Troubleshooting

### Common Issues

1. **Build errors during pip install**:
   ```bash
   pip install --upgrade pip setuptools wheel
   pip install -r requirements.txt
   ```

2. **DVC not finding data**:
   ```bash
   dvc pull  # Download data from remote
   ```

3. **Checksum mismatches**:
   - Verify the file wasn't corrupted during download
   - Check if the source URL has changed

### Getting Help

- [DVC Documentation](https://dvc.org/doc)
- [Zenodo DeepSTARR Dataset](https://doi.org/10.5281/zenodo.5502060)
- [Active Learning in ML](https://en.wikipedia.org/wiki/Active_learning_(machine_learning))

## License

This project is licensed under the MIT License - see the LICENSE file for details.

The DeepSTARR dataset is licensed under [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).
