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
