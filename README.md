# Active Learning for ML in Genomics

A research project focused on active learning strategies for machine learning in genomics, using the DeepSTARR dataset as a foundation.

## Dataset

This project uses the [DeepSTARR manuscript data](https://doi.org/10.5281/zenodo.5502060) from Zenodo, which includes:

- **Sequences**: FASTA files with DNA sequences from train/validation/test sets
- **Activity data**: Developmental and housekeeping activity measurements
- **Pre-trained model**: DeepSTARR Keras model files

**Citation**: de Almeida, Bernardo P. (2021). DeepSTARR manuscript data. Zenodo. https://doi.org/10.5281/zenodo.5502060

## Quick Start

### 1. Clone and Setup Environment

Choose one of the following methods:

#### Option A: Conda (Recommended)
```bash
git clone <your-repo-url>
cd active_learning_project
conda env create -f environment.yml
conda activate active-learning-genomics
```

#### Option B: Python venv
```bash
git clone <your-repo-url>
cd active_learning_project
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Initialize DVC and Download Data

```bash
# Initialize DVC (if not already done)
dvc init

# Download the DeepSTARR dataset
dvc repro

# Verify the download
ls -la data/raw/
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

### 3. Configure DVC Remote (Optional)

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
├── data/
│   ├── README.md           # Data documentation
│   ├── manifest.csv        # Dataset URLs and checksums
│   ├── .gitkeep           # Keeps data/ directory in Git
│   └── raw/               # Downloaded datasets (ignored by Git)
└── scripts/
    └── download_data.py    # Data download and verification script
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
