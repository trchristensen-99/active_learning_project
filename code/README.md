# Code Package

This package contains the core functionality for training and evaluating deep learning models for genomic sequence analysis.

## Architecture Overview

The package is organized into several modules:

### `models/` - Model Architectures

- **`dream_rnn.py`**: Implementation of DREAM-RNN using streamlined Prix Fixe framework
- **`config.py`**: Model registry and configuration management
- **`__init__.py`**: Package exports

### `prixfixe.py` - Streamlined Prix Fixe Framework

- **`DREAMRNN`**: Complete DREAM-RNN model implementation
- **`DREAMRNNDataset`**: Data loading and preprocessing
- **`DREAMRNNTrainer`**: Training loop with validation
- **`build_dream_rnn()`**: Model builder function

This streamlined implementation contains only the components needed for DREAM-RNN, making it easier to understand and maintain than the full Prix Fixe framework.

### `config.py` - Configuration Management

Central configuration for:
- Project paths and directories
- Model hyperparameters
- Training settings
- Device management

### `utils.py` - Utility Functions

Helper functions for:
- Sequence encoding and preprocessing
- Model loading and ensemble management
- Prediction with uncertainty quantification
- Metrics calculation
- Data validation

## Adding New Model Types

To add a new model architecture:

1. **Create model file**: `code/models/new_model.py`
   ```python
   def build_new_model(**kwargs):
       # Model implementation
       return model
   
   def get_new_model_config():
       return {...}
   ```

2. **Register in config**: Update `code/models/config.py`
   ```python
   MODEL_REGISTRY["new_model"] = build_new_model
   MODEL_CONFIGS["new_model"] = get_new_model_config()
   ```

3. **Update exports**: Add to `code/models/__init__.py`
   ```python
   from .new_model import build_new_model
   __all__ = [..., "build_new_model"]
   ```

## Extending to New Datasets

To support a new dataset:

1. **Update paths**: Add dataset paths to `code/config.py`
2. **Create preprocessor**: Add preprocessing script in `scripts/`
3. **Update DVC pipeline**: Add new stages to `dvc.yaml`
4. **Test compatibility**: Ensure data format matches model expectations

## Usage Examples

### Building a Model

```python
from code.models import build_model

# Build DREAM-RNN model
model = build_model(
    "dream_rnn",
    seqsize=249,
    out_channels=320,
    lstm_hidden_channels=320,
    kernel_sizes=[9, 15],
    generator=torch.Generator().manual_seed(42)
)
```

### Loading an Ensemble

```python
from code.utils import load_ensemble_models

ensemble = load_ensemble_models(
    "models/dream_rnn_ensemble",
    device=torch.device("cuda")
)
```

### Generating Predictions

```python
from code.utils import predict_with_uncertainty

dev_means, dev_stds, hk_means, hk_stds = predict_with_uncertainty(
    ensemble, sequences, device
)
```

## Configuration

All configuration is centralized in `code/config.py` and can be overridden:

```python
from code.config import get_model_config, get_training_config

# Get model configuration with overrides
model_config = get_model_config("dream_rnn", out_channels=512)

# Get training configuration
train_config = get_training_config(n_models=10, epochs=100)
```
