"""
Central configuration management for the active learning genomics project.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import torch


# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
CODE_DIR = PROJECT_ROOT / "code"

# Data file paths
DEEPSTARR_FILES = {
    'train_fasta': RAW_DATA_DIR / "Sequences_Train.fa",
    'val_fasta': RAW_DATA_DIR / "Sequences_Val.fa", 
    'test_fasta': RAW_DATA_DIR / "Sequences_Test.fa",
    'train_activity': RAW_DATA_DIR / "Sequences_activity_Train.txt",
    'val_activity': RAW_DATA_DIR / "Sequences_activity_Val.txt",
    'test_activity': RAW_DATA_DIR / "Sequences_activity_Test.txt",
}

PROCESSED_FILES = {
    'train': PROCESSED_DATA_DIR / "train.txt",
    'val': PROCESSED_DATA_DIR / "val.txt",
    'test': PROCESSED_DATA_DIR / "test.txt",
}

# Model configuration
DEFAULT_MODEL_CONFIG = {
    'dream_rnn': {
        'model_type': 'dream_rnn',
        'seqsize': 249,
        'in_channels': 5,
        'out_channels': 320,
        'lstm_hidden_channels': 320,
        'kernel_sizes': [9, 15],
        'pool_size': 1,
        'dropout1': 0.2,
        'dropout2': 0.5,
        'learning_rate': 0.005,
        'batch_size': 32,
        'epochs': 80
    }
}

# Training configuration
DEFAULT_TRAINING_CONFIG = {
    'n_models': 5,
    'base_seed': 42,
    'n_workers': 4,
    'device': 'auto'
}

# Prediction configuration
DEFAULT_PREDICTION_CONFIG = {
    'batch_size': 32,
    'device': 'auto'
}


def get_device(device_spec: str = "auto") -> torch.device:
    """
    Get PyTorch device based on specification.
    
    Args:
        device_spec: Device specification ("auto", "cpu", "cuda:0", etc.)
        
    Returns:
        PyTorch device
    """
    if device_spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        return torch.device(device_spec)


def get_model_config(model_type: str, **overrides) -> Dict[str, Any]:
    """
    Get configuration for a specific model type with optional overrides.
    
    Args:
        model_type: Type of model
        **overrides: Configuration overrides
        
    Returns:
        Model configuration dictionary
    """
    if model_type not in DEFAULT_MODEL_CONFIG:
        raise ValueError(f"Unknown model type: {model_type}")
    
    config = DEFAULT_MODEL_CONFIG[model_type].copy()
    config.update(overrides)
    return config


def get_training_config(**overrides) -> Dict[str, Any]:
    """
    Get training configuration with optional overrides.
    
    Args:
        **overrides: Configuration overrides
        
    Returns:
        Training configuration dictionary
    """
    config = DEFAULT_TRAINING_CONFIG.copy()
    config.update(overrides)
    return config


def get_prediction_config(**overrides) -> Dict[str, Any]:
    """
    Get prediction configuration with optional overrides.
    
    Args:
        **overrides: Configuration overrides
        
    Returns:
        Prediction configuration dictionary
    """
    config = DEFAULT_PREDICTION_CONFIG.copy()
    config.update(overrides)
    return config


def ensure_directories() -> None:
    """Ensure all required directories exist."""
    directories = [
        DATA_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODELS_DIR,
        RESULTS_DIR,
        CODE_DIR
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def get_model_output_dir(model_type: str, ensemble_name: str = None) -> Path:
    """
    Get output directory for trained models.
    
    Args:
        model_type: Type of model
        ensemble_name: Optional ensemble name
        
    Returns:
        Path to model output directory
    """
    if ensemble_name:
        return MODELS_DIR / f"{model_type}_{ensemble_name}"
    else:
        return MODELS_DIR / f"{model_type}_ensemble"


def get_results_dir(experiment_name: str) -> Path:
    """
    Get results directory for an experiment.
    
    Args:
        experiment_name: Name of the experiment
        
    Returns:
        Path to results directory
    """
    results_path = RESULTS_DIR / experiment_name
    results_path.mkdir(parents=True, exist_ok=True)
    return results_path
