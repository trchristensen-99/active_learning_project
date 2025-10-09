"""
Utility functions for genomic sequence analysis and model training.
"""

import random
import numpy as np
import torch
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
from pathlib import Path


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducible results.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def one_hot_encode(seq: str) -> List[List[int]]:
    """
    Convert DNA sequence to one-hot encoding.
    
    Args:
        seq: DNA sequence string
        
    Returns:
        List of one-hot encoded bases
    """
    mapping = {
        'A': [1, 0, 0, 0],
        'G': [0, 1, 0, 0], 
        'C': [0, 0, 1, 0],
        'T': [0, 0, 0, 1],
        'N': [0, 0, 0, 0]  # Unknown bases
    }
    return [mapping.get(base.upper(), [0, 0, 0, 0]) for base in seq]


def reverse_complement(seq: str) -> str:
    """
    Get reverse complement of DNA sequence.
    
    Args:
        seq: DNA sequence string
        
    Returns:
        Reverse complement sequence
    """
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}
    return ''.join(complement[base] for base in reversed(seq))


def sequence_to_tensor(seq: str, seqsize: int = 249, device: torch.device = None) -> torch.Tensor:
    """
    Convert DNA sequence to PyTorch tensor for Prix Fixe models.
    
    Args:
        seq: DNA sequence string
        seqsize: Expected sequence length
        device: Device to place tensor on
        
    Returns:
        Tensor of shape (1, 5, seqsize) for Prix Fixe input
    """
    if len(seq) != seqsize:
        raise ValueError(f"Sequence length {len(seq)} does not match expected {seqsize}")
    
    # One-hot encode
    encoded = one_hot_encode(seq)
    
    # Add reverse complement indicator (0 for forward)
    encoded_with_rev = [encoded_base + [0] for encoded_base in encoded]
    
    # Convert to tensor and reshape for Prix Fixe (batch, channels, sequence)
    tensor = torch.tensor(
        np.array(encoded_with_rev).transpose(1, 0),  # (seqsize, 5) -> (5, seqsize)
        dtype=torch.float32
    ).unsqueeze(0)  # Add batch dimension: (1, 5, seqsize)
    
    if device is not None:
        tensor = tensor.to(device)
    
    return tensor


def load_ensemble_models(checkpoint_dir: str, device: torch.device) -> List[torch.nn.Module]:
    """
    Load ensemble of trained models from checkpoint directory.
    
    Args:
        checkpoint_dir: Directory containing trained models
        device: Device to load models on
        
    Returns:
        List of loaded models
    """
    import json
    from .models import build_model
    
    models = []
    
    # Load training configuration
    config_path = Path(checkpoint_dir) / "training_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Training config not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Find all model directories
    checkpoint_path = Path(checkpoint_dir)
    model_dirs = [d for d in checkpoint_path.iterdir() 
                  if d.is_dir() and d.name.startswith('model_')]
    model_dirs.sort(key=lambda x: int(x.name.split('_')[1]))
    
    for model_dir in model_dirs:
        model_path = model_dir / "model_best_MSE.pth"
        
        if not model_path.exists():
            print(f"Warning: Model weights not found: {model_path}")
            continue
        
        try:
            # Build model with same configuration
            model = build_model(
                config.get('model_type', 'dream_rnn'),
                generator=torch.Generator().manual_seed(42),  # Dummy generator for loading
                **{k: v for k, v in config.items() 
                   if k not in ['n_models', 'base_seed', 'device', 'output_dir', 'n_workers']}
            )
            
            # Load weights
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()
            
            models.append(model)
            
        except Exception as e:
            print(f"Failed to load model from {model_dir}: {e}")
    
    if not models:
        raise RuntimeError("No models could be loaded")
    
    return models


def predict_with_uncertainty(
    ensemble: List[torch.nn.Module], 
    sequences: List[str], 
    device: torch.device,
    batch_size: int = 32
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate predictions with uncertainty using ensemble.
    
    Args:
        ensemble: List of trained models
        sequences: List of DNA sequences
        device: Device to run inference on
        batch_size: Batch size for processing
        
    Returns:
        Tuple of (dev_means, dev_stds, hk_means, hk_stds)
    """
    dev_means = []
    dev_stds = []
    hk_means = []
    hk_stds = []
    
    for i in range(0, len(sequences), batch_size):
        batch_sequences = sequences[i:i + batch_size]
        
        for seq in batch_sequences:
            # Encode sequence
            x = sequence_to_tensor(seq, device=device)
            
            # Get predictions from all models
            dev_predictions = []
            hk_predictions = []
            
            with torch.no_grad():
                for model in ensemble:
                    pred = model(x)
                    dev_pred = pred[0].detach().cpu().item()
                    hk_pred = pred[1].detach().cpu().item()
                    
                    dev_predictions.append(dev_pred)
                    hk_predictions.append(hk_pred)
            
            # Calculate mean and std
            dev_means.append(np.mean(dev_predictions))
            dev_stds.append(np.std(dev_predictions))
            hk_means.append(np.mean(hk_predictions))
            hk_stds.append(np.std(hk_predictions))
    
    return (
        np.array(dev_means),
        np.array(dev_stds), 
        np.array(hk_means),
        np.array(hk_stds)
    )


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate regression metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary of metrics
    """
    from scipy.stats import pearsonr
    
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    
    try:
        pearson_r, pearson_p = pearsonr(y_true, y_pred)
    except:
        pearson_r, pearson_p = 0.0, 1.0
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'pearson_r': pearson_r,
        'pearson_p': pearson_p
    }


def save_predictions(
    predictions: Dict[str, Any], 
    output_path: str,
    include_ground_truth: bool = True
) -> None:
    """
    Save predictions to CSV file.
    
    Args:
        predictions: Dictionary containing predictions and metadata
        output_path: Path to save CSV file
        include_ground_truth: Whether to include ground truth values
    """
    df = pd.DataFrame(predictions)
    
    # Reorder columns for better readability
    columns = ['ID', 'Sequence']
    if include_ground_truth:
        if 'Dev_true' in df.columns:
            columns.append('Dev_true')
        if 'Hk_true' in df.columns:
            columns.append('Hk_true')
    
    columns.extend(['Dev_pred_mean', 'Dev_pred_std', 'Hk_pred_mean', 'Hk_pred_std'])
    
    # Only include columns that exist
    columns = [col for col in columns if col in df.columns]
    df = df[columns]
    
    df.to_csv(output_path, index=False)


def load_data_splits(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load train, validation, and test data splits.
    
    Args:
        data_dir: Directory containing processed data files
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    data_path = Path(data_dir)
    
    train_df = pd.read_csv(data_path / "train.txt", sep='\t')
    val_df = pd.read_csv(data_path / "val.txt", sep='\t')
    test_df = pd.read_csv(data_path / "test.txt", sep='\t')
    
    return train_df, val_df, test_df


def validate_sequences(sequences: List[str], expected_length: int = 249) -> List[str]:
    """
    Validate DNA sequences and return valid ones.
    
    Args:
        sequences: List of DNA sequences
        expected_length: Expected sequence length
        
    Returns:
        List of valid sequences
    """
    valid_sequences = []
    valid_bases = set('ATCGN')
    
    for seq in sequences:
        if len(seq) == expected_length and all(base.upper() in valid_bases for base in seq):
            valid_sequences.append(seq.upper())
        else:
            print(f"Warning: Invalid sequence (length {len(seq)}, expected {expected_length}): {seq[:50]}...")
    
    return valid_sequences
