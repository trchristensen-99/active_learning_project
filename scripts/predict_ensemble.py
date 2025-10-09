#!/usr/bin/env python3
"""
Generate predictions using trained ensemble models.

This script loads an ensemble of trained models and generates predictions
with uncertainty quantification for genomic sequences.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from code.models import build_model


def one_hot_encode(seq: str) -> List[List[int]]:
    """Convert DNA sequence to one-hot encoding."""
    mapping = {
        'A': [1, 0, 0, 0],
        'G': [0, 1, 0, 0], 
        'C': [0, 0, 1, 0],
        'T': [0, 0, 0, 1],
        'N': [0, 0, 0, 0]
    }
    return [mapping[base] for base in seq]


def load_ensemble(checkpoint_dir: str, device: torch.device) -> List[torch.nn.Module]:
    """
    Load ensemble of trained models from checkpoint directory.
    
    Args:
        checkpoint_dir: Directory containing trained models
        device: Device to load models on
        
    Returns:
        List of loaded models
    """
    models = []
    
    # Load training configuration
    config_path = os.path.join(checkpoint_dir, "training_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Training config not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Find all model directories
    model_dirs = [d for d in os.listdir(checkpoint_dir) 
                  if d.startswith('model_') and os.path.isdir(os.path.join(checkpoint_dir, d))]
    model_dirs.sort(key=lambda x: int(x.split('_')[1]))
    
    print(f"Loading {len(model_dirs)} models from {checkpoint_dir}")
    
    for model_dir in model_dirs:
        model_path = os.path.join(checkpoint_dir, model_dir, "model_best_MSE.pth")
        
        if not os.path.exists(model_path):
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
            print(f"  Loaded model from {model_dir}")
            
        except Exception as e:
            print(f"  Failed to load model from {model_dir}: {e}")
    
    if not models:
        raise RuntimeError("No models could be loaded")
    
    print(f"Successfully loaded {len(models)} models")
    return models


def predict_sequence(ensemble: List[torch.nn.Module], sequence: str, device: torch.device) -> Tuple[float, float, float, float]:
    """
    Predict on a single sequence using ensemble.
    
    Args:
        ensemble: List of trained models
        sequence: DNA sequence to predict on
        device: Device to run inference on
        
    Returns:
        Tuple of (dev_mean, dev_std, hk_mean, hk_std)
    """
    # Encode sequence
    encoded_seq = one_hot_encode(sequence)
    
    # Add reverse complement indicator (0 for forward)
    encoded_seq_with_rev = [encoded_base + [0] for encoded_base in encoded_seq]
    
    # Convert to tensor
    x = torch.tensor(
        np.array(encoded_seq_with_rev).reshape(1, len(sequence), 5).transpose(0, 2, 1),
        device=device,
        dtype=torch.float32
    )
    
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
    dev_mean = np.mean(dev_predictions)
    dev_std = np.std(dev_predictions)
    hk_mean = np.mean(hk_predictions)
    hk_std = np.std(hk_predictions)
    
    return dev_mean, dev_std, hk_mean, hk_std


def predict_batch(ensemble: List[torch.nn.Module], sequences: List[str], device: torch.device, batch_size: int = 32) -> List[Tuple[float, float, float, float]]:
    """
    Predict on a batch of sequences using ensemble.
    
    Args:
        ensemble: List of trained models
        sequences: List of DNA sequences
        device: Device to run inference on
        batch_size: Batch size for processing
        
    Returns:
        List of prediction tuples (dev_mean, dev_std, hk_mean, hk_std)
    """
    predictions = []
    
    for i in tqdm(range(0, len(sequences), batch_size), desc="Predicting batches"):
        batch_sequences = sequences[i:i + batch_size]
        batch_predictions = []
        
        for seq in batch_sequences:
            pred = predict_sequence(ensemble, seq, device)
            batch_predictions.append(pred)
        
        predictions.extend(batch_predictions)
    
    return predictions


def main():
    parser = argparse.ArgumentParser(description="Generate predictions using trained ensemble")
    
    # Input/Output
    parser.add_argument("--checkpoint_dir", required=True,
                       help="Directory containing trained ensemble models")
    parser.add_argument("--input_data", required=True,
                       help="Path to input data (TSV with ID and Sequence columns)")
    parser.add_argument("--output", required=True,
                       help="Path to save predictions (CSV format)")
    
    # Prediction options
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for prediction")
    parser.add_argument("--device", default="auto",
                       help="Device to run inference on")
    
    args = parser.parse_args()
    
    # Determine device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Load ensemble
    ensemble = load_ensemble(args.checkpoint_dir, device)
    
    # Load input data
    print(f"Loading input data from {args.input_data}")
    df = pd.read_csv(args.input_data, sep='\t')
    
    if 'Sequence' not in df.columns:
        raise ValueError("Input data must contain 'Sequence' column")
    
    if 'ID' not in df.columns:
        df['ID'] = range(len(df))
    
    print(f"Loaded {len(df)} sequences")
    
    # Generate predictions
    print("Generating predictions...")
    predictions = predict_batch(ensemble, df['Sequence'].tolist(), device, args.batch_size)
    
    # Create results DataFrame
    results = []
    for i, (dev_mean, dev_std, hk_mean, hk_std) in enumerate(predictions):
        results.append({
            'ID': df.iloc[i]['ID'],
            'Sequence': df.iloc[i]['Sequence'],
            'Dev_pred_mean': dev_mean,
            'Dev_pred_std': dev_std,
            'Hk_pred_mean': hk_mean,
            'Hk_pred_std': hk_std
        })
    
    # Add ground truth if available
    if 'Dev_log2_enrichment' in df.columns:
        for i, result in enumerate(results):
            result['Dev_true'] = df.iloc[i]['Dev_log2_enrichment']
    
    if 'Hk_log2_enrichment' in df.columns:
        for i, result in enumerate(results):
            result['Hk_true'] = df.iloc[i]['Hk_log2_enrichment']
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(args.output, index=False)
    
    print(f"Predictions saved to {args.output}")
    print(f"Generated predictions for {len(results)} sequences using {len(ensemble)} models")
    
    # Print summary statistics
    print(f"\nPrediction summary:")
    print(f"  Dev activity - Mean: {results_df['Dev_pred_mean'].mean():.3f} ± {results_df['Dev_pred_std'].mean():.3f}")
    print(f"  Hk activity - Mean: {results_df['Hk_pred_mean'].mean():.3f} ± {results_df['Hk_pred_std'].mean():.3f}")


if __name__ == "__main__":
    main()
