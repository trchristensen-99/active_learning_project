#!/usr/bin/env python3
"""
Train ensemble of DREAM-RNN models for genomic sequence analysis.

This script trains multiple models with different random seeds to create
an ensemble for uncertainty quantification and improved predictions.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any
import torch
from torch import Generator
from tqdm import tqdm
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from code.models import build_model, get_model_config
from code.prixfixe import DREAMRNNTrainer
from code.active_learning.oracle_paths import build_evoaug_signature
# Using official evoaug package inside trainer; no custom augmentor here


def _build_evoaug_signature_simple(evoaug_cfg: Dict[str, Any]) -> str:
    """Back-compat wrapper calling canonical signature builder."""
    sig = build_evoaug_signature(evoaug_cfg)
    return sig or ''


def train_single_model(
    model_type: str,
    train_data_path: str,
    val_data_path: str,
    test_data_path: str,
    output_dir: str,
    model_idx: int,
    config: Dict[str, Any],
    device: torch.device,
    evoaug_config: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Train a single model with specified configuration.
    
    Args:
        model_type: Type of model to train
        train_data_path: Path to training data
        val_data_path: Path to validation data
        output_dir: Directory to save model
        model_idx: Index of this model in the ensemble
        config: Model and training configuration
        device: Device to train on
        evoaug_config: Optional EvoAug configuration
        
    Returns:
        Dictionary with training results
    """
    print(f"Training model {model_idx + 1}/{config['n_models']}")
    
    # Set random seed for reproducibility
    seed = config.get('base_seed', 42) + model_idx
    torch.manual_seed(seed)
    generator = Generator()
    generator.manual_seed(seed)
    
    # Build model
    model = build_model(
        model_type,
        generator=generator,
        **{k: v for k, v in config.items() 
           if k not in ['n_models', 'base_seed', 'device', 'output_dir']}
    )
    
    # Setup trainer
    model_output_dir = os.path.join(output_dir, f"model_{model_idx}")
    os.makedirs(model_output_dir, exist_ok=True)
    
    trainer = DREAMRNNTrainer(
        model,
        device=device,
        model_dir=model_output_dir,
        train_data_path=train_data_path,
        val_data_path=val_data_path,
        num_epochs=config['epochs'],
        lr=config['learning_rate'],
        batch_size=config['batch_size'],
        n_workers=config.get('n_workers', 4),
        evoaug_config=evoaug_config
    )
    
    # Train model
    try:
        trainer.fit()
        
        # Evaluate on test set
        test_results = trainer.evaluate_test_set(test_data_path)
        
        result = {
            'model_idx': model_idx,
            'seed': seed,
            'status': 'completed',
            'best_val_loss': trainer.best_val_loss,
            'model_path': os.path.join(model_output_dir, "model_best_MSE.pth"),
            'test_results': test_results,
            'output_dir': model_output_dir
        }
        
        print(f"  Model {model_idx + 1} training completed")
        return result
        
    except Exception as e:
        print(f"  Model {model_idx + 1} training failed: {e}")
        return {
            'model_idx': model_idx,
            'seed': seed,
            'status': 'failed',
            'error': str(e)
        }


def train_single_model_worker(args_tuple):
    """Worker function for parallel training."""
    (model_type, train_data_path, val_data_path, test_data_path, output_dir,
     model_idx, config, gpu_id, evoaug_config) = args_tuple

    # Set device for this worker
    device = torch.device(f"cuda:{gpu_id}")

    return train_single_model(
        model_type,
        train_data_path,
        val_data_path,
        test_data_path,
        output_dir,
        model_idx,
        config,
        device,
        evoaug_config
    )


def train_ensemble_parallel(
    model_type: str,
    train_data_path: str,
    val_data_path: str,
    test_data_path: str,
    output_dir: str,
    n_models: int,
    config: Dict[str, Any],
    max_parallel: int,
    evoaug_config: Dict[str, Any] = None
) -> List[Dict[str, Any]]:
    """
    Train ensemble models in parallel across multiple GPUs.
    
    Args:
        model_type: Type of model to train
        train_data_path: Path to training data
        val_data_path: Path to validation data
        output_dir: Directory to save models
        n_models: Number of models to train
        config: Training configuration
        max_parallel: Maximum number of parallel workers
        evoaug_config: Optional EvoAug configuration
        
    Returns:
        List of training results
    """
    num_gpus = torch.cuda.device_count()
    
    # Prepare arguments for each model
    worker_args = []
    for i in range(n_models):
        gpu_id = i % num_gpus  # Distribute models across available GPUs
        args_tuple = (
            model_type, train_data_path, val_data_path, test_data_path, output_dir,
            i, config, gpu_id, evoaug_config
        )
        worker_args.append(args_tuple)
    
    # Train models in parallel
    results = []
    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
        # Submit all training jobs
        future_to_model = {
            executor.submit(train_single_model_worker, args): args[5]  # args[5] is model_idx
            for args in worker_args
        }
        
        # Collect results as they complete
        for future in tqdm(as_completed(future_to_model), 
                          total=len(future_to_model), 
                          desc="Training models"):
            model_idx = future_to_model[future]
            try:
                result = future.result()
                results.append(result)
                print(f"Model {model_idx + 1} completed: {result['status']}")
            except Exception as e:
                print(f"Model {model_idx + 1} failed: {e}")
                results.append({
                    'model_idx': model_idx,
                    'status': 'failed',
                    'error': str(e)
                })
    
    # Sort results by model index
    results.sort(key=lambda x: x['model_idx'])
    return results


def main():
    parser = argparse.ArgumentParser(description="Train ensemble of DREAM-RNN models")
    
    # Model configuration
    parser.add_argument("--model_type", default="dream_rnn", 
                       help="Type of model to train")
    parser.add_argument("--train_data", required=True,
                       help="Path to training data (TSV format)")
    parser.add_argument("--val_data", required=True,
                       help="Path to validation data (TSV format)")
    parser.add_argument("--test_data", required=True,
                       help="Path to test data (TSV format)")
    parser.add_argument("--output_dir", required=True,
                       help="Directory to save trained models")
    
    # Training configuration
    parser.add_argument("--n_models", type=int, default=5,
                       help="Number of models in ensemble")
    parser.add_argument("--epochs", type=int, default=80,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1024,
                       help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.005,
                       help="Learning rate")
    parser.add_argument("--base_seed", type=int, default=42,
                       help="Base random seed (each model gets base_seed + model_idx)")
    
    # System configuration
    parser.add_argument("--device", default="auto",
                       help="Device to train on (auto, cpu, cuda:0, etc.)")
    parser.add_argument("--n_workers", type=int, default=16,
                       help="Number of data loading workers")
    parser.add_argument("--parallel", action="store_true",
                       help="Train models in parallel across multiple GPUs")
    parser.add_argument("--max_parallel", type=int, default=5,
                       help="Maximum number of models to train in parallel (default: 5 for ensemble)")
    
    # Model-specific hyperparameters
    parser.add_argument("--seqsize", type=int, default=249,
                       help="Sequence length")
    parser.add_argument("--in_channels", type=int, default=5,
                       help="Number of input channels")
    parser.add_argument("--out_channels", type=int, default=320,
                       help="Number of output channels")
    parser.add_argument("--lstm_hidden_channels", type=int, default=320,
                       help="LSTM hidden size")
    parser.add_argument("--kernel_sizes", nargs='+', type=int, default=[9, 15],
                       help="Convolution kernel sizes (space-separated)")
    parser.add_argument("--pool_size", type=int, default=1,
                       help="MaxPool size after first conv block (1 = no pooling)")
    parser.add_argument("--dropout1", type=float, default=0.2,
                       help="Dropout after conv and in LSTM (first block)")
    parser.add_argument("--dropout2", type=float, default=0.5,
                       help="Dropout before fully connected layers (final block)")
    
    # EvoAug configuration
    parser.add_argument("--evoaug-config", help="Path to EvoAug config JSON file")
    parser.add_argument("--evoaug-enabled", action="store_true", help="Enable EvoAug augmentations")
    parser.add_argument("--evoaug-max-augs", type=int, default=2, help="Maximum augmentations per sequence")
    parser.add_argument("--evoaug-mode", default="hard", choices=["hard", "soft"], help="EvoAug mode")
    
    args = parser.parse_args()
    
    # Determine device - GPU REQUIRED for training
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"✅ CUDA available! Using GPU: {torch.cuda.get_device_name()}")
            print(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            print(f"   CUDA version: {torch.version.cuda}")
        else:
            print("❌ CUDA not available!")
            print("   GPU training is required for this project.")
            print("   Please run: python scripts/setup_gpu.py")
            print("   Or update your NVIDIA drivers and reinstall PyTorch.")
            sys.exit(1)
    else:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            print("❌ CUDA requested but not available!")
            print("   Please run: python scripts/setup_gpu.py")
            sys.exit(1)
        elif device.type == "cpu":
            print("❌ CPU training is not supported!")
            print("   GPU training is required for this project.")
            print("   Please use --device auto or --device cuda")
            sys.exit(1)
    
    print(f"🚀 Using device: {device}")
    
    # Process EvoAug configuration
    evoaug_config = None
    if args.evoaug_config:
        with open(args.evoaug_config, 'r') as f:
            evoaug_config = json.load(f)
    elif args.evoaug_enabled:
        # Create default EvoAug config
        evoaug_config = {
            "enabled": True,
            "augmentations": {
                "mutation": {"enabled": True, "rate": 0.1},
                "translocation": {"enabled": True, "shift": 0.1},
                "deletion": {"enabled": True, "length": 0.05}
            },
            "max_augs_per_sequence": args.evoaug_max_augs,
            "mode": args.evoaug_mode
        }
    
    # Note: DREAMRNNTrainer now uses canonical AdamW+OneCycle training by default
    # The canonical base path models/oracles/{dataset}/dream_rnn/ represents this standard training
    # Non-canonical training configs should be placed in variant subdirectories
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    if evoaug_config and evoaug_config.get('enabled', False):
        print(f"🔬 EvoAug enabled: {evoaug_config}")
        # Build EvoAug signature for directory naming
        evoaug_sig = _build_evoaug_signature_simple(evoaug_config)
        if evoaug_sig:
            # Create EvoAug subdirectory within the specified output_dir
            output_path = Path(args.output_dir)
            new_output_dir = output_path / evoaug_sig
            args.output_dir = str(new_output_dir)
            os.makedirs(args.output_dir, exist_ok=True)
            print(f"📁 Updated output directory: {args.output_dir}")
    
    # Build configuration
    config = {
        'n_models': args.n_models,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'base_seed': args.base_seed,
        'device': device,
        'output_dir': args.output_dir,
        'n_workers': args.n_workers,
        'seqsize': args.seqsize,
        'in_channels': args.in_channels,
        'out_channels': args.out_channels,
        'lstm_hidden_channels': args.lstm_hidden_channels,
        'kernel_sizes': args.kernel_sizes,
        'pool_size': args.pool_size,
        'dropout1': args.dropout1,
        'dropout2': args.dropout2,
    }
    
    # Save configuration
    config_path = os.path.join(args.output_dir, "training_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2, default=str)
    
    print(f"Training {args.n_models} {args.model_type} models...")
    print(f"Configuration saved to {config_path}")
    
    # Determine parallel training strategy
    if args.parallel and torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        max_parallel = min(args.max_parallel, num_gpus)  # Use min to avoid exceeding available GPUs
        print(f"Parallel training enabled: {num_gpus} GPUs available, training {max_parallel} models in parallel")
        
        # Train ensemble in parallel
        results = train_ensemble_parallel(
            args.model_type,
            args.train_data,
            args.val_data,
            args.test_data,
            args.output_dir,
            args.n_models,
            config,
            max_parallel,
            evoaug_config
        )
    else:
        # Train ensemble sequentially
        results = []
        for i in range(args.n_models):
            result = train_single_model(
                args.model_type,
                args.train_data,
                args.val_data,
                args.test_data,
                args.output_dir,
                i,
                config,
                device,
                evoaug_config
            )
            results.append(result)
    
    # Save training results
    results_path = os.path.join(args.output_dir, "training_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Print summary
    completed = sum(1 for r in results if r['status'] == 'completed')
    failed = sum(1 for r in results if r['status'] == 'failed')
    
    print(f"\nTraining complete!")
    print(f"  Completed: {completed}/{args.n_models}")
    print(f"  Failed: {failed}/{args.n_models}")
    print(f"  Results saved to {results_path}")
    
    if failed > 0:
        print("\nFailed models:")
        for result in results:
            if result['status'] == 'failed':
                print(f"  Model {result['model_idx']}: {result['error']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
