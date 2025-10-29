#!/usr/bin/env python3
"""
Evaluate trained oracle models on test set.

Uses existing DREAMRNNTrainer.evaluate_test_set() method.
"""

import sys
import json
import argparse
from pathlib import Path
import torch
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from code.models import build_model
from code.prixfixe import DREAMRNNTrainer


def evaluate_model(model_dir: str, test_data_path: str, device: str = "cuda:0"):
    """Evaluate a single trained model."""
    model_dir = Path(model_dir)
    checkpoint_path = model_dir / "model_best_MSE.pth"
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    
    print(f"Evaluating model: {model_dir}")
    
    # Build model (default config for DREAM-RNN)
    model = build_model(
        "dream_rnn",
        seqsize=249,
        in_channels=5,
        generator=torch.Generator().manual_seed(42)
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    
    # Create trainer with actual data paths (train/val only used for initialization, not evaluation)
    # Use the test data path temporarily for train/val to satisfy initialization
    trainer = DREAMRNNTrainer(
        model,
        device=torch.device(device),
        model_dir=str(model_dir),
        train_data_path=test_data_path,  # Will be replaced, only used for dataset init
        val_data_path=test_data_path,    # Will be replaced, only used for dataset init
        num_epochs=1,
        lr=0.001,
        batch_size=1024,
        n_workers=4
    )
    
    # Replace the model with our already-loaded one (since evaluate_test_set tries to reload)
    trainer.model = model.to(device)
    
    # Now evaluate using the test set
    test_results = trainer.evaluate_test_set(test_data_path)
    
    return test_results


def evaluate_ensemble(ensemble_dir: str, test_data_path: str, device: str = "cuda:0"):
    """Evaluate all models in an ensemble."""
    ensemble_dir = Path(ensemble_dir)
    results = []
    
    # Find all model directories
    model_dirs = sorted([d for d in ensemble_dir.iterdir() if d.is_dir() and d.name.startswith("model_")])
    
    if not model_dirs:
        raise ValueError(f"No model directories found in {ensemble_dir}")
    
    print(f"Found {len(model_dirs)} models to evaluate")
    
    for model_dir in model_dirs:
        try:
            result = evaluate_model(str(model_dir), test_data_path, device)
            result['model_dir'] = str(model_dir)
            result['model_idx'] = int(model_dir.name.split('_')[1])
            results.append(result)
            print(f"✅ Model {model_dir.name}: Completed")
        except Exception as e:
            print(f"❌ Model {model_dir.name}: Failed - {e}")
            results.append({
                'model_dir': str(model_dir),
                'model_idx': int(model_dir.name.split('_')[1]),
                'status': 'failed',
                'error': str(e)
            })
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained oracle models")
    parser.add_argument("--model_dir", help="Path to single model directory")
    parser.add_argument("--ensemble_dir", help="Path to ensemble directory")
    parser.add_argument("--test_data", required=True, help="Path to test data (TSV)")
    parser.add_argument("--output", help="Output JSON file (default: evaluation_results.json)")
    parser.add_argument("--device", default="cuda:0", help="Device to use")
    
    args = parser.parse_args()
    
    if not args.model_dir and not args.ensemble_dir:
        parser.error("Must specify either --model_dir or --ensemble_dir")
    
    if args.model_dir and args.ensemble_dir:
        parser.error("Cannot specify both --model_dir and --ensemble_dir")
    
    # Evaluate
    if args.model_dir:
        results = evaluate_model(args.model_dir, args.test_data, args.device)
        output = args.output or Path(args.model_dir) / "evaluation_results.json"
        with open(output, 'w') as f:
            json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.float32, np.float64, np.int64)) else str(x))
        print(f"\nResults saved to: {output}")
    else:
        results = evaluate_ensemble(args.ensemble_dir, args.test_data, args.device)
        output = args.output or Path(args.ensemble_dir) / "evaluation_results.json"
        with open(output, 'w') as f:
            json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.float32, np.float64, np.int64)) else str(x))
        print(f"\nResults saved to: {output}")
        
        # Print summary
        completed = [r for r in results if 'status' not in r or r['status'] != 'failed']
        print(f"\nSummary: {len(completed)}/{len(results)} models evaluated successfully")


if __name__ == "__main__":
    main()

