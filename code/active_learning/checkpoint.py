"""
Checkpoint management for active learning rounds.

Handles saving and loading of model weights, metrics, and training data state.
"""

import json
import shutil
import numpy as np
from pathlib import Path
from typing import Dict, Any, List


class CheckpointManager:
    """
    Manages checkpoints for active learning rounds.
    
    Each round checkpoint includes:
    - Model weights (model_best.pth)
    - Training metrics (metrics.json)
    - Training data state (training_data.json)
    """
    
    def save_round_checkpoint(
        self,
        round_num: int,
        run_dir: Path,
        model_path: str,
        metrics: Dict[str, Any],
        training_sequences: List[str],
        training_labels: np.ndarray
    ):
        """
        Save complete checkpoint for a round.
        
        Args:
            round_num: Round number (0 for baseline, 1-N for AL cycles)
            run_dir: Run directory
            model_path: Path to trained model file
            metrics: Training and evaluation metrics
            training_sequences: All training sequences up to this round
            training_labels: All training labels up to this round
        """
        round_dir = run_dir / f"round_{round_num:03d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy model weights
        model_src = Path(model_path)
        if model_src.exists():
            shutil.copy(model_src, round_dir / "model_best.pth")
        else:
            print(f"Warning: Model file {model_path} not found, skipping copy")
        
        # Save metrics
        with open(round_dir / "metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save training data state
        training_data = {
            'sequences': training_sequences,
            'labels': training_labels.tolist() if isinstance(training_labels, np.ndarray) else training_labels,
            'n_sequences': len(training_sequences),
            'round': round_num
        }
        
        with open(round_dir / "training_data.json", 'w') as f:
            json.dump(training_data, f, indent=2)
        
        print(f"Checkpoint saved for round {round_num} at {round_dir}")
    
    def load_round_checkpoint(self, round_num: int, run_dir: Path) -> Dict[str, Any]:
        """
        Load checkpoint from a specific round.
        
        Args:
            round_num: Round number to load
            run_dir: Run directory
            
        Returns:
            Dictionary containing checkpoint data:
                - model_path: Path to model weights
                - metrics: Training and evaluation metrics
                - training_sequences: Training sequences
                - training_labels: Training labels
                - round_num: Round number
        """
        round_dir = run_dir / f"round_{round_num:03d}"
        
        if not round_dir.exists():
            raise FileNotFoundError(f"Round directory {round_dir} does not exist")
        
        # Load metrics
        with open(round_dir / "metrics.json") as f:
            metrics = json.load(f)
        
        # Load training data
        with open(round_dir / "training_data.json") as f:
            training_data = json.load(f)
        
        return {
            'model_path': str(round_dir / "model_best.pth"),
            'metrics': metrics,
            'training_sequences': training_data['sequences'],
            'training_labels': np.array(training_data['labels']),
            'round_num': round_num
        }
    
    def checkpoint_exists(self, round_num: int, run_dir: Path) -> bool:
        """
        Check if a complete checkpoint exists for a round.
        
        Args:
            round_num: Round number to check
            run_dir: Run directory
            
        Returns:
            True if checkpoint exists and is complete
        """
        round_dir = run_dir / f"round_{round_num:03d}"
        
        if not round_dir.exists():
            return False
        
        required_files = [
            round_dir / "model_best.pth",
            round_dir / "metrics.json",
            round_dir / "training_data.json"
        ]
        
        return all(f.exists() for f in required_files)

