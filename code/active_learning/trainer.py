"""
Active learning trainer module.

Handles training of student models with configurable strategies.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod

from .student import DeepSTARRStudent, DeepSTARRTrainer
from ..utils import one_hot_encode


class BaseActiveLearningTrainer(ABC):
    """Abstract base class for active learning trainers."""
    
    @abstractmethod
    def train_from_scratch(
        self,
        train_sequences: List[str],
        train_labels: np.ndarray,
        val_sequences: List[str],
        val_labels: np.ndarray,
        **kwargs
    ) -> Dict[str, Any]:
        """Train model from scratch."""
        pass
    
    @abstractmethod
    def fine_tune(
        self,
        new_sequences: List[str],
        new_labels: np.ndarray,
        **kwargs
    ) -> Dict[str, Any]:
        """Fine-tune existing model."""
        pass


class DeepSTARRActiveLearningTrainer(BaseActiveLearningTrainer):
    """
    Active learning trainer for DeepSTARR student models.
    
    Supports both training from scratch and continual learning with optional replay.
    """
    
    def __init__(
        self,
        model_dir: str,
        device: str = "auto",
        seqsize: int = 249,
        in_channels: int = 4,
        num_epochs: int = 100,
        lr: float = 0.001,
        weight_decay: float = 1e-6,
        batch_size: int = 32,
        n_workers: int = 4,
        enable_replay: bool = False,
        replay_buffer_size: int = 1000,
        early_stopping: bool = True,
        early_stopping_patience: int = 10,
        lr_scheduler: str = "reduce_on_plateau",
        lr_factor: float = 0.2,
        lr_patience: int = 5,
        finetune_config: Optional[Dict] = None
    ):
        """
        Initialize DeepSTARR active learning trainer.
        
        Args:
            model_dir: Directory to save/load models
            device: Device to train on
            seqsize: Sequence length
            in_channels: Number of input channels
            num_epochs: Number of training epochs
            lr: Learning rate
            batch_size: Batch size
            n_workers: Number of data loading workers
            enable_replay: Whether to use replay buffer for continual learning
            replay_buffer_size: Size of replay buffer
            finetune_config: Configuration for fine-tuning parameters
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.seqsize = seqsize
        self.in_channels = in_channels
        self.num_epochs = num_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.n_workers = n_workers
        
        # Continual learning settings
        self.enable_replay = enable_replay
        self.replay_buffer_size = replay_buffer_size
        self.replay_buffer = []  # List of (sequence, label) tuples

        # Training policy
        self.early_stopping = early_stopping
        self.early_stopping_patience = early_stopping_patience
        self.lr_scheduler = lr_scheduler
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        
        # Fine-tuning configuration
        self.finetune_config = finetune_config
        
        # Model and trainer
        self.model = None
        self.trainer = None
        
        # External validation data (optional)
        self.validation_sequences = None
        self.validation_labels = None
    
    def set_validation_data(self, sequences: List[str], labels: np.ndarray):
        """
        Set external validation data to use for all training runs.
        
        Args:
            sequences: Validation sequences
            labels: Validation labels (n, 2) for [dev, hk]
        """
        self.validation_sequences = sequences
        self.validation_labels = labels
    
    def _create_model(self) -> DeepSTARRStudent:
        """Create a new DeepSTARR model."""
        model = DeepSTARRStudent(
            seqsize=self.seqsize,
            in_channels=self.in_channels,
            generator=torch.Generator().manual_seed(42)
        )
        return model
    
    def _save_data_to_tsv(
        self,
        sequences: List[str],
        labels: np.ndarray,
        filepath: str
    ):
        """Save sequences and labels to TSV file."""
        # Convert sequences to Prix Fixe format for compatibility
        encoded_sequences = []
        for seq in sequences:
            encoded = one_hot_encode(seq)
            encoded_sequences.append(encoded)
        
        # Create DataFrame
        data = {
            'Sequence': sequences,  # Capital S to match DREAMRNNDataset expectations
            'Dev_log2_enrichment': labels[:, 0] if labels.ndim > 1 else labels,
            'Hk_log2_enrichment': labels[:, 1] if labels.ndim > 1 and labels.shape[1] > 1 else labels
        }
        
        df = pd.DataFrame(data)
        df.to_csv(filepath, sep='\t', index=False)
    
    def _load_model(self) -> Optional[DeepSTARRStudent]:
        """Load existing model if available."""
        model_path = self.model_dir / "model_best_MSE.pth"
        if model_path.exists():
            model = self._create_model()
            state_dict = torch.load(model_path, map_location=self.device)
            model.load_state_dict(state_dict)
            return model
        return None
    
    def train_from_scratch(
        self,
        train_sequences: List[str],
        train_labels: np.ndarray,
        val_sequences: List[str] = None,
        val_labels: np.ndarray = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train model from scratch.
        
        Args:
            train_sequences: Training sequences
            train_labels: Training labels (n_sequences, 2) for [dev, hk]
            val_sequences: Validation sequences (optional if external validation set)
            val_labels: Validation labels (optional if external validation set)
            **kwargs: Additional training parameters
            
        Returns:
            Training results dictionary
        """
        print("Training DeepSTARR model from scratch...")
        
        # Create new model
        self.model = self._create_model()
        
        # Use external validation data if set, otherwise use provided
        if self.validation_sequences is not None and self.validation_labels is not None:
            val_seqs = self.validation_sequences
            val_labs = self.validation_labels
        elif val_sequences is not None and val_labels is not None:
            val_seqs = val_sequences
            val_labs = val_labels
        else:
            raise ValueError("Validation data must be provided either via set_validation_data() or as arguments")
        
        # Save training data to temporary TSV file
        train_data_path = self.model_dir / "temp_train.tsv"
        self._save_data_to_tsv(train_sequences, train_labels, str(train_data_path))
        
        # Create trainer with external validation data
        self.trainer = DeepSTARRTrainer(
            model=self.model,
            device=self.device,
            model_dir=str(self.model_dir),
            train_data_path=str(train_data_path),
            val_sequences=val_seqs,
            val_labels=val_labs,
            num_epochs=self.num_epochs,
            lr=self.lr,
            weight_decay=self.weight_decay,
            batch_size=self.batch_size,
            n_workers=self.n_workers,
            early_stopping=self.early_stopping,
            early_stopping_patience=self.early_stopping_patience,
            lr_scheduler=self.lr_scheduler,
            lr_factor=self.lr_factor,
            lr_patience=self.lr_patience
        )
        
        # Train model
        self.trainer.fit()
        
        # Clean up temporary files
        train_data_path.unlink(missing_ok=True)
        
        # Update replay buffer if enabled
        if self.enable_replay:
            self._update_replay_buffer(train_sequences, train_labels)
        
        return {
            'training_type': 'from_scratch',
            'best_val_loss': self.trainer.best_val_loss,
            'model_path': str(self.model_dir / "model_best_MSE.pth")
        }
    
    def fine_tune(
        self,
        new_sequences: List[str],
        new_labels: np.ndarray,
        finetune_config: Optional[Dict] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Fine-tune existing model with new data using configurable parameters.
        
        Args:
            new_sequences: New sequences to add
            new_labels: New labels
            finetune_config: Fine-tuning configuration dictionary
            **kwargs: Additional training parameters (for backward compatibility)
            
        Returns:
            Training results dictionary
        """
        print("Fine-tuning DeepSTARR model...")
        
        # Use provided config or fall back to self.finetune_config
        config = finetune_config or self.finetune_config
        if config is None:
            raise ValueError("Fine-tuning requires finetune_config")
        
        # Load existing model
        self.model = self._load_model()
        if self.model is None:
            raise ValueError("No existing model found for fine-tuning")
        
        # Parse configuration
        lr_config = config['learning_rate']
        num_epochs = config['num_epochs']
        replay_config = config['replay_strategy']
        optimizer_config = config.get('optimizer', {})
        early_stopping_config = config.get('early_stopping', {})
        
        # Prepare training data based on replay strategy
        if replay_config['type'] == 'all_data':
            # Use all accumulated data (current behavior)
            if self.enable_replay and self.replay_buffer:
                replay_sequences, replay_labels = zip(*self.replay_buffer)
                all_sequences = list(replay_sequences) + new_sequences
                all_labels = np.vstack([np.array(replay_labels), new_labels])
            else:
                all_sequences = new_sequences
                all_labels = new_labels
                
        elif replay_config['type'] == 'fixed_ratio':
            # Sample old data based on ratio to new data
            ratio = replay_config['old_new_ratio']
            if self.enable_replay and self.replay_buffer:
                replay_sequences, replay_labels = zip(*self.replay_buffer)
                n_old_samples = int(len(new_sequences) * ratio)
                if n_old_samples > 0:
                    # Sample randomly from replay buffer
                    old_indices = np.random.choice(len(replay_sequences), 
                                                min(n_old_samples, len(replay_sequences)), 
                                                replace=False)
                    old_sequences = [replay_sequences[i] for i in old_indices]
                    old_labels = np.array([replay_labels[i] for i in old_indices])
                    all_sequences = old_sequences + new_sequences
                    all_labels = np.vstack([old_labels, new_labels])
                else:
                    all_sequences = new_sequences
                    all_labels = new_labels
            else:
                all_sequences = new_sequences
                all_labels = new_labels
                
        elif replay_config['type'] == 'percentage':
            # Sample fixed % of old data
            percentage = replay_config['percentage']
            if self.enable_replay and self.replay_buffer:
                replay_sequences, replay_labels = zip(*self.replay_buffer)
                n_old_samples = int(len(replay_sequences) * percentage / 100)
                if n_old_samples > 0:
                    # Sample randomly from replay buffer
                    old_indices = np.random.choice(len(replay_sequences), 
                                                min(n_old_samples, len(replay_sequences)), 
                                                replace=False)
                    old_sequences = [replay_sequences[i] for i in old_indices]
                    old_labels = np.array([replay_labels[i] for i in old_indices])
                    all_sequences = old_sequences + new_sequences
                    all_labels = np.vstack([old_labels, new_labels])
                else:
                    all_sequences = new_sequences
                    all_labels = new_labels
            else:
                all_sequences = new_sequences
                all_labels = new_labels
        else:
            raise ValueError(f"Unknown replay strategy: {replay_config['type']}")
        
        # Use external validation data if set, otherwise split train/val
        if self.validation_sequences is not None and self.validation_labels is not None:
            train_sequences = all_sequences
            train_labels = all_labels
            val_seqs = self.validation_sequences
            val_labs = self.validation_labels
        else:
            # Split into train/val (80/20)
            n_total = len(all_sequences)
            n_train = int(0.8 * n_total)
            
            indices = np.random.permutation(n_total)
            train_indices = indices[:n_train]
            val_indices = indices[n_train:]
            
            train_sequences = [all_sequences[i] for i in train_indices]
            train_labels = all_labels[train_indices]
            val_seqs = [all_sequences[i] for i in val_indices]
            val_labs = all_labels[val_indices]
        
        # Save training data to temporary TSV file
        train_data_path = self.model_dir / "temp_finetune_train.tsv"
        self._save_data_to_tsv(train_sequences, train_labels, str(train_data_path))
        
        # Configure learning rate
        if lr_config['type'] == 'fixed':
            finetune_lr = lr_config['value']
            lr_scheduler = None
            lr_factor = None
            lr_patience = None
        elif lr_config['type'] == 'scheduled':
            finetune_lr = lr_config['initial_value']
            lr_scheduler = lr_config['scheduler']
            lr_factor = lr_config['factor']
            lr_patience = lr_config['patience']
        else:
            raise ValueError(f"Unknown learning rate type: {lr_config['type']}")
        
        # Configure early stopping
        early_stopping = early_stopping_config.get('enabled', True)
        early_stopping_patience = early_stopping_config.get('patience', 10)
        
        # Create trainer with configured parameters
        self.trainer = DeepSTARRTrainer(
            model=self.model,
            device=self.device,
            model_dir=str(self.model_dir),
            train_data_path=str(train_data_path),
            val_sequences=val_seqs,
            val_labels=val_labs,
            num_epochs=num_epochs,
            lr=finetune_lr,
            weight_decay=optimizer_config.get('weight_decay', self.weight_decay),
            batch_size=self.batch_size,
            n_workers=self.n_workers,
            early_stopping=early_stopping,
            early_stopping_patience=early_stopping_patience,
            lr_scheduler=lr_scheduler,
            lr_factor=lr_factor,
            lr_patience=lr_patience
        )
        
        # Fine-tune model
        self.trainer.fit()
        
        # Clean up temporary files
        train_data_path.unlink(missing_ok=True)
        
        # Update replay buffer
        if self.enable_replay:
            self._update_replay_buffer(new_sequences, new_labels)
        
        return {
            'training_type': 'fine_tune',
            'best_val_loss': self.trainer.best_val_loss,
            'model_path': str(self.model_dir / "model_best_MSE.pth"),
            'n_new_samples': len(new_sequences),
            'n_replay_samples': len(self.replay_buffer) if self.enable_replay else 0
        }
    
    def _update_replay_buffer(self, sequences: List[str], labels: np.ndarray):
        """Update replay buffer with new data."""
        # Add new data to buffer
        for seq, label in zip(sequences, labels):
            self.replay_buffer.append((seq, label))
        
        # Trim buffer if it exceeds maximum size
        if len(self.replay_buffer) > self.replay_buffer_size:
            # Keep most recent samples
            self.replay_buffer = self.replay_buffer[-self.replay_buffer_size:]
    
    def predict(self, sequences: List[str]) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            sequences: Sequences to predict
            
        Returns:
            Predictions array (n_sequences, 2)
        """
        if self.model is None:
            self.model = self._load_model()
        
        if self.model is None:
            raise ValueError("No trained model available")
        
        self.model.eval()
        
        # Process in batches to avoid OOM
        batch_size = 512
        all_predictions = []
        
        for i in range(0, len(sequences), batch_size):
            batch_sequences = sequences[i:i + batch_size]
            
            # Encode sequences
            encoded_seqs = []
            for seq in batch_sequences:
                encoded = one_hot_encode(seq)
                # Convert list to tensor and add reverse complement indicator
                encoded_tensor = torch.tensor(encoded, dtype=torch.float32)
                # Add reverse complement indicator (0 for forward sequences)
                rev_indicator = torch.zeros(encoded_tensor.shape[0], 1)
                encoded_with_rev = torch.cat([encoded_tensor, rev_indicator], dim=1)
                # Transpose to (channels, sequence) format
                encoded_with_rev = encoded_with_rev.transpose(0, 1)
                encoded_seqs.append(encoded_with_rev)
            
            # Stack into batch
            batch = torch.stack(encoded_seqs).to(self.device)
            
            # Make predictions
            with torch.no_grad():
                dev_pred, hk_pred = self.model(batch)
                predictions = torch.cat([dev_pred, hk_pred], dim=1)
            
            all_predictions.append(predictions.cpu().numpy())
        
        return np.vstack(all_predictions)
    
    def evaluate(self, sequences: List[str], labels: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            sequences: Test sequences
            labels: True labels
            
        Returns:
            Evaluation metrics
        """
        predictions = self.predict(sequences)
        
        # Calculate MSE for each task
        dev_mse = np.mean((predictions[:, 0] - labels[:, 0]) ** 2)
        hk_mse = np.mean((predictions[:, 1] - labels[:, 1]) ** 2)
        total_mse = (dev_mse + hk_mse) / 2
        
        # Calculate Pearson correlation
        from scipy.stats import pearsonr
        dev_corr, _ = pearsonr(predictions[:, 0], labels[:, 0])
        hk_corr, _ = pearsonr(predictions[:, 1], labels[:, 1])
        avg_corr = (dev_corr + hk_corr) / 2
        
        return {
            'dev_mse': dev_mse,
            'hk_mse': hk_mse,
            'total_mse': total_mse,
            'dev_correlation': dev_corr,
            'hk_correlation': hk_corr,
            'avg_correlation': avg_corr
        }


