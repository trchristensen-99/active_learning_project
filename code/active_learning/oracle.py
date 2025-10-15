"""
Oracle module for active learning framework.

Provides oracle models that can label sequences for training student models.
"""

import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
import os
from pathlib import Path

from ..models import build_model
from ..utils import one_hot_encode, sequence_to_tensor


class BaseOracle(ABC):
    """Abstract base class for oracle models."""
    
    @abstractmethod
    def predict(self, sequences: List[str]) -> np.ndarray:
        """
        Predict regulatory activity for sequences.
        
        Args:
            sequences: List of DNA sequences
            
        Returns:
            Array of predictions (n_sequences, n_tasks)
        """
        pass
    
    @abstractmethod
    def get_uncertainty(self, sequences: List[str]) -> np.ndarray:
        """
        Get uncertainty estimates for sequences.
        
        Args:
            sequences: List of DNA sequences
            
        Returns:
            Array of uncertainty estimates (n_sequences,)
        """
        pass


class EnsembleOracle(BaseOracle):
    """
    Oracle ensemble that aggregates predictions from multiple models.
    
    Uses the trained DREAM-RNN ensemble to provide labels and uncertainty estimates.
    """
    
    def __init__(
        self,
        model_dir: str,
        model_type: str = "dream_rnn",
        device: str = "auto",
        seqsize: int = 249,
        batch_size: int = 32
    ):
        """
        Initialize ensemble oracle.
        
        Args:
            model_dir: Directory containing trained ensemble models
            model_type: Type of model to load
            device: Device to run inference on
            seqsize: Sequence length
            batch_size: Batch size for processing sequences
        """
        self.model_dir = Path(model_dir)
        self.model_type = model_type
        self.seqsize = seqsize
        self.batch_size = batch_size
        
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Load ensemble models
        self.models = self._load_ensemble()
        print(f"Loaded {len(self.models)} models for oracle ensemble")
    
    def _load_ensemble(self) -> List[nn.Module]:
        """Load all models in the ensemble."""
        models = []
        
        # Find all model directories
        model_dirs = [d for d in self.model_dir.iterdir() if d.is_dir() and d.name.startswith("model_")]
        model_dirs.sort(key=lambda x: int(x.name.split("_")[1]))
        
        for model_dir in model_dirs:
            model_path = model_dir / "model_best_MSE.pth"
            if model_path.exists():
                # Build model
                model = build_model(
                    self.model_type,
                    seqsize=self.seqsize,
                    in_channels=5,  # Prix Fixe encoding
                    generator=torch.Generator().manual_seed(42)
                )
                
                # Load weights
                state_dict = torch.load(model_path, map_location=self.device)
                model.load_state_dict(state_dict)
                model.to(self.device)
                model.eval()
                
                models.append(model)
            else:
                print(f"Warning: Model weights not found at {model_path}")
        
        if not models:
            raise ValueError(f"No trained models found in {self.model_dir}")
        
        return models
    
    def predict(self, sequences: List[str], chunk_size: int = None) -> np.ndarray:
        """
        Predict regulatory activity using ensemble with chunked processing for memory efficiency.
        
        Args:
            sequences: List of DNA sequences
            chunk_size: Batch size for processing (auto-determined if None)
            
        Returns:
            Array of predictions (n_sequences, 2) - [dev_activity, hk_activity]
        """
        if chunk_size is None:
            chunk_size = self.batch_size
        
        n_sequences = len(sequences)
        all_predictions = []
        
        # Process sequences in chunks to avoid memory issues
        for i in range(0, n_sequences, chunk_size):
            chunk_sequences = sequences[i:i + chunk_size]
            
            # Encode chunk
            encoded_seqs = []
            for seq in chunk_sequences:
                # Convert to Prix Fixe format (5 channels: A,C,G,T,rev)
                encoded = sequence_to_tensor(seq, self.seqsize, self.device)
                encoded_seqs.append(encoded)
            
            # Stack into batch and squeeze extra dimension
            batch = torch.stack(encoded_seqs).squeeze(1).to(self.device)
            
            # Get predictions from all models for this chunk
            chunk_predictions = []
            with torch.no_grad():
                for model in self.models:
                    dev_pred, hk_pred = model(batch)
                    pred = torch.cat([dev_pred, hk_pred], dim=1)  # (batch, 2)
                    chunk_predictions.append(pred.cpu().numpy())
            
            # Average predictions across ensemble for this chunk
            chunk_ensemble_pred = np.mean(chunk_predictions, axis=0)
            all_predictions.append(chunk_ensemble_pred)
            
            # Clear GPU memory
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        
        # Combine all chunk predictions
        ensemble_pred = np.vstack(all_predictions)
        return ensemble_pred
    
    def get_uncertainty(self, sequences: List[str], chunk_size: int = None) -> np.ndarray:
        """
        Get uncertainty estimates using ensemble variance with chunked processing.
        
        Args:
            sequences: List of DNA sequences
            chunk_size: Batch size for processing (auto-determined if None)
            
        Returns:
            Array of uncertainty estimates (n_sequences,)
        """
        if chunk_size is None:
            chunk_size = self.batch_size
        
        n_sequences = len(sequences)
        all_uncertainties = []
        
        # Process sequences in chunks to avoid memory issues
        for i in range(0, n_sequences, chunk_size):
            chunk_sequences = sequences[i:i + chunk_size]
            
            # Encode chunk
            encoded_seqs = []
            for seq in chunk_sequences:
                encoded = sequence_to_tensor(seq, self.seqsize, self.device)
                encoded_seqs.append(encoded)
            
            # Stack into batch and squeeze extra dimension
            batch = torch.stack(encoded_seqs).squeeze(1).to(self.device)
            
            # Get predictions from all models for this chunk
            chunk_predictions = []
            with torch.no_grad():
                for model in self.models:
                    dev_pred, hk_pred = model(batch)
                    pred = torch.cat([dev_pred, hk_pred], dim=1)  # (batch, 2)
                    chunk_predictions.append(pred.cpu().numpy())
            
            # Calculate uncertainty as standard deviation across ensemble for this chunk
            predictions_array = np.array(chunk_predictions)  # (n_models, batch_size, 2)
            
            # Calculate uncertainty for each task separately, then average
            dev_uncertainty = np.std(predictions_array[:, :, 0], axis=0)
            hk_uncertainty = np.std(predictions_array[:, :, 1], axis=0)
            
            # Average uncertainty across tasks
            chunk_uncertainty = (dev_uncertainty + hk_uncertainty) / 2
            all_uncertainties.append(chunk_uncertainty)
            
            # Clear GPU memory
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        
        # Combine all chunk uncertainties
        total_uncertainty = np.concatenate(all_uncertainties)
        return total_uncertainty
    
    def predict_with_uncertainty(self, sequences: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get both predictions and uncertainty estimates.
        
        Args:
            sequences: List of DNA sequences
            
        Returns:
            Tuple of (predictions, uncertainties)
        """
        predictions = self.predict(sequences)
        uncertainties = self.get_uncertainty(sequences)
        return predictions, uncertainties


