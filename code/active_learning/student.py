"""
Student model module for active learning framework.

Implements the DeepSTARR architecture as specified in the DeepSTARR paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Generator, List
import numpy as np


class DeepSTARRStudent(nn.Module):
    """
    DeepSTARR student model implementing the exact architecture from the paper.
    
    Architecture:
    - Input: (batch, 249, 4) - one-hot encoded DNA
    - Conv1D (256 filters, kernel=7) + BatchNorm + ReLU + MaxPool(2)
    - Conv1D (60 filters, kernel=3) + BatchNorm + ReLU + MaxPool(2)  
    - Conv1D (60 filters, kernel=5) + BatchNorm + ReLU + MaxPool(2)
    - Conv1D (120 filters, kernel=3) + BatchNorm + ReLU + MaxPool(2)
    - Flatten
    - Dense(256) + BatchNorm + ReLU + Dropout(0.4)
    - Dense(256) + BatchNorm + ReLU + Dropout(0.4)
    - Dense(1) for Dev + Dense(1) for Hk
    """
    
    def __init__(
        self,
        seqsize: int = 249,
        in_channels: int = 4,  # Standard one-hot encoding
        generator: Optional[Generator] = None
    ):
        """
        Initialize DeepSTARR model.
        
        Args:
            seqsize: Length of input sequences
            in_channels: Number of input channels (4 for one-hot)
            generator: Random number generator for weight initialization
        """
        super().__init__()
        self.seqsize = seqsize
        self.in_channels = in_channels
        
        # First convolutional block
        self.conv1 = nn.Conv1d(in_channels, 256, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(256)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Second convolutional block
        self.conv2 = nn.Conv1d(256, 60, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(60)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Third convolutional block
        self.conv3 = nn.Conv1d(60, 60, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(60)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Fourth convolutional block
        self.conv4 = nn.Conv1d(60, 120, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(120)
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Calculate flattened size after convolutions
        # 249 -> 124 -> 62 -> 31 -> 15 (after 4 max pools with stride 2)
        self.flattened_size = 120 * 15  # 1800
        
        # Dense layers
        self.dense1 = nn.Linear(self.flattened_size, 256)
        self.bn5 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.4)
        
        self.dense2 = nn.Linear(256, 256)
        self.bn6 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.4)
        
        # Output heads
        self.dev_head = nn.Linear(256, 1)
        self.hk_head = nn.Linear(256, 1)
        
        # Initialize weights
        if generator is not None:
            self._initialize_weights(generator)
    
    def _initialize_weights(self, generator: Generator):
        """Initialize model weights using Xavier uniform initialization."""
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                # Use generator if available (PyTorch 1.12+), otherwise use manual seeding
                try:
                    nn.init.xavier_uniform_(module.weight, generator=generator)
                except TypeError:
                    # Fallback for older PyTorch versions
                    nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, channels, seq_len) from DREAMRNNDataset
            
        Returns:
            Tuple of (dev_prediction, hk_prediction)
        """
        # Handle Prix Fixe format: (batch_size, 5, seq_len) -> (batch_size, 4, seq_len)
        # Remove the reverse complement indicator (last channel) for DeepSTARR
        if x.shape[1] == 5:
            x = x[:, :4, :]  # Keep only A,C,G,T channels
        
        # First conv block
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        # Second conv block
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        # Third conv block
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        # Fourth conv block
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Dense layers
        x = F.relu(self.bn5(self.dense1(x)))
        x = self.dropout1(x)
        
        x = F.relu(self.bn6(self.dense2(x)))
        x = self.dropout2(x)
        
        # Output heads
        dev_pred = self.dev_head(x)
        hk_pred = self.hk_head(x)
        
        return dev_pred, hk_pred


class DeepSTARRTrainer:
    """Trainer for DeepSTARR student model."""
    
    def __init__(
        self,
        model: DeepSTARRStudent,
        device: torch.device,
        model_dir: str,
        train_data_path: str = None,
        val_data_path: str = None,
        val_sequences: List[str] = None,
        val_labels: np.ndarray = None,
        num_epochs: int = 50,
        lr: float = 0.001,
        weight_decay: float = 1e-6,
        batch_size: int = 32,
        n_workers: int = 4,
        early_stopping: bool = True,
        early_stopping_patience: int = 10,
        lr_scheduler: str = "reduce_on_plateau",
        lr_factor: float = 0.2,
        lr_patience: int = 5
    ):
        """
        Initialize trainer.
        
        Args:
            model: DeepSTARR model to train
            device: Device to train on
            model_dir: Directory to save model
            train_data_path: Path to training data (TSV)
            val_data_path: Path to validation data (TSV) - optional if val_sequences/val_labels provided
            val_sequences: Validation sequences (alternative to val_data_path)
            val_labels: Validation labels (alternative to val_data_path)
            num_epochs: Number of training epochs
            lr: Learning rate
            weight_decay: Weight decay for optimizer
            batch_size: Batch size
            n_workers: Number of data loading workers
            early_stopping: Whether to use early stopping
            early_stopping_patience: Patience for early stopping
            lr_scheduler: Type of learning rate scheduler
            lr_factor: Factor for ReduceLROnPlateau
            lr_patience: Patience for ReduceLROnPlateau
        """
        self.model = model.to(device)
        self.device = device
        self.model_dir = model_dir
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.n_workers = n_workers
        
        # Early stopping parameters
        self.early_stopping = early_stopping
        self.early_stopping_patience = early_stopping_patience
        self.epochs_without_improvement = 0
        
        # Create model directory
        import os
        os.makedirs(model_dir, exist_ok=True)
        
        # Setup data loaders
        from ..prixfixe import DREAMRNNDataset
        from torch.utils.data import DataLoader
        
        # Training data loader (always from file)
        if train_data_path is None:
            raise ValueError("train_data_path is required")
        self.train_dataset = DREAMRNNDataset(train_data_path, model.seqsize)
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=n_workers,
            pin_memory=True
        )
        
        # Validation data loader (from file OR from sequences/labels)
        if val_sequences is not None and val_labels is not None:
            # Use provided validation data
            from ..prixfixe import DREAMRNNInMemoryDataset
            self.val_dataset = DREAMRNNInMemoryDataset(val_sequences, val_labels, model.seqsize)
        elif val_data_path is not None:
            # Use validation file
            self.val_dataset = DREAMRNNDataset(val_data_path, model.seqsize)
        else:
            raise ValueError("Either val_data_path or (val_sequences and val_labels) must be provided")
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=n_workers,
            pin_memory=True
        )
        
        # Setup optimizer and loss
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = nn.MSELoss()
        
        # Setup learning rate scheduler
        self.lr_scheduler_type = lr_scheduler
        if lr_scheduler == "reduce_on_plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=lr_factor,
                patience=lr_patience,
                verbose=True
            )
        else:
            self.scheduler = None
        
        # Training state
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.epoch = 0
        self.history = {'train_loss': [], 'val_loss': []}
    
    def fit(self):
        """Train the model."""
        print(f"Training DeepSTARR for {self.num_epochs} epochs...")
        
        for epoch in range(self.num_epochs):
            self.epoch = epoch
            
            # Train
            train_loss = self._train_epoch()
            
            # Validate
            val_loss = self._validate_epoch()
            
            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch + 1
                self.epochs_without_improvement = 0
                self._save_model('model_best_MSE.pth')
            else:
                self.epochs_without_improvement += 1
            
            # Update learning rate scheduler
            if self.scheduler is not None:
                self.scheduler.step(val_loss)
            
            print(f"Epoch {epoch+1}/{self.num_epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Early stopping check
            if self.early_stopping and self.epochs_without_improvement >= self.early_stopping_patience:
                print(f"Early stopping triggered after {epoch+1} epochs (no improvement for {self.early_stopping_patience} epochs)")
                break
        
        print(f"Training completed. Best validation loss: {self.best_val_loss:.4f} at epoch {self.best_epoch}")
    
    def _train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        from tqdm import tqdm
        for batch_x, (dev_target, hk_target) in tqdm(self.train_loader, desc="Training", leave=False):
            batch_x = batch_x.to(self.device)
            dev_target = dev_target.to(self.device)
            hk_target = hk_target.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            dev_pred, hk_pred = self.model(batch_x)
            
            # Calculate loss
            dev_loss = self.criterion(dev_pred.squeeze(), dev_target)
            hk_loss = self.criterion(hk_pred.squeeze(), hk_target)
            loss = dev_loss + hk_loss
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def _validate_epoch(self) -> float:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            from tqdm import tqdm
            for batch_x, (dev_target, hk_target) in tqdm(self.val_loader, desc="Validation", leave=False):
                batch_x = batch_x.to(self.device)
                dev_target = dev_target.to(self.device)
                hk_target = hk_target.to(self.device)
                
                # Forward pass
                dev_pred, hk_pred = self.model(batch_x)
                
                # Calculate loss
                dev_loss = self.criterion(dev_pred.squeeze(), dev_target)
                hk_loss = self.criterion(hk_pred.squeeze(), hk_target)
                loss = dev_loss + hk_loss
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def _save_model(self, filename: str):
        """Save model state dict."""
        import os
        model_path = os.path.join(self.model_dir, filename)
        torch.save(self.model.state_dict(), model_path)


