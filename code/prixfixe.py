"""
Streamlined Prix Fixe implementation for DREAM-RNN.

This module contains only the components needed for the DREAM-RNN model,
simplified from the original Prix Fixe framework.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
# TensorBoard removed - using simple console logging instead
class SummaryWriter:
    """Dummy SummaryWriter for compatibility - logging handled by console output."""
    def __init__(self, *args, **kwargs):
        pass
    def __getattr__(self, name):
        return lambda *args, **kwargs: None
from typing import Optional, Generator, Tuple
import pandas as pd
import numpy as np
import os
from tqdm import tqdm


class DREAMRNN(nn.Module):
    """
    DREAM-RNN model for genomic sequence analysis.
    
    Architecture:
    1. First Block: Convolutional layers with multiple kernel sizes
    2. Core Block: Bidirectional LSTM + convolutional layers
    3. Final Block: Global pooling + fully connected layers for two tasks
    """
    
    def __init__(
        self,
        seqsize: int = 249,
        in_channels: int = 5,
        out_channels: int = 320,
        lstm_hidden_channels: int = 320,
        kernel_sizes: list = [9, 15],
        pool_size: int = 1,
        dropout1: float = 0.2,
        dropout2: float = 0.5,
        generator: Optional[Generator] = None
    ):
        super().__init__()
        self.seqsize = seqsize
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lstm_hidden_channels = lstm_hidden_channels
        
        # First Block: Initial convolutional processing
        self.first_conv_layers = nn.ModuleList()
        for kernel_size in kernel_sizes:
            conv = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
                nn.ReLU(),
                nn.Dropout(dropout1)
            )
            self.first_conv_layers.append(conv)
        
        # Pooling after first block
        if pool_size > 1:
            self.first_pool = nn.MaxPool1d(pool_size)
            first_out_seqsize = seqsize // pool_size
        else:
            self.first_pool = nn.Identity()
            first_out_seqsize = seqsize
        
        # Core Block: Bidirectional LSTM + convolutional layers
        self.lstm = nn.LSTM(
            input_size=out_channels * len(kernel_sizes),
            hidden_size=lstm_hidden_channels,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=dropout1
        )
        
        self.core_conv_layers = nn.ModuleList()
        for kernel_size in kernel_sizes:
            conv = nn.Sequential(
                nn.Conv1d(lstm_hidden_channels * 2, out_channels, kernel_size, padding=kernel_size//2),
                nn.ReLU(),
                nn.Dropout(dropout2)
            )
            self.core_conv_layers.append(conv)
        
        # Pooling after core block
        if pool_size > 1:
            self.core_pool = nn.MaxPool1d(pool_size)
            core_out_seqsize = first_out_seqsize // pool_size
        else:
            self.core_pool = nn.Identity()
            core_out_seqsize = first_out_seqsize
        
        # Final Block: Global pooling + fully connected layers
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        final_in_channels = out_channels * len(kernel_sizes)
        self.fc_layers = nn.Sequential(
            nn.Linear(final_in_channels, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        # Task-specific output heads
        self.dev_head = nn.Linear(64, 1)
        self.hk_head = nn.Linear(64, 1)
        
        # Initialize weights
        if generator is not None:
            self._initialize_weights(generator)
    
    def _initialize_weights(self, generator: Generator):
        """Initialize model weights with given generator."""
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
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        try:
                            nn.init.xavier_uniform_(param, generator=generator)
                        except TypeError:
                            nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, in_channels, seq_len)
            
        Returns:
            Tuple of (dev_prediction, hk_prediction)
        """
        # First Block: Convolutional layers
        conv_outputs = []
        for conv_layer in self.first_conv_layers:
            conv_out = conv_layer(x)
            conv_outputs.append(conv_out)
        
        x = torch.cat(conv_outputs, dim=1)  # Concatenate different kernel sizes
        x = self.first_pool(x)
        
        # Core Block: LSTM + Convolutional layers
        batch_size, channels, seq_len = x.shape
        
        # Reshape for LSTM: (batch_size, seq_len, channels)
        x_lstm = x.transpose(1, 2)
        
        # Apply bidirectional LSTM
        lstm_out, _ = self.lstm(x_lstm)  # (batch_size, seq_len, lstm_hidden_channels * 2)
        
        # Reshape back for conv: (batch_size, lstm_hidden_channels * 2, seq_len)
        x_conv = lstm_out.transpose(1, 2)
        
        # Apply convolutional layers
        conv_outputs = []
        for conv_layer in self.core_conv_layers:
            conv_out = conv_layer(x_conv)
            conv_outputs.append(conv_out)
        
        x = torch.cat(conv_outputs, dim=1)  # Concatenate different kernel sizes
        x = self.core_pool(x)
        
        # Final Block: Global pooling + fully connected layers
        x = self.global_pool(x)  # (batch_size, channels, 1)
        x = x.squeeze(-1)  # (batch_size, channels)
        
        # Shared feature extraction
        features = self.fc_layers(x)
        
        # Task-specific predictions
        dev_pred = self.dev_head(features)
        hk_pred = self.hk_head(features)
        
        return dev_pred, hk_pred


class DREAMRNNDataset(Dataset):
    """Dataset for DREAM-RNN training."""
    
    def __init__(self, data_path: str, seqsize: int = 249):
        self.data = pd.read_csv(data_path, sep='\t')
        self.seqsize = seqsize
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Get sequence and encode
        sequence = row['Sequence']
        if len(sequence) != self.seqsize:
            # Pad or truncate to seqsize
            if len(sequence) > self.seqsize:
                sequence = sequence[:self.seqsize]
            else:
                sequence = sequence + 'N' * (self.seqsize - len(sequence))
        
        # One-hot encode sequence
        encoded_seq = self._one_hot_encode(sequence)
        
        # Add reverse complement indicator
        rev_indicator = row.get('rev', 0)
        encoded_seq_with_rev = np.concatenate([encoded_seq, [[rev_indicator]] * self.seqsize], axis=1)
        
        # Convert to tensor
        x = torch.tensor(encoded_seq_with_rev.transpose(1, 0), dtype=torch.float32)
        
        # Get targets
        dev_target = torch.tensor(row['Dev_log2_enrichment'], dtype=torch.float32)
        hk_target = torch.tensor(row['Hk_log2_enrichment'], dtype=torch.float32)
        
        return x, (dev_target, hk_target)
    
    def _one_hot_encode(self, seq: str) -> np.ndarray:
        """One-hot encode DNA sequence."""
        mapping = {'A': [1, 0, 0, 0], 'G': [0, 1, 0, 0], 'C': [0, 0, 1, 0], 'T': [0, 0, 0, 1], 'N': [0, 0, 0, 0]}
        return np.array([mapping.get(base.upper(), [0, 0, 0, 0]) for base in seq])


class DREAMRNNInMemoryDataset(Dataset):
    """In-memory dataset for DREAM-RNN training with pre-loaded sequences and labels."""
    
    def __init__(self, sequences: list, labels: np.ndarray, seqsize: int = 249):
        """
        Initialize in-memory dataset.
        
        Args:
            sequences: List of DNA sequences
            labels: Array of shape (n, 2) with dev and hk activities
            seqsize: Sequence length
        """
        self.sequences = sequences
        self.labels = labels
        self.seqsize = seqsize
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        # Get sequence and encode
        sequence = self.sequences[idx]
        if len(sequence) != self.seqsize:
            # Pad or truncate to seqsize
            if len(sequence) > self.seqsize:
                sequence = sequence[:self.seqsize]
            else:
                sequence = sequence + 'N' * (self.seqsize - len(sequence))
        
        # One-hot encode sequence
        encoded_seq = self._one_hot_encode(sequence)
        
        # Add reverse complement indicator (0 for non-genomic sequences)
        rev_indicator = 0
        encoded_seq_with_rev = np.concatenate([encoded_seq, [[rev_indicator]] * self.seqsize], axis=1)
        
        # Convert to tensor
        x = torch.tensor(encoded_seq_with_rev.transpose(1, 0), dtype=torch.float32)
        
        # Get targets
        dev_target = torch.tensor(self.labels[idx, 0], dtype=torch.float32)
        hk_target = torch.tensor(self.labels[idx, 1], dtype=torch.float32)
        
        return x, (dev_target, hk_target)
    
    def _one_hot_encode(self, seq: str) -> np.ndarray:
        """One-hot encode DNA sequence."""
        mapping = {'A': [1, 0, 0, 0], 'G': [0, 1, 0, 0], 'C': [0, 0, 1, 0], 'T': [0, 0, 0, 1], 'N': [0, 0, 0, 0]}
        return np.array([mapping.get(base.upper(), [0, 0, 0, 0]) for base in seq])


class DREAMRNNTrainer:
    """Trainer for DREAM-RNN model."""
    
    def __init__(
        self,
        model: DREAMRNN,
        device: torch.device,
        model_dir: str,
        train_data_path: str,
        val_data_path: str,
        num_epochs: int = 80,
        lr: float = 0.005,
        batch_size: int = 32,
        weight_decay: float = 1e-4,
        n_workers: int = 4
    ):
        self.model = model.to(device)
        self.device = device
        self.model_dir = model_dir
        self.num_epochs = num_epochs
        self.lr = lr
        self.batch_size = batch_size
        
        # Create model directory
        os.makedirs(model_dir, exist_ok=True)
        
        # Setup data loaders
        self.train_dataset = DREAMRNNDataset(train_data_path, model.seqsize)
        self.val_dataset = DREAMRNNDataset(val_data_path, model.seqsize)
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=n_workers,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=n_workers,
            pin_memory=True
        )
        
        # Setup optimizer and loss
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = nn.MSELoss()
        
        # Mixed precision scaler for faster training
        self.scaler = GradScaler()
        
        # Setup tensorboard
        self.writer = SummaryWriter(os.path.join(model_dir, 'logs'))
        
        # Training state
        self.best_val_loss = float('inf')
        self.epoch = 0
    
    def fit(self):
        """Train the model."""
        print(f"Training for {self.num_epochs} epochs...")
        
        for epoch in range(self.num_epochs):
            self.epoch = epoch
            
            # Training phase
            train_loss = self._train_epoch()
            
            # Validation phase
            val_loss = self._validate_epoch()
            
            # Log metrics
            self.writer.add_scalar('Loss/Train', train_loss, epoch)
            self.writer.add_scalar('Loss/Validation', val_loss, epoch)
            
            print(f"Epoch {epoch+1}/{self.num_epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save_model('model_best_MSE.pth')
        
        self.writer.close()
        print(f"Training completed. Best validation loss: {self.best_val_loss:.4f}")
    
    def evaluate_test_set(self, test_data_path: str) -> dict:
        """Evaluate the best model on test set."""
        # Load best model
        best_model_path = os.path.join(self.model_dir, "model_best_MSE.pth")
        if os.path.exists(best_model_path):
            self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
        
        self.model.eval()
        
        # Create test dataset and loader
        test_dataset = DREAMRNNDataset(test_data_path, self.model.seqsize)
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.n_workers,
            pin_memory=True
        )
        
        total_loss = 0.0
        dev_predictions = []
        hk_predictions = []
        dev_targets = []
        hk_targets = []
        
        with torch.no_grad():
            for batch_x, (dev_target, hk_target) in tqdm(test_loader, desc="Evaluating test set"):
                batch_x = batch_x.to(self.device)
                dev_target = dev_target.to(self.device)
                hk_target = hk_target.to(self.device)
                
                with autocast():
                    dev_pred, hk_pred = self.model(batch_x)
                    
                    # Calculate loss
                    dev_loss = self.criterion(dev_pred.squeeze(), dev_target)
                    hk_loss = self.criterion(hk_pred.squeeze(), hk_target)
                    loss = dev_loss + hk_loss
                
                total_loss += loss.item()
                
                # Store predictions and targets
                dev_predictions.extend(dev_pred.squeeze().cpu().numpy())
                hk_predictions.extend(hk_pred.squeeze().cpu().numpy())
                dev_targets.extend(dev_target.cpu().numpy())
                hk_targets.extend(hk_target.cpu().numpy())
        
        # Calculate metrics
        import numpy as np
        from scipy.stats import pearsonr
        
        dev_predictions = np.array(dev_predictions)
        hk_predictions = np.array(hk_predictions)
        dev_targets = np.array(dev_targets)
        hk_targets = np.array(hk_targets)
        
        # Pearson correlation
        dev_pearson, _ = pearsonr(dev_predictions, dev_targets)
        hk_pearson, _ = pearsonr(hk_predictions, hk_targets)
        
        # MSE
        dev_mse = np.mean((dev_predictions - dev_targets) ** 2)
        hk_mse = np.mean((hk_predictions - hk_targets) ** 2)
        
        results = {
            'test_loss': total_loss / len(test_loader),
            'dev_pearson': dev_pearson,
            'hk_pearson': hk_pearson,
            'dev_mse': dev_mse,
            'hk_mse': hk_mse,
            'avg_pearson': (dev_pearson + hk_pearson) / 2
        }
        
        print(f"Test Results - Dev Pearson: {dev_pearson:.4f}, Hk Pearson: {hk_pearson:.4f}, Avg: {results['avg_pearson']:.4f}")
        
        return results
    
    def _train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_x, (dev_target, hk_target) in tqdm(self.train_loader, desc="Training", leave=False):
            batch_x = batch_x.to(self.device)
            dev_target = dev_target.to(self.device)
            hk_target = hk_target.to(self.device)
            
            # Forward pass with mixed precision
            self.optimizer.zero_grad()
            with autocast():
                dev_pred, hk_pred = self.model(batch_x)
                
                # Calculate loss
                dev_loss = self.criterion(dev_pred.squeeze(), dev_target)
                hk_loss = self.criterion(hk_pred.squeeze(), hk_target)
                loss = dev_loss + hk_loss
            
            # Backward pass with mixed precision
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def _validate_epoch(self) -> float:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_x, (dev_target, hk_target) in tqdm(self.val_loader, desc="Validation", leave=False):
                batch_x = batch_x.to(self.device)
                dev_target = dev_target.to(self.device)
                hk_target = hk_target.to(self.device)
                
                # Forward pass with mixed precision
                with autocast():
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
        model_path = os.path.join(self.model_dir, filename)
        torch.save(self.model.state_dict(), model_path)
    
    def load_model(self, filename: str):
        """Load model state dict."""
        model_path = os.path.join(self.model_dir, filename)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))


def build_dream_rnn(
    seqsize: int = 249,
    in_channels: int = 5,
    out_channels: int = 320,
    lstm_hidden_channels: int = 320,
    kernel_sizes: list = [9, 15],
    pool_size: int = 1,
    dropout1: float = 0.2,
    dropout2: float = 0.5,
    generator: Optional[Generator] = None
) -> DREAMRNN:
    """
    Build DREAM-RNN model.
    
    Args:
        seqsize: Length of input sequences
        in_channels: Number of input channels (5 for Prix Fixe: A,C,G,T,rev)
        out_channels: Number of output channels for convolutions
        lstm_hidden_channels: LSTM hidden size
        kernel_sizes: List of kernel sizes for convolutions
        pool_size: Pooling size
        dropout1: Dropout rate for first/core blocks
        dropout2: Dropout rate for core block
        generator: PyTorch random number generator for reproducible initialization
        
    Returns:
        Configured DREAMRNN model
    """
    return DREAMRNN(
        seqsize=seqsize,
        in_channels=in_channels,
        out_channels=out_channels,
        lstm_hidden_channels=lstm_hidden_channels,
        kernel_sizes=kernel_sizes,
        pool_size=pool_size,
        dropout1=dropout1,
        dropout2=dropout2,
        generator=generator
    )
