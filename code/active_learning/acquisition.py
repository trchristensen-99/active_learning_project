"""
Acquisition functions for active learning.

Implements different strategies for selecting sequences from candidate pools.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
import torch


class BaseAcquisitionFunction(ABC):
    """Abstract base class for acquisition functions."""
    
    @abstractmethod
    def select_sequences(
        self,
        candidate_sequences: List[str],
        n_select: int,
        oracle_model: Optional[object] = None,
        **kwargs
    ) -> List[str]:
        """
        Select sequences from candidate pool.
        
        Args:
            candidate_sequences: Pool of candidate sequences
            n_select: Number of sequences to select
            oracle_model: Oracle model for uncertainty estimation
            **kwargs: Additional parameters
            
        Returns:
            List of selected sequences
        """
        pass


class RandomAcquisition(BaseAcquisitionFunction):
    """
    Random acquisition function.
    
    Randomly selects sequences from the candidate pool.
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize random acquisition.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
    
    def select_sequences(
        self,
        candidate_sequences: List[str],
        n_select: int,
        oracle_model: Optional[object] = None,
        **kwargs
    ) -> List[str]:
        """
        Randomly select sequences from candidate pool.
        
        Args:
            candidate_sequences: Pool of candidate sequences
            n_select: Number of sequences to select
            oracle_model: Not used for random selection
            
        Returns:
            List of randomly selected sequences
        """
        if n_select >= len(candidate_sequences):
            return candidate_sequences.copy()
        
        # Randomly sample without replacement
        selected_indices = np.random.choice(
            len(candidate_sequences),
            size=n_select,
            replace=False
        )
        
        return [candidate_sequences[i] for i in selected_indices]


class UncertaintyAcquisition(BaseAcquisitionFunction):
    """
    Uncertainty-based acquisition function.
    
    Selects sequences with highest predictive uncertainty.
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize uncertainty acquisition.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
    
    def select_sequences(
        self,
        candidate_sequences: List[str],
        n_select: int,
        oracle_model: Optional[object] = None,
        **kwargs
    ) -> List[str]:
        """
        Select sequences with highest uncertainty.
        
        Args:
            candidate_sequences: Pool of candidate sequences
            n_select: Number of sequences to select
            oracle_model: Oracle model for uncertainty estimation
            
        Returns:
            List of sequences with highest uncertainty
        """
        if oracle_model is None:
            # Fall back to random selection if no oracle
            random_acq = RandomAcquisition(self.seed)
            return random_acq.select_sequences(candidate_sequences, n_select)
        
        if n_select >= len(candidate_sequences):
            return candidate_sequences.copy()
        
        # Get uncertainty estimates
        uncertainties = oracle_model.get_uncertainty(candidate_sequences)
        
        # Select sequences with highest uncertainty
        uncertainty_indices = np.argsort(uncertainties)[::-1]  # Descending order
        selected_indices = uncertainty_indices[:n_select]
        
        return [candidate_sequences[i] for i in selected_indices]


class LargestClusterMaximumDistanceAcquisition(BaseAcquisitionFunction):
    """
    Largest Cluster Maximum Distance (LCMD) acquisition function.
    
    Balances informativeness, diversity, and representativeness when selecting batches.
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize LCMD acquisition.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
    
    def select_sequences(
        self,
        candidate_sequences: List[str],
        n_select: int,
        oracle_model: Optional[object] = None,
        **kwargs
    ) -> List[str]:
        """
        Select sequences using LCMD strategy.
        
        Args:
            candidate_sequences: Pool of candidate sequences
            n_select: Number of sequences to select
            oracle_model: Oracle model for uncertainty estimation
            
        Returns:
            List of sequences selected using LCMD
        """
        if n_select >= len(candidate_sequences):
            return candidate_sequences.copy()
        
        # For now, implement as uncertainty-based selection
        # TODO: Implement proper LCMD algorithm with clustering and diversity metrics
        uncertainty_acq = UncertaintyAcquisition(self.seed)
        return uncertainty_acq.select_sequences(
            candidate_sequences, n_select, oracle_model
        )


class BatchBALDAcquisition(BaseAcquisitionFunction):
    """
    BatchBALD acquisition function.
    
    Selects batches that maximize mutual information between model parameters and labels.
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize BatchBALD acquisition.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
    
    def select_sequences(
        self,
        candidate_sequences: List[str],
        n_select: int,
        oracle_model: Optional[object] = None,
        **kwargs
    ) -> List[str]:
        """
        Select sequences using BatchBALD strategy.
        
        Args:
            candidate_sequences: Pool of candidate sequences
            n_select: Number of sequences to select
            oracle_model: Oracle model for uncertainty estimation
            
        Returns:
            List of sequences selected using BatchBALD
        """
        if n_select >= len(candidate_sequences):
            return candidate_sequences.copy()
        
        # For now, implement as uncertainty-based selection
        # TODO: Implement proper BatchBALD algorithm
        uncertainty_acq = UncertaintyAcquisition(self.seed)
        return uncertainty_acq.select_sequences(
            candidate_sequences, n_select, oracle_model
        )


class DiversityAcquisition(BaseAcquisitionFunction):
    """
    Diversity-based acquisition function.
    
    Selects sequences that maximize diversity in the selected batch.
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize diversity acquisition.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
    
    def select_sequences(
        self,
        candidate_sequences: List[str],
        n_select: int,
        oracle_model: Optional[object] = None,
        **kwargs
    ) -> List[str]:
        """
        Select sequences to maximize diversity.
        
        Args:
            candidate_sequences: Pool of candidate sequences
            n_select: Number of sequences to select
            oracle_model: Not used for diversity selection
            
        Returns:
            List of diverse sequences
        """
        if n_select >= len(candidate_sequences):
            return candidate_sequences.copy()
        
        # Simple diversity strategy: select sequences with maximum Hamming distance
        selected = []
        remaining = candidate_sequences.copy()
        
        # Start with a random sequence
        first_idx = np.random.randint(len(remaining))
        selected.append(remaining.pop(first_idx))
        
        # Iteratively select sequences with maximum minimum distance to selected
        while len(selected) < n_select and remaining:
            max_min_distance = -1
            best_candidate = None
            best_idx = -1
            
            for i, candidate in enumerate(remaining):
                # Calculate minimum Hamming distance to already selected sequences
                min_distance = min(
                    self._hamming_distance(candidate, sel) for sel in selected
                )
                
                if min_distance > max_min_distance:
                    max_min_distance = min_distance
                    best_candidate = candidate
                    best_idx = i
            
            if best_candidate is not None:
                selected.append(remaining.pop(best_idx))
            else:
                # If no improvement possible, select randomly
                idx = np.random.randint(len(remaining))
                selected.append(remaining.pop(idx))
        
        return selected
    
    def _hamming_distance(self, seq1: str, seq2: str) -> int:
        """Calculate Hamming distance between two sequences."""
        if len(seq1) != len(seq2):
            raise ValueError("Sequences must have the same length")
        
        return sum(c1 != c2 for c1, c2 in zip(seq1, seq2))



