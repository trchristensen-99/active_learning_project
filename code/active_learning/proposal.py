"""
Sequence proposal strategies for active learning.

Implements different methods for generating candidate sequences.
"""

import random
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional
import torch


class BaseProposalStrategy(ABC):
    """Abstract base class for sequence proposal strategies."""
    
    @abstractmethod
    def propose_sequences(self, n_sequences: int, **kwargs) -> List[str]:
        """
        Propose candidate sequences.
        
        Args:
            n_sequences: Number of sequences to propose
            **kwargs: Additional strategy-specific parameters
            
        Returns:
            List of proposed DNA sequences
        """
        pass


class RandomProposalStrategy(BaseProposalStrategy):
    """
    Random sequence proposal strategy.
    
    Generates completely random DNA sequences of specified length.
    """
    
    def __init__(self, seqsize: int = 249, seed: Optional[int] = None):
        """
        Initialize random proposal strategy.
        
        Args:
            seqsize: Length of sequences to generate
            seed: Random seed for reproducibility
        """
        self.seqsize = seqsize
        self.bases = ['A', 'C', 'G', 'T']
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def propose_sequences(self, n_sequences: int, **kwargs) -> List[str]:
        """
        Generate random DNA sequences.
        
        Args:
            n_sequences: Number of sequences to generate
            
        Returns:
            List of random DNA sequences
        """
        sequences = []
        for _ in range(n_sequences):
            sequence = ''.join(random.choices(self.bases, k=self.seqsize))
            sequences.append(sequence)
        
        return sequences


class PartialRandomMutagenesisStrategy(BaseProposalStrategy):
    """
    Partial random mutagenesis strategy.
    
    Takes existing sequences and introduces random mutations at a specified rate.
    """
    
    def __init__(
        self,
        seqsize: int = 249,
        mutation_rate: float = 0.05,
        seed: Optional[int] = None
    ):
        """
        Initialize partial mutagenesis strategy.
        
        Args:
            seqsize: Length of sequences
            mutation_rate: Fraction of positions to mutate (0.0 to 1.0)
            seed: Random seed for reproducibility
        """
        self.seqsize = seqsize
        self.mutation_rate = mutation_rate
        self.bases = ['A', 'C', 'G', 'T']
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def propose_sequences(
        self,
        n_sequences: int,
        reference_sequences: Optional[List[str]] = None,
        **kwargs
    ) -> List[str]:
        """
        Generate sequences by mutating reference sequences.
        
        Args:
            n_sequences: Number of sequences to generate
            reference_sequences: Reference sequences to mutate (if None, generates random)
            
        Returns:
            List of mutated DNA sequences
        """
        sequences = []
        
        for i in range(n_sequences):
            if reference_sequences is not None:
                # Use reference sequence if available
                ref_seq = reference_sequences[i % len(reference_sequences)]
            else:
                # Generate random reference sequence
                ref_seq = ''.join(random.choices(self.bases, k=self.seqsize))
            
            # Mutate the sequence
            mutated_seq = self._mutate_sequence(ref_seq)
            sequences.append(mutated_seq)
        
        return sequences
    
    def _mutate_sequence(self, sequence: str) -> str:
        """Mutate a single sequence."""
        seq_list = list(sequence)
        n_mutations = int(len(sequence) * self.mutation_rate)
        
        # Randomly select positions to mutate
        positions = random.sample(range(len(sequence)), n_mutations)
        
        for pos in positions:
            # Replace with a random base (different from current)
            current_base = seq_list[pos]
            new_bases = [b for b in self.bases if b != current_base]
            seq_list[pos] = random.choice(new_bases)
        
        return ''.join(seq_list)


class UncertaintyGuidedMutagenesisStrategy(BaseProposalStrategy):
    """
    Uncertainty-guided mutagenesis strategy.
    
    Uses model uncertainty to guide which positions to mutate.
    """
    
    def __init__(
        self,
        seqsize: int = 249,
        mutation_rate: float = 0.05,
        oracle_model: Optional[object] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize uncertainty-guided mutagenesis strategy.
        
        Args:
            seqsize: Length of sequences
            mutation_rate: Fraction of positions to mutate
            oracle_model: Oracle model for uncertainty estimation
            seed: Random seed for reproducibility
        """
        self.seqsize = seqsize
        self.mutation_rate = mutation_rate
        self.oracle_model = oracle_model
        self.bases = ['A', 'C', 'G', 'T']
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def propose_sequences(
        self,
        n_sequences: int,
        reference_sequences: Optional[List[str]] = None,
        **kwargs
    ) -> List[str]:
        """
        Generate sequences using uncertainty-guided mutagenesis.
        
        Args:
            n_sequences: Number of sequences to generate
            reference_sequences: Reference sequences to mutate
            
        Returns:
            List of uncertainty-guided mutated sequences
        """
        if self.oracle_model is None:
            # Fall back to random mutagenesis if no oracle
            strategy = PartialRandomMutagenesisStrategy(
                self.seqsize, self.mutation_rate
            )
            return strategy.propose_sequences(n_sequences, reference_sequences)
        
        sequences = []
        
        for i in range(n_sequences):
            if reference_sequences is not None:
                ref_seq = reference_sequences[i % len(reference_sequences)]
            else:
                ref_seq = ''.join(random.choices(self.bases, k=self.seqsize))
            
            # Use uncertainty to guide mutations
            mutated_seq = self._uncertainty_guided_mutate(ref_seq)
            sequences.append(mutated_seq)
        
        return sequences
    
    def _uncertainty_guided_mutate(self, sequence: str) -> str:
        """Mutate sequence guided by uncertainty."""
        # For now, implement as random mutagenesis
        # TODO: Implement proper uncertainty-guided mutation using gradient-based methods
        strategy = PartialRandomMutagenesisStrategy(
            self.seqsize, self.mutation_rate
        )
        return strategy._mutate_sequence(sequence)


class MixedProposalStrategy(BaseProposalStrategy):
    """
    Mixed proposal strategy combining multiple approaches.
    
    Generates sequences using a combination of different proposal strategies.
    """
    
    def __init__(
        self,
        strategies: List[BaseProposalStrategy],
        weights: Optional[List[float]] = None,
        seqsize: int = 249
    ):
        """
        Initialize mixed proposal strategy.
        
        Args:
            strategies: List of proposal strategies to combine
            weights: Weights for each strategy (if None, equal weights)
            seqsize: Length of sequences
        """
        self.strategies = strategies
        self.seqsize = seqsize
        
        if weights is None:
            self.weights = [1.0 / len(strategies)] * len(strategies)
        else:
            # Normalize weights
            total_weight = sum(weights)
            self.weights = [w / total_weight for w in weights]
    
    def propose_sequences(self, n_sequences: int, **kwargs) -> List[str]:
        """
        Generate sequences using mixed strategies.
        
        Args:
            n_sequences: Number of sequences to generate
            
        Returns:
            List of sequences from mixed strategies
        """
        sequences = []
        
        for i, (strategy, weight) in enumerate(zip(self.strategies, self.weights)):
            n_from_strategy = int(n_sequences * weight)
            if i == len(self.strategies) - 1:  # Last strategy gets remaining sequences
                n_from_strategy = n_sequences - len(sequences)
            
            strategy_sequences = strategy.propose_sequences(n_from_strategy, **kwargs)
            sequences.extend(strategy_sequences)
        
        return sequences



