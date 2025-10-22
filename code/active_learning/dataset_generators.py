"""
Dataset generators for creating test/validation sets with different distribution shifts.

This module provides an extensible system for generating datasets with various
characteristics (genomic, mutated, random, etc.) for evaluating model robustness.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple
import random


class DatasetGenerator(ABC):
    """Base class for test/validation dataset generation."""
    
    def __init__(self, seed: int = 42):
        """
        Initialize dataset generator.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        random.seed(seed)
    
    @abstractmethod
    def generate(self, size: int, **kwargs) -> List[str]:
        """
        Generate sequences.
        
        Args:
            size: Number of sequences to generate
            **kwargs: Additional generator-specific parameters
            
        Returns:
            List of DNA sequences
        """
        pass
    
    def _mutate_sequence(self, sequence: str, mutation_rate: float) -> str:
        """
        Mutate a sequence with given per-position mutation rate.
        
        Args:
            sequence: Original DNA sequence
            mutation_rate: Probability of mutation per position
            
        Returns:
            Mutated sequence
        """
        bases = ['A', 'C', 'G', 'T']
        mutated = list(sequence)
        
        for i in range(len(mutated)):
            if self.rng.random() < mutation_rate:
                # Choose a different base
                current_base = mutated[i]
                other_bases = [b for b in bases if b != current_base]
                mutated[i] = self.rng.choice(other_bases)
        
        return ''.join(mutated)


class NoShiftGenerator(DatasetGenerator):
    """
    No distribution shift - uses original genomic sequences.
    
    This generator simply returns the provided genomic sequences without modification.
    """
    
    def generate(self, size: int, genomic_sequences: List[str] = None, **kwargs) -> List[str]:
        """
        Return original genomic sequences.
        
        Args:
            size: Number of sequences (must match len(genomic_sequences))
            genomic_sequences: Original sequences from dataset
            
        Returns:
            Original genomic sequences
        """
        if genomic_sequences is None:
            raise ValueError("genomic_sequences must be provided for NoShiftGenerator")
        
        if len(genomic_sequences) != size:
            raise ValueError(f"Expected {size} sequences, got {len(genomic_sequences)}")
        
        return genomic_sequences


class LowShiftGenerator(DatasetGenerator):
    """
    Low distribution shift - genomic sequences with per-position mutations.
    
    Each position in the sequence has a fixed probability of being mutated
    to a different base, creating sequences that are similar but not identical
    to the original genomic sequences.
    """
    
    def __init__(self, seed: int = 42, mutation_rate: float = 0.05):
        """
        Initialize low shift generator.
        
        Args:
            seed: Random seed for reproducibility
            mutation_rate: Probability of mutation per position (default: 0.05 = 5%)
        """
        super().__init__(seed)
        self.mutation_rate = mutation_rate
    
    def generate(self, size: int, genomic_sequences: List[str] = None, **kwargs) -> List[str]:
        """
        Generate mutated versions of genomic sequences.
        
        Args:
            size: Number of sequences (must match len(genomic_sequences))
            genomic_sequences: Original sequences to mutate
            
        Returns:
            Mutated sequences
        """
        if genomic_sequences is None:
            raise ValueError("genomic_sequences must be provided for LowShiftGenerator")
        
        if len(genomic_sequences) != size:
            raise ValueError(f"Expected {size} sequences, got {len(genomic_sequences)}")
        
        mutated_sequences = []
        for seq in genomic_sequences:
            mutated_seq = self._mutate_sequence(seq, self.mutation_rate)
            mutated_sequences.append(mutated_seq)
        
        return mutated_sequences


class HighShiftLowActivityGenerator(DatasetGenerator):
    """
    High distribution shift with low activity - random DNA sequences.
    
    Generates completely random DNA sequences, which typically have low
    regulatory activity and represent a significant distribution shift from
    genomic sequences.
    """
    
    def __init__(self, seed: int = 42, seqsize: int = 249):
        """
        Initialize high shift low activity generator.
        
        Args:
            seed: Random seed for reproducibility
            seqsize: Length of sequences to generate (default: 249 for DeepSTARR)
        """
        super().__init__(seed)
        self.seqsize = seqsize
    
    def generate(self, size: int, **kwargs) -> List[str]:
        """
        Generate random DNA sequences.
        
        Args:
            size: Number of sequences to generate
            
        Returns:
            Random DNA sequences
        """
        bases = ['A', 'C', 'G', 'T']
        sequences = []
        
        for _ in range(size):
            seq = ''.join(self.rng.choice(bases) for _ in range(self.seqsize))
            sequences.append(seq)
        
        return sequences


# Registry for easy extension
DATASET_GENERATORS = {
    'no_shift': NoShiftGenerator,
    'low_shift': LowShiftGenerator,
    'high_shift_low_activity': HighShiftLowActivityGenerator,
    # Future extensions can be added here:
    # 'high_shift_high_activity': HighShiftHighActivityGenerator,
    # 'medium_shift': MediumShiftGenerator,
    # etc.
}


def get_generator(generator_type: str, seed: int = 42, **kwargs) -> DatasetGenerator:
    """
    Get a dataset generator by type.
    
    Args:
        generator_type: Type of generator ('no_shift', 'low_shift', etc.)
        seed: Random seed for reproducibility
        **kwargs: Additional generator-specific parameters
        
    Returns:
        Initialized dataset generator
        
    Raises:
        ValueError: If generator_type is not recognized
    """
    if generator_type not in DATASET_GENERATORS:
        available = ', '.join(DATASET_GENERATORS.keys())
        raise ValueError(f"Unknown generator type '{generator_type}'. Available: {available}")
    
    generator_class = DATASET_GENERATORS[generator_type]
    return generator_class(seed=seed, **kwargs)

