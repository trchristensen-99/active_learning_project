"""
Evolution-inspired data augmentations for genomic sequences.

Based on the EvoAug paper: Lee et al., Genome Biology 2023
https://genomebiology.biomedcentral.com/articles/10.1186/s13059-023-02941-w
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any


class EvoAugConfig:
    """Configuration for EvoAug augmentations."""
    
    def __init__(self, config_dict: Dict[str, Any]):
        """
        Initialize from config dictionary.
        
        Expected format:
        {
            "enabled": true,
            "augmentations": {
                "mutation": {"enabled": true, "mutate_frac": 0.15},
                "translocation": {"enabled": true, "shift_min": 0, "shift_max": 30},
                ...
            },
            "max_augs_per_sequence": 3,
            "mode": "hard"  # or "soft"
        }
        """
        self.enabled = config_dict.get('enabled', False)
        self.augmentations = config_dict.get('augmentations', {})
        self.max_augs_per_sequence = config_dict.get('max_augs_per_sequence', 2)
        self.mode = config_dict.get('mode', 'hard')  # hard or soft
        
        # Get list of enabled augmentations
        self.enabled_augs = []
        for aug_name, aug_config in self.augmentations.items():
            if aug_config.get('enabled', False):
                self.enabled_augs.append(aug_name)
    
    def get_signature(self) -> str:
        """Get a short signature string for directory naming."""
        if not self.enabled or not self.enabled_augs:
            return ""
        
        # Short names for augmentations
        short_names = {
            'mutation': 'mut',
            'translocation': 'trans',
            'insertion': 'ins',
            'deletion': 'del',
            'inversion': 'inv',
            'reverse_complement': 'rc',
            'noise': 'noise'
        }
        
        aug_str = '_'.join([short_names.get(a, a) for a in sorted(self.enabled_augs)])
        return f"evoaug_{aug_str}{self.max_augs_per_sequence}_{self.mode}"


class EvoAugmentor:
    """Apply evolution-inspired augmentations to one-hot encoded sequences."""
    
    def __init__(self, config: EvoAugConfig, seed: Optional[int] = None):
        """
        Initialize augmentor.
        
        Args:
            config: EvoAugConfig object
            seed: Random seed for reproducibility
        """
        self.config = config
        self.seed = seed
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Augmentation priority order (as per EvoAug paper)
        self.aug_order = [
            'inversion', 'deletion', 'translocation', 
            'insertion', 'reverse_complement', 'mutation', 'noise'
        ]
    
    def augment_batch(self, sequences: torch.Tensor) -> torch.Tensor:
        """
        Apply augmentations to a batch of one-hot encoded sequences.
        
        Args:
            sequences: Tensor of shape (batch_size, seq_len, 4) - one-hot encoded
        
        Returns:
            Augmented sequences of same shape
        """
        if not self.config.enabled:
            return sequences
        
        batch_size = sequences.shape[0]
        augmented = sequences.clone()
        
        for i in range(batch_size):
            augmented[i] = self._augment_single(sequences[i])
        
        return augmented
    
    def _augment_single(self, sequence: torch.Tensor) -> torch.Tensor:
        """Augment a single sequence."""
        # Determine number of augmentations to apply
        if self.config.mode == 'hard':
            n_augs = self.config.max_augs_per_sequence
        else:  # soft mode
            n_augs = np.random.randint(1, self.config.max_augs_per_sequence + 1)
        
        # Sample augmentations to apply (without replacement)
        available_augs = [a for a in self.aug_order if a in self.config.enabled_augs]
        if not available_augs:
            return sequence
        
        n_augs = min(n_augs, len(available_augs))
        selected_augs = np.random.choice(available_augs, size=n_augs, replace=False)
        
        # Apply augmentations in priority order
        aug_seq = sequence.clone()
        for aug_name in self.aug_order:
            if aug_name in selected_augs:
                aug_seq = self._apply_augmentation(aug_seq, aug_name)
        
        return aug_seq
    
    def _apply_augmentation(self, sequence: torch.Tensor, aug_name: str) -> torch.Tensor:
        """Apply a specific augmentation."""
        aug_config = self.config.augmentations[aug_name]
        
        if aug_name == 'mutation':
            return self._mutate(sequence, aug_config)
        elif aug_name == 'translocation':
            return self._translocate(sequence, aug_config)
        elif aug_name == 'insertion':
            return self._insert(sequence, aug_config)
        elif aug_name == 'deletion':
            return self._delete(sequence, aug_config)
        elif aug_name == 'inversion':
            return self._invert(sequence, aug_config)
        elif aug_name == 'reverse_complement':
            return self._reverse_complement(sequence, aug_config)
        elif aug_name == 'noise':
            return self._add_noise(sequence, aug_config)
        else:
            return sequence
    
    def _mutate(self, sequence: torch.Tensor, config: Dict) -> torch.Tensor:
        """
        Apply random single nucleotide mutations.
        
        Args:
            sequence: (seq_len, 4) one-hot encoded
            config: {"mutate_frac": float}
        """
        mutate_frac = config.get('mutate_frac', 0.1)
        seq_len = sequence.shape[0]
        
        # Account for silent mutations (mutating to same base)
        adjusted_frac = mutate_frac / 0.75
        n_mutations = int(seq_len * adjusted_frac)
        
        if n_mutations == 0:
            return sequence
        
        # Sample positions (with replacement)
        device = sequence.device
        positions = torch.randint(0, seq_len, (n_mutations,), device=device)
        
        aug_seq = sequence.clone()
        for pos in positions:
            # Create random one-hot vector
            new_base = torch.zeros(4, device=device)
            new_base[torch.randint(0, 4, (1,), device=device)] = 1.0
            aug_seq[pos] = new_base
        
        return aug_seq
    
    def _translocate(self, sequence: torch.Tensor, config: Dict) -> torch.Tensor:
        """
        Circular shift (roll) the sequence.
        
        Args:
            sequence: (seq_len, 4) one-hot encoded
            config: {"shift_min": int, "shift_max": int}
        """
        shift_min = config.get('shift_min', 0)
        shift_max = config.get('shift_max', 30)
        
        # Random shift distance (can be negative for backward shift)
        shift = np.random.randint(-shift_max, shift_max + 1)
        if abs(shift) < shift_min:
            return sequence
        
        # Roll along sequence dimension
        return torch.roll(sequence, shifts=shift, dims=0)
    
    def _insert(self, sequence: torch.Tensor, config: Dict) -> torch.Tensor:
        """
        Insert random DNA sequence.
        
        Args:
            sequence: (seq_len, 4) one-hot encoded
            config: {"insert_min": int, "insert_max": int}
        """
        insert_min = config.get('insert_min', 0)
        insert_max = config.get('insert_max', 30)
        seq_len = sequence.shape[0]
        
        # Random insertion length
        insert_len = np.random.randint(insert_min, insert_max + 1)
        if insert_len == 0:
            return sequence
        
        # Generate random insertion
        device = sequence.device
        insertion = self._random_sequence(insert_len, device=device)
        
        # Random insertion position
        insert_pos = np.random.randint(0, seq_len - insert_len + 1)
        
        # Insert and maintain length
        # Split remaining padding evenly on flanks
        remaining_len = insert_max - insert_len
        pad_5prime = remaining_len // 2
        pad_3prime = remaining_len - pad_5prime
        
        # Create new sequence
        aug_seq = torch.cat([
            self._random_sequence(pad_5prime, device=device),
            sequence[:insert_pos],
            insertion,
            sequence[insert_pos:seq_len-insert_len-pad_3prime],
            self._random_sequence(pad_3prime, device=device)
        ], dim=0)
        
        # Ensure correct length
        if aug_seq.shape[0] > seq_len:
            aug_seq = aug_seq[:seq_len]
        elif aug_seq.shape[0] < seq_len:
            aug_seq = torch.cat([aug_seq, self._random_sequence(seq_len - aug_seq.shape[0], device=device)], dim=0)
        
        return aug_seq
    
    def _delete(self, sequence: torch.Tensor, config: Dict) -> torch.Tensor:
        """
        Delete contiguous segment and pad with random DNA.
        
        Args:
            sequence: (seq_len, 4) one-hot encoded
            config: {"delete_min": int, "delete_max": int}
        """
        delete_min = config.get('delete_min', 0)
        delete_max = config.get('delete_max', 30)
        seq_len = sequence.shape[0]
        
        # Random deletion length
        delete_len = np.random.randint(delete_min, delete_max + 1)
        if delete_len == 0 or delete_len >= seq_len:
            return sequence
        
        # Random deletion start position
        delete_start = np.random.randint(0, seq_len - delete_len + 1)
        delete_end = delete_start + delete_len
        
        # Concatenate parts before and after deletion
        remaining = torch.cat([sequence[:delete_start], sequence[delete_end:]], dim=0)
        
        # Pad to maintain length
        padding_len = seq_len - remaining.shape[0]
        if padding_len > 0:
            device = sequence.device
            pad_5prime = padding_len // 2
            pad_3prime = padding_len - pad_5prime
            aug_seq = torch.cat([
                self._random_sequence(pad_5prime, device=device),
                remaining,
                self._random_sequence(pad_3prime, device=device)
            ], dim=0)
        else:
            aug_seq = remaining
        
        return aug_seq
    
    def _invert(self, sequence: torch.Tensor, config: Dict) -> torch.Tensor:
        """
        Reverse-complement a random subsequence.
        
        Args:
            sequence: (seq_len, 4) one-hot encoded
            config: {"invert_min": int, "invert_max": int}
        """
        invert_min = config.get('invert_min', 0)
        invert_max = config.get('invert_max', 30)
        seq_len = sequence.shape[0]
        
        # Random inversion length
        invert_len = np.random.randint(invert_min, min(invert_max + 1, seq_len + 1))
        if invert_len == 0:
            return sequence
        
        # Random inversion start position
        invert_start = np.random.randint(0, seq_len - invert_len + 1)
        invert_end = invert_start + invert_len
        
        # Reverse complement the subsequence
        aug_seq = sequence.clone()
        subseq = sequence[invert_start:invert_end]
        aug_seq[invert_start:invert_end] = self._reverse_complement_seq(subseq)
        
        return aug_seq
    
    def _reverse_complement(self, sequence: torch.Tensor, config: Dict) -> torch.Tensor:
        """
        Reverse complement entire sequence with probability rc_prob.
        
        Args:
            sequence: (seq_len, 4) one-hot encoded
            config: {"rc_prob": float}
        """
        rc_prob = config.get('rc_prob', 0.5)
        
        if np.random.random() < rc_prob:
            return self._reverse_complement_seq(sequence)
        return sequence
    
    def _add_noise(self, sequence: torch.Tensor, config: Dict) -> torch.Tensor:
        """
        Add Gaussian noise to one-hot encoding.
        
        Args:
            sequence: (seq_len, 4) one-hot encoded
            config: {"noise_mean": float, "noise_std": float}
        """
        noise_mean = config.get('noise_mean', 0.0)
        noise_std = config.get('noise_std', 0.1)
        
        noise = torch.randn_like(sequence) * noise_std + noise_mean
        return sequence + noise
    
    def _reverse_complement_seq(self, sequence: torch.Tensor) -> torch.Tensor:
        """Reverse complement a one-hot encoded sequence."""
        # Reverse along sequence dimension
        rev_seq = torch.flip(sequence, dims=[0])
        # Complement: swap A<->T (indices 0<->3) and C<->G (indices 1<->2)
        comp_seq = torch.zeros_like(rev_seq)
        comp_seq[:, 0] = rev_seq[:, 3]  # T -> A
        comp_seq[:, 1] = rev_seq[:, 2]  # G -> C
        comp_seq[:, 2] = rev_seq[:, 1]  # C -> G
        comp_seq[:, 3] = rev_seq[:, 0]  # A -> T
        return comp_seq
    
    def _random_sequence(self, length: int, device: torch.device = None) -> torch.Tensor:
        """Generate random one-hot encoded DNA sequence."""
        if length == 0:
            return torch.zeros((0, 4), device=device)
        
        seq = torch.zeros((length, 4), device=device)
        bases = torch.randint(0, 4, (length,), device=device)
        seq[torch.arange(length, device=device), bases] = 1.0
        return seq


def apply_augmentations(sequences: torch.Tensor, config_dict: Dict[str, Any],
                       seed: Optional[int] = None) -> torch.Tensor:
    """
    Convenience function to apply augmentations.
    
    Args:
        sequences: (batch_size, seq_len, 4) one-hot encoded
        config_dict: EvoAug configuration dictionary
        seed: Random seed
    
    Returns:
        Augmented sequences of same shape
    """
    config = EvoAugConfig(config_dict)
    augmentor = EvoAugmentor(config, seed=seed)
    return augmentor.augment_batch(sequences)


