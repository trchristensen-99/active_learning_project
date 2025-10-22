"""
Generate test and validation datasets with different distribution shifts.

This script creates multiple test/validation datasets for evaluating model
robustness under distribution shift:
- no_shift: Original genomic sequences
- low_shift: Genomic sequences with 5% per-position mutations
- high_shift_low_activity: Random DNA sequences

Non-genomic datasets are labeled using an oracle ensemble.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import yaml
from typing import List, Tuple, Dict
import hashlib

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from code.active_learning.dataset_generators import get_generator, DATASET_GENERATORS
from code.active_learning.oracle import EnsembleOracle


def load_genomic_data(file_path: str) -> Tuple[List[str], np.ndarray]:
    """
    Load genomic sequences and labels from TSV file.
    
    Args:
        file_path: Path to TSV file with format: ID\tSequence\tDev_log2_enrichment\tHk_log2_enrichment\trev
        
    Returns:
        Tuple of (sequences, labels)
    """
    import pandas as pd
    
    # Load with pandas for easier handling
    df = pd.read_csv(file_path, sep='\t')
    
    sequences = df['Sequence'].tolist()
    labels = df[['Dev_log2_enrichment', 'Hk_log2_enrichment']].values
    
    return sequences, labels


def save_dataset(sequences: List[str], labels: np.ndarray, output_path: str):
    """
    Save sequences and labels to TSV file.
    
    Args:
        sequences: List of DNA sequences
        labels: Array of shape (n, 2) with dev and hk activities
        output_path: Path to output TSV file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for seq, (dev_act, hk_act) in zip(sequences, labels):
            f.write(f"{seq}\t{dev_act}\t{hk_act}\n")
    
    print(f"Saved {len(sequences)} sequences to {output_path}")


def generate_dataset_type(
    dataset_type: str,
    genomic_sequences: List[str],
    genomic_labels: np.ndarray,
    oracle,
    seed: int,
    config: Dict
) -> Tuple[List[str], np.ndarray]:
    """
    Generate a specific dataset type.
    
    Args:
        dataset_type: Type of dataset ('no_shift', 'low_shift', etc.)
        genomic_sequences: Original genomic sequences
        genomic_labels: Original genomic labels
        oracle: Oracle ensemble for labeling
        seed: Random seed
        config: Dataset type configuration
        
    Returns:
        Tuple of (sequences, labels)
    """
    print(f"\nGenerating {dataset_type} dataset...")
    
    # Get generator parameters
    generator_params = {}
    if 'mutation_rate' in config:
        generator_params['mutation_rate'] = config['mutation_rate']
    
    # Add seqsize for generators that need it
    # Infer from genomic sequences if not specified
    if 'seqsize' in config:
        seqsize = config['seqsize']
    else:
        seqsize = len(genomic_sequences[0])
    
    # Only pass seqsize to generators that accept it (high_shift_low_activity)
    if dataset_type == 'high_shift_low_activity':
        generator_params['seqsize'] = seqsize
    
    # Create generator
    generator = get_generator(dataset_type, seed=seed, **generator_params)
    
    # Generate sequences
    size = len(genomic_sequences)
    if dataset_type == 'no_shift':
        sequences = generator.generate(size, genomic_sequences=genomic_sequences)
        labels = genomic_labels  # Use original labels
    else:
        # Generate sequences
        if dataset_type == 'low_shift':
            sequences = generator.generate(size, genomic_sequences=genomic_sequences)
        else:
            sequences = generator.generate(size)
        
        # Label with oracle
        print(f"Labeling {len(sequences)} sequences with oracle ensemble...")
        
        # Process in batches with progress logging
        batch_size = 5000
        all_labels = []
        total_batches = (len(sequences) + batch_size - 1) // batch_size
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(sequences))
            batch_seqs = sequences[start_idx:end_idx]
            
            batch_labels = oracle.predict(batch_seqs)
            all_labels.append(batch_labels)
            
            # Log progress every 10% of completion
            progress = (batch_idx + 1) / total_batches
            if (batch_idx + 1) % max(1, total_batches // 10) == 0 or batch_idx + 1 == total_batches:
                print(f"    Progress: {progress*100:.0f}% ({end_idx}/{len(sequences)} sequences labeled)")
        
        labels = np.vstack(all_labels)
        print(f"Oracle labeling complete. Label shape: {labels.shape}")
    
    return sequences, labels


def generate_mixed_dataset(
    individual_datasets: Dict[str, Tuple[List[str], np.ndarray]],
    proportions: Dict[str, float],
    seed: int
) -> Tuple[List[str], np.ndarray]:
    """
    Generate a mixed dataset by sampling from individual datasets.
    
    Args:
        individual_datasets: Dict mapping dataset_type to (sequences, labels)
        proportions: Dict mapping dataset_type to proportion (should sum to 1.0)
        seed: Random seed for reproducible sampling
        
    Returns:
        Tuple of (mixed_sequences, mixed_labels)
    """
    rng = np.random.RandomState(seed)
    
    # Determine total size (use first dataset as reference)
    total_size = len(next(iter(individual_datasets.values()))[0])
    
    # Calculate number of samples from each type
    samples_per_type = {}
    remaining = total_size
    types_list = list(proportions.keys())
    
    for i, dataset_type in enumerate(types_list[:-1]):
        n_samples = int(total_size * proportions[dataset_type])
        samples_per_type[dataset_type] = n_samples
        remaining -= n_samples
    
    # Last type gets remaining samples to ensure exact total
    samples_per_type[types_list[-1]] = remaining
    
    # Sample from each dataset
    mixed_sequences = []
    mixed_labels = []
    
    for dataset_type, n_samples in samples_per_type.items():
        if n_samples == 0:
            continue
        
        sequences, labels = individual_datasets[dataset_type]
        indices = rng.choice(len(sequences), size=n_samples, replace=False)
        
        for idx in indices:
            mixed_sequences.append(sequences[idx])
            mixed_labels.append(labels[idx])
    
    # Shuffle the mixed dataset
    indices = rng.permutation(len(mixed_sequences))
    mixed_sequences = [mixed_sequences[i] for i in indices]
    mixed_labels = np.array([mixed_labels[i] for i in indices])
    
    return mixed_sequences, mixed_labels


def proportions_to_name(proportions: Dict[str, float]) -> str:
    """
    Convert proportions dict to descriptive name.
    
    Args:
        proportions: Dict mapping dataset_type to proportion
        
    Returns:
        Descriptive name (e.g., "70genomic20lowshift10highshift")
    """
    # Sort by proportion (descending) for consistency
    sorted_items = sorted(proportions.items(), key=lambda x: -x[1])
    
    parts = []
    for dataset_type, prop in sorted_items:
        if prop > 0:
            # Convert to percentage
            pct = int(round(prop * 100))
            # Simplify dataset type name
            type_name = dataset_type.replace('_', '')
            parts.append(f"{pct}{type_name}")
    
    return ''.join(parts)


def main():
    parser = argparse.ArgumentParser(
        description="Generate test/validation datasets with distribution shifts"
    )
    parser.add_argument(
        '--dataset',
        required=True,
        help='Dataset name (e.g., deepstarr)'
    )
    parser.add_argument(
        '--genomic-test',
        required=True,
        help='Path to genomic test set TSV file'
    )
    parser.add_argument(
        '--genomic-val',
        required=True,
        help='Path to genomic validation set TSV file'
    )
    parser.add_argument(
        '--oracle-dir',
        required=True,
        help='Path to oracle ensemble directory'
    )
    parser.add_argument(
        '--output-dir',
        default='data/test_val_sets',
        help='Output directory for generated datasets'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--config',
        default='configs/test_val_datasets.yaml',
        help='Path to dataset configuration YAML'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Test/Validation Dataset Generation")
    print("=" * 80)
    print(f"Dataset: {args.dataset}")
    print(f"Genomic test: {args.genomic_test}")
    print(f"Genomic val: {args.genomic_val}")
    print(f"Oracle: {args.oracle_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Seed: {args.seed}")
    print("=" * 80)
    
    # Load configuration
    with open(args.config, 'r') as f:
        full_config = yaml.safe_load(f)
    
    dataset_types_config = full_config['dataset_types']
    mixing_strategies = full_config.get('mixing_strategies', {})
    
    # Load genomic data
    print("\nLoading genomic data...")
    test_sequences, test_labels = load_genomic_data(args.genomic_test)
    val_sequences, val_labels = load_genomic_data(args.genomic_val)
    print(f"Loaded {len(test_sequences)} test sequences")
    print(f"Loaded {len(val_sequences)} validation sequences")
    
    # Load oracle
    print("\nLoading oracle ensemble...")
    oracle = EnsembleOracle(
        model_dir=args.oracle_dir,
        model_type='dream_rnn',
        device='auto',
        seqsize=249,
        batch_size=1024
    )
    print(f"Oracle loaded successfully")
    
    # Generate individual dataset types
    individual_test_datasets = {}
    individual_val_datasets = {}
    
    for dataset_type, config in dataset_types_config.items():
        # Use dataset-specific seed for reproducibility
        type_seed = args.seed + hash(dataset_type) % 10000
        
        # Generate test set
        test_seqs, test_labs = generate_dataset_type(
            dataset_type,
            test_sequences,
            test_labels,
            oracle,
            type_seed,
            config
        )
        individual_test_datasets[dataset_type] = (test_seqs, test_labs)
        
        # Generate validation set
        val_seqs, val_labs = generate_dataset_type(
            dataset_type,
            val_sequences,
            val_labels,
            oracle,
            type_seed + 1,  # Different seed for val
            config
        )
        individual_val_datasets[dataset_type] = (val_seqs, val_labs)
        
        # Save individual datasets
        output_base = Path(args.output_dir) / args.dataset / dataset_type
        save_dataset(test_seqs, test_labs, output_base / 'test.txt')
        save_dataset(val_seqs, val_labs, output_base / 'val.txt')
    
    # Generate mixed datasets
    print("\n" + "=" * 80)
    print("Generating mixed datasets...")
    print("=" * 80)
    
    for strategy_name, proportions in mixing_strategies.items():
        print(f"\nGenerating {strategy_name} mixed dataset...")
        print(f"Proportions: {proportions}")
        
        # Use strategy-specific seed
        strategy_seed = args.seed + hash(strategy_name) % 10000
        
        # Generate mixed test set
        mixed_test_seqs, mixed_test_labs = generate_mixed_dataset(
            individual_test_datasets,
            proportions,
            strategy_seed
        )
        
        # Generate mixed validation set
        mixed_val_seqs, mixed_val_labs = generate_mixed_dataset(
            individual_val_datasets,
            proportions,
            strategy_seed + 1
        )
        
        # Create descriptive name
        descriptive_name = proportions_to_name(proportions)
        
        # Save mixed datasets
        output_base = Path(args.output_dir) / args.dataset / descriptive_name
        save_dataset(mixed_test_seqs, mixed_test_labs, output_base / 'test.txt')
        save_dataset(mixed_val_seqs, mixed_val_labs, output_base / 'val.txt')
        
        print(f"Saved as: {descriptive_name}")
    
    print("\n" + "=" * 80)
    print("Dataset generation complete!")
    print("=" * 80)
    print(f"\nGenerated datasets in: {args.output_dir}/{args.dataset}/")
    print("\nAvailable dataset types:")
    for dataset_type in dataset_types_config.keys():
        print(f"  - {dataset_type}")
    print("\nAvailable mixed strategies:")
    for strategy_name, proportions in mixing_strategies.items():
        descriptive_name = proportions_to_name(proportions)
        print(f"  - {descriptive_name} ({strategy_name})")


if __name__ == '__main__':
    main()

