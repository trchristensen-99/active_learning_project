#!/usr/bin/env python3
"""
Active Learning Orchestration Script

Runs the complete active learning pipeline with configurable parameters.
"""

import argparse
import json
import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from code.active_learning import (
    EnsembleOracle,
    DeepSTARRActiveLearningTrainer,
    RandomProposalStrategy,
    MixedProposalStrategy,
    RandomAcquisition,
    UncertaintyAcquisition,
    ActiveLearningCycle,
    ConfigurationManager,
    CheckpointManager
)
from code.utils import one_hot_encode
import subprocess


def load_initial_data(data_path: str, n_initial: int = 20000) -> tuple:
    """
    Load initial training data.
    
    Args:
        data_path: Path to training data (TSV format)
        n_initial: Number of initial sequences to use
        
    Returns:
        Tuple of (sequences, labels)
    """
    print(f"Loading initial data from {data_path}...")
    
    # Load data
    df = pd.read_csv(data_path, sep='\t')
    
    # Sample initial sequences
    if len(df) > n_initial:
        df = df.sample(n=n_initial, random_state=42)
    
    sequences = df['Sequence'].tolist()
    labels = df[['Dev_log2_enrichment', 'Hk_log2_enrichment']].values
    
    print(f"Loaded {len(sequences)} initial sequences")
    return sequences, labels


def load_test_val_dataset(dataset_path: str) -> tuple:
    """
    Load test or validation dataset.
    
    Args:
        dataset_path: Path to dataset TSV file
        
    Returns:
        Tuple of (sequences, labels)
    """
    df = pd.read_csv(dataset_path, sep='\t', header=None, names=['Sequence', 'Dev', 'Hk'])
    sequences = df['Sequence'].tolist()
    labels = df[['Dev', 'Hk']].values
    return sequences, labels


def ensure_test_val_datasets(dataset_name: str, genomic_test: str, genomic_val: str, oracle_dir: str, seed: int = 42):
    """
    Ensure test/validation datasets exist, generate if needed.
    
    Args:
        dataset_name: Name of dataset (e.g., 'deepstarr')
        genomic_test: Path to genomic test set
        genomic_val: Path to genomic validation set
        oracle_dir: Path to oracle ensemble directory
        seed: Random seed for generation
    """
    output_dir = Path('data/test_val_sets') / dataset_name
    
    # Check if datasets exist
    required_types = ['no_shift', 'low_shift', 'high_shift_low_activity']
    all_exist = all(
        (output_dir / dtype / 'test.txt').exists() and 
        (output_dir / dtype / 'val.txt').exists()
        for dtype in required_types
    )
    
    if all_exist:
        print(f"Test/validation datasets for {dataset_name} already exist.")
        return
    
    print(f"\nGenerating test/validation datasets for {dataset_name}...")
    print("This may take several minutes as the oracle labels non-genomic sequences...")
    
    # Run generation script
    cmd = [
        'python', 'scripts/generate_test_val_datasets.py',
        '--dataset', dataset_name,
        '--genomic-test', genomic_test,
        '--genomic-val', genomic_val,
        '--oracle-dir', oracle_dir,
        '--output-dir', 'data/test_val_sets',
        '--seed', str(seed)
    ]
    
    result = subprocess.run(cmd, cwd=project_root, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error generating datasets:\n{result.stderr}")
        raise RuntimeError("Failed to generate test/validation datasets")
    
    print("Dataset generation complete!")


def load_all_test_datasets(dataset_name: str, validation_dataset: str = 'genomic') -> tuple:
    """
    Load all test datasets and the specified validation dataset.
    
    Args:
        dataset_name: Name of dataset (e.g., 'deepstarr')
        validation_dataset: Name of validation dataset to use (e.g., 'genomic', '33noshift33lowshift34highshiftlowactivity')
        
    Returns:
        Tuple of (test_datasets_dict, val_sequences, val_labels)
    """
    base_dir = Path('data/test_val_sets') / dataset_name
    
    # Load all test datasets
    test_datasets = {}
    test_types = ['no_shift', 'no_shift_oracle', 'low_shift', 'high_shift_low_activity']
    
    for test_type in test_types:
        test_path = base_dir / test_type / 'test.txt'
        if test_path.exists():
            sequences, labels = load_test_val_dataset(str(test_path))
            test_datasets[test_type] = (sequences, labels)
            print(f"Loaded {test_type} test set: {len(sequences)} sequences")
    
    # Load validation dataset
    # Map common names to actual directory names
    val_name_mapping = {
        'genomic': 'no_shift',
        'val_genomic': 'no_shift',
        'noshift': 'no_shift',
        'no_shift': 'no_shift',
        'genomic_oracle': 'no_shift_oracle',
        'val_genomic_oracle': 'no_shift_oracle',
        'no_shift_oracle': 'no_shift_oracle'
    }
    
    val_dir_name = val_name_mapping.get(validation_dataset.lower(), validation_dataset)
    val_path = base_dir / val_dir_name / 'val.txt'
    
    if not val_path.exists():
        raise FileNotFoundError(f"Validation dataset not found: {val_path}")
    
    val_sequences, val_labels = load_test_val_dataset(str(val_path))
    print(f"Loaded {validation_dataset} validation set: {len(val_sequences)} sequences")
    
    return test_datasets, val_sequences, val_labels


def create_oracle(oracle_config: dict) -> EnsembleOracle:
    """Create oracle ensemble."""
    # Check for new composition format
    if 'composition' in oracle_config:
        return EnsembleOracle(
            composition=oracle_config['composition'],
            device=oracle_config.get('device', 'auto'),
            seqsize=oracle_config.get('seqsize', 249),
            batch_size=oracle_config.get('batch_size', 32)
        )
    else:
        # Legacy format
        return EnsembleOracle(
            model_dir=oracle_config.get('model_dir'),
            model_type=oracle_config.get('model_type', 'dream_rnn'),
            device=oracle_config.get('device', 'auto'),
            seqsize=oracle_config.get('seqsize', 249),
            batch_size=oracle_config.get('batch_size', 32)
        )


def create_trainer(trainer_config: dict, seed: int = 42) -> DeepSTARRActiveLearningTrainer:
    """Create student model trainer."""
    return DeepSTARRActiveLearningTrainer(
        model_dir=trainer_config['model_dir'],
        device=trainer_config.get('device', 'auto'),
        seqsize=trainer_config.get('seqsize', 249),
        in_channels=trainer_config.get('in_channels', 4),
        num_epochs=trainer_config.get('num_epochs', 100),
        lr=trainer_config.get('lr', 0.002),
        weight_decay=trainer_config.get('weight_decay', 1e-6),
        batch_size=trainer_config.get('batch_size', 128),
        n_workers=trainer_config.get('n_workers', 4),
        enable_replay=trainer_config.get('enable_replay', False),
        replay_buffer_size=trainer_config.get('replay_buffer_size', 1000),
        early_stopping=trainer_config.get('early_stopping', True),
        early_stopping_patience=trainer_config.get('early_stopping_patience', 10),
        lr_scheduler=trainer_config.get('lr_scheduler', 'reduce_on_plateau'),
        lr_factor=trainer_config.get('lr_factor', 0.2),
        lr_patience=trainer_config.get('lr_patience', 5),
        seed=seed
    )


def create_proposal_strategy(strategy_config: dict) -> object:
    """Create proposal strategy."""
    strategy_type = strategy_config['type']
    
    if strategy_type == 'random':
        return RandomProposalStrategy(
            seqsize=strategy_config.get('seqsize', 249),
            seed=strategy_config.get('seed', 42)
        )
    elif strategy_type == 'mixed':
        # Create individual strategies
        strategies = []
        for sub_strategy in strategy_config['strategies']:
            if sub_strategy['type'] == 'random':
                strategies.append(RandomProposalStrategy(
                    seqsize=sub_strategy.get('seqsize', 249),
                    seed=sub_strategy.get('seed', 42)
                ))
            # Add more strategy types as needed
        
        return MixedProposalStrategy(
            strategies=strategies,
            weights=strategy_config.get('weights', None),
            seqsize=strategy_config.get('seqsize', 249)
        )
    else:
        raise ValueError(f"Unknown proposal strategy: {strategy_type}")


def create_acquisition_function(acquisition_config: dict) -> object:
    """Create acquisition function."""
    acquisition_type = acquisition_config['type']
    
    if acquisition_type == 'random':
        return RandomAcquisition(seed=acquisition_config.get('seed', 42))
    elif acquisition_type == 'uncertainty':
        return UncertaintyAcquisition(seed=acquisition_config.get('seed', 42))
    else:
        raise ValueError(f"Unknown acquisition function: {acquisition_type}")


def main():
    parser = argparse.ArgumentParser(description="Run active learning pipeline")
    
    # Configuration file
    parser.add_argument("--config", required=True, help="Path to configuration JSON file")
    
    # Run index for deterministic seeding
    parser.add_argument("--run-index", type=int, default=None,
                       help="Run index for deterministic seeding (seed = 42 + index * 1000)")
    
    # Override parameters
    parser.add_argument("--n_cycles", type=int, help="Number of active learning cycles")
    parser.add_argument("--n_candidates", type=int, help="Number of candidates per cycle")
    parser.add_argument("--n_acquire", type=int, help="Number of sequences to acquire per cycle")
    parser.add_argument("--training_strategy", choices=['from_scratch', 'fine_tune'], 
                       help="Training strategy")
    parser.add_argument("--output_dir", help="Output directory (overrides config-based directory)")
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Override run_index if provided
    if args.run_index is not None:
        config['run_index'] = args.run_index
    
    # Override with command line arguments
    if args.n_cycles is not None:
        config['active_learning']['n_cycles'] = args.n_cycles
    if args.n_candidates is not None:
        config['active_learning']['n_candidates_per_cycle'] = args.n_candidates
    if args.n_acquire is not None:
        config['active_learning']['n_acquire_per_cycle'] = args.n_acquire
    if args.training_strategy is not None:
        config['active_learning']['training_strategy'] = args.training_strategy
    
    # Create configuration manager
    config_manager = ConfigurationManager(config)
    checkpoint_manager = CheckpointManager()
    
    # Get run directory from config manager
    run_dir = config_manager.get_run_directory()
    
    # Override output_dir if specified on command line
    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        output_dir = str(run_dir)
    
    # Override seeds with deterministic value from config manager
    deterministic_seed = config_manager.seed
    config['seed'] = deterministic_seed
    config['proposal_strategy']['seed'] = deterministic_seed
    config['acquisition_function']['seed'] = deterministic_seed
    
    # Set global random seeds for reproducibility
    np.random.seed(deterministic_seed)
    torch.manual_seed(deterministic_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(deterministic_seed)
    
    print(f"\n=== Configuration ===")
    print(f"Run index: {config_manager.run_index}")
    print(f"Seed: {deterministic_seed}")
    print(f"Output directory: {output_dir}")
    print(f"Dataset: {config_manager.dataset}")
    print(f"Oracle composition: {config_manager.oracle_composition}")
    print(f"Student composition: {config_manager.student_composition}")
    print(f"Proposal strategy: {config_manager.proposal_strategy}")
    print(f"Acquisition strategy: {config_manager.acquisition_strategy}")
    print(f"Candidates per cycle: {config_manager.n_candidates}")
    print(f"Acquire per cycle: {config_manager.n_acquire}")
    print(f"Round 0 initialization: {config_manager.round0_init}")
    print(f"Validation dataset: {config_manager.validation_dataset}")
    
    # Show round 0 details
    if 'round0' in config:
        print(f"  - Round 0 proposal: {config['round0']['proposal_strategy']['type']}")
        print(f"  - Round 0 acquisition: {config['round0']['acquisition_function']['type']}")
        print(f"  - Round 0 candidates: {config['round0'].get('n_candidates', 100000)}")
        print(f"  - Round 0 acquire: {config['round0'].get('n_acquire', 20000)}")
    elif 'initial_data_path' in config.get('data', {}):
        print(f"  - Using provided genomic sequences from: {config['data']['initial_data_path']}")
        print(f"  - Number of initial sequences: {config['data'].get('n_initial', 20000)}")
    print()
    
    # Save configuration to run directory
    config_manager.save_config(Path(output_dir))
    config_manager.save_metadata(Path(output_dir))
    
    # Create components
    print("Creating active learning components...")
    
    # Oracle
    oracle = create_oracle(config['oracle'])
    
    # Trainer
    trainer = create_trainer(config['trainer'], seed=deterministic_seed)
    
    # Proposal strategy
    proposal_strategy = create_proposal_strategy(config['proposal_strategy'])
    
    # Acquisition function
    acquisition_function = create_acquisition_function(config['acquisition_function'])
    
    # Load initial data (may be None if using round 0 generation)
    initial_sequences = None
    initial_labels = None
    if 'initial_data_path' in config['data'] and config['data']['initial_data_path']:
        initial_sequences, initial_labels = load_initial_data(
            config['data']['initial_data_path'],
            config['data'].get('n_initial', 20000)
        )
    
    # Create round 0 strategies if configured
    round0_proposal_strategy = None
    round0_acquisition_function = None
    round0_n_candidates = None
    round0_n_acquire = None
    
    if 'round0' in config:
        if 'proposal_strategy' in config['round0']:
            round0_proposal_strategy = create_proposal_strategy(config['round0']['proposal_strategy'])
        if 'acquisition_function' in config['round0']:
            round0_acquisition_function = create_acquisition_function(config['round0']['acquisition_function'])
        round0_n_candidates = config['round0'].get('n_candidates', 100000)
        round0_n_acquire = config['round0'].get('n_acquire', 20000)
    
    # Load/generate test and validation datasets
    print("\nLoading test and validation datasets...")
    dataset_name = config['data'].get('dataset_name', 'deepstarr').replace('_train', '')  # e.g., 'deepstarr_train' -> 'deepstarr'
    validation_dataset = config.get('validation_dataset', 'genomic')
    
    # Determine genomic test/val paths
    genomic_test_path = config['data'].get('genomic_test_path', 'data/processed/test.txt')
    genomic_val_path = config['data'].get('genomic_val_path', 'data/processed/val.txt')
    
    # Ensure test/val datasets exist (generate if needed)
    # Get oracle_dir from composition or legacy format
    if 'composition' in config['oracle']:
        oracle_dir = config['oracle']['composition'][0]['model_dir']
    else:
        oracle_dir = config['oracle']['model_dir']
    
    ensure_test_val_datasets(
        dataset_name=dataset_name,
        genomic_test=genomic_test_path,
        genomic_val=genomic_val_path,
        oracle_dir=oracle_dir,
        seed=deterministic_seed
    )
    
    # Load all test datasets and validation dataset
    test_datasets, val_sequences, val_labels = load_all_test_datasets(
        dataset_name=dataset_name,
        validation_dataset=validation_dataset
    )
    
    # Set validation data on trainer
    trainer.set_validation_data(val_sequences, val_labels)
    print(f"Validation dataset '{validation_dataset}' set on trainer")
    
    # Create active learning cycle with checkpoint support
    al_cycle = ActiveLearningCycle(
        oracle=oracle,
        trainer=trainer,
        proposal_strategy=proposal_strategy,
        acquisition_function=acquisition_function,
        output_dir=output_dir,
        initial_sequences=initial_sequences,
        initial_labels=initial_labels,
        n_cycles=config['active_learning']['n_cycles'],
        n_candidates_per_cycle=config['active_learning']['n_candidates_per_cycle'],
        n_acquire_per_cycle=config['active_learning']['n_acquire_per_cycle'],
        training_strategy=config['active_learning']['training_strategy'],
        seed=deterministic_seed,
        config_manager=config_manager,
        checkpoint_manager=checkpoint_manager,
        round0_proposal_strategy=round0_proposal_strategy,
        round0_acquisition_function=round0_acquisition_function,
        round0_n_candidates=round0_n_candidates,
        round0_n_acquire=round0_n_acquire,
        test_datasets=test_datasets,
        finetune_config=config['active_learning'].get('finetune_config')
    )
    
    # Run active learning (will auto-resume if checkpoints exist)
    print("Starting active learning cycles...")
    results = al_cycle.run_all_cycles()
    
    print("\nActive learning completed successfully!")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()


