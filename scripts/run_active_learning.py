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


def create_oracle(oracle_config: dict) -> EnsembleOracle:
    """Create oracle ensemble."""
    return EnsembleOracle(
        model_dir=oracle_config['model_dir'],
        model_type=oracle_config.get('model_type', 'dream_rnn'),
        device=oracle_config.get('device', 'auto'),
        seqsize=oracle_config.get('seqsize', 249),
        batch_size=oracle_config.get('batch_size', 32)
    )


def create_trainer(trainer_config: dict) -> DeepSTARRActiveLearningTrainer:
    """Create student model trainer."""
    return DeepSTARRActiveLearningTrainer(
        model_dir=trainer_config['model_dir'],
        device=trainer_config.get('device', 'auto'),
        seqsize=trainer_config.get('seqsize', 249),
        in_channels=trainer_config.get('in_channels', 4),
        num_epochs=trainer_config.get('num_epochs', 100),
        lr=trainer_config.get('lr', 0.001),
        weight_decay=trainer_config.get('weight_decay', 1e-6),
        batch_size=trainer_config.get('batch_size', 32),
        n_workers=trainer_config.get('n_workers', 4),
        enable_replay=trainer_config.get('enable_replay', False),
        replay_buffer_size=trainer_config.get('replay_buffer_size', 1000),
        early_stopping=trainer_config.get('early_stopping', True),
        early_stopping_patience=trainer_config.get('early_stopping_patience', 10),
        lr_scheduler=trainer_config.get('lr_scheduler', 'reduce_on_plateau'),
        lr_factor=trainer_config.get('lr_factor', 0.2),
        lr_patience=trainer_config.get('lr_patience', 5)
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
    print(f"Proposal strategy: {config_manager.proposal_strategy}")
    print(f"Acquisition strategy: {config_manager.acquisition_strategy}")
    print(f"Candidates per cycle: {config_manager.n_candidates}")
    print(f"Acquire per cycle: {config_manager.n_acquire}")
    print(f"Student architecture: {config_manager.student_arch}")
    print(f"Oracle architecture: {config_manager.oracle_arch}")
    print(f"Dataset: {config_manager.dataset}")
    print()
    
    # Save configuration to run directory
    config_manager.save_config(Path(output_dir))
    
    # Create components
    print("Creating active learning components...")
    
    # Oracle
    oracle = create_oracle(config['oracle'])
    
    # Trainer
    trainer = create_trainer(config['trainer'])
    
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
        round0_n_acquire=round0_n_acquire
    )
    
    # Run active learning (will auto-resume if checkpoints exist)
    print("Starting active learning cycles...")
    results = al_cycle.run_all_cycles()
    
    print("\nActive learning completed successfully!")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()


