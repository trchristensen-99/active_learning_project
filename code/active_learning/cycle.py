"""
Active learning cycle orchestration.

Implements the complete active learning cycle following the PIONEER approach.
"""

import numpy as np
import pandas as pd
import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from .oracle import BaseOracle, EnsembleOracle
from .student import DeepSTARRStudent
from .proposal import BaseProposalStrategy, RandomProposalStrategy, MixedProposalStrategy
from .acquisition import BaseAcquisitionFunction, RandomAcquisition, UncertaintyAcquisition
from .trainer import BaseActiveLearningTrainer, DeepSTARRActiveLearningTrainer
from .config_manager import ConfigurationManager
from .checkpoint import CheckpointManager
from ..utils import one_hot_encode

# Set up logger
logger = logging.getLogger(__name__)


class ActiveLearningCycle:
    """
    Complete active learning cycle implementation.
    
    Follows the PIONEER approach: generation -> acquisition -> labeling -> retraining
    """
    
    def __init__(
        self,
        oracle: BaseOracle,
        trainer: BaseActiveLearningTrainer,
        proposal_strategy: BaseProposalStrategy,
        acquisition_function: BaseAcquisitionFunction,
        output_dir: str,
        initial_sequences: Optional[List[str]] = None,
        initial_labels: Optional[np.ndarray] = None,
        n_cycles: int = 5,
        n_candidates_per_cycle: int = 100000,
        n_acquire_per_cycle: int = 20000,
        training_strategy: str = "from_scratch",  # "from_scratch" or "fine_tune"
        seed: int = 42,
        config_manager: Optional[ConfigurationManager] = None,
        checkpoint_manager: Optional[CheckpointManager] = None,
        round0_proposal_strategy: Optional[BaseProposalStrategy] = None,
        round0_acquisition_function: Optional[BaseAcquisitionFunction] = None,
        round0_n_candidates: Optional[int] = None,
        round0_n_acquire: Optional[int] = None,
        test_datasets: Optional[Dict[str, Tuple[List[str], np.ndarray]]] = None,
        finetune_config: Optional[Dict] = None
    ):
        """
        Initialize active learning cycle.
        
        Args:
            oracle: Oracle model for labeling sequences
            trainer: Trainer for student model
            proposal_strategy: Strategy for proposing candidate sequences
            acquisition_function: Function for selecting sequences from candidates
            output_dir: Directory to save results
            initial_sequences: Initial training sequences
            initial_labels: Initial training labels
            n_cycles: Number of active learning cycles
            n_candidates_per_cycle: Number of candidate sequences to generate per cycle
            n_acquire_per_cycle: Number of sequences to acquire per cycle
            training_strategy: Training strategy ("from_scratch" or "fine_tune")
            seed: Random seed for reproducibility
            finetune_config: Configuration for fine-tuning parameters
        """
        self.oracle = oracle
        self.trainer = trainer
        self.proposal_strategy = proposal_strategy
        self.acquisition_function = acquisition_function
        
        # Round 0 (pretraining) strategies - if None, use provided initial data
        self.round0_proposal_strategy = round0_proposal_strategy
        self.round0_acquisition_function = round0_acquisition_function
        self.round0_n_candidates = round0_n_candidates
        self.round0_n_acquire = round0_n_acquire
        
        # Checkpoint managers
        self.config_manager = config_manager
        self.checkpoint_manager = checkpoint_manager or CheckpointManager()
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Active learning parameters
        self.n_cycles = n_cycles
        self.n_candidates_per_cycle = n_candidates_per_cycle
        self.n_acquire_per_cycle = n_acquire_per_cycle
        self.training_strategy = training_strategy
        self.finetune_config = finetune_config
        
        # Set random seed
        np.random.seed(seed)
        
        # Initialize training data
        self.training_sequences = initial_sequences or []
        self.training_labels = initial_labels if initial_labels is not None else np.array([])
        
        # Test datasets for evaluation
        self.test_datasets = test_datasets or {}
        
        # Check for existing checkpoints and resume if available
        self.last_completed_round = -1
        if self.config_manager:
            self.last_completed_round = self.config_manager.find_last_completed_round(
                self.output_dir, self.n_cycles
            )
            
            # Load checkpoint if resuming
            if self.last_completed_round >= 0:
                print(f"\n*** Resuming from round {self.last_completed_round} ***")
                checkpoint = self.checkpoint_manager.load_round_checkpoint(
                    self.last_completed_round, self.output_dir
                )
                self.training_sequences = checkpoint['training_sequences']
                self.training_labels = checkpoint['training_labels']
                print(f"Loaded {len(self.training_sequences)} training sequences from checkpoint")
        
        # Results tracking
        self.cycle_results = []
        self.best_model_path = None
        
        # Save configuration
        self._save_config()
    
    def _save_config(self):
        """Save active learning configuration."""
        config = {
            'n_cycles': self.n_cycles,
            'n_candidates_per_cycle': self.n_candidates_per_cycle,
            'n_acquire_per_cycle': self.n_acquire_per_cycle,
            'training_strategy': self.training_strategy,
            'initial_n_sequences': len(self.training_sequences),
            'proposal_strategy': type(self.proposal_strategy).__name__,
            'acquisition_function': type(self.acquisition_function).__name__,
            'oracle_type': type(self.oracle).__name__,
            'trainer_type': type(self.trainer).__name__,
            'timestamp': datetime.now().isoformat()
        }
        
        config_path = self.output_dir / "active_learning_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    def run_cycle(self, cycle_idx: int) -> Dict[str, Any]:
        """
        Run a single active learning cycle.
        
        Args:
            cycle_idx: Index of the current cycle
            
        Returns:
            Cycle results dictionary
        """
        logger.info(f"=== Starting Cycle {cycle_idx} ===")
        print(f"\n=== Active Learning Cycle {cycle_idx + 1}/{self.n_cycles} ===")
        logger.info(f"Current training sequences: {len(self.training_sequences)}")
        
        # Step 1: Generate candidate sequences
        logger.info(f"Generating {self.n_candidates_per_cycle} candidate sequences...")
        print(f"1. Generating {self.n_candidates_per_cycle} candidate sequences...")
        candidate_sequences = self.proposal_strategy.propose_sequences(
            self.n_candidates_per_cycle
        )
        logger.info(f"Generated {len(candidate_sequences)} candidates")
        print(f"   Generated {len(candidate_sequences)} candidates")
        
        # Step 2: Acquire sequences using acquisition function
        logger.info(f"Selecting {self.n_acquire_per_cycle} sequences for labeling...")
        print(f"2. Selecting {self.n_acquire_per_cycle} sequences for labeling...")
        selected_sequences = self.acquisition_function.select_sequences(
            candidate_sequences,
            self.n_acquire_per_cycle,
            oracle_model=self.oracle
        )
        logger.info(f"Selected {len(selected_sequences)} sequences")
        print(f"   Selected {len(selected_sequences)} sequences")
        
        # Step 3: Label sequences using oracle
        logger.info("Labeling selected sequences with oracle...")
        print("3. Labeling selected sequences with oracle...")
        oracle_predictions, oracle_uncertainties = self.oracle.predict_with_uncertainty(
            selected_sequences
        )
        logger.info(f"Labeled {len(selected_sequences)} sequences")
        logger.info(f"Mean oracle uncertainty: {np.mean(oracle_uncertainties):.4f}")
        print(f"   Labeled {len(selected_sequences)} sequences")
        print(f"   Mean uncertainty: {np.mean(oracle_uncertainties):.4f}")
        
        # Step 4: Add new data to training set
        logger.info("Adding new data to training set...")
        print("4. Adding new data to training set...")
        self.training_sequences.extend(selected_sequences)
        if self.training_labels.size == 0:
            self.training_labels = oracle_predictions
        else:
            self.training_labels = np.vstack([self.training_labels, oracle_predictions])
        
        logger.info(f"New total training sequences: {len(self.training_sequences)}")
        print(f"   Total training sequences: {len(self.training_sequences)}")
        
        # Step 5: Train/retrain student model
        logger.info(f"Training model from scratch with {len(self.training_sequences)} sequences")
        print("5. Training student model...")
        training_results = self._train_student_model(cycle_idx)
        logger.info(f"Training completed. Best val loss: {training_results.get('best_val_loss', 'N/A')}")
        
        # Step 6: Evaluate model performance
        logger.info("Evaluating model performance on all test sets...")
        print("6. Evaluating model performance...")
        evaluation_results = self._evaluate_model(cycle_idx)
        
        # Log evaluation results
        for test_set, metrics in evaluation_results.items():
            logger.info(f"{test_set}: avg_corr={metrics.get('avg_correlation', 0):.4f}, mse={metrics.get('total_mse', 0):.4f}")
        
        # Compile cycle results
        cycle_result = {
            'cycle': cycle_idx,
            'n_candidates_generated': len(candidate_sequences),
            'n_sequences_acquired': len(selected_sequences),
            'mean_oracle_uncertainty': float(np.mean(oracle_uncertainties)),
            'std_oracle_uncertainty': float(np.std(oracle_uncertainties)),
            'total_training_sequences': len(self.training_sequences),
            'training_results': training_results,
            'evaluation_results': evaluation_results,
            'timestamp': datetime.now().isoformat()
        }
        
        # NOTE: Do NOT append to cycle_results here - it will be added in run_all_cycles
        # to avoid duplicates when saving to all_cycle_results.json
        
        # Save cycle results to individual cycle directory
        self._save_cycle_results(cycle_idx, cycle_result)
        logger.info(f"Saved results for cycle {cycle_idx} to {self.output_dir}/cycle_{cycle_idx:02d}")
        
        print(f"Cycle {cycle_idx + 1} completed successfully!")
        return cycle_result
    
    def _train_student_model(self, cycle_idx: int) -> Dict[str, Any]:
        """Train the student model using external validation set."""
        # Use all training data (no splitting) - validation set is external
        train_sequences = self.training_sequences
        train_labels = self.training_labels
        
        # Train model
        if self.training_strategy == "from_scratch" or cycle_idx == 0:
            # Pass None for val_sequences/val_labels to use external validation set
            training_results = self.trainer.train_from_scratch(
                train_sequences, train_labels, None, None
            )
        else:  # fine_tune
            training_results = self.trainer.fine_tune(
                train_sequences, train_labels,
                finetune_config=self.finetune_config
            )
        
        return training_results
    
    def _evaluate_model(self, cycle_idx: int) -> Dict[str, Any]:
        """
        Evaluate model performance on all test datasets.
        
        Returns:
            Dictionary with evaluation results for each test set type
        """
        from scipy.stats import pearsonr
        
        evaluation_results = {}
        
        # Evaluate on each test dataset type
        for test_type, (test_sequences, test_labels) in self.test_datasets.items():
            print(f"  Evaluating on {test_type} test set ({len(test_sequences)} sequences)...")
            
            # Get model predictions
            model_predictions = self.trainer.predict(test_sequences)
            
            # Calculate metrics
            dev_mse = np.mean((model_predictions[:, 0] - test_labels[:, 0]) ** 2)
            hk_mse = np.mean((model_predictions[:, 1] - test_labels[:, 1]) ** 2)
            total_mse = (dev_mse + hk_mse) / 2
            
            # Calculate correlations
            dev_corr, _ = pearsonr(model_predictions[:, 0], test_labels[:, 0])
            hk_corr, _ = pearsonr(model_predictions[:, 1], test_labels[:, 1])
            avg_corr = (dev_corr + hk_corr) / 2
            
            evaluation_results[test_type] = {
                'dev_mse': float(dev_mse),
                'hk_mse': float(hk_mse),
                'total_mse': float(total_mse),
                'dev_correlation': float(dev_corr),
                'hk_correlation': float(hk_corr),
                'avg_correlation': float(avg_corr),
                'n_test_samples': len(test_sequences)
            }
        
        # If no test datasets provided, fall back to training data subset (for backwards compatibility)
        if not evaluation_results:
            print("  Warning: No test datasets provided, evaluating on training subset...")
            n_eval = min(1000, len(self.training_sequences))
            eval_indices = np.random.choice(len(self.training_sequences), n_eval, replace=False)
            
            eval_sequences = [self.training_sequences[i] for i in eval_indices]
            eval_labels = self.training_labels[eval_indices]
            
            model_predictions = self.trainer.predict(eval_sequences)
            
            dev_mse = np.mean((model_predictions[:, 0] - eval_labels[:, 0]) ** 2)
            hk_mse = np.mean((model_predictions[:, 1] - eval_labels[:, 1]) ** 2)
            total_mse = (dev_mse + hk_mse) / 2
            
            dev_corr, _ = pearsonr(model_predictions[:, 0], eval_labels[:, 0])
            hk_corr, _ = pearsonr(model_predictions[:, 1], eval_labels[:, 1])
            avg_corr = (dev_corr + hk_corr) / 2
            
            evaluation_results['training_subset'] = {
                'dev_mse': float(dev_mse),
                'hk_mse': float(hk_mse),
                'total_mse': float(total_mse),
                'dev_correlation': float(dev_corr),
                'hk_correlation': float(hk_corr),
                'avg_correlation': float(avg_corr),
                'n_eval_samples': n_eval
            }
        
        return evaluation_results
    
    def _save_cycle_results(self, cycle_idx: int, results: Dict[str, Any]):
        """Save results for a single cycle."""
        cycle_dir = self.output_dir / f"cycle_{cycle_idx:02d}"
        cycle_dir.mkdir(exist_ok=True)
        
        # Save cycle results
        results_path = cycle_dir / "cycle_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save training data for this cycle
        training_data = {
            'sequences': self.training_sequences,
            'labels': self.training_labels.tolist()
        }
        data_path = cycle_dir / "training_data.json"
        with open(data_path, 'w') as f:
            json.dump(training_data, f, indent=2)
    
    def run_all_cycles(self) -> List[Dict[str, Any]]:
        """
        Run all active learning cycles with automatic resumption from checkpoints.
        
        n_cycles=5 means 5 AL iterations AFTER initial model, producing cycles 0-5:
        - Cycle 0: Initial model evaluation (no data acquisition)
        - Cycles 1-5: Active learning iterations (acquire data, train, evaluate)
        
        Returns:
            List of cycle results
        """
        # Determine starting round
        start_round = self.last_completed_round + 1
        
        # Check if already complete (n_cycles AL iterations + cycle 0 = n_cycles + 1 total)
        if start_round > self.n_cycles:
            print(f"\n*** All {self.n_cycles + 1} cycles (0-{self.n_cycles}) already completed! ***")
            print(f"Results are in: {self.output_dir}")
            return self._load_final_results()
        
        logger.info(f"Starting active learning with {self.n_cycles} AL cycles (cycles 0-{self.n_cycles})")
        print(f"Starting active learning with {self.n_cycles} AL cycles...")
        print(f"Initial training sequences: {len(self.training_sequences)}")
        print(f"Training strategy: {self.training_strategy}")
        print(f"Starting from round {start_round} (0=initial model, 1-{self.n_cycles}=AL cycles)")
        logger.info(f"Starting from round {start_round}")
        
        # Round 0: Baseline training (either on initial data or generated sequences)
        if start_round == 0:
            try:
                logger.info("=== Starting Round 0 (baseline pretraining) ===")
                print("\n=== Round 0 (baseline pretraining) ===")
                
                # If round 0 strategies are provided, generate and acquire sequences
                if self.round0_proposal_strategy and self.round0_acquisition_function:
                    logger.info(f"Generating {self.round0_n_candidates} candidate sequences for pretraining...")
                    print(f"Generating {self.round0_n_candidates} candidate sequences for pretraining...")
                    candidate_sequences = self.round0_proposal_strategy.propose_sequences(
                        self.round0_n_candidates
                    )
                    logger.info(f"Generated {len(candidate_sequences)} candidates")
                    print(f"   Generated {len(candidate_sequences)} candidates")
                    
                    logger.info(f"Selecting {self.round0_n_acquire} sequences for pretraining...")
                    print(f"Selecting {self.round0_n_acquire} sequences for pretraining...")
                    selected_sequences = self.round0_acquisition_function.select_sequences(
                        candidate_sequences,
                        self.round0_n_acquire,
                        oracle_model=self.oracle
                    )
                    logger.info(f"Selected {len(selected_sequences)} sequences")
                    print(f"   Selected {len(selected_sequences)} sequences")
                    
                    logger.info("Labeling selected sequences with oracle...")
                    print("Labeling selected sequences with oracle...")
                    oracle_predictions, oracle_uncertainties = self.oracle.predict_with_uncertainty(
                        selected_sequences
                    )
                    logger.info(f"Labeled {len(selected_sequences)} sequences")
                    logger.info(f"Mean oracle uncertainty: {np.mean(oracle_uncertainties):.4f}")
                    print(f"   Labeled {len(selected_sequences)} sequences")
                    print(f"   Mean uncertainty: {np.mean(oracle_uncertainties):.4f}")
                    
                    # Use generated sequences as initial training data
                    self.training_sequences = selected_sequences
                    self.training_labels = oracle_predictions
                    
                    n_candidates_generated = len(candidate_sequences)
                    n_sequences_acquired = len(selected_sequences)
                    mean_uncertainty = float(np.mean(oracle_uncertainties))
                    std_uncertainty = float(np.std(oracle_uncertainties))
                else:
                    # Use provided initial data
                    logger.info(f"Using {len(self.training_sequences)} provided initial sequences")
                    print(f"Using {len(self.training_sequences)} provided initial sequences")
                    n_candidates_generated = 0
                    n_sequences_acquired = 0
                    mean_uncertainty = None
                    std_uncertainty = None
                
                # Train on all initial data using external validation set
                logger.info(f"Training initial model with {len(self.training_sequences)} sequences")
                # Pass None for val_sequences/val_labels to use external validation set
                results = self.trainer.train_from_scratch(
                    self.training_sequences, self.training_labels, None, None
                )
                
                logger.info(f"Training completed. Best val loss: {results.get('best_val_loss', 'N/A')}")
                logger.info("Evaluating initial model on all test sets...")
                eval_results = self._evaluate_model(-1)
                
                # Log evaluation results
                for test_set, metrics in eval_results.items():
                    logger.info(f"{test_set}: avg_corr={metrics.get('avg_correlation', 0):.4f}, mse={metrics.get('total_mse', 0):.4f}")
                
                round0 = {
                    'cycle': 0,
                    'round': 0,
                    'n_candidates_generated': n_candidates_generated,
                    'n_sequences_acquired': n_sequences_acquired,
                    'mean_oracle_uncertainty': mean_uncertainty,
                    'std_oracle_uncertainty': std_uncertainty,
                    'total_training_sequences': len(self.training_sequences),
                    'training_results': results,
                    'evaluation_results': eval_results,
                    'timestamp': datetime.now().isoformat()
                }
                self.cycle_results.append(round0)
                
                # Save checkpoint for round 0
                if self.checkpoint_manager:
                    self.checkpoint_manager.save_round_checkpoint(
                        0, self.output_dir, results['model_path'], round0,
                        self.training_sequences, self.training_labels
                    )
                
                # Also save under cycle_00 for compatibility
                self._save_cycle_results(0, round0)
                with open(self.output_dir / 'round0_results.json', 'w') as f:
                    json.dump(round0, f, indent=2)
                logger.info("Round 0 completed successfully!")
                print("Round 0 completed successfully!")
                start_round = 1
            except Exception as e:
                logger.error(f"Round 0 failed: {e}", exc_info=True)
                print(f"Round 0 failed: {e}")
                raise
        
        # Run AL cycles 1 through n_cycles (inclusive)
        # This gives us cycles 0-n_cycles total (n_cycles + 1 evaluations)
        for cycle_idx in range(start_round, self.n_cycles + 1):
            try:
                logger.info(f"Starting AL cycle {cycle_idx}/{self.n_cycles}")
                cycle_result = self.run_cycle(cycle_idx)
                self.cycle_results.append(cycle_result)
                
                # Save checkpoint after each cycle
                if self.checkpoint_manager:
                    model_path = cycle_result['training_results'].get('model_path', '')
                    self.checkpoint_manager.save_round_checkpoint(
                        cycle_idx, self.output_dir, model_path, cycle_result,
                        self.training_sequences, self.training_labels
                    )
                logger.info(f"Cycle {cycle_idx} completed and checkpointed")
            except Exception as e:
                logger.error(f"Error in cycle {cycle_idx}: {e}", exc_info=True)
                print(f"Error in cycle {cycle_idx}: {e}")
                break
        
        # Save final results
        logger.info("Saving final results...")
        self._save_final_results()
        logger.info(f"Active learning completed! Ran {len(self.cycle_results)} cycles (0-{len(self.cycle_results)-1})")
        
        print(f"\nActive learning completed! Ran {len(self.cycle_results)} cycles.")
        return self.cycle_results
    
    def _save_final_results(self):
        """Save final results and summary."""
        # Save all cycle results
        results_path = self.output_dir / "all_cycle_results.json"
        with open(results_path, 'w') as f:
            json.dump(self.cycle_results, f, indent=2)
        
        # Create summary
        summary = self._create_summary()
        summary_path = self.output_dir / "active_learning_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save training data
        final_data = {
            'sequences': self.training_sequences,
            'labels': self.training_labels.tolist()
        }
        data_path = self.output_dir / "final_training_data.json"
        with open(data_path, 'w') as f:
            json.dump(final_data, f, indent=2)
    
    def _load_final_results(self) -> List[Dict[str, Any]]:
        """Load final results from disk when all cycles are already complete."""
        results_path = self.output_dir / "all_cycle_results.json"
        if results_path.exists():
            with open(results_path) as f:
                return json.load(f)
        return []
    
    def _create_summary(self) -> Dict[str, Any]:
        """Create summary of active learning results."""
        if not self.cycle_results:
            return {}
        
        # Extract key metrics across cycles
        cycles = [r['cycle'] for r in self.cycle_results]
        uncertainties = [r['mean_oracle_uncertainty'] for r in self.cycle_results if r.get('mean_oracle_uncertainty') is not None]
        correlations = [r['evaluation_results']['no_shift']['avg_correlation'] for r in self.cycle_results]
        mses = [r['evaluation_results']['no_shift']['total_mse'] for r in self.cycle_results]
        
        summary = {
            'total_cycles': len(self.cycle_results),
            'final_training_sequences': len(self.training_sequences),
            'uncertainty_trend': {
                'initial': uncertainties[0] if uncertainties else None,
                'final': uncertainties[-1] if uncertainties else None,
                'mean': float(np.mean(uncertainties)) if uncertainties else None
            },
            'performance_trend': {
                'initial_correlation': correlations[0] if correlations else None,
                'final_correlation': correlations[-1] if correlations else None,
                'best_correlation': float(np.max(correlations)) if correlations else None,
                'initial_mse': mses[0] if mses else None,
                'final_mse': mses[-1] if mses else None,
                'best_mse': float(np.min(mses)) if mses else None
            },
            'training_strategy': self.training_strategy,
            'proposal_strategy': type(self.proposal_strategy).__name__,
            'acquisition_function': type(self.acquisition_function).__name__
        }
        
        return summary


