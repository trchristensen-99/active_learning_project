"""
Configuration management for reproducible active learning runs.

Handles configuration hashing, directory structure, and seed management.
"""

import json
from pathlib import Path
from typing import Dict, Any


class ConfigurationManager:
    """
    Manages configuration-based directory structure and seeding for reproducible runs.
    
    Each configuration gets a unique hierarchical directory based on:
    - Proposal strategy
    - Acquisition strategy
    - Pool sizes (candidates, acquisitions)
    - Student architecture
    - Oracle architecture
    - Dataset
    - Run index (determines seed)
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize configuration manager.
        
        Args:
            config: Full configuration dictionary
        """
        self.config = config
        
        # Extract key parameters that define unique configurations
        self.proposal_strategy = config['proposal_strategy']['type']
        self.acquisition_strategy = config['acquisition_function']['type']
        self.n_candidates = config['active_learning']['n_candidates_per_cycle']
        self.n_acquire = config['active_learning']['n_acquire_per_cycle']
        self.run_index = config.get('run_index', 0)
        self.student_arch = config['trainer'].get('architecture', 'deepstarr')
        self.oracle_arch = config['oracle'].get('architecture', 'dream_rnn_ensemble')
        self.dataset = config['data'].get('dataset_name', 'train')
        
        # Calculate deterministic seed from index
        # Formula: seed = 42 + index * 1000
        self.seed = 42 + self.run_index * 1000
    
    def get_run_directory(self) -> Path:
        """
        Generate hierarchical directory path for this configuration.
        
        Returns:
            Path object for the run directory
            
        Example:
            results/random/uncertainty/100000cand_20000acq/deepstarr/dream_rnn_ensemble/train/idx0/
        """
        return Path('results') / \
               self.proposal_strategy / \
               self.acquisition_strategy / \
               f"{self.n_candidates}cand_{self.n_acquire}acq" / \
               self.student_arch / \
               self.oracle_arch / \
               self.dataset / \
               f"idx{self.run_index}"
    
    def find_last_completed_round(self, run_dir: Path, n_total_cycles: int) -> int:
        """
        Find the last round with complete results.
        
        Args:
            run_dir: Run directory to check
            n_total_cycles: Total number of AL cycles (not including round 0)
            
        Returns:
            Last completed round number (-1 if none, 0 for baseline, 1-N for AL cycles)
        """
        # Start from -1 (no rounds completed)
        last_completed = -1
        
        # Check round 0 (baseline) through round n_total_cycles
        for round_num in range(n_total_cycles + 1):
            round_dir = run_dir / f"round_{round_num:03d}"
            
            # Check if round directory exists and has required files
            if not round_dir.exists():
                break
            
            required_files = [
                round_dir / "model_best.pth",
                round_dir / "metrics.json",
                round_dir / "training_data.json"
            ]
            
            if all(f.exists() for f in required_files):
                last_completed = round_num
            else:
                # Incomplete round found, stop here
                break
        
        return last_completed
    
    def save_config(self, run_dir: Path):
        """
        Save full configuration to run directory.
        
        Args:
            run_dir: Directory to save configuration
        """
        run_dir.mkdir(parents=True, exist_ok=True)
        config_path = run_dir / "config.json"
        
        # Add computed values to config
        config_with_metadata = self.config.copy()
        config_with_metadata['_metadata'] = {
            'seed': self.seed,
            'run_directory': str(run_dir),
            'configuration_hash': self._get_config_hash()
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_with_metadata, f, indent=2)
    
    def _get_config_hash(self) -> str:
        """
        Generate a human-readable configuration identifier.
        
        Returns:
            Configuration hash string
        """
        return f"{self.proposal_strategy}_{self.acquisition_strategy}_" \
               f"{self.n_candidates}c_{self.n_acquire}a_" \
               f"{self.student_arch}_{self.oracle_arch}_" \
               f"{self.dataset}_idx{self.run_index}"

