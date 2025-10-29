"""
Results parser for extracting experiment data from hierarchical directory structure.

Parses the 11-level directory hierarchy and loads all_cycle_results.json files.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import numpy as np


class ExperimentResult:
    """Represents a single experiment result."""
    
    def __init__(self, path: Path):
        """Initialize from directory path."""
        self.path = path
        self._parse_directory_structure()
        self._load_results()
    
    def _parse_directory_structure(self):
        """Parse the 11-level directory hierarchy."""
        parts = self.path.parts
        
        # Find the results directory index
        try:
            results_idx = parts.index('results')
        except ValueError:
            raise ValueError(f"No 'results' directory found in path: {self.path}")
        
        # Extract the 11 levels after 'results'
        if len(parts) < results_idx + 12:
            raise ValueError(f"Invalid directory structure. Expected 11 levels after 'results', got {len(parts) - results_idx - 1}")
        
        hierarchy_parts = parts[results_idx + 1:results_idx + 12]
        
        # Level 1: Dataset
        self.dataset = hierarchy_parts[0]
        
        # Level 2: Oracle composition
        self.oracle_composition = hierarchy_parts[1]
        
        # Level 3: Student composition
        self.student_composition = hierarchy_parts[2]
        
        # Level 4: Proposal strategy
        self.proposal_strategy = hierarchy_parts[3]
        
        # Level 5: Acquisition strategy
        self.acquisition_strategy = hierarchy_parts[4]
        
        # Level 6: Pool sizes (parse into separate variables)
        pool_sizes = hierarchy_parts[5]
        pool_match = re.match(r'(\d+)cand_(\d+)acq', pool_sizes)
        if pool_match:
            self.n_candidates_per_cycle = int(pool_match.group(1))
            self.n_acquire_per_cycle = int(pool_match.group(2))
        else:
            raise ValueError(f"Invalid pool sizes format: {pool_sizes}")
        
        # Level 7: Round 0 initialization
        self.round0_init = hierarchy_parts[6]
        
        # Level 8: Training mode
        self.training_mode = hierarchy_parts[7]
        
        # Level 9: Validation dataset
        self.validation_dataset = hierarchy_parts[8]
        
        # Level 10: Data source
        self.data_source = hierarchy_parts[9]
        
        # Level 11: Run index
        run_index_str = hierarchy_parts[10]
        run_match = re.match(r'idx(\d+)', run_index_str)
        if run_match:
            self.run_index = int(run_match.group(1))
        else:
            raise ValueError(f"Invalid run index format: {run_index_str}")
    
    def _load_results(self):
        """Load all_cycle_results.json file."""
        results_file = self.path / "all_cycle_results.json"
        if not results_file.exists():
            raise FileNotFoundError(f"Results file not found: {results_file}")
        
        with open(results_file, 'r') as f:
            self.raw_results = json.load(f)
        
        # Organize results by cycle, keeping entry with max training sequences
        # This handles duplicate entries in existing data
        self.cycle_results = {}
        for result in self.raw_results:
            cycle = result['cycle']
            n_train = result.get('total_training_sequences', 0)
            
            if cycle not in self.cycle_results:
                self.cycle_results[cycle] = result
            else:
                # Keep entry with more training data (final trained model)
                current_n_train = self.cycle_results[cycle].get('total_training_sequences', 0)
                if n_train > current_n_train:
                    print(f"Warning: Multiple entries for cycle {cycle} in {self.path.name}. "
                          f"Keeping entry with n_train={n_train} over {current_n_train}")
                    self.cycle_results[cycle] = result
    
    def get_metrics(self, cycle: int, test_set: str, metric: str) -> Optional[float]:
        """Extract specific metric for a cycle and test set."""
        if cycle not in self.cycle_results:
            return None
        
        cycle_data = self.cycle_results[cycle]
        evaluation = cycle_data.get('evaluation_results', {})
        test_data = evaluation.get(test_set, {})
        
        return test_data.get(metric)
    
    def get_available_cycles(self) -> List[int]:
        """Get list of available cycle numbers."""
        return sorted(self.cycle_results.keys())
    
    def get_available_test_sets(self) -> List[str]:
        """Get list of available test sets."""
        test_sets = set()
        for cycle_data in self.cycle_results.values():
            evaluation = cycle_data.get('evaluation_results', {})
            test_sets.update(evaluation.keys())
        return sorted(list(test_sets))
    
    def get_available_metrics(self) -> List[str]:
        """Get list of available metrics."""
        metrics = set()
        for cycle_data in self.cycle_results.values():
            evaluation = cycle_data.get('evaluation_results', {})
            for test_data in evaluation.values():
                metrics.update(test_data.keys())
        return sorted(list(metrics))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'path': str(self.path),
            'dataset': self.dataset,
            'oracle_composition': self.oracle_composition,
            'student_composition': self.student_composition,
            'proposal_strategy': self.proposal_strategy,
            'acquisition_strategy': self.acquisition_strategy,
            'n_candidates_per_cycle': self.n_candidates_per_cycle,
            'n_acquire_per_cycle': self.n_acquire_per_cycle,
            'round0_init': self.round0_init,
            'training_mode': self.training_mode,
            'validation_dataset': self.validation_dataset,
            'run_index': self.run_index,
            'available_cycles': self.get_available_cycles(),
            'available_test_sets': self.get_available_test_sets(),
            'available_metrics': self.get_available_metrics()
        }


class ResultsCollection:
    """Collection of experiment results."""
    
    def __init__(self, experiments: List[ExperimentResult]):
        """Initialize with list of experiments."""
        self.experiments = experiments
    
    @classmethod
    def from_directory(cls, results_dir: Path) -> 'ResultsCollection':
        """Load all experiments from a results directory."""
        experiments = []
        
        # Find all idx* directories
        for idx_dir in results_dir.rglob('idx*'):
            if idx_dir.is_dir() and (idx_dir / "all_cycle_results.json").exists():
                try:
                    exp = ExperimentResult(idx_dir)
                    experiments.append(exp)
                except Exception as e:
                    print(f"Warning: Failed to load experiment from {idx_dir}: {e}")
        
        return cls(experiments)
    
    def find_experiments(self, **filters) -> 'ResultsCollection':
        """Find experiments matching the given filters."""
        matching = []
        
        for exp in self.experiments:
            match = True
            for key, value in filters.items():
                if not hasattr(exp, key):
                    print(f"Warning: Unknown filter key: {key}")
                    match = False
                    break
                
                exp_value = getattr(exp, key)
                if isinstance(value, list):
                    if exp_value not in value:
                        match = False
                        break
                else:
                    if exp_value != value:
                        match = False
                        break
            
            if match:
                matching.append(exp)
        
        return ResultsCollection(matching)
    
    def group_by(self, variable: str) -> Dict[Any, 'ResultsCollection']:
        """Group experiments by a variable."""
        groups = {}
        
        for exp in self.experiments:
            if not hasattr(exp, variable):
                raise ValueError(f"Unknown variable: {variable}")
            
            value = getattr(exp, variable)
            if value not in groups:
                groups[value] = []
            groups[value].append(exp)
        
        return {k: ResultsCollection(v) for k, v in groups.items()}
    
    def filter(self, **criteria) -> 'ResultsCollection':
        """Filter experiments by criteria."""
        return self.find_experiments(**criteria)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics of the collection."""
        if not self.experiments:
            return {'n_experiments': 0}
        
        # Get all unique values for each variable
        variables = [
            'dataset', 'oracle_composition', 'student_composition',
            'proposal_strategy', 'acquisition_strategy', 'n_candidates_per_cycle',
            'n_acquire_per_cycle', 'round0_init', 'training_mode',
            'validation_dataset', 'run_index'
        ]
        
        summary = {'n_experiments': len(self.experiments)}
        for var in variables:
            values = [getattr(exp, var) for exp in self.experiments]
            summary[var] = {
                'unique_values': sorted(list(set(values))),
                'count': len(set(values))
            }
        
        return summary
    
    def __len__(self) -> int:
        """Return number of experiments."""
        return len(self.experiments)
    
    def __iter__(self):
        """Iterate over experiments."""
        return iter(self.experiments)
