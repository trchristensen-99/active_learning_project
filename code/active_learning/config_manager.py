"""
Configuration management for reproducible active learning runs.

Handles configuration hashing, directory structure, and seed management.
"""

import json
import subprocess
import socket
from pathlib import Path
from typing import Dict, Any, List, Tuple
from datetime import datetime


class ConfigurationManager:
    """
    Manages configuration-based directory structure and seeding for reproducible runs.
    
    New hierarchical structure (v2):
    - Dataset (e.g., deepstarr, lentimpra)
    - Oracle composition (e.g., 5dreamrnn, 3deepstarr+5dreamrnn)
    - Student composition (e.g., 1deepstarr, 5deepstarr)
    - Proposal strategy (e.g., 100random_proposal, 50mixed_50random_proposal)
    - Acquisition strategy (e.g., 100random_acquisition, 50random_50uncertainty_acquisition)
    - Pool sizes (e.g., 100000cand_20000acq)
    - Round 0 initialization (e.g., init_prop_genomic_acq_random_20k)
    - Validation dataset (e.g., val_genomic, val_33noshift_33lowshift_34highshift)
    - Run index (determines seed: 42 + index * 1000)
    
    Example:
        results/deepstarr/5dreamrnn/1deepstarr/100random_proposal/
        100random_acquisition/100000cand_20000acq/init_prop_genomic_acq_random_20k/
        val_genomic/idx1/
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize configuration manager.
        
        Args:
            config: Full configuration dictionary
        """
        self.config = config
        self.run_index = config.get('run_index', 0)
        
        # Extract and format all components
        self.dataset = self._get_dataset_name(config)
        self.oracle_composition = self._get_oracle_composition(config)
        self.student_composition = self._get_student_composition(config)
        self.proposal_strategy = self._get_proposal_strategy(config)
        self.acquisition_strategy = self._get_acquisition_strategy(config)
        self.n_candidates = config['active_learning']['n_candidates_per_cycle']
        self.n_acquire = config['active_learning']['n_acquire_per_cycle']
        self.round0_init = self._get_round0_init(config)
        self.validation_dataset = self._get_validation_dataset(config)
        
        # Calculate deterministic seed from index
        # Formula: seed = 42 + index * 1000
        self.seed = 42 + self.run_index * 1000
    
    def _get_dataset_name(self, config: Dict[str, Any]) -> str:
        """Extract and clean dataset name."""
        dataset = config['data'].get('dataset_name', 'unknown')
        # Remove '_train' suffix if present
        if dataset.endswith('_train'):
            dataset = dataset[:-6]
        return dataset.lower()
    
    def _get_oracle_composition(self, config: Dict[str, Any]) -> str:
        """
        Get oracle composition string.
        
        Format: {n}{modeltype}[+{n}{modeltype}]*
        Models in alphabetical order for mixed ensembles.
        
        Examples:
            - 5dreamrnn
            - 3deepstarr+5dreamrnn
        """
        if 'composition' in config.get('oracle', {}):
            # New composition format
            composition = config['oracle']['composition']
            if isinstance(composition, list):
                # Sort by model type alphabetically
                sorted_comp = sorted(composition, key=lambda x: x['type'])
                parts = []
                for comp in sorted_comp:
                    count = comp.get('count', 1)
                    model_type = comp['type'].replace('_', '').lower()
                    parts.append(f"{count}{model_type}")
                return '+'.join(parts)
        
        # Fallback: infer from model_dir or architecture
        model_dir = config.get('oracle', {}).get('model_dir', '')
        architecture = config.get('oracle', {}).get('architecture', 'unknown')
        
        # Try to count models in directory
        if model_dir and Path(model_dir).exists():
            model_dirs = [d for d in Path(model_dir).iterdir() 
                         if d.is_dir() and d.name.startswith('model_')]
            count = len(model_dirs) if model_dirs else 1
        else:
            # Default count for known ensembles
            count = 5 if 'ensemble' in architecture else 1
        
        # Clean architecture name
        arch_clean = architecture.replace('_ensemble', '').replace('_', '').lower()
        return f"{count}{arch_clean}"
    
    def _get_student_composition(self, config: Dict[str, Any]) -> str:
        """
        Get student composition string.
        
        Format: {n}{modeltype}[+{n}{modeltype}]*[_hyperparam_suffix]
        
        Examples:
            - 1deepstarr
            - 5deepstarr
            - 3deepstarr+2cnn
            - 1deepstarr_e200_lr0005
        """
        trainer_config = config.get('trainer', {})
        
        if 'composition' in trainer_config:
            # New composition format for ensembles
            composition = trainer_config['composition']
            if isinstance(composition, list):
                sorted_comp = sorted(composition, key=lambda x: x['type'])
                parts = []
                for comp in sorted_comp:
                    count = comp.get('count', 1)
                    model_type = comp['type'].replace('_', '').lower()
                    parts.append(f"{count}{model_type}")
                base_name = '+'.join(parts)
            else:
                base_name = f"1{composition.lower()}"
        else:
            # Single model (default)
            architecture = trainer_config.get('architecture', 'deepstarr')
            arch_clean = architecture.replace('_', '').lower()
            base_name = f"1{arch_clean}"
        
        # Add hyperparameter suffix for non-default configs
        hyperparam_suffix = self._get_hyperparam_suffix(trainer_config)
        if hyperparam_suffix:
            return f"{base_name}_{hyperparam_suffix}"
        return base_name
    
    def _get_hyperparam_suffix(self, trainer_config: Dict[str, Any]) -> str:
        """
        Generate hyperparameter suffix for non-default configurations.
        
        Format: {param1}{value1}_{param2}{value2}
        Parameters in alphabetical order.
        
        Returns empty string if all parameters are default.
        """
        # Define default values
        defaults = {
            'num_epochs': 100,
            'batch_size': 8192,
            'lr': 0.001,
            'weight_decay': 1e-6,
            'lr_scheduler': 'reduce_on_plateau'
        }
        
        # Check for explicit non-default hyperparameters
        if 'hyperparameters' in trainer_config:
            hyperparams = trainer_config['hyperparameters']
        else:
            # Check individual fields
            hyperparams = {}
            for key, default_val in defaults.items():
                if key in trainer_config and trainer_config[key] != default_val:
                    hyperparams[key] = trainer_config[key]
        
        if not hyperparams:
            return ""
        
        # Format hyperparameters (alphabetically)
        parts = []
        for key in sorted(hyperparams.keys()):
            value = hyperparams[key]
            # Abbreviate common parameter names
            abbrev_map = {
                'num_epochs': 'e',
                'batch_size': 'bs',
                'lr': 'lr',
                'weight_decay': 'wd',
                'lr_scheduler': 'sched'
            }
            abbrev = abbrev_map.get(key, key)
            
            # Format value (remove decimal points, handle scientific notation)
            if isinstance(value, float):
                # Convert to string, remove decimal and leading zeros
                val_str = f"{value:.10f}".rstrip('0').rstrip('.').replace('.', '')
                if val_str.startswith('0'):
                    val_str = val_str.lstrip('0')
                parts.append(f"{abbrev}{val_str}")
            else:
                parts.append(f"{abbrev}{value}")
        
        return '_'.join(parts)
    
    def _get_proposal_strategy(self, config: Dict[str, Any]) -> str:
        """
        Get proposal strategy string with percentages.
        
        Format: {pct1}{strategy1}[_{pct2}{strategy2}]*_proposal
        Strategies in alphabetical order.
        
        Examples:
            - 100random_proposal
            - 50mixed_50random_proposal
        """
        proposal_config = config.get('proposal_strategy', {})
        
        if 'strategies' in proposal_config:
            # Multiple strategies with percentages
            strategies = proposal_config['strategies']
            sorted_strats = sorted(strategies, key=lambda x: x['type'])
            parts = []
            for strat in sorted_strats:
                pct = strat.get('percentage', 100)
                strategy_type = strat['type'].lower()
                parts.append(f"{pct}{strategy_type}")
            return '_'.join(parts) + '_proposal'
        else:
            # Single strategy (100%)
            strategy_type = proposal_config.get('type', 'unknown').lower()
            return f"100{strategy_type}_proposal"
    
    def _get_acquisition_strategy(self, config: Dict[str, Any]) -> str:
        """
        Get acquisition strategy string with percentages.
        
        Format: {pct1}{strategy1}[_{pct2}{strategy2}]*_acquisition
        Strategies in alphabetical order.
        
        Examples:
            - 100random_acquisition
            - 30diversity_70uncertainty_acquisition
        """
        acq_config = config.get('acquisition_function', {})
        
        if 'strategies' in acq_config:
            # Multiple strategies with percentages
            strategies = acq_config['strategies']
            sorted_strats = sorted(strategies, key=lambda x: x['type'])
            parts = []
            for strat in sorted_strats:
                pct = strat.get('percentage', 100)
                strategy_type = strat['type'].lower()
                parts.append(f"{pct}{strategy_type}")
            return '_'.join(parts) + '_acquisition'
        else:
            # Single strategy (100%)
            strategy_type = acq_config.get('type', 'unknown').lower()
            return f"100{strategy_type}_acquisition"
    
    def _get_round0_init(self, config: Dict[str, Any]) -> str:
        """
        Get round 0 initialization string.
        
        Format: init_prop_{prop_strategies}_acq_{acq_strategies}_{size}
        For multiple data sources: append _{source1}{size1}_{source2}{size2}
        
        Examples:
            - init_prop_genomic_acq_random_20k
            - init_prop_random_acq_uncertainty_10k
            - init_prop_genomic_random_acq_random_genomic10k_random10k
        """
        data_config = config.get('data', {})
        n_initial = data_config.get('n_initial', 0)
        
        # Format size
        if n_initial >= 1000:
            size_str = f"{n_initial // 1000}k"
        else:
            size_str = str(n_initial)
        
        if 'round0' in config and config['round0']:
            # Generated sequences (round0 config)
            round0_config = config['round0']
            
            # Proposal strategies
            prop_config = round0_config.get('proposal_strategy', {})
            if 'strategies' in prop_config:
                prop_parts = sorted([s['type'].lower() for s in prop_config['strategies']])
                prop_str = '_'.join(prop_parts)
            else:
                prop_str = prop_config.get('type', 'unknown').lower()
            
            # Acquisition strategies
            acq_config = round0_config.get('acquisition_function', {})
            if 'strategies' in acq_config:
                acq_parts = sorted([s['type'].lower() for s in acq_config['strategies']])
                acq_str = '_'.join(acq_parts)
            else:
                acq_str = acq_config.get('type', 'unknown').lower()
            
            return f"init_prop_{prop_str}_acq_{acq_str}_{size_str}"
        
        elif 'initial_data_path' in data_config:
            # Genomic sequences from file
            return f"init_prop_genomic_acq_random_{size_str}"
        
        elif 'initial_data_sources' in data_config:
            # Multiple data sources
            sources = data_config['initial_data_sources']
            sorted_sources = sorted(sources, key=lambda x: x['type'])
            
            # Extract unique proposal and acquisition strategies
            prop_types = sorted(set(s.get('proposal', 'genomic').lower() for s in sources))
            acq_types = sorted(set(s.get('acquisition', 'random').lower() for s in sources))
            
            prop_str = '_'.join(prop_types)
            acq_str = '_'.join(acq_types)
            
            # Add source-specific sizes
            source_parts = []
            for source in sorted_sources:
                source_type = source['type'].lower()
                source_size = source.get('size', 0)
                if source_size >= 1000:
                    source_size_str = f"{source_size // 1000}k"
                else:
                    source_size_str = str(source_size)
                source_parts.append(f"{source_type}{source_size_str}")
            
            return f"init_prop_{prop_str}_acq_{acq_str}_{'_'.join(source_parts)}"
        
        else:
            return f"init_unknown_{size_str}"
    
    def _get_validation_dataset(self, config: Dict[str, Any]) -> str:
        """
        Get validation dataset string.
        
        Format: val_{name} or val_{pct1}{type1}_{pct2}{type2}*
        Types in alphabetical order for mixed datasets.
        
        Examples:
            - val_genomic
            - val_noshift
            - val_33highshiftlowactivity_33lowshift_34noshift
        """
        val_config = config.get('validation_dataset', 'genomic')
        
        if isinstance(val_config, str):
            # Simple string name - ensure it has val_ prefix
            if val_config.startswith('val_'):
                return val_config.lower()
            else:
                return f"val_{val_config.lower()}"
        
        elif isinstance(val_config, dict) and 'mix' in val_config:
            # Mixed validation dataset with percentages
            mix = val_config['mix']
            sorted_types = sorted(mix.items(), key=lambda x: x[0])
            parts = []
            for dataset_type, percentage in sorted_types:
                # Clean dataset type name (remove spaces, lowercase)
                type_clean = dataset_type.replace('_', '').replace(' ', '').lower()
                parts.append(f"{percentage}{type_clean}")
            return 'val_' + '_'.join(parts)
        
        else:
            return "val_unknown"
    
    def get_run_directory(self) -> Path:
        """
        Generate hierarchical directory path for this configuration.
        
        Returns:
            Path object for the run directory
            
        Example:
            results/deepstarr/5dreamrnn/1deepstarr/100random_proposal/
            100random_acquisition/100000cand_20000acq/init_prop_genomic_acq_random_20k/
            val_genomic/idx1/
        """
        return Path('results') / \
               self.dataset / \
               self.oracle_composition / \
               self.student_composition / \
               self.proposal_strategy / \
               self.acquisition_strategy / \
               f"{self.n_candidates}cand_{self.n_acquire}acq" / \
               self.round0_init / \
               self.validation_dataset / \
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
    
    def save_metadata(self, run_dir: Path):
        """
        Save experiment metadata (git hash, timestamp, hardware info, etc.).
        
        Args:
            run_dir: Directory to save metadata
        """
        run_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = run_dir / "metadata.json"
        
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'hostname': socket.gethostname(),
            'run_index': self.run_index,
            'seed': self.seed,
            'directory_structure': {
                'dataset': self.dataset,
                'oracle_composition': self.oracle_composition,
                'student_composition': self.student_composition,
                'proposal_strategy': self.proposal_strategy,
                'acquisition_strategy': self.acquisition_strategy,
                'pool_sizes': f"{self.n_candidates}cand_{self.n_acquire}acq",
                'round0_init': self.round0_init,
                'validation_dataset': self.validation_dataset
            }
        }
        
        # Try to get git commit hash
        try:
            git_hash = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'],
                stderr=subprocess.DEVNULL,
                text=True
            ).strip()
            metadata['git_commit'] = git_hash
        except:
            metadata['git_commit'] = 'unknown'
        
        # Try to get GPU info
        try:
            import torch
            if torch.cuda.is_available():
                metadata['gpu'] = {
                    'count': torch.cuda.device_count(),
                    'devices': [torch.cuda.get_device_name(i) 
                               for i in range(torch.cuda.device_count())]
                }
                metadata['cuda_version'] = torch.version.cuda
                metadata['pytorch_version'] = torch.__version__
        except:
            pass
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _get_config_hash(self) -> str:
        """
        Generate a human-readable configuration identifier.
        
        Returns:
            Configuration hash string
        """
        return f"{self.dataset}_{self.oracle_composition}_{self.student_composition}_" \
               f"{self.proposal_strategy}_{self.acquisition_strategy}_" \
               f"{self.n_candidates}c_{self.n_acquire}a_idx{self.run_index}"
