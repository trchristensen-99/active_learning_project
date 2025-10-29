"""
Helper functions for constructing oracle model directory paths.

Provides standardized path construction for the hierarchical oracle model directory structure:
    models/oracles/{dataset}/{architecture}/[{evoaug_sig}/]
"""

from pathlib import Path
from typing import Dict, Any, Optional


def get_oracle_path(
    dataset: str,
    architecture: str,
    evoaug_config: Optional[Dict[str, Any]] = None
) -> Path:
    """
    Construct standardized oracle model directory path.
    
    The path follows the structure:
        models/oracles/{dataset}/{architecture}/[{evoaug_sig}/]
    
    Where evoaug_sig is only present if EvoAug augmentations are enabled.
    
    Args:
        dataset: Dataset name (e.g., 'deepstarr', 'lentimpra')
        architecture: Model architecture (e.g., 'dream_rnn', 'deepstarr')
        evoaug_config: Optional EvoAug configuration dict with 'enabled' and 'augmentations'
        
    Returns:
        Path object pointing to the oracle model directory
        
    Examples:
        >>> get_oracle_path('deepstarr', 'dream_rnn')
        PosixPath('models/oracles/deepstarr/dream_rnn')
        
        >>> evoaug = {'enabled': True, 'augmentations': {'mutation': {'enabled': True, 'rate': 0.1}}}
        >>> get_oracle_path('deepstarr', 'dream_rnn', evoaug)
        PosixPath('models/oracles/deepstarr/dream_rnn/evoaug_mut0p1_2_hard')
    """
    # Clean dataset and architecture names
    dataset = dataset.lower().replace('_', '')
    architecture = architecture.lower().replace('_', '')
    
    # Base path
    base = Path("models/oracles") / dataset / architecture
    
    # Add EvoAug subdirectory if enabled
    if evoaug_config and evoaug_config.get('enabled', False):
        from code.active_learning.config_manager import ConfigurationManager
        temp_config = {"oracle": {"evoaug": evoaug_config}}
        cfgm = ConfigurationManager(temp_config)
        evoaug_sig = cfgm._build_evoaug_signature(evoaug_config)
        if evoaug_sig:
            return base / evoaug_sig
    
    return base


def parse_oracle_path(path: Path) -> Dict[str, Any]:
    """
    Parse an oracle model path to extract metadata.
    
    Args:
        path: Path to oracle model directory
        
    Returns:
        Dictionary with 'dataset', 'architecture', and optionally 'evoaug_sig' keys
        
    Examples:
        >>> parse_oracle_path(Path('models/oracles/deepstarr/dream_rnn'))
        {'dataset': 'deepstarr', 'architecture': 'dream_rnn', 'evoaug_sig': None}
        
        >>> parse_oracle_path(Path('models/oracles/deepstarr/dream_rnn/evoaug_mut0p1_2_hard'))
        {'dataset': 'deepstarr', 'architecture': 'dream_rnn', 'evoaug_sig': 'evoaug_mut0p1_2_hard'}
    """
    parts = path.parts
    
    # Find 'oracles' in path
    try:
        oracle_idx = parts.index('oracles')
    except ValueError:
        raise ValueError(f"Path does not contain 'oracles' directory: {path}")
    
    # Extract components
    result = {}
    if len(parts) > oracle_idx + 1:
        result['dataset'] = parts[oracle_idx + 1]
    if len(parts) > oracle_idx + 2:
        result['architecture'] = parts[oracle_idx + 2]
    if len(parts) > oracle_idx + 3 and parts[oracle_idx + 3].startswith('evoaug_'):
        result['evoaug_sig'] = parts[oracle_idx + 3]
    else:
        result['evoaug_sig'] = None
    
    return result


