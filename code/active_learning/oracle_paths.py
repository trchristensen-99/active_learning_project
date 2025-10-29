"""
Helper functions for constructing oracle model directory paths.

Provides standardized path construction for the hierarchical oracle model directory structure:
    models/oracles/{dataset}/{architecture}/[{training_variant}/][{evoaug_sig}/]

The base path (no variants) represents the canonical DREAM-RNN training:
    - AdamW optimizer with weight_decay=0.01
    - OneCycleLR scheduler
    - Gradient clipping (max_norm=1.0)

Non-canonical training configs go in subdirectories (e.g., 'adam_no_scheduler').
"""

from pathlib import Path
from typing import Dict, Any, Optional


def get_oracle_path(
    dataset: str,
    architecture: str,
    evoaug_config: Optional[Dict[str, Any]] = None,
    training_config: Optional[Dict[str, Any]] = None,
    use_canonical_base: bool = True
) -> Path:
    """
    Construct standardized oracle model directory path.
    
    The path follows the structure:
        models/oracles/{dataset}/{architecture}/[{training_variant}/][{evoaug_sig}/]
    
    The base path (no training_variant) represents canonical DREAM-RNN training:
    - AdamW optimizer (weight_decay=0.01)
    - OneCycleLR scheduler
    - Gradient clipping (max_norm=1.0)
    
    Non-canonical training configs are placed in subdirectories.
    
    Args:
        dataset: Dataset name (e.g., 'deepstarr', 'lentimpra')
        architecture: Model architecture (e.g., 'dream_rnn', 'deepstarr')
        evoaug_config: Optional EvoAug configuration dict with 'enabled' and 'augmentations'
        training_config: Optional training configuration dict to detect if canonical
        use_canonical_base: If True (default), base path = canonical training
        
    Returns:
        Path object pointing to the oracle model directory
        
    Examples:
        >>> get_oracle_path('deepstarr', 'dream_rnn')
        PosixPath('models/oracles/deepstarr/dream_rnn')
        
        >>> evoaug = {'enabled': True, 'augmentations': {'mutation': {'enabled': True, 'rate': 0.1}}}
        >>> get_oracle_path('deepstarr', 'dream_rnn', evoaug=evoaug)
        PosixPath('models/oracles/deepstarr/dream_rnn/evoaug_mut0p1_2_hard')
        
        >>> training = {'optimizer': 'adam', 'scheduler': None}  # Non-canonical
        >>> get_oracle_path('deepstarr', 'dream_rnn', training_config=training)
        PosixPath('models/oracles/deepstarr/dream_rnn/adam_no_scheduler')
    """
    # Clean dataset and architecture names
    dataset = dataset.lower().replace('_', '')
    architecture = architecture.lower().replace('_', '')
    
    # Base path (canonical unless training_config indicates otherwise)
    base = Path("models/oracles") / dataset / architecture
    
    # Check if training config is non-canonical
    training_variant = None
    if training_config is not None and not use_canonical_base:
        training_variant = _build_training_variant_sig(training_config)
        if training_variant:
            base = base / training_variant
    
    # Add EvoAug subdirectory if enabled
    if evoaug_config and evoaug_config.get('enabled', False):
        evoaug_sig = build_evoaug_signature(evoaug_config)
        if evoaug_sig:
            return base / evoaug_sig
    
    return base


def _build_training_variant_sig(training_config: Dict[str, Any]) -> Optional[str]:
    """
    Build signature for non-canonical training configurations.
    
    Returns None if training is canonical (AdamW + OneCycleLR).
    Otherwise returns a signature like 'adam_no_scheduler'.
    
    Canonical training:
    - optimizer: 'adamw' (or AdamW)
    - scheduler: 'onecyclelr' (or OneCycleLR)
    - weight_decay: 0.01
    - gradient_clipping: True (max_norm=1.0)
    """
    optimizer = training_config.get('optimizer', 'adamw').lower()
    scheduler = training_config.get('scheduler', 'onecyclelr')
    if scheduler:
        scheduler = str(scheduler).lower()
    
    # Check if canonical
    is_canonical = (
        'adamw' in optimizer and
        ('onecycle' in scheduler or scheduler == 'onecyclelr') and
        training_config.get('weight_decay', 0.01) == 0.01 and
        training_config.get('gradient_clipping', True)
    )
    
    if is_canonical:
        return None
    
    # Build variant signature
    parts = []
    if 'adamw' not in optimizer:
        parts.append(optimizer.replace('adam', 'adam'))
    
    if not scheduler or 'onecycle' not in scheduler:
        if scheduler:
            parts.append('nosched')
        else:
            parts.append('nosched')
    else:
        parts.append(scheduler.replace('lr', '').replace('scheduler', ''))
    
    return '_'.join(parts) if parts else None


def _fmt_float_2dec(value: Any) -> str:
    """Format float with 2 decimals and replace '.' with 'p'."""
    try:
        f = float(value)
    except Exception:
        return str(value)
    s = f"{f:.2f}"
    # strip trailing zeros and dot for cleanliness, while keeping 2-dec policy overall
    s = s.rstrip('0').rstrip('.') if '.' in s else s
    if s == '':
        s = '0'
    return s.replace('.', 'p')


def build_evoaug_signature(evoaug_cfg: Dict[str, Any]) -> Optional[str]:
    """
    Build EvoAug signature per canonical spec:
    evoaug_{components}_m{MAX}_{MODE}

    Components in fixed order (included only if enabled):
      - mutation       -> mut{mutate_frac}
      - deletion       -> del{delete_min}_{delete_max}
      - translocation  -> trans{shift_min}_{shift_max}
      - insertion      -> ins{insert_min}_{insert_max}
      - inversion      -> inv{inv_min}_{inv_max}
      - reverse_complement -> rc
      - noise          -> noise{noise_std}
    Floats are rounded to 2 decimals and '.' replaced with 'p'.
    Always append _m{max_augs}_mode.
    """
    if not evoaug_cfg or not evoaug_cfg.get('enabled', False):
        return None

    augs: Dict[str, Any] = evoaug_cfg.get('augmentations', {}) or {}
    components = []

    # mutation
    mut = augs.get('mutation') or {}
    if mut.get('enabled'):
        # Accept keys mutate_frac or rate
        frac = mut.get('mutate_frac', mut.get('rate'))
        if frac is not None:
            components.append(f"mut{_fmt_float_2dec(frac)}")
        else:
            components.append("mut")

    # deletion
    dele = augs.get('deletion') or {}
    if dele.get('enabled'):
        dmin = dele.get('delete_min', dele.get('min', dele.get('start', 0)))
        dmax = dele.get('delete_max', dele.get('max', dele.get('end', dmin)))
        components.append(f"del{int(dmin)}_{int(dmax)}")

    # translocation
    trans = augs.get('translocation') or {}
    if trans.get('enabled'):
        tmin = trans.get('shift_min', trans.get('min', 0))
        tmax = trans.get('shift_max', trans.get('max', tmin))
        components.append(f"trans{int(tmin)}_{int(tmax)}")

    # insertion
    ins = augs.get('insertion') or {}
    if ins.get('enabled'):
        imin = ins.get('insert_min', ins.get('min', 0))
        imax = ins.get('insert_max', ins.get('max', imin))
        components.append(f"ins{int(imin)}_{int(imax)}")

    # inversion
    inv = augs.get('inversion') or {}
    if inv.get('enabled'):
        vmin = inv.get('inv_min', inv.get('min', 0))
        vmax = inv.get('inv_max', inv.get('max', vmin))
        components.append(f"inv{int(vmin)}_{int(vmax)}")

    # reverse complement
    rc = augs.get('reverse_complement') or {}
    if rc.get('enabled'):
        components.append("rc")

    # noise
    noise = augs.get('noise') or {}
    if noise.get('enabled'):
        std = noise.get('noise_std', noise.get('std', noise.get('sigma')))
        if std is not None:
            components.append(f"noise{_fmt_float_2dec(std)}")
        else:
            components.append("noise")

    comp_str = '_'.join(components) if components else 'none'
    max_augs = evoaug_cfg.get('max_augs_per_sequence', evoaug_cfg.get('max_augs', 2))
    mode = evoaug_cfg.get('mode', 'hard')
    return f"evoaug_{comp_str}_m{int(max_augs)}_{mode}"

def parse_oracle_path(path: Path) -> Dict[str, Any]:
    """
    Parse an oracle model path to extract metadata.
    
    Args:
        path: Path to oracle model directory
        
    Returns:
        Dictionary with 'dataset', 'architecture', 'training_variant', and 'evoaug_sig' keys.
        'training_variant' is None if canonical training (AdamW+OneCycle).
        
    Examples:
        >>> parse_oracle_path(Path('models/oracles/deepstarr/dream_rnn'))
        {'dataset': 'deepstarr', 'architecture': 'dream_rnn', 'training_variant': None, 'evoaug_sig': None}
        
        >>> parse_oracle_path(Path('models/oracles/deepstarr/dream_rnn/evoaug_mut0p1_2_hard'))
        {'dataset': 'deepstarr', 'architecture': 'dream_rnn', 'training_variant': None, 'evoaug_sig': 'evoaug_mut0p1_2_hard'}
        
        >>> parse_oracle_path(Path('models/oracles/deepstarr/dream_rnn/adam_no_scheduler'))
        {'dataset': 'deepstarr', 'architecture': 'dream_rnn', 'training_variant': 'adam_no_scheduler', 'evoaug_sig': None}
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
    
    # Check for training variant and/or evoaug
    if len(parts) > oracle_idx + 3:
        third = parts[oracle_idx + 3]
        if third.startswith('evoaug_'):
            result['training_variant'] = None
            result['evoaug_sig'] = third
        else:
            # Could be training variant
            result['training_variant'] = third
            if len(parts) > oracle_idx + 4:
                result['evoaug_sig'] = parts[oracle_idx + 4] if parts[oracle_idx + 4].startswith('evoaug_') else None
            else:
                result['evoaug_sig'] = None
    else:
        result['training_variant'] = None
        result['evoaug_sig'] = None
    
    return result


