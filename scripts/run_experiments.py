#!/usr/bin/env python3
"""
Flexible experiment runner for active learning with GPU management.

This script allows running multiple active learning experiments in parallel
with different configurations, automatically assigning them to available GPUs.

Usage:
    # Run from a YAML configuration file
    python scripts/run_experiments.py --config experiments.yaml
    
    # Run specific experiments
    python scripts/run_experiments.py \
        --base-config configs/active_learning_random_random.json \
        --run-indices 1 2 3 \
        --gpus 1 2 3
    
    # Auto-detect available GPUs
    python scripts/run_experiments.py \
        --base-config configs/active_learning_random_random.json \
        --run-indices 1 2 3 4 5 \
        --auto-gpu
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import yaml


def get_gpu_info() -> List[Dict[str, Any]]:
    """
    Get information about available GPUs.
    
    Returns:
        List of dicts with GPU info (index, name, utilization, memory)
    """
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used,memory.total',
             '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            check=True
        )
        
        gpus = []
        for line in result.stdout.strip().split('\n'):
            if line:
                parts = [p.strip() for p in line.split(',')]
                gpus.append({
                    'index': int(parts[0]),
                    'name': parts[1],
                    'utilization': int(parts[2]),
                    'memory_used': int(parts[3]),
                    'memory_total': int(parts[4])
                })
        return gpus
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Warning: Could not query GPU information")
        return []


def get_available_gpus(min_free_memory_mb: int = 1000, max_utilization: int = 10) -> List[int]:
    """
    Get list of available GPU indices based on usage criteria.
    
    Args:
        min_free_memory_mb: Minimum free memory required (MB)
        max_utilization: Maximum GPU utilization percentage
        
    Returns:
        List of available GPU indices
    """
    gpus = get_gpu_info()
    available = []
    
    for gpu in gpus:
        free_memory = gpu['memory_total'] - gpu['memory_used']
        if free_memory >= min_free_memory_mb and gpu['utilization'] <= max_utilization:
            available.append(gpu['index'])
    
    return available


def print_gpu_status():
    """Print current GPU status."""
    gpus = get_gpu_info()
    if not gpus:
        print("No GPU information available")
        return
    
    print("\n" + "="*80)
    print("GPU Status")
    print("="*80)
    print(f"{'GPU':<5} {'Name':<25} {'Util%':<8} {'Memory':<20} {'Status':<10}")
    print("-"*80)
    
    for gpu in gpus:
        free_memory = gpu['memory_total'] - gpu['memory_used']
        memory_str = f"{gpu['memory_used']}/{gpu['memory_total']} MB"
        status = "Available" if free_memory > 1000 and gpu['utilization'] < 10 else "In Use"
        
        print(f"{gpu['index']:<5} {gpu['name']:<25} {gpu['utilization']:<8} {memory_str:<20} {status:<10}")
    print("="*80 + "\n")


def create_experiment_config(
    base_config_path: str,
    run_index: int,
    proposal_strategy: Optional[Dict] = None,
    acquisition_function: Optional[Dict] = None,
    round0_config: Optional[Dict] = None,
    **overrides
) -> Dict[str, Any]:
    """
    Create an experiment configuration by modifying a base config.
    
    Args:
        base_config_path: Path to base configuration JSON
        run_index: Run index for this experiment
        proposal_strategy: Override proposal strategy config
        acquisition_function: Override acquisition function config
        round0_config: Override round 0 configuration
        **overrides: Additional config overrides
        
    Returns:
        Modified configuration dictionary
    """
    with open(base_config_path) as f:
        config = json.load(f)
    
    # Set run index
    config['run_index'] = run_index
    
    # Override strategies if provided
    if proposal_strategy:
        config['proposal_strategy'] = proposal_strategy
    if acquisition_function:
        config['acquisition_function'] = acquisition_function
    if round0_config:
        config['round0'] = round0_config
    
    # Apply additional overrides
    for key, value in overrides.items():
        keys = key.split('.')
        current = config
        for k in keys[:-1]:
            current = current.setdefault(k, {})
        current[keys[-1]] = value
    
    return config


def run_experiment(
    config: Dict[str, Any],
    gpu_id: int,
    log_dir: Path,
    use_gpu_wrapper: bool = True
) -> subprocess.Popen:
    """
    Launch an active learning experiment on a specific GPU.
    
    Args:
        config: Experiment configuration
        gpu_id: GPU index to use
        log_dir: Directory for log files
        use_gpu_wrapper: Whether to use GPU wrapper for CUDA compatibility
        
    Returns:
        Subprocess handle
    """
    # Create temporary config file
    config_path = log_dir / f"config_idx{config['run_index']}_gpu{gpu_id}.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Create log file path
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"experiment_idx{config['run_index']}_gpu{gpu_id}_{timestamp}.log"
    
    # Build command
    if use_gpu_wrapper:
        cmd = ['./scripts/gpu_wrapper.sh', 'python', 'scripts/run_active_learning.py']
    else:
        cmd = ['python', 'scripts/run_active_learning.py']
    
    cmd.extend(['--config', str(config_path)])
    
    # Set environment with GPU assignment
    env = subprocess.os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # Launch process
    print(f"Launching experiment (index={config['run_index']}, GPU={gpu_id})")
    print(f"  Config: {config_path}")
    print(f"  Log: {log_path}")
    
    with open(log_path, 'w') as log_file:
        process = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=Path.cwd()
        )
    
    return process


def load_experiment_batch(yaml_path: str) -> List[Dict[str, Any]]:
    """
    Load a batch of experiments from a YAML file.
    
    YAML format:
        base_config: path/to/base_config.json
        experiments:
          - run_index: 1
            gpu: 1
            proposal_strategy:
              type: random
            acquisition_function:
              type: uncertainty
          - run_index: 2
            gpu: 2
            ...
    
    Args:
        yaml_path: Path to YAML configuration
        
    Returns:
        List of experiment specifications
    """
    with open(yaml_path) as f:
        batch = yaml.safe_load(f)
    
    return batch


def main():
    parser = argparse.ArgumentParser(
        description="Run active learning experiments in parallel with GPU management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run from YAML batch file
  python scripts/run_experiments.py --config experiments.yaml
  
  # Run specific indices on specific GPUs
  python scripts/run_experiments.py \\
      --base-config configs/active_learning_random_random.json \\
      --run-indices 1 2 3 \\
      --gpus 1 2 3
  
  # Auto-assign to available GPUs
  python scripts/run_experiments.py \\
      --base-config configs/active_learning_random_random.json \\
      --run-indices 1 2 3 4 5 \\
      --auto-gpu
  
  # Show GPU status and exit
  python scripts/run_experiments.py --show-gpus
        """
    )
    
    # Input methods
    parser.add_argument('--config', type=str,
                       help='YAML file with batch experiment configuration')
    parser.add_argument('--base-config', type=str,
                       help='Base JSON configuration file')
    parser.add_argument('--run-indices', type=int, nargs='+',
                       help='Run indices to execute')
    
    # GPU assignment
    parser.add_argument('--gpus', type=int, nargs='+',
                       help='GPU indices to use (must match number of run indices)')
    parser.add_argument('--auto-gpu', action='store_true',
                       help='Automatically assign to available GPUs')
    parser.add_argument('--show-gpus', action='store_true',
                       help='Show GPU status and exit')
    
    # Options
    parser.add_argument('--log-dir', type=str, default='logs',
                       help='Directory for log files (default: logs)')
    parser.add_argument('--no-gpu-wrapper', action='store_true',
                       help='Do not use GPU wrapper (disable CUDA compatibility layer)')
    parser.add_argument('--wait', action='store_true',
                       help='Wait for all experiments to complete')
    
    args = parser.parse_args()
    
    # Show GPU status if requested
    if args.show_gpus:
        print_gpu_status()
        return 0
    
    # Create log directory
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine experiments to run
    experiments = []
    
    if args.config:
        # Load from YAML batch file
        print(f"Loading experiments from {args.config}")
        batch = load_experiment_batch(args.config)
        base_config = batch.get('base_config')
        
        for exp in batch.get('experiments', []):
            # Extract special keys (make a copy to avoid modifying original)
            exp_copy = exp.copy()
            run_index = exp_copy.pop('run_index')
            gpu = exp_copy.pop('gpu', None)
            proposal_strategy = exp_copy.pop('proposal_strategy', None)
            acquisition_function = exp_copy.pop('acquisition_function', None)
            round0_config = exp_copy.pop('round0', None)
            
            # Remaining keys are treated as overrides (with dot notation support)
            overrides = {}
            for key, value in exp_copy.items():
                if isinstance(value, dict):
                    # Flatten nested dicts with dot notation
                    for subkey, subvalue in value.items():
                        overrides[f"{key}.{subkey}"] = subvalue
                else:
                    overrides[key] = value
            
            config = create_experiment_config(
                base_config,
                run_index,
                proposal_strategy=proposal_strategy,
                acquisition_function=acquisition_function,
                round0_config=round0_config,
                **overrides
            )
            experiments.append({
                'config': config,
                'gpu': gpu
            })
    
    elif args.base_config and args.run_indices:
        # Create from command-line arguments
        print(f"Creating experiments from {args.base_config}")
        
        # Determine GPU assignments
        if args.auto_gpu:
            available_gpus = get_available_gpus()
            if len(available_gpus) < len(args.run_indices):
                print(f"Warning: Only {len(available_gpus)} GPUs available for {len(args.run_indices)} experiments")
                print("Some experiments will be queued")
            gpus = available_gpus * (len(args.run_indices) // len(available_gpus) + 1)
            gpus = gpus[:len(args.run_indices)]
        elif args.gpus:
            if len(args.gpus) != len(args.run_indices):
                print(f"Error: Number of GPUs ({len(args.gpus)}) must match number of run indices ({len(args.run_indices)})")
                return 1
            gpus = args.gpus
        else:
            print("Error: Must specify --gpus or --auto-gpu")
            return 1
        
        for run_index, gpu in zip(args.run_indices, gpus):
            config = create_experiment_config(args.base_config, run_index)
            experiments.append({
                'config': config,
                'gpu': gpu
            })
    
    else:
        print("Error: Must provide either --config or (--base-config and --run-indices)")
        parser.print_help()
        return 1
    
    # Show GPU status
    print_gpu_status()
    
    # Launch experiments
    print(f"\nLaunching {len(experiments)} experiment(s)...\n")
    processes = []
    
    for exp in experiments:
        process = run_experiment(
            exp['config'],
            exp['gpu'],
            log_dir,
            use_gpu_wrapper=not args.no_gpu_wrapper
        )
        processes.append({
            'process': process,
            'config': exp['config'],
            'gpu': exp['gpu']
        })
        time.sleep(2)  # Stagger launches slightly
    
    print(f"\nAll experiments launched!")
    print(f"Monitor progress: tail -f {log_dir}/experiment_*.log")
    print(f"Check GPU usage: watch -n 1 nvidia-smi\n")
    
    # Wait for completion if requested
    if args.wait:
        print("Waiting for all experiments to complete...\n")
        failed = []
        
        for p in processes:
            returncode = p['process'].wait()
            idx = p['config']['run_index']
            gpu = p['gpu']
            
            if returncode == 0:
                print(f"✓ Experiment index={idx} (GPU {gpu}) completed successfully")
            else:
                print(f"✗ Experiment index={idx} (GPU {gpu}) failed with code {returncode}")
                failed.append(idx)
        
        if failed:
            print(f"\n{len(failed)} experiment(s) failed: {failed}")
            return 1
        else:
            print(f"\nAll {len(experiments)} experiments completed successfully!")
            return 0
    else:
        print("Experiments running in background. PIDs:")
        for p in processes:
            print(f"  Index {p['config']['run_index']} (GPU {p['gpu']}): PID {p['process'].pid}")
        return 0


if __name__ == '__main__':
    sys.exit(main())

