#!/usr/bin/env python3
"""
Validation script to check experiment results for correctness.

Checks for:
1. No duplicate cycle entries
2. Correct cycle count (0-5 for n_cycles=5)
3. Monotonic training set sizes
4. Performance trends
5. Variance statistics
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any
import numpy as np


def validate_experiment(exp_dir: Path) -> Dict[str, Any]:
    """Validate a single experiment directory."""
    results_file = exp_dir / "all_cycle_results.json"
    
    if not results_file.exists():
        return {
            'valid': False,
            'error': f"Results file not found: {results_file}"
        }
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    validation = {
        'valid': True,
        'experiment': str(exp_dir),
        'issues': [],
        'warnings': []
    }
    
    # Check 1: No duplicate cycles
    cycles = [r['cycle'] for r in results]
    unique_cycles = set(cycles)
    if len(cycles) != len(unique_cycles):
        duplicates = [c for c in unique_cycles if cycles.count(c) > 1]
        validation['issues'].append(f"Duplicate cycles found: {duplicates}")
        validation['valid'] = False
    
    # Check 2: Correct cycle count (should be 0-5 for n_cycles=5)
    expected_cycles = list(range(6))  # 0-5
    if sorted(unique_cycles) != expected_cycles:
        validation['issues'].append(
            f"Incorrect cycle count. Expected {expected_cycles}, got {sorted(unique_cycles)}"
        )
        validation['valid'] = False
    
    # Check 3: Monotonic training set sizes
    training_sizes = []
    for cycle in sorted(unique_cycles):
        cycle_results = [r for r in results if r['cycle'] == cycle]
        if len(cycle_results) > 1:
            validation['warnings'].append(f"Multiple entries for cycle {cycle}, using first")
        training_sizes.append(cycle_results[0]['total_training_sequences'])
    
    if training_sizes != sorted(training_sizes):
        validation['issues'].append(
            f"Training sizes not monotonic: {training_sizes}"
        )
        validation['valid'] = False
    
    # Check 4: Performance trends
    correlations = []
    for cycle in sorted(unique_cycles):
        cycle_results = [r for r in results if r['cycle'] == cycle]
        if cycle_results and 'evaluation_results' in cycle_results[0]:
            eval_results = cycle_results[0]['evaluation_results']
            if 'no_shift' in eval_results:
                corr = eval_results['no_shift'].get('avg_correlation', None)
                if corr is not None:
                    correlations.append((cycle, corr))
    
    # Check for catastrophic drops (>0.2 decrease)
    for i in range(1, len(correlations)):
        prev_cycle, prev_corr = correlations[i-1]
        curr_cycle, curr_corr = correlations[i]
        if curr_corr < prev_corr - 0.2:
            validation['warnings'].append(
                f"Large performance drop: cycle {prev_cycle} ({prev_corr:.4f}) -> "
                f"cycle {curr_cycle} ({curr_corr:.4f})"
            )
    
    # Check 5: Training size increments
    expected_increment = None
    for i in range(1, len(training_sizes)):
        increment = training_sizes[i] - training_sizes[i-1]
        if expected_increment is None:
            expected_increment = increment
        elif abs(increment - expected_increment) > 100:  # Allow small variation
            validation['warnings'].append(
                f"Inconsistent training size increment: cycle {i-1}->{i} "
                f"added {increment} sequences (expected ~{expected_increment})"
            )
    
    # Summary statistics
    validation['summary'] = {
        'n_cycles': len(unique_cycles),
        'training_sizes': training_sizes,
        'correlations': [c for _, c in correlations],
        'final_correlation': correlations[-1][1] if correlations else None,
        'final_training_size': training_sizes[-1] if training_sizes else None
    }
    
    return validation


def main():
    """Main validation function."""
    if len(sys.argv) < 2:
        print("Usage: python validate_results.py <results_directory>")
        print("Example: python validate_results.py results/deepstarr")
        sys.exit(1)
    
    results_dir = Path(sys.argv[1])
    if not results_dir.exists():
        print(f"Error: Directory not found: {results_dir}")
        sys.exit(1)
    
    # Find all experiment directories (containing all_cycle_results.json)
    experiments = []
    for results_file in results_dir.rglob('all_cycle_results.json'):
        experiments.append(results_file.parent)
    
    if not experiments:
        print(f"No experiments found in {results_dir}")
        sys.exit(1)
    
    print(f"Found {len(experiments)} experiments to validate\n")
    print("="*80)
    
    all_valid = True
    for exp_dir in sorted(experiments):
        validation = validate_experiment(exp_dir)
        
        # Print experiment name
        rel_path = exp_dir.relative_to(results_dir)
        print(f"\n{rel_path}")
        print("-" * 80)
        
        if validation['valid']:
            print("✓ VALID")
        else:
            print("✗ INVALID")
            all_valid = False
        
        # Print issues
        if validation.get('issues'):
            print("\nIssues:")
            for issue in validation['issues']:
                print(f"  ✗ {issue}")
        
        # Print warnings
        if validation.get('warnings'):
            print("\nWarnings:")
            for warning in validation['warnings']:
                print(f"  ⚠ {warning}")
        
        # Print summary
        if 'summary' in validation:
            summary = validation['summary']
            print(f"\nSummary:")
            print(f"  Cycles: {summary['n_cycles']}")
            print(f"  Training sizes: {summary['training_sizes']}")
            if summary['correlations']:
                print(f"  Correlations: {[f'{c:.4f}' for c in summary['correlations']]}")
                print(f"  Final correlation: {summary['final_correlation']:.4f}")
    
    print("\n" + "="*80)
    if all_valid:
        print("✓ All experiments VALID")
        sys.exit(0)
    else:
        print("✗ Some experiments INVALID")
        sys.exit(1)


if __name__ == "__main__":
    main()


