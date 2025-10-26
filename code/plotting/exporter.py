"""
Results exporter for CSV and summary table generation.
"""

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd
from datetime import datetime


class ResultsExporter:
    """Export aggregated results to various formats."""
    
    def __init__(self):
        """Initialize exporter."""
        pass
    
    def export_to_csv(self, aggregated_data: Dict[str, Dict[str, Dict[int, Dict[str, float]]]], 
                     output_path: Path, compared_variable: str, compared_values: List[str]) -> Path:
        """Export aggregated metrics to CSV."""
        rows = []
        
        for test_set, test_data in aggregated_data.items():
            for metric, metric_data in test_data.items():
                for compared_value, cycle_data in metric_data.items():
                    for cycle, stats in cycle_data.items():
                        row = {
                            'cycle': cycle,
                            'test_set': test_set,
                            'metric': metric,
                            'compared_variable': compared_variable,
                            'compared_value': compared_value,
                            'mean': stats['mean'],
                            'std': stats['std'],
                            'sem': stats['sem'],
                            'min': stats['min'],
                            'max': stats['max'],
                            'ci_lower': stats['ci_lower'],
                            'ci_upper': stats['ci_upper'],
                            'n_replicates': stats['n_replicates']
                        }
                        rows.append(row)
        
        # Write CSV
        with open(output_path, 'w', newline='') as f:
            if rows:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
        
        return output_path
    
    def export_summary_table(self, aggregated_data: Dict[str, Dict[str, Dict[int, Dict[str, float]]]], 
                           output_path: Path, compared_variable: str, compared_values: List[str]) -> Path:
        """Export summary table as Markdown."""
        lines = []
        
        # Header
        lines.append(f"# Analysis Summary")
        lines.append(f"")
        lines.append(f"**Compared Variable:** {compared_variable}")
        lines.append(f"**Compared Values:** {', '.join(map(str, compared_values))}")
        lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"")
        
        # Summary for each test set and metric
        for test_set, test_data in aggregated_data.items():
            lines.append(f"## {self._format_test_set_name(test_set)}")
            lines.append(f"")
            
            for metric, metric_data in test_data.items():
                lines.append(f"### {self._format_metric_name(metric)}")
                lines.append(f"")
                
                # Create table
                lines.append("| Configuration | Final Cycle | Mean | Std | SEM | 95% CI |")
                lines.append("|---------------|-------------|------|-----|-----|--------|")
                
                for compared_value in compared_values:
                    if compared_value in metric_data:
                        # Get final cycle data
                        cycles = sorted(metric_data[compared_value].keys())
                        if cycles:
                            final_cycle = max(cycles)
                            stats = metric_data[compared_value][final_cycle]
                            
                            ci_str = f"[{stats['ci_lower']:.3f}, {stats['ci_upper']:.3f}]"
                            
                            lines.append(f"| {compared_variable}={compared_value} | {final_cycle} | "
                                       f"{stats['mean']:.3f} | {stats['std']:.3f} | "
                                       f"{stats['sem']:.3f} | {ci_str} |")
                        else:
                            lines.append(f"| {compared_variable}={compared_value} | - | - | - | - | - |")
                    else:
                        lines.append(f"| {compared_variable}={compared_value} | - | - | - | - | - |")
                
                lines.append(f"")
        
        # Write Markdown file
        with open(output_path, 'w') as f:
            f.write('\n'.join(lines))
        
        return output_path
    
    def export_raw_data(self, experiments: List[Any], output_path: Path) -> Path:
        """Export all raw experiment data."""
        rows = []
        
        for exp in experiments:
            exp_dict = exp.to_dict()
            
            # Add cycle-level data
            for cycle in exp.get_available_cycles():
                cycle_data = exp.cycle_results[cycle]
                evaluation = cycle_data.get('evaluation_results', {})
                
                for test_set, test_data in evaluation.items():
                    for metric, value in test_data.items():
                        row = {
                            'experiment_path': str(exp_dict['path']),
                            'dataset': str(exp_dict['dataset']),
                            'oracle_composition': str(exp_dict['oracle_composition']),
                            'student_composition': str(exp_dict['student_composition']),
                            'proposal_strategy': str(exp_dict['proposal_strategy']),
                            'acquisition_strategy': str(exp_dict['acquisition_strategy']),
                            'n_candidates_per_cycle': str(exp_dict['n_candidates_per_cycle']),
                            'n_acquire_per_cycle': str(exp_dict['n_acquire_per_cycle']),
                            'round0_init': str(exp_dict['round0_init']),
                            'training_mode': str(exp_dict['training_mode']),
                            'validation_dataset': str(exp_dict['validation_dataset']),
                            'run_index': str(exp_dict['run_index']),
                            'cycle': str(cycle),
                            'test_set': str(test_set),
                            'metric': str(metric),
                            'value': str(value) if value is not None else ''
                        }
                        rows.append(row)
        
        # Write CSV
        with open(output_path, 'w', newline='') as f:
            if rows:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
        
        return output_path
    
    def export_analysis_config(self, config: Dict[str, Any], output_path: Path) -> Path:
        """Export analysis configuration for reproducibility."""
        config['timestamp'] = datetime.now().isoformat()
        
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        return output_path
    
    def _format_metric_name(self, metric: str) -> str:
        """Format metric name for display."""
        replacements = {
            'avg_correlation': 'Average Correlation',
            'dev_correlation': 'Dev Correlation',
            'hk_correlation': 'Hk Correlation',
            'total_mse': 'Total MSE',
            'dev_mse': 'Dev MSE',
            'hk_mse': 'Hk MSE'
        }
        return replacements.get(metric, metric.replace('_', ' ').title())
    
    def _format_test_set_name(self, test_set: str) -> str:
        """Format test set name for display."""
        replacements = {
            'no_shift': 'No Shift (Genomic)',
            'low_shift': 'Low Shift (5% Mutation)',
            'high_shift_low_activity': 'High Shift (Random)'
        }
        return replacements.get(test_set, test_set.replace('_', ' ').title())
