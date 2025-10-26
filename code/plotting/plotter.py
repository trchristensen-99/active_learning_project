"""
Matplotlib plotter for generating publication-ready plots.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import warnings

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class ResultsPlotter:
    """Generate matplotlib plots for experiment results."""
    
    def __init__(self, figure_size: Tuple[float, float] = (8, 6), 
                 font_size: int = 12, dpi: int = 300):
        """Initialize plotter with styling options."""
        self.figure_size = figure_size
        self.font_size = font_size
        self.dpi = dpi
        
        # Set matplotlib parameters
        plt.rcParams.update({
            'font.size': font_size,
            'axes.titlesize': font_size + 2,
            'axes.labelsize': font_size,
            'xtick.labelsize': font_size - 1,
            'ytick.labelsize': font_size - 1,
            'legend.fontsize': font_size - 1,
            'figure.titlesize': font_size + 4
        })
    
    def plot_metric_over_cycles(self, aggregated_data: Dict[str, Dict[str, Dict[int, Dict[str, float]]]], 
                               output_dir: Path, compared_variable: str, compared_values: List[str],
                               test_set: str, metric: str, plot_format: str = 'png') -> Path:
        """Plot a single metric over cycles for one test set."""
        if test_set not in aggregated_data or metric not in aggregated_data[test_set]:
            raise ValueError(f"No data found for test_set={test_set}, metric={metric}")
        
        fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
        
        colors = sns.color_palette("husl", len(compared_values))
        
        for i, value in enumerate(compared_values):
            if value not in aggregated_data[test_set][metric]:
                warnings.warn(f"No data found for {compared_variable}={value}")
                continue
            
            cycle_data = aggregated_data[test_set][metric][value]
            
            # Extract cycles and statistics
            cycles = sorted(cycle_data.keys())
            means = [cycle_data[c]['mean'] for c in cycles]
            ci_lower = [cycle_data[c]['ci_lower'] for c in cycles]
            ci_upper = [cycle_data[c]['ci_upper'] for c in cycles]
            
            # Remove NaN values
            valid_indices = [i for i, m in enumerate(means) if not np.isnan(m)]
            if not valid_indices:
                continue
            
            cycles_clean = [cycles[i] for i in valid_indices]
            means_clean = [means[i] for i in valid_indices]
            ci_lower_clean = [ci_lower[i] for i in valid_indices]
            ci_upper_clean = [ci_upper[i] for i in valid_indices]
            
            # Plot line
            ax.plot(cycles_clean, means_clean, 
                   color=colors[i], linewidth=2, marker='o', markersize=6,
                   label=f"{compared_variable}={value}")
            
            # Plot confidence interval as shaded region
            ax.fill_between(cycles_clean, ci_lower_clean, ci_upper_clean,
                           color=colors[i], alpha=0.2)
        
        # Customize plot
        ax.set_xlabel('Cycle')
        ax.set_ylabel(self._format_metric_name(metric))
        ax.set_title(f'{self._format_test_set_name(test_set)} - {self._format_metric_name(metric)}')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Set x-axis to integer ticks
        ax.set_xticks(range(min(cycles) if cycles else 0, max(cycles) + 1 if cycles else 1))
        
        plt.tight_layout()
        
        # Save plot
        filename = f"{test_set}_{metric}.{plot_format}"
        output_path = output_dir / filename
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def plot_combined_test_sets(self, aggregated_data: Dict[str, Dict[str, Dict[int, Dict[str, float]]]], 
                               output_dir: Path, compared_variable: str, compared_values: List[str],
                               metric: str, plot_format: str = 'png') -> Path:
        """Plot a metric across all test sets in subplots."""
        test_sets = list(aggregated_data.keys())
        n_test_sets = len(test_sets)
        
        if n_test_sets == 0:
            raise ValueError("No test sets found in data")
        
        # Create subplots
        fig, axes = plt.subplots(1, n_test_sets, figsize=(self.figure_size[0] * n_test_sets, self.figure_size[1]), dpi=self.dpi)
        if n_test_sets == 1:
            axes = [axes]
        
        colors = sns.color_palette("husl", len(compared_values))
        
        for test_idx, test_set in enumerate(test_sets):
            ax = axes[test_idx]
            
            if metric not in aggregated_data[test_set]:
                ax.text(0.5, 0.5, f'No data for {metric}', 
                       transform=ax.transAxes, ha='center', va='center')
                continue
            
            for i, value in enumerate(compared_values):
                if value not in aggregated_data[test_set][metric]:
                    continue
                
                cycle_data = aggregated_data[test_set][metric][value]
                
                # Extract cycles and statistics
                cycles = sorted(cycle_data.keys())
                means = [cycle_data[c]['mean'] for c in cycles]
                ci_lower = [cycle_data[c]['ci_lower'] for c in cycles]
                ci_upper = [cycle_data[c]['ci_upper'] for c in cycles]
                
                # Remove NaN values
                valid_indices = [i for i, m in enumerate(means) if not np.isnan(m)]
                if not valid_indices:
                    continue
                
                cycles_clean = [cycles[i] for i in valid_indices]
                means_clean = [means[i] for i in valid_indices]
                ci_lower_clean = [ci_lower[i] for i in valid_indices]
                ci_upper_clean = [ci_upper[i] for i in valid_indices]
                
                # Plot line
                ax.plot(cycles_clean, means_clean, 
                       color=colors[i], linewidth=2, marker='o', markersize=6,
                       label=f"{compared_variable}={value}")
                
                # Plot confidence interval
                ax.fill_between(cycles_clean, ci_lower_clean, ci_upper_clean,
                               color=colors[i], alpha=0.2)
            
            # Customize subplot
            ax.set_xlabel('Cycle')
            ax.set_ylabel(self._format_metric_name(metric))
            ax.set_title(self._format_test_set_name(test_set))
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Set x-axis to integer ticks
            all_cycles = []
            for i, value in enumerate(compared_values):
                if value in aggregated_data[test_set][metric]:
                    cycle_data = aggregated_data[test_set][metric][value]
                    cycles = sorted(cycle_data.keys())
                    means = [cycle_data[c]['mean'] for c in cycles]
                    valid_indices = [i for i, m in enumerate(means) if not np.isnan(m)]
                    if valid_indices:
                        cycles_clean = [cycles[i] for i in valid_indices]
                        all_cycles.extend(cycles_clean)
            
            if all_cycles:
                ax.set_xticks(range(min(all_cycles), max(all_cycles) + 1))
        
        plt.suptitle(f'{self._format_metric_name(metric)} Across Test Sets', fontsize=self.font_size + 4)
        plt.tight_layout()
        
        # Save plot
        filename = f"combined_{metric}.{plot_format}"
        output_path = output_dir / filename
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def plot_all_metrics(self, aggregated_data: Dict[str, Dict[str, Dict[int, Dict[str, float]]]], 
                        output_dir: Path, compared_variable: str, compared_values: List[str],
                        metrics: List[str], plot_format: str = 'png') -> List[Path]:
        """Plot all specified metrics."""
        output_paths = []
        
        # Use output_dir directly (plots directory is created by the script)
        plots_dir = output_dir
        
        # Get all test sets
        test_sets = list(aggregated_data.keys())
        
        for metric in metrics:
            # Plot individual test sets
            for test_set in test_sets:
                try:
                    path = self.plot_metric_over_cycles(
                        aggregated_data, plots_dir, compared_variable, compared_values,
                        test_set, metric, plot_format
                    )
                    output_paths.append(path)
                except ValueError as e:
                    warnings.warn(f"Skipping {test_set}_{metric}: {e}")
            
            # Plot combined test sets
            try:
                path = self.plot_combined_test_sets(
                    aggregated_data, plots_dir, compared_variable, compared_values,
                    metric, plot_format
                )
                output_paths.append(path)
            except ValueError as e:
                warnings.warn(f"Skipping combined_{metric}: {e}")
        
        return output_paths
    
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
