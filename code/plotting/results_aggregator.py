"""
Metrics aggregator for computing statistics across replicates.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from scipy import stats
from .results_parser import ExperimentResult, ResultsCollection


class MetricsAggregator:
    """Aggregates metrics across replicates with proper statistics."""
    
    def __init__(self, confidence_level: float = 0.95):
        """Initialize with confidence level for CI calculation."""
        self.confidence_level = confidence_level
    
    def aggregate_replicates(self, experiments: List[ExperimentResult]) -> Dict[str, Any]:
        """Aggregate metrics across replicates."""
        if not experiments:
            return {}
        
        # Get all available cycles, test sets, and metrics
        all_cycles = set()
        all_test_sets = set()
        all_metrics = set()
        
        for exp in experiments:
            all_cycles.update(exp.get_available_cycles())
            all_test_sets.update(exp.get_available_test_sets())
            all_metrics.update(exp.get_available_metrics())
        
        all_cycles = sorted(list(all_cycles))
        all_test_sets = sorted(list(all_test_sets))
        all_metrics = sorted(list(all_metrics))
        
        # Aggregate for each combination
        aggregated = {}
        
        for test_set in all_test_sets:
            aggregated[test_set] = {}
            for metric in all_metrics:
                aggregated[test_set][metric] = {}
                
                for cycle in all_cycles:
                    values = []
                    for exp in experiments:
                        value = exp.get_metrics(cycle, test_set, metric)
                        if value is not None:
                            values.append(value)
                    
                    if values:
                        stats_dict = self.compute_statistics(values)
                        stats_dict['n_replicates'] = len(values)
                        aggregated[test_set][metric][cycle] = stats_dict
                    else:
                        aggregated[test_set][metric][cycle] = {
                            'mean': np.nan,
                            'std': np.nan,
                            'sem': np.nan,
                            'min': np.nan,
                            'max': np.nan,
                            'ci_lower': np.nan,
                            'ci_upper': np.nan,
                            'n_replicates': 0
                        }
        
        return aggregated
    
    def compute_statistics(self, values: List[float]) -> Dict[str, float]:
        """Compute comprehensive statistics for a list of values."""
        values = np.array(values)
        n = len(values)
        
        if n == 0:
            return {
                'mean': np.nan,
                'std': np.nan,
                'sem': np.nan,
                'min': np.nan,
                'max': np.nan,
                'ci_lower': np.nan,
                'ci_upper': np.nan
            }
        
        mean = np.mean(values)
        std = np.std(values, ddof=1)  # Sample standard deviation
        sem = std / np.sqrt(n)  # Standard error of the mean
        
        min_val = np.min(values)
        max_val = np.max(values)
        
        # 95% confidence interval using t-distribution
        if n > 1:
            alpha = 1 - self.confidence_level
            t_critical = stats.t.ppf(1 - alpha/2, df=n-1)
            margin_error = t_critical * sem
            ci_lower = mean - margin_error
            ci_upper = mean + margin_error
        else:
            ci_lower = mean
            ci_upper = mean
        
        return {
            'mean': float(mean),
            'std': float(std),
            'sem': float(sem),
            'min': float(min_val),
            'max': float(max_val),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper)
        }
    
    def get_metric_over_cycles(self, experiments: List[ExperimentResult], 
                             test_set: str, metric: str) -> Dict[int, Dict[str, float]]:
        """Extract time series for a specific metric."""
        aggregated = self.aggregate_replicates(experiments)
        
        if test_set not in aggregated or metric not in aggregated[test_set]:
            return {}
        
        return aggregated[test_set][metric]
    
    def get_cycle_range(self, experiments: List[ExperimentResult]) -> Tuple[int, int]:
        """Get the range of cycles available across all experiments."""
        if not experiments:
            return 0, 0
        
        min_cycle = min(min(exp.get_available_cycles()) for exp in experiments)
        max_cycle = max(max(exp.get_available_cycles()) for exp in experiments)
        
        return min_cycle, max_cycle
    
    def get_available_combinations(self, experiments: List[ExperimentResult]) -> List[Tuple[str, str]]:
        """Get all available test_set, metric combinations."""
        if not experiments:
            return []
        
        combinations = set()
        for exp in experiments:
            for test_set in exp.get_available_test_sets():
                for metric in exp.get_available_metrics():
                    combinations.add((test_set, metric))
        
        return sorted(list(combinations))
