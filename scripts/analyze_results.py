#!/usr/bin/env python3
"""
Results analysis script for comparing experiment configurations.

Analyzes hierarchical experiment results and generates plots/CSVs comparing
any variable while holding others constant.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Any
import warnings

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from code.plotting import ResultsCollection, MetricsAggregator, ResultsPlotter, ResultsExporter


def parse_hold_constant(hold_constant_str: str) -> Dict[str, Any]:
    """Parse hold-constant string into dictionary."""
    if not hold_constant_str:
        return {}
    
    filters = {}
    for pair in hold_constant_str.split(','):
        if '=' not in pair:
            raise ValueError(f"Invalid hold-constant format: {pair}. Expected 'key=value'")
        
        key, value = pair.split('=', 1)
        key = key.strip()
        value = value.strip()
        
        # Try to convert to appropriate type
        if value.isdigit():
            value = int(value)
        elif value.replace('.', '').isdigit():
            value = float(value)
        
        filters[key] = value
    
    return filters


def parse_compare_values(compare_values_str: str) -> List[Any]:
    """Parse compare-values string into list."""
    if not compare_values_str:
        return []
    
    values = []
    for value in compare_values_str.split(','):
        value = value.strip()
        
        # Try to convert to appropriate type
        if value.isdigit():
            value = int(value)
        elif value.replace('.', '').isdigit():
            value = float(value)
        
        values.append(value)
    
    return values


def main():
    """Main analysis function."""
    parser = argparse.ArgumentParser(description='Analyze experiment results')
    
    # Required arguments
    parser.add_argument('--results-dir', type=str, default='results/',
                       help='Root results directory')
    parser.add_argument('--compare-variable', type=str, required=True,
                       help='Variable to compare (e.g., n_acquire_per_cycle, training_mode)')
    parser.add_argument('--compare-values', type=str, required=True,
                       help='Values to compare (comma-separated, e.g., 10000,20000)')
    
    # Optional arguments
    parser.add_argument('--hold-constant', type=str, default='',
                       help='Variables to hold constant (format: var1=val1,var2=val2)')
    parser.add_argument('--metrics', type=str, default='avg_correlation,total_mse',
                       help='Metrics to plot (comma-separated)')
    parser.add_argument('--test-sets', type=str, default='',
                       help='Test sets to include (comma-separated, default: all)')
    parser.add_argument('--output-dir', type=str, default='analysis_output/',
                       help='Output directory for plots/CSVs')
    parser.add_argument('--plot-format', type=str, default='png',
                       choices=['png', 'pdf', 'svg'],
                       help='Plot format')
    parser.add_argument('--figure-size', type=str, default='8,6',
                       help='Figure size as width,height (default: 8,6)')
    parser.add_argument('--font-size', type=int, default=12,
                       help='Font size (default: 12)')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip plot generation')
    parser.add_argument('--no-csv', action='store_true',
                       help='Skip CSV export')
    
    args = parser.parse_args()
    
    # Parse arguments
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        sys.exit(1)
    
    hold_constant = parse_hold_constant(args.hold_constant)
    compare_values = parse_compare_values(args.compare_values)
    metrics = [m.strip() for m in args.metrics.split(',')]
    
    test_sets = None
    if args.test_sets:
        test_sets = [t.strip() for t in args.test_sets.split(',')]
    
    output_dir = Path(args.output_dir)
    
    # Parse figure size
    try:
        figure_size = tuple(map(float, args.figure_size.split(',')))
        if len(figure_size) != 2:
            raise ValueError("Figure size must be width,height")
    except ValueError as e:
        print(f"Error parsing figure size: {e}")
        sys.exit(1)
    
    print(f"Loading experiments from: {results_dir}")
    print(f"Comparing variable: {args.compare_variable}")
    print(f"Comparing values: {compare_values}")
    print(f"Holding constant: {hold_constant}")
    print(f"Metrics: {metrics}")
    print(f"Output directory: {output_dir}")
    
    # Load experiments
    try:
        collection = ResultsCollection.from_directory(results_dir)
        print(f"Found {len(collection)} total experiments")
    except Exception as e:
        print(f"Error loading experiments: {e}")
        sys.exit(1)
    
    # Filter experiments
    filtered_collection = collection.find_experiments(**hold_constant)
    print(f"Found {len(filtered_collection)} experiments matching filters")
    
    if len(filtered_collection) == 0:
        print("No experiments found matching the criteria")
        sys.exit(1)
    
    # Group by compared variable
    try:
        groups = filtered_collection.group_by(args.compare_variable)
        print(f"Found {len(groups)} groups for {args.compare_variable}")
    except ValueError as e:
        print(f"Error grouping by {args.compare_variable}: {e}")
        sys.exit(1)
    
    # Check if all compared values are present
    missing_values = [v for v in compare_values if v not in groups]
    if missing_values:
        print(f"Warning: Missing values for {args.compare_variable}: {missing_values}")
        print(f"Available values: {list(groups.keys())}")
    
    # Filter to only requested values
    groups = {k: v for k, v in groups.items() if k in compare_values}
    
    if len(groups) < 2:
        print(f"Error: Need at least 2 groups to compare, found {len(groups)}")
        print(f"Available groups: {list(groups.keys())}")
        sys.exit(1)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    data_dir = output_dir / "data"
    plots_dir.mkdir(exist_ok=True)
    data_dir.mkdir(exist_ok=True)
    
    # Initialize components
    aggregator = MetricsAggregator()
    plotter = ResultsPlotter(figure_size=figure_size, font_size=args.font_size)
    exporter = ResultsExporter()
    
    # Aggregate data for each group
    aggregated_data = {}
    for test_set in ['no_shift', 'low_shift', 'high_shift_low_activity']:  # Default test sets
        aggregated_data[test_set] = {}
        for metric in metrics:
            aggregated_data[test_set][metric] = {}
            
            for value, group in groups.items():
                if len(group) == 0:
                    print(f"Warning: No experiments found for {args.compare_variable}={value}")
                    continue
                
                print(f"Aggregating {len(group)} experiments for {args.compare_variable}={value}")
                group_aggregated = aggregator.aggregate_replicates(list(group))
                
                if test_set in group_aggregated and metric in group_aggregated[test_set]:
                    aggregated_data[test_set][metric][value] = group_aggregated[test_set][metric]
    
    # Filter test sets if specified
    if test_sets:
        aggregated_data = {k: v for k, v in aggregated_data.items() if k in test_sets}
    
    # Generate plots
    if not args.no_plots:
        print("Generating plots...")
        try:
            plot_paths = plotter.plot_all_metrics(
                aggregated_data, plots_dir, args.compare_variable, compare_values,
                metrics, args.plot_format
            )
            print(f"Generated {len(plot_paths)} plots")
        except Exception as e:
            print(f"Error generating plots: {e}")
            if not args.no_csv:  # Continue with CSV export
                pass
    
    # Export CSV data
    if not args.no_csv:
        print("Exporting CSV data...")
        try:
            # Export aggregated data
            csv_path = exporter.export_to_csv(
                aggregated_data, data_dir / "aggregated_metrics.csv",
                args.compare_variable, compare_values
            )
            print(f"Exported aggregated data to: {csv_path}")
            
            # Export summary table
            summary_path = exporter.export_summary_table(
                aggregated_data, data_dir / "summary_table.md",
                args.compare_variable, compare_values
            )
            print(f"Exported summary table to: {summary_path}")
            
            # Export raw data
            raw_path = exporter.export_raw_data(
                list(filtered_collection), data_dir / "raw_data.csv"
            )
            print(f"Exported raw data to: {raw_path}")
            
        except Exception as e:
            print(f"Error exporting CSV data: {e}")
    
    # Export analysis configuration
    config = {
        'results_dir': str(results_dir),
        'compare_variable': args.compare_variable,
        'compare_values': compare_values,
        'hold_constant': hold_constant,
        'metrics': metrics,
        'test_sets': test_sets,
        'figure_size': figure_size,
        'font_size': args.font_size,
        'plot_format': args.plot_format,
        'n_experiments_total': len(collection),
        'n_experiments_filtered': len(filtered_collection),
        'n_groups': len(groups)
    }
    
    config_path = exporter.export_analysis_config(config, output_dir / "config.json")
    print(f"Exported analysis config to: {config_path}")
    
    # Print summary
    print("\n" + "="*50)
    print("ANALYSIS SUMMARY")
    print("="*50)
    print(f"Compared variable: {args.compare_variable}")
    print(f"Compared values: {compare_values}")
    print(f"Total experiments: {len(collection)}")
    print(f"Filtered experiments: {len(filtered_collection)}")
    print(f"Groups found: {len(groups)}")
    
    for value, group in groups.items():
        print(f"  {args.compare_variable}={value}: {len(group)} experiments")
    
    print(f"\nOutput directory: {output_dir}")
    print("Analysis complete!")


if __name__ == "__main__":
    main()
