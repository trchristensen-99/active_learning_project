# Analysis Results: 10k vs 20k Acquisition Size Comparison

## Overview

This analysis compares the performance of active learning experiments with different acquisition sizes (10,000 vs 20,000 samples per cycle) while holding all other variables constant.

## Configuration

- **Compared Variable**: `n_acquire_per_cycle`
- **Compared Values**: 10,000, 20,000
- **Held Constant**:
  - Dataset: `deepstarr`
  - Oracle Composition: `5dreamrnn`
  - Student Composition: `1deepstarr`
  - Proposal Strategy: `100random_proposal`
  - Acquisition Strategy: `100random_acquisition`
  - Candidates per Cycle: 100,000
  - Round 0 Init: `init_prop_genomic_acq_random_20k`
  - Training Mode: `train_scratch`
  - Validation Dataset: `val_genomic`

## Results Summary

### Experiments Analyzed
- **Total Experiments**: 6
- **10k Acquisition**: 3 experiments (replicates)
- **20k Acquisition**: 3 experiments (replicates)

### Key Findings

#### No Shift (Genomic) Test Set
- **Average Correlation**: 10k acquisition performs slightly better (0.397 vs 0.372)
- **Total MSE**: 10k acquisition performs better (2.208 vs 2.230)

#### Low Shift (5% Mutation) Test Set
- **Average Correlation**: 20k acquisition performs better (0.354 vs 0.333)
- **Total MSE**: 20k acquisition performs better (2.245 vs 2.260)

#### High Shift (Random) Test Set
- **Average Correlation**: 20k acquisition performs better (0.301 vs 0.285)
- **Total MSE**: 20k acquisition performs better (2.280 vs 2.295)

## Files Generated

### Plots (`plots/` directory)
- `no_shift_avg_correlation.png` - Average correlation on genomic test set
- `no_shift_total_mse.png` - Total MSE on genomic test set
- `low_shift_avg_correlation.png` - Average correlation on low shift test set
- `low_shift_total_mse.png` - Total MSE on low shift test set
- `high_shift_low_activity_avg_correlation.png` - Average correlation on high shift test set
- `high_shift_low_activity_total_mse.png` - Total MSE on high shift test set
- `combined_avg_correlation.png` - Average correlation across all test sets
- `combined_total_mse.png` - Total MSE across all test sets

### Data (`data/` directory)
- `aggregated_metrics.csv` - Aggregated statistics (mean, std, sem, CI) by cycle
- `summary_table.md` - Markdown summary table with final cycle performance
- `raw_data.csv` - Complete raw data from all experiments

### Configuration
- `config.json` - Analysis configuration for reproducibility

## Interpretation

The results show an interesting pattern:
- **10k acquisition** performs better on the **genomic (no shift)** test set
- **20k acquisition** performs better on the **shifted** test sets (low shift and high shift)

This suggests that:
1. Smaller acquisition sizes may be sufficient for learning from the original data distribution
2. Larger acquisition sizes provide better generalization to shifted distributions
3. The choice of acquisition size should depend on the target distribution shift

## Statistical Notes

- All statistics computed across 3 replicates
- 95% confidence intervals calculated using t-distribution
- Error bars in plots represent 95% confidence intervals
- Standard error of the mean (SEM) reported for precision assessment
