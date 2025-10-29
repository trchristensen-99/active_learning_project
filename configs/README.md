# Configuration Files

This directory contains JSON configuration files for running active learning experiments.

## Naming Convention

Config files follow a pattern based on their key settings:

- **Oracle-labeled experiments**: `oracle_labeled_{acq}k_{init}k_init.json`
  - `acq`: Acquisition size per cycle (e.g., `5k`, `10k`, `20k`)
  - `init`: Initial pretraining size (e.g., `20k`, `40k`, `80k`)
  - Example: `oracle_labeled_20k_40k_init.json` = 20k acquisition, 40k initial pretraining

- **Ground truth experiments**: `active_learning_*.json`
  - Various naming patterns for different experiment types
  - Examples: `active_learning_genomic_init.json`, `active_learning_random_random.json`

## Usage

Run experiments using:

```bash
python scripts/run_active_learning.py --config configs/<config_file>.json --run-index <index>
```

For multiple replicates, use different `--run-index` values (0, 1, 2, ...).

## Key Settings

### Oracle-Labeled Configs

| Config File | Acquisition Size | Initial Pretraining | Use Oracle Labels |
|------------|------------------|---------------------|-------------------|
| `oracle_labeled_5k.json` | 5,000 | 20,000 | Yes |
| `oracle_labeled_10k.json` | 10,000 | 20,000 | Yes |
| `oracle_labeled_20k.json` | 20,000 | 20,000 | Yes |
| `oracle_labeled_20k_40k_init.json` | 20,000 | 40,000 | Yes |
| `oracle_labeled_20k_80k_init.json` | 20,000 | 80,000 | Yes |

### Common Settings

- **Student Model**: DeepSTARR architecture
  - Batch size: 128
  - Learning rate: 0.002
  - Epochs: 100
- **Active Learning**: 5 cycles, 100k candidates per cycle
- **Oracle**: 5 DREAM-RNN ensemble
- **Strategy**: Random proposal and acquisition

## Adding New Configs

1. Copy an existing config file as a template
2. Modify the relevant parameters:
   - `data.n_initial`: Initial pretraining size
   - `active_learning.n_acquire_per_cycle`: Acquisition size
   - `active_learning.n_candidates_per_cycle`: Candidate pool size
3. Follow the naming convention above
4. Update this README with the new config's details

## Canonical Configs

These configs are considered canonical and are actively used:

- `oracle_labeled_5k.json`, `oracle_labeled_10k.json`, `oracle_labeled_20k.json`: Standard oracle-labeled experiments
- `oracle_labeled_20k_40k_init.json`, `oracle_labeled_20k_80k_init.json`: Variants with larger pretraining sizes
- `active_learning_genomic_init.json`: Ground truth genomic initialization baseline

