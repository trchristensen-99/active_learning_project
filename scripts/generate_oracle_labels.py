#!/usr/bin/env python3
"""
Generate oracle-labeled copies of genomic train/val/test with structured paths:
  data/oracle_labels/{dataset}/{oracle_sig}/no_shift/{train,val,test}.txt

Uses the active-learning config to compute dataset/oracle signatures.
"""

import argparse
from pathlib import Path
import sys
import json
import pandas as pd
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from code.active_learning import ConfigurationManager, EnsembleOracle


def predict_in_batches(oracle: EnsembleOracle, seqs, batch_size: int):
    labels = []
    for start in range(0, len(seqs), batch_size):
        batch = seqs[start:start + batch_size]
        preds = oracle.predict(batch)
        labels.append(preds)
    return np.vstack(labels)


def main():
    p = argparse.ArgumentParser(description="Generate oracle-labeled genomic datasets")
    p.add_argument('--config', required=True, help='Path to active learning config JSON')
    p.add_argument('--train-path', default='data/processed/train.txt')
    p.add_argument('--val-path', default='data/processed/val.txt')
    p.add_argument('--test-path', default='data/processed/test.txt')
    p.add_argument('--output-root', default='data/oracle_labels')
    p.add_argument('--device', default='auto')
    p.add_argument('--batch-size', type=int, default=512)
    args = p.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)
    cfgm = ConfigurationManager(config)

    # Create oracle (from new composition or legacy)
    oracle_cfg = config.get('oracle', {})
    if 'composition' in oracle_cfg:
        oracle = EnsembleOracle(
            composition=oracle_cfg['composition'],
            device=args.device,
            seqsize=oracle_cfg.get('seqsize', 249),
            batch_size=args.batch_size
        )
    else:
        oracle = EnsembleOracle(
            model_dir=oracle_cfg.get('model_dir'),
            model_type=oracle_cfg.get('model_type', 'dream_rnn'),
            device=args.device,
            seqsize=oracle_cfg.get('seqsize', 249),
            batch_size=args.batch_size
        )

    dataset = cfgm.dataset
    oracle_sig = cfgm.oracle_composition

    out_base = Path(args.output_root) / dataset / oracle_sig / 'no_shift'
    out_base.mkdir(parents=True, exist_ok=True)

    # Process splits
    splits = [
        ('train', Path(args.train_path)),
        ('val', Path(args.val_path)),
        ('test', Path(args.test_path)),
    ]

    for split_name, split_path in splits:
        if not split_path.exists():
            print(f"Skipping {split_name} (missing): {split_path}")
            continue
        print(f"Loading {split_name}: {split_path}")
        df = pd.read_csv(split_path, sep='\t')
        seqs = df['Sequence'].tolist()
        print(f"Predicting {len(seqs)} sequences for {split_name}...")
        preds = predict_in_batches(oracle, seqs, args.batch_size)
        out_df = pd.DataFrame({
            'Sequence': seqs,
            'Dev_log2_enrichment': preds[:, 0],
            'Hk_log2_enrichment': preds[:, 1]
        })
        out_path = out_base / f"{split_name}.txt"
        out_df.to_csv(out_path, sep='\t', index=False)
        print(f"Wrote {split_name}: {out_path}")


if __name__ == '__main__':
    main()



