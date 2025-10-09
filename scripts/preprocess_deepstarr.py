#!/usr/bin/env python3
"""
Preprocess DeepSTARR dataset for DREAM-RNN training.

Merges FASTA sequence files with activity measurements into TSV format
expected by the Prix Fixe framework.
"""

import argparse
import os
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import sys


def parse_fasta(fasta_path: str) -> Dict[str, str]:
    """Parse FASTA file and return dictionary of ID -> sequence."""
    sequences = {}
    current_id = None
    current_seq = []
    
    with open(fasta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                # Save previous sequence
                if current_id is not None:
                    sequences[current_id] = ''.join(current_seq)
                
                # Start new sequence
                current_id = line[1:]  # Remove '>'
                current_seq = []
            else:
                current_seq.append(line)
        
        # Save last sequence
        if current_id is not None:
            sequences[current_id] = ''.join(current_seq)
    
    return sequences


def parse_activity_file(activity_path: str) -> pd.DataFrame:
    """Parse activity file and return DataFrame."""
    return pd.read_csv(activity_path, sep='\t')


def reverse_complement(seq: str) -> str:
    """Get reverse complement of DNA sequence."""
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}
    return ''.join(complement[base] for base in reversed(seq))


def process_split(raw_dir: str, out_dir: str, split: str) -> None:
    """Process a single split (train/val/test)."""
    print(f"Processing {split} split...")
    
    # File paths
    fasta_path = os.path.join(raw_dir, f"Sequences_{split.capitalize()}.fa")
    activity_path = os.path.join(raw_dir, f"Sequences_activity_{split.capitalize()}.txt")
    
    if not os.path.exists(fasta_path):
        print(f"Warning: {fasta_path} not found, skipping {split}")
        return
    
    if not os.path.exists(activity_path):
        print(f"Warning: {activity_path} not found, skipping {split}")
        return
    
    # Parse sequences and activities
    sequences = parse_fasta(fasta_path)
    activities = parse_activity_file(activity_path)
    
    print(f"  Loaded {len(sequences)} sequences and {len(activities)} activity measurements")
    
    # Create DataFrame with sequences
    data = []
    for i, (seq_id, sequence) in enumerate(sequences.items()):
        if i < len(activities):
            row = {
                'ID': seq_id,
                'Sequence': sequence,
                'Dev_log2_enrichment': activities.iloc[i]['Dev_log2_enrichment'],
                'Hk_log2_enrichment': activities.iloc[i]['Hk_log2_enrichment'],
                'rev': 0  # Forward sequence
            }
            data.append(row)
            
            # Add reverse complement
            rev_seq = reverse_complement(sequence)
            rev_row = {
                'ID': f"{seq_id}_rev",
                'Sequence': rev_seq,
                'Dev_log2_enrichment': activities.iloc[i]['Dev_log2_enrichment'],
                'Hk_log2_enrichment': activities.iloc[i]['Hk_log2_enrichment'],
                'rev': 1  # Reverse complement
            }
            data.append(rev_row)
    
    # Create DataFrame and save
    df = pd.DataFrame(data)
    output_path = os.path.join(out_dir, f"{split}.txt")
    df.to_csv(output_path, sep='\t', index=False)
    
    print(f"  Saved {len(df)} sequences (including reverse complements) to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess DeepSTARR dataset for DREAM-RNN training")
    parser.add_argument("--raw_dir", default="data/raw", 
                       help="Directory containing raw DeepSTARR files")
    parser.add_argument("--out_dir", default="data/processed",
                       help="Output directory for processed files")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                       help="Data splits to process")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Process each split
    for split in args.splits:
        try:
            process_split(args.raw_dir, args.out_dir, split)
        except Exception as e:
            print(f"Error processing {split}: {e}")
            sys.exit(1)
    
    print("Preprocessing complete!")


if __name__ == "__main__":
    main()
