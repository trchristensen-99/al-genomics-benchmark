#!/usr/bin/env python3
"""
Create a test split for yeast dataset by holding out 10% of training data.
"""

import numpy as np
from pathlib import Path

def create_test_split(data_dir: str = 'data/yeast', test_fraction: float = 0.1, seed: int = 42):
    """
    Split the yeast training data into train and test sets.
    
    Args:
        data_dir: Path to yeast data directory
        test_fraction: Fraction of data to hold out for test
        seed: Random seed for reproducibility
    """
    data_path = Path(data_dir)
    train_file = data_path / 'train.txt'
    
    if not train_file.exists():
        raise FileNotFoundError(f"Training file not found: {train_file}")
    
    print(f"Loading training data from {train_file}")
    
    # Read all lines
    with open(train_file, 'r') as f:
        lines = f.readlines()
    
    print(f"Total sequences: {len(lines):,}")
    
    # Set random seed
    np.random.seed(seed)
    
    # Calculate split sizes
    n_total = len(lines)
    n_test = int(n_total * test_fraction)
    n_train = n_total - n_test
    
    print(f"Test split: {test_fraction*100:.0f}% = {n_test:,} sequences")
    print(f"New train: {n_train:,} sequences")
    
    # Randomly shuffle indices
    indices = np.arange(n_total)
    np.random.shuffle(indices)
    
    # Split indices
    test_indices = sorted(indices[:n_test])
    train_indices = sorted(indices[n_test:])
    
    # Create new files
    train_new = data_path / 'train_new.txt'
    test_new = data_path / 'test.txt'
    
    print(f"\nWriting new train file: {train_new}")
    with open(train_new, 'w') as f:
        for idx in train_indices:
            f.write(lines[idx])
    
    print(f"Writing test file: {test_new}")
    with open(test_new, 'w') as f:
        for idx in test_indices:
            f.write(lines[idx])
    
    # Backup original and replace
    train_backup = data_path / 'train_original.txt'
    print(f"\nBacking up original train to: {train_backup}")
    train_file.rename(train_backup)
    train_new.rename(train_file)
    
    print("\n✅ Done!")
    print(f"   Train: {n_train:,} sequences")
    print(f"   Val: {len(open(data_path / 'val.txt').readlines()):,} sequences")
    print(f"   Test: {n_test:,} sequences")
    print(f"   Original train backed up to: {train_backup}")

if __name__ == '__main__':
    create_test_split()
