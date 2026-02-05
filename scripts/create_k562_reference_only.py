#!/usr/bin/env python3
"""
Create K562 reference-allele-only dataset matching the benchmark paper.

Paper specifies: 367,364 200-bp reference allele genomic sequences
Current dataset has: ~392K reference + ~392K alternate = 784K variant sequences
"""

import pandas as pd
from pathlib import Path

def create_reference_only_dataset():
    """Filter K562 dataset to reference alleles only."""
    
    input_file = "data/k562/DATA-Table_S2__MPRA_dataset.txt"
    output_file = "data/k562/K562_reference_only.txt"
    
    print("Loading full K562 dataset...")
    df = pd.read_csv(input_file, sep='\t', dtype={'OL': str})
    
    print(f"Total sequences: {len(df):,}")
    
    # Parse ID to extract allele type
    id_parts = df['IDs'].str.split(':', expand=True)
    df['allele_type'] = id_parts[4]  # R=reference, A=alternate, empty=no variant
    df['ref_allele'] = id_parts[2]
    df['alt_allele'] = id_parts[3]
    
    # Count by type
    print("\nSequence breakdown:")
    print(f"  Reference (R): {len(df[df['allele_type']=='R']):,}")
    print(f"  Alternate (A): {len(df[df['allele_type']=='A']):,}")
    print(f"  No variant (CRE, etc): {len(df[df['allele_type']==''].astype(bool)):,}")
    
    # Filter to reference alleles only (R) and non-variant sequences
    # Non-variant sequences have NA in both ref and alt columns
    is_reference = df['allele_type'] == 'R'
    is_non_variant = (df['ref_allele'] == 'NA') & (df['alt_allele'] == 'NA')
    
    reference_only = df[is_reference | is_non_variant].copy()
    
    print(f"\nReference-only dataset: {len(reference_only):,} sequences")
    
    # Additional filtering to match paper's 367K
    # The paper may have filtered out some sequences (e.g., chromosomes, quality)
    # Let's check if excluding certain projects gets us closer
    
    print("\nBreakdown by project:")
    for project in reference_only['data_project'].value_counts().index:
        count = len(reference_only[reference_only['data_project']==project])
        print(f"  {project}: {count:,}")
    
    # Apply chromosome filter (exclude validation/test chromosomes during counting)
    train_chr = ~reference_only['chr'].isin(['7', '13', '19', '21', 'X'])
    training_pool = reference_only[train_chr].copy()
    
    print(f"\nTraining pool (excluding test/val chromosomes): {len(training_pool):,}")
    
    # Save reference-only dataset
    reference_only.to_csv(output_file, sep='\t', index=False)
    print(f"\nSaved reference-only dataset to: {output_file}")
    
    # Save just the training pool
    training_pool.to_csv(output_file.replace('.txt', '_train.txt'), sep='\t', index=False)
    print(f"Saved training pool to: {output_file.replace('.txt', '_train.txt')}")
    
    return len(reference_only), len(training_pool)

if __name__ == '__main__':
    total, train = create_reference_only_dataset()
    print(f"\n✅ Created reference-only dataset:")
    print(f"   Total: {total:,} sequences")
    print(f"   Training pool: {train:,} sequences")
    print(f"   Expected from paper: 367,364 sequences")
    print(f"   Difference: {train - 367364:,} ({(train/367364-1)*100:.1f}%)")
