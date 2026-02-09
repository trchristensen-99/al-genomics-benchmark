#!/usr/bin/env python3
"""
Create SNV test set from HashFrag test set only.

Following the pilot study methodology:
- SNV test set = alternate alleles of sequences in the HashFrag test set
- Only include pairs where both ref and alt alleles are available
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.k562 import K562Dataset


def create_hashfrag_snv_pairs(
    data_file: str = "data/k562/DATA-Table_S2__MPRA_dataset.txt",
    output_file: str = "data/k562/test_sets/test_hashfrag_snv_pairs.tsv",
    hashfrag_cache_dir: str = "data/k562/hashfrag_splits"
):
    """
    Create SNV pairs from HashFrag test set only.
    
    Args:
        data_file: Path to full K562 MPRA dataset
        output_file: Path to save SNV pairs
        hashfrag_cache_dir: Directory with HashFrag splits
    """
    print("=" * 80)
    print("CREATING SNV PAIRS FROM HASHFRAG TEST SET")
    print("=" * 80)
    
    # Step 1: Load HashFrag test set (reference alleles only)
    print("\n1. Loading HashFrag test set...")
    try:
        test_dataset = K562Dataset(
            data_path=str(Path(data_file).parent),
            split='test',
            use_hashfrag=True,
            hashfrag_cache_dir=hashfrag_cache_dir
        )
        test_sequences = test_dataset.sequences
        test_labels = test_dataset.labels
        test_indices = test_dataset.indices
        
        print(f"   HashFrag test set: {len(test_sequences):,} sequences")
    except Exception as e:
        raise RuntimeError(
            f"Could not load HashFrag test set: {e}\n"
            "Make sure HashFrag splits have been created first:\n"
            "  python scripts/create_hashfrag_splits.py"
        )
    
    # Step 2: Load full dataset to find alternate alleles
    print("\n2. Loading full K562 dataset to find alternate alleles...")
    df = pd.read_csv(data_file, sep='\t', dtype={'OL': str})
    print(f"   Total sequences in dataset: {len(df):,}")
    
    # Parse ID components
    id_parts = df['IDs'].str.split(':', expand=True)
    df['chr'] = id_parts[0]
    df['pos'] = id_parts[1]
    df['ref'] = id_parts[2]
    df['alt'] = id_parts[3]
    df['allele'] = id_parts[4]  # R=reference, A=alternate
    df['wc'] = id_parts[5]
    
    # Step 3: Map HashFrag test indices back to original dataset
    # The test_indices are indices into the filtered dataset (reference alleles only)
    # We need to map these back to the original dataset IDs
    
    print("\n3. Mapping HashFrag test sequences to original dataset...")
    
    # Load the filtered dataset to get the mapping
    # We need to apply the same filtering as K562Dataset._load_and_filter_data
    is_reference = df['allele'] == 'R'
    is_non_variant = (df['ref'] == 'NA') & (df['alt'] == 'NA')
    df_filtered = df[is_reference | is_non_variant].copy()
    
    # Filter by length
    df_filtered['seq_len'] = df_filtered['sequence'].str.len()
    df_filtered = df_filtered[df_filtered['seq_len'] >= 198].copy()
    df_filtered = df_filtered.drop(columns=['seq_len'])
    
    # Reset index to get mapping
    df_filtered = df_filtered.reset_index(drop=True)
    
    # Get the IDs of sequences in HashFrag test set
    test_ids_in_filtered = df_filtered.iloc[test_indices]['IDs'].values
    
    print(f"   Found {len(test_ids_in_filtered):,} test sequence IDs")
    
    # Step 4: Extract reference alleles from test set
    print("\n4. Extracting reference alleles from HashFrag test set...")
    test_ref_df = df[df['IDs'].isin(test_ids_in_filtered) & (df['allele'] == 'R')].copy()
    print(f"   Reference alleles in test set: {len(test_ref_df):,}")
    
    # Step 5: Find corresponding alternate alleles
    print("\n5. Finding corresponding alternate alleles...")
    
    # Create pair key: chr:pos:ref:alt
    test_ref_df['pair_key'] = (
        test_ref_df['chr'].astype(str) + ':' +
        test_ref_df['pos'].astype(str) + ':' +
        test_ref_df['ref'].astype(str) + ':' +
        test_ref_df['alt'].astype(str)
    )
    
    # Find alternate alleles with same pair_key
    alt_df = df[df['allele'] == 'A'].copy()
    alt_df['pair_key'] = (
        alt_df['chr'].astype(str) + ':' +
        alt_df['pos'].astype(str) + ':' +
        alt_df['ref'].astype(str) + ':' +
        alt_df['alt'].astype(str)
    )
    
    # Merge to create pairs
    snv_pairs = pd.merge(
        test_ref_df[[
            'pair_key', 'IDs', 'sequence', 'K562_log2FC', 'K562_lfcSE',
            'chr', 'pos', 'ref', 'alt'
        ]],
        alt_df[[
            'pair_key', 'IDs', 'sequence', 'K562_log2FC', 'K562_lfcSE'
        ]],
        on='pair_key',
        suffixes=('_ref', '_alt'),
        how='inner'  # Only keep pairs where both ref and alt exist
    )
    
    print(f"   Found {len(snv_pairs):,} ref/alt pairs")
    
    # Step 6: Calculate variant effect (delta log2FC)
    snv_pairs['delta_log2FC'] = (
        snv_pairs['K562_log2FC_alt'] - snv_pairs['K562_log2FC_ref']
    )
    
    # Step 7: Select and rename columns for evaluation script
    output_columns = [
        'pair_key',
        'IDs_ref',
        'sequence_ref',
        'K562_log2FC_ref',
        'K562_lfcSE_ref',
        'IDs_alt',
        'sequence_alt',
        'K562_log2FC_alt',
        'K562_lfcSE_alt',
        'delta_log2FC'
    ]
    
    snv_pairs_output = snv_pairs.rename(columns={
        'IDs_ref': 'IDs_ref',
        'sequence_ref': 'sequence_ref',
        'K562_log2FC_ref': 'K562_log2FC_ref',
        'K562_lfcSE_ref': 'K562_lfcSE_ref',
        'IDs_alt': 'IDs_alt',
        'sequence_alt': 'sequence_alt',
        'K562_log2FC_alt': 'K562_log2FC_alt',
        'K562_lfcSE_alt': 'K562_lfcSE_alt'
    })[output_columns]
    
    # Step 8: Save output
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    snv_pairs_output.to_csv(output_path, sep='\t', index=False)
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"HashFrag test set size: {len(test_sequences):,} sequences")
    print(f"Reference alleles in test set: {len(test_ref_df):,}")
    print(f"SNV pairs created: {len(snv_pairs):,}")
    print(f"\nSNV pair statistics:")
    print(f"  Mean |delta_log2FC|: {snv_pairs['delta_log2FC'].abs().mean():.3f}")
    print(f"  Median |delta_log2FC|: {snv_pairs['delta_log2FC'].abs().median():.3f}")
    print(f"  Min delta_log2FC: {snv_pairs['delta_log2FC'].min():.3f}")
    print(f"  Max delta_log2FC: {snv_pairs['delta_log2FC'].max():.3f}")
    print(f"\nSaved to: {output_path}")
    
    return snv_pairs_output


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Create SNV pairs from HashFrag test set only"
    )
    parser.add_argument(
        '--data-file',
        type=str,
        default="data/k562/DATA-Table_S2__MPRA_dataset.txt",
        help="Path to full K562 MPRA dataset"
    )
    parser.add_argument(
        '--output-file',
        type=str,
        default="data/k562/test_sets/test_hashfrag_snv_pairs.tsv",
        help="Path to save SNV pairs"
    )
    parser.add_argument(
        '--hashfrag-cache-dir',
        type=str,
        default="data/k562/hashfrag_splits",
        help="Directory with HashFrag splits"
    )
    
    args = parser.parse_args()
    
    create_hashfrag_snv_pairs(
        data_file=args.data_file,
        output_file=args.output_file,
        hashfrag_cache_dir=args.hashfrag_cache_dir
    )
