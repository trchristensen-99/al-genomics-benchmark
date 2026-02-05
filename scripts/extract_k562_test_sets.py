#!/usr/bin/env python3
"""
Extract K562 test sets from Gosai et al. MPRA data.

Based on paper methodology:
- Test chromosomes: 7 and 13 (held out)
- Validation chromosomes: 19, 21, X (held out)
- SNV pairs: ref/alt alleles for variant effect prediction
"""

import pandas as pd
import numpy as np
from pathlib import Path

def extract_test_sets(data_file: str = "data/k562/DATA-Table_S2__MPRA_dataset.txt",
                     output_dir: str = "data/k562/test_sets"):
    """Extract and organize K562 test sets."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("Loading MPRA data...")
    df = pd.read_csv(data_file, sep='\t', dtype={'OL': str})
    
    print(f"Total sequences: {len(df):,}")
    
    # Parse ID components
    print("\nParsing sequence IDs...")
    id_parts = df['IDs'].str.split(':', expand=True)
    df['chr'] = id_parts[0]
    df['pos'] = id_parts[1]
    df['ref'] = id_parts[2]
    df['alt'] = id_parts[3]
    df['allele'] = id_parts[4]  # A=alternate, R=reference
    df['wc'] = id_parts[5]
    
    # 1. Extract test set (chromosomes 7 and 13)
    print("\n" + "="*80)
    print("EXTRACTING TEST SET (Chromosomes 7 and 13)")
    print("="*80)
    
    test_chr = df['chr'].isin(['7', '13'])
    test_df = df[test_chr].copy()
    
    print(f"Test set size: {len(test_df):,} sequences")
    print(f"  Chr 7: {len(test_df[test_df['chr']=='7']):,}")
    print(f"  Chr 13: {len(test_df[test_df['chr']=='13']):,}")
    
    # Save full test set
    test_df.to_csv(output_path / "test_chr7_13_all.tsv", sep='\t', index=False)
    print(f"Saved: {output_path / 'test_chr7_13_all.tsv'}")
    
    # 2. Extract SNV pairs (ref/alt)
    print("\n" + "="*80)
    print("EXTRACTING SNV PAIRS FOR VARIANT EFFECT PREDICTION")
    print("="*80)
    
    # Sequences with ref/alt annotations
    has_snv = (test_df['ref'] != 'NA') & (test_df['alt'] != 'NA') & (test_df['allele'].isin(['A', 'R']))
    snv_df = test_df[has_snv].copy()
    
    print(f"SNV sequences: {len(snv_df):,}")
    print(f"  Reference alleles (R): {len(snv_df[snv_df['allele']=='R']):,}")
    print(f"  Alternate alleles (A): {len(snv_df[snv_df['allele']=='A']):,}")
    
    # Create SNV pairs dataframe
    snv_ref = snv_df[snv_df['allele']=='R'].copy()
    snv_alt = snv_df[snv_df['allele']=='A'].copy()
    
    # Create pair key
    snv_ref['pair_key'] = snv_ref['chr'] + ':' + snv_ref['pos'] + ':' + snv_ref['ref'] + ':' + snv_ref['alt']
    snv_alt['pair_key'] = snv_alt['chr'] + ':' + snv_alt['pos'] + ':' + snv_alt['ref'] + ':' + snv_alt['alt']
    
    # Merge to create pairs
    snv_pairs = pd.merge(
        snv_ref[['pair_key', 'IDs', 'sequence', 'K562_log2FC', 'K562_lfcSE']],
        snv_alt[['pair_key', 'IDs', 'sequence', 'K562_log2FC', 'K562_lfcSE']],
        on='pair_key',
        suffixes=('_ref', '_alt')
    )
    
    # Calculate variant effect (delta log2FC)
    snv_pairs['delta_log2FC'] = snv_pairs['K562_log2FC_alt'] - snv_pairs['K562_log2FC_ref']
    
    print(f"\nMatched SNV pairs: {len(snv_pairs):,}")
    print(f"Mean |delta_log2FC|: {snv_pairs['delta_log2FC'].abs().mean():.3f}")
    print(f"Median |delta_log2FC|: {snv_pairs['delta_log2FC'].abs().median():.3f}")
    
    snv_pairs.to_csv(output_path / "test_snv_pairs.tsv", sep='\t', index=False)
    print(f"Saved: {output_path / 'test_snv_pairs.tsv'}")
    
    # 3. Extract CRE sequences (natural regulatory elements)
    print("\n" + "="*80)
    print("EXTRACTING CRE SEQUENCES (Natural Regulatory Elements)")
    print("="*80)
    
    cre_df = df[df['data_project'] == 'CRE'].copy()
    
    print(f"CRE sequences: {len(cre_df):,}")
    print(f"CRE classes: {cre_df['class'].value_counts().to_dict()}")
    
    cre_df.to_csv(output_path / "cre_sequences.tsv", sep='\t', index=False)
    print(f"Saved: {output_path / 'cre_sequences.tsv'}")
    
    # 4. Extract validation set (chromosomes 19, 21, X)
    print("\n" + "="*80)
    print("EXTRACTING VALIDATION SET (Chromosomes 19, 21, X)")
    print("="*80)
    
    val_chr = df['chr'].isin(['19', '21', 'X'])
    val_df = df[val_chr].copy()
    
    print(f"Validation set size: {len(val_df):,} sequences")
    print(f"  Chr 19: {len(val_df[val_df['chr']=='19']):,}")
    print(f"  Chr 21: {len(val_df[val_df['chr']=='21']):,}")
    print(f"  Chr X: {len(val_df[val_df['chr']=='X']):,}")
    
    val_df.to_csv(output_path / "val_chr19_21_X.tsv", sep='\t', index=False)
    print(f"Saved: {output_path / 'val_chr19_21_X.tsv'}")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total sequences: {len(df):,}")
    print(f"Test set (chr 7, 13): {len(test_df):,}")
    print(f"  - SNV pairs: {len(snv_pairs):,}")
    print(f"  - CRE sequences: {len(cre_df):,}")
    print(f"Validation set (chr 19, 21, X): {len(val_df):,}")
    print(f"\nNote: Synthetic sequences (OOD test) not found in this dataset.")
    print(f"      They may be in a separate CODA library repository.")

if __name__ == '__main__':
    extract_test_sets()
