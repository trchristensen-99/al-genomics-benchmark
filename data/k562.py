"""
K562 human cell line MPRA dataset loader.

Dataset from: Gosai et al., Nature 2023
Zenodo: https://zenodo.org/records/10698014
"""

import os
from typing import Dict, Optional
import numpy as np
import pandas as pd
from pathlib import Path

from .base import SequenceDataset
from .utils import one_hot_encode


class K562Dataset(SequenceDataset):
    """
    K562 human MPRA dataset.
    
    Dataset characteristics:
    - ~370K regulatory sequences
    - ~230bp each
    - Functional readout: Gene expression values (continuous)
    - Metadata: singleton flags, genomic coordinates
    
    Expected file structure:
        data_path/
        ├── train.txt (or .csv, .tsv)
        ├── val.txt
        └── test.txt
    
    Expected columns:
        - sequence: DNA sequence string
        - activity: Expression measurement (continuous)
        - [optional] chrom, start, end: Genomic coordinates
        
    Note: is_singleton flag removed (was all zeros, not used)
    """
    
    SEQUENCE_LENGTH = 230  # Target length after adding flanking regions
    NATIVE_LENGTH = 200  # Most common native sequence length in the dataset
    NUM_CHANNELS = 4  # ACGT only (is_singleton flag removed as it's all zeros)
    FLANKING_SEQUENCE = "N" * 15  # Placeholder for constant plasmid flanking regions (15bp each side)
    
    def __init__(
        self,
        data_path: str,
        split: str = "train",
        transform: Optional[any] = None,
        target_transform: Optional[any] = None,
        use_flanking: bool = True
    ):
        """
        Initialize K562 dataset.
        
        Args:
            data_path: Path to data directory containing train/val/test files
            split: Which split to load ("train", "val", or "test")
            transform: Optional transform to apply to sequences
            target_transform: Optional transform to apply to labels
            use_flanking: Whether to add constant flanking regions to sequences
        """
        self.use_flanking = use_flanking
        super().__init__(data_path, split, transform, target_transform)
    
    def load_data(self) -> None:
        """
        Load K562 MPRA data from files.
        
        The K562 dataset comes as a single file with all data.
        """
        data_dir = Path(self.data_path)
        
        # The actual filename from the Zenodo download
        file_path = data_dir / 'DATA-Table_S2__MPRA_dataset.txt'
        
        if not file_path.exists():
            raise FileNotFoundError(
                f"Could not find K562 data file at {file_path}. "
                f"Please run: python scripts/download_data.py --dataset k562"
            )
        
        print(f"Loading K562 {self.split} data from {file_path}")
        
        # Load data (tab-separated with header)
        try:
            df = pd.read_csv(file_path, sep='\t', dtype={'OL': str})
        except Exception as e:
            raise RuntimeError(f"Error loading K562 data from {file_path}: {e}")
        
        # Filter to reference alleles only (matching benchmark paper: 367K sequences)
        # Parse ID format: chr:pos:ref:alt:allele_type:wc
        id_parts = df['IDs'].str.split(':', expand=True)
        allele_type = id_parts[4]  # R=reference, A=alternate, empty=CRE/no variant
        ref_col = id_parts[2]
        alt_col = id_parts[3]
        
        # Keep reference alleles (R) and non-variant sequences (NA:NA)
        is_reference = allele_type == 'R'
        is_non_variant = (ref_col == 'NA') & (alt_col == 'NA')
        n_before = len(df)
        df = df[is_reference | is_non_variant].copy()
        n_after = len(df)
        
        print(f"  Filtered to {n_after:,} reference alleles (excluded {n_before - n_after:,} alternate alleles)")
        
        # Filter by sequence length (paper uses sequences >= 198bp)
        # This matches the paper's 367,364 sequences (our data: 367,793 with >= 198bp filter)
        df['seq_len'] = df['sequence'].str.len()
        n_before_len = len(df)
        df = df[df['seq_len'] >= 198].copy()
        n_after_len = len(df)
        
        print(f"  Length filter (>= 198bp): {n_after_len:,} sequences (excluded {n_before_len - n_after_len:,} shorter sequences)")
        df = df.drop(columns=['seq_len'])
        
        # Expected columns from the K562 lentiMPRA dataset:
        # IDs, chr, data_project, OL, class, K562_log2FC, HepG2_log2FC, SKNSH_log2FC, 
        # K562_lfcSE, HepG2_lfcSE, SKNSH_lfcSE, sequence
        
        if 'sequence' not in df.columns:
            raise ValueError(f"Data file must contain 'sequence' column. Found: {df.columns.tolist()}")
        
        if 'K562_log2FC' not in df.columns:
            raise ValueError(f"Data file must contain 'K562_log2FC' column. Found: {df.columns.tolist()}")
        
        # Apply chromosome-based splits (as per paper)
        # Train: all chromosomes except test (7, 13) - includes validation chromosomes
        # Val: chromosomes 19, 21, X (subset of training pool for early stopping)
        # Test: chromosomes 7, 13 (held-out for final evaluation)
        if self.split == 'train':
            df = df[~df['chr'].isin(['7', '13'])].copy()
            print(f"  Train split (excluding test chr 7, 13): {len(df):,} sequences")
        elif self.split == 'val':
            df = df[df['chr'].isin(['19', '21', 'X'])].copy()
            print(f"  Validation split (chr 19, 21, X): {len(df):,} sequences")
        elif self.split == 'test':
            df = df[df['chr'].isin(['7', '13'])].copy()
            print(f"  Test split (chr 7, 13): {len(df):,} sequences")
        
        # Use K562_log2FC as the expression value (K562 cell line specific data)
        self.sequences = df['sequence'].values
        self.labels = df['K562_log2FC'].values.astype(np.float32)
        
        # Process sequences (add flanking if needed)
        if self.use_flanking:
            self.sequences = self._add_flanking_regions(self.sequences)
        
        # Store metadata (singleton flag removed as it was all zeros)
        self.metadata = {}
        
        # Store additional metadata if available
        metadata_cols = ['IDs', 'chr', 'data_project', 'OL', 'class', 'K562_lfcSE']
        for col in metadata_cols:
            if col in df.columns:
                self.metadata[col] = df[col].values
        
        # Store sequence length
        self.sequence_length = self.SEQUENCE_LENGTH
        
        print(f"Loaded {len(self.sequences)} sequences for {self.split} split")
        print(f"Label range: [{np.min(self.labels):.3f}, {np.max(self.labels):.3f}]")
    
    def _find_column(self, df: pd.DataFrame, possible_names: list, required: bool = True) -> Optional[str]:
        """Find a column in the dataframe by trying multiple possible names."""
        for name in possible_names:
            if name in df.columns:
                return name
        
        if required:
            raise ValueError(
                f"Could not find required column. Tried: {possible_names}. "
                f"Available columns: {list(df.columns)}"
            )
        return None
    
    def _add_flanking_regions(self, sequences: np.ndarray) -> np.ndarray:
        """
        Add constant plasmid flanking regions to sequences.
        
        This extends sequences to exactly 230bp using the same flanking
        regions that were present in the experimental plasmid construct.
        """
        # TODO: Replace with actual constant flanking sequences from paper
        # For now, just pad with Ns if needed
        processed = []
        for seq in sequences:
            curr_len = len(seq)
            if curr_len < self.SEQUENCE_LENGTH:
                # Center the sequence and pad with flanking regions
                pad_needed = self.SEQUENCE_LENGTH - curr_len
                left_pad = pad_needed // 2
                right_pad = pad_needed - left_pad
                padded = 'N' * left_pad + seq + 'N' * right_pad
                processed.append(padded)
            elif curr_len > self.SEQUENCE_LENGTH:
                # Truncate to target length (center)
                start = (curr_len - self.SEQUENCE_LENGTH) // 2
                processed.append(seq[start:start + self.SEQUENCE_LENGTH])
            else:
                processed.append(seq)
        
        # Ensure all sequences are exactly SEQUENCE_LENGTH
        processed = [seq if len(seq) == self.SEQUENCE_LENGTH else seq[:self.SEQUENCE_LENGTH] for seq in processed]
        
        return np.array(processed)
    
    def encode_sequence(self, sequence: str, metadata: Optional[Dict] = None) -> np.ndarray:
        """
        Encode a K562 sequence with one-hot encoding + singleton channel.
        
        Args:
            sequence: DNA sequence string (230bp)
            metadata: Dictionary with 'is_singleton' flag
            
        Returns:
            Encoded sequence of shape (5, 230)
            - First 4 channels: one-hot encoded ACGT
            - 5th channel: is_singleton flag (1.0 or 0.0)
        """
        is_singleton = False
        if metadata is not None and 'is_singleton' in metadata:
            is_singleton = metadata['is_singleton']
        
        encoded = one_hot_encode(
            sequence,
            add_singleton_channel=True,
            is_singleton=is_singleton
        )
        
        return encoded
    
    def get_num_channels(self) -> int:
        """Return number of input channels (5 for K562: ACGT + singleton flag)."""
        return self.NUM_CHANNELS
