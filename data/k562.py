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
        - is_singleton: Boolean flag indicating if sequence appears only once
        - [optional] chrom, start, end: Genomic coordinates
    """
    
    SEQUENCE_LENGTH = 230  # Target length after adding flanking regions
    NATIVE_LENGTH = 200  # Most common native sequence length in the dataset
    NUM_CHANNELS = 5  # ACGT + is_singleton flag
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
            df = pd.read_csv(file_path, sep='\t')
        except Exception as e:
            raise RuntimeError(f"Error loading K562 data from {file_path}: {e}")
        
        # Expected columns from the K562 lentiMPRA dataset:
        # IDs, chr, data_project, OL, class, K562_log2FC, HepG2_log2FC, SKNSH_log2FC, 
        # K562_lfcSE, HepG2_lfcSE, SKNSH_lfcSE, sequence
        
        if 'sequence' not in df.columns:
            raise ValueError(f"Data file must contain 'sequence' column. Found: {df.columns.tolist()}")
        
        if 'K562_log2FC' not in df.columns:
            raise ValueError(f"Data file must contain 'K562_log2FC' column. Found: {df.columns.tolist()}")
        
        # Use K562_log2FC as the expression value (K562 cell line specific data)
        self.sequences = df['sequence'].values
        self.labels = df['K562_log2FC'].values.astype(np.float32)
        
        # For now, we'll use the entire dataset for all splits
        # In a real scenario, you'd want to create proper train/val/test splits
        # TODO: Implement proper data splitting strategy
        
        # Process sequences (add flanking if needed)
        if self.use_flanking:
            self.sequences = self._add_flanking_regions(self.sequences)
        
        # Store metadata
        self.metadata = {}
        # The K562 dataset doesn't have explicit singleton flags in this format
        # Default: assume all are non-singletons
        self.metadata['is_singleton'] = np.zeros(len(self.sequences), dtype=bool)
        
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
            if len(seq) < self.SEQUENCE_LENGTH:
                # Center the sequence and pad with flanking regions
                pad_needed = self.SEQUENCE_LENGTH - len(seq)
                left_pad = pad_needed // 2
                right_pad = pad_needed - left_pad
                padded = self.FLANKING_SEQUENCE[:left_pad] + seq + self.FLANKING_SEQUENCE[:right_pad]
                processed.append(padded)
            elif len(seq) > self.SEQUENCE_LENGTH:
                # Truncate to target length (center)
                start = (len(seq) - self.SEQUENCE_LENGTH) // 2
                processed.append(seq[start:start + self.SEQUENCE_LENGTH])
            else:
                processed.append(seq)
        
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
