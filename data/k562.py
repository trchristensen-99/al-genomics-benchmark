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
    
    SEQUENCE_LENGTH = 230
    NUM_CHANNELS = 5  # ACGT + is_singleton flag
    FLANKING_SEQUENCE = "N" * 10  # Placeholder for constant plasmid flanking regions
    
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
        
        Tries multiple file formats: .txt, .csv, .tsv
        """
        data_dir = Path(self.data_path)
        
        # Try different file extensions
        for ext in ['.txt', '.csv', '.tsv']:
            file_path = data_dir / f"{self.split}{ext}"
            if file_path.exists():
                break
        else:
            raise FileNotFoundError(
                f"Could not find {self.split} data file in {data_dir}. "
                f"Expected one of: {self.split}.txt, {self.split}.csv, {self.split}.tsv"
            )
        
        print(f"Loading K562 {self.split} data from {file_path}")
        
        # Determine delimiter
        if ext == '.csv':
            delimiter = ','
        elif ext == '.tsv':
            delimiter = '\t'
        else:
            delimiter = None  # Let pandas infer
        
        # Load data
        # TODO: Adjust column names based on actual data format once downloaded
        try:
            df = pd.read_csv(file_path, delimiter=delimiter)
        except Exception as e:
            raise RuntimeError(f"Error loading K562 data from {file_path}: {e}")
        
        # Extract sequences and labels
        # TODO: Update column names to match actual dataset
        sequence_col = self._find_column(df, ['sequence', 'seq', 'Sequence', 'SEQ'])
        activity_col = self._find_column(df, ['activity', 'expression', 'Activity', 'Expression', 'label'])
        singleton_col = self._find_column(df, ['is_singleton', 'singleton', 'Singleton'], required=False)
        
        self.sequences = df[sequence_col].values
        self.labels = df[activity_col].values.astype(np.float32)
        
        # Process sequences (add flanking if needed)
        if self.use_flanking:
            self.sequences = self._add_flanking_regions(self.sequences)
        
        # Store metadata
        self.metadata = {}
        if singleton_col is not None:
            self.metadata['is_singleton'] = df[singleton_col].values.astype(bool)
        else:
            # Default: assume all are non-singletons
            self.metadata['is_singleton'] = np.zeros(len(self.sequences), dtype=bool)
        
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
