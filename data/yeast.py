"""
Yeast promoter MPRA dataset loader.

Dataset from: de Boer et al., Nature Biotechnology 2024
Zenodo: https://zenodo.org/records/10633252
"""

import os
from typing import Dict, Optional
import numpy as np
import pandas as pd
from pathlib import Path

from .base import SequenceDataset
from .utils import one_hot_encode


class YeastDataset(SequenceDataset):
    """
    Yeast promoter MPRA dataset.
    
    Dataset characteristics:
    - Promoter sequences
    - ~80bp each
    - Functional readout: Promoter activity (continuous)
    
    Expected file structure:
        data_path/
        ├── train.txt
        ├── val.txt
        └── test.txt
    
    Expected format (from Zenodo):
        Tab-separated or space-separated file with columns:
        - sequence: DNA sequence string
        - activity: Promoter activity measurement
    """
    
    SEQUENCE_LENGTH = 80
    NUM_CHANNELS = 4  # ACGT only (no singleton flag for yeast)
    
    def load_data(self) -> None:
        """
        Load yeast MPRA data from files.
        
        The data is from the Random Promoter DREAM Challenge.
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
        
        print(f"Loading Yeast {self.split} data from {file_path}")
        
        # Load data - try different delimiters
        df = None
        for delimiter in ['\t', ',', ' ', None]:
            try:
                df = pd.read_csv(file_path, delimiter=delimiter)
                if len(df.columns) >= 2:  # Need at least sequence and activity
                    break
            except Exception:
                continue
        
        if df is None or len(df.columns) < 2:
            raise RuntimeError(f"Could not parse data from {file_path}")
        
        # Extract sequences and labels
        # TODO: Update column names to match actual dataset once downloaded
        sequence_col = self._find_column(df, ['sequence', 'seq', 'Sequence', 'SEQ', 'promoter'])
        activity_col = self._find_column(df, ['activity', 'expression', 'Activity', 'Expression', 'label', 'score'])
        
        self.sequences = df[sequence_col].values
        self.labels = df[activity_col].values.astype(np.float32)
        
        # Validate and standardize sequence lengths
        seq_lengths = [len(seq) for seq in self.sequences]
        unique_lengths = set(seq_lengths)
        
        if len(unique_lengths) == 1:
            self.sequence_length = list(unique_lengths)[0]
        else:
            print(f"Warning: Found variable sequence lengths: {unique_lengths}")
            print(f"Using most common length as target: {self.SEQUENCE_LENGTH}")
            self.sequence_length = self.SEQUENCE_LENGTH
            # Filter or pad sequences to target length
            self.sequences, self.labels = self._standardize_lengths(
                self.sequences, self.labels
            )
        
        # No additional metadata for yeast dataset
        self.metadata = None
        
        print(f"Loaded {len(self.sequences)} sequences for {self.split} split")
        print(f"Sequence length: {self.sequence_length}")
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
    
    def _standardize_lengths(
        self, 
        sequences: np.ndarray, 
        labels: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Standardize sequence lengths by filtering to target length.
        
        For now, just keep sequences that match target length.
        Could be extended to pad/truncate if needed.
        """
        mask = np.array([len(seq) == self.sequence_length for seq in sequences])
        filtered_seqs = sequences[mask]
        filtered_labels = labels[mask]
        
        print(f"Filtered {len(sequences)} -> {len(filtered_seqs)} sequences matching length {self.sequence_length}")
        
        return filtered_seqs, filtered_labels
    
    def encode_sequence(self, sequence: str, metadata: Optional[Dict] = None) -> np.ndarray:
        """
        Encode a yeast sequence with one-hot encoding.
        
        Args:
            sequence: DNA sequence string (~80bp)
            metadata: Optional metadata (not used for yeast)
            
        Returns:
            Encoded sequence of shape (4, seq_length)
            - 4 channels: one-hot encoded ACGT
        """
        encoded = one_hot_encode(
            sequence,
            add_singleton_channel=False
        )
        
        return encoded
    
    def get_num_channels(self) -> int:
        """Return number of input channels (4 for yeast: ACGT only)."""
        return self.NUM_CHANNELS
