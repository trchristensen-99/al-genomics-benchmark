#!/usr/bin/env python3
"""
Yeast-specific evaluation script.

Evaluates yeast models on three test sets:
1. In-distribution (filtered test set)
2. Single Nucleotide Variants (SNV pairs)
3. Out-of-distribution (native/genomic sequences)
"""

import sys
import argparse
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader, TensorDataset
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.dream_rnn import DREAMRNN
from data.utils import reverse_complement, one_hot_encode


def load_model(checkpoint_path: str, device: torch.device) -> DREAMRNN:
    """Load a trained DREAM-RNN model from checkpoint."""
    config = {
        'input_channels': 6,  # ACGT + reverse complement flag + singleton flag
        'sequence_length': 150,  # 57bp + 80bp + 13bp plasmid context
        'dropout_cnn': 0.1,
        'dropout_lstm': 0.1,
        'hidden_dim': 320,
        'cnn_filters': 160  # Original Prix Fixe default
    }
    
    model = DREAMRNN(
        input_channels=config['input_channels'],
        sequence_length=config['sequence_length'],
        dropout_cnn=config['dropout_cnn'],
        dropout_lstm=config['dropout_lstm'],
        hidden_dim=config['hidden_dim'],
        cnn_filters=config['cnn_filters']
    )
    model.load_checkpoint(checkpoint_path)
    model = model.to(device)
    model.eval()
    return model


def predict_with_reverse_complement(
    model: DREAMRNN,
    sequences: np.ndarray,
    batch_size: int,
    device: torch.device,
    use_reverse_complement: bool = True
) -> np.ndarray:
    """
    Make predictions, optionally averaging with reverse complement predictions.
    Uses yeast-specific 6-channel encoding (ACGT + reverse complement flag + singleton flag).
    """
    # Encode sequences with 6 channels for yeast
    encoded = []
    for seq in sequences:
        # One-hot encode ACGT (4 channels)
        one_hot = one_hot_encode(seq, add_singleton_channel=False)  # Shape: (4, 150)
        # Add reverse complement channel (0 for forward strand)
        rc_channel = np.zeros((1, len(seq)), dtype=np.float32)
        # Add singleton channel (0 for test sequences - all are non-singleton)
        singleton_channel = np.zeros((1, len(seq)), dtype=np.float32)
        # Concatenate: (4, 150) + (1, 150) + (1, 150) = (6, 150)
        encoded_seq = np.concatenate([one_hot, rc_channel, singleton_channel], axis=0)
        encoded.append(encoded_seq)
    encoded = np.array(encoded)
    encoded_tensor = torch.from_numpy(encoded).float()
    
    dataset = TensorDataset(encoded_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    predictions = []
    with torch.no_grad():
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            pred = model(batch_x).squeeze().cpu().numpy()
            predictions.extend(pred if pred.ndim > 0 else [pred.item()])
    
    predictions = np.array(predictions)
    
    if use_reverse_complement:
        # Get reverse complement predictions
        rc_sequences = [reverse_complement(seq) for seq in sequences]
        # Encode reverse complement with RC flag = 1
        rc_encoded = []
        for seq in rc_sequences:
            one_hot = one_hot_encode(seq, add_singleton_channel=False)
            rc_channel = np.ones((1, len(seq)), dtype=np.float32)  # RC flag = 1
            singleton_channel = np.zeros((1, len(seq)), dtype=np.float32)  # Still non-singleton
            encoded_seq = np.concatenate([one_hot, rc_channel, singleton_channel], axis=0)
            rc_encoded.append(encoded_seq)
        rc_encoded = np.array(rc_encoded)
        rc_tensor = torch.from_numpy(rc_encoded).float()
        
        rc_dataset = TensorDataset(rc_tensor)
        rc_loader = DataLoader(rc_dataset, batch_size=batch_size, shuffle=False)
        
        rc_predictions = []
        with torch.no_grad():
            for (batch_x,) in rc_loader:
                batch_x = batch_x.to(device)
                pred = model(batch_x).squeeze().cpu().numpy()
                rc_predictions.extend(pred if pred.ndim > 0 else [pred.item()])
        
        rc_predictions = np.array(rc_predictions)
        
        # Average forward and reverse complement predictions
        predictions = (predictions + rc_predictions) / 2
    
    return predictions


def evaluate_yeast_in_distribution(
    model: DREAMRNN,
    test_file: Path,
    device: torch.device,
    batch_size: int = 512
) -> dict:
    """Evaluate on yeast in-distribution test set."""
    print("\n" + "="*80)
    print("Yeast In-Distribution Test")
    print("="*80)
    
    # Load test file (tab-separated, no header, sequence + expression)
    df = pd.read_csv(test_file, sep='\t', header=None, names=['sequence', 'expression'])
    sequences = df['sequence'].values
    labels = df['expression'].values.astype(np.float32)
    
    print(f"Test sequences: {len(sequences):,}")
    
    predictions = predict_with_reverse_complement(
        model, sequences, batch_size, device, use_reverse_complement=True
    )
    
    pearson_r, _ = pearsonr(labels, predictions)
    spearman_r, _ = spearmanr(labels, predictions)
    mse = np.mean((predictions - labels) ** 2)
    
    print(f"Pearson R: {pearson_r:.4f}")
    print(f"Spearman R: {spearman_r:.4f}")
    print(f"MSE: {mse:.4f}")
    
    return {
        'test_type': 'in_distribution',
        'dataset': 'yeast',
        'n_samples': len(sequences),
        'pearson_r': float(pearson_r),
        'spearman_r': float(spearman_r),
        'mse': float(mse)
    }


def evaluate_yeast_snv(
    model: DREAMRNN,
    test_file: Path,
    snv_file: Path,
    device: torch.device,
    batch_size: int = 512
) -> dict:
    """Evaluate on yeast SNV test subset."""
    print("\n" + "="*80)
    print("Yeast Single Nucleotide Variant (SNV) Test")
    print("="*80)
    
    # Load full test set (tab-separated, no header)
    df_test = pd.read_csv(test_file, sep='\t', header=None, names=['sequence', 'expression'])
    
    # Load SNV data with sequences
    df_snv = pd.read_csv(snv_file)
    
    # Get unique sequences from SNV data (both ref and alt)
    snv_sequences = set()
    if 'ref_sequence' in df_snv.columns:
        snv_sequences.update(df_snv['ref_sequence'].dropna().values)
    if 'alt_sequence' in df_snv.columns:
        snv_sequences.update(df_snv['alt_sequence'].dropna().values)
    
    # Filter test set to SNV sequences
    df_filtered = df_test[df_test['sequence'].isin(snv_sequences)].copy()
    
    print(f"SNV test sequences: {len(df_filtered):,}")
    
    sequences = df_filtered['sequence'].values
    labels = df_filtered['expression'].values.astype(np.float32)
    
    predictions = predict_with_reverse_complement(
        model, sequences, batch_size, device, use_reverse_complement=True
    )
    
    pearson_r, _ = pearsonr(labels, predictions)
    spearman_r, _ = spearmanr(labels, predictions)
    mse = np.mean((predictions - labels) ** 2)
    
    print(f"Pearson R: {pearson_r:.4f}")
    print(f"Spearman R: {spearman_r:.4f}")
    print(f"MSE: {mse:.4f}")
    
    return {
        'test_type': 'snv',
        'dataset': 'yeast',
        'n_samples': len(df_filtered),
        'pearson_r': float(pearson_r),
        'spearman_r': float(spearman_r),
        'mse': float(mse)
    }


def evaluate_yeast_ood(
    model: DREAMRNN,
    test_file: Path,
    ood_file: Path,
    device: torch.device,
    batch_size: int = 512
) -> dict:
    """Evaluate on yeast out-of-distribution (native/genomic) sequences."""
    print("\n" + "="*80)
    print("Yeast Out-of-Distribution (OOD) Test - Native/Genomic Sequences")
    print("="*80)
    
    # Load test set
    df_test = pd.read_csv(test_file, sep='\t', header=None, names=['sequence', 'expression'])
    
    # Load OOD sequences (native yeast sequences)
    # Format: CSV with columns: tag, sequence, pos, exp
    df_ood = pd.read_csv(ood_file)
    
    # Get sequences from OOD file (column name is 'sequence')
    ood_sequences = set(df_ood['sequence'].dropna().values)
    
    # Filter test set to OOD sequences
    df_filtered = df_test[df_test['sequence'].isin(ood_sequences)].copy()
    
    print(f"OOD test sequences (native/genomic): {len(df_filtered):,}")
    
    sequences = df_filtered['sequence'].values
    labels = df_filtered['expression'].values.astype(np.float32)
    
    predictions = predict_with_reverse_complement(
        model, sequences, batch_size, device, use_reverse_complement=True
    )
    
    pearson_r, _ = pearsonr(labels, predictions)
    spearman_r, _ = spearmanr(labels, predictions)
    mse = np.mean((predictions - labels) ** 2)
    
    print(f"Pearson R: {pearson_r:.4f}")
    print(f"Spearman R: {spearman_r:.4f}")
    print(f"MSE: {mse:.4f}")
    
    return {
        'test_type': 'ood',
        'dataset': 'yeast',
        'n_samples': len(df_filtered),
        'pearson_r': float(pearson_r),
        'spearman_r': float(spearman_r),
        'mse': float(mse)
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate yeast model on all test sets (in-distribution, SNV, OOD)"
    )
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file for results')
    parser.add_argument('--batch-size', type=int, default=512,
                        help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"\nLoading model from {args.checkpoint}...")
    model = load_model(args.checkpoint, device)
    print("Model loaded successfully!")
    
    # Run evaluations
    results = []
    data_dir = Path('./data/yeast')
    
    # 1. In-distribution test
    test_file = data_dir / 'filtered_test_data_with_MAUDE_expression.txt'
    if test_file.exists():
        results.append(evaluate_yeast_in_distribution(
            model, test_file, device, args.batch_size
        ))
    else:
        print(f"Warning: {test_file} not found")
    
    # 2. SNV test
    snv_file = data_dir / 'test_subset_ids' / 'all_SNVs_seqs.csv'
    if snv_file.exists() and test_file.exists():
        results.append(evaluate_yeast_snv(
            model, test_file, snv_file, device, args.batch_size
        ))
    else:
        print(f"Warning: SNV test files not found, skipping SNV test")
    
    # 3. OOD test (native/genomic sequences)
    ood_file = data_dir / 'test_subset_ids' / 'yeast_seqs.csv'
    if ood_file.exists() and test_file.exists():
        results.append(evaluate_yeast_ood(
            model, test_file, ood_file, device, args.batch_size
        ))
    else:
        print(f"Warning: OOD test files not found, skipping OOD test")
    
    # Print summary
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    
    for result in results:
        test_type = result['test_type'].upper()
        n_samples = result.get('n_samples', result.get('n_pairs', 'N/A'))
        print(f"\n{test_type} (yeast):")
        print(f"  Samples: {n_samples:,}" if isinstance(n_samples, int) else f"  Samples: {n_samples}")
        print(f"  Pearson R: {result['pearson_r']:.4f}")
        print(f"  Spearman R: {result['spearman_r']:.4f}")
    
    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")
    else:
        print("\nResults not saved (use --output to specify output file)")


if __name__ == '__main__':
    main()
