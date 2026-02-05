#!/usr/bin/env python3
"""
Comprehensive evaluation script for all test sets:
1. In-distribution (held-out chromosomes for K562, filtered test for yeast)
2. Single Nucleotide Variants (SNV pairs for variant effect prediction)
3. Out-of-distribution (synthetic for K562, native/genomic for yeast)
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
from data.utils import one_hot_encode


def load_model(checkpoint_path: str, config: dict, device: torch.device) -> DREAMRNN:
    """Load a trained DREAM-RNN model from checkpoint."""
    model = DREAMRNN(
        input_channels=config['input_channels'],
        sequence_length=config['sequence_length'],
        dropout=config.get('dropout', 0.2)
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
    """
    from data.utils import reverse_complement
    
    # Encode sequences
    encoded = np.array([one_hot_encode(seq) for seq in sequences])
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
        rc_encoded = np.array([one_hot_encode(seq) for seq in rc_sequences])
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


def evaluate_k562_in_distribution(
    model: DREAMRNN,
    test_file: Path,
    device: torch.device,
    batch_size: int = 512
) -> dict:
    """Evaluate on K562 in-distribution test set (chromosomes 7, 13)."""
    print("\n" + "="*80)
    print("K562 In-Distribution Test (Chromosomes 7, 13)")
    print("="*80)
    
    df = pd.read_csv(test_file, sep='\t')
    sequences = df['sequence'].values
    labels = df['K562_log2FC'].values.astype(np.float32)
    
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
        'dataset': 'k562',
        'n_samples': len(sequences),
        'pearson_r': float(pearson_r),
        'spearman_r': float(spearman_r),
        'mse': float(mse)
    }


def evaluate_k562_snv(
    model: DREAMRNN,
    snv_file: Path,
    device: torch.device,
    batch_size: int = 512
) -> dict:
    """Evaluate on K562 SNV pairs for variant effect prediction."""
    print("\n" + "="*80)
    print("K562 Single Nucleotide Variant (SNV) Test")
    print("="*80)
    
    df = pd.read_csv(snv_file, sep='\t')
    
    print(f"SNV pairs: {len(df):,}")
    
    # Predict reference and alternate sequences
    ref_pred = predict_with_reverse_complement(
        model, df['sequence_ref'].values, batch_size, device
    )
    alt_pred = predict_with_reverse_complement(
        model, df['sequence_alt'].values, batch_size, device
    )
    
    # Calculate predicted delta
    predicted_delta = alt_pred - ref_pred
    true_delta = df['delta_log2FC'].values
    
    pearson_r, _ = pearsonr(true_delta, predicted_delta)
    spearman_r, _ = spearmanr(true_delta, predicted_delta)
    mse = np.mean((predicted_delta - true_delta) ** 2)
    
    print(f"Pearson R (delta): {pearson_r:.4f}")
    print(f"Spearman R (delta): {spearman_r:.4f}")
    print(f"MSE (delta): {mse:.4f}")
    
    return {
        'test_type': 'snv',
        'dataset': 'k562',
        'n_pairs': len(df),
        'pearson_r': float(pearson_r),
        'spearman_r': float(spearman_r),
        'mse': float(mse)
    }


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
    snv_ids_file: Path,
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
    df_snv = pd.read_csv(snv_ids_file)
    
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
    ood_ids_file: Path,
    device: torch.device,
    batch_size: int = 512
) -> dict:
    """Evaluate on yeast out-of-distribution (native/genomic) test subset."""
    print("\n" + "="*80)
    print("Yeast Out-of-Distribution Test (Native/Genomic Sequences)")
    print("="*80)
    
    # Load full test set (tab-separated, no header)
    df_test = pd.read_csv(test_file, sep='\t', header=None, names=['sequence', 'expression'])
    
    # Load OOD sequences (native/genomic)
    df_ood = pd.read_csv(ood_ids_file)
    
    # Get sequences from OOD file
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
        description="Evaluate model on all test sets (in-distribution, SNV, OOD)"
    )
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['k562', 'yeast'],
                        help='Dataset to evaluate')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file for results')
    parser.add_argument('--batch-size', type=int, default=512,
                        help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model configuration
    if args.dataset == 'k562':
        config = {
            'input_channels': 5,
            'sequence_length': 230,
            'dropout': 0.2
        }
        data_dir = Path('./data/k562')
    else:  # yeast
        config = {
            'input_channels': 4,
            'sequence_length': 110,
            'dropout': 0.2
        }
        data_dir = Path('./data/yeast')
    
    # Load model
    print(f"\nLoading model from {args.checkpoint}...")
    model = load_model(args.checkpoint, config, device)
    print("Model loaded successfully!")
    
    # Run evaluations
    results = []
    
    if args.dataset == 'k562':
        # 1. In-distribution test
        test_file = data_dir / 'test_sets' / 'test_chr7_13_all.tsv'
        if test_file.exists():
            results.append(evaluate_k562_in_distribution(
                model, test_file, device, args.batch_size
            ))
        else:
            print(f"Warning: {test_file} not found, skipping in-distribution test")
        
        # 2. SNV test
        snv_file = data_dir / 'test_sets' / 'test_snv_pairs.tsv'
        if snv_file.exists():
            results.append(evaluate_k562_snv(
                model, snv_file, device, args.batch_size
            ))
        else:
            print(f"Warning: {snv_file} not found, skipping SNV test")
        
        # 3. OOD test (synthetic sequences - if available)
        print("\nNote: K562 OOD test (synthetic sequences) not yet available")
        print("      This requires the CODA library data from the paper")
    
    else:  # yeast
        test_file = data_dir / 'filtered_test_data_with_MAUDE_expression.txt'
        
        # 1. In-distribution test
        if test_file.exists():
            results.append(evaluate_yeast_in_distribution(
                model, test_file, device, args.batch_size
            ))
        else:
            print(f"Warning: {test_file} not found")
        
        # 2. SNV test
        snv_ids_file = data_dir / 'test_subset_ids' / 'all_SNVs_seqs.csv'
        if test_file.exists() and snv_ids_file.exists():
            results.append(evaluate_yeast_snv(
                model, test_file, snv_ids_file, device, args.batch_size
            ))
        else:
            print(f"Warning: SNV test files not found")
        
        # 3. OOD test (native/genomic sequences)
        ood_ids_file = data_dir / 'test_subset_ids' / 'yeast_seqs.csv'
        if test_file.exists() and ood_ids_file.exists():
            results.append(evaluate_yeast_ood(
                model, test_file, ood_ids_file, device, args.batch_size
            ))
        else:
            print(f"Warning: OOD test files not found")
    
    # Print summary
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    for result in results:
        print(f"\n{result['test_type'].upper()} ({result['dataset']}):")
        print(f"  Samples: {result.get('n_samples', result.get('n_pairs', 0)):,}")
        print(f"  Pearson R: {result['pearson_r']:.4f}")
        print(f"  Spearman R: {result['spearman_r']:.4f}")
    
    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")
    
    return results


if __name__ == '__main__':
    main()
