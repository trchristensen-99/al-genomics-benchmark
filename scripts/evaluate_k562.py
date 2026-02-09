#!/usr/bin/env python3
"""
K562-specific evaluation script.

Evaluates K562 models on test sets:
1. In-distribution (chromosomes 7, 13)
2. Single Nucleotide Variants (SNV pairs)
3. Out-of-distribution (synthetic sequences - if available)
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
from data.utils import reverse_complement, one_hot_encode, pad_sequence
from data.k562 import K562Dataset
from torch.utils.data import DataLoader


def load_model(checkpoint_path: str, device: torch.device) -> DREAMRNN:
    """Load a trained DREAM-RNN model from checkpoint."""
    config = {
        'input_channels': 5,  # ACGT + reverse complement flag
        'sequence_length': 200,  # K562 sequences are 200bp
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


def encode_k562_sequence(sequence: str) -> np.ndarray:
    """
    Encode a K562 sequence with 5 channels (ACGT + reverse complement flag).
    Pads/truncates to 200bp as needed.
    """
    # Pad or truncate to 200bp
    if len(sequence) < 200:
        padded = pad_sequence(sequence, target_length=200, pad_char='N', mode='both')
    elif len(sequence) > 200:
        start = (len(sequence) - 200) // 2
        padded = sequence[start:start + 200]
    else:
        padded = sequence
    
    # One-hot encode ACGT (4 channels)
    one_hot = one_hot_encode(padded, add_singleton_channel=False)  # Shape: (4, 200)
    
    # Add reverse complement channel (0 for forward strand)
    rc_channel = np.zeros((1, len(padded)), dtype=np.float32)
    
    # Concatenate: (4, 200) + (1, 200) = (5, 200)
    encoded = np.concatenate([one_hot, rc_channel], axis=0)
    
    return encoded


def predict_with_reverse_complement(
    model: DREAMRNN,
    sequences: np.ndarray,
    batch_size: int,
    device: torch.device,
    use_reverse_complement: bool = True
) -> np.ndarray:
    """
    Make predictions, optionally averaging with reverse complement predictions.
    Uses K562-specific encoding (5 channels).
    """
    # Encode sequences
    encoded = np.array([encode_k562_sequence(seq) for seq in sequences])
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
        rc_encoded = []
        for seq in rc_sequences:
            # Pad/truncate
            if len(seq) < 200:
                padded = pad_sequence(seq, target_length=200, pad_char='N', mode='both')
            elif len(seq) > 200:
                start = (len(seq) - 200) // 2
                padded = seq[start:start + 200]
            else:
                padded = seq
            
            # Encode with RC flag = 1
            one_hot = one_hot_encode(padded, add_singleton_channel=False)
            rc_channel = np.ones((1, len(padded)), dtype=np.float32)  # RC flag = 1
            encoded_seq = np.concatenate([one_hot, rc_channel], axis=0)
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


def evaluate_k562_in_distribution(
    model: DREAMRNN,
    data_dir: Path,
    device: torch.device,
    batch_size: int = 512,
    use_hashfrag: bool = True
) -> dict:
    """Evaluate on K562 in-distribution test set (HashFrag-based or chromosome-based)."""
    print("\n" + "="*80)
    if use_hashfrag:
        print("K562 In-Distribution Test (HashFrag-based, homology-aware split)")
    else:
        print("K562 In-Distribution Test (Chromosomes 7, 13)")
    print("="*80)
    
    # Load test set using K562Dataset (uses HashFrag splits by default)
    try:
        test_dataset = K562Dataset(
            data_path=str(data_dir),
            split='test',
            use_hashfrag=use_hashfrag
        )
        sequences = test_dataset.sequences  # These are numpy arrays of strings
        labels = test_dataset.labels
        
        print(f"Test sequences: {len(sequences):,}")
        
        # Convert sequences to list of strings for prediction function
        sequences_list = [seq if isinstance(seq, str) else str(seq) for seq in sequences]
        
        # Predict using reverse complement
        predictions = predict_with_reverse_complement(
            model, sequences_list, batch_size, device, use_reverse_complement=True
        )
        
        test_set_type = 'hashfrag'
        
    except Exception as e:
        # Fallback to old chromosome-based test set if HashFrag fails
        print(f"Warning: Could not load HashFrag test set: {e}")
        print("Falling back to chromosome-based test set...")
        
        test_file = data_dir / 'test_sets' / 'test_chr7_13_all.tsv'
        if not test_file.exists():
            raise FileNotFoundError(f"Neither HashFrag test set nor {test_file} found")
        
        df = pd.read_csv(test_file, sep='\t')
        sequences_list = df['sequence'].values.tolist()
        labels = df['K562_log2FC'].values.astype(np.float32)
        
        print(f"Test sequences: {len(sequences_list):,}")
        
        predictions = predict_with_reverse_complement(
            model, sequences_list, batch_size, device, use_reverse_complement=True
        )
        
        test_set_type = 'chromosome_based'
    
    pearson_r, _ = pearsonr(labels, predictions)
    spearman_r, _ = spearmanr(labels, predictions)
    mse = np.mean((predictions - labels) ** 2)
    
    print(f"Pearson R: {pearson_r:.4f}")
    print(f"Spearman R: {spearman_r:.4f}")
    print(f"MSE: {mse:.4f}")
    
    return {
        'test_type': 'in_distribution',
        'dataset': 'k562',
        'n_samples': len(labels),
        'pearson_r': float(pearson_r),
        'spearman_r': float(spearman_r),
        'mse': float(mse),
        'test_set_type': test_set_type
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
        model, df['sequence_ref'].values, batch_size, device, use_reverse_complement=True
    )
    alt_pred = predict_with_reverse_complement(
        model, df['sequence_alt'].values, batch_size, device, use_reverse_complement=True
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


def evaluate_k562_ood(
    model: DREAMRNN,
    ood_file: Path,
    device: torch.device,
    batch_size: int = 512
) -> dict:
    """Evaluate on K562 out-of-distribution (synthetic sequences)."""
    print("\n" + "="*80)
    print("K562 Out-of-Distribution (OOD) Test - Synthetic Sequences")
    print("="*80)
    
    # Load OOD sequences (synthetic sequences)
    df = pd.read_csv(ood_file, sep='\t')
    
    # Check for sequence and expression columns
    if 'sequence' not in df.columns:
        raise ValueError(f"OOD file must have 'sequence' column. Found columns: {df.columns.tolist()}")
    
    # Expression column might be named differently
    expr_col = None
    for col in ['K562_log2FC', 'expression', 'log2FC', 'expr']:
        if col in df.columns:
            expr_col = col
            break
    
    if expr_col is None:
        raise ValueError(f"OOD file must have expression column. Found columns: {df.columns.tolist()}")
    
    sequences = df['sequence'].values
    labels = df[expr_col].values.astype(np.float32)
    
    print(f"OOD test sequences (synthetic): {len(sequences):,}")
    
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
        'dataset': 'k562',
        'n_samples': len(sequences),
        'pearson_r': float(pearson_r),
        'spearman_r': float(spearman_r),
        'mse': float(mse)
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate K562 model on all test sets (in-distribution, SNV, OOD)"
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
    data_dir = Path('./data/k562')
    
    # 1. In-distribution test (use HashFrag test set by default)
    try:
        results.append(evaluate_k562_in_distribution(
            model, data_dir, device, args.batch_size, use_hashfrag=True
        ))
    except Exception as e:
        print(f"Warning: Could not evaluate on HashFrag test set: {e}")
        print("Attempting fallback to chromosome-based test set...")
        try:
            results.append(evaluate_k562_in_distribution(
                model, data_dir, device, args.batch_size, use_hashfrag=False
            ))
        except Exception as e2:
            print(f"Error: Could not evaluate on either test set: {e2}")
            print("Skipping in-distribution test")
    
    # 2. SNV test (use HashFrag-based SNV pairs if available, fallback to old)
    snv_file = data_dir / 'test_sets' / 'test_hashfrag_snv_pairs.tsv'
    if not snv_file.exists():
        # Fallback to old chromosome-based SNV pairs
        snv_file = data_dir / 'test_sets' / 'test_snv_pairs.tsv'
    
    if snv_file.exists():
        results.append(evaluate_k562_snv(
            model, snv_file, device, args.batch_size
        ))
    else:
        print(f"Warning: SNV test file not found, skipping SNV test")
        print(f"  Checked: {data_dir / 'test_sets' / 'test_hashfrag_snv_pairs.tsv'}")
        print(f"  Checked: {data_dir / 'test_sets' / 'test_snv_pairs.tsv'}")
        print(f"  Create HashFrag SNV pairs with: python scripts/create_k562_hashfrag_snv_pairs.py")
    
    # 3. OOD test (synthetic sequences - check multiple possible locations)
    ood_file = None
    possible_ood_paths = [
        data_dir / 'test_sets' / 'synthetic_sequences.tsv',
        data_dir / 'test_sets' / 'coda_library.tsv',
        data_dir / 'test_sets' / 'ood_synthetic.tsv',
        data_dir / 'synthetic_sequences.tsv',
        data_dir / 'coda_library.tsv',
    ]
    
    for path in possible_ood_paths:
        if path.exists():
            ood_file = path
            break
    
    if ood_file:
        try:
            results.append(evaluate_k562_ood(
                model, ood_file, device, args.batch_size
            ))
        except Exception as e:
            print(f"\nWarning: Failed to evaluate OOD test: {e}")
            print("      This may require the CODA library data from the paper")
    else:
        print("\nNote: K562 OOD test (synthetic sequences) not found")
        print("      Checked locations:")
        for path in possible_ood_paths:
            print(f"        - {path}")
        print("      This requires the CODA library data from the paper")
    
    # Print summary
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    
    for result in results:
        test_type = result['test_type'].upper()
        n_samples = result.get('n_samples', result.get('n_pairs', 'N/A'))
        print(f"\n{test_type} (k562):")
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
