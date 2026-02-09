#!/usr/bin/env python3
"""
Plot training curves comparing different subsample fractions.

Extracts training metrics from log files and plots:
- Training loss
- Validation loss  
- Validation Pearson R
"""

import sys
import re
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, str(Path(__file__).parent.parent))


def extract_training_metrics(log_file: Path):
    """Extract training metrics from a log file."""
    epochs = []
    train_loss = []
    val_loss = []
    val_pearson_r = []
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    # Pattern: Epoch X/80
    epoch_pattern = r'Epoch (\d+)/80'
    # Pattern: Train - Loss: X.XXXX, Pearson R: X.XXXX
    train_pattern = r'Train - Loss: ([\d.]+),'
    # Pattern: Val   - Loss: X.XXXX, Pearson R: X.XXXX
    val_pattern = r'Val   - Loss: ([\d.]+),.*?Pearson R: ([\d.]+)'
    
    # Find all epoch sections
    epoch_matches = list(re.finditer(epoch_pattern, content))
    train_matches = list(re.finditer(train_pattern, content))
    val_matches = list(re.finditer(val_pattern, content, re.DOTALL))
    
    # Match them up
    for i, epoch_match in enumerate(epoch_matches):
        epoch_num = int(epoch_match.group(1))
        
        if i < len(train_matches) and i < len(val_matches):
            train_loss_val = float(train_matches[i].group(1))
            val_loss_val = float(val_matches[i].group(1))
            val_pearson_val = float(val_matches[i].group(2))
            
            epochs.append(epoch_num)
            train_loss.append(train_loss_val)
            val_loss.append(val_loss_val)
            val_pearson_r.append(val_pearson_val)
    
    return {
        'epochs': np.array(epochs),
        'train_loss': np.array(train_loss),
        'val_loss': np.array(val_loss),
        'val_pearson_r': np.array(val_pearson_r)
    }


def plot_training_comparison(log_files_dict, output_file: Path):
    """Plot training curves for comparison."""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Training Curves Comparison: 4% vs 8% vs 16% Subsamples', 
                 fontsize=16, fontweight='bold')
    
    # Colors for each fraction
    colors = {
        '4%': '#2E86AB',
        '8%': '#A23B72',
        '16%': '#F18F01'
    }
    
    # Plot 1: Training Loss
    ax1 = axes[0]
    for fraction, logs in log_files_dict.items():
        for log_file, seed, metrics in logs:
            if metrics and len(metrics['epochs']) > 0:
                # Check if this is the unfinished 16% run
                is_unfinished = (fraction == "16%" and seed == "42")
                linestyle = '--' if is_unfinished else '-'
                label = f"{fraction} (seed {seed})" + (" [incomplete]" if is_unfinished else "")
                fmt = 'o--' if is_unfinished else 'o-'
                ax1.plot(metrics['epochs'], metrics['train_loss'], 
                        fmt, linewidth=2, markersize=4,
                        label=label, color=colors.get(fraction, 'gray'),
                        alpha=0.7)
    
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Training Loss', fontsize=12)
    ax1.set_title('Training Loss', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='best', fontsize=9)
    
    # Plot 2: Validation Loss
    ax2 = axes[1]
    for fraction, logs in log_files_dict.items():
        for log_file, seed, metrics in logs:
            if metrics and len(metrics['epochs']) > 0:
                # Check if this is the unfinished 16% run
                is_unfinished = (fraction == "16%" and seed == "42")
                linestyle = '--' if is_unfinished else '-'
                label = f"{fraction} (seed {seed})" + (" [incomplete]" if is_unfinished else "")
                fmt = 'o--' if is_unfinished else 'o-'
                ax2.plot(metrics['epochs'], metrics['val_loss'], 
                        fmt, linewidth=2, markersize=4,
                        label=label, color=colors.get(fraction, 'gray'),
                        alpha=0.7)
    
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Validation Loss', fontsize=12)
    ax2.set_title('Validation Loss', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(loc='best', fontsize=9)
    
    # Plot 3: Validation Pearson R
    ax3 = axes[2]
    for fraction, logs in log_files_dict.items():
        for log_file, seed, metrics in logs:
            if metrics and len(metrics['epochs']) > 0:
                # Check if this is the unfinished 16% run
                is_unfinished = (fraction == "16%" and seed == "42")
                linestyle = '--' if is_unfinished else '-'
                label = f"{fraction} (seed {seed})" + (" [incomplete]" if is_unfinished else "")
                fmt = 'o--' if is_unfinished else 'o-'
                ax3.plot(metrics['epochs'], metrics['val_pearson_r'], 
                        fmt, linewidth=2, markersize=4,
                        label=label, color=colors.get(fraction, 'gray'),
                        alpha=0.7)
    
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Validation Pearson R', fontsize=12)
    ax3.set_title('Validation Pearson R', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.legend(loc='best', fontsize=9)
    ax3.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved plot to: {output_file}")
    plt.close()


def main():
    logs_dir = Path("logs")
    output_dir = Path("plots")
    output_dir.mkdir(exist_ok=True)
    
    # Specific log files to compare (as requested by user)
    # 4%: both finished runs (seeds 225, 42)
    # 8%: both finished runs (seeds 300, 42)
    # 16%: finished run (seed 226) and unfinished run (seed 42)
    target_logs = {
        "4%": [
            "train_yeast_4pct_no_earlystop_seed225.log",
            "train_yeast_4pct_seed42_gpu1.log",
        ],
        "8%": [
            "train_yeast_8pct_no_earlystop_seed300.log",
            "train_yeast_8pct_seed42_gpu2.log",
        ],
        "16%": [
            "train_yeast_16pct_no_earlystop_seed226.log",
            "train_yeast_16pct_seed42_gpu3.log",  # Unfinished (54 epochs)
        ],
    }
    
    log_files_dict = defaultdict(list)
    
    for label, log_names in target_logs.items():
        for log_name in log_names:
            log_file = logs_dir / log_name
            if not log_file.exists():
                print(f"Warning: {log_file} not found")
                continue
            
            # Extract seed
            seed_match = re.search(r'seed(\d+)', log_name)
            seed = seed_match.group(1) if seed_match else "unknown"
            
            # Extract metrics
            try:
                metrics = extract_training_metrics(log_file)
                if len(metrics['epochs']) > 0:
                    log_files_dict[label].append((log_name, seed, metrics))
                    print(f"Found {label} log: {log_name} (seed {seed}) - {len(metrics['epochs'])} epochs")
                else:
                    print(f"Warning: {log_name} has no metrics")
            except Exception as e:
                print(f"Error extracting metrics from {log_name}: {e}")
    
    # Plot comparison
    if log_files_dict:
        output_file = output_dir / "training_curves_comparison.png"
        plot_training_comparison(log_files_dict, output_file)
    else:
        print("No log files found!")


if __name__ == '__main__':
    main()
