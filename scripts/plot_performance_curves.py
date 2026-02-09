#!/usr/bin/env python3
"""
Plot performance curves for baseline experiments.

Creates separate plots for:
- Yeast vs K562 datasets
- In-distribution, SNV, and OOD test sets
- X-axis: Dataset downsample fraction (0-100%)
- Y-axis: Pearson correlation (0-100%)
"""

import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

sys.path.insert(0, str(Path(__file__).parent.parent))


def find_evaluation_results(results_dir: Path, dataset: str) -> dict:
    """
    Find all evaluation results for a dataset.
    
    Returns:
        Dict mapping (fraction, seed) -> list of evaluation results
    """
    results = defaultdict(list)
    
    # Look for evaluation JSON files in results/evaluations/
    eval_dir = results_dir / 'evaluations'
    if eval_dir.exists():
        pattern = f"{dataset}_*.json"
        for eval_file in eval_dir.glob(pattern):
            try:
                # Parse filename: {dataset}_{fraction}_eval.json or {dataset}_{fraction}_seed{seed}_eval.json
                name = eval_file.stem.replace(f"{dataset}_", "").replace("_eval", "")
                if "_seed" in name:
                    fraction_str, seed_str = name.split("_seed", 1)
                    fraction = float(fraction_str)
                    try:
                        # Extract numeric part from seed string (handle cases like "21X" -> None)
                        seed = int(''.join(filter(str.isdigit, seed_str)))
                    except (ValueError, TypeError):
                        seed = None
                else:
                    fraction = float(name)
                    seed = None
                
                with open(eval_file, 'r') as f:
                    eval_data = json.load(f)
                    # eval_data is a list, so extend with it
                    if isinstance(eval_data, list):
                        results[(fraction, seed)].extend(eval_data)
                    else:
                        results[(fraction, seed)].append(eval_data)
            except (ValueError, json.JSONDecodeError) as e:
                print(f"Warning: Could not parse {eval_file}: {e}")
                continue
    
    # Also check training results directories for evaluation outputs
    for result_dir in results_dir.glob(f"baseline_{dataset}_*"):
        if not result_dir.is_dir():
            continue
        
        # Look for evaluation files in subdirectories
        for eval_file in result_dir.rglob("*eval*.json"):
            try:
                with open(eval_file, 'r') as f:
                    eval_data = json.load(f)
                    
                # Try to extract fraction from parent directory or filename
                # Check if there's a fraction_* directory
                fraction = None
                seed = None
                
                # Check parent directories for fraction
                for parent in eval_file.parents:
                    if parent.name.startswith("fraction_"):
                        fraction = float(parent.name.replace("fraction_", ""))
                        break
                
                # Try to extract seed from result directory name
                dir_name = result_dir.name
                if "seed" in dir_name:
                    try:
                        seed_part = dir_name.split("seed")[1].split("_")[0]
                        seed = int(seed_part)
                    except (ValueError, IndexError):
                        pass
                
                if fraction is not None:
                    results[(fraction, seed)].extend(eval_data)
            except (ValueError, json.JSONDecodeError) as e:
                continue
    
    return results


def find_training_results(results_dir: Path, dataset: str) -> dict:
    """
    Extract performance metrics from training results.
    
    Returns:
        Dict mapping fraction -> list of (seed, pearson_r, spearman_r) tuples
    """
    results = defaultdict(list)
    
    for result_dir in results_dir.glob(f"baseline_{dataset}_*"):
        if not result_dir.is_dir():
            continue
        
        results_file = result_dir / "results.json"
        if not results_file.exists():
            continue
        
        try:
            with open(results_file, 'r') as f:
                data = json.load(f)
            
            # Extract seed from directory name
            seed = None
            dir_name = result_dir.name
            if "seed" in dir_name:
                try:
                    seed_part = dir_name.split("seed")[1].split("_")[0]
                    seed = int(seed_part)
                except (ValueError, IndexError):
                    pass
            
            # Extract subset results
            if 'subset_results' in data:
                for subset_result in data['subset_results']:
                    fraction = subset_result.get('fraction')
                    if fraction is not None:
                        test_metrics = subset_result.get('test_metrics', {})
                        # Note: training results use validation metrics, not test set metrics
                        # We'll prefer evaluation results when available
                        pearson_r = test_metrics.get('pearson_r', 0.0)
                        spearman_r = test_metrics.get('spearman_r', 0.0)
                        results[fraction].append({
                            'seed': seed,
                            'pearson_r': pearson_r,
                            'spearman_r': spearman_r,
                            'source': 'training'
                        })
        except (json.JSONDecodeError, KeyError) as e:
            continue
    
    return results


def organize_results(eval_results: dict, dataset: str) -> dict:
    """
    Organize evaluation results by test type and fraction.
    
    Returns:
        Dict: {test_type: {fraction: [pearson_r values]}}
    """
    organized = defaultdict(lambda: defaultdict(list))
    
    for (fraction, seed), eval_data_list in eval_results.items():
        for eval_data in eval_data_list:
            if isinstance(eval_data, dict):
                test_type = eval_data.get('test_type')
                if test_type and eval_data.get('dataset') == dataset:
                    pearson_r = eval_data.get('pearson_r', 0.0)
                    organized[test_type][fraction].append(pearson_r)
    
    return organized


def plot_curves(organized_results: dict, dataset: str, output_dir: Path):
    """Plot performance curves for each test type."""
    
    test_types = {
        'in_distribution': 'In-Distribution',
        'snv': 'SNV',
        'ood': 'Out-of-Distribution'
    }
    
    # Create figure with subplots for each test type
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'{dataset.upper()} Dataset Performance Curves', fontsize=16, fontweight='bold')
    
    for idx, (test_type, test_label) in enumerate(test_types.items()):
        ax = axes[idx]
        
        if test_type not in organized_results:
            ax.text(0.5, 0.5, f'No {test_label} data available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_xlabel('Training Data Fraction')
            ax.set_ylabel('Pearson Correlation')
            ax.set_title(test_label)
            continue
        
        # Collect data points with standard deviations
        fractions = []
        pearson_values = []
        pearson_stds = []
        
        for fraction, values in sorted(organized_results[test_type].items()):
            if values:
                # Average multiple runs at same fraction
                avg_pearson = np.mean(values)
                std_pearson = np.std(values) if len(values) > 1 else 0.0
                # Keep fraction as decimal (0.005, 0.02, etc.) for log scale
                # We'll format labels as percentages later
                fractions.append(fraction)  # Keep as fraction (0.005, 0.02, etc.) for log scale
                pearson_values.append(avg_pearson)  # Keep as 0-1 for y-axis
                pearson_stds.append(std_pearson)  # Standard deviation
        
        if fractions:
            # Sort by fraction
            sorted_data = sorted(zip(fractions, pearson_values, pearson_stds))
            fractions, pearson_values, pearson_stds = zip(*sorted_data)
            
            # Convert to lists for errorbar
            fractions = list(fractions)
            pearson_values = list(pearson_values)
            pearson_stds = list(pearson_stds)
            
            # Plot curve with shaded region (no error bars - shading shows std dev)
            ax.plot(fractions, pearson_values, 
                   'o-', linewidth=2, markersize=8, 
                   label='Pearson R', color='#2E86AB', 
                   alpha=0.8)
            ax.fill_between(fractions, 
                           [v - s for v, s in zip(pearson_values, pearson_stds)],
                           [v + s for v, s in zip(pearson_values, pearson_stds)],
                           alpha=0.15, color='#2E86AB')
            
            # Add grid
            ax.grid(True, alpha=0.3, linestyle='--')
            
            # Set log scale for x-axis (using fraction values, not percentages)
            ax.set_xscale('log')
            
            # Set axis limits (fractions are already in 0.005-1.0 range)
            # For log scale, we need to avoid 0, so set a minimum
            min_frac = min(fractions) if fractions else 0.001
            max_frac = max(fractions) if fractions else 1.0
            ax.set_xlim(max(0.001, min_frac * 0.5), min(1.2, max_frac * 1.2))
            ax.set_ylim(0, 1)
            
            # Format axes
            ax.set_xlabel('Training Data Fraction', fontsize=12)
            ax.set_ylabel('Pearson Correlation', fontsize=12)
            ax.set_title(test_label, fontsize=14, fontweight='bold')
            
            # Set log scale tick locations and labels
            # Format as percentage for readability
            from matplotlib.ticker import LogLocator, FuncFormatter
            ax.xaxis.set_major_locator(LogLocator(base=10, numticks=10))
            # Format as percentage
            def log_formatter(x, pos):
                if x < 0.01:
                    return f'{x*100:.2f}%'
                elif x < 0.1:
                    return f'{x*100:.1f}%'
                else:
                    return f'{x*100:.0f}%'
            ax.xaxis.set_major_formatter(FuncFormatter(log_formatter))
            
            # Point labels removed to reduce clutter - values can be read from the plot
        else:
            ax.text(0.5, 0.5, 'No data available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_xlabel('Training Data Fraction')
            ax.set_ylabel('Pearson Correlation')
            ax.set_title(test_label)
    
    plt.tight_layout()
    
    # Save figure
    output_file = output_dir / f'{dataset}_performance_curves.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved plot to: {output_file}")
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Plot performance curves for baseline experiments"
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default='./results',
        help='Directory containing results (default: ./results)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./plots',
        help='Directory to save plots (default: ./plots)'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['yeast', 'k562', 'all'],
        default='all',
        help='Dataset to plot (default: all)'
    )
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    datasets = ['yeast', 'k562'] if args.dataset == 'all' else [args.dataset]
    
    for dataset in datasets:
        print(f"\n{'='*80}")
        print(f"Processing {dataset.upper()} dataset")
        print(f"{'='*80}")
        
        # Find evaluation results
        eval_results = find_evaluation_results(results_dir, dataset)
        print(f"Found {len(eval_results)} evaluation result sets")
        
        # Organize by test type
        organized = organize_results(eval_results, dataset)
        
        # Print summary
        for test_type, data in organized.items():
            print(f"  {test_type}: {len(data)} fractions")
            for fraction, values in sorted(data.items()):
                avg = np.mean(values)
                print(f"    {fraction*100:5.1f}%: {avg:.4f} (n={len(values)})")
        
        # Plot curves
        if organized:
            plot_curves(organized, dataset, output_dir)
        else:
            print(f"Warning: No evaluation results found for {dataset}")
    
    print(f"\n{'='*80}")
    print("Plotting complete!")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
