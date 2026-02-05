"""
Baseline subset experiment: Train models on random subsets of different sizes.

This establishes baseline data efficiency curves WITHOUT active learning,
showing how model performance scales with random data sampling.

Usage:
    python experiments/01_baseline_subsets.py --config experiments/configs/baseline.yaml
    python experiments/01_baseline_subsets.py --config experiments/configs/baseline.yaml --dataset yeast
"""

import argparse
import os
import sys
from pathlib import Path
import yaml
import json
import time
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.k562 import K562Dataset
from data.yeast import YeastDataset
from models.dream_rnn import create_dream_rnn
from models.training import (
    train_model,
    create_optimizer_and_scheduler,
    evaluate
)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if seed is not None:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_dataset(dataset_name: str, data_path: str, split: str):
    """Create dataset instance based on name."""
    if dataset_name.lower() == 'k562':
        return K562Dataset(data_path=data_path, split=split)
    elif dataset_name.lower() == 'yeast':
        return YeastDataset(data_path=data_path, split=split)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def create_subset_indices(dataset_size: int, fraction: float, seed: int) -> np.ndarray:
    """Create random subset indices."""
    np.random.seed(seed)
    num_samples = int(dataset_size * fraction)
    indices = np.random.choice(dataset_size, size=num_samples, replace=False)
    return indices


def run_experiment(
    config: dict, 
    dataset_override: str = None,
    seed_override: int = None,
    fractions_override: list = None
):
    """
    Run the baseline subset experiment.
    
    Args:
        config: Configuration dictionary
        dataset_override: Optional dataset name to override config
        seed_override: Optional random seed to override config
        fractions_override: Optional list of fractions to override config
    """
    print("="*80)
    print("BASELINE SUBSET EXPERIMENT")
    print("="*80)
    
    # Override seed if specified
    if seed_override is not None:
        config['experiment']['random_seed'] = seed_override
        print(f"Seed override: {seed_override}")
    
    # Set random seed
    seed = config['experiment']['random_seed']
    set_seed(seed)
    
    # Override dataset if specified
    if dataset_override is not None:
        config['data']['dataset_name'] = dataset_override
        print(f"Dataset override: {dataset_override}")
    
    # Override fractions if specified
    if fractions_override is not None:
        config['data']['subset_fractions'] = fractions_override
        print(f"Fractions override: {fractions_override}")
    
    dataset_name = config['data']['dataset_name']
    print(f"Dataset: {dataset_name}")
    print(f"Random seed: {seed}")
    print(f"Subset fractions: {config['data']['subset_fractions']}")
    
    # Setup directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{config['experiment']['name']}_{dataset_name}_{timestamp}"
    
    checkpoint_dir = Path(config['output']['checkpoint_dir']) / experiment_name
    results_dir = Path(config['output']['results_dir']) / experiment_name
    log_dir = Path(config['output']['log_dir']) / experiment_name
    
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {results_dir}")
    
    # Save config
    with open(results_dir / "config.yaml", 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Load datasets
    print("\nLoading datasets...")
    data_path = config['data']['data_path']
    
    try:
        train_dataset_full = create_dataset(dataset_name, data_path, 'train')
        val_dataset = create_dataset(dataset_name, data_path, 'val')
        test_dataset = create_dataset(dataset_name, data_path, 'test')
        
        print(f"Full training set: {len(train_dataset_full)} samples")
        print(f"Validation set: {len(val_dataset)} samples")
        print(f"Test set: {len(test_dataset)} samples")
        print(f"Sequence length: {train_dataset_full.get_sequence_length()}")
        print(f"Input channels: {train_dataset_full.get_num_channels()}")
    except Exception as e:
        print(f"\nError loading dataset: {e}")
        print("\nNote: Make sure you've downloaded the data first:")
        print(f"  python scripts/download_data.py --dataset {dataset_name}")
        return
    
    # Setup device
    if torch.cuda.is_available() and config['hardware']['device'] == 'cuda':
        device = torch.device(f"cuda:{config['hardware']['device_id']}")
    else:
        device = torch.device('cpu')
    print(f"\nDevice: {device}")
    
    # Create data loaders for validation and test (full sets)
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory']
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory']
    )
    
    # Results storage
    all_results = {
        'config': config,
        'subset_results': []
    }
    
    # Train on each subset fraction
    subset_fractions = config['data']['subset_fractions']
    
    for fraction in subset_fractions:
        print("\n" + "="*80)
        print(f"TRAINING WITH {fraction*100:.1f}% OF DATA")
        print("="*80)
        
        # Create subset
        subset_indices = create_subset_indices(
            len(train_dataset_full),
            fraction,
            seed=seed
        )
        train_subset = Subset(train_dataset_full, subset_indices)
        
        print(f"Training samples: {len(train_subset)}")
        
        # Create data loader for this subset
        train_loader = DataLoader(
            train_subset,
            batch_size=config['data']['batch_size'],
            shuffle=True,
            num_workers=config['data']['num_workers'],
            pin_memory=config['data']['pin_memory']
        )
        
        # Create model
        model = create_dream_rnn(
            input_channels=train_dataset_full.get_num_channels(),
            sequence_length=train_dataset_full.get_sequence_length(),
            hidden_dim=config['model']['hidden_dim'],
            cnn_filters=config['model']['cnn_filters'],
            dropout_cnn=config['model']['dropout_cnn'],
            dropout_lstm=config['model']['dropout_lstm']
        )
        model = model.to(device)
        
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Create optimizer and scheduler
        optimizer, scheduler = create_optimizer_and_scheduler(
            model=model,
            train_loader=train_loader,
            num_epochs=config['training']['num_epochs'],
            lr=config['training']['lr'],
            lr_lstm=config['training']['lr_lstm'],
            weight_decay=config['training']['weight_decay'],
            pct_start=config['training']['pct_start']
        )
        
        # Create loss function
        criterion = nn.MSELoss()
        
        # Create checkpoint directory for this subset
        subset_checkpoint_dir = checkpoint_dir / f"fraction_{fraction:.3f}"
        subset_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Train model
        start_time = time.time()
        
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            num_epochs=config['training']['num_epochs'],
            device=device,
            scheduler=scheduler,
            checkpoint_dir=subset_checkpoint_dir,
            use_reverse_complement=config['training']['use_reverse_complement'],
            early_stopping_patience=config['training']['early_stopping_patience'],
            metric_for_best=config['training']['metric_for_best']
        )
        
        training_time = time.time() - start_time
        
        # Evaluate on test set
        print("\nEvaluating on test set...")
        
        # Load best model
        best_model_path = subset_checkpoint_dir / "best_model.pt"
        if best_model_path.exists():
            model.load_checkpoint(str(best_model_path))
        
        test_metrics = evaluate(
            model=model,
            dataloader=test_loader,
            criterion=criterion,
            device=device,
            use_reverse_complement=config['training']['use_reverse_complement']
        )
        
        print(f"Test metrics:")
        print(f"  Pearson R: {test_metrics['pearson_r']:.4f}")
        print(f"  Spearman R: {test_metrics['spearman_r']:.4f}")
        print(f"  MSE: {test_metrics['mse']:.4f}")
        
        # Store results
        subset_result = {
            'fraction': fraction,
            'num_samples': len(train_subset),
            'training_time_seconds': training_time,
            'history': history,
            'test_metrics': test_metrics
        }
        all_results['subset_results'].append(subset_result)
        
        # Save intermediate results
        with open(results_dir / "results.json", 'w') as f:
            json.dump(all_results, f, indent=2)
    
    # Save final results
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)
    
    with open(results_dir / "results.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to: {results_dir}")
    
    # Print summary
    print("\nSummary:")
    print(f"{'Fraction':<10} {'Samples':<10} {'Train Time':<12} {'Test Pearson R':<15} {'Test Spearman R'}")
    print("-" * 70)
    for result in all_results['subset_results']:
        print(f"{result['fraction']:<10.3f} "
              f"{result['num_samples']:<10} "
              f"{result['training_time_seconds']/60:<12.1f} "
              f"{result['test_metrics']['pearson_r']:<15.4f} "
              f"{result['test_metrics']['spearman_r']:<15.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Run baseline subset experiment"
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to configuration YAML file'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default=None,
        choices=['k562', 'yeast'],
        help='Override dataset specified in config'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Override random seed specified in config'
    )
    parser.add_argument(
        '--fractions',
        type=float,
        nargs='+',
        default=None,
        help='Override subset fractions (e.g., --fractions 0.05 0.1 0.25)'
    )
    parser.add_argument(
        '--gpu',
        type=int,
        default=None,
        help='GPU device ID to use'
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override GPU if specified
    if args.gpu is not None:
        config['hardware']['device_id'] = args.gpu
    
    # Run experiment
    try:
        run_experiment(
            config, 
            dataset_override=args.dataset,
            seed_override=args.seed,
            fractions_override=args.fractions
        )
    except KeyboardInterrupt:
        print("\n\nExperiment interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nExperiment failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
