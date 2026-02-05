"""
Active Learning Simulation Experiment.

Simulates active learning loops with different acquisition strategies,
comparing them against random (passive) baseline.

Usage:
    python experiments/02_al_simulation.py --config experiments/configs/al_simulation.yaml
    python experiments/02_al_simulation.py --config experiments/configs/al_simulation.yaml --strategy uncertainty
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
from torch.utils.data import DataLoader

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
from al_genomics import (
    DataPool,
    RandomAcquisition,
    UncertaintyAcquisition,
    DiversityAcquisition,
    HybridAcquisition
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


def create_acquisition_strategy(strategy_name: str, random_seed: int, **kwargs):
    """Create acquisition strategy instance."""
    if strategy_name.lower() == 'random':
        return RandomAcquisition(random_seed=random_seed)
    elif strategy_name.lower() == 'uncertainty':
        return UncertaintyAcquisition(
            random_seed=random_seed,
            mc_dropout_samples=kwargs.get('mc_dropout_samples', 10),
            batch_size=kwargs.get('batch_size', 256)
        )
    elif strategy_name.lower() == 'diversity':
        return DiversityAcquisition(
            random_seed=random_seed,
            batch_size=kwargs.get('batch_size', 256)
        )
    elif strategy_name.lower() == 'hybrid':
        return HybridAcquisition(
            random_seed=random_seed,
            uncertainty_weight=kwargs.get('uncertainty_weight', 0.5),
            diversity_weight=kwargs.get('diversity_weight', 0.5),
            batch_size=kwargs.get('batch_size', 256),
            mc_dropout_samples=kwargs.get('mc_dropout_samples', 10)
        )
    else:
        raise ValueError(f"Unknown acquisition strategy: {strategy_name}")


def run_al_simulation(config: dict, strategy_override: str = None):
    """
    Run active learning simulation.
    
    Args:
        config: Configuration dictionary
        strategy_override: Optional strategy name to override config
    """
    print("="*80)
    print("ACTIVE LEARNING SIMULATION")
    print("="*80)
    
    # Set random seed
    seed = config['experiment']['random_seed']
    set_seed(seed)
    
    # Override strategy if specified
    if strategy_override is not None:
        config['active_learning']['strategy'] = strategy_override
        print(f"Strategy override: {strategy_override}")
    
    strategy_name = config['active_learning']['strategy']
    dataset_name = config['data']['dataset_name']
    
    print(f"Dataset: {dataset_name}")
    print(f"Strategy: {strategy_name}")
    print(f"Random seed: {seed}")
    print(f"Initial samples: {config['active_learning']['initial_samples']}")
    print(f"Batch size: {config['active_learning']['batch_size']}")
    print(f"Number of rounds: {config['active_learning']['num_rounds']}")
    
    # Setup directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"al_{strategy_name}_{dataset_name}_{timestamp}"
    
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
    
    # Create validation and test loaders (full sets)
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
    
    # Initialize data pool
    print("\nInitializing data pool...")
    
    # Create initial labeled set
    initial_size = config['active_learning']['initial_samples']
    initial_indices = np.random.choice(
        len(train_dataset_full),
        size=initial_size,
        replace=False
    )
    
    pool = DataPool(
        full_dataset=train_dataset_full,
        initial_labeled_indices=initial_indices,
        random_seed=seed
    )
    
    print(f"Initial pool state: {pool}")
    
    # Create acquisition strategy
    acquisition_strategy = create_acquisition_strategy(
        strategy_name,
        random_seed=seed,
        mc_dropout_samples=config['active_learning'].get('mc_dropout_samples', 10),
        batch_size=config['data']['batch_size'],
        uncertainty_weight=config['active_learning'].get('uncertainty_weight', 0.5),
        diversity_weight=config['active_learning'].get('diversity_weight', 0.5)
    )
    
    print(f"Acquisition strategy: {acquisition_strategy.get_name()}")
    
    # Results storage
    all_results = {
        'config': config,
        'strategy': strategy_name,
        'rounds': []
    }
    
    # Active learning loop
    num_rounds = config['active_learning']['num_rounds']
    batch_size = config['active_learning']['batch_size']
    
    for round_num in range(num_rounds + 1):  # +1 to include initial round
        print("\n" + "="*80)
        if round_num == 0:
            print(f"INITIAL ROUND (Random {initial_size} samples)")
        else:
            print(f"ACTIVE LEARNING ROUND {round_num}/{num_rounds}")
        print("="*80)
        
        # Get current labeled dataset
        labeled_dataset = pool.get_labeled_dataset()
        pool_dataset = pool.get_pool_dataset()
        
        print(f"Labeled samples: {len(labeled_dataset)}")
        print(f"Pool samples: {len(pool_dataset)}")
        
        # Create data loader for training
        train_loader = DataLoader(
            labeled_dataset,
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
        
        # Create checkpoint directory for this round
        round_checkpoint_dir = checkpoint_dir / f"round_{round_num:03d}"
        round_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
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
            checkpoint_dir=round_checkpoint_dir,
            use_reverse_complement=config['training']['use_reverse_complement'],
            early_stopping_patience=config['training']['early_stopping_patience'],
            metric_for_best=config['training']['metric_for_best']
        )
        
        training_time = time.time() - start_time
        
        # Evaluate on test set
        print("\nEvaluating on test set...")
        
        # Load best model
        best_model_path = round_checkpoint_dir / "best_model.pt"
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
        
        # Store round results
        round_result = {
            'round': round_num,
            'num_labeled': len(labeled_dataset),
            'num_pool': len(pool_dataset),
            'training_time_seconds': training_time,
            'history': history,
            'test_metrics': test_metrics,
            'pool_stats': pool.get_statistics()
        }
        all_results['rounds'].append(round_result)
        
        # Save intermediate results
        with open(results_dir / "results.json", 'w') as f:
            json.dump(all_results, f, indent=2)
        
        # Acquisition phase (if not last round and pool not exhausted)
        if round_num < num_rounds and not pool.is_exhausted():
            print(f"\nAcquiring {batch_size} new samples using {strategy_name} strategy...")
            
            acquisition_start = time.time()
            
            # Select batch using acquisition strategy
            pool_dataset = pool.get_pool_dataset()
            
            # Ensure we don't try to acquire more than available
            n_to_acquire = min(batch_size, len(pool_dataset))
            
            selected_pool_indices = acquisition_strategy.select_batch(
                model=model,
                pool_dataset=pool_dataset,
                n_samples=n_to_acquire,
                device=device
            )
            
            # Convert pool indices to full dataset indices
            pool_indices = pool.get_pool_indices()
            selected_full_indices = pool_indices[selected_pool_indices]
            
            # Add to labeled set
            pool.add_to_labeled(selected_full_indices, round_num=round_num + 1)
            
            acquisition_time = time.time() - acquisition_start
            
            print(f"Acquired {len(selected_full_indices)} samples in {acquisition_time:.2f}s")
            print(f"New pool state: {pool}")
            
            # Store acquisition info
            round_result['acquisition'] = {
                'num_acquired': len(selected_full_indices),
                'acquisition_time_seconds': acquisition_time,
                'selected_indices': selected_full_indices.tolist()
            }
    
    # Save final results
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)
    
    with open(results_dir / "results.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to: {results_dir}")
    
    # Print summary
    print("\nSummary:")
    print(f"{'Round':<8} {'Labeled':<10} {'Train Time':<12} {'Test Pearson R':<15} {'Test Spearman R'}")
    print("-" * 70)
    for result in all_results['rounds']:
        print(f"{result['round']:<8} "
              f"{result['num_labeled']:<10} "
              f"{result['training_time_seconds']/60:<12.1f} "
              f"{result['test_metrics']['pearson_r']:<15.4f} "
              f"{result['test_metrics']['spearman_r']:<15.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Run active learning simulation"
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to configuration YAML file'
    )
    parser.add_argument(
        '--strategy',
        type=str,
        default=None,
        choices=['random', 'uncertainty', 'diversity', 'hybrid'],
        help='Override acquisition strategy specified in config'
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Run experiment
    try:
        run_al_simulation(config, strategy_override=args.strategy)
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
