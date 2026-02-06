#!/usr/bin/env python
"""
Pre-compute HashFrag splits for K562 dataset.

This script creates homology-aware train/validation/test splits using HashFrag
and caches them for future use. Running this once allows all subsequent experiments
to use the cached splits without re-running the slow HashFrag computation.

Usage:
    # Create splits with defaults
    python scripts/create_hashfrag_splits.py
    
    # Create with custom threshold
    python scripts/create_hashfrag_splits.py --threshold 70
    
    # Force recompute even if cache exists
    python scripts/create_hashfrag_splits.py --force
    
    # Specify custom data and output directories
    python scripts/create_hashfrag_splits.py --data-dir ./data/k562 --output-dir ./my_splits
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.k562 import K562Dataset

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='Pre-compute HashFrag splits for K562 dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='./data/k562',
        help='Directory containing K562 data (DATA-Table_S2__MPRA_dataset.txt)'
    )
    
    parser.add_argument(
        '--threshold',
        type=int,
        default=60,
        help='Smith-Waterman score threshold for defining homology'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Directory to save splits (default: {data-dir}/hashfrag_splits)'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force recompute even if cached splits exist'
    )
    
    args = parser.parse_args()
    
    # Set output directory
    data_dir = Path(args.data_dir)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = data_dir / 'hashfrag_splits'
    
    # Check if splits already exist
    cache_files = [
        output_dir / 'train_indices.npy',
        output_dir / 'pool_indices.npy',
        output_dir / 'val_indices.npy',
        output_dir / 'test_indices.npy'
    ]
    
    if all(f.exists() for f in cache_files) and not args.force:
        logger.info("=" * 70)
        logger.info("HashFrag splits already exist!")
        logger.info(f"Location: {output_dir}")
        logger.info("=" * 70)
        logger.info("")
        logger.info("To recompute, use --force flag")
        logger.info("To use existing splits, no action needed - experiments will use them automatically")
        return
    
    if args.force and any(f.exists() for f in cache_files):
        logger.info("Forcing recomputation of splits...")
    
    # Print configuration
    logger.info("=" * 70)
    logger.info("HashFrag Split Creation")
    logger.info("=" * 70)
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Homology threshold: {args.threshold}")
    logger.info(f"Split ratio: 80% train+pool / 10% val / 10% test")
    logger.info(f"  (train pool will be further split: 100K train + rest pool)")
    logger.info("=" * 70)
    logger.info("")
    
    # Warn about computation time
    logger.info("⚠️  WARNING: This will take several hours!")
    logger.info("HashFrag must:")
    logger.info("  1. Run BLAST on ~367K sequences")
    logger.info("  2. Compute Smith-Waterman scores for candidate pairs")
    logger.info("  3. Cluster sequences by homology")
    logger.info("  4. Create orthogonal splits")
    logger.info("")
    logger.info("Consider running on HPC with: sbatch scripts/slurm/create_hashfrag_splits.sh")
    logger.info("")
    
    # Start timer
    start_time = time.time()
    
    try:
        # Create dataset - this will trigger HashFrag split creation
        logger.info("Loading K562 dataset and creating HashFrag splits...")
        logger.info("")
        
        dataset = K562Dataset(
            data_path=str(data_dir),
            split='train',  # We load train split, but all splits are created
            use_hashfrag=True,
            hashfrag_threshold=args.threshold,
            hashfrag_cache_dir=str(output_dir)
        )
        
        # Print statistics
        elapsed_time = time.time() - start_time
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = int(elapsed_time % 60)
        
        logger.info("")
        logger.info("=" * 70)
        logger.info("✓ HashFrag splits created successfully!")
        logger.info("=" * 70)
        logger.info(f"Time taken: {hours}h {minutes}m {seconds}s")
        logger.info(f"Splits saved to: {output_dir}")
        logger.info("")
        
        # Load and print split statistics
        import numpy as np
        
        total_seqs = 0
        logger.info("Split Statistics:")
        logger.info("-" * 70)
        
        for split_name in ['train', 'pool', 'val', 'test']:
            indices_file = output_dir / f'{split_name}_indices.npy'
            if indices_file.exists():
                indices = np.load(indices_file)
                n_seqs = len(indices)
                total_seqs += n_seqs
                logger.info(f"  {split_name.capitalize():6s}: {n_seqs:7,} sequences")
        
        logger.info("-" * 70)
        logger.info(f"  Total:  {total_seqs:7,} sequences")
        logger.info("")
        
        # Calculate percentages
        if total_seqs > 0:
            logger.info("Split Percentages:")
            logger.info("-" * 70)
            for split_name in ['train', 'pool', 'val', 'test']:
                indices_file = output_dir / f'{split_name}_indices.npy'
                if indices_file.exists():
                    indices = np.load(indices_file)
                    pct = 100 * len(indices) / total_seqs
                    logger.info(f"  {split_name.capitalize():6s}: {pct:5.1f}%")
            logger.info("")
        
        logger.info("Next steps:")
        logger.info("  1. Run baseline experiments: python experiments/01_baseline_subsets.py")
        logger.info("  2. Experiments will automatically use these splits")
        logger.info("")
        
    except Exception as e:
        logger.error(f"Failed to create HashFrag splits: {e}")
        logger.error("")
        logger.error("Common issues:")
        logger.error("  1. BLAST+ not installed - see BLAST_INSTALL.md")
        logger.error("  2. HashFrag not in PATH - run: ./scripts/setup_hashfrag.sh")
        logger.error("  3. Insufficient memory - try running on HPC with more RAM")
        logger.error("")
        sys.exit(1)


if __name__ == '__main__':
    main()
