#!/usr/bin/env python3
"""
Launch baseline experiments across multiple GPUs.

This script distributes baseline subset experiments across available GPUs,
running multiple replicates with different random seeds.
"""

import subprocess
import time
import argparse
from pathlib import Path
from typing import List, Tuple
import sys

# Experiment configurations
K562_FRACTIONS = [0.05, 0.1, 0.25, 0.5]
YEAST_FRACTIONS = [0.01, 0.02, 0.04, 0.08]

# Default seeds for replicates
DEFAULT_SEEDS = [42, 123, 456]


def create_job(
    dataset: str,
    fraction: float,
    seed: int,
    gpu_id: int,
    config_path: str
) -> Tuple[List[str], str]:
    """
    Create a job command and identifier.
    
    Args:
        dataset: Dataset name (k562 or yeast)
        fraction: Data fraction to use
        seed: Random seed
        gpu_id: GPU device ID
        config_path: Path to config file
        
    Returns:
        Tuple of (command_list, job_id)
    """
    job_id = f"{dataset}_frac{fraction:.3f}_seed{seed}_gpu{gpu_id}"
    
    cmd = [
        "python", "-u",  # -u for unbuffered output
        "experiments/01_baseline_subsets.py",
        "--config", config_path,
        "--dataset", dataset,
        "--seed", str(seed),
        "--fractions", str(fraction),
        "--gpu", str(gpu_id)
    ]
    
    return cmd, job_id


def launch_job_background(cmd: List[str], job_id: str, log_dir: Path) -> subprocess.Popen:
    """
    Launch a job in the background with logging.
    
    Args:
        cmd: Command to run
        job_id: Job identifier for logging
        log_dir: Directory to save logs
        
    Returns:
        Subprocess handle
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f"{job_id}.log"
    
    print(f"Launching: {job_id}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"  Log: {log_file}")
    
    with open(log_file, 'w') as f:
        f.write(f"Command: {' '.join(cmd)}\n")
        f.write(f"Job ID: {job_id}\n")
        f.write("="*80 + "\n\n")
        f.flush()
        
        process = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            cwd=Path(__file__).parent.parent,  # Run from project root
            text=True
        )
    
    return process


def main():
    parser = argparse.ArgumentParser(
        description="Launch baseline experiments across multiple GPUs"
    )
    parser.add_argument(
        '--gpus',
        type=int,
        nargs='+',
        default=list(range(8)),
        help='GPU IDs to use (default: 0-7)'
    )
    parser.add_argument(
        '--datasets',
        type=str,
        nargs='+',
        choices=['k562', 'yeast', 'all'],
        default=['all'],
        help='Datasets to run (default: all)'
    )
    parser.add_argument(
        '--seeds',
        type=int,
        nargs='+',
        default=DEFAULT_SEEDS,
        help=f'Random seeds for replicates (default: {DEFAULT_SEEDS})'
    )
    parser.add_argument(
        '--log-dir',
        type=str,
        default='./logs/baseline_runs',
        help='Directory for job logs'
    )
    parser.add_argument(
        '--sequential',
        action='store_true',
        help='Run jobs sequentially instead of parallel'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print jobs without running them'
    )
    
    args = parser.parse_args()
    
    # Determine which datasets to run
    if 'all' in args.datasets:
        datasets_to_run = ['k562', 'yeast']
    else:
        datasets_to_run = args.datasets
    
    log_dir = Path(args.log_dir)
    
    # Create job list
    jobs = []
    
    for dataset in datasets_to_run:
        if dataset == 'k562':
            fractions = K562_FRACTIONS
            config_path = "experiments/configs/baseline_k562.yaml"
        else:  # yeast
            fractions = YEAST_FRACTIONS
            config_path = "experiments/configs/baseline_yeast.yaml"
        
        for fraction in fractions:
            for seed in args.seeds:
                jobs.append((dataset, fraction, seed, config_path))
    
    print("="*80)
    print("BASELINE EXPERIMENT LAUNCHER")
    print("="*80)
    print(f"Datasets: {', '.join(datasets_to_run)}")
    print(f"Total jobs: {len(jobs)}")
    print(f"Seeds per fraction: {len(args.seeds)}")
    print(f"Available GPUs: {args.gpus}")
    print(f"Log directory: {log_dir}")
    print("="*80)
    print()
    
    if args.dry_run:
        print("DRY RUN - Jobs that would be launched:")
        for i, (dataset, fraction, seed, config_path) in enumerate(jobs):
            gpu_id = args.gpus[i % len(args.gpus)]
            cmd, job_id = create_job(dataset, fraction, seed, gpu_id, config_path)
            print(f"\n{i+1}. {job_id}")
            print(f"   GPU: {gpu_id}")
            print(f"   Command: {' '.join(cmd)}")
        return
    
    if args.sequential:
        # Run jobs sequentially
        print("Running jobs SEQUENTIALLY...")
        for i, (dataset, fraction, seed, config_path) in enumerate(jobs):
            gpu_id = args.gpus[i % len(args.gpus)]
            cmd, job_id = create_job(dataset, fraction, seed, gpu_id, config_path)
            
            print(f"\nJob {i+1}/{len(jobs)}: {job_id}")
            process = launch_job_background(cmd, job_id, log_dir)
            process.wait()
            
            if process.returncode != 0:
                print(f"  ❌ Job failed with exit code {process.returncode}")
            else:
                print(f"  ✓ Job completed successfully")
    else:
        # Run jobs in parallel across GPUs
        print("Running jobs in PARALLEL across GPUs...")
        print()
        
        # Track running jobs per GPU
        gpu_jobs = {gpu_id: None for gpu_id in args.gpus}
        pending_jobs = list(enumerate(jobs))
        completed = 0
        failed = 0
        
        while pending_jobs or any(proc is not None for proc in gpu_jobs.values()):
            # Check for completed jobs and free up GPUs
            for gpu_id in list(gpu_jobs.keys()):
                if gpu_jobs[gpu_id] is not None:
                    process, job_id = gpu_jobs[gpu_id]
                    if process.poll() is not None:  # Job finished
                        if process.returncode == 0:
                            print(f"✓ Completed: {job_id} (GPU {gpu_id})")
                            completed += 1
                        else:
                            print(f"❌ Failed: {job_id} (GPU {gpu_id}, exit code {process.returncode})")
                            failed += 1
                        gpu_jobs[gpu_id] = None
            
            # Launch new jobs on free GPUs
            for gpu_id in args.gpus:
                if gpu_jobs[gpu_id] is None and pending_jobs:
                    i, (dataset, fraction, seed, config_path) = pending_jobs.pop(0)
                    cmd, job_id = create_job(dataset, fraction, seed, gpu_id, config_path)
                    process = launch_job_background(cmd, job_id, log_dir)
                    gpu_jobs[gpu_id] = (process, job_id)
                    running = sum(1 for v in gpu_jobs.values() if v is not None)
                    print(f"→ Started: {job_id} (GPU {gpu_id}) [{completed+failed+running}/{len(jobs)}]")
            
            # Sleep briefly before checking again
            time.sleep(2)
        
        print()
        print("="*80)
        print("ALL JOBS FINISHED")
        print("="*80)
        print(f"Completed: {completed}")
        print(f"Failed: {failed}")
        print(f"Total: {len(jobs)}")
        print(f"Logs: {log_dir}")
        
        if failed > 0:
            print("\n⚠️  Some jobs failed. Check logs for details.")
            sys.exit(1)


if __name__ == '__main__':
    main()
