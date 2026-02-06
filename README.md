# Active Learning Benchmarking Platform for Genomics

A modular platform for systematically benchmarking active learning (AL) strategies for genomic sequence-to-function prediction models. This project enables testing AL approaches on existing MPRA datasets before conducting expensive wet-lab experiments.

## Overview

This platform simulates active learning rounds using:
- **Student models**: Smaller models trained on subsets (e.g., DREAM-RNN)
- **Oracle models**: Better models or ground truth labels to simulate experimental results

The goal is to establish which AL strategies most efficiently improve model performance, reducing the number of sequences that need to be experimentally tested.

## Key Features

- **Modular architecture**: Easy to swap datasets, models, and AL strategies
- **Config-driven experiments**: Reproducible experiments via YAML configs
- **HPC-ready**: SLURM scripts for running large-scale experiments
- **Comprehensive logging**: Track training dynamics, performance metrics, and resource usage

## Installation

### Prerequisites

- Python 3.11
- [uv](https://github.com/astral-sh/uv) (fast Python package installer)
- Git

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd al-genomics-benchmark
```

2. Install dependencies using uv:
```bash
uv sync --all-extras
```

3. Activate the virtual environment:
```bash
source .venv/bin/activate
```

4. Set up environment variables:
```bash
cp env.template .env
# Edit .env with your specific paths
```

## Project Structure

```
al-genomics-benchmark/
├── data/                    # Data loading and preprocessing
│   ├── base.py              # Abstract dataset interface
│   ├── k562.py              # K562 human dataset loader
│   ├── yeast.py             # Yeast dataset loader
│   └── utils.py             # Encoding and sampling utilities
├── models/                  # Student model implementations
│   ├── base.py              # Abstract model interface
│   ├── dream_rnn.py         # DREAM-RNN architecture
│   └── training.py          # Training utilities
├── oracles/                 # Oracle models for AL simulation
│   ├── base.py              # Abstract oracle interface
│   └── perfect_oracle.py    # Perfect oracle (uses true labels)
├── acquisition/             # Acquisition functions
│   ├── base.py              # Abstract acquisition interface
│   ├── random.py            # Random acquisition baseline
│   └── uncertainty.py       # Uncertainty-based acquisition
├── proposal/                # Sequence proposal strategies
│   └── base.py              # Abstract proposal interface
├── simulation/              # AL simulation engine
│   ├── simulator.py         # Main AL simulation loop
│   └── evaluation.py        # Evaluation metrics
├── experiments/             # Experiment scripts
│   ├── 01_baseline_subsets.py
│   └── configs/
│       └── baseline.yaml
├── scripts/                 # Utility scripts
│   ├── download_data.py     # Download MPRA datasets
│   └── slurm/
│       └── baseline.sh      # SLURM job script
├── analysis/                # Results analysis
│   └── notebooks/
│       └── baseline_results.ipynb
└── tests/                   # Unit tests
```

## Datasets

### K562 Human MPRA Dataset
- **Source**: Gosai et al., Nature 2023 (Zenodo: 10698014)
- **Training**: 798,064 genomic sequences (230bp) from human genome with MPRA-measured activity
  - Includes reference and alternate alleles for common genetic variants (UKBB, GTEx)
  - Sequences from chromosomes 1-6, 8-12, 14-18, 20, 22 (chr 7, 13 held out for test)
- **Test Sets**:
  - **In-distribution**: Chr 7 & 13 (66,712 sequences)
  - **SNV/Variant Effect**: 45,543 ref/alt pairs for predicting expression differences
  - **OOD (Out-of-Domain)**: Synthetic sequences (not yet located - may require CODA library)
- **Task**: Predict regulatory activity and variant effects from sequence
- **Download**: `python scripts/download_data.py --dataset k562`

### Yeast Random Promoter Dataset
- **Source**: de Boer et al., Random Promoter DREAM Challenge (Zenodo: 10633252)
- **Training**: 6,739,258 random synthetic promoter sequences (110bp) with measured expression
  - Randomly generated DNA sequences in promoter-like context
  - Split: 5.46M train, 674K validation, 607K test (from random holdout)
- **Test Sets** (71,103 sequences total):
  - **In-distribution**: 6,349 random sequences (similar to training)
  - **SNV/Variant Effect**: 46,237 variant pairs with single nucleotide changes
  - **OOD (Out-of-Domain)**: 965 native yeast genomic promoters (vs. random training)
  - **Additional**: High/low expression, challenging, motif perturbation/tiling subsets
- **Task**: Predict promoter activity from sequence
- **Download**: `python scripts/download_data.py --dataset yeast`

**Note**: Training data distributions differ between datasets:
- Human: Genomic sequences (natural) → Test on synthetic (OOD)
- Yeast: Random sequences (synthetic) → Test on genomic (OOD)

## Data Splitting with HashFrag

This project uses **HashFrag** to create homology-aware train/validation/test splits for the K562 dataset, preventing data leakage from homologous (similar) sequences spanning different splits.

### Why HashFrag?

DNA sequences can be homologous due to duplication, repeat elements, or evolutionary relationships. If a sequence in the training set is very similar to a sequence in the test set, a model can "memorize" the training sequence and use it to predict the test sequence without actually learning the underlying regulatory logic. HashFrag solves this by ensuring train/val/test splits are truly independent.

**How it works:**
1. Uses BLAST to find candidate homologous pairs
2. Computes exact Smith-Waterman alignment scores
3. Groups sequences into homology clusters (graph-based)
4. Distributes entire clusters across train/val/test splits
5. No cluster spans multiple splits → no leakage!

### Setup HashFrag

**BLAST+ and HashFrag are already installed!**

To use them, simply source the environment setup script:

```bash
source setup_env.sh
```

This adds BLAST+ and HashFrag to your PATH. To make this automatic, add to your `~/.bashrc`:

```bash
echo 'source /home/trevor/al-genomics-benchmark/setup_env.sh' >> ~/.bashrc
```

### Creating Splits

**For K562 dataset only** (yeast uses provided splits):

**Option 1: Local (slow, 4-8 hours)**
```bash
python scripts/create_hashfrag_splits.py
```

**Option 2: HPC (recommended)**
```bash
# Submit to SLURM
sbatch scripts/slurm/create_hashfrag_splits.sh

# Monitor progress
tail -f logs/hashfrag_k562_*.log
```

Once created, splits are cached at `data/k562/hashfrag_splits/` and experiments will use them automatically.

### Split Configuration

- **Threshold**: Smith-Waterman score ≥60 defines homology
- **Split ratio**: 80% train+pool / 10% validation / 10% test
- **Train/Pool split**: First 100K from train for active learning experiments, rest for pool
- **Expected sizes**: ~293K train+pool, ~37K validation, ~37K test

### Verifying Splits

```bash
# Check if splits exist
ls -lh data/k562/hashfrag_splits/

# View split sizes
python -c "
import numpy as np
for split in ['train', 'pool', 'val', 'test']:
    idx = np.load(f'data/k562/hashfrag_splits/{split}_indices.npy')
    print(f'{split:5s}: {len(idx):>7,} sequences')
"
```

### Using in Experiments

Experiments automatically use HashFrag splits when available:

```python
from data.k562 import K562Dataset

# Will use cached HashFrag splits if they exist
dataset = K562Dataset(
    data_path="./data/k562",
    split='train',
    use_hashfrag=True  # default
)
```

## Quick Start

### 1. Download Data

```bash
python scripts/download_data.py --dataset k562
python scripts/download_data.py --dataset yeast
```

### 2. Run Baseline Experiments

Establish baseline data efficiency curves (no active learning):

```bash
python experiments/01_baseline_subsets.py --config experiments/configs/baseline.yaml
```

### 3. Analyze Results

```bash
jupyter notebook analysis/notebooks/baseline_results.ipynb
```

## Usage Examples

### Training DREAM-RNN on a Dataset

```python
from data.k562 import K562Dataset
from models.dream_rnn import DREAMRNN
from models.training import train_model

# Load dataset
dataset = K562Dataset(data_path="data/k562", split="train")

# Initialize model
model = DREAMRNN(input_channels=5, sequence_length=230)

# Train
train_model(
    model=model,
    train_dataset=dataset,
    val_dataset=val_dataset,
    epochs=80,
    batch_size=1024,
    lr=0.005,
)
```

### Running Active Learning Simulation

```python
from simulation.simulator import ALSimulator
from acquisition.uncertainty import UncertaintyAcquisition

# Initialize simulator
simulator = ALSimulator(
    student_model=model,
    oracle=oracle,
    acquisition_fn=UncertaintyAcquisition(),
    initial_pool_size=1000,
    budget_per_round=100,
    num_rounds=10,
)

# Run simulation
results = simulator.run()
```

## Running on HPC (CSHL bamdev4)

### Initial Setup on HPC

1. **Clone repository on HPC:**
```bash
ssh tmartin@bamdev4
cd /path/to/your/workspace
git clone <repository-url>
cd al-genomics-benchmark
```

2. **Install uv (if not already installed):**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env
```

3. **Set up environment:**
```bash
uv sync --all-extras
```

4. **Download data:**
```bash
source .venv/bin/activate
python scripts/download_data.py --dataset k562
python scripts/download_data.py --dataset yeast
```

### Submit SLURM Jobs

**For K562 dataset:**
```bash
sbatch scripts/slurm/baseline.sh k562
```

**For Yeast dataset:**
```bash
sbatch scripts/slurm/baseline.sh yeast
```

### Monitor Jobs

**Check job status:**
```bash
squeue -u tmartin
```

**View live logs:**
```bash
tail -f logs/al_baseline_*.log
```

**Check job details:**
```bash
scontrol show job <job_id>
```

**Cancel job:**
```bash
scancel <job_id>
```

### Job Configuration

The SLURM scripts (`scripts/slurm/baseline.sh`) are configured for:
- **Partition**: `kooq`
- **QOS**: `koolab`
- **GPU**: 1x H100 (96GB)
- **CPUs**: 14 per GPU
- **Memory**: 128GB per GPU
- **Time limit**: 48 hours

**To adjust resources**, edit the SBATCH directives in `scripts/slurm/baseline.sh`:
```bash
#SBATCH --gres=gpu:h100:1      # Change to v100:1 for V100 GPU
#SBATCH --cpus-per-gpu=14      # Adjust CPU count
#SBATCH --mem-per-gpu=128G     # Adjust memory
#SBATCH --time=48:00:00        # Adjust time limit
```

### Retrieving Results

After job completion, results are saved in:
- **Checkpoints**: `checkpoints/baseline_subsets_<dataset>_<timestamp>/`
- **Results**: `results/baseline_subsets_<dataset>_<timestamp>/`
- **Logs**: `logs/al_baseline_<job_id>.log`

**Download results to local machine:**
```bash
# From your local machine
scp -r tmartin@bamdev4:/path/to/al-genomics-benchmark/results/<experiment_dir> ./results/
```

**Or analyze on HPC using Jupyter:**
```bash
# On HPC login node
jupyter notebook --no-browser --port=8888
# Then tunnel from local: ssh -L 8888:localhost:8888 tmartin@bamdev4
```

## Development

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black .
```

### Type Checking

```bash
mypy data/ models/ experiments/
```

## Model Architecture: DREAM-RNN

The DREAM-RNN architecture used for student models:

1. **Input**: One-hot encoded sequences
   - K562: (batch, 5, 230) - ACGT + is_singleton flag
   - Yeast: (batch, 4, 80) - ACGT only

2. **Architecture**:
   - Dual CNN layers (kernel sizes 9 and 15)
   - Bi-directional LSTM (320 hidden units each direction)
   - Final CNN → global average pooling → linear output

3. **Training**:
   - Optimizer: AdamW (weight_decay=0.01)
   - Scheduler: OneCycleLR with cosine annealing
   - Loss: MSE
   - Metrics: Pearson R, Spearman R

## Citations

If you use this platform, please cite:

### Datasets

- **K562**: Gosai et al. (2023). "Machine-guided design of cell-type-targeting cis-regulatory elements." Nature. DOI: 10.5281/zenodo.10698014

- **Yeast**: de Boer et al. (2024). "A high-resolution gene expression atlas of epistasis between gene-specific transcription factors exposes potential mechanisms for genetic interactions." Nature Biotechnology. DOI: 10.5281/zenodo.10633252

## License

[To be determined]

## Contact

Trevor Martin - Cold Spring Harbor Laboratory
[Contact information]

## Acknowledgments

- de Boer Lab (Yale University) for collaboration and DREAM-RNN architecture
- Gosai et al. for K562 MPRA dataset
- de Boer et al. for yeast MPRA dataset
