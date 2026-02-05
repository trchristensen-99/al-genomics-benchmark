# Active Learning Framework Documentation

## Overview

This document explains the active learning (AL) framework infrastructure, how it differs from baseline experiments, and how to use it for benchmarking acquisition strategies.

## Table of Contents

1. [Key Concepts](#key-concepts)
2. [Architecture](#architecture)
3. [Baseline vs. AL Experiments](#baseline-vs-al-experiments)
4. [Components](#components)
5. [Usage Examples](#usage-examples)
6. [Model Checkpoints](#model-checkpoints)

---

## Key Concepts

### What is Active Learning?

Active learning is a machine learning paradigm where the model actively selects which samples to label next, rather than passively receiving random samples. The goal is to achieve better performance with fewer labeled samples by strategically choosing informative data points.

### AL Loop

```
1. Start with small labeled set (e.g., 1000 samples)
2. Train model on labeled data
3. Evaluate model
4. Use acquisition strategy to select next batch from unlabeled pool
5. Add selected samples to labeled set
6. Repeat from step 2
```

### Pool Management

The **DataPool** maintains two disjoint sets:
- **Labeled set**: Samples that have been "acquired" and can be used for training
- **Pool set**: Unlabeled samples available for selection

This ensures:
- No duplicate selections across rounds
- Fair comparison between strategies (all see same pool)
- Realistic simulation of sequential data acquisition

---

## Architecture

```
al_genomics/
├── __init__.py           # Package exports
├── pool.py               # DataPool class for managing labeled/unlabeled splits
└── acquisition.py        # Acquisition strategy implementations

experiments/
├── 01_baseline_subsets.py    # Baseline: random subsets (no AL)
├── 02_al_simulation.py       # AL simulation with iterative loop
└── configs/
    ├── baseline_yeast.yaml
    ├── baseline_k562.yaml
    ├── al_simulation.yaml        # AL config for yeast
    └── al_simulation_k562.yaml   # AL config for K562
```

---

## Baseline vs. AL Experiments

### Baseline Subset Experiment (`01_baseline_subsets.py`)

**Purpose**: Establish passive learning baseline

**How it works**:
1. For each subset fraction (1%, 5%, 10%, 25%, 50%, 75%, 100%):
   - Randomly sample that fraction of training data
   - Train a **fresh model from scratch** on that subset
   - Evaluate on full validation/test sets
   - Save model and metrics
2. Output: Data efficiency curve (performance vs. training set size)

**Key characteristics**:
- Each model is independent
- No pool management (just random sampling)
- No iterative acquisition
- Simulates: "What if we had randomly collected X samples?"

**When to use**:
- Establishing baseline data efficiency
- Comparing to random acquisition strategy
- Understanding model scaling behavior

### AL Simulation Experiment (`02_al_simulation.py`)

**Purpose**: Benchmark active learning strategies

**How it works**:
1. Initialize pool with small labeled set (e.g., 1000 samples)
2. For each AL round:
   - Train model on current labeled set
   - Evaluate on validation/test sets
   - Use acquisition strategy to select next batch from pool
   - Move selected samples from pool to labeled set
3. Output: AL curves for each strategy (performance vs. acquisition rounds)

**Key characteristics**:
- Iterative process with pool management
- Strategic sample selection (not random)
- Models can be trained from scratch or warm-started
- Simulates: "What if we strategically chose which sequences to measure?"

**When to use**:
- Comparing acquisition strategies (random, uncertainty, diversity, hybrid)
- Optimizing data collection strategy
- Understanding which samples are most informative

---

## Components

### 1. DataPool (`al_genomics/pool.py`)

Manages the partition between labeled and unlabeled data.

**Key methods**:

```python
from al_genomics import DataPool

# Initialize with dataset and optional initial labeled samples
pool = DataPool(
    full_dataset=train_dataset,
    initial_labeled_indices=initial_indices,  # np.ndarray of indices
    random_seed=42
)

# Get current datasets
labeled_dataset = pool.get_labeled_dataset()  # Subset for training
pool_dataset = pool.get_pool_dataset()        # Subset for acquisition

# Acquire new samples
selected_indices = np.array([100, 200, 300])  # Indices relative to pool
pool_indices = pool.get_pool_indices()        # Map to full dataset
full_indices = pool_indices[selected_indices]
pool.add_to_labeled(full_indices, round_num=1)

# Get statistics
stats = pool.get_statistics()
# Returns: {
#   'total_samples': 6065325,
#   'num_labeled': 1003,
#   'num_unlabeled': 6064322,
#   'fraction_labeled': 0.000165,
#   'num_acquisition_rounds': 2,
#   'is_exhausted': False
# }
```

**Features**:
- Validates no duplicate acquisitions
- Tracks acquisition history per round
- Provides clean Subset interfaces for PyTorch DataLoader

### 2. Acquisition Strategies (`al_genomics/acquisition.py`)

All strategies inherit from `AcquisitionStrategy` ABC.

#### RandomAcquisition (Baseline)

Random sampling without using model information.

```python
from al_genomics import RandomAcquisition

strategy = RandomAcquisition(random_seed=42)
selected_indices = strategy.select_batch(
    model=model,              # Not used
    pool_dataset=pool_dataset,
    n_samples=500,
    device=device
)
```

**Use case**: Passive learning baseline that AL strategies should beat.

#### UncertaintyAcquisition

Selects samples where the model is most uncertain.

```python
from al_genomics import UncertaintyAcquisition

strategy = UncertaintyAcquisition(
    random_seed=42,
    mc_dropout_samples=10,    # Number of forward passes with dropout
    batch_size=256            # For inference
)

selected_indices = strategy.select_batch(
    model=model,
    pool_dataset=pool_dataset,
    n_samples=500,
    device=device
)
```

**Methods**:
- **MC Dropout**: Enable dropout during inference, compute variance across multiple forward passes
- **Ensemble** (if provided): Compute variance across multiple models

**Use case**: Exploit model uncertainty to find hard/ambiguous samples.

#### DiversityAcquisition

Selects diverse samples for good sequence space coverage.

```python
from al_genomics import DiversityAcquisition

strategy = DiversityAcquisition(
    random_seed=42,
    use_model_embeddings=True,  # Use model embeddings vs. k-mer features
    batch_size=256
)

selected_indices = strategy.select_batch(
    model=model,
    pool_dataset=pool_dataset,
    n_samples=500,
    device=device
)
```

**Method**:
- Extract embeddings from model (or use k-mer features)
- Use K-means clustering to find diverse cluster centers
- Select samples nearest to cluster centers

**Use case**: Explore sequence space, avoid redundant similar sequences.

#### HybridAcquisition

Combines uncertainty and diversity with weighted scoring.

```python
from al_genomics import HybridAcquisition

strategy = HybridAcquisition(
    random_seed=42,
    uncertainty_weight=0.5,   # Balance between uncertainty and diversity
    diversity_weight=0.5,
    mc_dropout_samples=10,
    batch_size=256
)

selected_indices = strategy.select_batch(
    model=model,
    pool_dataset=pool_dataset,
    n_samples=500,
    device=device
)
```

**Method**:
- Greedy selection maximizing: `uncertainty_weight * uncertainty + diversity_weight * diversity`

**Use case**: Balance exploration (diversity) and exploitation (uncertainty).

---

## Usage Examples

### Running Baseline Experiments

```bash
# Yeast dataset
python experiments/01_baseline_subsets.py \
    --config experiments/configs/baseline_yeast.yaml

# K562 dataset
python experiments/01_baseline_subsets.py \
    --config experiments/configs/baseline_k562.yaml
```

**Output structure**:
```
checkpoints/baseline_subsets_yeast_20260205_143022/
├── fraction_0.010/
│   ├── best_model.pt
│   ├── final_model.pt
│   └── training_history.json
├── fraction_0.050/
│   └── ...
└── ...

results/baseline_subsets_yeast_20260205_143022/
├── config.yaml
└── results.json  # Contains all metrics
```

### Running AL Simulations

```bash
# Random acquisition (baseline)
python experiments/02_al_simulation.py \
    --config experiments/configs/al_simulation.yaml \
    --strategy random

# Uncertainty-based acquisition
python experiments/02_al_simulation.py \
    --config experiments/configs/al_simulation.yaml \
    --strategy uncertainty

# Diversity-based acquisition
python experiments/02_al_simulation.py \
    --config experiments/configs/al_simulation.yaml \
    --strategy diversity

# Hybrid acquisition
python experiments/02_al_simulation.py \
    --config experiments/configs/al_simulation.yaml \
    --strategy hybrid
```

**Output structure**:
```
checkpoints/al_uncertainty_yeast_20260205_150000/
├── round_000/  # Initial random set
│   ├── best_model.pt
│   ├── final_model.pt
│   └── training_history.json
├── round_001/  # After 1st acquisition
│   └── ...
├── round_002/  # After 2nd acquisition
│   └── ...
└── ...

results/al_uncertainty_yeast_20260205_150000/
├── config.yaml
└── results.json  # Contains metrics per round + acquisition info
```

### Configuration Parameters

Key parameters in `al_simulation.yaml`:

```yaml
active_learning:
  strategy: "random"           # random, uncertainty, diversity, hybrid
  initial_samples: 1000        # Initial random labeled set
  batch_size: 500              # Samples to acquire per round
  num_rounds: 10               # Number of AL rounds
  
  # Strategy-specific
  mc_dropout_samples: 10       # For uncertainty
  uncertainty_weight: 0.5      # For hybrid
  diversity_weight: 0.5        # For hybrid
```

---

## Model Checkpoints

### Checkpoint Format

All models are saved using the `SequenceModel.save_checkpoint()` method:

```python
checkpoint = {
    'model_state_dict': model.state_dict(),
    'model_config': {
        'input_channels': 4,
        'sequence_length': 110,
        'output_dim': 1
    },
    'epoch': 79,
    'optimizer_state_dict': optimizer.state_dict(),
    'train_metrics': {...},
    'val_metrics': {...}
}
```

### Loading Checkpoints

**Option 1: Using model's load_checkpoint method** (recommended):

```python
from models.dream_rnn import create_dream_rnn

# Create model with same architecture
model = create_dream_rnn(
    input_channels=4,
    sequence_length=110,
    hidden_dim=320,
    cnn_filters=160
)

# Load weights
checkpoint_path = "checkpoints/al_uncertainty_yeast_20260205/round_005/best_model.pt"
model.load_checkpoint(checkpoint_path)

# Use for prediction
model.eval()
predictions = model.predict(sequences, use_reverse_complement=True)
```

**Option 2: Manual loading for inspection**:

```python
import torch

checkpoint = torch.load(checkpoint_path)
state_dict = checkpoint['model_state_dict']
config = checkpoint['model_config']
val_metrics = checkpoint['val_metrics']

print(f"Validation Pearson R: {val_metrics['pearson_r']:.4f}")

model.load_state_dict(state_dict)
```

### Checkpoint Organization

**Baseline experiments**:
```
checkpoints/baseline_subsets_{dataset}_{timestamp}/
└── fraction_{X.XXX}/
    ├── best_model.pt      # Best by validation metric
    └── final_model.pt     # After all epochs
```

**AL experiments**:
```
checkpoints/al_{strategy}_{dataset}_{timestamp}/
└── round_{NNN}/
    ├── best_model.pt      # Best by validation metric
    └── final_model.pt     # After all epochs
```

### Oracle Models

For AL experiments, you'll want to train an "oracle" model once on the full dataset and reuse it:

```bash
# Train oracle on full yeast dataset
python experiments/01_baseline_subsets.py \
    --config experiments/configs/baseline_yeast.yaml

# This will create a checkpoint at fraction_1.000/best_model.pt
# Copy to a dedicated oracle directory:
mkdir -p checkpoints/oracle_yeast
cp checkpoints/baseline_subsets_yeast_*/fraction_1.000/best_model.pt \
   checkpoints/oracle_yeast/oracle.pt
```

Then load in your AL experiments:

```python
oracle_model = create_dream_rnn(...)
oracle_model.load_checkpoint("checkpoints/oracle_yeast/oracle.pt")
oracle_model.eval()

# Use oracle for generating "ground truth" labels
with torch.no_grad():
    oracle_predictions = oracle_model.predict(sequences, use_reverse_complement=True)
```

---

## Key Differences Summary

| Aspect | Baseline Subsets | AL Simulation |
|--------|-----------------|---------------|
| **Purpose** | Passive learning baseline | Active learning benchmark |
| **Data selection** | Random sampling | Strategic acquisition |
| **Pool management** | No (independent subsets) | Yes (DataPool tracks labeled/unlabeled) |
| **Training** | One model per subset size | One model per AL round |
| **Iterations** | None (single training per subset) | Iterative (train → acquire → repeat) |
| **Comparison** | Different data amounts | Different acquisition strategies |
| **Output** | Data efficiency curve | AL learning curves per strategy |

---

## Next Steps

1. **Run baseline experiments** to establish passive learning curves
2. **Train oracle models** on full datasets for comparison
3. **Run AL simulations** with different strategies
4. **Compare results** in `analysis/notebooks/` to see which strategies are most data-efficient

For analysis and visualization, see `analysis/notebooks/baseline_results.ipynb`.
