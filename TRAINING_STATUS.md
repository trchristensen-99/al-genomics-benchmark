# Training Status Report

**Generated:** February 5, 2026

## ✅ All Jobs Running Successfully

### Architecture Verification

Both K562 and Yeast datasets are using **correctly configured** DREAM-RNN models:

| Dataset | Sequence Length | Input Channels | Model Parameters | Architecture |
|---------|----------------|----------------|------------------|--------------|
| **K562** | 230 bp | 5 (ACGT + singleton) | 4,203,393 | DREAM-RNN |
| **Yeast** | 110 bp | 4 (ACGT only) | 4,199,553 | DREAM-RNN |

### Model Architecture Details

The DREAM-RNN model **automatically adapts** to different input dimensions:

```python
model = create_dream_rnn(
    input_channels=train_dataset.get_num_channels(),  # 5 for K562, 4 for Yeast
    sequence_length=train_dataset.get_sequence_length(),  # 230 for K562, 110 for Yeast
    hidden_dim=320,
    cnn_filters=160,
    dropout_cnn=0.2,
    dropout_lstm=0.5
)
```

**Architecture Components:**
1. **First Layer**: Dual CNNs (kernel sizes 9, 15) → 160 filters each → 320 total after concat
2. **Core Layer**: Bi-LSTM (320 units × 2 directions = 640) → Dual CNNs (same as first layer)
3. **Final Layer**: 1D Conv (256 filters) → Global Average Pooling → Linear output

The model uses `padding='same'` in all conv layers, so sequence length is preserved throughout.

### Current Training Status

**GPU Utilization:**
```
GPU 0 (K562 5%):   15GB memory, 100% utilization
GPU 1 (K562 10%):  15GB memory, 100% utilization
GPU 2 (K562 25%):  15GB memory,  83% utilization
GPU 3 (K562 50%):  15GB memory,  94% utilization
GPU 4 (Yeast 1%):   8GB memory,  68% utilization
GPU 5 (Yeast 2%):   8GB memory,  76% utilization
GPU 6 (Yeast 4%):   8GB memory,  98% utilization
GPU 7 (Yeast 8%):   8GB memory,  81% utilization
```

### Training Configuration

**Common Settings:**
- Epochs: 80
- Optimizer: AdamW (weight_decay=0.01)
- Scheduler: OneCycleLR with cosine annealing
- Learning rates: 0.005 (CNN), 0.001 (LSTM)
- Batch size: 1024
- Random seed: 42

**K562 Fractions:**
- 5%: 39,903 samples
- 10%: 79,806 samples
- 25%: 199,516 samples
- 50%: 399,032 samples

**Yeast Fractions:**
- 1%: 54,588 samples (from new 5.46M train set)
- 2%: 109,176 samples
- 4%: 218,352 samples
- 8%: 436,703 samples

### Estimated Completion Times

Based on current progress and training speed:

| Job | Dataset | Fraction | Samples | Est. Time per Epoch | Total Est. Time | Status |
|-----|---------|----------|---------|---------------------|-----------------|--------|
| GPU 0 | K562 | 5% | 39,903 | ~5 min | ~6.5 hours | Running |
| GPU 1 | K562 | 10% | 79,806 | ~10 min | ~13 hours | Running |
| GPU 2 | K562 | 25% | 199,516 | ~25 min | ~33 hours | Running |
| GPU 3 | K562 | 50% | 399,032 | ~50 min | ~67 hours | Running |
| GPU 4 | Yeast | 1% | 54,588 | ~3 min | ~4 hours | Running |
| GPU 5 | Yeast | 2% | 109,176 | ~6 min | ~8 hours | Running |
| GPU 6 | Yeast | 4% | 218,352 | ~12 min | ~16 hours | Running |
| GPU 7 | Yeast | 8% | 436,703 | ~24 min | ~32 hours | Running |

**Note:** These are rough estimates. Actual times may vary based on:
- Early stopping (if validation performance plateaus)
- GPU thermal throttling
- System load

### Output Locations

**Checkpoints:**
```
checkpoints/baseline_subsets_{dataset}_{dataset}_{timestamp}/
└── fraction_{X.XXX}/
    ├── best_model.pt
    ├── final_model.pt
    └── training_history.json
```

**Results:**
```
results/baseline_subsets_{dataset}_{dataset}_{timestamp}/
├── config.yaml
└── results.json
```

**Logs:**
```
logs/baseline_runs/
├── k562_5pct_gpu0.log
├── k562_10pct_gpu1.log
├── k562_25pct_gpu2.log
├── k562_50pct_gpu3.log
├── yeast_1pct_gpu4.log
├── yeast_2pct_gpu5.log
├── yeast_4pct_gpu6.log
└── yeast_8pct_gpu7.log
```

### Monitoring Commands

```bash
# Check all job status
./scripts/monitor_jobs.sh

# Watch specific log
tail -f logs/baseline_runs/k562_5pct_gpu0.log

# GPU utilization
watch -n 5 nvidia-smi

# Check running processes
ps aux | grep "01_baseline_subsets.py"
```

### Issues Resolved

1. ✅ **Yeast test set missing** → Created test.txt by randomly holding out 10% of train data
2. ✅ **CUDA compatibility** → Reinstalled PyTorch with cu118 for CUDA 11.4
3. ✅ **Sequence length inconsistency** → Updated K562 loader to pad/truncate to 230bp
4. ✅ **Gradient computation error** → Fixed reverse complement handling in training loop
5. ✅ **DataLoader multiprocessing** → Set num_workers=0 for PyTorch 2.4.1 compatibility

### Next Steps

After training completes:
1. Analyze results with `analysis/notebooks/baseline_results.ipynb`
2. Plot data efficiency curves (Pearson R vs. training set size)
3. Train oracle models on 100% of data
4. Run active learning simulations comparing strategies
