# Baseline Experiments - Currently Running

## Status: ✅ TRAINING IN PROGRESS

Started: February 5, 2026 at 15:20 UTC

## Jobs Running

### K562 Human Dataset (GPUs 0-3)
- **GPU 0**: K562 5% (39,903 samples) - `logs/baseline_runs/k562_5pct_gpu0.log`
- **GPU 1**: K562 10% (79,806 samples) - `logs/baseline_runs/k562_10pct_gpu1.log`
- **GPU 2**: K562 25% (199,516 samples) - `logs/baseline_runs/k562_25pct_gpu2.log`
- **GPU 3**: K562 50% (399,032 samples) - `logs/baseline_runs/k562_50pct_gpu3.log`

### Yeast Dataset (GPUs 4-7)
- **GPU 4**: Yeast 1% (60,653 samples) - `logs/baseline_runs/yeast_1pct_gpu4.log`
- **GPU 5**: Yeast 2% (121,307 samples) - `logs/baseline_runs/yeast_2pct_gpu5.log`
- **GPU 6**: Yeast 4% (242,613 samples) - `logs/baseline_runs/yeast_4pct_gpu6.log`
- **GPU 7**: Yeast 8% (485,226 samples) - `logs/baseline_runs/yeast_8pct_gpu7.log`

## Configuration

- **Random seed**: 42 (single replicate)
- **Model**: DREAM-RNN (4.2M parameters)
- **Training**: 80 epochs with OneCycleLR
- **Batch size**: 1024
- **Learning rate**: 0.005 (CNN), 0.001 (LSTM)

## Monitoring

### Check job status:
```bash
./scripts/monitor_jobs.sh
```

### Watch a specific job:
```bash
tail -f logs/baseline_runs/k562_5pct_gpu0.log
```

### Check GPU utilization:
```bash
watch -n 5 nvidia-smi
```

### Check running processes:
```bash
ps aux | grep "01_baseline_subsets.py"
```

## Expected Runtime

- **K562 5%**: ~1-2 hours
- **K562 10%**: ~2-3 hours  
- **K562 25%**: ~4-6 hours
- **K562 50%**: ~8-12 hours
- **Yeast 1%**: ~2-3 hours
- **Yeast 2%**: ~4-6 hours
- **Yeast 4%**: ~8-12 hours
- **Yeast 8%**: ~16-24 hours

## Output Locations

### Checkpoints:
```
checkpoints/baseline_subsets_{dataset}_{dataset}_{timestamp}/
└── fraction_{X.XXX}/
    ├── best_model.pt
    ├── final_model.pt
    └── training_history.json
```

### Results:
```
results/baseline_subsets_{dataset}_{dataset}_{timestamp}/
├── config.yaml
└── results.json
```

## Issues Fixed

1. ✅ CUDA compatibility (PyTorch 2.4.1+cu118 for CUDA 11.4)
2. ✅ Sequence length inconsistency (K562 padding)
3. ✅ Gradient computation (removed reverse complement from training)
4. ✅ DataLoader multiprocessing (set num_workers=0)

## Next Steps

After completion:
1. Check results: `find results/ -name "results.json"`
2. Analyze performance: Open `analysis/notebooks/baseline_results.ipynb`
3. Compare data efficiency curves
4. Train oracle models on 100% data
5. Run active learning simulations
