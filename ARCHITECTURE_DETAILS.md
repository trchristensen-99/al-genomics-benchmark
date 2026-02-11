# DREAM-RNN Architecture Details

## Overview

The DREAM-RNN model follows the Prix Fixe framework architecture from the de Boer Lab, used in the Random Promoter DREAM Challenge. The architecture consists of three main blocks:

1. **First Layer Block**: Dual parallel CNNs with different kernel sizes
2. **Core Block**: Bidirectional LSTM followed by another dual CNN block
3. **Final Layer Block**: Point-wise convolution → global average pooling → task-specific output

---

## Shared Architecture (Both Yeast and K562)

### Input Processing
- **Input Format**: One-hot encoded sequences with additional metadata channels
- **Tensor Shape**: `(batch_size, input_channels, sequence_length)`

### First Layer Block
```
Input: (batch, input_channels, seq_len)
  ↓
Conv1D (kernel=9, filters=160)  ──┐
                                   ├──→ Concat → (batch, 320, seq_len)
Conv1D (kernel=15, filters=160) ──┘
  ↓
ReLU activation
  ↓
Dropout (rate: 0.1 for training, 0.2 in original paper)
  ↓
Output: (batch, 320, seq_len)
```

**Rationale**: Dual CNNs with kernel sizes 9 and 15 capture motifs of different lengths (typical regulatory motifs are 9-15bp).

### Core Block

#### Part 1: Bidirectional LSTM
```
Input: (batch, 320, seq_len)
  ↓
Permute: (batch, seq_len, 320)
  ↓
Bidirectional LSTM
  - Hidden size per direction: 320
  - Total output size: 640 (320 forward + 320 backward)
  - Num layers: 1
  - Dropout between layers: 0.0 (only one layer)
  ↓
Output: (batch, seq_len, 640)
  ↓
Permute: (batch, 640, seq_len)
```

#### Part 2: Dual CNN Block (Same as First Layer)
```
Input: (batch, 640, seq_len)
  ↓
Conv1D (kernel=9, filters=160)  ──┐
                                   ├──→ Concat → (batch, 320, seq_len)
Conv1D (kernel=15, filters=160) ──┘
  ↓
ReLU activation
  ↓
Dropout (rate: 0.1 for training, 0.5 in original paper)
  ↓
Output: (batch, 320, seq_len)
```

### Final Layer Block
```
Input: (batch, 320, seq_len)
  ↓
Conv1D (kernel=1, filters=256)  # Point-wise convolution
  ↓
ReLU activation
  ↓
Global Average Pooling (along sequence dimension)
  ↓
Output: (batch, 256)
  ↓
Linear Layer
  ↓
Output: (batch, output_dim)
```

---

## Task-Specific Configurations

### Yeast (Random Promoter DREAM Challenge)

#### Input Specifications
- **Sequence Length**: 150 bp
  - 57 bp 5' flanking (plasmid context)
  - 80 bp random promoter region
  - 13 bp 3' flanking (plasmid context)
- **Input Channels**: 6
  - Channel 0-3: ACGT one-hot encoding
  - Channel 4: Reverse complement flag (0 = forward, 1 = reverse)
  - Channel 5: Singleton flag (1 = singleton expression, 0 = non-singleton)
- **Input Shape**: `(batch, 6, 150)`

#### Final Layer Block (Yeast-Specific)
```
Linear Layer Output: (batch, 18)  # 18 logits for bins
  ↓
SoftMax activation
  ↓
Bin Probabilities: (batch, 18)  # p = [p₀, p₁, ..., p₁₇]
  ↓
Weighted Average: expression = Σᵢ (bin_center_i × pᵢ)
  where bin_centers = [0.0, 1.0, 2.0, ..., 17.0]
  ↓
Final Output: (batch,)  # Continuous expression value
```

**Bin Centers**: Integer values from 0.0 to 17.0 (18 bins total)
- Bin 0: center = 0.0
- Bin 1: center = 1.0
- ...
- Bin 17: center = 17.0

#### Loss Function
- **KL Divergence Loss** (Kullback-Leibler divergence)
- Target: One-hot encoded bin probabilities from continuous expression values
- Input to loss: Log-probabilities (log_softmax of logits)
- Formula: `KL(target_probs || predicted_probs)`

#### Training Configuration
- **Epochs**: 80 (no early stopping)
- **Batch Size**: 1024
- **Optimizer**: AdamW
  - Learning rate (CNN): 0.005
  - Learning rate (LSTM): 0.001 (lower for attention-like components)
  - Weight decay: 0.01
- **Scheduler**: OneCycleLR
  - pct_start: 0.3 (30% of training spent increasing LR)
- **Dropout**:
  - CNN dropout: 0.1 (0.2 in original paper)
  - LSTM dropout: 0.1 (0.5 in original paper)
- **Mixed Precision**: Enabled (FP16)
- **Reverse Complement**: Enabled (averages forward and reverse predictions)

---

### K562 (Human MPRA)

#### Input Specifications
- **Sequence Length**: 200 bp (genomic sequences, padded with Ns if shorter)
- **Input Channels**: 5
  - Channel 0-3: ACGT one-hot encoding
  - Channel 4: Reverse complement flag (0 = forward, 1 = reverse)
- **Input Shape**: `(batch, 5, 200)`

#### Final Layer Block (K562-Specific)
```
Linear Layer Output: (batch, 1)  # Direct regression
  ↓
Squeeze dimension
  ↓
Final Output: (batch,)  # Continuous expression value (log2 fold change)
```

**No SoftMax or binning** - direct regression output.

#### Loss Function
- **MSE Loss** (Mean Squared Error)
- Target: Continuous expression values (log2 fold change)
- Formula: `MSE = mean((predictions - targets)²)`

#### Training Configuration
- **Epochs**: 80 (no early stopping)
- **Batch Size**: 1024
- **Optimizer**: AdamW
  - Learning rate (CNN): 0.005
  - Learning rate (LSTM): 0.001
  - Weight decay: 0.01
- **Scheduler**: OneCycleLR
  - pct_start: 0.3
- **Dropout**:
  - CNN dropout: 0.1 (0.2 in original paper)
  - LSTM dropout: 0.1 (0.5 in original paper)
- **Mixed Precision**: Enabled (FP16)
- **Reverse Complement**: Enabled (averages forward and reverse predictions)

---

## Architecture Summary Table

| Component | Yeast | K562 |
|-----------|-------|------|
| **Input Sequence Length** | 150 bp | 200 bp |
| **Input Channels** | 6 (ACGT + RC + singleton) | 5 (ACGT + RC) |
| **First Layer Block** | Dual CNNs (k=9,15, f=160 each) → 320 channels | Same |
| **LSTM Hidden Size** | 320 per direction (640 total) | Same |
| **Core CNN Block** | Dual CNNs (k=9,15, f=160 each) → 320 channels | Same |
| **Final Conv** | Point-wise (k=1, f=256) | Same |
| **Final Linear** | 256 → 18 logits | 256 → 1 value |
| **Output Processing** | SoftMax → Weighted Average | Direct output |
| **Output Dimension** | 1 (after weighted average) | 1 |
| **Loss Function** | KL Divergence | MSE |
| **Total Parameters** | ~4.2M | ~4.2M |

---

## Key Architectural Details

### Weight Initialization
- **Conv1D layers**: Kaiming normal (He initialization) for ReLU
- **Linear layers**: Xavier normal (Glorot initialization)
- **LSTM**: Xavier normal for weights, zeros for biases

### Dropout Strategy
- Applied after ReLU activations
- CNN dropout: 0.1 (reduced from 0.2 for MC dropout compatibility)
- LSTM dropout: 0.1 (reduced from 0.5 for MC dropout compatibility)
- No dropout between LSTM layers (only one layer)

### Reverse Complement Augmentation
- During inference: Average predictions from forward and reverse complement sequences
- Reverse complement transformation:
  1. Reverse sequence along length dimension
  2. Swap ACGT channels: A↔T (0↔3), C↔G (1↔2)
  3. Set reverse complement flag to 1.0
  4. Keep other metadata channels unchanged

### Model Parameters
- **Total parameters**: ~4,211,602 (both yeast and K562)
- Breakdown:
  - First layer CNNs: ~(6×160×9 + 6×160×15) = ~23K
  - LSTM: ~(320×640×4 + biases) = ~820K
  - Core CNNs: ~(640×160×9 + 640×160×15) = ~2.5M
  - Final conv: ~(320×256) = ~82K
  - Final linear: ~(256×18 for yeast, 256×1 for K562) = ~4.6K or ~256

---

## Differences from Original Prix Fixe Paper

1. **Dropout Rates**: Reduced from (0.2, 0.5) to (0.1, 0.1) for MC dropout compatibility
2. **Final Layer**: 
   - Yeast: 18-bin classification (matches paper)
   - K562: Direct regression (matches paper's adaptation for human MPRA)
3. **Training**: No early stopping (trains full 80 epochs as in paper)

---

## Verification Checklist

✅ **Yeast Architecture**:
- [x] 6 input channels (ACGT + RC + singleton)
- [x] 150 bp sequence length
- [x] Dual CNNs (k=9,15) with 160 filters each
- [x] Bidirectional LSTM with 320 hidden units per direction
- [x] Final layer: 18 logits → SoftMax → weighted average
- [x] KL divergence loss
- [x] Bin centers: [0.0, 1.0, ..., 17.0]

✅ **K562 Architecture**:
- [x] 5 input channels (ACGT + RC)
- [x] 200 bp sequence length
- [x] Same CNN/LSTM architecture as yeast
- [x] Final layer: 1 output → direct regression
- [x] MSE loss

✅ **Training Configuration**:
- [x] 80 epochs, no early stopping
- [x] Batch size: 1024
- [x] AdamW optimizer with separate LRs for CNN (0.005) and LSTM (0.001)
- [x] OneCycleLR scheduler (pct_start=0.3)
- [x] Reverse complement augmentation enabled
- [x] Mixed precision (FP16) enabled
