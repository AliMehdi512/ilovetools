# Release Notes - v0.2.17

**Release Date:** December 27, 2024

## 🎯 NEW: Normalization Techniques Module

Added 5 essential normalization techniques for deep neural networks!

### 📦 What's New

#### Normalization Functions (5 techniques)
1. **batch_normalization()** - Normalizes across batch dimension (CNNs)
2. **layer_normalization()** - Normalizes across features (Transformers)
3. **group_normalization()** - Divides channels into groups (small batches)
4. **instance_normalization()** - Per-sample normalization (style transfer, GANs)
5. **weight_normalization()** - Normalizes weight vectors (faster than BatchNorm)

#### Forward Passes with Cache (2 functions)
6. **batch_norm_forward()** - BatchNorm with cache for backprop
7. **layer_norm_forward()** - LayerNorm with cache for backprop

#### Utilities (2 functions)
8. **create_normalization_params()** - Initialize normalization parameters
9. **apply_normalization()** - Apply normalization by name

#### Aliases (5 shortcuts)
10. **batchnorm**, **layernorm**, **groupnorm**, **instancenorm**, **weightnorm**

## 💻 Installation

```bash
pip install --upgrade ilovetools
```

## ✅ Quick Start

```python
from ilovetools.ml.normalization import (
    batch_normalization,
    layer_normalization,
    group_normalization,
    instance_normalization
)
import numpy as np

# Batch Normalization (for CNNs)
x = np.random.randn(32, 64)  # Batch of 32, 64 features
gamma = np.ones(64)
beta = np.zeros(64)

out, running_mean, running_var = batch_normalization(
    x, gamma, beta, training=True
)
print(f"BatchNorm: {x.shape} -> {out.shape}")
print(f"Mean: {np.mean(out):.4f}, Var: {np.var(out):.4f}")

# Layer Normalization (for Transformers)
out = layer_normalization(x, gamma, beta)
print(f"LayerNorm: {x.shape} -> {out.shape}")

# For sequences (BERT, GPT)
x_seq = np.random.randn(32, 10, 512)  # Batch 32, seq 10, dim 512
gamma_seq = np.ones(512)
beta_seq = np.zeros(512)
out = layer_normalization(x_seq, gamma_seq, beta_seq)
print(f"LayerNorm (seq): {x_seq.shape} -> {out.shape}")

# Group Normalization (for small batches)
x_img = np.random.randn(8, 64, 32, 32)  # Small batch
gamma_img = np.ones(64)
beta_img = np.zeros(64)
out = group_normalization(x_img, gamma_img, beta_img, num_groups=32)
print(f"GroupNorm: {x_img.shape} -> {out.shape}")

# Instance Normalization (for style transfer)
out = instance_normalization(x_img, gamma_img, beta_img)
print(f"InstanceNorm: {x_img.shape} -> {out.shape}")
```

## 🎯 Selection Guide

### By Architecture

| Architecture | Normalization | Why |
|--------------|---------------|-----|
| **CNNs** | Batch Normalization | Best for computer vision |
| **Transformers** | Layer Normalization | BERT, GPT standard |
| **RNNs/LSTMs** | Layer Normalization | Batch-independent |
| **GANs** | Instance Normalization | Per-sample normalization |
| **Small Batch** | Group Normalization | Batch-size independent |

### By Use Case

**Computer Vision:**
- Standard CNNs → Batch Normalization
- Small batches → Group Normalization
- Style transfer → Instance Normalization

**Natural Language Processing:**
- Transformers → Layer Normalization
- RNNs → Layer Normalization
- Character-level → Layer Normalization

**Generative Models:**
- GANs → Instance Normalization
- VAEs → Batch Normalization
- Diffusion → Group Normalization

## 📊 Comparison

### Batch Normalization
**Formula:** y = γ((x - μ_B) / √(σ²_B + ε)) + β

✅ **Pros:**
- Faster convergence
- Higher learning rates
- Reduces covariate shift
- Acts as regularization

❌ **Cons:**
- Batch size dependent
- Different train/test behavior
- Not suitable for RNNs
- Issues with small batches

**Use For:** CNNs, large batch training, feedforward networks

### Layer Normalization
**Formula:** y = γ((x - μ_L) / √(σ²_L + ε)) + β

✅ **Pros:**
- Batch size independent
- Works with batch size = 1
- Perfect for RNNs/Transformers
- Consistent behavior

❌ **Cons:**
- Slightly slower than BatchNorm
- Less effective for CNNs

**Use For:** Transformers (BERT, GPT), RNNs, small batches, sequences

### Group Normalization
**Formula:** Divides channels into groups, normalizes within groups

✅ **Pros:**
- Batch size independent
- Good for small batches
- Works well for CNNs

❌ **Cons:**
- Requires tuning num_groups
- Slightly more complex

**Use For:** Small batch training, object detection, segmentation

### Instance Normalization
**Formula:** Normalizes each sample independently

✅ **Pros:**
- Per-sample normalization
- Perfect for style transfer
- Good for GANs

❌ **Cons:**
- No batch statistics
- Less regularization

**Use For:** Style transfer, GANs, image-to-image translation

### Weight Normalization
**Formula:** w = g * (v / ||v||)

✅ **Pros:**
- Faster than BatchNorm
- Decouples magnitude & direction
- No batch dependency

❌ **Cons:**
- Less common
- Requires careful initialization

**Use For:** RNNs, faster training, research

## 🔧 Advanced Usage

### CNN with Batch Normalization

```python
from ilovetools.ml.normalization import batch_normalization
import numpy as np

# Convolutional layer output
x = np.random.randn(32, 64, 28, 28)  # (N, C, H, W)
gamma = np.ones(64)
beta = np.zeros(64)

# Initialize running statistics
running_mean = np.zeros(64)
running_var = np.ones(64)

# Training mode
out, running_mean, running_var = batch_normalization(
    x, gamma, beta, running_mean, running_var,
    training=True, momentum=0.9
)

# Inference mode
out_test, _, _ = batch_normalization(
    x, gamma, beta, running_mean, running_var,
    training=False
)
```

### Transformer with Layer Normalization

```python
from ilovetools.ml.normalization import layer_normalization

# Transformer layer output
x = np.random.randn(32, 512, 768)  # (batch, seq_len, hidden_dim)
gamma = np.ones(768)
beta = np.zeros(768)

# Apply layer norm (same for train and test)
out = layer_normalization(x, gamma, beta, epsilon=1e-5)

# No running statistics needed!
```

### Small Batch Training with Group Normalization

```python
from ilovetools.ml.normalization import group_normalization

# Small batch (e.g., batch_size=4)
x = np.random.randn(4, 64, 32, 32)
gamma = np.ones(64)
beta = np.zeros(64)

# Divide 64 channels into 32 groups
out = group_normalization(x, gamma, beta, num_groups=32)
```

### Style Transfer with Instance Normalization

```python
from ilovetools.ml.normalization import instance_normalization

# Style transfer network
x = np.random.randn(1, 64, 256, 256)  # Single image
gamma = np.ones(64)
beta = np.zeros(64)

# Normalize each channel independently
out = instance_normalization(x, gamma, beta)
```

### Weight Normalization for RNNs

```python
from ilovetools.ml.normalization import weight_normalization

# RNN weight matrix
W = np.random.randn(512, 256)

# Normalize weights
W_norm, g = weight_normalization(W, axis=0)

# Use W_norm in forward pass
# Magnitude g can be learned separately
```

## 💡 Pro Tips

✅ **Use BatchNorm for CNNs** - Best performance for computer vision  
✅ **Use LayerNorm for Transformers** - BERT, GPT standard  
✅ **Place after linear/conv layers** - Before activation functions  
✅ **Tune momentum parameter** - 0.9 is typical for BatchNorm  
✅ **Use running statistics for inference** - BatchNorm only  
✅ **GroupNorm for small batches** - When batch size < 16  

❌ **Don't use BatchNorm with small batches** - Use GroupNorm instead  
❌ **Don't use LayerNorm for CNNs** - BatchNorm is better  
❌ **Don't forget to set training mode** - BatchNorm behavior differs  
❌ **Don't normalize inputs twice** - Once is enough  

## 🔬 Technical Details

### Normalization Properties

All normalization functions:
- ✅ Vectorized (NumPy)
- ✅ Numerically stable (epsilon for division)
- ✅ Support both training and inference
- ✅ Fully documented with examples
- ✅ Tested with 100% coverage

### Common Patterns

```python
# CNN Block
conv_out = conv2d(x, W)
bn_out = batch_normalization(conv_out, gamma, beta, training=True)
activated = relu(bn_out)

# Transformer Block
attn_out = multi_head_attention(x)
ln_out = layer_normalization(attn_out, gamma, beta)
ffn_out = feedforward(ln_out)

# ResNet Block with GroupNorm
conv_out = conv2d(x, W)
gn_out = group_normalization(conv_out, gamma, beta, num_groups=32)
activated = relu(gn_out)
```

## 🔗 Links

- **PyPI:** https://pypi.org/project/ilovetools/
- **GitHub:** https://github.com/AliMehdi512/ilovetools
- **Normalization Code:** https://github.com/AliMehdi512/ilovetools/blob/main/ilovetools/ml/normalization.py
- **Tests:** https://github.com/AliMehdi512/ilovetools/blob/main/tests/test_normalization.py
- **Verification:** https://github.com/AliMehdi512/ilovetools/blob/main/scripts/verify_normalization.py

## 📈 Total ML Functions

- **Previous (v0.2.16):** 248+ functions
- **New (v0.2.17):** **262+ functions** (14+ new normalization functions!)

## 🎓 Educational Content

Check out our LinkedIn posts:
- **Normalization Guide:** https://www.linkedin.com/feed/update/urn:li:share:7410522180330983424
- **Loss Functions:** https://www.linkedin.com/feed/update/urn:li:share:7410204146366189569
- **Activation Functions:** https://www.linkedin.com/feed/update/urn:li:share:7409972257818775552

## 📚 Research Papers

These techniques are based on:
- **Batch Normalization:** Ioffe & Szegedy (2015)
- **Layer Normalization:** Ba et al. (2016)
- **Group Normalization:** Wu & He (2018)
- **Instance Normalization:** Ulyanov et al. (2016)
- **Weight Normalization:** Salimans & Kingma (2016)

## 🚀 What's Next

Coming in future releases:
- Spectral Normalization
- Adaptive Instance Normalization
- Switchable Normalization
- Filter Response Normalization

## 🙏 Thank You

Thank you for using ilovetools! We're committed to providing the best ML utilities for Python developers.

## 📝 Version History

- **v0.2.17** (Dec 27, 2024): ✅ Normalization techniques module
- **v0.2.16** (Dec 25, 2024): Advanced optimizers module
- **v0.2.15** (Dec 25, 2024): Activation functions module
- **v0.2.14** (Dec 21, 2024): Loss functions module
- **v0.2.13** (Dec 20, 2024): Regularization techniques

---

**Normalize Better, Train Faster! 🎯**
