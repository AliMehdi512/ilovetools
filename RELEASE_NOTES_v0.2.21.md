# Release Notes - v0.2.21

**Release Date:** December 31, 2024

## 🎯 NEW: Advanced Normalization Techniques Module

Added complete normalization suite - essential for stable deep learning training!

### 📦 What's New

#### Batch Normalization (1 function)
1. **batch_norm_forward()** - Normalizes across batch dimension

#### Layer Normalization (1 function)
2. **layer_norm_forward()** - Normalizes across feature dimension

#### Instance Normalization (1 function)
3. **instance_norm_forward()** - Normalizes per sample, per channel

#### Group Normalization (1 function)
4. **group_norm_forward()** - Normalizes within channel groups

#### Weight Normalization (1 function)
5. **weight_norm()** - Decouples weight magnitude and direction

#### Spectral Normalization (1 function)
6. **spectral_norm()** - Normalizes by largest singular value

#### Utilities (2 functions)
7. **initialize_norm_params()** - Initialize γ and β
8. **compute_norm_stats()** - Compute normalization statistics

#### Aliases (4 shortcuts)
9. **batch_norm**, **layer_norm**, **instance_norm**, **group_norm**

## 💻 Installation

```bash
pip install --upgrade ilovetools
```

## ✅ Quick Start

```python
from ilovetools.ml.normalization_advanced import (
    batch_norm_forward,
    layer_norm_forward,
    instance_norm_forward,
    group_norm_forward,
    initialize_norm_params
)
import numpy as np

# Batch Normalization (CNNs)
x = np.random.randn(32, 64, 28, 28)  # (batch, channels, H, W)
gamma, beta = initialize_norm_params(64)

output, running_mean, running_var = batch_norm_forward(
    x, gamma, beta, training=True
)
print(f"BatchNorm: {x.shape} -> {output.shape}")  # (32, 64, 28, 28)

# Layer Normalization (Transformers)
x = np.random.randn(32, 10, 512)  # (batch, seq_len, features)
gamma, beta = initialize_norm_params(512)

output = layer_norm_forward(x, gamma, beta)
print(f"LayerNorm: {x.shape} -> {output.shape}")  # (32, 10, 512)

# Instance Normalization (Style Transfer)
x = np.random.randn(32, 64, 28, 28)
gamma, beta = initialize_norm_params(64)

output = instance_norm_forward(x, gamma, beta)
print(f"InstanceNorm: {x.shape} -> {output.shape}")  # (32, 64, 28, 28)

# Group Normalization (Small Batches)
output = group_norm_forward(x, gamma, beta, num_groups=32)
print(f"GroupNorm: {x.shape} -> {output.shape}")  # (32, 64, 28, 28)
```

## 🔧 Advanced Usage

### CNN with Batch Normalization

```python
from ilovetools.ml.cnn import conv2d
from ilovetools.ml.normalization_advanced import batch_norm_forward
from ilovetools.ml.activations import relu

# Input
x = np.random.randn(32, 3, 224, 224)

# Conv -> BatchNorm -> ReLU
kernel = np.random.randn(64, 3, 3, 3)
conv_out = conv2d(x, kernel, stride=1, padding='same')

gamma, beta = initialize_norm_params(64)
bn_out, _, _ = batch_norm_forward(conv_out, gamma, beta, training=True)

relu_out = relu(bn_out)
print(f"CNN Block: {x.shape} -> {relu_out.shape}")
```

### Transformer with Layer Normalization

```python
from ilovetools.ml.attention import multi_head_self_attention
from ilovetools.ml.normalization_advanced import layer_norm_forward

# Input
x = np.random.randn(32, 10, 512)

# Multi-head attention
attn_out, _ = multi_head_self_attention(x, num_heads=8, d_model=512)

# Add & Norm
x = x + attn_out
gamma, beta = initialize_norm_params(512)
x = layer_norm_forward(x, gamma, beta)

print(f"Transformer block: {x.shape}")
```

### Style Transfer with Instance Normalization

```python
# Content and style images
content = np.random.randn(1, 64, 256, 256)
style = np.random.randn(1, 64, 256, 256)

gamma, beta = initialize_norm_params(64)

# Normalize each independently
content_norm = instance_norm_forward(content, gamma, beta)
style_norm = instance_norm_forward(style, gamma, beta)
```

## 💡 Pro Tips

✅ **Use BatchNorm for CNNs** - Faster training, higher learning rates  
✅ **Use LayerNorm for Transformers** - Batch size independent  
✅ **Use GroupNorm for small batches** - Works with batch size = 1  
✅ **Place after linear/conv** - Before activation usually  
✅ **Initialize γ=1, β=0** - Standard initialization  
✅ **Train/eval mode** - Different behavior for BatchNorm  

❌ **Don't use BatchNorm with batch=1** - Use LayerNorm or GroupNorm  
❌ **Don't forget train/eval mode** - BatchNorm behaves differently  
❌ **Don't use BatchNorm in RNNs** - Use LayerNorm instead  

## 📊 Comparison

### Normalization Techniques

| Technique | Axis | Batch Dependent | Use Case |
|-----------|------|-----------------|----------|
| Batch Norm | Batch, H, W | Yes | CNNs |
| Layer Norm | Features | No | Transformers, RNNs |
| Instance Norm | H, W | No | Style transfer |
| Group Norm | Groups | No | Small batches |

### Benefits

**Batch Normalization:**
- 2-10x faster training
- Higher learning rates possible
- Reduces internal covariate shift
- Acts as regularization

**Layer Normalization:**
- Batch size independent
- Same behavior train/test
- Perfect for RNNs/Transformers
- Works with batch size = 1

**Instance Normalization:**
- Per-sample normalization
- Style transfer
- Image generation

**Group Normalization:**
- Good for small batches
- Object detection
- Semantic segmentation

## 🔗 Links

- **PyPI:** https://pypi.org/project/ilovetools/
- **GitHub:** https://github.com/AliMehdi512/ilovetools
- **Code:** https://github.com/AliMehdi512/ilovetools/blob/main/ilovetools/ml/normalization_advanced.py
- **Tests:** https://github.com/AliMehdi512/ilovetools/blob/main/tests/test_normalization_advanced.py
- **Verification:** https://github.com/AliMehdi512/ilovetools/blob/main/scripts/verify_normalization_advanced.py

## 📈 Total ML Functions

- **Previous (v0.2.20):** 316+ functions
- **New (v0.2.21):** **328+ functions** (12+ new normalization functions!)

## 🎓 Educational Content

Check out our LinkedIn posts:
- **Normalization Guide:** https://www.linkedin.com/feed/update/urn:li:share:7412548761010409472
- **RNNs:** https://www.linkedin.com/feed/update/urn:li:share:7411784681609895937
- **CNNs:** https://www.linkedin.com/feed/update/urn:li:share:7411263336107036672

## 📚 Research Papers

- **Batch Normalization:** Ioffe & Szegedy (2015)
- **Layer Normalization:** Ba et al. (2016)
- **Instance Normalization:** Ulyanov et al. (2016)
- **Group Normalization:** Wu & He (2018)
- **Weight Normalization:** Salimans & Kingma (2016)
- **Spectral Normalization:** Miyato et al. (2018)

## 🚀 What's Next

Coming in future releases:
- Adaptive normalization
- Conditional normalization
- Switchable normalization

## 📝 Version History

- **v0.2.21** (Dec 31, 2024): ✅ Advanced normalization techniques module
- **v0.2.20** (Dec 30, 2024): RNN operations module
- **v0.2.19** (Dec 29, 2024): CNN operations module
- **v0.2.18** (Dec 28, 2024): Attention mechanisms module

---

**Normalize, Stabilize, Train Faster! 🎯**
