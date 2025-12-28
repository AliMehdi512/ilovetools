# Release Notes - v0.2.18

**Release Date:** December 28, 2024

## 🎯 NEW: Attention Mechanisms Module

Added complete attention mechanisms suite - the foundation of Transformers, BERT, and GPT!

### 📦 What's New

#### Core Attention Functions (5 functions)
1. **scaled_dot_product_attention()** - Fundamental attention mechanism
2. **multi_head_attention()** - Multiple attention heads in parallel
3. **self_attention()** - Q, K, V from same input (BERT, GPT)
4. **multi_head_self_attention()** - Multi-head + self-attention
5. **cross_attention()** - Attention between different sequences

#### Attention Masks (3 functions)
6. **create_padding_mask()** - Mask padding tokens
7. **create_causal_mask()** - Prevent looking ahead (GPT)
8. **create_look_ahead_mask()** - Alias for causal mask

#### Positional Encoding (2 functions)
9. **positional_encoding()** - Sinusoidal position encoding
10. **learned_positional_encoding()** - Learnable positions

#### Utilities (3 functions)
11. **softmax()** - Numerically stable softmax
12. **dropout()** - Dropout for attention weights
13. **attention_score_visualization()** - Prepare attention for visualization

#### Aliases (7 shortcuts)
14. **sdp_attention**, **mha**, **self_attn**, **cross_attn**, **pos_encoding**, **causal_mask**, **padding_mask**

## 💻 Installation

```bash
pip install --upgrade ilovetools
```

## ✅ Quick Start

```python
from ilovetools.ml.attention import (
    scaled_dot_product_attention,
    multi_head_attention,
    self_attention,
    cross_attention,
    positional_encoding,
    create_causal_mask
)
import numpy as np

# Scaled Dot-Product Attention
q = np.random.randn(32, 10, 64)  # (batch, seq_len, d_k)
k = np.random.randn(32, 10, 64)
v = np.random.randn(32, 10, 64)

output, weights = scaled_dot_product_attention(q, k, v)
print(f"Output: {output.shape}")  # (32, 10, 64)
print(f"Weights: {weights.shape}")  # (32, 10, 10)
print(f"Weights sum to 1: {np.allclose(np.sum(weights, axis=-1), 1.0)}")

# Multi-Head Attention (Transformers)
output, weights = multi_head_attention(
    q, k, v, num_heads=8, d_model=64
)
print(f"Multi-head output: {output.shape}")  # (32, 10, 64)
print(f"Multi-head weights: {weights.shape}")  # (32, 8, 10, 10)

# Self-Attention (BERT, GPT)
x = np.random.randn(32, 10, 512)
output, weights = self_attention(x, d_model=512)
print(f"Self-attention: {x.shape} -> {output.shape}")

# Cross-Attention (Encoder-Decoder)
query = np.random.randn(32, 10, 512)  # Decoder
context = np.random.randn(32, 20, 512)  # Encoder
output, weights = cross_attention(query, context, d_model=512)
print(f"Cross-attention: query {query.shape}, context {context.shape}")
print(f"Output: {output.shape}")  # (32, 10, 512)

# Positional Encoding
pos_enc = positional_encoding(seq_len=100, d_model=512)
print(f"Positional encoding: {pos_enc.shape}")  # (100, 512)

# Add to embeddings
embeddings = np.random.randn(32, 100, 512)
embeddings_with_pos = embeddings + pos_enc

# Causal Mask (for GPT-style models)
mask = create_causal_mask(seq_len=10)
output, weights = scaled_dot_product_attention(q, k, v, mask=mask)
print("Causal attention applied!")
```

## 🎯 Attention Types

### 1. Scaled Dot-Product Attention

**Formula:** Attention(Q, K, V) = softmax(QK^T / √d_k) × V

**Use Cases:**
- Foundation of all attention mechanisms
- Used in every Transformer layer
- Enables parallel processing

**Example:**
```python
# Basic attention
q = np.random.randn(32, 10, 64)
k = np.random.randn(32, 10, 64)
v = np.random.randn(32, 10, 64)

output, weights = scaled_dot_product_attention(q, k, v)
```

### 2. Multi-Head Attention

**Formula:** MultiHead(Q,K,V) = Concat(head₁,...,headₕ)W^O

**Use Cases:**
- Standard in Transformers
- BERT uses 12 heads
- GPT uses 12-96 heads
- Attends to different representation subspaces

**Example:**
```python
# 8 attention heads
output, weights = multi_head_attention(
    q, k, v, num_heads=8, d_model=512
)
```

### 3. Self-Attention

**Use Cases:**
- BERT (bidirectional)
- GPT (causal)
- Vision Transformers
- Q, K, V all from same input

**Example:**
```python
# Self-attention
x = np.random.randn(32, 10, 512)
output, weights = self_attention(x, d_model=512)
```

### 4. Cross-Attention

**Use Cases:**
- Machine translation (decoder → encoder)
- Image captioning (text → image)
- CLIP (vision ↔ language)
- Multimodal models

**Example:**
```python
# Decoder attending to encoder
decoder_out = np.random.randn(32, 10, 512)
encoder_out = np.random.randn(32, 20, 512)
output, weights = cross_attention(decoder_out, encoder_out, d_model=512)
```

## 🔧 Advanced Usage

### Building a Transformer Block

```python
from ilovetools.ml.attention import (
    multi_head_self_attention,
    positional_encoding,
    create_causal_mask
)
from ilovetools.ml.normalization import layer_normalization

# Input
x = np.random.randn(32, 10, 512)  # (batch, seq_len, d_model)

# Add positional encoding
pos_enc = positional_encoding(seq_len=10, d_model=512)
x = x + pos_enc

# Multi-head self-attention
attn_output, attn_weights = multi_head_self_attention(
    x, num_heads=8, d_model=512
)

# Residual connection + layer norm
x = layer_normalization(x + attn_output, gamma=np.ones(512), beta=np.zeros(512))

# Feed-forward network would go here
# ...

print(f"Transformer block output: {x.shape}")
```

### GPT-Style Causal Attention

```python
from ilovetools.ml.attention import (
    scaled_dot_product_attention,
    create_causal_mask,
    positional_encoding
)

# Input sequence
x = np.random.randn(32, 10, 512)

# Add positional encoding
pos_enc = positional_encoding(seq_len=10, d_model=512)
x = x + pos_enc

# Create causal mask (prevent looking ahead)
mask = create_causal_mask(seq_len=10)

# Apply causal attention
output, weights = scaled_dot_product_attention(x, x, x, mask=mask)

print("GPT-style causal attention applied!")
print(f"Position 0 can only attend to position 0")
print(f"Position 5 can attend to positions 0-5")
```

### BERT-Style Bidirectional Attention

```python
from ilovetools.ml.attention import (
    multi_head_self_attention,
    create_padding_mask,
    positional_encoding
)

# Input with padding
seq = np.array([[1, 2, 3, 4, 0, 0], [1, 2, 3, 0, 0, 0]])  # Token IDs
x = np.random.randn(2, 6, 768)  # Embeddings

# Add positional encoding
pos_enc = positional_encoding(seq_len=6, d_model=768)
x = x + pos_enc

# Create padding mask
mask = create_padding_mask(seq, pad_token=0)

# Apply bidirectional attention
output, weights = multi_head_self_attention(
    x, num_heads=12, d_model=768
)

print("BERT-style bidirectional attention applied!")
```

### Encoder-Decoder with Cross-Attention

```python
from ilovetools.ml.attention import (
    multi_head_self_attention,
    cross_attention,
    create_causal_mask
)

# Encoder (source sequence)
encoder_input = np.random.randn(32, 20, 512)
encoder_output, _ = multi_head_self_attention(
    encoder_input, num_heads=8, d_model=512
)

# Decoder (target sequence)
decoder_input = np.random.randn(32, 10, 512)

# Decoder self-attention (causal)
mask = create_causal_mask(seq_len=10)
decoder_self_attn, _ = multi_head_self_attention(
    decoder_input, num_heads=8, d_model=512
)

# Cross-attention (decoder → encoder)
decoder_output, cross_weights = cross_attention(
    decoder_self_attn, encoder_output, d_model=512
)

print(f"Encoder-decoder translation complete!")
print(f"Cross-attention weights: {cross_weights.shape}")  # (32, 10, 20)
```

## 💡 Pro Tips

✅ **Use scaled dot-product** - Prevents gradient issues  
✅ **Multi-head for Transformers** - 8-16 heads typical  
✅ **Add positional encoding** - Attention has no position info  
✅ **Use causal mask for GPT** - Prevents looking ahead  
✅ **Use padding mask for BERT** - Ignore padding tokens  
✅ **Visualize attention weights** - Interpretable!  

❌ **Don't forget scaling** - QK^T can be very large  
❌ **Don't skip positional encoding** - Order matters  
❌ **Don't use wrong mask** - Causal vs padding  
❌ **Don't ignore O(n²) complexity** - Long sequences expensive  

## 📊 Complexity Analysis

### Time Complexity
- **Attention:** O(n² × d) where n = sequence length, d = dimension
- **Multi-head:** O(n² × d) (parallelizable)
- **Positional encoding:** O(n × d)

### Space Complexity
- **Attention weights:** O(n²) per head
- **Multi-head:** O(h × n²) where h = number of heads
- **Activations:** O(n × d)

### Optimization Strategies
- **Sparse attention:** Reduce to O(n × √n)
- **Linear attention:** Reduce to O(n × d²)
- **Local attention:** Sliding window
- **Flash attention:** Memory-efficient

## 🔗 Links

- **PyPI:** https://pypi.org/project/ilovetools/
- **GitHub:** https://github.com/AliMehdi512/ilovetools
- **Attention Code:** https://github.com/AliMehdi512/ilovetools/blob/main/ilovetools/ml/attention.py
- **Tests:** https://github.com/AliMehdi512/ilovetools/blob/main/tests/test_attention.py
- **Verification:** https://github.com/AliMehdi512/ilovetools/blob/main/scripts/verify_attention.py

## 📈 Total ML Functions

- **Previous (v0.2.17):** 262+ functions
- **New (v0.2.18):** **283+ functions** (21+ new attention functions!)

## 🎓 Educational Content

Check out our LinkedIn posts:
- **Attention Mechanisms:** https://www.linkedin.com/feed/update/urn:li:share:7410931572444385280
- **Normalization Guide:** https://www.linkedin.com/feed/update/urn:li:share:7410522180330983424
- **Loss Functions:** https://www.linkedin.com/feed/update/urn:li:share:7410204146366189569

## 📚 Research Papers

These mechanisms are based on:
- **Attention Is All You Need:** Vaswani et al. (2017)
- **BERT:** Devlin et al. (2018)
- **GPT:** Radford et al. (2018)
- **Vision Transformers:** Dosovitskiy et al. (2020)
- **CLIP:** Radford et al. (2021)

## 🚀 What's Next

Coming in future releases:
- Sparse attention mechanisms
- Linear attention (Performer)
- Flash attention
- Relative positional encoding

## 🙏 Thank You

Thank you for using ilovetools! We're committed to providing the best ML utilities for Python developers.

## 📝 Version History

- **v0.2.18** (Dec 28, 2024): ✅ Attention mechanisms module
- **v0.2.17** (Dec 27, 2024): Normalization techniques module
- **v0.2.16** (Dec 25, 2024): Advanced optimizers module
- **v0.2.15** (Dec 25, 2024): Activation functions module
- **v0.2.14** (Dec 21, 2024): Loss functions module

---

**Attention Is All You Need! 🎯**
