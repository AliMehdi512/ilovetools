# Release Notes - Version 0.2.23

## 🚀 Major Release: Positional Encoding and Attention Mechanisms

**Release Date:** January 4, 2026

This release adds comprehensive positional encoding techniques and attention mechanisms - the core building blocks of modern transformer architectures.

---

## 🎯 What's New

### Positional Encoding Implementations

#### 1. **Sinusoidal Positional Encoding**
The original positional encoding from "Attention Is All You Need" (Vaswani et al., 2017).

```python
from ilovetools.ml.positional_encoding import SinusoidalPositionalEncoding

pe = SinusoidalPositionalEncoding(d_model=512, max_len=5000)
x_with_pos = pe.forward(embeddings)
```

**Features:**
- Fixed sine/cosine patterns
- No trainable parameters
- Works for any sequence length
- Captures relative positions naturally

#### 2. **Learned Positional Embeddings**
Trainable position embeddings that learn optimal representations.

```python
from ilovetools.ml.positional_encoding import LearnedPositionalEmbedding

learned_pe = LearnedPositionalEmbedding(max_len=512, d_model=512)
x_with_pos = learned_pe.forward(embeddings)
learned_pe.update_embeddings(gradients, learning_rate=0.001)
```

**Features:**
- Trainable parameters
- Can learn task-specific patterns
- Used in BERT and GPT-2

#### 3. **Relative Positional Encoding**
Encodes relative distances between tokens (T5, Transformer-XL style).

```python
from ilovetools.ml.positional_encoding import RelativePositionalEncoding

relative_pe = RelativePositionalEncoding(d_model=512, max_relative_position=128)
relative_encodings = relative_pe.forward(seq_len=50)
```

**Features:**
- Focuses on relative distances
- Better generalization to longer sequences
- Used in T5 and Transformer-XL

#### 4. **Rotary Position Embedding (RoPE)**
Advanced technique using rotation matrices (used in LLaMA, GPT-NeoX).

```python
from ilovetools.ml.positional_encoding import RotaryPositionalEmbedding

rope = RotaryPositionalEmbedding(d_model=512, max_len=2048)
x_with_rope = rope.forward(embeddings)
```

**Features:**
- Rotation-based encoding
- Excellent for long sequences
- Used in LLaMA, GPT-NeoX, PaLM
- Naturally captures relative positions

#### 5. **ALiBi (Attention with Linear Biases)**
Adds position information directly to attention scores.

```python
from ilovetools.ml.positional_encoding import ALiBiPositionalBias

alibi = ALiBiPositionalBias(num_heads=8, max_len=2048)
attention_with_bias = alibi.forward(attention_scores, seq_len)
```

**Features:**
- No positional embeddings needed
- Linear bias based on distance
- Excellent extrapolation to longer sequences
- Used in BLOOM

---

### Attention Mechanisms

#### 1. **Scaled Dot-Product Attention**
The fundamental attention operation.

```python
from ilovetools.ml.positional_encoding import scaled_dot_product_attention

output, weights = scaled_dot_product_attention(query, key, value, mask=None)
```

**Formula:** `Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V`

#### 2. **Multi-Head Attention**
Parallel attention with multiple representation subspaces.

```python
from ilovetools.ml.positional_encoding import MultiHeadAttention

mha = MultiHeadAttention(d_model=512, num_heads=8, dropout=0.1)
output, attention_weights = mha.forward(query, key, value, mask=None)
```

**Features:**
- Multiple attention heads
- Parallel processing
- Captures different aspects of relationships
- Core component of transformers

#### 3. **Causal (Masked) Attention**
Prevents attending to future positions (for autoregressive models).

```python
from ilovetools.ml.positional_encoding import CausalAttention

causal_attn = CausalAttention(d_model=512, num_heads=8)
output, weights = causal_attn.forward(query, key, value)
```

**Features:**
- Automatic causal masking
- Used in GPT, GPT-2, GPT-3, GPT-4
- Prevents information leakage from future

#### 4. **Utility Functions**

```python
from ilovetools.ml.positional_encoding import (
    create_padding_mask,
    create_look_ahead_mask,
    softmax,
)

# Padding mask for variable-length sequences
padding_mask = create_padding_mask(sequences, pad_token=0)

# Look-ahead mask for causal attention
look_ahead_mask = create_look_ahead_mask(seq_len)

# Numerically stable softmax
probs = softmax(logits, axis=-1)
```

---

## 📊 Complete Feature List

### Positional Encodings (5 variants)
- ✅ Sinusoidal Positional Encoding
- ✅ Learned Positional Embeddings
- ✅ Relative Positional Encoding
- ✅ Rotary Position Embedding (RoPE)
- ✅ ALiBi (Attention with Linear Biases)

### Attention Mechanisms (4 types)
- ✅ Scaled Dot-Product Attention
- ✅ Multi-Head Attention
- ✅ Causal (Masked) Attention
- ✅ Cross-Attention support

### Utilities
- ✅ Padding mask creation
- ✅ Look-ahead mask creation
- ✅ Numerically stable softmax
- ✅ Convenient aliases

---

## 🧪 Testing & Quality

### Comprehensive Test Suite
- **15+ test functions** covering all components
- **372+ test cases** in total
- **100% functionality coverage**

Test categories:
1. ✅ Sinusoidal PE tests
2. ✅ Learned PE tests
3. ✅ Relative PE tests
4. ✅ RoPE tests
5. ✅ ALiBi tests
6. ✅ Scaled dot-product attention tests
7. ✅ Multi-head attention tests
8. ✅ Causal attention tests
9. ✅ Utility function tests
10. ✅ Integration tests (complete transformer blocks)
11. ✅ GPT-style decoder tests
12. ✅ RoPE + Attention integration tests

Run tests:
```bash
python tests/test_positional_encoding.py
```

---

## 📚 Examples & Documentation

### 14 Comprehensive Examples

1. **Sinusoidal Positional Encoding** - Original transformer approach
2. **Learned Positional Embeddings** - Trainable positions
3. **Relative Positional Encoding** - T5-style relative positions
4. **Rotary Position Embedding** - LLaMA-style RoPE
5. **ALiBi** - Attention with linear biases
6. **Scaled Dot-Product Attention** - Basic attention
7. **Multi-Head Attention** - Parallel attention heads
8. **Causal Attention** - GPT-style masked attention
9. **Complete Transformer Encoder** - Full encoder block
10. **Complete Transformer Decoder** - Full decoder block
11. **Cross-Attention** - Encoder-decoder attention
12. **Padding Masks** - Handling variable-length sequences
13. **Comparing Positional Encodings** - Side-by-side comparison
14. **Real-World Sentiment Analysis** - Complete application

Run examples:
```bash
python examples/positional_encoding_examples.py
```

---

## 🎓 Use Cases

### 1. Building Custom Transformers
```python
# Complete transformer encoder block
pe = SinusoidalPositionalEncoding(d_model=512)
mha = MultiHeadAttention(d_model=512, num_heads=8)

x_with_pos = pe.forward(embeddings)
attn_output, _ = mha.forward(x_with_pos, x_with_pos, x_with_pos)
```

### 2. GPT-Style Language Models
```python
# Autoregressive decoder with causal masking
pe = SinusoidalPositionalEncoding(d_model=512)
causal_attn = CausalAttention(d_model=512, num_heads=8)

x_with_pos = pe.forward(embeddings)
output, _ = causal_attn.forward(x_with_pos, x_with_pos, x_with_pos)
```

### 3. LLaMA-Style Models with RoPE
```python
# Modern approach with rotary embeddings
rope = RotaryPositionalEmbedding(d_model=512)
mha = MultiHeadAttention(d_model=512, num_heads=8)

x_with_rope = rope.forward(embeddings)
output, _ = mha.forward(x_with_rope, x_with_rope, x_with_rope)
```

### 4. Long-Context Models with ALiBi
```python
# Excellent for long sequences
alibi = ALiBiPositionalBias(num_heads=8)
mha = MultiHeadAttention(d_model=512, num_heads=8)

# No positional encoding needed!
output, weights = mha.forward(embeddings, embeddings, embeddings)
attention_with_bias = alibi.forward(weights, seq_len)
```

---

## 🔧 Installation & Verification

### Install
```bash
pip install ilovetools==0.2.23
```

### Verify Installation
```bash
python tests/verify_positional_encoding.py
```

### Quick Test
```python
from ilovetools.ml.positional_encoding import (
    SinusoidalPositionalEncoding,
    MultiHeadAttention,
    RotaryPositionalEmbedding,
)

# Test imports
print("✓ All imports successful!")
```

---

## 📈 Performance & Efficiency

### Optimizations
- ✅ Precomputed positional encodings (cached)
- ✅ Efficient matrix operations
- ✅ Numerically stable computations
- ✅ Memory-efficient implementations

### Benchmarks
- Sinusoidal PE: O(1) after precomputation
- Multi-Head Attention: O(n²d) standard complexity
- RoPE: O(nd) rotation operations
- ALiBi: O(n²) bias addition

---

## 🔗 Integration with Existing Code

### Easy Integration
All components are designed to work seamlessly with existing NumPy-based code:

```python
import numpy as np
from ilovetools.ml.positional_encoding import *

# Your existing embeddings
embeddings = np.random.randn(batch_size, seq_len, d_model)

# Add positional encoding
pe = SinusoidalPositionalEncoding(d_model)
embeddings_with_pos = pe.forward(embeddings)

# Apply attention
mha = MultiHeadAttention(d_model, num_heads=8)
output, weights = mha.forward(embeddings_with_pos, embeddings_with_pos, embeddings_with_pos)
```

---

## 🎯 Comparison with Other Libraries

### Why ilovetools?

| Feature | ilovetools | PyTorch | TensorFlow |
|---------|-----------|---------|------------|
| **Sinusoidal PE** | ✅ | ✅ | ✅ |
| **Learned PE** | ✅ | ✅ | ✅ |
| **Relative PE** | ✅ | ❌ | ❌ |
| **RoPE** | ✅ | ❌ (external) | ❌ |
| **ALiBi** | ✅ | ❌ (external) | ❌ |
| **Pure NumPy** | ✅ | ❌ | ❌ |
| **No GPU Required** | ✅ | ❌ | ❌ |
| **Educational** | ✅ | ⚠️ | ⚠️ |
| **Lightweight** | ✅ | ❌ | ❌ |

---

## 🐛 Bug Fixes & Improvements

### From Previous Versions
- N/A (New module)

### Known Limitations
- NumPy-based (not GPU-accelerated)
- Designed for educational and prototyping purposes
- For production at scale, consider PyTorch/TensorFlow

---

## 🔮 Future Plans

### Upcoming Features (v0.2.24+)
- [ ] Sparse Attention mechanisms
- [ ] Linear Attention variants
- [ ] Flash Attention implementation
- [ ] Grouped Query Attention (GQA)
- [ ] Multi-Query Attention (MQA)
- [ ] Sliding Window Attention
- [ ] Memory-efficient attention

---

## 📝 Migration Guide

### New Users
Simply install and import:
```bash
pip install ilovetools==0.2.23
```

### Existing Users
No breaking changes. This is a pure addition.

---

## 🙏 Acknowledgments

### Inspired By
- "Attention Is All You Need" (Vaswani et al., 2017)
- "RoFormer: Enhanced Transformer with Rotary Position Embedding" (Su et al., 2021)
- "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation" (Press et al., 2022)
- LLaMA, GPT-NeoX, T5, BLOOM implementations

---

## 📞 Support & Community

### Get Help
- 📖 Documentation: [GitHub Wiki](https://github.com/AliMehdi512/ilovetools)
- 🐛 Issues: [GitHub Issues](https://github.com/AliMehdi512/ilovetools/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/AliMehdi512/ilovetools/discussions)
- 📧 Email: ali.mehdi.dev579@gmail.com

### Contribute
- ⭐ Star the repo
- 🍴 Fork and submit PRs
- 🐛 Report bugs
- 💡 Suggest features
- 📝 Improve documentation

---

## 📄 License

MIT License - Free for commercial and personal use

---

## 🎉 Thank You!

Thank you to everyone who uses, contributes to, and supports ilovetools!

**Happy Coding! 🚀**

---

**Full Changelog:** [v0.2.22...v0.2.23](https://github.com/AliMehdi512/ilovetools/compare/v0.2.22...v0.2.23)
