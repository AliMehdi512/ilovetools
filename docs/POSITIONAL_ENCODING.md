# Positional Encoding and Attention Mechanisms

Complete implementation of positional encoding techniques and attention mechanisms for building transformer models.

## 📚 Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Positional Encodings](#positional-encodings)
- [Attention Mechanisms](#attention-mechanisms)
- [Examples](#examples)
- [API Reference](#api-reference)
- [Performance](#performance)
- [Contributing](#contributing)

## 🎯 Overview

This module provides production-ready implementations of:

### Positional Encodings (5 variants)
1. **Sinusoidal Positional Encoding** - Original Transformer (Vaswani et al., 2017)
2. **Learned Positional Embeddings** - Trainable positions (BERT, GPT-2)
3. **Relative Positional Encoding** - Relative distances (T5, Transformer-XL)
4. **Rotary Position Embedding (RoPE)** - Rotation-based (LLaMA, GPT-NeoX)
5. **ALiBi** - Attention with Linear Biases (BLOOM)

### Attention Mechanisms (4 types)
1. **Scaled Dot-Product Attention** - Basic attention operation
2. **Multi-Head Attention** - Parallel attention heads
3. **Causal Attention** - Masked attention for autoregressive models
4. **Cross-Attention** - Encoder-decoder attention

## 📦 Installation

```bash
pip install ilovetools==0.2.23
```

## 🚀 Quick Start

### Basic Usage

```python
import numpy as np
from ilovetools.ml.positional_encoding import (
    SinusoidalPositionalEncoding,
    MultiHeadAttention,
)

# Create embeddings
batch_size, seq_len, d_model = 32, 50, 512
embeddings = np.random.randn(batch_size, seq_len, d_model)

# Add positional encoding
pe = SinusoidalPositionalEncoding(d_model=512, max_len=5000)
embeddings_with_pos = pe.forward(embeddings)

# Apply multi-head attention
mha = MultiHeadAttention(d_model=512, num_heads=8)
output, attention_weights = mha.forward(
    embeddings_with_pos,
    embeddings_with_pos,
    embeddings_with_pos
)

print(f"Output shape: {output.shape}")  # (32, 50, 512)
print(f"Attention weights shape: {attention_weights.shape}")  # (32, 8, 50, 50)
```

## 🔢 Positional Encodings

### 1. Sinusoidal Positional Encoding

The original positional encoding from "Attention Is All You Need".

```python
from ilovetools.ml.positional_encoding import SinusoidalPositionalEncoding

pe = SinusoidalPositionalEncoding(
    d_model=512,      # Model dimension
    max_len=5000,     # Maximum sequence length
    dropout=0.1       # Dropout rate
)

# Add to embeddings
x_with_pos = pe.forward(embeddings)

# Get encoding for specific positions
positions = np.array([0, 1, 2, 3, 4])
encodings = pe.get_encoding(positions)
```

**When to use:**
- Default choice for most transformers
- No trainable parameters
- Works for any sequence length
- Good for relative position understanding

### 2. Learned Positional Embeddings

Trainable position embeddings that learn optimal representations.

```python
from ilovetools.ml.positional_encoding import LearnedPositionalEmbedding

learned_pe = LearnedPositionalEmbedding(
    max_len=512,      # Maximum sequence length
    d_model=512       # Model dimension
)

# Add to embeddings
x_with_pos = learned_pe.forward(embeddings)

# Update during training
gradients = compute_gradients()  # Your gradient computation
learned_pe.update_embeddings(gradients, learning_rate=0.001)
```

**When to use:**
- When you have enough training data
- Task-specific position patterns
- Used in BERT, GPT-2

### 3. Relative Positional Encoding

Encodes relative distances between tokens (T5, Transformer-XL style).

```python
from ilovetools.ml.positional_encoding import RelativePositionalEncoding

relative_pe = RelativePositionalEncoding(
    d_model=512,
    max_relative_position=128  # Maximum relative distance
)

# Generate relative encodings
seq_len = 50
relative_encodings = relative_pe.forward(seq_len)
# Shape: (seq_len, seq_len, d_model)
```

**When to use:**
- Better generalization to longer sequences
- Focus on relative rather than absolute positions
- Used in T5, Transformer-XL

### 4. Rotary Position Embedding (RoPE)

Advanced technique using rotation matrices (LLaMA, GPT-NeoX).

```python
from ilovetools.ml.positional_encoding import RotaryPositionalEmbedding

rope = RotaryPositionalEmbedding(
    d_model=512,      # Must be even
    max_len=2048,     # Maximum sequence length
    base=10000.0      # Frequency base
)

# Apply RoPE
x_with_rope = rope.forward(embeddings)

# With custom positions
positions = np.arange(seq_len)
x_with_rope = rope.forward(embeddings, positions)
```

**When to use:**
- State-of-the-art for long sequences
- Used in LLaMA, GPT-NeoX, PaLM
- Excellent relative position modeling
- Natural extrapolation to longer sequences

### 5. ALiBi (Attention with Linear Biases)

Adds position information directly to attention scores.

```python
from ilovetools.ml.positional_encoding import ALiBiPositionalBias

alibi = ALiBiPositionalBias(
    num_heads=8,      # Number of attention heads
    max_len=2048      # Maximum sequence length
)

# Add bias to attention scores
attention_scores = compute_attention_scores()  # Your attention computation
attention_with_bias = alibi.forward(attention_scores, seq_len)
```

**When to use:**
- Excellent extrapolation to longer sequences
- No positional embeddings needed
- Used in BLOOM
- Memory efficient

## 🎯 Attention Mechanisms

### 1. Scaled Dot-Product Attention

The fundamental attention operation.

```python
from ilovetools.ml.positional_encoding import scaled_dot_product_attention

# Create Q, K, V
query = np.random.randn(batch_size, num_heads, seq_len, d_k)
key = np.random.randn(batch_size, num_heads, seq_len, d_k)
value = np.random.randn(batch_size, num_heads, seq_len, d_v)

# Apply attention
output, attention_weights = scaled_dot_product_attention(
    query, key, value,
    mask=None,        # Optional attention mask
    dropout=0.1       # Dropout rate
)
```

**Formula:** `Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V`

### 2. Multi-Head Attention

Parallel attention with multiple representation subspaces.

```python
from ilovetools.ml.positional_encoding import MultiHeadAttention

mha = MultiHeadAttention(
    d_model=512,      # Model dimension
    num_heads=8,      # Number of attention heads
    dropout=0.1       # Dropout rate
)

# Self-attention
output, weights = mha.forward(x, x, x)

# Cross-attention
output, weights = mha.forward(
    query=decoder_hidden,
    key=encoder_hidden,
    value=encoder_hidden
)

# With mask
mask = create_padding_mask(sequences)
output, weights = mha.forward(x, x, x, mask=mask)
```

**When to use:**
- Core component of transformers
- Captures different aspects of relationships
- Standard for encoder and decoder layers

### 3. Causal (Masked) Attention

Prevents attending to future positions (for autoregressive models).

```python
from ilovetools.ml.positional_encoding import CausalAttention

causal_attn = CausalAttention(
    d_model=512,
    num_heads=8,
    dropout=0.1
)

# Automatically applies causal mask
output, weights = causal_attn.forward(x, x, x)

# Attention weights will have upper triangle = 0
```

**When to use:**
- GPT-style language models
- Any autoregressive model
- Prevents information leakage from future

### 4. Utility Functions

```python
from ilovetools.ml.positional_encoding import (
    create_padding_mask,
    create_look_ahead_mask,
    softmax,
)

# Padding mask for variable-length sequences
sequences = np.array([[1, 2, 3, 0, 0], [4, 5, 0, 0, 0]])
padding_mask = create_padding_mask(sequences, pad_token=0)
# Output: [[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]]

# Look-ahead mask for causal attention
look_ahead_mask = create_look_ahead_mask(seq_len=5)
# Output: Lower triangular matrix

# Numerically stable softmax
probs = softmax(logits, axis=-1)
```

## 📖 Examples

### Complete Transformer Encoder Block

```python
from ilovetools.ml.positional_encoding import (
    SinusoidalPositionalEncoding,
    MultiHeadAttention,
)

# Components
pe = SinusoidalPositionalEncoding(d_model=512)
mha = MultiHeadAttention(d_model=512, num_heads=8)

# Input
x = np.random.randn(32, 50, 512)

# Encoder block
x_with_pos = pe.forward(x)
attn_output, _ = mha.forward(x_with_pos, x_with_pos, x_with_pos)
x = x_with_pos + attn_output  # Residual connection

# Feed-forward (simplified)
W1 = np.random.randn(512, 2048) * 0.02
W2 = np.random.randn(2048, 512) * 0.02
ff_output = np.maximum(0, np.dot(x, W1))  # ReLU
ff_output = np.dot(ff_output, W2)
encoder_output = x + ff_output  # Residual connection
```

### GPT-Style Decoder

```python
from ilovetools.ml.positional_encoding import (
    SinusoidalPositionalEncoding,
    CausalAttention,
)

# Components
pe = SinusoidalPositionalEncoding(d_model=512)
causal_attn = CausalAttention(d_model=512, num_heads=8)

# Input
x = np.random.randn(32, 50, 512)

# Decoder block with causal masking
x_with_pos = pe.forward(x)
attn_output, _ = causal_attn.forward(x_with_pos, x_with_pos, x_with_pos)
decoder_output = x_with_pos + attn_output
```

### LLaMA-Style with RoPE

```python
from ilovetools.ml.positional_encoding import (
    RotaryPositionalEmbedding,
    MultiHeadAttention,
)

# Components
rope = RotaryPositionalEmbedding(d_model=512)
mha = MultiHeadAttention(d_model=512, num_heads=8)

# Apply RoPE
x_with_rope = rope.forward(embeddings)
output, _ = mha.forward(x_with_rope, x_with_rope, x_with_rope)
```

### Sentiment Analysis Model

```python
from ilovetools.ml.positional_encoding import (
    SinusoidalPositionalEncoding,
    MultiHeadAttention,
)

# Model components
embedding_dim = 256
pe = SinusoidalPositionalEncoding(embedding_dim)
mha = MultiHeadAttention(embedding_dim, num_heads=8)

# Input: batch of sentences
sentences = np.random.randint(0, 10000, (32, 20))
embeddings = embedding_matrix[sentences]

# Add positional encoding
embeddings_with_pos = pe.forward(embeddings)

# Self-attention
attn_out, _ = mha.forward(embeddings_with_pos, embeddings_with_pos, embeddings_with_pos)

# Pooling and classification
pooled = np.mean(attn_out, axis=1)
logits = np.dot(pooled, W_class)
```

## 📚 API Reference

### Positional Encoding Classes

#### `SinusoidalPositionalEncoding(d_model, max_len=5000, dropout=0.1)`
- `forward(x)` - Add positional encoding to input
- `get_encoding(positions)` - Get encoding for specific positions

#### `LearnedPositionalEmbedding(max_len, d_model)`
- `forward(x)` - Add learned embeddings
- `update_embeddings(gradients, learning_rate)` - Update embeddings

#### `RelativePositionalEncoding(d_model, max_relative_position=128)`
- `forward(seq_len)` - Generate relative encodings

#### `RotaryPositionalEmbedding(d_model, max_len=2048, base=10000.0)`
- `forward(x, positions=None)` - Apply RoPE

#### `ALiBiPositionalBias(num_heads, max_len=2048)`
- `forward(attention_scores, seq_len)` - Add ALiBi bias

### Attention Classes

#### `MultiHeadAttention(d_model, num_heads, dropout=0.1)`
- `forward(query, key, value, mask=None)` - Apply multi-head attention

#### `CausalAttention(d_model, num_heads, dropout=0.1)`
- `forward(query, key, value, mask=None)` - Apply causal attention

### Functions

#### `scaled_dot_product_attention(query, key, value, mask=None, dropout=0.0)`
Returns: `(output, attention_weights)`

#### `create_padding_mask(seq, pad_token=0)`
Returns: Padding mask tensor

#### `create_look_ahead_mask(size)`
Returns: Lower triangular mask

#### `softmax(x, axis=-1)`
Returns: Softmax probabilities

### Aliases

```python
# Positional encoding aliases
sinusoidal_pe = SinusoidalPositionalEncoding
learned_pe = LearnedPositionalEmbedding
relative_pe = RelativePositionalEncoding
rope = RotaryPositionalEmbedding
alibi = ALiBiPositionalBias

# Attention aliases
mha = MultiHeadAttention
causal_attn = CausalAttention
sdpa = scaled_dot_product_attention
```

## ⚡ Performance

### Complexity Analysis

| Operation | Time Complexity | Space Complexity |
|-----------|----------------|------------------|
| Sinusoidal PE | O(1) cached | O(max_len × d_model) |
| Learned PE | O(seq_len × d_model) | O(max_len × d_model) |
| RoPE | O(seq_len × d_model) | O(max_len × d_model/2) |
| ALiBi | O(num_heads × seq_len²) | O(num_heads × max_len²) |
| Multi-Head Attention | O(seq_len² × d_model) | O(batch × heads × seq_len²) |

### Optimization Tips

1. **Precompute positional encodings** - Done automatically for sinusoidal and RoPE
2. **Use appropriate sequence length** - Don't set max_len unnecessarily high
3. **Batch processing** - Process multiple sequences together
4. **Attention masking** - Use masks to ignore padding tokens

## 🧪 Testing

Run comprehensive tests:

```bash
# All tests
python tests/test_positional_encoding.py

# Verification only
python tests/verify_positional_encoding.py

# Examples
python examples/positional_encoding_examples.py
```

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📄 License

MIT License - See LICENSE file for details

## 📞 Support

- 📖 Documentation: [GitHub Wiki](https://github.com/AliMehdi512/ilovetools)
- 🐛 Issues: [GitHub Issues](https://github.com/AliMehdi512/ilovetools/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/AliMehdi512/ilovetools/discussions)
- 📧 Email: ali.mehdi.dev579@gmail.com

## 🙏 Acknowledgments

Inspired by:
- "Attention Is All You Need" (Vaswani et al., 2017)
- "RoFormer: Enhanced Transformer with Rotary Position Embedding" (Su et al., 2021)
- "Train Short, Test Long: Attention with Linear Biases" (Press et al., 2022)
- LLaMA, GPT-NeoX, T5, BLOOM implementations

## 🔗 Links

- **GitHub:** https://github.com/AliMehdi512/ilovetools
- **PyPI:** https://pypi.org/project/ilovetools/
- **Documentation:** https://github.com/AliMehdi512/ilovetools/tree/main/docs

---

**Happy Coding! 🚀**
