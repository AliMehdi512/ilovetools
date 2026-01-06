# Trigger PyPI Publication for v0.2.23

This file triggers the GitHub Actions workflow to publish version 0.2.23 to PyPI.

## Changes in v0.2.23

### New Module: Positional Encoding and Attention Mechanisms

**5 Positional Encoding Variants:**
1. SinusoidalPositionalEncoding - Original Transformer
2. LearnedPositionalEmbedding - Trainable positions
3. RelativePositionalEncoding - T5-style relative positions
4. RotaryPositionalEmbedding (RoPE) - LLaMA-style
5. ALiBiPositionalBias - BLOOM-style

**4 Attention Mechanism Types:**
1. scaled_dot_product_attention - Basic attention
2. MultiHeadAttention - Parallel attention heads
3. CausalAttention - GPT-style masked attention
4. Cross-Attention support

**Complete Testing:**
- 15+ test functions
- 372+ test cases
- 14 comprehensive examples
- Full documentation

## Installation

After publication:
```bash
pip install ilovetools==0.2.23
```

## Usage

```python
from ilovetools.ml.positional_encoding import (
    SinusoidalPositionalEncoding,
    MultiHeadAttention,
    RotaryPositionalEmbedding,
    ALiBiPositionalBias,
    CausalAttention,
)
```

## Links

- GitHub: https://github.com/AliMehdi512/ilovetools
- PyPI: https://pypi.org/project/ilovetools/
- Documentation: https://github.com/AliMehdi512/ilovetools/tree/main/docs

---

**Version:** 0.2.23  
**Date:** January 4, 2026  
**Status:** Ready for PyPI publication
