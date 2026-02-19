"""
Comprehensive Examples: Attention Mechanisms and Transformers

This file demonstrates all attention components with practical examples.

Author: Ali Mehdi
Date: February 19, 2026
"""

import numpy as np
from ilovetools.ml.attention import (
    scaled_dot_product_attention,
    MultiHeadAttention,
    PositionalEncoding,
    FeedForward,
    TransformerEncoderLayer,
    TransformerDecoderLayer,
    Transformer,
    layer_norm,
    create_causal_mask,
    create_padding_mask,
)

print("=" * 80)
print("ATTENTION MECHANISMS & TRANSFORMERS - COMPREHENSIVE EXAMPLES")
print("=" * 80)
print()

# ============================================================================
# EXAMPLE 1: Scaled Dot-Product Attention
# ============================================================================
print("EXAMPLE 1: Scaled Dot-Product Attention")
print("-" * 80)

# Create query, key, value matrices
batch_size = 2
seq_len = 10
d_k = 64

query = np.random.randn(batch_size, seq_len, d_k)
key = np.random.randn(batch_size, seq_len, d_k)
value = np.random.randn(batch_size, seq_len, d_k)

print(f"Query shape: {query.shape}")
print(f"Key shape: {key.shape}")
print(f"Value shape: {value.shape}")
print()

# Compute attention
output, attention_weights = scaled_dot_product_attention(query, key, value)

print(f"Output shape: {output.shape}")
print(f"Attention weights shape: {attention_weights.shape}")
print(f"Attention weights sum (should be 1.0): {attention_weights[0, 0].sum():.4f}")

print("\n✓ Example 1 completed\n")

# ============================================================================
# EXAMPLE 2: Multi-Head Attention
# ============================================================================
print("EXAMPLE 2: Multi-Head Attention")
print("-" * 80)

d_model = 512
num_heads = 8

mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)

query = np.random.randn(batch_size, seq_len, d_model)
key = np.random.randn(batch_size, seq_len, d_model)
value = np.random.randn(batch_size, seq_len, d_model)

print(f"Model dimension: {d_model}")
print(f"Number of heads: {num_heads}")
print(f"Dimension per head: {d_model // num_heads}")
print()

output, attention_weights = mha(query, key, value)

print(f"Output shape: {output.shape}")
print(f"Attention weights shape: {attention_weights.shape}")
print(f"  [batch, num_heads, seq_len, seq_len]")

print("\n✓ Example 2 completed\n")

# ============================================================================
# EXAMPLE 3: Positional Encoding
# ============================================================================
print("EXAMPLE 3: Positional Encoding")
print("-" * 80)

d_model = 512
max_len = 100

pe = PositionalEncoding(d_model=d_model, max_len=max_len)

x = np.random.randn(batch_size, seq_len, d_model)

print(f"Input shape: {x.shape}")
print()

x_with_pos = pe(x)

print(f"Output shape: {x_with_pos.shape}")
print(f"Positional encoding added!")
print()

print("Positional encoding formula:")
print("  PE(pos, 2i) = sin(pos / 10000^(2i/d_model))")
print("  PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))")

print("\n✓ Example 3 completed\n")

# ============================================================================
# EXAMPLE 4: Feed-Forward Network
# ============================================================================
print("EXAMPLE 4: Feed-Forward Network")
print("-" * 80)

d_model = 512
d_ff = 2048

ffn = FeedForward(d_model=d_model, d_ff=d_ff)

x = np.random.randn(batch_size, seq_len, d_model)

print(f"Input dimension: {d_model}")
print(f"Hidden dimension: {d_ff}")
print()

output = ffn(x)

print(f"Output shape: {output.shape}")
print()

print("Feed-forward formula:")
print("  FFN(x) = max(0, xW_1 + b_1)W_2 + b_2")
print("  Two linear layers with ReLU activation")

print("\n✓ Example 4 completed\n")

# ============================================================================
# EXAMPLE 5: Transformer Encoder Layer
# ============================================================================
print("EXAMPLE 5: Transformer Encoder Layer")
print("-" * 80)

encoder_layer = TransformerEncoderLayer(d_model=512, num_heads=8, d_ff=2048)

x = np.random.randn(batch_size, seq_len, 512)

print("Encoder layer components:")
print("  1. Multi-head self-attention")
print("  2. Add & Norm (residual + layer norm)")
print("  3. Feed-forward network")
print("  4. Add & Norm")
print()

output = encoder_layer(x)

print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")

print("\n✓ Example 5 completed\n")

# ============================================================================
# EXAMPLE 6: Transformer Decoder Layer
# ============================================================================
print("EXAMPLE 6: Transformer Decoder Layer")
print("-" * 80)

decoder_layer = TransformerDecoderLayer(d_model=512, num_heads=8, d_ff=2048)

x = np.random.randn(batch_size, seq_len, 512)
encoder_output = np.random.randn(batch_size, seq_len, 512)

print("Decoder layer components:")
print("  1. Masked multi-head self-attention")
print("  2. Add & Norm")
print("  3. Multi-head cross-attention (with encoder)")
print("  4. Add & Norm")
print("  5. Feed-forward network")
print("  6. Add & Norm")
print()

output = decoder_layer(x, encoder_output)

print(f"Input shape: {x.shape}")
print(f"Encoder output shape: {encoder_output.shape}")
print(f"Output shape: {output.shape}")

print("\n✓ Example 6 completed\n")

# ============================================================================
# EXAMPLE 7: Full Transformer Model
# ============================================================================
print("EXAMPLE 7: Full Transformer Model")
print("-" * 80)

transformer = Transformer(
    d_model=512,
    num_heads=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
    d_ff=2048
)

src = np.random.randn(batch_size, 10, 512)  # Source sequence
tgt = np.random.randn(batch_size, 8, 512)   # Target sequence

print("Transformer configuration:")
print(f"  Model dimension: 512")
print(f"  Number of heads: 8")
print(f"  Encoder layers: 6")
print(f"  Decoder layers: 6")
print(f"  Feed-forward dimension: 2048")
print()

output = transformer(src, tgt)

print(f"Source shape: {src.shape}")
print(f"Target shape: {tgt.shape}")
print(f"Output shape: {output.shape}")

print("\n✓ Example 7 completed\n")

# ============================================================================
# EXAMPLE 8: Causal Mask (for Decoder)
# ============================================================================
print("EXAMPLE 8: Causal Mask for Decoder")
print("-" * 80)

seq_len = 5
mask = create_causal_mask(seq_len)

print("Causal mask (prevents attending to future positions):")
print(mask)
print()

print("Interpretation:")
print("  1 = can attend")
print("  0 = cannot attend (future position)")
print()

print("Position 0 can attend to: [0]")
print("Position 1 can attend to: [0, 1]")
print("Position 2 can attend to: [0, 1, 2]")
print("...")

print("\n✓ Example 8 completed\n")

# ============================================================================
# EXAMPLE 9: Padding Mask
# ============================================================================
print("EXAMPLE 9: Padding Mask")
print("-" * 80)

seq = np.array([[1, 2, 3, 0, 0], [1, 2, 0, 0, 0]])

print("Sequences (0 = padding):")
print(seq)
print()

mask = create_padding_mask(seq, pad_idx=0)

print(f"Padding mask shape: {mask.shape}")
print()

print("Interpretation:")
print("  Sequence 1: tokens [1, 2, 3] are valid, [0, 0] are padding")
print("  Sequence 2: tokens [1, 2] are valid, [0, 0, 0] are padding")

print("\n✓ Example 9 completed\n")

# ============================================================================
# EXAMPLE 10: Machine Translation Example
# ============================================================================
print("EXAMPLE 10: Machine Translation with Transformer")
print("-" * 80)

print("Task: Translate English to French")
print()

# Create transformer
transformer = Transformer(
    d_model=256,
    num_heads=4,
    num_encoder_layers=3,
    num_decoder_layers=3,
    d_ff=1024
)

# Simulate embeddings
src_seq_len = 12  # "Hello, how are you today?"
tgt_seq_len = 10  # "Bonjour, comment allez-vous?"

src = np.random.randn(1, src_seq_len, 256)  # English sentence
tgt = np.random.randn(1, tgt_seq_len, 256)  # French sentence (teacher forcing)

print(f"Source (English): {src_seq_len} tokens")
print(f"Target (French): {tgt_seq_len} tokens")
print()

# Create causal mask for decoder
tgt_mask = create_causal_mask(tgt_seq_len)

# Forward pass
output = transformer(src, tgt, tgt_mask=tgt_mask)

print(f"Output shape: {output.shape}")
print()

print("Training process:")
print("  1. Encode English sentence")
print("  2. Decode to French (with causal mask)")
print("  3. Compute loss against ground truth")
print("  4. Backpropagate and update weights")

print("\n✓ Example 10 completed\n")

# ============================================================================
# EXAMPLE 11: Self-Attention Visualization
# ============================================================================
print("EXAMPLE 11: Self-Attention Visualization")
print("-" * 80)

# Small example for visualization
seq_len = 5
d_model = 64

x = np.random.randn(1, seq_len, d_model)

# Self-attention (Q=K=V=x)
output, weights = scaled_dot_product_attention(x, x, x)

print("Self-attention weights (how much each position attends to others):")
print(weights[0])
print()

print("Interpretation:")
print("  Row i, Column j = how much position i attends to position j")
print("  Each row sums to 1.0")

print("\n✓ Example 11 completed\n")

# ============================================================================
# EXAMPLE 12: Comparing RNN vs Transformer
# ============================================================================
print("EXAMPLE 12: RNN vs Transformer Comparison")
print("-" * 80)

print("RNN (Sequential Processing):")
print("  ✗ Sequential (slow)")
print("  ✗ Vanishing gradients")
print("  ✗ Limited context window")
print("  ✗ No parallelization")
print()

print("Transformer (Parallel Processing):")
print("  ✓ Parallel (fast)")
print("  ✓ No vanishing gradients")
print("  ✓ Unlimited context (with attention)")
print("  ✓ Full parallelization")
print()

print("Speed comparison (seq_len=100):")
print("  RNN: 100 sequential steps")
print("  Transformer: 1 parallel step")

print("\n✓ Example 12 completed\n")

# ============================================================================
# EXAMPLE 13: Real-World Applications
# ============================================================================
print("EXAMPLE 13: Real-World Applications")
print("-" * 80)

print("Natural Language Processing:")
print("  • GPT (Decoder-only): Text generation")
print("  • BERT (Encoder-only): Text understanding")
print("  • T5 (Encoder-Decoder): Text-to-text tasks")
print()

print("Computer Vision:")
print("  • Vision Transformer (ViT): Image classification")
print("  • CLIP: Image-text matching")
print("  • DALL-E: Text-to-image generation")
print()

print("Speech:")
print("  • Whisper: Speech recognition")
print("  • Speech Transformer: Speech synthesis")
print()

print("Multimodal:")
print("  • Flamingo: Visual question answering")
print("  • Gato: Generalist agent")

print("\n✓ Example 13 completed\n")

# ============================================================================
# SUMMARY
# ============================================================================
print("=" * 80)
print("ALL EXAMPLES COMPLETED SUCCESSFULLY! ✓")
print("=" * 80)
print()
print("Summary of what we covered:")
print("1. ✓ Scaled Dot-Product Attention")
print("2. ✓ Multi-Head Attention")
print("3. ✓ Positional Encoding")
print("4. ✓ Feed-Forward Network")
print("5. ✓ Transformer Encoder Layer")
print("6. ✓ Transformer Decoder Layer")
print("7. ✓ Full Transformer Model")
print("8. ✓ Causal Mask")
print("9. ✓ Padding Mask")
print("10. ✓ Machine Translation Example")
print("11. ✓ Self-Attention Visualization")
print("12. ✓ RNN vs Transformer")
print("13. ✓ Real-World Applications")
print()
print("You now have a complete understanding of attention mechanisms!")
print()
print("Next steps:")
print("- Use for NLP tasks (GPT, BERT)")
print("- Apply to computer vision (ViT)")
print("- Build multimodal models (CLIP, DALL-E)")
print("- Implement custom transformer architectures")
