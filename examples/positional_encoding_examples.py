"""
Comprehensive Examples: Positional Encoding and Attention Mechanisms

This file demonstrates all positional encoding techniques and attention
mechanisms with practical examples and use cases.
"""

import numpy as np
from ilovetools.ml.positional_encoding import (
    SinusoidalPositionalEncoding,
    LearnedPositionalEmbedding,
    RelativePositionalEncoding,
    RotaryPositionalEmbedding,
    ALiBiPositionalBias,
    MultiHeadAttention,
    CausalAttention,
    scaled_dot_product_attention,
    create_padding_mask,
    create_look_ahead_mask,
)

print("=" * 80)
print("POSITIONAL ENCODING AND ATTENTION MECHANISMS - COMPREHENSIVE EXAMPLES")
print("=" * 80)
print()

# ============================================================================
# EXAMPLE 1: Sinusoidal Positional Encoding (Original Transformer)
# ============================================================================
print("EXAMPLE 1: Sinusoidal Positional Encoding")
print("-" * 80)

d_model = 512
max_len = 100
batch_size = 32
seq_len = 50

# Create sinusoidal positional encoding
pe = SinusoidalPositionalEncoding(d_model, max_len, dropout=0.1)

# Simulate word embeddings
word_embeddings = np.random.randn(batch_size, seq_len, d_model)

# Add positional encoding
embeddings_with_position = pe.forward(word_embeddings)

print(f"Input shape: {word_embeddings.shape}")
print(f"Output shape: {embeddings_with_position.shape}")
print(f"Positional encoding added successfully!")

# Visualize position encodings for first few positions
print("\nPositional encodings for positions 0-4 (first 10 dimensions):")
for pos in range(5):
    encoding = pe.get_encoding(np.array([pos]))[0, :10]
    print(f"Position {pos}: {encoding}")

print("\n✓ Example 1 completed\n")

# ============================================================================
# EXAMPLE 2: Learned Positional Embeddings
# ============================================================================
print("EXAMPLE 2: Learned Positional Embeddings")
print("-" * 80)

# Create learned positional embeddings
learned_pe = LearnedPositionalEmbedding(max_len, d_model)

# Add to word embeddings
embeddings_with_learned_pos = learned_pe.forward(word_embeddings)

print(f"Input shape: {word_embeddings.shape}")
print(f"Output shape: {embeddings_with_learned_pos.shape}")

# Simulate training update
gradients = np.random.randn(max_len, d_model) * 0.001
learned_pe.update_embeddings(gradients, learning_rate=0.001)
print("Embeddings updated with gradients")

print("\n✓ Example 2 completed\n")

# ============================================================================
# EXAMPLE 3: Relative Positional Encoding (T5-style)
# ============================================================================
print("EXAMPLE 3: Relative Positional Encoding")
print("-" * 80)

# Create relative positional encoding
relative_pe = RelativePositionalEncoding(d_model, max_relative_position=128)

# Generate relative position encodings
seq_len_rel = 20
relative_encodings = relative_pe.forward(seq_len_rel)

print(f"Relative encodings shape: {relative_encodings.shape}")
print(f"Shape: (seq_len, seq_len, d_model) = ({seq_len_rel}, {seq_len_rel}, {d_model})")
print("Each pair of positions has a relative encoding")

print("\n✓ Example 3 completed\n")

# ============================================================================
# EXAMPLE 4: Rotary Position Embedding (RoPE) - LLaMA style
# ============================================================================
print("EXAMPLE 4: Rotary Position Embedding (RoPE)")
print("-" * 80)

# Create RoPE
rope = RotaryPositionalEmbedding(d_model, max_len=2048)

# Apply RoPE to embeddings
embeddings_with_rope = rope.forward(word_embeddings)

print(f"Input shape: {word_embeddings.shape}")
print(f"Output shape: {embeddings_with_rope.shape}")
print("RoPE applied through rotation matrices")

# Apply to specific positions
custom_positions = np.array([0, 5, 10, 15, 20])
embeddings_custom = rope.forward(word_embeddings[:, :5, :], custom_positions)
print(f"\nCustom positions shape: {embeddings_custom.shape}")

print("\n✓ Example 4 completed\n")

# ============================================================================
# EXAMPLE 5: ALiBi (Attention with Linear Biases)
# ============================================================================
print("EXAMPLE 5: ALiBi (Attention with Linear Biases)")
print("-" * 80)

num_heads = 8

# Create ALiBi
alibi = ALiBiPositionalBias(num_heads, max_len=2048)

# Simulate attention scores
attention_scores = np.random.randn(batch_size, num_heads, seq_len, seq_len)

# Add ALiBi bias
attention_with_alibi = alibi.forward(attention_scores, seq_len)

print(f"Attention scores shape: {attention_scores.shape}")
print(f"With ALiBi bias shape: {attention_with_alibi.shape}")
print("ALiBi bias added to attention scores")

# Show bias pattern for first head
print(f"\nBias pattern for head 0 (first 5x5):")
print(alibi.bias[0, :5, :5])

print("\n✓ Example 5 completed\n")

# ============================================================================
# EXAMPLE 6: Scaled Dot-Product Attention
# ============================================================================
print("EXAMPLE 6: Scaled Dot-Product Attention")
print("-" * 80)

d_k = 64
d_v = 64

# Create Q, K, V
query = np.random.randn(batch_size, 1, seq_len, d_k)
key = np.random.randn(batch_size, 1, seq_len, d_k)
value = np.random.randn(batch_size, 1, seq_len, d_v)

# Apply attention
output, attention_weights = scaled_dot_product_attention(query, key, value)

print(f"Query shape: {query.shape}")
print(f"Key shape: {key.shape}")
print(f"Value shape: {value.shape}")
print(f"Output shape: {output.shape}")
print(f"Attention weights shape: {attention_weights.shape}")

# Verify attention weights sum to 1
weights_sum = np.sum(attention_weights[0, 0, 0, :])
print(f"\nAttention weights sum: {weights_sum:.6f} (should be ~1.0)")

print("\n✓ Example 6 completed\n")

# ============================================================================
# EXAMPLE 7: Multi-Head Attention
# ============================================================================
print("EXAMPLE 7: Multi-Head Attention")
print("-" * 80)

# Create multi-head attention
mha = MultiHeadAttention(d_model, num_heads, dropout=0.1)

# Prepare inputs
query_mha = np.random.randn(batch_size, seq_len, d_model)
key_mha = np.random.randn(batch_size, seq_len, d_model)
value_mha = np.random.randn(batch_size, seq_len, d_model)

# Apply multi-head attention
output_mha, weights_mha = mha.forward(query_mha, key_mha, value_mha)

print(f"Input shape: {query_mha.shape}")
print(f"Output shape: {output_mha.shape}")
print(f"Attention weights shape: {weights_mha.shape}")
print(f"Number of heads: {num_heads}")
print(f"Dimension per head: {d_model // num_heads}")

print("\n✓ Example 7 completed\n")

# ============================================================================
# EXAMPLE 8: Causal (Masked) Attention for GPT-style Models
# ============================================================================
print("EXAMPLE 8: Causal Attention (GPT-style)")
print("-" * 80)

# Create causal attention
causal_attn = CausalAttention(d_model, num_heads, dropout=0.1)

# Apply causal attention
output_causal, weights_causal = causal_attn.forward(query_mha, key_mha, value_mha)

print(f"Input shape: {query_mha.shape}")
print(f"Output shape: {output_causal.shape}")
print(f"Attention weights shape: {weights_causal.shape}")

# Verify causality (upper triangle should be zero)
print("\nCausal mask verification (head 0, first 5x5):")
print(weights_causal[0, 0, :5, :5])
print("Upper triangle should be ~0 (no attending to future)")

print("\n✓ Example 8 completed\n")

# ============================================================================
# EXAMPLE 9: Complete Transformer Encoder Block
# ============================================================================
print("EXAMPLE 9: Complete Transformer Encoder Block")
print("-" * 80)

# Components
pe_encoder = SinusoidalPositionalEncoding(d_model, max_len)
mha_encoder = MultiHeadAttention(d_model, num_heads)

# Input
x_encoder = np.random.randn(batch_size, seq_len, d_model)

# Step 1: Add positional encoding
x_with_pe = pe_encoder.forward(x_encoder)
print(f"After positional encoding: {x_with_pe.shape}")

# Step 2: Self-attention
attn_output, attn_weights = mha_encoder.forward(x_with_pe, x_with_pe, x_with_pe)
print(f"After self-attention: {attn_output.shape}")

# Step 3: Add & Norm (residual connection)
x_norm = x_with_pe + attn_output
print(f"After residual connection: {x_norm.shape}")

# Step 4: Feed-forward (simplified)
W1 = np.random.randn(d_model, d_model * 4) * 0.02
W2 = np.random.randn(d_model * 4, d_model) * 0.02
ff_output = np.maximum(0, np.dot(x_norm, W1))  # ReLU
ff_output = np.dot(ff_output, W2)
print(f"After feed-forward: {ff_output.shape}")

# Step 5: Add & Norm
encoder_output = x_norm + ff_output
print(f"Final encoder output: {encoder_output.shape}")

print("\n✓ Example 9 completed\n")

# ============================================================================
# EXAMPLE 10: Complete Transformer Decoder Block (GPT-style)
# ============================================================================
print("EXAMPLE 10: Complete Transformer Decoder Block (GPT-style)")
print("-" * 80)

# Components
pe_decoder = SinusoidalPositionalEncoding(d_model, max_len)
causal_attn_decoder = CausalAttention(d_model, num_heads)

# Input
x_decoder = np.random.randn(batch_size, seq_len, d_model)

# Step 1: Add positional encoding
x_dec_pe = pe_decoder.forward(x_decoder)
print(f"After positional encoding: {x_dec_pe.shape}")

# Step 2: Masked self-attention
masked_attn_output, masked_weights = causal_attn_decoder.forward(
    x_dec_pe, x_dec_pe, x_dec_pe
)
print(f"After masked self-attention: {masked_attn_output.shape}")

# Step 3: Add & Norm
x_dec_norm = x_dec_pe + masked_attn_output
print(f"After residual connection: {x_dec_norm.shape}")

# Step 4: Feed-forward
ff_dec_output = np.maximum(0, np.dot(x_dec_norm, W1))
ff_dec_output = np.dot(ff_dec_output, W2)
print(f"After feed-forward: {ff_dec_output.shape}")

# Step 5: Add & Norm
decoder_output = x_dec_norm + ff_dec_output
print(f"Final decoder output: {decoder_output.shape}")

print("\n✓ Example 10 completed\n")

# ============================================================================
# EXAMPLE 11: Cross-Attention (Encoder-Decoder)
# ============================================================================
print("EXAMPLE 11: Cross-Attention (Encoder-Decoder)")
print("-" * 80)

# Create cross-attention layer
cross_attn = MultiHeadAttention(d_model, num_heads)

# Decoder queries, Encoder keys and values
decoder_queries = np.random.randn(batch_size, seq_len, d_model)
encoder_keys = np.random.randn(batch_size, seq_len, d_model)
encoder_values = np.random.randn(batch_size, seq_len, d_model)

# Apply cross-attention
cross_output, cross_weights = cross_attn.forward(
    decoder_queries, encoder_keys, encoder_values
)

print(f"Decoder queries shape: {decoder_queries.shape}")
print(f"Encoder keys shape: {encoder_keys.shape}")
print(f"Encoder values shape: {encoder_values.shape}")
print(f"Cross-attention output: {cross_output.shape}")
print(f"Cross-attention weights: {cross_weights.shape}")

print("\n✓ Example 11 completed\n")

# ============================================================================
# EXAMPLE 12: Using Padding Masks
# ============================================================================
print("EXAMPLE 12: Using Padding Masks")
print("-" * 80)

# Create sequence with padding
seq_with_padding = np.array([
    [1, 2, 3, 4, 5, 0, 0, 0],
    [6, 7, 8, 0, 0, 0, 0, 0],
    [9, 10, 11, 12, 13, 14, 15, 16]
])

# Create padding mask
padding_mask = create_padding_mask(seq_with_padding, pad_token=0)

print("Sequence with padding:")
print(seq_with_padding)
print("\nPadding mask (1=real token, 0=padding):")
print(padding_mask)

# Use mask in attention
seq_len_pad = seq_with_padding.shape[1]
query_pad = np.random.randn(3, 1, seq_len_pad, 64)
key_pad = np.random.randn(3, 1, seq_len_pad, 64)
value_pad = np.random.randn(3, 1, seq_len_pad, 64)

# Expand mask for attention
attention_mask = padding_mask[:, None, None, :]  # (batch, 1, 1, seq_len)

output_pad, weights_pad = scaled_dot_product_attention(
    query_pad, key_pad, value_pad, mask=attention_mask
)

print(f"\nAttention output with padding mask: {output_pad.shape}")

print("\n✓ Example 12 completed\n")

# ============================================================================
# EXAMPLE 13: Comparing Different Positional Encodings
# ============================================================================
print("EXAMPLE 13: Comparing Different Positional Encodings")
print("-" * 80)

# Same input for all
x_compare = np.random.randn(1, 10, 128)

# Sinusoidal
pe_sin = SinusoidalPositionalEncoding(128, 100, dropout=0.0)
out_sin = pe_sin.forward(x_compare.copy())

# Learned
pe_learned = LearnedPositionalEmbedding(100, 128)
out_learned = pe_learned.forward(x_compare.copy())

# RoPE
pe_rope = RotaryPositionalEmbedding(128, 100)
out_rope = pe_rope.forward(x_compare.copy())

print("Comparison of positional encodings:")
print(f"Sinusoidal output shape: {out_sin.shape}")
print(f"Learned output shape: {out_learned.shape}")
print(f"RoPE output shape: {out_rope.shape}")

# Calculate differences from original
diff_sin = np.mean(np.abs(out_sin - x_compare))
diff_learned = np.mean(np.abs(out_learned - x_compare))
diff_rope = np.mean(np.abs(out_rope - x_compare))

print(f"\nMean absolute difference from input:")
print(f"Sinusoidal: {diff_sin:.6f}")
print(f"Learned: {diff_learned:.6f}")
print(f"RoPE: {diff_rope:.6f}")

print("\n✓ Example 13 completed\n")

# ============================================================================
# EXAMPLE 14: Real-World Use Case - Sentiment Analysis
# ============================================================================
print("EXAMPLE 14: Real-World Use Case - Sentiment Analysis")
print("-" * 80)

# Simulate a sentiment analysis model
vocab_size = 10000
embedding_dim = 256
num_classes = 3  # positive, negative, neutral

# Input: batch of sentences (token IDs)
sentences = np.random.randint(0, vocab_size, (32, 20))  # 32 sentences, 20 tokens each

# Step 1: Embedding layer (simplified)
embedding_matrix = np.random.randn(vocab_size, embedding_dim) * 0.02
embeddings = embedding_matrix[sentences]

print(f"Input sentences shape: {sentences.shape}")
print(f"Embeddings shape: {embeddings.shape}")

# Step 2: Add positional encoding
pe_sentiment = SinusoidalPositionalEncoding(embedding_dim, max_len=100)
embeddings_with_pos = pe_sentiment.forward(embeddings)

# Step 3: Self-attention
mha_sentiment = MultiHeadAttention(embedding_dim, num_heads=8)
attn_out, _ = mha_sentiment.forward(embeddings_with_pos, embeddings_with_pos, embeddings_with_pos)

# Step 4: Pooling (mean over sequence)
pooled = np.mean(attn_out, axis=1)

# Step 5: Classification head
W_class = np.random.randn(embedding_dim, num_classes) * 0.02
logits = np.dot(pooled, W_class)

print(f"Attention output shape: {attn_out.shape}")
print(f"Pooled shape: {pooled.shape}")
print(f"Logits shape: {logits.shape}")
print(f"Predictions shape: (batch_size={logits.shape[0]}, num_classes={logits.shape[1]})")

print("\n✓ Example 14 completed\n")

# ============================================================================
# SUMMARY
# ============================================================================
print("=" * 80)
print("ALL EXAMPLES COMPLETED SUCCESSFULLY! ✓")
print("=" * 80)
print()
print("Summary of what we covered:")
print("1. ✓ Sinusoidal Positional Encoding (Transformer)")
print("2. ✓ Learned Positional Embeddings")
print("3. ✓ Relative Positional Encoding (T5)")
print("4. ✓ Rotary Position Embedding (LLaMA)")
print("5. ✓ ALiBi (Attention with Linear Biases)")
print("6. ✓ Scaled Dot-Product Attention")
print("7. ✓ Multi-Head Attention")
print("8. ✓ Causal Attention (GPT)")
print("9. ✓ Complete Transformer Encoder")
print("10. ✓ Complete Transformer Decoder")
print("11. ✓ Cross-Attention")
print("12. ✓ Padding Masks")
print("13. ✓ Comparing Positional Encodings")
print("14. ✓ Real-World Sentiment Analysis")
print()
print("You now have a complete understanding of positional encoding")
print("and attention mechanisms used in modern transformers!")
print()
print("Next steps:")
print("- Implement your own transformer model")
print("- Experiment with different positional encodings")
print("- Try different attention patterns")
print("- Build real-world NLP applications")
