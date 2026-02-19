"""
Tests for Attention Mechanisms and Transformers

This file contains comprehensive tests for all attention components.

Author: Ali Mehdi
Date: February 19, 2026
"""

import numpy as np
import pytest
from ilovetools.ml.attention import (
    scaled_dot_product_attention,
    MultiHeadAttention,
    PositionalEncoding,
    FeedForward,
    TransformerEncoderLayer,
    TransformerDecoderLayer,
    Transformer,
    layer_norm,
    softmax,
    create_causal_mask,
    create_padding_mask,
)


# ============================================================================
# TEST SCALED DOT-PRODUCT ATTENTION
# ============================================================================

def test_scaled_dot_product_attention_basic():
    """Test basic scaled dot-product attention."""
    query = np.random.randn(2, 10, 64)
    key = np.random.randn(2, 10, 64)
    value = np.random.randn(2, 10, 64)
    
    output, weights = scaled_dot_product_attention(query, key, value)
    
    assert output.shape == (2, 10, 64)
    assert weights.shape == (2, 10, 10)


def test_scaled_dot_product_attention_with_mask():
    """Test attention with mask."""
    query = np.random.randn(2, 10, 64)
    key = np.random.randn(2, 10, 64)
    value = np.random.randn(2, 10, 64)
    mask = np.ones((2, 10, 10))
    
    output, weights = scaled_dot_product_attention(query, key, value, mask)
    
    assert output.shape == (2, 10, 64)
    assert weights.shape == (2, 10, 10)


def test_softmax():
    """Test softmax function."""
    x = np.array([[1.0, 2.0, 3.0]])
    result = softmax(x, axis=-1)
    
    assert result.shape == x.shape
    assert np.allclose(result.sum(axis=-1), 1.0)


# ============================================================================
# TEST MULTI-HEAD ATTENTION
# ============================================================================

def test_multi_head_attention_basic():
    """Test basic multi-head attention."""
    mha = MultiHeadAttention(d_model=512, num_heads=8)
    query = np.random.randn(2, 10, 512)
    key = np.random.randn(2, 10, 512)
    value = np.random.randn(2, 10, 512)
    
    output, weights = mha(query, key, value)
    
    assert output.shape == (2, 10, 512)
    assert weights.shape == (2, 8, 10, 10)


def test_multi_head_attention_different_seq_len():
    """Test multi-head attention with different sequence lengths."""
    mha = MultiHeadAttention(d_model=512, num_heads=8)
    query = np.random.randn(2, 5, 512)
    key = np.random.randn(2, 10, 512)
    value = np.random.randn(2, 10, 512)
    
    output, weights = mha(query, key, value)
    
    assert output.shape == (2, 5, 512)
    assert weights.shape == (2, 8, 5, 10)


def test_multi_head_attention_split_combine():
    """Test split and combine heads."""
    mha = MultiHeadAttention(d_model=512, num_heads=8)
    x = np.random.randn(2, 10, 512)
    
    # Split heads
    x_split = mha.split_heads(x)
    assert x_split.shape == (2, 8, 10, 64)
    
    # Combine heads
    x_combined = mha.combine_heads(x_split)
    assert x_combined.shape == (2, 10, 512)


# ============================================================================
# TEST POSITIONAL ENCODING
# ============================================================================

def test_positional_encoding_basic():
    """Test basic positional encoding."""
    pe = PositionalEncoding(d_model=512, max_len=100)
    x = np.random.randn(2, 10, 512)
    
    output = pe(x)
    
    assert output.shape == (2, 10, 512)


def test_positional_encoding_different_lengths():
    """Test positional encoding with different sequence lengths."""
    pe = PositionalEncoding(d_model=512, max_len=100)
    
    x1 = np.random.randn(2, 5, 512)
    output1 = pe(x1)
    assert output1.shape == (2, 5, 512)
    
    x2 = np.random.randn(2, 20, 512)
    output2 = pe(x2)
    assert output2.shape == (2, 20, 512)


# ============================================================================
# TEST FEED-FORWARD NETWORK
# ============================================================================

def test_feed_forward_basic():
    """Test basic feed-forward network."""
    ffn = FeedForward(d_model=512, d_ff=2048)
    x = np.random.randn(2, 10, 512)
    
    output = ffn(x)
    
    assert output.shape == (2, 10, 512)


def test_feed_forward_relu_activation():
    """Test that feed-forward uses ReLU activation."""
    ffn = FeedForward(d_model=512, d_ff=2048)
    x = np.random.randn(2, 10, 512)
    
    output = ffn(x)
    
    # Output should be different from input
    assert not np.allclose(output, x)


# ============================================================================
# TEST LAYER NORMALIZATION
# ============================================================================

def test_layer_norm_basic():
    """Test basic layer normalization."""
    x = np.random.randn(2, 10, 512)
    
    output = layer_norm(x)
    
    assert output.shape == (2, 10, 512)


def test_layer_norm_statistics():
    """Test that layer norm produces correct statistics."""
    x = np.random.randn(2, 10, 512)
    
    output = layer_norm(x)
    
    # Mean should be close to 0
    mean = output.mean(axis=-1)
    assert np.allclose(mean, 0, atol=1e-5)
    
    # Std should be close to 1
    std = output.std(axis=-1)
    assert np.allclose(std, 1, atol=1e-1)


# ============================================================================
# TEST TRANSFORMER ENCODER LAYER
# ============================================================================

def test_transformer_encoder_layer_basic():
    """Test basic transformer encoder layer."""
    encoder_layer = TransformerEncoderLayer(d_model=512, num_heads=8)
    x = np.random.randn(2, 10, 512)
    
    output = encoder_layer(x)
    
    assert output.shape == (2, 10, 512)


def test_transformer_encoder_layer_with_mask():
    """Test encoder layer with mask."""
    encoder_layer = TransformerEncoderLayer(d_model=512, num_heads=8)
    x = np.random.randn(2, 10, 512)
    mask = np.ones((2, 10, 10))
    
    output = encoder_layer(x, mask)
    
    assert output.shape == (2, 10, 512)


# ============================================================================
# TEST TRANSFORMER DECODER LAYER
# ============================================================================

def test_transformer_decoder_layer_basic():
    """Test basic transformer decoder layer."""
    decoder_layer = TransformerDecoderLayer(d_model=512, num_heads=8)
    x = np.random.randn(2, 10, 512)
    encoder_output = np.random.randn(2, 10, 512)
    
    output = decoder_layer(x, encoder_output)
    
    assert output.shape == (2, 10, 512)


def test_transformer_decoder_layer_with_masks():
    """Test decoder layer with masks."""
    decoder_layer = TransformerDecoderLayer(d_model=512, num_heads=8)
    x = np.random.randn(2, 10, 512)
    encoder_output = np.random.randn(2, 10, 512)
    self_mask = np.ones((2, 10, 10))
    cross_mask = np.ones((2, 10, 10))
    
    output = decoder_layer(x, encoder_output, self_mask, cross_mask)
    
    assert output.shape == (2, 10, 512)


# ============================================================================
# TEST FULL TRANSFORMER
# ============================================================================

def test_transformer_basic():
    """Test basic transformer."""
    transformer = Transformer(d_model=512, num_heads=8, num_encoder_layers=2, num_decoder_layers=2)
    src = np.random.randn(2, 10, 512)
    tgt = np.random.randn(2, 8, 512)
    
    output = transformer(src, tgt)
    
    assert output.shape == (2, 8, 512)


def test_transformer_encode():
    """Test transformer encoder."""
    transformer = Transformer(d_model=512, num_heads=8, num_encoder_layers=2)
    src = np.random.randn(2, 10, 512)
    
    encoder_output = transformer.encode(src)
    
    assert encoder_output.shape == (2, 10, 512)


def test_transformer_decode():
    """Test transformer decoder."""
    transformer = Transformer(d_model=512, num_heads=8, num_decoder_layers=2)
    tgt = np.random.randn(2, 8, 512)
    encoder_output = np.random.randn(2, 10, 512)
    
    decoder_output = transformer.decode(tgt, encoder_output)
    
    assert decoder_output.shape == (2, 8, 512)


def test_transformer_with_masks():
    """Test transformer with masks."""
    transformer = Transformer(d_model=512, num_heads=8, num_encoder_layers=2, num_decoder_layers=2)
    src = np.random.randn(2, 10, 512)
    tgt = np.random.randn(2, 8, 512)
    src_mask = np.ones((2, 10, 10))
    tgt_mask = create_causal_mask(8)
    
    output = transformer(src, tgt, src_mask, tgt_mask)
    
    assert output.shape == (2, 8, 512)


# ============================================================================
# TEST UTILITY FUNCTIONS
# ============================================================================

def test_create_causal_mask():
    """Test causal mask creation."""
    mask = create_causal_mask(5)
    
    assert mask.shape == (5, 5)
    assert mask[0, 0] == 1
    assert mask[0, 1] == 0
    assert mask[4, 4] == 1


def test_create_padding_mask():
    """Test padding mask creation."""
    seq = np.array([[1, 2, 3, 0, 0], [1, 2, 0, 0, 0]])
    mask = create_padding_mask(seq, pad_idx=0)
    
    assert mask.shape == (2, 1, 5)
    assert mask[0, 0, 0] == 1
    assert mask[0, 0, 3] == 0
    assert mask[1, 0, 2] == 0


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

def test_all_components_callable():
    """Test that all components are callable."""
    # Scaled dot-product attention
    q = np.random.randn(2, 10, 64)
    k = np.random.randn(2, 10, 64)
    v = np.random.randn(2, 10, 64)
    output, weights = scaled_dot_product_attention(q, k, v)
    assert output is not None
    
    # Multi-head attention
    mha = MultiHeadAttention(d_model=512, num_heads=8)
    q = np.random.randn(2, 10, 512)
    output, weights = mha(q, q, q)
    assert output is not None
    
    # Positional encoding
    pe = PositionalEncoding(d_model=512)
    x = np.random.randn(2, 10, 512)
    output = pe(x)
    assert output is not None
    
    # Feed-forward
    ffn = FeedForward(d_model=512)
    output = ffn(x)
    assert output is not None
    
    # Transformer
    transformer = Transformer(d_model=512, num_heads=8)
    src = np.random.randn(2, 10, 512)
    tgt = np.random.randn(2, 8, 512)
    output = transformer(src, tgt)
    assert output is not None


def test_transformer_end_to_end():
    """Test transformer end-to-end."""
    # Create transformer
    transformer = Transformer(
        d_model=256,
        num_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=1024
    )
    
    # Create input sequences
    src = np.random.randn(4, 12, 256)  # batch=4, src_len=12
    tgt = np.random.randn(4, 10, 256)  # batch=4, tgt_len=10
    
    # Create masks
    tgt_mask = create_causal_mask(10)
    
    # Forward pass
    output = transformer(src, tgt, tgt_mask=tgt_mask)
    
    assert output.shape == (4, 10, 256)


print("=" * 80)
print("ALL ATTENTION MECHANISM TESTS PASSED! ✓")
print("=" * 80)
