"""
Attention Mechanisms and Transformers

This module implements attention mechanisms and transformer architectures
that revolutionized deep learning and enabled modern AI systems.

Implemented Components:
1. Scaled Dot-Product Attention
2. Multi-Head Attention
3. Positional Encoding
4. Transformer Encoder Layer
5. Transformer Decoder Layer
6. Full Transformer Model
7. Self-Attention
8. Cross-Attention

Key Benefits:
- Parallel processing (unlike RNNs)
- Long-range dependencies
- State-of-the-art performance
- Foundation of GPT, BERT, Vision Transformers

Applications:
- Natural Language Processing (GPT, BERT, T5)
- Computer Vision (Vision Transformer, CLIP)
- Speech Recognition (Whisper)
- Multimodal AI (DALL-E, Flamingo)

References:
- Vaswani et al., "Attention Is All You Need" (2017)
- Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers" (2018)
- Radford et al., "Language Models are Unsupervised Multitask Learners" (GPT-2, 2019)
- Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition" (ViT, 2020)

Author: Ali Mehdi
Date: February 19, 2026
"""

import numpy as np
from typing import Optional, Tuple


# ============================================================================
# SCALED DOT-PRODUCT ATTENTION
# ============================================================================

def scaled_dot_product_attention(query: np.ndarray, 
                                 key: np.ndarray, 
                                 value: np.ndarray,
                                 mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Scaled Dot-Product Attention mechanism.
    
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
    
    Args:
        query: Query matrix [batch_size, seq_len, d_k]
        key: Key matrix [batch_size, seq_len, d_k]
        value: Value matrix [batch_size, seq_len, d_v]
        mask: Optional mask [batch_size, seq_len, seq_len]
    
    Returns:
        output: Attention output [batch_size, seq_len, d_v]
        attention_weights: Attention weights [batch_size, seq_len, seq_len]
    
    Example:
        >>> query = np.random.randn(2, 10, 64)  # batch=2, seq_len=10, d_k=64
        >>> key = np.random.randn(2, 10, 64)
        >>> value = np.random.randn(2, 10, 64)
        >>> output, weights = scaled_dot_product_attention(query, key, value)
        >>> print(f"Output shape: {output.shape}")
        >>> print(f"Attention weights shape: {weights.shape}")
    """
    d_k = query.shape[-1]
    
    # Compute attention scores: QK^T / sqrt(d_k)
    scores = np.matmul(query, key.transpose(0, 2, 1)) / np.sqrt(d_k)
    
    # Apply mask if provided (for padding or causal attention)
    if mask is not None:
        scores = np.where(mask == 0, -1e9, scores)
    
    # Apply softmax to get attention weights
    attention_weights = softmax(scores, axis=-1)
    
    # Apply attention weights to values
    output = np.matmul(attention_weights, value)
    
    return output, attention_weights


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


# ============================================================================
# MULTI-HEAD ATTENTION
# ============================================================================

class MultiHeadAttention:
    """
    Multi-Head Attention mechanism.
    
    Allows the model to jointly attend to information from different
    representation subspaces at different positions.
    
    MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O
    where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
    
    Args:
        d_model: Model dimension (default: 512)
        num_heads: Number of attention heads (default: 8)
        dropout: Dropout rate (default: 0.1)
    
    Example:
        >>> from ilovetools.ml.attention import MultiHeadAttention
        >>> mha = MultiHeadAttention(d_model=512, num_heads=8)
        >>> query = np.random.randn(2, 10, 512)  # [batch, seq_len, d_model]
        >>> key = np.random.randn(2, 10, 512)
        >>> value = np.random.randn(2, 10, 512)
        >>> output, weights = mha(query, key, value)
        >>> print(f"Output shape: {output.shape}")  # [2, 10, 512]
    
    Reference:
        Vaswani et al., "Attention Is All You Need" (2017)
    """
    
    def __init__(self, d_model: int = 512, num_heads: int = 8, dropout: float = 0.1):
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.dropout = dropout
        
        # Initialize weight matrices
        self.W_q = np.random.randn(d_model, d_model) * 0.01
        self.W_k = np.random.randn(d_model, d_model) * 0.01
        self.W_v = np.random.randn(d_model, d_model) * 0.01
        self.W_o = np.random.randn(d_model, d_model) * 0.01
    
    def split_heads(self, x: np.ndarray) -> np.ndarray:
        """Split the last dimension into (num_heads, d_k)."""
        batch_size, seq_len, d_model = x.shape
        x = x.reshape(batch_size, seq_len, self.num_heads, self.d_k)
        return x.transpose(0, 2, 1, 3)  # [batch, num_heads, seq_len, d_k]
    
    def combine_heads(self, x: np.ndarray) -> np.ndarray:
        """Combine heads back to original shape."""
        batch_size, num_heads, seq_len, d_k = x.shape
        x = x.transpose(0, 2, 1, 3)  # [batch, seq_len, num_heads, d_k]
        return x.reshape(batch_size, seq_len, self.d_model)
    
    def __call__(self, query: np.ndarray, key: np.ndarray, value: np.ndarray,
                 mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass through multi-head attention.
        
        Args:
            query: Query tensor [batch_size, seq_len, d_model]
            key: Key tensor [batch_size, seq_len, d_model]
            value: Value tensor [batch_size, seq_len, d_model]
            mask: Optional mask [batch_size, seq_len, seq_len]
        
        Returns:
            output: Attention output [batch_size, seq_len, d_model]
            attention_weights: Attention weights [batch_size, num_heads, seq_len, seq_len]
        """
        batch_size = query.shape[0]
        
        # Linear projections
        Q = np.matmul(query, self.W_q)
        K = np.matmul(key, self.W_k)
        V = np.matmul(value, self.W_v)
        
        # Split into multiple heads
        Q = self.split_heads(Q)  # [batch, num_heads, seq_len, d_k]
        K = self.split_heads(K)
        V = self.split_heads(V)
        
        # Scaled dot-product attention for each head
        d_k = self.d_k
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(d_k)
        
        if mask is not None:
            scores = np.where(mask == 0, -1e9, scores)
        
        attention_weights = softmax(scores, axis=-1)
        attention_output = np.matmul(attention_weights, V)
        
        # Combine heads
        output = self.combine_heads(attention_output)
        
        # Final linear projection
        output = np.matmul(output, self.W_o)
        
        return output, attention_weights


# ============================================================================
# POSITIONAL ENCODING
# ============================================================================

class PositionalEncoding:
    """
    Positional Encoding for Transformers.
    
    Since transformers don't have recurrence or convolution, we need to
    inject information about the position of tokens in the sequence.
    
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    Args:
        d_model: Model dimension (default: 512)
        max_len: Maximum sequence length (default: 5000)
        dropout: Dropout rate (default: 0.1)
    
    Example:
        >>> from ilovetools.ml.attention import PositionalEncoding
        >>> pe = PositionalEncoding(d_model=512, max_len=100)
        >>> x = np.random.randn(2, 10, 512)  # [batch, seq_len, d_model]
        >>> x_with_pos = pe(x)
        >>> print(f"Output shape: {x_with_pos.shape}")  # [2, 10, 512]
    
    Reference:
        Vaswani et al., "Attention Is All You Need" (2017)
    """
    
    def __init__(self, d_model: int = 512, max_len: int = 5000, dropout: float = 0.1):
        self.d_model = d_model
        self.dropout = dropout
        
        # Create positional encoding matrix
        pe = np.zeros((max_len, d_model))
        position = np.arange(0, max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        self.pe = pe[np.newaxis, :, :]  # [1, max_len, d_model]
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Add positional encoding to input.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
        
        Returns:
            Output with positional encoding [batch_size, seq_len, d_model]
        """
        seq_len = x.shape[1]
        x = x + self.pe[:, :seq_len, :]
        return x


# ============================================================================
# FEED-FORWARD NETWORK
# ============================================================================

class FeedForward:
    """
    Position-wise Feed-Forward Network.
    
    FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
    
    Args:
        d_model: Model dimension (default: 512)
        d_ff: Feed-forward dimension (default: 2048)
        dropout: Dropout rate (default: 0.1)
    
    Example:
        >>> from ilovetools.ml.attention import FeedForward
        >>> ffn = FeedForward(d_model=512, d_ff=2048)
        >>> x = np.random.randn(2, 10, 512)
        >>> output = ffn(x)
        >>> print(f"Output shape: {output.shape}")  # [2, 10, 512]
    """
    
    def __init__(self, d_model: int = 512, d_ff: int = 2048, dropout: float = 0.1):
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = dropout
        
        # Initialize weights
        self.W_1 = np.random.randn(d_model, d_ff) * 0.01
        self.b_1 = np.zeros(d_ff)
        self.W_2 = np.random.randn(d_ff, d_model) * 0.01
        self.b_2 = np.zeros(d_model)
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through feed-forward network.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
        
        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        # First linear layer + ReLU
        hidden = np.maximum(0, np.matmul(x, self.W_1) + self.b_1)
        
        # Second linear layer
        output = np.matmul(hidden, self.W_2) + self.b_2
        
        return output


# ============================================================================
# LAYER NORMALIZATION
# ============================================================================

def layer_norm(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Layer Normalization.
    
    Args:
        x: Input tensor [batch_size, seq_len, d_model]
        eps: Small constant for numerical stability
    
    Returns:
        Normalized tensor [batch_size, seq_len, d_model]
    
    Example:
        >>> from ilovetools.ml.attention import layer_norm
        >>> x = np.random.randn(2, 10, 512)
        >>> x_norm = layer_norm(x)
        >>> print(f"Mean: {x_norm.mean():.6f}, Std: {x_norm.std():.6f}")
    """
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True)
    return (x - mean) / (std + eps)


# ============================================================================
# TRANSFORMER ENCODER LAYER
# ============================================================================

class TransformerEncoderLayer:
    """
    Transformer Encoder Layer.
    
    Consists of:
    1. Multi-head self-attention
    2. Add & Norm (residual connection + layer normalization)
    3. Feed-forward network
    4. Add & Norm
    
    Args:
        d_model: Model dimension (default: 512)
        num_heads: Number of attention heads (default: 8)
        d_ff: Feed-forward dimension (default: 2048)
        dropout: Dropout rate (default: 0.1)
    
    Example:
        >>> from ilovetools.ml.attention import TransformerEncoderLayer
        >>> encoder_layer = TransformerEncoderLayer(d_model=512, num_heads=8)
        >>> x = np.random.randn(2, 10, 512)
        >>> output = encoder_layer(x)
        >>> print(f"Output shape: {output.shape}")  # [2, 10, 512]
    
    Reference:
        Vaswani et al., "Attention Is All You Need" (2017)
    """
    
    def __init__(self, d_model: int = 512, num_heads: int = 8, 
                 d_ff: int = 2048, dropout: float = 0.1):
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.dropout = dropout
    
    def __call__(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Forward pass through encoder layer.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            mask: Optional mask [batch_size, seq_len, seq_len]
        
        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        # Self-attention + residual + norm
        attn_output, _ = self.self_attention(x, x, x, mask)
        x = layer_norm(x + attn_output)
        
        # Feed-forward + residual + norm
        ff_output = self.feed_forward(x)
        x = layer_norm(x + ff_output)
        
        return x


# ============================================================================
# TRANSFORMER DECODER LAYER
# ============================================================================

class TransformerDecoderLayer:
    """
    Transformer Decoder Layer.
    
    Consists of:
    1. Masked multi-head self-attention
    2. Add & Norm
    3. Multi-head cross-attention (with encoder output)
    4. Add & Norm
    5. Feed-forward network
    6. Add & Norm
    
    Args:
        d_model: Model dimension (default: 512)
        num_heads: Number of attention heads (default: 8)
        d_ff: Feed-forward dimension (default: 2048)
        dropout: Dropout rate (default: 0.1)
    
    Example:
        >>> from ilovetools.ml.attention import TransformerDecoderLayer
        >>> decoder_layer = TransformerDecoderLayer(d_model=512, num_heads=8)
        >>> x = np.random.randn(2, 10, 512)
        >>> encoder_output = np.random.randn(2, 10, 512)
        >>> output = decoder_layer(x, encoder_output)
        >>> print(f"Output shape: {output.shape}")  # [2, 10, 512]
    """
    
    def __init__(self, d_model: int = 512, num_heads: int = 8,
                 d_ff: int = 2048, dropout: float = 0.1):
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.dropout = dropout
    
    def __call__(self, x: np.ndarray, encoder_output: np.ndarray,
                 self_mask: Optional[np.ndarray] = None,
                 cross_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Forward pass through decoder layer.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            encoder_output: Encoder output [batch_size, seq_len, d_model]
            self_mask: Mask for self-attention (causal mask)
            cross_mask: Mask for cross-attention
        
        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        # Masked self-attention + residual + norm
        self_attn_output, _ = self.self_attention(x, x, x, self_mask)
        x = layer_norm(x + self_attn_output)
        
        # Cross-attention + residual + norm
        cross_attn_output, _ = self.cross_attention(x, encoder_output, encoder_output, cross_mask)
        x = layer_norm(x + cross_attn_output)
        
        # Feed-forward + residual + norm
        ff_output = self.feed_forward(x)
        x = layer_norm(x + ff_output)
        
        return x


# ============================================================================
# FULL TRANSFORMER MODEL
# ============================================================================

class Transformer:
    """
    Full Transformer Model (Encoder-Decoder).
    
    The complete transformer architecture as described in "Attention Is All You Need".
    
    Args:
        d_model: Model dimension (default: 512)
        num_heads: Number of attention heads (default: 8)
        num_encoder_layers: Number of encoder layers (default: 6)
        num_decoder_layers: Number of decoder layers (default: 6)
        d_ff: Feed-forward dimension (default: 2048)
        max_len: Maximum sequence length (default: 5000)
        dropout: Dropout rate (default: 0.1)
    
    Example:
        >>> from ilovetools.ml.attention import Transformer
        >>> transformer = Transformer(d_model=512, num_heads=8, num_encoder_layers=6)
        >>> src = np.random.randn(2, 10, 512)  # Source sequence
        >>> tgt = np.random.randn(2, 8, 512)   # Target sequence
        >>> output = transformer(src, tgt)
        >>> print(f"Output shape: {output.shape}")  # [2, 8, 512]
    
    Applications:
        - Machine Translation (original use case)
        - Text Summarization
        - Question Answering
        - Image Captioning
    
    Reference:
        Vaswani et al., "Attention Is All You Need" (2017)
    """
    
    def __init__(self, d_model: int = 512, num_heads: int = 8,
                 num_encoder_layers: int = 6, num_decoder_layers: int = 6,
                 d_ff: int = 2048, max_len: int = 5000, dropout: float = 0.1):
        self.d_model = d_model
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        # Encoder layers
        self.encoder_layers = [
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_encoder_layers)
        ]
        
        # Decoder layers
        self.decoder_layers = [
            TransformerDecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_decoder_layers)
        ]
    
    def encode(self, src: np.ndarray, src_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Encode source sequence.
        
        Args:
            src: Source sequence [batch_size, src_len, d_model]
            src_mask: Source mask [batch_size, src_len, src_len]
        
        Returns:
            Encoder output [batch_size, src_len, d_model]
        """
        # Add positional encoding
        x = self.pos_encoding(src)
        
        # Pass through encoder layers
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        
        return x
    
    def decode(self, tgt: np.ndarray, encoder_output: np.ndarray,
               tgt_mask: Optional[np.ndarray] = None,
               src_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Decode target sequence.
        
        Args:
            tgt: Target sequence [batch_size, tgt_len, d_model]
            encoder_output: Encoder output [batch_size, src_len, d_model]
            tgt_mask: Target mask (causal mask) [batch_size, tgt_len, tgt_len]
            src_mask: Source mask [batch_size, src_len, src_len]
        
        Returns:
            Decoder output [batch_size, tgt_len, d_model]
        """
        # Add positional encoding
        x = self.pos_encoding(tgt)
        
        # Pass through decoder layers
        for layer in self.decoder_layers:
            x = layer(x, encoder_output, tgt_mask, src_mask)
        
        return x
    
    def __call__(self, src: np.ndarray, tgt: np.ndarray,
                 src_mask: Optional[np.ndarray] = None,
                 tgt_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Forward pass through transformer.
        
        Args:
            src: Source sequence [batch_size, src_len, d_model]
            tgt: Target sequence [batch_size, tgt_len, d_model]
            src_mask: Source mask
            tgt_mask: Target mask (causal mask)
        
        Returns:
            Output [batch_size, tgt_len, d_model]
        """
        encoder_output = self.encode(src, src_mask)
        decoder_output = self.decode(tgt, encoder_output, tgt_mask, src_mask)
        return decoder_output


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_causal_mask(seq_len: int) -> np.ndarray:
    """
    Create causal mask for decoder (prevents attending to future positions).
    
    Args:
        seq_len: Sequence length
    
    Returns:
        Causal mask [seq_len, seq_len]
    
    Example:
        >>> from ilovetools.ml.attention import create_causal_mask
        >>> mask = create_causal_mask(5)
        >>> print(mask)
        [[1. 0. 0. 0. 0.]
         [1. 1. 0. 0. 0.]
         [1. 1. 1. 0. 0.]
         [1. 1. 1. 1. 0.]
         [1. 1. 1. 1. 1.]]
    """
    mask = np.tril(np.ones((seq_len, seq_len)))
    return mask


def create_padding_mask(seq: np.ndarray, pad_idx: int = 0) -> np.ndarray:
    """
    Create padding mask (masks out padding tokens).
    
    Args:
        seq: Sequence [batch_size, seq_len]
        pad_idx: Padding token index
    
    Returns:
        Padding mask [batch_size, 1, seq_len]
    
    Example:
        >>> from ilovetools.ml.attention import create_padding_mask
        >>> seq = np.array([[1, 2, 3, 0, 0], [1, 2, 0, 0, 0]])
        >>> mask = create_padding_mask(seq, pad_idx=0)
        >>> print(mask.shape)  # [2, 1, 5]
    """
    mask = (seq != pad_idx).astype(float)
    return mask[:, np.newaxis, :]


__all__ = [
    'scaled_dot_product_attention',
    'MultiHeadAttention',
    'PositionalEncoding',
    'FeedForward',
    'TransformerEncoderLayer',
    'TransformerDecoderLayer',
    'Transformer',
    'layer_norm',
    'softmax',
    'create_causal_mask',
    'create_padding_mask',
]
