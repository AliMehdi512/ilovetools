"""
Verification script to test attention mechanisms imports
"""

print("Testing attention imports from ilovetools.ml.attention...")

try:
    from ilovetools.ml.attention import (
        scaled_dot_product_attention,
        multi_head_attention,
        self_attention,
        multi_head_self_attention,
        cross_attention,
        create_padding_mask,
        create_causal_mask,
        create_look_ahead_mask,
        positional_encoding,
        learned_positional_encoding,
        attention_score_visualization,
        softmax,
        dropout,
        sdp_attention,
        mha,
        self_attn,
        cross_attn,
        pos_encoding,
        causal_mask,
        padding_mask,
    )
    print("✓ All attention functions imported successfully")
    
    # Test basic functionality
    import numpy as np
    
    # Test Scaled Dot-Product Attention
    q = np.random.randn(32, 10, 64)
    k = np.random.randn(32, 10, 64)
    v = np.random.randn(32, 10, 64)
    
    output, weights = scaled_dot_product_attention(q, k, v)
    print(f"✓ Scaled Dot-Product Attention works: {q.shape} -> {output.shape}")
    print(f"  Attention weights shape: {weights.shape}")
    print(f"  Weights sum to 1: {np.allclose(np.sum(weights, axis=-1), 1.0)}")
    
    # Test Multi-Head Attention
    output, weights = multi_head_attention(q, k, v, num_heads=8, d_model=64)
    print(f"✓ Multi-Head Attention works: {q.shape} -> {output.shape}")
    print(f"  Multi-head weights shape: {weights.shape}")
    
    # Test Self-Attention
    x = np.random.randn(32, 10, 512)
    output, weights = self_attention(x, d_model=512)
    print(f"✓ Self-Attention works: {x.shape} -> {output.shape}")
    
    # Test Cross-Attention
    query = np.random.randn(32, 10, 512)
    context = np.random.randn(32, 20, 512)
    output, weights = cross_attention(query, context, d_model=512)
    print(f"✓ Cross-Attention works: query {query.shape}, context {context.shape} -> {output.shape}")
    
    # Test Positional Encoding
    pos_enc = positional_encoding(seq_len=100, d_model=512)
    print(f"✓ Positional Encoding works: shape {pos_enc.shape}")
    print(f"  Values range: [{np.min(pos_enc):.4f}, {np.max(pos_enc):.4f}]")
    
    # Test Causal Mask
    mask = create_causal_mask(seq_len=10)
    print(f"✓ Causal Mask works: shape {mask.shape}")
    
    # Test Padding Mask
    seq = np.array([[1, 2, 3, 0, 0], [1, 2, 0, 0, 0]])
    mask = create_padding_mask(seq, pad_token=0)
    print(f"✓ Padding Mask works: shape {mask.shape}")
    
    # Test Softmax
    x = np.random.randn(32, 10, 10)
    result = softmax(x, axis=-1)
    print(f"✓ Softmax works: {x.shape} -> {result.shape}")
    print(f"  Sums to 1: {np.allclose(np.sum(result, axis=-1), 1.0)}")
    
    print("\n✅ All verifications passed!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
