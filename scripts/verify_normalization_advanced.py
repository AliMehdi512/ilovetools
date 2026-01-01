"""
Verification script to test advanced normalization techniques imports
"""

print("Testing advanced normalization imports from ilovetools.ml.normalization_advanced...")

try:
    from ilovetools.ml.normalization_advanced import (
        batch_norm_forward,
        layer_norm_forward,
        instance_norm_forward,
        group_norm_forward,
        weight_norm,
        spectral_norm,
        initialize_norm_params,
        compute_norm_stats,
        batch_norm,
        layer_norm,
        instance_norm,
        group_norm,
    )
    print("✓ All normalization functions imported successfully")
    
    # Test basic functionality
    import numpy as np
    
    # Test Batch Normalization
    x_cnn = np.random.randn(32, 64, 28, 28)
    gamma, beta = initialize_norm_params(64)
    
    output, running_mean, running_var = batch_norm_forward(
        x_cnn, gamma, beta, training=True
    )
    print(f"✓ Batch Normalization works: {x_cnn.shape} -> {output.shape}")
    print(f"  Running mean shape: {running_mean.shape}")
    print(f"  Running var shape: {running_var.shape}")
    
    # Check normalization
    output_mean = np.mean(output, axis=(0, 2, 3))
    output_var = np.var(output, axis=(0, 2, 3))
    print(f"  Output mean (should be ~0): {np.mean(np.abs(output_mean)):.6f}")
    print(f"  Output var (should be ~1): {np.mean(output_var):.6f}")
    
    # Test Layer Normalization
    x_transformer = np.random.randn(32, 10, 512)
    gamma_ln, beta_ln = initialize_norm_params(512)
    
    output_ln = layer_norm_forward(x_transformer, gamma_ln, beta_ln)
    print(f"✓ Layer Normalization works: {x_transformer.shape} -> {output_ln.shape}")
    
    # Check per-sample normalization
    sample_mean = np.mean(output_ln[0, 0])
    sample_var = np.var(output_ln[0, 0])
    print(f"  Sample mean (should be ~0): {sample_mean:.6f}")
    print(f"  Sample var (should be ~1): {sample_var:.6f}")
    
    # Test Instance Normalization
    output_in = instance_norm_forward(x_cnn, gamma, beta)
    print(f"✓ Instance Normalization works: {x_cnn.shape} -> {output_in.shape}")
    
    # Test Group Normalization
    output_gn = group_norm_forward(x_cnn, gamma, beta, num_groups=32)
    print(f"✓ Group Normalization works: {x_cnn.shape} -> {output_gn.shape}")
    
    # Test Weight Normalization
    weight = np.random.randn(64, 128)
    w_normalized, g = weight_norm(weight, dim=1)
    print(f"✓ Weight Normalization works: {weight.shape} -> {w_normalized.shape}")
    print(f"  Norm shape: {g.shape}")
    
    # Check unit norm
    row_norms = np.linalg.norm(w_normalized, axis=1)
    print(f"  Row norms (should be ~1): mean={np.mean(row_norms):.6f}, std={np.std(row_norms):.6f}")
    
    # Test Spectral Normalization
    w_spectral = spectral_norm(weight, num_iterations=1)
    print(f"✓ Spectral Normalization works: {weight.shape} -> {w_spectral.shape}")
    
    # Check largest singular value
    u, s, v = np.linalg.svd(w_spectral, full_matrices=False)
    print(f"  Largest singular value (should be ~1): {s[0]:.6f}")
    
    # Test Compute Norm Stats
    mean_batch, var_batch = compute_norm_stats(x_cnn, norm_type='batch')
    print(f"✓ Compute norm stats works:")
    print(f"  Batch norm stats: mean {mean_batch.shape}, var {var_batch.shape}")
    
    mean_layer, var_layer = compute_norm_stats(x_cnn, norm_type='layer')
    print(f"  Layer norm stats: mean {mean_layer.shape}, var {var_layer.shape}")
    
    # Test with FC layer
    x_fc = np.random.randn(32, 256)
    gamma_fc, beta_fc = initialize_norm_params(256)
    output_fc, _, _ = batch_norm_forward(x_fc, gamma_fc, beta_fc, training=True)
    print(f"✓ Batch Normalization (FC) works: {x_fc.shape} -> {output_fc.shape}")
    
    print("\n✅ All verifications passed!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
