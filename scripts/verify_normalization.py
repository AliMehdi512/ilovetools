"""
Verification script to test normalization imports
"""

print("Testing normalization imports from ilovetools.ml.normalization...")

try:
    from ilovetools.ml.normalization import (
        batch_normalization,
        layer_normalization,
        group_normalization,
        instance_normalization,
        weight_normalization,
        batch_norm_forward,
        layer_norm_forward,
        create_normalization_params,
        apply_normalization,
        batchnorm,
        layernorm,
        groupnorm,
        instancenorm,
        weightnorm,
    )
    print("✓ All normalization functions imported successfully")
    
    # Test basic functionality
    import numpy as np
    
    # Test Batch Normalization
    x = np.random.randn(32, 64)
    gamma = np.ones(64)
    beta = np.zeros(64)
    
    out, running_mean, running_var = batch_normalization(x, gamma, beta, training=True)
    print(f"✓ Batch Normalization works: {x.shape} -> {out.shape}")
    print(f"  Mean: {np.mean(out):.4f}, Var: {np.var(out):.4f}")
    
    # Test Layer Normalization
    out = layer_normalization(x, gamma, beta)
    print(f"✓ Layer Normalization works: {x.shape} -> {out.shape}")
    print(f"  Mean: {np.mean(out):.4f}, Var: {np.var(out):.4f}")
    
    # Test Group Normalization
    x_4d = np.random.randn(8, 64, 32, 32)
    gamma_4d = np.ones(64)
    beta_4d = np.zeros(64)
    out = group_normalization(x_4d, gamma_4d, beta_4d, num_groups=32)
    print(f"✓ Group Normalization works: {x_4d.shape} -> {out.shape}")
    
    # Test Instance Normalization
    out = instance_normalization(x_4d, gamma_4d, beta_4d)
    print(f"✓ Instance Normalization works: {x_4d.shape} -> {out.shape}")
    
    # Test Weight Normalization
    w = np.random.randn(512, 256)
    w_norm, g = weight_normalization(w)
    print(f"✓ Weight Normalization works: {w.shape} -> {w_norm.shape}")
    
    # Test utilities
    params = create_normalization_params(64, 'batch')
    print(f"✓ create_normalization_params works: {list(params.keys())}")
    
    out = apply_normalization(x, 'layer', gamma, beta)
    print(f"✓ apply_normalization works: {x.shape} -> {out.shape}")
    
    print("\n✅ All verifications passed!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
