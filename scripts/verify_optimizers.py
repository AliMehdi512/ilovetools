"""
Verification script to test optimizer imports
"""

print("Testing optimizer imports from ilovetools.ml.optimizers...")

try:
    from ilovetools.ml.optimizers import (
        adam_optimizer,
        adamw_optimizer,
        adamax_optimizer,
        nadam_optimizer,
        amsgrad_optimizer,
        rmsprop_optimizer,
        rmsprop_momentum_optimizer,
        radam_optimizer,
        lamb_optimizer,
        lookahead_optimizer,
        adabelief_optimizer,
        create_optimizer_state,
        get_optimizer_function,
    )
    print("✓ All optimizers imported successfully from ilovetools.ml.optimizers")
    
    # Test basic functionality
    import numpy as np
    params = np.array([1.0, 2.0, 3.0])
    grads = np.array([0.1, 0.2, 0.3])
    m = np.zeros_like(params)
    v = np.zeros_like(params)
    
    new_params, m, v = adam_optimizer(params, grads, m, v, t=1)
    print(f"✓ Adam optimizer works: {params} -> {new_params}")
    
    print("\n✅ All verifications passed!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
except Exception as e:
    print(f"❌ Error: {e}")
