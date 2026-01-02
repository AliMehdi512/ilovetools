"""
Verify that all regularization functions are accessible from ilovetools.ml
"""

print("Testing imports from ilovetools.ml.regularization...")
print("=" * 60)

try:
    # Test dropout imports
    from ilovetools.ml.regularization import (
        dropout,
        spatial_dropout,
        variational_dropout,
        dropconnect,
    )
    print("✓ Dropout functions imported successfully")
    
    # Test L1 regularization imports
    from ilovetools.ml.regularization import (
        l1_regularization,
        l1_gradient,
        l1_penalty,
    )
    print("✓ L1 regularization functions imported successfully")
    
    # Test L2 regularization imports
    from ilovetools.ml.regularization import (
        l2_regularization,
        l2_gradient,
        l2_penalty,
        weight_decay,
    )
    print("✓ L2 regularization functions imported successfully")
    
    # Test Elastic Net imports
    from ilovetools.ml.regularization import (
        elastic_net_regularization,
        elastic_net_gradient,
        elastic_net_penalty,
    )
    print("✓ Elastic Net functions imported successfully")
    
    # Test Early Stopping imports
    from ilovetools.ml.regularization import (
        EarlyStopping,
        early_stopping_monitor,
        should_stop_early,
    )
    print("✓ Early Stopping utilities imported successfully")
    
    # Test utility imports
    from ilovetools.ml.regularization import (
        apply_regularization,
        compute_regularization_gradient,
        get_dropout_rate_schedule,
    )
    print("✓ Utility functions imported successfully")
    
    # Test aliases
    from ilovetools.ml.regularization import (
        inverted_dropout,
        dropout_mask,
    )
    print("✓ Aliases imported successfully")
    
    print("\n" + "=" * 60)
    print("ALL IMPORTS SUCCESSFUL! ✓")
    print("=" * 60)
    
    # Quick functionality test
    print("\nQuick functionality test:")
    import numpy as np
    
    # Test dropout
    x = np.random.randn(10, 20)
    output, mask = dropout(x, dropout_rate=0.5, training=True, seed=42)
    print(f"✓ dropout: input shape {x.shape} -> output shape {output.shape}")
    
    # Test L1 regularization
    weights = np.random.randn(10, 10)
    penalty = l1_regularization(weights, lambda_=0.01)
    print(f"✓ l1_regularization: penalty = {penalty:.6f}")
    
    # Test L2 regularization
    penalty = l2_regularization(weights, lambda_=0.01)
    print(f"✓ l2_regularization: penalty = {penalty:.6f}")
    
    # Test Early Stopping
    early_stopping = EarlyStopping(patience=5)
    print(f"✓ EarlyStopping: initialized with patience={early_stopping.patience}")
    
    print("\n" + "=" * 60)
    print("FUNCTIONALITY TEST PASSED! ✓")
    print("=" * 60)
    
except ImportError as e:
    print(f"\n❌ Import Error: {e}")
    import traceback
    traceback.print_exc()
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
