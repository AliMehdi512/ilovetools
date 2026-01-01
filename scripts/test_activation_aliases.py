"""
Quick test to verify activation function aliases work
"""

print("Testing activation function aliases...")

try:
    from ilovetools.ml.activations import (
        relu,
        sigmoid,
        tanh,
        softmax,
        gelu,
        swish
    )
    print("✓ All activation aliases imported successfully")
    
    import numpy as np
    
    # Test ReLU
    x = np.array([-2, -1, 0, 1, 2])
    output = relu(x)
    print(f"✓ relu({x}) = {output}")
    assert np.array_equal(output, np.array([0, 0, 0, 1, 2])), "ReLU output incorrect"
    
    # Test Sigmoid
    output = sigmoid(x)
    print(f"✓ sigmoid({x}) = {output}")
    assert output.shape == x.shape, "Sigmoid shape incorrect"
    
    # Test Tanh
    output = tanh(x)
    print(f"✓ tanh({x}) = {output}")
    assert output.shape == x.shape, "Tanh shape incorrect"
    
    # Test Softmax
    output = softmax(x)
    print(f"✓ softmax({x}) = {output}")
    assert np.isclose(np.sum(output), 1.0), "Softmax should sum to 1"
    
    # Test GELU
    output = gelu(x)
    print(f"✓ gelu({x}) = {output}")
    assert output.shape == x.shape, "GELU shape incorrect"
    
    # Test Swish
    output = swish(x)
    print(f"✓ swish({x}) = {output}")
    assert output.shape == x.shape, "Swish shape incorrect"
    
    print("\n✅ All activation function aliases work correctly!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
