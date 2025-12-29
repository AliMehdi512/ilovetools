"""
Verification script to test CNN operations imports
"""

print("Testing CNN imports from ilovetools.ml.cnn...")

try:
    from ilovetools.ml.cnn import (
        conv2d,
        conv2d_fast,
        max_pool2d,
        avg_pool2d,
        global_avg_pool2d,
        global_max_pool2d,
        im2col,
        col2im,
        depthwise_conv2d,
        separable_conv2d,
        calculate_output_size,
        maxpool2d,
        avgpool2d,
        global_avgpool,
        global_maxpool,
        depthwise_conv,
        separable_conv,
    )
    print("✓ All CNN functions imported successfully")
    
    # Test basic functionality
    import numpy as np
    
    # Test 2D Convolution
    input = np.random.randn(8, 3, 28, 28)
    kernel = np.random.randn(32, 3, 3, 3)
    
    output = conv2d(input, kernel, stride=1, padding='same')
    print(f"✓ Conv2D works: {input.shape} -> {output.shape}")
    
    # Test with stride
    output_stride = conv2d(input, kernel, stride=2, padding=1)
    print(f"✓ Conv2D with stride=2: {input.shape} -> {output_stride.shape}")
    
    # Test Max Pooling
    pool_input = np.random.randn(8, 64, 28, 28)
    pool_output = max_pool2d(pool_input, pool_size=2, stride=2)
    print(f"✓ Max Pooling works: {pool_input.shape} -> {pool_output.shape}")
    
    # Test Average Pooling
    avg_output = avg_pool2d(pool_input, pool_size=2, stride=2)
    print(f"✓ Avg Pooling works: {pool_input.shape} -> {avg_output.shape}")
    
    # Test Global Average Pooling
    global_input = np.random.randn(8, 512, 7, 7)
    global_output = global_avg_pool2d(global_input)
    print(f"✓ Global Avg Pooling works: {global_input.shape} -> {global_output.shape}")
    
    # Test Im2Col
    im2col_input = np.random.randn(2, 3, 5, 5)
    col = im2col(im2col_input, kernel_h=3, kernel_w=3, stride_h=1, stride_w=1)
    print(f"✓ Im2Col works: {im2col_input.shape} -> {col.shape}")
    
    # Test Col2Im
    img = col2im(col, im2col_input.shape, kernel_h=3, kernel_w=3, stride_h=1, stride_w=1)
    print(f"✓ Col2Im works: {col.shape} -> {img.shape}")
    
    # Test Depthwise Convolution
    dw_input = np.random.randn(8, 32, 56, 56)
    dw_kernel = np.random.randn(32, 1, 3, 3)
    dw_output = depthwise_conv2d(dw_input, dw_kernel, stride=1, padding='same')
    print(f"✓ Depthwise Conv works: {dw_input.shape} -> {dw_output.shape}")
    
    # Test Separable Convolution
    pw_kernel = np.random.randn(64, 32, 1, 1)
    sep_output = separable_conv2d(dw_input, dw_kernel, pw_kernel, stride=1, padding='same')
    print(f"✓ Separable Conv works: {dw_input.shape} -> {sep_output.shape}")
    
    # Test Calculate Output Size
    out_size = calculate_output_size(224, 3, stride=2, padding=1)
    print(f"✓ Calculate output size works: 224 -> {out_size}")
    
    # Test edge detection
    edge_input = np.zeros((1, 1, 5, 5))
    edge_input[0, 0, 2, :] = 1
    edge_kernel = np.array([[[[-1, -1, -1], [0, 0, 0], [1, 1, 1]]]])
    edge_output = conv2d(edge_input, edge_kernel, stride=1, padding=0)
    print(f"✓ Edge detection works: {edge_input.shape} -> {edge_output.shape}")
    
    print("\n✅ All verifications passed!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
