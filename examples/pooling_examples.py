"""
Comprehensive Examples: Pooling Layers

This file demonstrates all pooling operations with practical examples and use cases.

Author: Ali Mehdi
Date: January 17, 2026
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ilovetools.ml.pooling import (
    MaxPool1D,
    MaxPool2D,
    AvgPool1D,
    AvgPool2D,
    GlobalMaxPool,
    GlobalAvgPool,
    AdaptiveMaxPool,
    AdaptiveAvgPool,
)

print("=" * 80)
print("POOLING LAYERS - COMPREHENSIVE EXAMPLES")
print("=" * 80)
print()

# ============================================================================
# EXAMPLE 1: MaxPool2D - Image Classification
# ============================================================================
print("EXAMPLE 1: MaxPool2D - Image Classification (CNN)")
print("-" * 80)

pool = MaxPool2D(pool_size=2, stride=2)

print("Simulating CNN feature extraction:")
print(f"Pool size: {pool.pool_h}x{pool.pool_w}")
print(f"Stride: {pool.stride_h}x{pool.stride_w}")
print()

# Simulate feature maps from conv layer
feature_maps = np.random.randn(32, 64, 28, 28)  # (batch, channels, h, w)
print(f"Input shape: {feature_maps.shape}")

output = pool.forward(feature_maps)
print(f"Output shape: {output.shape}")
print(f"Spatial reduction: {28}x{28} → {output.shape[2]}x{output.shape[3]}")
print(f"Parameters: 0 (pooling has no learnable parameters)")

print("\n✓ Example 1 completed\n")

# ============================================================================
# EXAMPLE 2: AvgPool2D - Smooth Downsampling
# ============================================================================
print("EXAMPLE 2: AvgPool2D - Smooth Downsampling")
print("-" * 80)

pool = AvgPool2D(pool_size=2, stride=2)

print("Average pooling for smoother features:")
x = np.array([[[[1, 2, 3, 4],
                 [5, 6, 7, 8],
                 [9, 10, 11, 12],
                 [13, 14, 15, 16]]]])

output = pool.forward(x)
print(f"Input:\n{x[0, 0]}")
print(f"\nOutput:\n{output[0, 0]}")
print(f"\nEach output value is average of 2x2 window")

print("\n✓ Example 2 completed\n")

# ============================================================================
# EXAMPLE 3: GlobalAvgPool - Classification Head
# ============================================================================
print("EXAMPLE 3: GlobalAvgPool - Classification Head")
print("-" * 80)

pool = GlobalAvgPool()

print("Global Average Pooling for classification:")
print("Used in ResNet, MobileNet, EfficientNet")
print()

# Simulate final conv layer output
feature_maps = np.random.randn(32, 512, 7, 7)
print(f"Input shape: {feature_maps.shape}")

output = pool.forward(feature_maps)
print(f"Output shape: {output.shape}")
print(f"Reduction: 7x7 spatial → single value per channel")
print(f"Ready for fully connected layer (32, 512) → (32, num_classes)")

print("\n✓ Example 3 completed\n")

# ============================================================================
# EXAMPLE 4: GlobalMaxPool - Feature Extraction
# ============================================================================
print("EXAMPLE 4: GlobalMaxPool - Feature Extraction")
print("-" * 80)

pool = GlobalMaxPool()

print("Global Max Pooling for strongest features:")
feature_maps = np.random.randn(16, 256, 14, 14)
print(f"Input shape: {feature_maps.shape}")

output = pool.forward(feature_maps)
print(f"Output shape: {output.shape}")
print(f"Extracts maximum activation per channel")
print(f"Useful for: Object detection, feature matching")

print("\n✓ Example 4 completed\n")

# ============================================================================
# EXAMPLE 5: AdaptiveMaxPool - Variable Input Sizes
# ============================================================================
print("EXAMPLE 5: AdaptiveMaxPool - Variable Input Sizes")
print("-" * 80)

pool = AdaptiveMaxPool(output_size=(7, 7))

print("Adaptive pooling for variable input sizes:")
print(f"Target output size: 7x7")
print()

inputs = [
    np.random.randn(2, 512, 14, 14),
    np.random.randn(2, 512, 28, 28),
    np.random.randn(2, 512, 35, 35),
]

for i, x in enumerate(inputs):
    output = pool.forward(x)
    print(f"Input {i+1}: {x.shape} → Output: {output.shape}")

print(f"\nAll outputs have same size: (batch, 512, 7, 7)")
print(f"Useful for: Transfer learning, variable image sizes")

print("\n✓ Example 5 completed\n")

# ============================================================================
# EXAMPLE 6: AdaptiveAvgPool - Flexible Architecture
# ============================================================================
print("EXAMPLE 6: AdaptiveAvgPool - Flexible Architecture")
print("-" * 80)

pool = AdaptiveAvgPool(output_size=(1, 1))

print("Adaptive average pooling as global pooling:")
x = np.random.randn(8, 2048, 14, 14)
print(f"Input shape: {x.shape}")

output = pool.forward(x)
print(f"Output shape: {output.shape}")
print(f"Equivalent to GlobalAvgPool")
print(f"Can specify any output size: (1,1), (7,7), etc.")

print("\n✓ Example 6 completed\n")

# ============================================================================
# EXAMPLE 7: MaxPool1D - Sequence Processing
# ============================================================================
print("EXAMPLE 7: MaxPool1D - Sequence Processing")
print("-" * 80)

pool = MaxPool1D(pool_size=2, stride=2)

print("1D pooling for sequences (text, time series):")
sequence = np.random.randn(16, 128, 100)  # (batch, channels, length)
print(f"Input shape: {sequence.shape}")

output = pool.forward(sequence)
print(f"Output shape: {output.shape}")
print(f"Sequence length: 100 → {output.shape[2]}")
print(f"Use case: Text CNN, audio processing")

print("\n✓ Example 7 completed\n")

# ============================================================================
# EXAMPLE 8: AvgPool1D - Smooth Sequence Downsampling
# ============================================================================
print("EXAMPLE 8: AvgPool1D - Smooth Sequence Downsampling")
print("-" * 80)

pool = AvgPool1D(pool_size=3, stride=2)

print("Average pooling for sequences:")
sequence = np.array([[[1, 2, 3, 4, 5, 6, 7, 8]]])
print(f"Input: {sequence[0, 0]}")

output = pool.forward(sequence)
print(f"Output: {output[0, 0]}")
print(f"Smoother than max pooling")

print("\n✓ Example 8 completed\n")

# ============================================================================
# EXAMPLE 9: Comparing Max vs Average Pooling
# ============================================================================
print("EXAMPLE 9: Comparing Max vs Average Pooling")
print("-" * 80)

x = np.array([[[[1, 2, 3, 4],
                 [5, 6, 7, 8],
                 [9, 10, 11, 12],
                 [13, 14, 15, 16]]]])

max_pool = MaxPool2D(pool_size=2, stride=2)
avg_pool = AvgPool2D(pool_size=2, stride=2)

max_out = max_pool.forward(x)
avg_out = avg_pool.forward(x)

print("Input:")
print(x[0, 0])
print("\nMax Pooling Output:")
print(max_out[0, 0])
print("\nAverage Pooling Output:")
print(avg_out[0, 0])
print("\nMax pooling: Preserves strong features (edges)")
print("Avg pooling: Smoother, preserves more information")

print("\n✓ Example 9 completed\n")

# ============================================================================
# EXAMPLE 10: CNN Architecture with Pooling
# ============================================================================
print("EXAMPLE 10: CNN Architecture with Pooling")
print("-" * 80)

print("Typical CNN architecture:")
print()

x = np.random.randn(32, 3, 224, 224)  # ImageNet input
print(f"Input: {x.shape}")

# Conv1 + Pool1
print(f"Conv1 (64 filters, 3x3) → (32, 64, 224, 224)")
pool1 = MaxPool2D(pool_size=2, stride=2)
x = np.random.randn(32, 64, 224, 224)
x = pool1.forward(x)
print(f"Pool1 (2x2, stride 2) → {x.shape}")

# Conv2 + Pool2
print(f"Conv2 (128 filters, 3x3) → (32, 128, 112, 112)")
pool2 = MaxPool2D(pool_size=2, stride=2)
x = np.random.randn(32, 128, 112, 112)
x = pool2.forward(x)
print(f"Pool2 (2x2, stride 2) → {x.shape}")

# Conv3 + Pool3
print(f"Conv3 (256 filters, 3x3) → (32, 256, 56, 56)")
pool3 = MaxPool2D(pool_size=2, stride=2)
x = np.random.randn(32, 256, 56, 56)
x = pool3.forward(x)
print(f"Pool3 (2x2, stride 2) → {x.shape}")

# Global pooling
print(f"Conv4 (512 filters, 3x3) → (32, 512, 28, 28)")
global_pool = GlobalAvgPool()
x = np.random.randn(32, 512, 28, 28)
x = global_pool.forward(x)
print(f"GlobalAvgPool → {x.shape}")

print(f"\nFinal: (32, 512) ready for classification")

print("\n✓ Example 10 completed\n")

# ============================================================================
# EXAMPLE 11: Pooling for Different Tasks
# ============================================================================
print("EXAMPLE 11: Pooling for Different Tasks")
print("-" * 80)

print("Task-specific pooling strategies:")
print()

print("1. Image Classification (ResNet):")
print("   - MaxPool after first conv")
print("   - GlobalAvgPool before FC layer")
print("   - Reduces overfitting")
print()

print("2. Object Detection (YOLO, Faster R-CNN):")
print("   - MaxPool for feature extraction")
print("   - Preserve spatial information")
print("   - Multiple scales")
print()

print("3. Semantic Segmentation (U-Net):")
print("   - MaxPool in encoder")
print("   - Upsampling in decoder")
print("   - Skip connections")
print()

print("4. Text Classification (TextCNN):")
print("   - MaxPool1D over sequence")
print("   - GlobalMaxPool for final features")
print("   - Captures key phrases")
print()

print("5. Time Series (WaveNet):")
print("   - AvgPool1D for smoothing")
print("   - Preserve temporal patterns")
print()

print("\n✓ Example 11 completed\n")

# ============================================================================
# EXAMPLE 12: Pooling Parameters Impact
# ============================================================================
print("EXAMPLE 12: Pooling Parameters Impact")
print("-" * 80)

x = np.random.randn(1, 1, 16, 16)

configs = [
    (2, 2, "Standard: 2x2, stride 2"),
    (3, 2, "Overlapping: 3x3, stride 2"),
    (2, 1, "Heavy overlap: 2x2, stride 1"),
    (4, 4, "Aggressive: 4x4, stride 4"),
]

print("Different pooling configurations:")
print()

for pool_size, stride, desc in configs:
    pool = MaxPool2D(pool_size=pool_size, stride=stride)
    output = pool.forward(x)
    print(f"{desc}")
    print(f"  Input: {x.shape} → Output: {output.shape}")
    print(f"  Reduction: {x.shape[2]/output.shape[2]:.1f}x")
    print()

print("\n✓ Example 12 completed\n")

# ============================================================================
# EXAMPLE 13: Memory and Computation Savings
# ============================================================================
print("EXAMPLE 13: Memory and Computation Savings")
print("-" * 80)

print("Pooling reduces memory and computation:")
print()

# Before pooling
h, w = 224, 224
channels = 64
print(f"Before pooling: {channels} x {h} x {w} = {channels * h * w:,} values")

# After pooling
pool = MaxPool2D(pool_size=2, stride=2)
h_out, w_out = h // 2, w // 2
print(f"After pooling: {channels} x {h_out} x {w_out} = {channels * h_out * w_out:,} values")

reduction = (channels * h * w) / (channels * h_out * w_out)
print(f"\nMemory reduction: {reduction:.1f}x")
print(f"Computation reduction: ~{reduction:.1f}x for next layer")

print("\n✓ Example 13 completed\n")

# ============================================================================
# EXAMPLE 14: Translation Invariance
# ============================================================================
print("EXAMPLE 14: Translation Invariance")
print("-" * 80)

print("Pooling provides translation invariance:")
print()

# Original image
x1 = np.array([[[[0, 0, 1, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0]]]])

# Shifted image
x2 = np.array([[[[0, 1, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0]]]])

pool = MaxPool2D(pool_size=2, stride=2)

out1 = pool.forward(x1)
out2 = pool.forward(x2)

print("Original:")
print(x1[0, 0])
print(f"Pooled: {out1[0, 0]}")
print()

print("Shifted:")
print(x2[0, 0])
print(f"Pooled: {out2[0, 0]}")
print()

print("Both produce same pooled output!")
print("Pooling makes network robust to small translations")

print("\n✓ Example 14 completed\n")

# ============================================================================
# EXAMPLE 15: Pooling Selection Guide
# ============================================================================
print("EXAMPLE 15: Pooling Selection Guide")
print("-" * 80)

print("When to use each pooling type:")
print()

print("MaxPool2D:")
print("  ✓ Image classification (CNNs)")
print("  ✓ Feature detection (edges, textures)")
print("  ✓ When strong features matter")
print("  ✓ Standard choice for most tasks")
print()

print("AvgPool2D:")
print("  ✓ Smooth downsampling")
print("  ✓ When all features matter equally")
print("  ✓ Less common than MaxPool")
print()

print("GlobalMaxPool:")
print("  ✓ Feature extraction")
print("  ✓ Object detection")
print("  ✓ When strongest activation matters")
print()

print("GlobalAvgPool:")
print("  ✓ Classification (ResNet, MobileNet)")
print("  ✓ Reduces overfitting vs FC layers")
print("  ✓ Modern best practice")
print("  ✓ No parameters to learn")
print()

print("AdaptiveMaxPool:")
print("  ✓ Variable input sizes")
print("  ✓ Transfer learning")
print("  ✓ Fixed output size needed")
print()

print("AdaptiveAvgPool:")
print("  ✓ Flexible architectures")
print("  ✓ Multi-scale features")
print("  ✓ SPP (Spatial Pyramid Pooling)")
print()

print("MaxPool1D / AvgPool1D:")
print("  ✓ Text classification (CNN)")
print("  ✓ Time series analysis")
print("  ✓ Audio processing")
print("  ✓ Sequence modeling")

print("\n✓ Example 15 completed\n")

# ============================================================================
# SUMMARY
# ============================================================================
print("=" * 80)
print("ALL EXAMPLES COMPLETED SUCCESSFULLY! ✓")
print("=" * 80)
print()
print("Summary of what we covered:")
print("1. ✓ MaxPool2D - Image Classification")
print("2. ✓ AvgPool2D - Smooth Downsampling")
print("3. ✓ GlobalAvgPool - Classification Head")
print("4. ✓ GlobalMaxPool - Feature Extraction")
print("5. ✓ AdaptiveMaxPool - Variable Input Sizes")
print("6. ✓ AdaptiveAvgPool - Flexible Architecture")
print("7. ✓ MaxPool1D - Sequence Processing")
print("8. ✓ AvgPool1D - Smooth Sequence Downsampling")
print("9. ✓ Comparing Max vs Average Pooling")
print("10. ✓ CNN Architecture with Pooling")
print("11. ✓ Pooling for Different Tasks")
print("12. ✓ Pooling Parameters Impact")
print("13. ✓ Memory and Computation Savings")
print("14. ✓ Translation Invariance")
print("15. ✓ Pooling Selection Guide")
print()
print("You now have a complete understanding of pooling layers!")
print()
print("Next steps:")
print("- Choose pooling based on your task")
print("- Experiment with pool sizes and strides")
print("- Use GlobalAvgPool for classification")
print("- Try AdaptivePool for variable inputs")
