"""
Comprehensive Examples: Convolution Operations

This file demonstrates all convolution types with practical examples and use cases.

Author: Ali Mehdi
Date: January 22, 2026
"""

import numpy as np
from ilovetools.ml.convolution import (
    Conv1D,
    Conv2D,
    Conv3D,
    DepthwiseConv2D,
    SeparableConv2D,
    DilatedConv2D,
    TransposedConv2D,
    Conv1x1,
)

print("=" * 80)
print("CONVOLUTION OPERATIONS - COMPREHENSIVE EXAMPLES")
print("=" * 80)
print()

# ============================================================================
# EXAMPLE 1: Conv2D - Image Classification (Standard CNN)
# ============================================================================
print("EXAMPLE 1: Conv2D - Image Classification")
print("-" * 80)

conv = Conv2D(in_channels=3, out_channels=64, kernel_size=3, padding='same')

print("Standard 2D convolution for images:")
print(f"Input channels: {conv.in_channels} (RGB)")
print(f"Output channels: {conv.out_channels} (filters)")
print(f"Kernel size: {conv.kernel_h}x{conv.kernel_w}")
print(f"Padding: 'same' (preserve spatial dimensions)")
print()

# Simulate ImageNet input
x = np.random.randn(32, 3, 224, 224)
print(f"Input shape: {x.shape}")

output = conv.forward(x)
print(f"Output shape: {output.shape}")
print(f"Spatial dimensions preserved: {output.shape[2]}x{output.shape[3]}")
print(f"Parameters: {conv.out_channels * conv.in_channels * conv.kernel_h * conv.kernel_w:,}")

print("\n✓ Example 1 completed\n")

# ============================================================================
# EXAMPLE 2: Conv1D - Text Classification
# ============================================================================
print("EXAMPLE 2: Conv1D - Text Classification (TextCNN)")
print("-" * 80)

conv = Conv1D(in_channels=128, out_channels=256, kernel_size=3, padding=1)

print("1D convolution for sequences:")
print(f"Input channels: {conv.in_channels} (embedding dim)")
print(f"Output channels: {conv.out_channels} (filters)")
print(f"Kernel size: {conv.kernel_size} (n-gram size)")
print()

# Simulate text embeddings
x = np.random.randn(32, 128, 100)  # (batch, embedding_dim, seq_len)
print(f"Input shape: {x.shape}")
print(f"Batch: 32 sentences")
print(f"Embedding: 128-dim word vectors")
print(f"Sequence length: 100 words")
print()

output = conv.forward(x)
print(f"Output shape: {output.shape}")
print(f"Captures 3-gram features")

print("\n✓ Example 2 completed\n")

# ============================================================================
# EXAMPLE 3: Conv3D - Video Classification
# ============================================================================
print("EXAMPLE 3: Conv3D - Video Classification")
print("-" * 80)

conv = Conv3D(in_channels=3, out_channels=64, kernel_size=3, padding=1)

print("3D convolution for videos:")
print(f"Input channels: {conv.in_channels} (RGB)")
print(f"Output channels: {conv.out_channels} (filters)")
print(f"Kernel size: {conv.kernel_d}x{conv.kernel_h}x{conv.kernel_w}")
print()

# Simulate video input
x = np.random.randn(8, 3, 16, 112, 112)  # (batch, channels, frames, height, width)
print(f"Input shape: {x.shape}")
print(f"Batch: 8 videos")
print(f"Frames: 16 frames per video")
print(f"Resolution: 112x112")
print()

output = conv.forward(x)
print(f"Output shape: {output.shape}")
print(f"Captures spatio-temporal features")

print("\n✓ Example 3 completed\n")

# ============================================================================
# EXAMPLE 4: DepthwiseConv2D - Efficient Mobile Networks
# ============================================================================
print("EXAMPLE 4: DepthwiseConv2D - Efficient Mobile Networks")
print("-" * 80)

conv = DepthwiseConv2D(in_channels=64, kernel_size=3, padding=1)

print("Depthwise convolution (spatial filtering per channel):")
print(f"Input channels: {conv.in_channels}")
print(f"Output channels: {conv.in_channels} (same)")
print(f"Kernel size: {conv.kernel_h}x{conv.kernel_w}")
print()

x = np.random.randn(32, 64, 56, 56)
print(f"Input shape: {x.shape}")

output = conv.forward(x)
print(f"Output shape: {output.shape}")

# Calculate parameter reduction
standard_params = 64 * 64 * 3 * 3
depthwise_params = 64 * 3 * 3
print(f"\nParameter comparison:")
print(f"Standard Conv2D: {standard_params:,} parameters")
print(f"DepthwiseConv2D: {depthwise_params:,} parameters")
print(f"Reduction: {standard_params / depthwise_params:.1f}x")

print("\n✓ Example 4 completed\n")

# ============================================================================
# EXAMPLE 5: SeparableConv2D - MobileNet Architecture
# ============================================================================
print("EXAMPLE 5: SeparableConv2D - MobileNet Architecture")
print("-" * 80)

conv = SeparableConv2D(in_channels=64, out_channels=128, kernel_size=3, padding=1)

print("Depthwise separable convolution:")
print("Step 1: Depthwise (spatial filtering)")
print("Step 2: Pointwise 1x1 (channel mixing)")
print()

x = np.random.randn(32, 64, 56, 56)
print(f"Input shape: {x.shape}")

output = conv.forward(x)
print(f"Output shape: {output.shape}")

# Calculate parameter reduction
standard_params = 128 * 64 * 3 * 3
separable_params = (64 * 3 * 3) + (64 * 128)
print(f"\nParameter comparison:")
print(f"Standard Conv2D: {standard_params:,} parameters")
print(f"SeparableConv2D: {separable_params:,} parameters")
print(f"Reduction: {standard_params / separable_params:.1f}x")
print(f"Used in: MobileNet, Xception, EfficientNet")

print("\n✓ Example 5 completed\n")

# ============================================================================
# EXAMPLE 6: DilatedConv2D - Semantic Segmentation
# ============================================================================
print("EXAMPLE 6: DilatedConv2D - Semantic Segmentation")
print("-" * 80)

conv = DilatedConv2D(in_channels=64, out_channels=64, kernel_size=3, dilation=2, padding=2)

print("Dilated convolution (expanded receptive field):")
print(f"Kernel size: {conv.kernel_h}x{conv.kernel_w}")
print(f"Dilation rate: {conv.dilation_h}")
print(f"Effective receptive field: 5x5")
print()

x = np.random.randn(16, 64, 56, 56)
print(f"Input shape: {x.shape}")

output = conv.forward(x)
print(f"Output shape: {output.shape}")
print(f"Spatial dimensions preserved")
print(f"\nReceptive field comparison:")
print(f"Standard 3x3: 3x3 receptive field")
print(f"Dilated 3x3 (rate=2): 5x5 receptive field")
print(f"Dilated 3x3 (rate=3): 7x7 receptive field")
print(f"Used in: DeepLab, WaveNet")

print("\n✓ Example 6 completed\n")

# ============================================================================
# EXAMPLE 7: TransposedConv2D - Image Upsampling
# ============================================================================
print("EXAMPLE 7: TransposedConv2D - Image Upsampling")
print("-" * 80)

conv = TransposedConv2D(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)

print("Transposed convolution (upsampling):")
print(f"Input channels: {conv.in_channels}")
print(f"Output channels: {conv.out_channels}")
print(f"Kernel size: {conv.kernel_h}x{conv.kernel_w}")
print(f"Stride: {conv.stride_h} (2x upsampling)")
print()

x = np.random.randn(32, 128, 28, 28)
print(f"Input shape: {x.shape}")

output = conv.forward(x)
print(f"Output shape: {output.shape}")
print(f"Upsampling: 28x28 → 56x56 (2x)")
print(f"Used in: U-Net, GANs, Segmentation decoders")

print("\n✓ Example 7 completed\n")

# ============================================================================
# EXAMPLE 8: Conv1x1 - Channel Reduction (Bottleneck)
# ============================================================================
print("EXAMPLE 8: Conv1x1 - Channel Reduction (Bottleneck)")
print("-" * 80)

conv = Conv1x1(in_channels=256, out_channels=64)

print("1x1 convolution (pointwise):")
print(f"Input channels: {conv.in_channels}")
print(f"Output channels: {conv.out_channels}")
print(f"Kernel size: 1x1")
print()

x = np.random.randn(32, 256, 56, 56)
print(f"Input shape: {x.shape}")

output = conv.forward(x)
print(f"Output shape: {output.shape}")
print(f"Spatial dimensions preserved: {output.shape[2]}x{output.shape[3]}")
print(f"Channels reduced: 256 → 64 (4x reduction)")
print(f"Used in: ResNet bottleneck, Inception, channel mixing")

print("\n✓ Example 8 completed\n")

# ============================================================================
# EXAMPLE 9: Building a CNN Architecture
# ============================================================================
print("EXAMPLE 9: Building a CNN Architecture")
print("-" * 80)

print("Typical CNN architecture:")
print()

x = np.random.randn(32, 3, 224, 224)
print(f"Input: {x.shape} (ImageNet)")

# Block 1
conv1 = Conv2D(3, 64, 3, padding=1)
x = conv1.forward(x)
print(f"Conv1 (64 filters, 3x3) → {x.shape}")

# Block 2
conv2 = Conv2D(64, 128, 3, stride=2, padding=1)
x = conv2.forward(x)
print(f"Conv2 (128 filters, 3x3, stride=2) → {x.shape}")

# Block 3
conv3 = Conv2D(128, 256, 3, stride=2, padding=1)
x = conv3.forward(x)
print(f"Conv3 (256 filters, 3x3, stride=2) → {x.shape}")

# Block 4
conv4 = Conv2D(256, 512, 3, stride=2, padding=1)
x = conv4.forward(x)
print(f"Conv4 (512 filters, 3x3, stride=2) → {x.shape}")

print(f"\nFinal feature maps: {x.shape}")
print(f"Ready for global pooling + classification")

print("\n✓ Example 9 completed\n")

# ============================================================================
# EXAMPLE 10: MobileNet-style Block
# ============================================================================
print("EXAMPLE 10: MobileNet-style Block")
print("-" * 80)

print("Depthwise Separable Convolution Block:")
print()

x = np.random.randn(32, 64, 56, 56)
print(f"Input: {x.shape}")

# Depthwise
depthwise = DepthwiseConv2D(64, 3, padding=1)
x = depthwise.forward(x)
print(f"Depthwise (3x3) → {x.shape}")

# Pointwise
pointwise = Conv1x1(64, 128)
x = pointwise.forward(x)
print(f"Pointwise (1x1) → {x.shape}")

print(f"\nEquivalent to SeparableConv2D(64, 128, 3)")
print(f"9x fewer parameters than standard Conv2D")

print("\n✓ Example 10 completed\n")

# ============================================================================
# EXAMPLE 11: U-Net Encoder-Decoder
# ============================================================================
print("EXAMPLE 11: U-Net Encoder-Decoder")
print("-" * 80)

print("U-Net architecture for segmentation:")
print()

x = np.random.randn(8, 3, 256, 256)
print(f"Input: {x.shape}")

# Encoder
print("\nEncoder (downsampling):")
conv_down1 = Conv2D(3, 64, 3, stride=2, padding=1)
x = conv_down1.forward(x)
print(f"Down1 → {x.shape}")

conv_down2 = Conv2D(64, 128, 3, stride=2, padding=1)
x = conv_down2.forward(x)
print(f"Down2 → {x.shape}")

conv_down3 = Conv2D(128, 256, 3, stride=2, padding=1)
x = conv_down3.forward(x)
print(f"Down3 → {x.shape}")

# Decoder
print("\nDecoder (upsampling):")
conv_up1 = TransposedConv2D(256, 128, 4, stride=2, padding=1)
x = conv_up1.forward(x)
print(f"Up1 → {x.shape}")

conv_up2 = TransposedConv2D(128, 64, 4, stride=2, padding=1)
x = conv_up2.forward(x)
print(f"Up2 → {x.shape}")

conv_up3 = TransposedConv2D(64, 3, 4, stride=2, padding=1)
x = conv_up3.forward(x)
print(f"Up3 → {x.shape}")

print(f"\nOutput: {x.shape} (same as input)")
print(f"Used for: Semantic segmentation, medical imaging")

print("\n✓ Example 11 completed\n")

# ============================================================================
# EXAMPLE 12: ResNet Bottleneck Block
# ============================================================================
print("EXAMPLE 12: ResNet Bottleneck Block")
print("-" * 80)

print("ResNet bottleneck with 1x1 convolutions:")
print()

x = np.random.randn(32, 256, 56, 56)
print(f"Input: {x.shape}")

# 1x1 reduce
conv1x1_reduce = Conv1x1(256, 64)
x = conv1x1_reduce.forward(x)
print(f"1x1 reduce (256→64) → {x.shape}")

# 3x3 conv
conv3x3 = Conv2D(64, 64, 3, padding=1)
x = conv3x3.forward(x)
print(f"3x3 conv → {x.shape}")

# 1x1 expand
conv1x1_expand = Conv1x1(64, 256)
x = conv1x1_expand.forward(x)
print(f"1x1 expand (64→256) → {x.shape}")

print(f"\nBottleneck reduces computation:")
print(f"Direct 3x3 (256→256): {256*256*3*3:,} params")
print(f"Bottleneck: {(256*64) + (64*64*3*3) + (64*256):,} params")
print(f"Reduction: ~3x")

print("\n✓ Example 12 completed\n")

# ============================================================================
# EXAMPLE 13: Inception Module
# ============================================================================
print("EXAMPLE 13: Inception Module")
print("-" * 80)

print("Inception module (multi-scale features):")
print()

x = np.random.randn(32, 256, 28, 28)
print(f"Input: {x.shape}")

# Branch 1: 1x1
branch1 = Conv1x1(256, 64)
out1 = branch1.forward(x)
print(f"Branch 1 (1x1) → {out1.shape}")

# Branch 2: 1x1 → 3x3
branch2_reduce = Conv1x1(256, 96)
branch2_conv = Conv2D(96, 128, 3, padding=1)
out2 = branch2_conv.forward(branch2_reduce.forward(x))
print(f"Branch 2 (1x1→3x3) → {out2.shape}")

# Branch 3: 1x1 → 5x5
branch3_reduce = Conv1x1(256, 16)
branch3_conv = Conv2D(16, 32, 5, padding=2)
out3 = branch3_conv.forward(branch3_reduce.forward(x))
print(f"Branch 3 (1x1→5x5) → {out3.shape}")

print(f"\nConcatenate all branches:")
print(f"Total output channels: 64 + 128 + 32 = 224")
print(f"Multi-scale feature extraction")

print("\n✓ Example 13 completed\n")

# ============================================================================
# EXAMPLE 14: Convolution Selection Guide
# ============================================================================
print("EXAMPLE 14: Convolution Selection Guide")
print("-" * 80)

print("When to use each convolution type:")
print()

print("Conv2D (Standard):")
print("  ✓ Image classification (AlexNet, VGG)")
print("  ✓ Object detection")
print("  ✓ General-purpose CNNs")
print("  ✓ When accuracy > efficiency")
print()

print("Conv1D:")
print("  ✓ Text classification (TextCNN)")
print("  ✓ Audio processing")
print("  ✓ Time series analysis")
print("  ✓ Sequence modeling")
print()

print("Conv3D:")
print("  ✓ Video classification")
print("  ✓ Action recognition")
print("  ✓ 3D medical imaging")
print("  ✓ Spatio-temporal features")
print()

print("DepthwiseConv2D:")
print("  ✓ Mobile networks (MobileNet)")
print("  ✓ Efficient architectures")
print("  ✓ Spatial filtering per channel")
print("  ✓ First step of separable conv")
print()

print("SeparableConv2D:")
print("  ✓ MobileNet, Xception, EfficientNet")
print("  ✓ Mobile/edge deployment")
print("  ✓ 9x parameter reduction")
print("  ✓ When efficiency matters")
print()

print("DilatedConv2D:")
print("  ✓ Semantic segmentation (DeepLab)")
print("  ✓ Audio generation (WaveNet)")
print("  ✓ Expand receptive field")
print("  ✓ No downsampling needed")
print()

print("TransposedConv2D:")
print("  ✓ U-Net decoder")
print("  ✓ GANs (generator)")
print("  ✓ Semantic segmentation")
print("  ✓ Image upsampling")
print()

print("Conv1x1:")
print("  ✓ ResNet bottleneck")
print("  ✓ Inception module")
print("  ✓ Channel reduction/expansion")
print("  ✓ Adding non-linearity")

print("\n✓ Example 14 completed\n")

# ============================================================================
# EXAMPLE 15: Parameter Comparison
# ============================================================================
print("EXAMPLE 15: Parameter Comparison")
print("-" * 80)

in_ch, out_ch, k = 64, 128, 3

print(f"Input channels: {in_ch}")
print(f"Output channels: {out_ch}")
print(f"Kernel size: {k}x{k}")
print()

# Standard Conv2D
standard_params = out_ch * in_ch * k * k
print(f"Standard Conv2D: {standard_params:,} parameters")

# Depthwise
depthwise_params = in_ch * k * k
print(f"DepthwiseConv2D: {depthwise_params:,} parameters")

# Separable
separable_params = (in_ch * k * k) + (in_ch * out_ch)
print(f"SeparableConv2D: {separable_params:,} parameters")

# 1x1
conv1x1_params = in_ch * out_ch
print(f"Conv1x1: {conv1x1_params:,} parameters")

print()
print("Reductions:")
print(f"Depthwise vs Standard: {standard_params / depthwise_params:.1f}x")
print(f"Separable vs Standard: {standard_params / separable_params:.1f}x")
print(f"1x1 vs Standard: {standard_params / conv1x1_params:.1f}x")

print("\n✓ Example 15 completed\n")

# ============================================================================
# SUMMARY
# ============================================================================
print("=" * 80)
print("ALL EXAMPLES COMPLETED SUCCESSFULLY! ✓")
print("=" * 80)
print()
print("Summary of what we covered:")
print("1. ✓ Conv2D - Image Classification")
print("2. ✓ Conv1D - Text Classification")
print("3. ✓ Conv3D - Video Classification")
print("4. ✓ DepthwiseConv2D - Efficient Mobile Networks")
print("5. ✓ SeparableConv2D - MobileNet Architecture")
print("6. ✓ DilatedConv2D - Semantic Segmentation")
print("7. ✓ TransposedConv2D - Image Upsampling")
print("8. ✓ Conv1x1 - Channel Reduction")
print("9. ✓ Building a CNN Architecture")
print("10. ✓ MobileNet-style Block")
print("11. ✓ U-Net Encoder-Decoder")
print("12. ✓ ResNet Bottleneck Block")
print("13. ✓ Inception Module")
print("14. ✓ Convolution Selection Guide")
print("15. ✓ Parameter Comparison")
print()
print("You now have a complete understanding of convolution operations!")
print()
print("Next steps:")
print("- Choose convolution based on your task")
print("- Use SeparableConv for mobile deployment")
print("- Use DilatedConv for segmentation")
print("- Use TransposedConv for upsampling")
print("- Combine different types for best results")
