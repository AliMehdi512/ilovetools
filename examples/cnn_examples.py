"""
Comprehensive Examples: Convolutional Neural Networks (CNNs)

This file demonstrates all CNN components with practical examples
and real-world applications.

Author: Ali Mehdi
Date: February 22, 2026
"""

import numpy as np
from ilovetools.ml.cnn import (
    conv2d,
    depthwise_conv2d,
    max_pool2d,
    avg_pool2d,
    global_avg_pool2d,
    relu,
    leaky_relu,
    gelu,
    swish,
    batch_norm2d,
    dropout,
    LeNet5,
)

print("=" * 80)
print("CONVOLUTIONAL NEURAL NETWORKS (CNNs) - COMPREHENSIVE EXAMPLES")
print("=" * 80)
print()

# ============================================================================
# EXAMPLE 1: Basic 2D Convolution
# ============================================================================
print("EXAMPLE 1: Basic 2D Convolution")
print("-" * 80)

# RGB image (batch=1, channels=3, height=32, width=32)
input_image = np.random.randn(1, 3, 32, 32)
print(f"Input image shape: {input_image.shape}")
print(f"  Batch size: 1")
print(f"  Channels: 3 (RGB)")
print(f"  Height: 32 pixels")
print(f"  Width: 32 pixels")
print()

# 64 filters, each 3x3
kernel = np.random.randn(64, 3, 3, 3)
print(f"Kernel shape: {kernel.shape}")
print(f"  Output channels: 64")
print(f"  Input channels: 3")
print(f"  Kernel size: 3x3")
print()

# Apply convolution
output = conv2d(input_image, kernel, stride=1, padding=1)
print(f"Output shape: {output.shape}")
print(f"  Feature maps: 64")
print(f"  Spatial size: 32x32 (preserved with padding=1)")
print()

print("Key Concepts:")
print("  • Receptive Field: Each output neuron 'sees' a 3x3 region")
print("  • Parameter Sharing: Same 64 filters applied across entire image")
print("  • Translation Invariance: Detects features anywhere in image")
print("  • Parameters: 64 × 3 × 3 × 3 = 1,728 weights")

print("\n✓ Example 1 completed\n")

# ============================================================================
# EXAMPLE 2: Depthwise Separable Convolution (MobileNet)
# ============================================================================
print("EXAMPLE 2: Depthwise Separable Convolution (MobileNet)")
print("-" * 80)

input_tensor = np.random.randn(1, 32, 64, 64)
print(f"Input shape: {input_tensor.shape}")
print()

# Depthwise convolution (one filter per channel)
dw_kernel = np.random.randn(32, 1, 3, 3)
output_dw = depthwise_conv2d(input_tensor, dw_kernel, stride=1, padding=1)

print(f"Depthwise output shape: {output_dw.shape}")
print()

print("Parameter Comparison:")
print("  Standard Conv (32→64, 3x3): 32 × 64 × 3 × 3 = 18,432 params")
print("  Depthwise Conv (32, 3x3): 32 × 1 × 3 × 3 = 288 params")
print("  Reduction: 64x fewer parameters!")
print()

print("Benefits:")
print("  ✓ 8-9x fewer parameters")
print("  ✓ 8-9x less computation (FLOPs)")
print("  ✓ Perfect for mobile/edge devices")
print("  ✓ Used in MobileNet, EfficientNet")

print("\n✓ Example 2 completed\n")

# ============================================================================
# EXAMPLE 3: Max Pooling
# ============================================================================
print("EXAMPLE 3: Max Pooling")
print("-" * 80)

feature_maps = np.random.randn(1, 64, 32, 32)
print(f"Input shape: {feature_maps.shape}")
print()

# 2x2 max pooling with stride 2
pooled = max_pool2d(feature_maps, kernel_size=2, stride=2)
print(f"After max pooling: {pooled.shape}")
print(f"  Spatial reduction: 32×32 → 16×16 (4x reduction)")
print()

print("Benefits:")
print("  • Translation invariance (small shifts don't affect output)")
print("  • Reduces spatial dimensions (less computation)")
print("  • Reduces overfitting (fewer parameters in next layer)")
print("  • Preserves dominant features (max operation)")
print()

# Demonstrate max pooling behavior
demo_input = np.array([[[[1, 2, 3, 4],
                          [5, 6, 7, 8],
                          [9, 10, 11, 12],
                          [13, 14, 15, 16]]]])
demo_output = max_pool2d(demo_input, kernel_size=2, stride=2)
print("Max pooling example:")
print("Input:")
print(demo_input[0, 0])
print("\nOutput (2x2 pooling):")
print(demo_output[0, 0])
print("  Top-left: max(1,2,5,6) = 6")
print("  Top-right: max(3,4,7,8) = 8")
print("  Bottom-left: max(9,10,13,14) = 14")
print("  Bottom-right: max(11,12,15,16) = 16")

print("\n✓ Example 3 completed\n")

# ============================================================================
# EXAMPLE 4: Global Average Pooling
# ============================================================================
print("EXAMPLE 4: Global Average Pooling")
print("-" * 80)

deep_features = np.random.randn(1, 2048, 7, 7)
print(f"Input shape: {deep_features.shape}")
print(f"  Channels: 2048")
print(f"  Spatial size: 7×7")
print()

# Global average pooling
global_pooled = global_avg_pool2d(deep_features)
print(f"After global average pooling: {global_pooled.shape}")
print(f"  Each feature map → single value")
print()

print("Parameter Comparison:")
print("  Fully Connected: 7×7×2048 → 1000 = 100,352,000 params")
print("  Global Avg Pool: 0 parameters!")
print()

print("Benefits:")
print("  ✓ No parameters to learn")
print("  ✓ Dramatically reduces overfitting")
print("  ✓ Works with any input size (resolution invariant)")
print("  ✓ Used in ResNet, EfficientNet, MobileNet")

print("\n✓ Example 4 completed\n")

# ============================================================================
# EXAMPLE 5: Batch Normalization
# ============================================================================
print("EXAMPLE 5: Batch Normalization")
print("-" * 80)

# Batch of feature maps
batch_features = np.random.randn(32, 64, 28, 28)
print(f"Input shape: {batch_features.shape}")
print(f"  Batch size: 32")
print(f"  Channels: 64")
print()

# Batch norm parameters
gamma = np.ones(64)  # Scale
beta = np.zeros(64)  # Shift

# Apply batch normalization
normalized = batch_norm2d(batch_features, gamma, beta)
print(f"Output shape: {normalized.shape}")
print()

print("Formula: y = γ × (x - μ) / √(σ² + ε) + β")
print()

print("Benefits:")
print("  ✓ 10-15x faster training")
print("  ✓ Enables higher learning rates")
print("  ✓ Reduces internal covariate shift")
print("  ✓ Regularization effect (reduces overfitting)")
print("  ✓ Less sensitive to weight initialization")
print()

print("Impact:")
print("  • Enabled training of very deep networks (100+ layers)")
print("  • Standard in modern CNNs (ResNet, Inception, EfficientNet)")
print("  • One of the most important innovations in deep learning")

print("\n✓ Example 5 completed\n")

# ============================================================================
# EXAMPLE 6: Activation Functions
# ============================================================================
print("EXAMPLE 6: Activation Functions")
print("-" * 80)

x = np.array([-2, -1, 0, 1, 2])
print(f"Input: {x}")
print()

# ReLU
relu_out = relu(x)
print(f"ReLU: {relu_out}")
print("  Formula: f(x) = max(0, x)")
print("  Most popular in CNNs")
print()

# Leaky ReLU
leaky_out = leaky_relu(x, alpha=0.01)
print(f"Leaky ReLU: {leaky_out}")
print("  Formula: f(x) = max(0.01x, x)")
print("  Prevents dying ReLU problem")
print()

# GELU
gelu_out = gelu(x)
print(f"GELU: {gelu_out}")
print("  Formula: f(x) = x × Φ(x)")
print("  Used in BERT, GPT, Vision Transformers")
print()

# Swish
swish_out = swish(x)
print(f"Swish: {swish_out}")
print("  Formula: f(x) = x × sigmoid(x)")
print("  Used in EfficientNet, MobileNetV3")

print("\n✓ Example 6 completed\n")

# ============================================================================
# EXAMPLE 7: Dropout Regularization
# ============================================================================
print("EXAMPLE 7: Dropout Regularization")
print("-" * 80)

features = np.ones((32, 512))
print(f"Input shape: {features.shape}")
print()

# Apply dropout (training mode)
dropped = dropout(features, p=0.5, training=True)
zero_ratio = (dropped == 0).sum() / dropped.size
print(f"Dropout (p=0.5, training=True):")
print(f"  Zero ratio: {zero_ratio:.2%}")
print(f"  Expected: ~50%")
print()

# Inference mode (no dropout)
inference = dropout(features, p=0.5, training=False)
print(f"Dropout (p=0.5, training=False):")
print(f"  Output unchanged (no dropout during inference)")
print()

print("Benefits:")
print("  • Prevents overfitting (co-adaptation of neurons)")
print("  • Ensemble effect (trains multiple sub-networks)")
print("  • Simple and effective regularization")
print()

print("Typical Values:")
print("  • Hidden layers: p=0.5")
print("  • Input layer: p=0.2")
print("  • Convolutional layers: p=0.1-0.3")

print("\n✓ Example 7 completed\n")

# ============================================================================
# EXAMPLE 8: LeNet-5 Architecture
# ============================================================================
print("EXAMPLE 8: LeNet-5 Architecture (1998)")
print("-" * 80)

model = LeNet5()
print(f"Model: {model.name}")
print()

# MNIST image (28x28 grayscale, padded to 32x32)
mnist_image = np.random.randn(1, 1, 32, 32)
print(f"Input: {mnist_image.shape}")
print("  MNIST handwritten digit (0-9)")
print()

# Forward pass
output = model.forward(mnist_image)
print(f"Output: {output.shape}")
print()

print("Architecture:")
print("  Input (32×32×1)")
print("  → Conv1 (6 filters, 5×5) → ReLU → AvgPool(2×2)")
print("  → Conv2 (16 filters, 5×5) → ReLU → AvgPool(2×2)")
print("  → FC(120) → FC(84) → FC(10)")
print()

print("Historical Impact:")
print("  • First successful CNN for real-world problems")
print("  • MNIST digit recognition: 99.2% accuracy")
print("  • Used for check reading in banks")
print("  • Established convolution + pooling pattern")
print("  • Foundation for modern CNNs")
print()

print("Parameters: ~60,000")

print("\n✓ Example 8 completed\n")

# ============================================================================
# EXAMPLE 9: Complete CNN Pipeline
# ============================================================================
print("EXAMPLE 9: Complete CNN Pipeline")
print("-" * 80)

print("Building a modern CNN for image classification...")
print()

# Input: RGB image
x = np.random.randn(1, 3, 224, 224)
print(f"1. Input: {x.shape} (224×224 RGB image)")

# Conv Block 1
conv1_kernel = np.random.randn(64, 3, 3, 3)
x = conv2d(x, conv1_kernel, stride=1, padding=1)
print(f"2. Conv2D (64 filters, 3×3): {x.shape}")

gamma1 = np.ones(64)
beta1 = np.zeros(64)
x = batch_norm2d(x, gamma1, beta1)
print(f"3. BatchNorm: {x.shape}")

x = relu(x)
print(f"4. ReLU: {x.shape}")

x = max_pool2d(x, kernel_size=2, stride=2)
print(f"5. MaxPool (2×2): {x.shape}")

# Conv Block 2
conv2_kernel = np.random.randn(128, 64, 3, 3)
x = conv2d(x, conv2_kernel, stride=1, padding=1)
print(f"6. Conv2D (128 filters, 3×3): {x.shape}")

gamma2 = np.ones(128)
beta2 = np.zeros(128)
x = batch_norm2d(x, gamma2, beta2)
print(f"7. BatchNorm: {x.shape}")

x = relu(x)
print(f"8. ReLU: {x.shape}")

x = max_pool2d(x, kernel_size=2, stride=2)
print(f"9. MaxPool (2×2): {x.shape}")

# Global pooling
x = global_avg_pool2d(x)
print(f"10. GlobalAvgPool: {x.shape}")

print()
print("Final output: 128 features ready for classification!")
print()

print("This pipeline demonstrates:")
print("  ✓ Spatial hierarchy (224→112→56→1)")
print("  ✓ Feature hierarchy (3→64→128 channels)")
print("  ✓ Modern best practices (BatchNorm, ReLU, GlobalAvgPool)")

print("\n✓ Example 9 completed\n")

# ============================================================================
# EXAMPLE 10: Real-World Applications
# ============================================================================
print("EXAMPLE 10: Real-World Applications")
print("-" * 80)

print("Image Classification:")
print("  • ImageNet: 1000 classes, 14M images")
print("  • CIFAR-10: 10 classes, 60K images")
print("  • MNIST: Handwritten digits, 99.7% accuracy")
print("  • Medical imaging: X-ray, MRI, CT scan analysis")
print()

print("Object Detection:")
print("  • YOLO: Real-time object detection (30+ FPS)")
print("  • Faster R-CNN: High-accuracy detection")
print("  • SSD: Single Shot Detector")
print("  • Applications: Autonomous driving, surveillance")
print()

print("Semantic Segmentation:")
print("  • U-Net: Medical image segmentation")
print("  • DeepLab: Scene understanding")
print("  • Mask R-CNN: Instance segmentation")
print("  • Applications: Self-driving cars, medical diagnosis")
print()

print("Face Recognition:")
print("  • FaceNet: Face verification and recognition")
print("  • ArcFace: State-of-the-art face recognition")
print("  • DeepFace: Facebook's face recognition")
print("  • Applications: Security, authentication, photo tagging")
print()

print("Autonomous Driving:")
print("  • Tesla Autopilot: 8 cameras, 360° vision")
print("  • Waymo: LiDAR + camera fusion")
print("  • Lane detection, traffic sign recognition")
print("  • Pedestrian detection, obstacle avoidance")
print()

print("Style Transfer & Generation:")
print("  • Neural Style Transfer: Artistic style application")
print("  • CycleGAN: Image-to-image translation")
print("  • SRGAN: Super-resolution (upscaling)")
print("  • Applications: Photo editing, art generation")

print("\n✓ Example 10 completed\n")

# ============================================================================
# EXAMPLE 11: CNN Evolution Timeline
# ============================================================================
print("EXAMPLE 11: CNN Evolution Timeline")
print("-" * 80)

print("1998 - LeNet-5:")
print("  • First successful CNN")
print("  • 60K parameters")
print("  • MNIST: 99.2% accuracy")
print()

print("2012 - AlexNet:")
print("  • ImageNet breakthrough")
print("  • 60M parameters, 8 layers")
print("  • Top-5 error: 15.3% (vs 26% previous)")
print("  • Introduced ReLU, Dropout, GPU training")
print()

print("2014 - VGG:")
print("  • Very deep (16-19 layers)")
print("  • 138M parameters")
print("  • Simple 3×3 conv architecture")
print("  • Top-5 error: 7.3%")
print()

print("2015 - ResNet:")
print("  • Residual connections (skip connections)")
print("  • 152 layers possible!")
print("  • Top-5 error: 3.57% (superhuman)")
print("  • Solved vanishing gradient problem")
print()

print("2017 - DenseNet:")
print("  • Dense connections (every layer to every other)")
print("  • Fewer parameters, better gradient flow")
print("  • State-of-the-art efficiency")
print()

print("2019 - EfficientNet:")
print("  • Compound scaling (depth, width, resolution)")
print("  • 66M parameters")
print("  • Top-1 accuracy: 84.3%")
print("  • 8.4x smaller, 6.1x faster than best CNN")

print("\n✓ Example 11 completed\n")

# ============================================================================
# EXAMPLE 12: Parameter Efficiency Comparison
# ============================================================================
print("EXAMPLE 12: Parameter Efficiency Comparison")
print("-" * 80)

print("Standard Convolution (32→64 channels, 3×3 kernel):")
print("  Parameters: 32 × 64 × 3 × 3 = 18,432")
print("  FLOPs (56×56 image): 18,432 × 56 × 56 = 57.8M")
print()

print("Depthwise Separable Convolution:")
print("  Depthwise: 32 × 1 × 3 × 3 = 288 params")
print("  Pointwise: 32 × 64 × 1 × 1 = 2,048 params")
print("  Total: 2,336 params (7.9x reduction!)")
print("  FLOPs: 7.3M (7.9x reduction!)")
print()

print("Fully Connected vs Global Average Pooling:")
print("  FC Layer (7×7×2048 → 1000): 100,352,000 params")
print("  Global Avg Pool: 0 params (infinite reduction!)")
print()

print("Key Insight:")
print("  Modern CNNs achieve better accuracy with fewer parameters")
print("  through architectural innovations!")

print("\n✓ Example 12 completed\n")

# ============================================================================
# SUMMARY
# ============================================================================
print("=" * 80)
print("ALL EXAMPLES COMPLETED SUCCESSFULLY! ✓")
print("=" * 80)
print()
print("Summary of what we covered:")
print("1. ✓ Basic 2D Convolution")
print("2. ✓ Depthwise Separable Convolution")
print("3. ✓ Max Pooling")
print("4. ✓ Global Average Pooling")
print("5. ✓ Batch Normalization")
print("6. ✓ Activation Functions")
print("7. ✓ Dropout Regularization")
print("8. ✓ LeNet-5 Architecture")
print("9. ✓ Complete CNN Pipeline")
print("10. ✓ Real-World Applications")
print("11. ✓ CNN Evolution Timeline")
print("12. ✓ Parameter Efficiency")
print()
print("You now have a complete understanding of CNNs!")
print()
print("Next steps:")
print("- Implement AlexNet, VGG, ResNet")
print("- Train on CIFAR-10, ImageNet")
print("- Build object detection models (YOLO)")
print("- Explore semantic segmentation (U-Net)")
print("- Apply to your own image classification tasks")
print()
print("GitHub: https://github.com/AliMehdi512/ilovetools")
print("Install: pip install ilovetools")
