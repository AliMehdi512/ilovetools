"""
Comprehensive Example: Batch Normalization and Layer Normalization

This example demonstrates all normalization techniques:
- Batch Normalization (BatchNorm1d, BatchNorm2d)
- Layer Normalization
- Group Normalization
- Instance Normalization
- Training vs Inference modes
- Complete neural network example
"""

import numpy as np
from ilovetools.ml.normalization import (
    BatchNorm1d,
    BatchNorm2d,
    LayerNorm,
    GroupNorm,
    InstanceNorm,
)

print("=" * 70)
print("BATCH NORMALIZATION AND LAYER NORMALIZATION - COMPREHENSIVE EXAMPLE")
print("=" * 70)

# ============================================================================
# 1. BATCH NORMALIZATION 1D (Fully Connected Layers)
# ============================================================================
print("\n1. BATCH NORMALIZATION 1D (Fully Connected Layers)")
print("-" * 70)

# Initialize
bn1d = BatchNorm1d(num_features=256)

# Training mode
x_train = np.random.randn(64, 256)
output_train = bn1d.forward(x_train, training=True)

print(f"Input shape: {x_train.shape}")
print(f"Output shape: {output_train.shape}")
print(f"Output mean: {np.mean(output_train, axis=0)[:5]}")  # First 5 features
print(f"Output std: {np.std(output_train, axis=0)[:5]}")
print(f"Batches tracked: {bn1d.num_batches_tracked}")

# Inference mode
x_test = np.random.randn(1, 256)
output_test = bn1d.forward(x_test, training=False)
print(f"\nInference output shape: {output_test.shape}")

# ============================================================================
# 2. BATCH NORMALIZATION 2D (Convolutional Layers)
# ============================================================================
print("\n2. BATCH NORMALIZATION 2D (Convolutional Layers)")
print("-" * 70)

# Initialize
bn2d = BatchNorm2d(num_features=64)

# Training mode
x_conv = np.random.randn(32, 64, 28, 28)  # (batch, channels, height, width)
output_conv = bn2d.forward(x_conv, training=True)

print(f"Input shape: {x_conv.shape}")
print(f"Output shape: {output_conv.shape}")

# Check normalization per channel
channel_means = np.mean(output_conv, axis=(0, 2, 3))
channel_stds = np.std(output_conv, axis=(0, 2, 3))

print(f"Channel means (first 5): {channel_means[:5]}")
print(f"Channel stds (first 5): {channel_stds[:5]}")

# ============================================================================
# 3. LAYER NORMALIZATION (RNNs and Transformers)
# ============================================================================
print("\n3. LAYER NORMALIZATION (RNNs and Transformers)")
print("-" * 70)

# Initialize
ln = LayerNorm(normalized_shape=512)

# Forward pass (same for training and inference)
x_seq = np.random.randn(32, 10, 512)  # (batch, seq_len, features)
output_seq = ln.forward(x_seq)

print(f"Input shape: {x_seq.shape}")
print(f"Output shape: {output_seq.shape}")

# Check normalization per sample
sample_means = np.mean(output_seq, axis=-1)
sample_stds = np.std(output_seq, axis=-1)

print(f"Sample means (first 5 samples, first timestep): {sample_means[:5, 0]}")
print(f"Sample stds (first 5 samples, first timestep): {sample_stds[:5, 0]}")

# ============================================================================
# 4. GROUP NORMALIZATION
# ============================================================================
print("\n4. GROUP NORMALIZATION")
print("-" * 70)

# Initialize (64 channels, 8 groups)
gn = GroupNorm(num_groups=8, num_channels=64)

# Forward pass
x_group = np.random.randn(32, 64, 28, 28)
output_group = gn.forward(x_group)

print(f"Input shape: {x_group.shape}")
print(f"Output shape: {output_group.shape}")
print(f"Number of groups: {gn.num_groups}")
print(f"Channels per group: {gn.num_channels // gn.num_groups}")

# ============================================================================
# 5. INSTANCE NORMALIZATION
# ============================================================================
print("\n5. INSTANCE NORMALIZATION")
print("-" * 70)

# Initialize
in_norm = InstanceNorm(num_features=64)

# Forward pass
x_instance = np.random.randn(32, 64, 28, 28)
output_instance = in_norm.forward(x_instance)

print(f"Input shape: {x_instance.shape}")
print(f"Output shape: {output_instance.shape}")

# Check normalization per instance and channel
instance_means = np.mean(output_instance, axis=(2, 3))
instance_stds = np.std(output_instance, axis=(2, 3))

print(f"Instance means (first sample, first 5 channels): {instance_means[0, :5]}")
print(f"Instance stds (first sample, first 5 channels): {instance_stds[0, :5]}")

# ============================================================================
# 6. COMPARISON: BATCH NORM VS LAYER NORM
# ============================================================================
print("\n6. COMPARISON: BATCH NORM VS LAYER NORM")
print("-" * 70)

# Same input for both
x_compare = np.random.randn(32, 128)

# Batch Normalization
bn_compare = BatchNorm1d(num_features=128)
output_bn = bn_compare.forward(x_compare, training=True)

# Layer Normalization
ln_compare = LayerNorm(normalized_shape=128)
output_ln = ln_compare.forward(x_compare)

print("Batch Normalization:")
print(f"  Normalizes across: batch dimension")
print(f"  Mean per feature: {np.mean(output_bn, axis=0)[:3]}")
print(f"  Std per feature: {np.std(output_bn, axis=0)[:3]}")

print("\nLayer Normalization:")
print(f"  Normalizes across: feature dimension")
print(f"  Mean per sample: {np.mean(output_ln, axis=1)[:3]}")
print(f"  Std per sample: {np.std(output_ln, axis=1)[:3]}")

# ============================================================================
# 7. COMPLETE NEURAL NETWORK EXAMPLE
# ============================================================================
print("\n7. COMPLETE NEURAL NETWORK EXAMPLE")
print("-" * 70)

print("Building a simple neural network with normalization...")

# Network architecture
input_size = 784  # 28x28 images
hidden_size = 256
output_size = 10
batch_size = 64

# Initialize layers
bn_layer1 = BatchNorm1d(num_features=hidden_size)
bn_layer2 = BatchNorm1d(num_features=hidden_size)

# Simulate training
print("\nTraining mode:")
for epoch in range(5):
    # Generate random batch
    x_batch = np.random.randn(batch_size, input_size)
    
    # Layer 1: Linear + BatchNorm + ReLU
    W1 = np.random.randn(hidden_size, input_size) * 0.01
    h1 = np.dot(x_batch, W1.T)
    h1_bn = bn_layer1.forward(h1, training=True)
    h1_relu = np.maximum(0, h1_bn)
    
    # Layer 2: Linear + BatchNorm + ReLU
    W2 = np.random.randn(hidden_size, hidden_size) * 0.01
    h2 = np.dot(h1_relu, W2.T)
    h2_bn = bn_layer2.forward(h2, training=True)
    h2_relu = np.maximum(0, h2_bn)
    
    # Output layer
    W3 = np.random.randn(output_size, hidden_size) * 0.01
    output = np.dot(h2_relu, W3.T)
    
    print(f"Epoch {epoch + 1}: Output shape {output.shape}, "
          f"Layer1 batches tracked: {bn_layer1.num_batches_tracked}")

# Simulate inference
print("\nInference mode:")
x_test = np.random.randn(1, input_size)

h1 = np.dot(x_test, W1.T)
h1_bn = bn_layer1.forward(h1, training=False)
h1_relu = np.maximum(0, h1_bn)

h2 = np.dot(h1_relu, W2.T)
h2_bn = bn_layer2.forward(h2, training=False)
h2_relu = np.maximum(0, h2_bn)

output = np.dot(h2_relu, W3.T)

print(f"Test output shape: {output.shape}")
print(f"Predictions: {output[0]}")

# ============================================================================
# 8. TRANSFORMER BLOCK WITH LAYER NORMALIZATION
# ============================================================================
print("\n8. TRANSFORMER BLOCK WITH LAYER NORMALIZATION")
print("-" * 70)

# Transformer parameters
seq_len = 10
d_model = 512
batch_size = 32

# Initialize Layer Norms
ln1 = LayerNorm(normalized_shape=d_model)
ln2 = LayerNorm(normalized_shape=d_model)

# Input
x_transformer = np.random.randn(batch_size, seq_len, d_model)

print(f"Input shape: {x_transformer.shape}")

# Self-attention (simplified)
attention_output = np.random.randn(batch_size, seq_len, d_model)

# Add & Norm 1
x_norm1 = ln1.forward(x_transformer + attention_output)
print(f"After LayerNorm 1: {x_norm1.shape}")

# Feed-forward (simplified)
ff_output = np.random.randn(batch_size, seq_len, d_model)

# Add & Norm 2
x_norm2 = ln2.forward(x_norm1 + ff_output)
print(f"After LayerNorm 2: {x_norm2.shape}")

print("\nTransformer block completed successfully!")

# ============================================================================
# 9. PERFORMANCE COMPARISON
# ============================================================================
print("\n9. PERFORMANCE COMPARISON")
print("-" * 70)

import time

# Test data
x_perf = np.random.randn(128, 512)

# Batch Normalization
bn_perf = BatchNorm1d(num_features=512)
start = time.time()
for _ in range(100):
    _ = bn_perf.forward(x_perf, training=True)
bn_time = time.time() - start

# Layer Normalization
ln_perf = LayerNorm(normalized_shape=512)
start = time.time()
for _ in range(100):
    _ = ln_perf.forward(x_perf)
ln_time = time.time() - start

print(f"Batch Normalization: {bn_time:.4f}s for 100 iterations")
print(f"Layer Normalization: {ln_time:.4f}s for 100 iterations")
print(f"Speed ratio (BN/LN): {bn_time/ln_time:.2f}x")

# ============================================================================
# 10. BEST PRACTICES
# ============================================================================
print("\n10. BEST PRACTICES")
print("-" * 70)

print("""
✓ Use Batch Normalization for:
  • CNNs (image classification, object detection)
  • Large batch training (batch size ≥ 32)
  • Feed-forward networks
  • When you need regularization effect

✓ Use Layer Normalization for:
  • RNNs and LSTMs
  • Transformers (BERT, GPT, Vision Transformers)
  • Small batch sizes or online learning
  • Reinforcement learning
  • When batch statistics are unreliable

✓ Use Group Normalization for:
  • Small batch sizes with CNNs
  • Object detection and segmentation
  • When batch size varies

✓ Use Instance Normalization for:
  • Style transfer
  • Image generation (GANs)
  • Per-image normalization needed

Key Tips:
1. Place normalization AFTER linear/conv layer, BEFORE activation
2. Use momentum=0.1 for BatchNorm (PyTorch default)
3. Use eps=1e-5 for numerical stability
4. Always set training=True/False correctly
5. Reset running stats when fine-tuning
""")

print("\n" + "=" * 70)
print("EXAMPLE COMPLETED SUCCESSFULLY! ✓")
print("=" * 70)
print("\nAll normalization techniques demonstrated:")
print("✓ Batch Normalization (1D and 2D)")
print("✓ Layer Normalization")
print("✓ Group Normalization")
print("✓ Instance Normalization")
print("✓ Training vs Inference modes")
print("✓ Complete neural network example")
print("✓ Transformer block example")
print("✓ Performance comparison")
print("\nReady for production use!")
