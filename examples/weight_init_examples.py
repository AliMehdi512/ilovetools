"""
Comprehensive Examples: Weight Initialization Techniques

This file demonstrates all weight initialization techniques
with practical examples and use cases.
"""

import numpy as np
from ilovetools.ml.weight_init import (
    xavier_uniform,
    xavier_normal,
    he_uniform,
    he_normal,
    lecun_uniform,
    lecun_normal,
    orthogonal,
    identity,
    sparse,
    variance_scaling,
    constant,
    uniform,
    normal as normal_init,  # Renamed to avoid conflict
    calculate_gain,
    get_initializer,
    WeightInitializer,
)

print("=" * 80)
print("WEIGHT INITIALIZATION - COMPREHENSIVE EXAMPLES")
print("=" * 80)
print()

# ============================================================================
# EXAMPLE 1: Xavier/Glorot Initialization (Sigmoid/Tanh Networks)
# ============================================================================
print("EXAMPLE 1: Xavier/Glorot Initialization (Sigmoid/Tanh Networks)")
print("-" * 80)

input_size = 784
hidden_size = 256
output_size = 10

# Initialize weights for a 3-layer network
W1 = xavier_normal((input_size, hidden_size))
W2 = xavier_normal((hidden_size, hidden_size))
W3 = xavier_normal((hidden_size, output_size))

print(f"Layer 1 weights: shape={W1.shape}, mean={np.mean(W1):.6f}, std={np.std(W1):.6f}")
print(f"Layer 2 weights: shape={W2.shape}, mean={np.mean(W2):.6f}, std={np.std(W2):.6f}")
print(f"Layer 3 weights: shape={W3.shape}, mean={np.mean(W3):.6f}, std={np.std(W3):.6f}")

# Simulate forward pass
x = np.random.randn(32, input_size)
h1 = np.tanh(np.dot(x, W1))
h2 = np.tanh(np.dot(h1, W2))
output = np.dot(h2, W3)

print(f"\nActivation statistics:")
print(f"Hidden 1: mean={np.mean(h1):.6f}, std={np.std(h1):.6f}")
print(f"Hidden 2: mean={np.mean(h2):.6f}, std={np.std(h2):.6f}")
print(f"Output: mean={np.mean(output):.6f}, std={np.std(output):.6f}")

print("\n✓ Example 1 completed\n")

# ============================================================================
# EXAMPLE 2: He/Kaiming Initialization (ReLU Networks)
# ============================================================================
print("EXAMPLE 2: He/Kaiming Initialization (ReLU Networks)")
print("-" * 80)

# Initialize weights for ResNet-style network
W1 = he_normal((784, 512))
W2 = he_normal((512, 512))
W3 = he_normal((512, 256))
W4 = he_normal((256, 10))

print(f"Layer 1: shape={W1.shape}, std={np.std(W1):.6f}")
print(f"Layer 2: shape={W2.shape}, std={np.std(W2):.6f}")
print(f"Layer 3: shape={W3.shape}, std={np.std(W3):.6f}")
print(f"Layer 4: shape={W4.shape}, std={np.std(W4):.6f}")

# Simulate forward pass with ReLU
x = np.random.randn(32, 784)
h1 = np.maximum(0, np.dot(x, W1))
h2 = np.maximum(0, np.dot(h1, W2))
h3 = np.maximum(0, np.dot(h2, W3))
output = np.dot(h3, W4)

print(f"\nActivation statistics (ReLU):")
print(f"Hidden 1: mean={np.mean(h1):.6f}, std={np.std(h1):.6f}")
print(f"Hidden 2: mean={np.mean(h2):.6f}, std={np.std(h2):.6f}")
print(f"Hidden 3: mean={np.mean(h3):.6f}, std={np.std(h3):.6f}")
print(f"Output: mean={np.mean(output):.6f}, std={np.std(output):.6f}")

print("\n✓ Example 2 completed\n")

# ============================================================================
# EXAMPLE 3: LeCun Initialization (SELU Networks)
# ============================================================================
print("EXAMPLE 3: LeCun Initialization (SELU Networks)")
print("-" * 80)

# SELU activation parameters
alpha = 1.6732632423543772848170429916717
scale = 1.0507009873554804934193349852946

def selu(x):
    return scale * np.where(x > 0, x, alpha * (np.exp(x) - 1))

# Initialize weights
W1 = lecun_normal((784, 256))
W2 = lecun_normal((256, 128))
W3 = lecun_normal((128, 10))

print(f"Layer 1: shape={W1.shape}, std={np.std(W1):.6f}")
print(f"Layer 2: shape={W2.shape}, std={np.std(W2):.6f}")
print(f"Layer 3: shape={W3.shape}, std={np.std(W3):.6f}")

# Simulate forward pass with SELU
x = np.random.randn(32, 784)
h1 = selu(np.dot(x, W1))
h2 = selu(np.dot(h1, W2))
output = np.dot(h2, W3)

print(f"\nActivation statistics (SELU):")
print(f"Hidden 1: mean={np.mean(h1):.6f}, std={np.std(h1):.6f}")
print(f"Hidden 2: mean={np.mean(h2):.6f}, std={np.std(h2):.6f}")
print(f"Output: mean={np.mean(output):.6f}, std={np.std(output):.6f}")

print("\n✓ Example 3 completed\n")

# ============================================================================
# EXAMPLE 4: Orthogonal Initialization (RNNs)
# ============================================================================
print("EXAMPLE 4: Orthogonal Initialization (RNNs)")
print("-" * 80)

hidden_size = 128

# Initialize RNN weights
W_input = xavier_normal((100, hidden_size))
W_hidden = orthogonal((hidden_size, hidden_size))
W_output = xavier_normal((hidden_size, 10))

print(f"Input weights: shape={W_input.shape}")
print(f"Hidden weights: shape={W_hidden.shape}")
print(f"Output weights: shape={W_output.shape}")

# Check orthogonality
product = np.dot(W_hidden, W_hidden.T)
identity_mat = np.eye(hidden_size)
orthogonality_error = np.max(np.abs(product - identity_mat))

print(f"\nOrthogonality check:")
print(f"Max deviation from identity: {orthogonality_error:.6f}")
print(f"Is orthogonal: {orthogonality_error < 1e-5}")

# Simulate RNN forward pass
sequence_length = 10
batch_size = 32
input_size = 100

x_seq = np.random.randn(sequence_length, batch_size, input_size)
h = np.zeros((batch_size, hidden_size))

hidden_states = []
for t in range(sequence_length):
    h = np.tanh(np.dot(x_seq[t], W_input) + np.dot(h, W_hidden))
    hidden_states.append(h)

print(f"\nHidden state statistics over time:")
for t in [0, 4, 9]:
    print(f"  t={t}: mean={np.mean(hidden_states[t]):.6f}, std={np.std(hidden_states[t]):.6f}")

print("\n✓ Example 4 completed\n")

# ============================================================================
# EXAMPLE 5: Convolutional Layer Initialization
# ============================================================================
print("EXAMPLE 5: Convolutional Layer Initialization")
print("-" * 80)

# Conv2D: (out_channels, in_channels, kernel_h, kernel_w)
conv1_weights = he_normal((64, 3, 3, 3))
conv2_weights = he_normal((128, 64, 3, 3))
conv3_weights = he_normal((256, 128, 3, 3))

print(f"Conv1: shape={conv1_weights.shape}, std={np.std(conv1_weights):.6f}")
print(f"Conv2: shape={conv2_weights.shape}, std={np.std(conv2_weights):.6f}")
print(f"Conv3: shape={conv3_weights.shape}, std={np.std(conv3_weights):.6f}")

# Calculate receptive field sizes
for i, w in enumerate([conv1_weights, conv2_weights, conv3_weights], 1):
    receptive_field = w.shape[2] * w.shape[3]
    fan_in = w.shape[1] * receptive_field
    expected_std = np.sqrt(2.0 / fan_in)
    actual_std = np.std(w)
    print(f"\nConv{i}:")
    print(f"  Receptive field: {receptive_field}")
    print(f"  Fan-in: {fan_in}")
    print(f"  Expected std: {expected_std:.6f}")
    print(f"  Actual std: {actual_std:.6f}")

print("\n✓ Example 5 completed\n")

# ============================================================================
# EXAMPLE 6: Variance Scaling (Generalized)
# ============================================================================
print("EXAMPLE 6: Variance Scaling (Generalized)")
print("-" * 80)

shape = (100, 50)

# Different modes
w_fan_in = variance_scaling(shape, scale=2.0, mode='fan_in')
w_fan_out = variance_scaling(shape, scale=2.0, mode='fan_out')
w_fan_avg = variance_scaling(shape, scale=2.0, mode='fan_avg')

print(f"Fan-in mode: std={np.std(w_fan_in):.6f}")
print(f"Fan-out mode: std={np.std(w_fan_out):.6f}")
print(f"Fan-avg mode: std={np.std(w_fan_avg):.6f}")

# Different distributions
w_normal_dist = variance_scaling(shape, scale=2.0, mode='fan_in', distribution='normal')
w_uniform_dist = variance_scaling(shape, scale=2.0, mode='fan_in', distribution='uniform')

print(f"\nNormal distribution: std={np.std(w_normal_dist):.6f}")
print(f"Uniform distribution: std={np.std(w_uniform_dist):.6f}")

print("\n✓ Example 6 completed\n")

# ============================================================================
# EXAMPLE 7: Sparse Initialization
# ============================================================================
print("EXAMPLE 7: Sparse Initialization")
print("-" * 80)

shape = (100, 100)

# Different sparsity levels
for sparsity in [0.1, 0.5, 0.9]:
    weights = sparse(shape, sparsity=sparsity)
    zero_fraction = np.sum(weights == 0) / weights.size
    print(f"Sparsity {sparsity}: actual zero fraction = {zero_fraction:.3f}")

print("\n✓ Example 7 completed\n")

# ============================================================================
# EXAMPLE 8: Identity Initialization (Residual Connections)
# ============================================================================
print("EXAMPLE 8: Identity Initialization (Residual Connections)")
print("-" * 80)

# Residual block
input_size = 256
W_residual = identity((input_size, input_size))
W_transform = he_normal((input_size, input_size))

print(f"Residual weights: shape={W_residual.shape}")
print(f"Transform weights: shape={W_transform.shape}")

# Simulate residual connection
x_input = np.random.randn(32, input_size)
residual_output = np.dot(x_input, W_residual)
transform_output = np.maximum(0, np.dot(x_input, W_transform))
final_output = residual_output + transform_output

print(f"\nResidual connection:")
print(f"Input: mean={np.mean(x_input):.6f}, std={np.std(x_input):.6f}")
print(f"Residual: mean={np.mean(residual_output):.6f}, std={np.std(residual_output):.6f}")
print(f"Transform: mean={np.mean(transform_output):.6f}, std={np.std(transform_output):.6f}")
print(f"Output: mean={np.mean(final_output):.6f}, std={np.std(final_output):.6f}")

print("\n✓ Example 8 completed\n")

# ============================================================================
# EXAMPLE 9: Using WeightInitializer Class
# ============================================================================
print("EXAMPLE 9: Using WeightInitializer Class")
print("-" * 80)

# Create initializers
xavier_init = WeightInitializer('xavier_normal')
he_init = WeightInitializer('he_normal')
orthogonal_init = WeightInitializer('orthogonal', gain=1.0)

# Initialize layers
W1 = xavier_init((784, 256))
W2 = he_init((256, 128))
W3 = orthogonal_init((128, 128))

print(f"Xavier initialized: shape={W1.shape}, std={np.std(W1):.6f}")
print(f"He initialized: shape={W2.shape}, std={np.std(W2):.6f}")
print(f"Orthogonal initialized: shape={W3.shape}")

print("\n✓ Example 9 completed\n")

# ============================================================================
# EXAMPLE 10: Comparing Initializations
# ============================================================================
print("EXAMPLE 10: Comparing Initializations")
print("-" * 80)

shape = (100, 50)

initializations = {
    'Xavier Uniform': xavier_uniform(shape),
    'Xavier Normal': xavier_normal(shape),
    'He Uniform': he_uniform(shape),
    'He Normal': he_normal(shape),
    'LeCun Uniform': lecun_uniform(shape),
    'LeCun Normal': lecun_normal(shape),
}

print("Comparison of initialization methods:")
print(f"{'Method':<20} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
print("-" * 60)

for name, weights in initializations.items():
    print(f"{name:<20} {np.mean(weights):>10.6f} {np.std(weights):>10.6f} "
          f"{np.min(weights):>10.6f} {np.max(weights):>10.6f}")

print("\n✓ Example 10 completed\n")

# ============================================================================
# EXAMPLE 11: Calculate Gain for Different Activations
# ============================================================================
print("EXAMPLE 11: Calculate Gain for Different Activations")
print("-" * 80)

activations = ['linear', 'sigmoid', 'tanh', 'relu', 'leaky_relu', 'selu']

print("Recommended gains for different activations:")
for activation in activations:
    gain = calculate_gain(activation)
    print(f"  {activation:<15}: {gain:.6f}")

print("\n✓ Example 11 completed\n")

# ============================================================================
# EXAMPLE 12: Deep Network Initialization (10 layers)
# ============================================================================
print("EXAMPLE 12: Deep Network Initialization (10 layers)")
print("-" * 80)

# Initialize a very deep network
layer_sizes = [784, 512, 512, 256, 256, 128, 128, 64, 64, 32, 10]
weights = []

for i in range(len(layer_sizes) - 1):
    w = he_normal((layer_sizes[i], layer_sizes[i+1]))
    weights.append(w)
    print(f"Layer {i+1}: {layer_sizes[i]:4d} -> {layer_sizes[i+1]:4d}, std={np.std(w):.6f}")

# Simulate forward pass
x = np.random.randn(32, 784)
activations = [x]

for w in weights:
    x = np.maximum(0, np.dot(x, w))
    activations.append(x)

print(f"\nActivation statistics through depth:")
for i, act in enumerate(activations):
    print(f"  Layer {i}: mean={np.mean(act):.6f}, std={np.std(act):.6f}")

print("\n✓ Example 12 completed\n")

# ============================================================================
# EXAMPLE 13: Using get_initializer Factory
# ============================================================================
print("EXAMPLE 13: Using get_initializer Factory")
print("-" * 80)

shape = (100, 50)

methods = ['xavier_normal', 'he_normal', 'lecun_normal', 'orthogonal']

print("Creating initializers using factory:")
for method in methods:
    weights = get_initializer(method, shape)
    print(f"✓ {method:<20}: shape={weights.shape}, std={np.std(weights):.6f}")

print("\n✓ Example 13 completed\n")

# ============================================================================
# EXAMPLE 14: Real-World: ResNet Block Initialization
# ============================================================================
print("EXAMPLE 14: Real-World: ResNet Block Initialization")
print("-" * 80)

# ResNet block with skip connection
in_channels = 64
out_channels = 64

# Main path
conv1 = he_normal((out_channels, in_channels, 3, 3))
conv2 = he_normal((out_channels, out_channels, 3, 3))

# Skip connection (identity if same dimensions)
skip = identity((in_channels, out_channels)) if in_channels == out_channels else he_normal((out_channels, in_channels, 1, 1))

print("ResNet block initialization:")
print(f"Conv1: shape={conv1.shape}, std={np.std(conv1):.6f}")
print(f"Conv2: shape={conv2.shape}, std={np.std(conv2):.6f}")
print(f"Skip: shape={skip.shape}, std={np.std(skip):.6f}")

print("\n✓ Example 14 completed\n")

# ============================================================================
# EXAMPLE 15: Transformer Initialization
# ============================================================================
print("EXAMPLE 15: Transformer Initialization")
print("-" * 80)

d_model = 512
num_heads = 8
d_ff = 2048

# Multi-head attention
W_q = xavier_normal((d_model, d_model))
W_k = xavier_normal((d_model, d_model))
W_v = xavier_normal((d_model, d_model))
W_o = xavier_normal((d_model, d_model))

# Feed-forward
W_ff1 = xavier_normal((d_model, d_ff))
W_ff2 = xavier_normal((d_ff, d_model))

print("Transformer layer initialization:")
print(f"Query projection: shape={W_q.shape}, std={np.std(W_q):.6f}")
print(f"Key projection: shape={W_k.shape}, std={np.std(W_k):.6f}")
print(f"Value projection: shape={W_v.shape}, std={np.std(W_v):.6f}")
print(f"Output projection: shape={W_o.shape}, std={np.std(W_o):.6f}")
print(f"FF layer 1: shape={W_ff1.shape}, std={np.std(W_ff1):.6f}")
print(f"FF layer 2: shape={W_ff2.shape}, std={np.std(W_ff2):.6f}")

print("\n✓ Example 15 completed\n")

# ============================================================================
# SUMMARY
# ============================================================================
print("=" * 80)
print("ALL EXAMPLES COMPLETED SUCCESSFULLY! ✓")
print("=" * 80)
print()
print("Summary of what we covered:")
print("1. ✓ Xavier/Glorot Initialization (Sigmoid/Tanh)")
print("2. ✓ He/Kaiming Initialization (ReLU)")
print("3. ✓ LeCun Initialization (SELU)")
print("4. ✓ Orthogonal Initialization (RNNs)")
print("5. ✓ Convolutional Layer Initialization")
print("6. ✓ Variance Scaling (Generalized)")
print("7. ✓ Sparse Initialization")
print("8. ✓ Identity Initialization (Residual)")
print("9. ✓ WeightInitializer Class")
print("10. ✓ Comparing Initializations")
print("11. ✓ Calculate Gain")
print("12. ✓ Deep Network (10 layers)")
print("13. ✓ Factory Function")
print("14. ✓ ResNet Block")
print("15. ✓ Transformer Layer")
print()
print("You now have a complete understanding of weight initialization!")
print()
print("Next steps:")
print("- Match initialization to your activation function")
print("- Monitor gradient flow during training")
print("- Combine with batch normalization")
print("- Experiment with different methods")
