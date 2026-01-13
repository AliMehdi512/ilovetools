"""
Comprehensive Examples: Dropout Regularization Techniques

This file demonstrates all dropout regularization techniques
with practical examples and use cases.

Author: Ali Mehdi
Date: January 12, 2026
"""

import numpy as np
from ilovetools.ml.dropout import (
    Dropout,
    SpatialDropout2D,
    SpatialDropout3D,
    VariationalDropout,
    DropConnect,
    AlphaDropout,
    dropout,
    spatial_dropout_2d,
    spatial_dropout_3d,
    variational_dropout,
    dropconnect,
    alpha_dropout,
)

print("=" * 80)
print("DROPOUT REGULARIZATION - COMPREHENSIVE EXAMPLES")
print("=" * 80)
print()

# ============================================================================
# EXAMPLE 1: Standard Dropout (Fully Connected Networks)
# ============================================================================
print("EXAMPLE 1: Standard Dropout (Fully Connected Networks)")
print("-" * 80)

# Create dropout layer
drop = Dropout(rate=0.5, seed=42)

# Simulate training
x_train = np.random.randn(32, 128)
x_train_dropped = drop(x_train, training=True)

print(f"Training mode:")
print(f"  Input shape: {x_train.shape}")
print(f"  Output shape: {x_train_dropped.shape}")
print(f"  Zeros in output: {np.sum(x_train_dropped == 0)} / {x_train_dropped.size}")
print(f"  Mean (before): {np.mean(x_train):.6f}")
print(f"  Mean (after): {np.mean(x_train_dropped):.6f}")

# Simulate inference
x_test = np.random.randn(32, 128)
x_test_output = drop(x_test, training=False)

print(f"\nInference mode:")
print(f"  Input shape: {x_test.shape}")
print(f"  Output shape: {x_test_output.shape}")
print(f"  Unchanged: {np.allclose(x_test, x_test_output)}")

print("\n✓ Example 1 completed\n")

# ============================================================================
# EXAMPLE 2: Dropout in Multi-Layer Network
# ============================================================================
print("EXAMPLE 2: Dropout in Multi-Layer Network")
print("-" * 80)

# Network architecture
input_dim = 784
hidden1_dim = 512
hidden2_dim = 256
output_dim = 10

# Initialize weights
W1 = np.random.randn(input_dim, hidden1_dim) * 0.01
W2 = np.random.randn(hidden1_dim, hidden2_dim) * 0.01
W3 = np.random.randn(hidden2_dim, output_dim) * 0.01

# Dropout layers
drop1 = Dropout(rate=0.2, seed=42)  # Lower rate for input
drop2 = Dropout(rate=0.5, seed=43)  # Higher rate for hidden

# Forward pass (training)
x = np.random.randn(32, input_dim)

h1 = np.maximum(0, np.dot(x, W1))  # ReLU
h1_dropped = drop1(h1, training=True)

h2 = np.maximum(0, np.dot(h1_dropped, W2))  # ReLU
h2_dropped = drop2(h2, training=True)

output = np.dot(h2_dropped, W3)

print(f"Layer 1: {input_dim} -> {hidden1_dim}, dropout=0.2")
print(f"  Active neurons: {np.sum(h1_dropped != 0)} / {h1_dropped.size}")

print(f"\nLayer 2: {hidden1_dim} -> {hidden2_dim}, dropout=0.5")
print(f"  Active neurons: {np.sum(h2_dropped != 0)} / {h2_dropped.size}")

print(f"\nOutput: {hidden2_dim} -> {output_dim}")
print(f"  Shape: {output.shape}")

print("\n✓ Example 2 completed\n")

# ============================================================================
# EXAMPLE 3: Spatial Dropout for CNNs
# ============================================================================
print("EXAMPLE 3: Spatial Dropout for CNNs")
print("-" * 80)

# Convolutional feature maps
batch_size = 16
channels = 64
height = 28
width = 28

x_conv = np.random.randn(batch_size, channels, height, width)

# Apply spatial dropout
drop_spatial = SpatialDropout2D(rate=0.2, data_format='channels_first', seed=42)
x_dropped = drop_spatial(x_conv, training=True)

print(f"Input shape: {x_conv.shape}")
print(f"Output shape: {x_dropped.shape}")

# Check how many feature maps were dropped
dropped_maps = 0
for b in range(batch_size):
    for c in range(channels):
        if np.all(x_dropped[b, c] == 0):
            dropped_maps += 1

total_maps = batch_size * channels
print(f"\nDropped feature maps: {dropped_maps} / {total_maps}")
print(f"Dropout rate: {dropped_maps / total_maps:.2%}")

print("\n✓ Example 3 completed\n")

# ============================================================================
# EXAMPLE 4: Spatial Dropout with Channels Last
# ============================================================================
print("EXAMPLE 4: Spatial Dropout with Channels Last")
print("-" * 80)

# TensorFlow/Keras format
x_tf = np.random.randn(16, 28, 28, 64)  # (batch, height, width, channels)

drop_tf = SpatialDropout2D(rate=0.2, data_format='channels_last', seed=42)
x_dropped_tf = drop_tf(x_tf, training=True)

print(f"Input shape (channels_last): {x_tf.shape}")
print(f"Output shape: {x_dropped_tf.shape}")

print("\n✓ Example 4 completed\n")

# ============================================================================
# EXAMPLE 5: Spatial Dropout 3D for Video/3D CNNs
# ============================================================================
print("EXAMPLE 5: Spatial Dropout 3D for Video/3D CNNs")
print("-" * 80)

# 3D convolutional feature maps
batch_size = 8
channels = 32
depth = 16
height = 28
width = 28

x_3d = np.random.randn(batch_size, channels, depth, height, width)

drop_3d = SpatialDropout3D(rate=0.2, data_format='channels_first', seed=42)
x_dropped_3d = drop_3d(x_3d, training=True)

print(f"Input shape: {x_3d.shape}")
print(f"Output shape: {x_dropped_3d.shape}")

print("\n✓ Example 5 completed\n")

# ============================================================================
# EXAMPLE 6: Variational Dropout for RNNs
# ============================================================================
print("EXAMPLE 6: Variational Dropout for RNNs")
print("-" * 80)

# RNN sequence
batch_size = 32
time_steps = 10
features = 128

x_rnn = np.random.randn(batch_size, time_steps, features)

# Apply variational dropout
drop_var = VariationalDropout(rate=0.3, seed=42)
x_dropped_var = drop_var(x_rnn, training=True)

print(f"Input shape: {x_rnn.shape}")
print(f"Output shape: {x_dropped_var.shape}")

# Verify same mask across time steps
print(f"\nVerifying consistent mask across time:")
for b in range(min(3, batch_size)):
    for f in range(min(5, features)):
        time_series = x_dropped_var[b, :, f]
        is_dropped = np.all(time_series == 0)
        print(f"  Batch {b}, Feature {f}: {'Dropped' if is_dropped else 'Active'}")

print("\n✓ Example 6 completed\n")

# ============================================================================
# EXAMPLE 7: DropConnect
# ============================================================================
print("EXAMPLE 7: DropConnect")
print("-" * 80)

# Input and weights
x_input = np.random.randn(32, 128)
weights = np.random.randn(128, 64)

# Apply DropConnect
drop_conn = DropConnect(rate=0.5, seed=42)

# Training mode
output_train = drop_conn.apply(x_input, weights, training=True)
print(f"Training mode:")
print(f"  Input shape: {x_input.shape}")
print(f"  Weights shape: {weights.shape}")
print(f"  Output shape: {output_train.shape}")

# Inference mode
output_test = drop_conn.apply(x_input, weights, training=False)
print(f"\nInference mode:")
print(f"  Output shape: {output_test.shape}")
print(f"  Same as np.dot: {np.allclose(output_test, np.dot(x_input, weights))}")

print("\n✓ Example 7 completed\n")

# ============================================================================
# EXAMPLE 8: Alpha Dropout for Self-Normalizing Networks
# ============================================================================
print("EXAMPLE 8: Alpha Dropout for Self-Normalizing Networks")
print("-" * 80)

# SELU activation
alpha_selu = 1.6732632423543772848170429916717
scale_selu = 1.0507009873554804934193349852946

def selu(x):
    return scale_selu * np.where(x > 0, x, alpha_selu * (np.exp(x) - 1))

# Create network with SELU and Alpha Dropout
x_snn = np.random.randn(100, 128)

# Layer 1
W1 = np.random.randn(128, 64) * np.sqrt(1.0 / 128)
h1 = selu(np.dot(x_snn, W1))

drop_alpha = AlphaDropout(rate=0.1, seed=42)
h1_dropped = drop_alpha(h1, training=True)

print(f"Input statistics:")
print(f"  Mean: {np.mean(x_snn):.6f}")
print(f"  Std: {np.std(x_snn):.6f}")

print(f"\nAfter SELU:")
print(f"  Mean: {np.mean(h1):.6f}")
print(f"  Std: {np.std(h1):.6f}")

print(f"\nAfter Alpha Dropout:")
print(f"  Mean: {np.mean(h1_dropped):.6f}")
print(f"  Std: {np.std(h1_dropped):.6f}")

print("\n✓ Example 8 completed\n")

# ============================================================================
# EXAMPLE 9: Comparing Dropout Rates
# ============================================================================
print("EXAMPLE 9: Comparing Dropout Rates")
print("-" * 80)

x_compare = np.ones((1000, 100))

rates = [0.1, 0.3, 0.5, 0.7, 0.9]

print("Dropout rate comparison:")
print(f"{'Rate':<10} {'Active %':<15} {'Mean':<15} {'Std':<15}")
print("-" * 60)

for rate in rates:
    drop_cmp = Dropout(rate=rate, seed=42)
    x_dropped_cmp = drop_cmp(x_compare, training=True)
    
    active_pct = np.sum(x_dropped_cmp != 0) / x_dropped_cmp.size * 100
    mean_val = np.mean(x_dropped_cmp)
    std_val = np.std(x_dropped_cmp)
    
    print(f"{rate:<10.1f} {active_pct:<15.2f} {mean_val:<15.6f} {std_val:<15.6f}")

print("\n✓ Example 9 completed\n")

# ============================================================================
# EXAMPLE 10: Dropout vs No Dropout (Overfitting Demo)
# ============================================================================
print("EXAMPLE 10: Dropout vs No Dropout (Overfitting Demo)")
print("-" * 80)

# Simulate small dataset (prone to overfitting)
np.random.seed(42)
X_train_small = np.random.randn(50, 20)
y_train_small = np.random.randn(50, 1)

# Network without dropout
W_no_drop = np.random.randn(20, 10) * 0.1
W2_no_drop = np.random.randn(10, 1) * 0.1

h_no_drop = np.maximum(0, np.dot(X_train_small, W_no_drop))
pred_no_drop = np.dot(h_no_drop, W2_no_drop)

# Network with dropout
drop_demo = Dropout(rate=0.5, seed=42)
h_with_drop = np.maximum(0, np.dot(X_train_small, W_no_drop))
h_with_drop = drop_demo(h_with_drop, training=True)
pred_with_drop = np.dot(h_with_drop, W2_no_drop)

print(f"Without dropout:")
print(f"  Hidden activations mean: {np.mean(h_no_drop):.6f}")
print(f"  Hidden activations std: {np.std(h_no_drop):.6f}")

print(f"\nWith dropout (rate=0.5):")
print(f"  Hidden activations mean: {np.mean(h_with_drop):.6f}")
print(f"  Hidden activations std: {np.std(h_with_drop):.6f}")

print("\n✓ Example 10 completed\n")

# ============================================================================
# EXAMPLE 11: Using Convenience Functions
# ============================================================================
print("EXAMPLE 11: Using Convenience Functions")
print("-" * 80)

x_func = np.random.randn(32, 128)

# Standard dropout
x_drop = dropout(x_func, rate=0.5, training=True, seed=42)
print(f"dropout(): {x_drop.shape}")

# Spatial dropout 2D
x_2d = np.random.randn(16, 64, 28, 28)
x_spatial_2d = spatial_dropout_2d(x_2d, rate=0.2, training=True, seed=42)
print(f"spatial_dropout_2d(): {x_spatial_2d.shape}")

# Spatial dropout 3D
x_3d_func = np.random.randn(8, 32, 16, 28, 28)
x_spatial_3d = spatial_dropout_3d(x_3d_func, rate=0.2, training=True, seed=42)
print(f"spatial_dropout_3d(): {x_spatial_3d.shape}")

# Variational dropout
x_var_func = np.random.randn(32, 10, 128)
x_var_drop = variational_dropout(x_var_func, rate=0.3, training=True, seed=42)
print(f"variational_dropout(): {x_var_drop.shape}")

# DropConnect
x_dc = np.random.randn(32, 128)
W_dc = np.random.randn(128, 64)
output_dc = dropconnect(x_dc, W_dc, rate=0.5, training=True, seed=42)
print(f"dropconnect(): {output_dc.shape}")

# Alpha dropout
x_alpha_func = np.random.randn(32, 128)
x_alpha_drop = alpha_dropout(x_alpha_func, rate=0.1, training=True, seed=42)
print(f"alpha_dropout(): {x_alpha_drop.shape}")

print("\n✓ Example 11 completed\n")

# ============================================================================
# EXAMPLE 12: ResNet-style Network with Dropout
# ============================================================================
print("EXAMPLE 12: ResNet-style Network with Dropout")
print("-" * 80)

# Residual block with dropout
def residual_block_with_dropout(x_res, W1_res, W2_res, drop_rate=0.1):
    """Residual block with dropout."""
    drop_res = Dropout(rate=drop_rate, seed=42)
    
    # Main path
    h_res = np.maximum(0, np.dot(x_res, W1_res))
    h_res = drop_res(h_res, training=True)
    h_res = np.maximum(0, np.dot(h_res, W2_res))
    h_res = drop_res(h_res, training=True)
    
    # Skip connection
    output_res = x_res + h_res
    
    return output_res

# Test residual block
x_resnet = np.random.randn(32, 256)
W1_resnet = np.random.randn(256, 256) * 0.01
W2_resnet = np.random.randn(256, 256) * 0.01

output_resnet = residual_block_with_dropout(x_resnet, W1_resnet, W2_resnet, drop_rate=0.1)

print(f"Input shape: {x_resnet.shape}")
print(f"Output shape: {output_resnet.shape}")
print(f"Residual preserved: {output_resnet.shape == x_resnet.shape}")

print("\n✓ Example 12 completed\n")

# ============================================================================
# EXAMPLE 13: LSTM with Variational Dropout
# ============================================================================
print("EXAMPLE 13: LSTM with Variational Dropout")
print("-" * 80)

# Simplified LSTM with variational dropout
batch_size_lstm = 32
seq_length = 10
input_size = 64
hidden_size = 128

# Input sequence
x_lstm = np.random.randn(batch_size_lstm, seq_length, input_size)

# Weights
W_input_lstm = np.random.randn(input_size, hidden_size) * 0.01
W_hidden_lstm = np.random.randn(hidden_size, hidden_size) * 0.01

# Variational dropout
drop_lstm = VariationalDropout(rate=0.3, seed=42)
x_lstm_dropped = drop_lstm(x_lstm, training=True)

# Process sequence
h_lstm = np.zeros((batch_size_lstm, hidden_size))
outputs_lstm = []

for t in range(seq_length):
    h_lstm = np.tanh(np.dot(x_lstm_dropped[:, t, :], W_input_lstm) + 
                     np.dot(h_lstm, W_hidden_lstm))
    outputs_lstm.append(h_lstm)

final_output_lstm = np.stack(outputs_lstm, axis=1)

print(f"Input sequence shape: {x_lstm.shape}")
print(f"Output sequence shape: {final_output_lstm.shape}")
print(f"Hidden state shape: {h_lstm.shape}")

print("\n✓ Example 13 completed\n")

# ============================================================================
# EXAMPLE 14: Transformer with Dropout
# ============================================================================
print("EXAMPLE 14: Transformer with Dropout")
print("-" * 80)

# Simplified transformer layer with dropout
d_model = 512
num_heads = 8
d_ff = 2048

# Input
x_transformer = np.random.randn(32, 10, d_model)  # (batch, seq_len, d_model)

# Attention dropout
drop_attn = Dropout(rate=0.1, seed=42)

# Feed-forward dropout
drop_ff = Dropout(rate=0.1, seed=43)

# Simulate attention
attn_output = x_transformer  # Simplified
attn_output = drop_attn(attn_output, training=True)

# Feed-forward
W_ff1 = np.random.randn(d_model, d_ff) * 0.01
W_ff2 = np.random.randn(d_ff, d_model) * 0.01

ff_output = np.maximum(0, np.dot(attn_output, W_ff1))
ff_output = drop_ff(ff_output, training=True)
ff_output = np.dot(ff_output, W_ff2)
ff_output = drop_ff(ff_output, training=True)

# Residual connection
output_transformer = x_transformer + ff_output

print(f"Input shape: {x_transformer.shape}")
print(f"After attention + dropout: {attn_output.shape}")
print(f"After feed-forward + dropout: {ff_output.shape}")
print(f"Final output: {output_transformer.shape}")

print("\n✓ Example 14 completed\n")

# ============================================================================
# EXAMPLE 15: Dropout Schedule (Varying Rates During Training)
# ============================================================================
print("EXAMPLE 15: Dropout Schedule (Varying Rates During Training)")
print("-" * 80)

# Simulate training epochs with decreasing dropout
x_schedule = np.random.randn(100, 128)

epochs = [1, 10, 20, 30, 40, 50]
dropout_rates = [0.5, 0.4, 0.3, 0.2, 0.1, 0.05]

print("Dropout schedule during training:")
print(f"{'Epoch':<10} {'Dropout Rate':<15} {'Active Neurons %':<20}")
print("-" * 50)

for epoch, rate in zip(epochs, dropout_rates):
    drop_sched = Dropout(rate=rate, seed=42)
    x_dropped_sched = drop_sched(x_schedule, training=True)
    active_pct = np.sum(x_dropped_sched != 0) / x_dropped_sched.size * 100
    
    print(f"{epoch:<10} {rate:<15.2f} {active_pct:<20.2f}")

print("\n✓ Example 15 completed\n")

# ============================================================================
# SUMMARY
# ============================================================================
print("=" * 80)
print("ALL EXAMPLES COMPLETED SUCCESSFULLY! ✓")
print("=" * 80)
print()
print("Summary of what we covered:")
print("1. ✓ Standard Dropout (Fully Connected)")
print("2. ✓ Multi-Layer Network with Dropout")
print("3. ✓ Spatial Dropout for CNNs")
print("4. ✓ Spatial Dropout (Channels Last)")
print("5. ✓ Spatial Dropout 3D")
print("6. ✓ Variational Dropout for RNNs")
print("7. ✓ DropConnect")
print("8. ✓ Alpha Dropout (Self-Normalizing)")
print("9. ✓ Comparing Dropout Rates")
print("10. ✓ Dropout vs No Dropout")
print("11. ✓ Convenience Functions")
print("12. ✓ ResNet with Dropout")
print("13. ✓ LSTM with Variational Dropout")
print("14. ✓ Transformer with Dropout")
print("15. ✓ Dropout Schedule")
print()
print("You now have a complete understanding of dropout regularization!")
print()
print("Next steps:")
print("- Choose dropout rate based on network size")
print("- Use spatial dropout for CNNs")
print("- Use variational dropout for RNNs")
print("- Combine with other regularization techniques")
print("- Monitor validation performance")
