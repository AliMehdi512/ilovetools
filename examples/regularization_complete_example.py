"""
Comprehensive Example: Dropout and Regularization Techniques

This example demonstrates all regularization techniques available in ilovetools:
- Standard Dropout
- Spatial Dropout (CNNs)
- Variational Dropout (RNNs)
- DropConnect
- L1 Regularization (Lasso)
- L2 Regularization (Ridge)
- Elastic Net
- Early Stopping
"""

import numpy as np
from ilovetools.ml.regularization import (
    # Dropout
    dropout,
    spatial_dropout,
    variational_dropout,
    dropconnect,
    # Regularization
    l1_regularization,
    l2_regularization,
    elastic_net_regularization,
    # Gradients
    l1_gradient,
    l2_gradient,
    weight_decay,
    # Early Stopping
    EarlyStopping,
    # Utilities
    get_dropout_rate_schedule,
)

print("=" * 70)
print("DROPOUT AND REGULARIZATION TECHNIQUES - COMPREHENSIVE EXAMPLE")
print("=" * 70)

# ============================================================================
# 1. STANDARD DROPOUT
# ============================================================================
print("\n1. STANDARD DROPOUT")
print("-" * 70)

# Simulate a fully connected layer
x = np.random.randn(32, 128)  # (batch_size, features)

# Training mode with dropout
output_train, mask = dropout(x, dropout_rate=0.5, training=True, seed=42)
print(f"Input shape: {x.shape}")
print(f"Output shape: {output_train.shape}")
print(f"Neurons dropped: {np.sum(mask == 0)} / {mask.size} ({np.sum(mask == 0) / mask.size * 100:.1f}%)")

# Inference mode (no dropout)
output_inference, _ = dropout(x, dropout_rate=0.5, training=False)
print(f"Inference output equals input: {np.array_equal(output_inference, x)}")

# ============================================================================
# 2. SPATIAL DROPOUT (for CNNs)
# ============================================================================
print("\n2. SPATIAL DROPOUT (for CNNs)")
print("-" * 70)

# Simulate a convolutional layer output
x_cnn = np.random.randn(32, 64, 28, 28)  # (batch, channels, height, width)

output_spatial, mask_spatial = spatial_dropout(x_cnn, dropout_rate=0.2, training=True, seed=42)
print(f"Input shape: {x_cnn.shape}")
print(f"Output shape: {output_spatial.shape}")

# Count dropped channels
dropped_channels = 0
for b in range(x_cnn.shape[0]):
    for c in range(x_cnn.shape[1]):
        if mask_spatial[b, c, 0, 0] == 0:
            dropped_channels += 1

print(f"Channels dropped: {dropped_channels} / {x_cnn.shape[0] * x_cnn.shape[1]} ({dropped_channels / (x_cnn.shape[0] * x_cnn.shape[1]) * 100:.1f}%)")

# ============================================================================
# 3. VARIATIONAL DROPOUT (for RNNs)
# ============================================================================
print("\n3. VARIATIONAL DROPOUT (for RNNs)")
print("-" * 70)

# Simulate an RNN layer output
x_rnn = np.random.randn(32, 10, 512)  # (batch, seq_len, features)

output_var, mask_var = variational_dropout(x_rnn, dropout_rate=0.3, training=True, seed=42)
print(f"Input shape: {x_rnn.shape}")
print(f"Output shape: {output_var.shape}")

# Verify same mask across time steps
same_mask_across_time = True
for b in range(x_rnn.shape[0]):
    for f in range(x_rnn.shape[2]):
        if not np.all(mask_var[b, :, f] == mask_var[b, 0, f]):
            same_mask_across_time = False
            break

print(f"Same mask across time steps: {same_mask_across_time}")

# ============================================================================
# 4. DROPCONNECT
# ============================================================================
print("\n4. DROPCONNECT")
print("-" * 70)

# Simulate a fully connected layer
x_fc = np.random.randn(32, 128)
weights = np.random.randn(256, 128)

output_dc, mask_dc = dropconnect(x_fc, weights, dropout_rate=0.5, training=True, seed=42)
print(f"Input shape: {x_fc.shape}")
print(f"Weights shape: {weights.shape}")
print(f"Output shape: {output_dc.shape}")
print(f"Connections dropped: {np.sum(mask_dc == 0)} / {mask_dc.size} ({np.sum(mask_dc == 0) / mask_dc.size * 100:.1f}%)")

# ============================================================================
# 5. L1 REGULARIZATION (Lasso)
# ============================================================================
print("\n5. L1 REGULARIZATION (Lasso)")
print("-" * 70)

weights = np.random.randn(256, 128)

# Compute L1 penalty
l1_penalty = l1_regularization(weights, lambda_=0.01)
print(f"Weights shape: {weights.shape}")
print(f"L1 penalty: {l1_penalty:.6f}")

# Compute L1 gradient
l1_grad = l1_gradient(weights, lambda_=0.01)
print(f"L1 gradient shape: {l1_grad.shape}")
print(f"L1 gradient mean: {np.mean(np.abs(l1_grad)):.6f}")

# ============================================================================
# 6. L2 REGULARIZATION (Ridge)
# ============================================================================
print("\n6. L2 REGULARIZATION (Ridge)")
print("-" * 70)

# Compute L2 penalty
l2_penalty = l2_regularization(weights, lambda_=0.01)
print(f"Weights shape: {weights.shape}")
print(f"L2 penalty: {l2_penalty:.6f}")

# Compute L2 gradient
l2_grad = l2_gradient(weights, lambda_=0.01)
print(f"L2 gradient shape: {l2_grad.shape}")
print(f"L2 gradient mean: {np.mean(np.abs(l2_grad)):.6f}")

# Apply weight decay
weights_decayed = weight_decay(weights, learning_rate=0.01, decay_rate=0.01)
print(f"Weight decay applied: {np.mean(np.abs(weights - weights_decayed)):.6f}")

# ============================================================================
# 7. ELASTIC NET (L1 + L2)
# ============================================================================
print("\n7. ELASTIC NET (L1 + L2)")
print("-" * 70)

# Compute Elastic Net penalty
en_penalty = elastic_net_regularization(weights, lambda_=0.01, alpha=0.5)
print(f"Elastic Net penalty (α=0.5): {en_penalty:.6f}")

# Compare with pure L1 and L2
print(f"Pure L1 (α=1.0): {elastic_net_regularization(weights, lambda_=0.01, alpha=1.0):.6f}")
print(f"Pure L2 (α=0.0): {elastic_net_regularization(weights, lambda_=0.01, alpha=0.0):.6f}")

# ============================================================================
# 8. EARLY STOPPING
# ============================================================================
print("\n8. EARLY STOPPING")
print("-" * 70)

# Simulate training with early stopping
early_stopping = EarlyStopping(patience=5, min_delta=0.001, mode='min')

# Simulate validation losses
val_losses = [1.0, 0.9, 0.85, 0.82, 0.81, 0.805, 0.804, 0.803, 0.803, 0.803]

print("Epoch | Val Loss | Should Stop | Should Save")
print("-" * 50)

for epoch, val_loss in enumerate(val_losses, 1):
    should_stop = early_stopping(val_loss)
    should_save = early_stopping.should_save()
    
    print(f"{epoch:5d} | {val_loss:8.3f} | {str(should_stop):11s} | {str(should_save):11s}")
    
    if should_stop:
        print(f"\nEarly stopping triggered at epoch {epoch}!")
        print(f"Best validation loss: {early_stopping.best_score:.6f}")
        break

# ============================================================================
# 9. DROPOUT RATE SCHEDULE
# ============================================================================
print("\n9. DROPOUT RATE SCHEDULE")
print("-" * 70)

# Generate dropout schedules
linear_schedule = get_dropout_rate_schedule(0.5, 0.1, 100, 'linear')
exp_schedule = get_dropout_rate_schedule(0.5, 0.1, 100, 'exponential')
cosine_schedule = get_dropout_rate_schedule(0.5, 0.1, 100, 'cosine')

print("Epoch | Linear | Exponential | Cosine")
print("-" * 50)
for epoch in [0, 25, 50, 75, 99]:
    print(f"{epoch:5d} | {linear_schedule[epoch]:.4f} | {exp_schedule[epoch]:.4f} | {cosine_schedule[epoch]:.4f}")

# ============================================================================
# 10. COMPLETE TRAINING EXAMPLE
# ============================================================================
print("\n10. COMPLETE TRAINING EXAMPLE")
print("-" * 70)

# Simulate a simple neural network training loop
np.random.seed(42)

# Model parameters
input_size = 784
hidden_size = 256
output_size = 10
batch_size = 32
num_epochs = 20

# Initialize weights
W1 = np.random.randn(hidden_size, input_size) * 0.01
W2 = np.random.randn(output_size, hidden_size) * 0.01

# Training settings
learning_rate = 0.01
dropout_rate = 0.5
l2_lambda = 0.01

# Early stopping
early_stopping = EarlyStopping(patience=3, min_delta=0.001)

print("Training neural network with regularization...")
print(f"Architecture: {input_size} -> {hidden_size} -> {output_size}")
print(f"Dropout rate: {dropout_rate}")
print(f"L2 regularization: λ={l2_lambda}")
print()

for epoch in range(num_epochs):
    # Simulate training
    X_batch = np.random.randn(batch_size, input_size)
    
    # Forward pass with dropout
    h1 = np.dot(X_batch, W1.T)
    h1_dropout, _ = dropout(h1, dropout_rate=dropout_rate, training=True)
    h1_relu = np.maximum(0, h1_dropout)
    
    output = np.dot(h1_relu, W2.T)
    
    # Simulate loss
    train_loss = 1.0 / (epoch + 1) + np.random.randn() * 0.01
    
    # Add L2 regularization to loss
    l2_penalty = l2_regularization(W1, l2_lambda) + l2_regularization(W2, l2_lambda)
    total_loss = train_loss + l2_penalty
    
    # Simulate validation loss
    val_loss = train_loss + np.random.randn() * 0.02
    
    # Check early stopping
    should_stop = early_stopping(val_loss)
    
    print(f"Epoch {epoch+1:2d}: Train Loss={total_loss:.4f}, Val Loss={val_loss:.4f}, L2 Penalty={l2_penalty:.6f}")
    
    if should_stop:
        print(f"\nEarly stopping at epoch {epoch+1}")
        print(f"Best validation loss: {early_stopping.best_score:.4f}")
        break

print("\n" + "=" * 70)
print("EXAMPLE COMPLETED SUCCESSFULLY! ✓")
print("=" * 70)
print("\nKey Takeaways:")
print("✓ Standard dropout for fully connected layers (50% rate)")
print("✓ Spatial dropout for CNNs (drops entire channels)")
print("✓ Variational dropout for RNNs (same mask across time)")
print("✓ DropConnect drops connections instead of neurons")
print("✓ L1 regularization encourages sparsity")
print("✓ L2 regularization prevents large weights")
print("✓ Elastic Net combines L1 and L2")
print("✓ Early stopping prevents overfitting")
print("✓ Dropout schedules can improve training")
print("\nAll techniques are production-ready and optimized!")
