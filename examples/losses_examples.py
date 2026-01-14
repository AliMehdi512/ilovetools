"""
Comprehensive Examples: Loss Functions

This file demonstrates all loss functions with practical examples and use cases.

Author: Ali Mehdi
Date: January 13, 2026
"""

import numpy as np
from ilovetools.ml.losses import (
    MSE, MAE, HuberLoss,
    CrossEntropy, BinaryCrossEntropy, FocalLoss,
    DiceLoss, HingeLoss, KLDivergence,
    CosineSimilarityLoss, TripletLoss
)

print("=" * 80)
print("LOSS FUNCTIONS - COMPREHENSIVE EXAMPLES")
print("=" * 80)
print()

# ============================================================================
# EXAMPLE 1: MSE for Regression
# ============================================================================
print("EXAMPLE 1: MSE for Regression")
print("-" * 80)

mse = MSE()

# House price prediction
y_true_prices = np.array([300000, 450000, 250000, 600000])  # Actual prices
y_pred_prices = np.array([310000, 440000, 260000, 580000])  # Predicted prices

loss = mse(y_true_prices, y_pred_prices)
print(f"House Price Prediction MSE: ${loss:,.2f}")
print(f"RMSE: ${np.sqrt(loss):,.2f}")

print("\n✓ Example 1 completed\n")

# ============================================================================
# EXAMPLE 2: MAE vs MSE - Outlier Robustness
# ============================================================================
print("EXAMPLE 2: MAE vs MSE - Outlier Robustness")
print("-" * 80)

mse = MSE()
mae = MAE()

# Data with outlier
y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y_pred_normal = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
y_pred_outlier = np.array([1.1, 2.1, 3.1, 4.1, 15.0])  # Large outlier

print("Without outlier:")
print(f"  MSE: {mse(y_true, y_pred_normal):.4f}")
print(f"  MAE: {mae(y_true, y_pred_normal):.4f}")

print("\nWith outlier:")
print(f"  MSE: {mse(y_true, y_pred_outlier):.4f} (heavily penalized)")
print(f"  MAE: {mae(y_true, y_pred_outlier):.4f} (less affected)")

print("\n✓ Example 2 completed\n")

# ============================================================================
# EXAMPLE 3: Huber Loss - Best of Both Worlds
# ============================================================================
print("EXAMPLE 3: Huber Loss - Best of Both Worlds")
print("-" * 80)

huber = HuberLoss(delta=1.0)

y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y_pred = np.array([1.1, 2.1, 3.1, 4.1, 10.0])

loss_huber = huber(y_true, y_pred)
loss_mse = MSE()(y_true, y_pred)
loss_mae = MAE()(y_true, y_pred)

print(f"Huber Loss: {loss_huber:.4f}")
print(f"MSE Loss: {loss_mse:.4f}")
print(f"MAE Loss: {loss_mae:.4f}")
print("\nHuber combines smoothness of MSE with robustness of MAE")

print("\n✓ Example 3 completed\n")

# ============================================================================
# EXAMPLE 4: Cross Entropy for Multi-Class Classification
# ============================================================================
print("EXAMPLE 4: Cross Entropy for Multi-Class Classification")
print("-" * 80)

ce = CrossEntropy()

# Image classification (3 classes: cat, dog, bird)
y_true = np.array([
    [1, 0, 0],  # Cat
    [0, 1, 0],  # Dog
    [0, 0, 1],  # Bird
])

# Model predictions (probabilities)
y_pred_good = np.array([
    [0.9, 0.05, 0.05],  # Confident cat
    [0.1, 0.8, 0.1],    # Confident dog
    [0.05, 0.05, 0.9],  # Confident bird
])

y_pred_bad = np.array([
    [0.4, 0.3, 0.3],    # Uncertain
    [0.3, 0.4, 0.3],    # Uncertain
    [0.3, 0.3, 0.4],    # Uncertain
])

loss_good = ce(y_true, y_pred_good)
loss_bad = ce(y_true, y_pred_bad)

print(f"Good predictions loss: {loss_good:.4f}")
print(f"Bad predictions loss: {loss_bad:.4f}")
print(f"Difference: {loss_bad - loss_good:.4f}")

print("\n✓ Example 4 completed\n")

# ============================================================================
# EXAMPLE 5: Binary Cross Entropy for Binary Classification
# ============================================================================
print("EXAMPLE 5: Binary Cross Entropy for Binary Classification")
print("-" * 80)

bce = BinaryCrossEntropy()

# Spam detection
y_true = np.array([1, 0, 1, 0, 1, 0])  # 1=spam, 0=not spam
y_pred = np.array([0.95, 0.05, 0.90, 0.10, 0.85, 0.15])

loss = bce(y_true, y_pred)
print(f"Spam Detection BCE Loss: {loss:.4f}")

# Calculate accuracy
predictions = (y_pred > 0.5).astype(int)
accuracy = np.mean(predictions == y_true)
print(f"Accuracy: {accuracy * 100:.1f}%")

print("\n✓ Example 5 completed\n")

# ============================================================================
# EXAMPLE 6: Focal Loss for Imbalanced Classification
# ============================================================================
print("EXAMPLE 6: Focal Loss for Imbalanced Classification")
print("-" * 80)

bce = BinaryCrossEntropy()
focal = FocalLoss(alpha=0.25, gamma=2.0)

# Imbalanced dataset (rare disease detection)
y_true = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])  # 90% negative, 10% positive

# Easy examples (high confidence)
y_pred_easy = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9])

# Hard examples (low confidence)
y_pred_hard = np.array([0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.6])

print("Easy examples:")
print(f"  BCE Loss: {bce(y_true, y_pred_easy):.4f}")
print(f"  Focal Loss: {focal(y_true, y_pred_easy):.4f}")

print("\nHard examples:")
print(f"  BCE Loss: {bce(y_true, y_pred_hard):.4f}")
print(f"  Focal Loss: {focal(y_true, y_pred_hard):.4f}")

print("\nFocal loss focuses more on hard examples!")

print("\n✓ Example 6 completed\n")

# ============================================================================
# EXAMPLE 7: Dice Loss for Segmentation
# ============================================================================
print("EXAMPLE 7: Dice Loss for Segmentation")
print("-" * 80)

dice = DiceLoss()

# Medical image segmentation (tumor detection)
# 1 = tumor, 0 = healthy tissue
y_true = np.array([
    [0, 0, 1, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 1, 0, 0],
])

# Good segmentation
y_pred_good = np.array([
    [0.1, 0.1, 0.9, 0.9, 0.1],
    [0.1, 0.9, 0.9, 0.9, 0.1],
    [0.1, 0.1, 0.9, 0.1, 0.1],
])

# Poor segmentation
y_pred_poor = np.array([
    [0.5, 0.5, 0.5, 0.5, 0.5],
    [0.5, 0.5, 0.5, 0.5, 0.5],
    [0.5, 0.5, 0.5, 0.5, 0.5],
])

loss_good = dice(y_true, y_pred_good)
loss_poor = dice(y_true, y_pred_poor)

print(f"Good segmentation Dice loss: {loss_good:.4f}")
print(f"Poor segmentation Dice loss: {loss_poor:.4f}")
print(f"Dice coefficient (good): {1 - loss_good:.4f}")

print("\n✓ Example 7 completed\n")

# ============================================================================
# EXAMPLE 8: Hinge Loss for SVM-style Classification
# ============================================================================
print("EXAMPLE 8: Hinge Loss for SVM-style Classification")
print("-" * 80)

hinge = HingeLoss()

# Binary classification with margin
y_true = np.array([1, -1, 1, -1, 1, -1])
y_pred = np.array([1.5, -1.5, 0.8, -0.8, 2.0, -2.0])

loss = hinge(y_true, y_pred)
print(f"Hinge Loss: {loss:.4f}")

# Check margin violations
margins = y_true * y_pred
print(f"\nMargins: {margins}")
print(f"Violations (< 1): {np.sum(margins < 1)}")

print("\n✓ Example 8 completed\n")

# ============================================================================
# EXAMPLE 9: KL Divergence for Distribution Matching
# ============================================================================
print("EXAMPLE 9: KL Divergence for Distribution Matching")
print("-" * 80)

kl = KLDivergence()

# Teacher-student knowledge distillation
teacher_probs = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
student_probs = np.array([[0.6, 0.3, 0.1], [0.2, 0.7, 0.1]])

loss = kl(teacher_probs, student_probs)
print(f"KL Divergence (teacher → student): {loss:.4f}")

# Reverse KL
loss_reverse = kl(student_probs, teacher_probs)
print(f"KL Divergence (student → teacher): {loss_reverse:.4f}")
print("\nKL divergence is asymmetric!")

print("\n✓ Example 9 completed\n")

# ============================================================================
# EXAMPLE 10: Cosine Similarity Loss for Embeddings
# ============================================================================
print("EXAMPLE 10: Cosine Similarity Loss for Embeddings")
print("-" * 80)

cosine = CosineSimilarityLoss()

# Sentence embeddings
sentence1 = np.array([[0.5, 0.3, 0.2, 0.1]])
sentence2_similar = np.array([[0.52, 0.28, 0.18, 0.12]])
sentence2_different = np.array([[0.1, 0.2, 0.3, 0.5]])

loss_similar = cosine(sentence1, sentence2_similar)
loss_different = cosine(sentence1, sentence2_different)

print(f"Similar sentences loss: {loss_similar:.4f}")
print(f"Different sentences loss: {loss_different:.4f}")
print(f"Similarity score (similar): {1 - loss_similar:.4f}")
print(f"Similarity score (different): {1 - loss_different:.4f}")

print("\n✓ Example 10 completed\n")

# ============================================================================
# EXAMPLE 11: Triplet Loss for Face Recognition
# ============================================================================
print("EXAMPLE 11: Triplet Loss for Face Recognition")
print("-" * 80)

triplet = TripletLoss(margin=1.0)

# Face embeddings
anchor = np.array([[1.0, 2.0, 3.0]])  # Person A
positive = np.array([[1.1, 2.1, 3.1]])  # Person A (different photo)
negative = np.array([[5.0, 6.0, 7.0]])  # Person B

loss = triplet(anchor, positive, negative)
print(f"Triplet Loss: {loss:.4f}")

# Calculate distances
pos_dist = np.sum((anchor - positive) ** 2)
neg_dist = np.sum((anchor - negative) ** 2)

print(f"\nPositive distance: {pos_dist:.4f}")
print(f"Negative distance: {neg_dist:.4f}")
print(f"Margin satisfied: {neg_dist - pos_dist > 1.0}")

print("\n✓ Example 11 completed\n")

# ============================================================================
# EXAMPLE 12: Comparing Losses for Regression
# ============================================================================
print("EXAMPLE 12: Comparing Losses for Regression")
print("-" * 80)

y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y_pred = np.array([1.2, 2.3, 2.8, 4.5, 8.0])  # Last prediction is outlier

mse = MSE()
mae = MAE()
huber = HuberLoss(delta=1.0)

print(f"{'Loss Type':<15} {'Value':<10}")
print("-" * 25)
print(f"{'MSE':<15} {mse(y_true, y_pred):<10.4f}")
print(f"{'MAE':<15} {mae(y_true, y_pred):<10.4f}")
print(f"{'Huber':<15} {huber(y_true, y_pred):<10.4f}")

print("\n✓ Example 12 completed\n")

# ============================================================================
# EXAMPLE 13: Training Loop with Loss Functions
# ============================================================================
print("EXAMPLE 13: Training Loop with Loss Functions")
print("-" * 80)

# Simulate training
np.random.seed(42)
epochs = 5
batch_size = 32

mse = MSE()

print("Epoch | Loss")
print("-" * 20)

for epoch in range(epochs):
    # Simulate predictions getting better
    y_true = np.random.randn(batch_size, 10)
    noise = np.random.randn(batch_size, 10) * (1.0 - epoch * 0.15)
    y_pred = y_true + noise
    
    loss = mse(y_true, y_pred)
    print(f"{epoch + 1:5d} | {loss:.6f}")

print("\n✓ Example 13 completed\n")

# ============================================================================
# SUMMARY
# ============================================================================
print("=" * 80)
print("ALL EXAMPLES COMPLETED SUCCESSFULLY! ✓")
print("=" * 80)
print()
print("Summary of what we covered:")
print("1. ✓ MSE for Regression")
print("2. ✓ MAE vs MSE - Outlier Robustness")
print("3. ✓ Huber Loss - Best of Both Worlds")
print("4. ✓ Cross Entropy for Multi-Class")
print("5. ✓ Binary Cross Entropy")
print("6. ✓ Focal Loss for Imbalanced Data")
print("7. ✓ Dice Loss for Segmentation")
print("8. ✓ Hinge Loss for SVM")
print("9. ✓ KL Divergence for Distribution Matching")
print("10. ✓ Cosine Similarity for Embeddings")
print("11. ✓ Triplet Loss for Face Recognition")
print("12. ✓ Comparing Regression Losses")
print("13. ✓ Training Loop Example")
print()
print("You now have a complete understanding of loss functions!")
print()
print("Next steps:")
print("- Choose loss based on task (regression/classification/segmentation)")
print("- Consider data characteristics (outliers, imbalance)")
print("- Monitor loss during training")
print("- Combine losses when appropriate")
