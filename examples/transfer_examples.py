"""
Comprehensive Examples: Transfer Learning & Fine-Tuning

This file demonstrates all transfer learning strategies with practical examples.

Author: Ali Mehdi
Date: January 31, 2026
"""

import numpy as np
from ilovetools.ml.transfer import (
    FeatureExtractor,
    FineTuner,
    GradualUnfreezer,
    DiscriminativeLR,
    DomainAdapter,
    compute_transfer_gap,
    learning_rate_warmup,
)

print("=" * 80)
print("TRANSFER LEARNING & FINE-TUNING - COMPREHENSIVE EXAMPLES")
print("=" * 80)
print()

# ============================================================================
# EXAMPLE 1: Feature Extraction - Small Dataset
# ============================================================================
print("EXAMPLE 1: Feature Extraction - ImageNet → Custom Dataset")
print("-" * 80)

# Pretrained ResNet50 on ImageNet (2048 features)
input_dim = 2048
num_classes = 10  # Custom dataset classes
num_samples = 500  # Small dataset

extractor = FeatureExtractor(
    input_dim=input_dim,
    num_classes=num_classes,
    hidden_dims=[512],
    pretrained_name='resnet50'
)

print("Transfer learning setup:")
print(f"Pretrained model: ResNet50 (ImageNet)")
print(f"Target dataset: {num_samples} samples, {num_classes} classes")
print(f"Strategy: Feature extraction (frozen backbone)")
print()

# Freeze backbone
extractor.freeze_backbone()
print()

# Simulate features from frozen ResNet50
features = np.random.randn(num_samples, input_dim)

print(f"Features from backbone: {features.shape}")

# Train classifier head
logits = extractor.forward(features)

print(f"Classifier output: {logits.shape}")
print()

print("Benefits:")
print("✓ Fast training (only classifier)")
print("✓ Works with small datasets")
print("✓ Low compute requirements")
print("✓ No risk of overfitting backbone")

print("\n✓ Example 1 completed\n")

# ============================================================================
# EXAMPLE 2: Fine-Tuning - Larger Dataset
# ============================================================================
print("EXAMPLE 2: Fine-Tuning - BERT → Domain-Specific NLP")
print("-" * 80)

num_layers = 12  # BERT-base layers
num_classes = 5
base_lr = 1e-4

finetuner = FineTuner(
    num_layers=num_layers,
    num_classes=num_classes,
    base_lr=base_lr
)

print("Fine-tuning setup:")
print(f"Pretrained model: BERT-base ({num_layers} layers)")
print(f"Target task: Sentiment classification ({num_classes} classes)")
print(f"Base learning rate: {base_lr:.2e}")
print()

# Set discriminative learning rates
finetuner.set_layer_lrs(discriminative=True, lr_decay=0.95)
print()

# Unfreeze top 4 layers
finetuner.unfreeze_layers(start_layer=8, end_layer=12)
print()

# Show layer-wise LRs
print("Layer-wise learning rates:")
for i in [0, 4, 8, 11]:
    lr = finetuner.get_layer_lr(i)
    frozen = "frozen" if finetuner.layer_frozen[i] else "trainable"
    print(f"  Layer {i}: {lr:.2e} ({frozen})")

print("\n✓ Example 2 completed\n")

# ============================================================================
# EXAMPLE 3: Gradual Unfreezing - Prevent Catastrophic Forgetting
# ============================================================================
print("EXAMPLE 3: Gradual Unfreezing - Stable Fine-Tuning")
print("-" * 80)

num_layers = 50  # ResNet50
num_groups = 4
epochs_per_group = 3
total_epochs = num_groups * epochs_per_group

unfreezer = GradualUnfreezer(
    num_layers=num_layers,
    num_groups=num_groups,
    epochs_per_group=epochs_per_group
)

print("Gradual unfreezing strategy:")
print(f"Total layers: {num_layers}")
print(f"Layer groups: {num_groups}")
print(f"Epochs per group: {epochs_per_group}")
print(f"Total epochs: {total_epochs}")
print()

print("Training progression:")
for epoch in range(total_epochs):
    unfreezer.step(epoch)
    trainable = unfreezer.get_trainable_layers()
    
    if epoch % epochs_per_group == 0:
        print(f"Epoch {epoch}: {len(trainable)} trainable layers")

print()
print("Benefits:")
print("✓ Prevents catastrophic forgetting")
print("✓ Stable training")
print("✓ Better final performance")

print("\n✓ Example 3 completed\n")

# ============================================================================
# EXAMPLE 4: Discriminative Learning Rates
# ============================================================================
print("EXAMPLE 4: Discriminative Learning Rates - Layer-Wise Adaptation")
print("-" * 80)

num_layers = 50
base_lr = 1e-3
decay_factor = 0.95

scheduler = DiscriminativeLR(
    num_layers=num_layers,
    base_lr=base_lr,
    decay_factor=decay_factor
)

print("Discriminative LR setup:")
scheduler.print_summary()
print()

# Show LRs for different layers
print("Learning rates by layer group:")
print(f"  Early layers (0-10): {scheduler.get_layer_lr(5):.2e}")
print(f"  Middle layers (20-30): {scheduler.get_layer_lr(25):.2e}")
print(f"  Late layers (40-50): {scheduler.get_layer_lr(45):.2e}")
print()

print("Rationale:")
print("✓ Early layers: General features (edges, textures)")
print("  → Lower LR to preserve pretrained knowledge")
print("✓ Late layers: Task-specific features")
print("  → Higher LR for better adaptation")

print("\n✓ Example 4 completed\n")

# ============================================================================
# EXAMPLE 5: Domain Adaptation - Medical Images
# ============================================================================
print("EXAMPLE 5: Domain Adaptation - Natural → Medical Images")
print("-" * 80)

feature_dim = 2048
num_classes = 3  # Disease categories

adapter = DomainAdapter(
    feature_dim=feature_dim,
    num_classes=num_classes,
    adaptation_method='adversarial'
)

print("Domain adaptation setup:")
print(f"Source domain: ImageNet (natural images)")
print(f"Target domain: Medical images (X-rays)")
print(f"Adaptation method: Domain-adversarial training")
print()

# Source features (ImageNet)
source_features = np.random.randn(64, feature_dim)

# Target features (Medical)
target_features = np.random.randn(64, feature_dim)

print(f"Source features: {source_features.shape}")
print(f"Target features: {target_features.shape}")
print()

# Compute domain adaptation loss
domain_loss = adapter.domain_loss(source_features, target_features)

print(f"Domain adaptation loss: {domain_loss:.4f}")
print()

# Task prediction on target domain
task_logits = adapter.task_forward(target_features)

print(f"Task predictions: {task_logits.shape}")
print()

print("Goal:")
print("✓ Minimize domain loss (confuse domain classifier)")
print("✓ Minimize task loss (accurate disease classification)")
print("✓ Learn domain-invariant features")

print("\n✓ Example 5 completed\n")

# ============================================================================
# EXAMPLE 6: Learning Rate Warmup
# ============================================================================
print("EXAMPLE 6: Learning Rate Warmup - Stable Fine-Tuning")
print("-" * 80)

warmup_epochs = 5
base_lr = 1e-3
total_epochs = 20

print("LR warmup schedule:")
print(f"Warmup epochs: {warmup_epochs}")
print(f"Base LR: {base_lr:.2e}")
print()

print("Learning rate progression:")
for epoch in range(total_epochs):
    lr = learning_rate_warmup(epoch, warmup_epochs, base_lr)
    
    if epoch < warmup_epochs or epoch % 5 == 0:
        print(f"  Epoch {epoch}: LR = {lr:.2e}")

print()
print("Benefits:")
print("✓ Prevents early instability")
print("✓ Better convergence")
print("✓ Especially important for fine-tuning")

print("\n✓ Example 6 completed\n")

# ============================================================================
# EXAMPLE 7: Transfer Gap Analysis
# ============================================================================
print("EXAMPLE 7: Transfer Gap Analysis")
print("-" * 80)

# Scenario 1: Similar domains
source_acc_1 = 0.95
target_acc_1 = 0.92
gap_1 = compute_transfer_gap(source_acc_1, target_acc_1)

# Scenario 2: Different domains
source_acc_2 = 0.95
target_acc_2 = 0.75
gap_2 = compute_transfer_gap(source_acc_2, target_acc_2)

print("Transfer gap comparison:")
print()

print("Scenario 1: ImageNet → Similar dataset")
print(f"  Source accuracy: {source_acc_1:.1%}")
print(f"  Target accuracy: {target_acc_1:.1%}")
print(f"  Transfer gap: {gap_1:.1%}")
print()

print("Scenario 2: ImageNet → Medical images")
print(f"  Source accuracy: {source_acc_2:.1%}")
print(f"  Target accuracy: {target_acc_2:.1%}")
print(f"  Transfer gap: {gap_2:.1%}")
print()

print("Interpretation:")
print("✓ Small gap (3%): Good transfer, similar domains")
print("✗ Large gap (20%): Poor transfer, domain shift")
print("  → Need domain adaptation or more fine-tuning")

print("\n✓ Example 7 completed\n")

# ============================================================================
# EXAMPLE 8: Complete Fine-Tuning Pipeline
# ============================================================================
print("EXAMPLE 8: Complete Fine-Tuning Pipeline")
print("-" * 80)

print("Step-by-step fine-tuning:")
print()

# Step 1: Feature extraction (warmup)
print("Step 1: Feature extraction (3 epochs)")
extractor = FeatureExtractor(input_dim=2048, num_classes=10)
extractor.freeze_backbone()
print("  ✓ Backbone frozen, train classifier only")
print()

# Step 2: Gradual unfreezing
print("Step 2: Gradual unfreezing (9 epochs)")
unfreezer = GradualUnfreezer(num_layers=50, num_groups=3, epochs_per_group=3)
for epoch in range(9):
    unfreezer.step(epoch)
    if epoch % 3 == 0:
        trainable = unfreezer.get_trainable_layers()
        print(f"  Epoch {epoch}: {len(trainable)} trainable layers")
print()

# Step 3: Full fine-tuning with discriminative LR
print("Step 3: Full fine-tuning (8 epochs)")
finetuner = FineTuner(num_layers=50, num_classes=10, base_lr=1e-4)
finetuner.unfreeze_layers(start_layer=0)
finetuner.set_layer_lrs(discriminative=True)
print("  ✓ All layers trainable")
print("  ✓ Discriminative learning rates")
print()

print("Total training: 20 epochs")
print("Benefits:")
print("✓ Stable training (no catastrophic forgetting)")
print("✓ Better final performance")
print("✓ Efficient use of pretrained knowledge")

print("\n✓ Example 8 completed\n")

# ============================================================================
# EXAMPLE 9: Few-Shot Learning with Transfer
# ============================================================================
print("EXAMPLE 9: Few-Shot Learning - 10 Samples per Class")
print("-" * 80)

num_classes = 5
samples_per_class = 10
total_samples = num_classes * samples_per_class

print("Few-shot learning scenario:")
print(f"Classes: {num_classes}")
print(f"Samples per class: {samples_per_class}")
print(f"Total samples: {total_samples}")
print()

# Feature extraction (best for few-shot)
extractor = FeatureExtractor(input_dim=2048, num_classes=num_classes)
extractor.freeze_backbone()

features = np.random.randn(total_samples, 2048)
logits = extractor.forward(features)

print(f"Features: {features.shape}")
print(f"Predictions: {logits.shape}")
print()

print("Why feature extraction for few-shot:")
print("✓ Prevents overfitting with tiny dataset")
print("✓ Leverages pretrained features")
print("✓ Fast training")
print("✓ Surprisingly good performance")

print("\n✓ Example 9 completed\n")

# ============================================================================
# EXAMPLE 10: When to Use Each Strategy
# ============================================================================
print("EXAMPLE 10: Strategy Selection Guide")
print("-" * 80)

print("Feature Extraction:")
print("  Use when:")
print("    ✓ Small dataset (< 1,000 samples)")
print("    ✓ Similar to source domain")
print("    ✓ Limited compute")
print("    ✓ Fast prototyping")
print("  Example: ImageNet → Cats vs Dogs")
print()

print("Fine-Tuning:")
print("  Use when:")
print("    ✓ Medium dataset (1,000 - 100,000 samples)")
print("    ✓ Some domain shift")
print("    ✓ Higher accuracy needed")
print("    ✓ Sufficient compute")
print("  Example: ImageNet → Medical images")
print()

print("Gradual Unfreezing:")
print("  Use when:")
print("    ✓ Large domain shift")
print("    ✓ Risk of catastrophic forgetting")
print("    ✓ Stable training needed")
print("  Example: Natural images → Satellite imagery")
print()

print("Domain Adaptation:")
print("  Use when:")
print("    ✓ Significant domain shift")
print("    ✓ Unlabeled target data available")
print("    ✓ Distribution mismatch")
print("  Example: Synthetic → Real images")

print("\n✓ Example 10 completed\n")

# ============================================================================
# SUMMARY
# ============================================================================
print("=" * 80)
print("ALL EXAMPLES COMPLETED SUCCESSFULLY! ✓")
print("=" * 80)
print()
print("Summary of what we covered:")
print("1. ✓ Feature Extraction - Small Dataset")
print("2. ✓ Fine-Tuning - Larger Dataset")
print("3. ✓ Gradual Unfreezing - Prevent Forgetting")
print("4. ✓ Discriminative Learning Rates")
print("5. ✓ Domain Adaptation - Medical Images")
print("6. ✓ Learning Rate Warmup")
print("7. ✓ Transfer Gap Analysis")
print("8. ✓ Complete Fine-Tuning Pipeline")
print("9. ✓ Few-Shot Learning")
print("10. ✓ Strategy Selection Guide")
print()
print("You now have a complete understanding of transfer learning!")
print()
print("Next steps:")
print("- Use feature extraction for small datasets")
print("- Use fine-tuning for domain shift")
print("- Use gradual unfreezing for stability")
print("- Use discriminative LR for better adaptation")
print("- Use domain adaptation for distribution mismatch")
