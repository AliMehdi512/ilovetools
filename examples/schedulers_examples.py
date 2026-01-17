"""
Comprehensive Examples: Learning Rate Schedulers

This file demonstrates all learning rate schedulers with practical examples and use cases.

Author: Ali Mehdi
Date: January 17, 2026
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from ilovetools.ml.schedulers import (
    StepLR,
    ExponentialLR,
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    OneCycleLR,
    CyclicLR,
    ReduceLROnPlateau,
    PolynomialLR,
    WarmupLR,
    MultiStepLR,
)

print("=" * 80)
print("LEARNING RATE SCHEDULERS - COMPREHENSIVE EXAMPLES")
print("=" * 80)
print()

# ============================================================================
# EXAMPLE 1: Step LR - Image Classification
# ============================================================================
print("EXAMPLE 1: Step LR - Image Classification")
print("-" * 80)

scheduler = StepLR(initial_lr=0.1, step_size=30, gamma=0.1)

print("Training ResNet on ImageNet with Step LR:")
print(f"Initial LR: {scheduler.initial_lr}")
print(f"Step size: {scheduler.step_size} epochs")
print(f"Gamma: {scheduler.gamma}")
print()

epochs = [0, 10, 30, 50, 90, 100]
for epoch in epochs:
    lr = scheduler.step(epoch)
    print(f"Epoch {epoch:3d}: LR = {lr:.6f}")

print("\n✓ Example 1 completed\n")

# ============================================================================
# EXAMPLE 2: Exponential LR - Continuous Decay
# ============================================================================
print("EXAMPLE 2: Exponential LR - Continuous Decay")
print("-" * 80)

scheduler = ExponentialLR(initial_lr=0.1, gamma=0.95)

print("Smooth exponential decay:")
lrs = []
for epoch in range(100):
    lr = scheduler.step(epoch)
    lrs.append(lr)
    if epoch % 20 == 0:
        print(f"Epoch {epoch:3d}: LR = {lr:.6f}")

print(f"\nFinal LR after 100 epochs: {lrs[-1]:.6f}")
print(f"Decay rate: {(lrs[-1] / lrs[0]) * 100:.2f}%")

print("\n✓ Example 2 completed\n")

# ============================================================================
# EXAMPLE 3: Cosine Annealing - Smooth Decay
# ============================================================================
print("EXAMPLE 3: Cosine Annealing - Smooth Decay")
print("-" * 80)

scheduler = CosineAnnealingLR(initial_lr=0.1, T_max=100, eta_min=0.001)

print("Cosine annealing schedule:")
lrs = []
for epoch in range(100):
    lr = scheduler.step(epoch)
    lrs.append(lr)
    if epoch % 25 == 0:
        print(f"Epoch {epoch:3d}: LR = {lr:.6f}")

print(f"\nMinimum LR: {min(lrs):.6f}")
print(f"Maximum LR: {max(lrs):.6f}")

print("\n✓ Example 3 completed\n")

# ============================================================================
# EXAMPLE 4: Cosine Annealing with Warm Restarts (SGDR)
# ============================================================================
print("EXAMPLE 4: Cosine Annealing with Warm Restarts (SGDR)")
print("-" * 80)

scheduler = CosineAnnealingWarmRestarts(initial_lr=0.1, T_0=10, T_mult=2, eta_min=0.001)

print("SGDR schedule with periodic restarts:")
lrs = []
for epoch in range(50):
    lr = scheduler.step(epoch)
    lrs.append(lr)
    if epoch in [0, 10, 30]:
        print(f"Epoch {epoch:3d}: LR = {lr:.6f} (Restart)")

print(f"\nNumber of restarts in 50 epochs: 3")
print(f"Restart periods: 10, 20, 40 epochs")

print("\n✓ Example 4 completed\n")

# ============================================================================
# EXAMPLE 5: One Cycle LR - Super Convergence
# ============================================================================
print("EXAMPLE 5: One Cycle LR - Super Convergence")
print("-" * 80)

scheduler = OneCycleLR(initial_lr=0.001, max_lr=0.1, total_steps=1000, pct_start=0.3)

print("One Cycle Policy for fast training:")
print(f"Initial LR: {scheduler.initial_lr}")
print(f"Max LR: {scheduler.max_lr}")
print(f"Total steps: {scheduler.total_steps}")
print(f"Warmup percentage: {scheduler.pct_start * 100}%")
print()

steps = [0, 150, 300, 500, 750, 999]
for step in steps:
    lr = scheduler.step(step)
    print(f"Step {step:4d}: LR = {lr:.6f}")

print("\n✓ Example 5 completed\n")

# ============================================================================
# EXAMPLE 6: Cyclic LR - Triangular Policy
# ============================================================================
print("EXAMPLE 6: Cyclic LR - Triangular Policy")
print("-" * 80)

scheduler = CyclicLR(base_lr=0.001, max_lr=0.01, step_size=500, mode='triangular')

print("Cyclic learning rate with triangular mode:")
lrs = []
for step in range(2000):
    lr = scheduler.step(step)
    lrs.append(lr)

print(f"Base LR: {scheduler.base_lr}")
print(f"Max LR: {scheduler.max_lr}")
print(f"Step size: {scheduler.step_size}")
print(f"Number of cycles in 2000 steps: {2000 / (2 * scheduler.step_size)}")
print(f"Min LR observed: {min(lrs):.6f}")
print(f"Max LR observed: {max(lrs):.6f}")

print("\n✓ Example 6 completed\n")

# ============================================================================
# EXAMPLE 7: Reduce LR on Plateau - Adaptive
# ============================================================================
print("EXAMPLE 7: Reduce LR on Plateau - Adaptive")
print("-" * 80)

scheduler = ReduceLROnPlateau(initial_lr=0.1, mode='min', patience=5, factor=0.5)

print("Adaptive LR reduction based on validation loss:")
print(f"Initial LR: {scheduler.initial_lr}")
print(f"Patience: {scheduler.patience} epochs")
print(f"Reduction factor: {scheduler.factor}")
print()

# Simulate validation losses
val_losses = [1.0, 0.9, 0.8, 0.75, 0.74, 0.74, 0.74, 0.74, 0.74, 0.73]

for epoch, val_loss in enumerate(val_losses):
    lr = scheduler.step(val_loss)
    print(f"Epoch {epoch}: Val Loss = {val_loss:.2f}, LR = {lr:.6f}")

print("\n✓ Example 7 completed\n")

# ============================================================================
# EXAMPLE 8: Polynomial LR - Smooth Decay
# ============================================================================
print("EXAMPLE 8: Polynomial LR - Smooth Decay")
print("-" * 80)

scheduler = PolynomialLR(initial_lr=0.1, total_epochs=100, end_lr=0.001, power=2.0)

print("Polynomial decay with power=2:")
lrs = []
for epoch in range(100):
    lr = scheduler.step(epoch)
    lrs.append(lr)
    if epoch % 25 == 0:
        print(f"Epoch {epoch:3d}: LR = {lr:.6f}")

print(f"\nFinal LR: {lrs[-1]:.6f}")

print("\n✓ Example 8 completed\n")

# ============================================================================
# EXAMPLE 9: Warmup LR - Gradual Start
# ============================================================================
print("EXAMPLE 9: Warmup LR - Gradual Start")
print("-" * 80)

scheduler = WarmupLR(target_lr=0.1, warmup_steps=1000)

print("Linear warmup for stable training start:")
print(f"Target LR: {scheduler.target_lr}")
print(f"Warmup steps: {scheduler.warmup_steps}")
print()

steps = [0, 250, 500, 750, 1000, 1500]
for step in steps:
    lr = scheduler.step(step)
    print(f"Step {step:4d}: LR = {lr:.6f}")

print("\n✓ Example 9 completed\n")

# ============================================================================
# EXAMPLE 10: Multi-Step LR - Multiple Milestones
# ============================================================================
print("EXAMPLE 10: Multi-Step LR - Multiple Milestones")
print("-" * 80)

scheduler = MultiStepLR(initial_lr=0.1, milestones=[30, 60, 90], gamma=0.1)

print("Multi-step decay at specific epochs:")
print(f"Initial LR: {scheduler.initial_lr}")
print(f"Milestones: {scheduler.milestones}")
print(f"Gamma: {scheduler.gamma}")
print()

epochs = [0, 29, 30, 59, 60, 89, 90, 100]
for epoch in epochs:
    lr = scheduler.step(epoch)
    print(f"Epoch {epoch:3d}: LR = {lr:.6f}")

print("\n✓ Example 10 completed\n")

# ============================================================================
# EXAMPLE 11: Combining Warmup + Cosine Annealing
# ============================================================================
print("EXAMPLE 11: Combining Warmup + Cosine Annealing")
print("-" * 80)

warmup = WarmupLR(target_lr=0.1, warmup_steps=100)
cosine = CosineAnnealingLR(initial_lr=0.1, T_max=900, eta_min=0.001)

print("Warmup followed by cosine annealing:")
lrs = []
for step in range(1000):
    if step < 100:
        lr = warmup.step(step)
    else:
        lr = cosine.step(step - 100)
    lrs.append(lr)
    if step in [0, 50, 100, 500, 999]:
        print(f"Step {step:4d}: LR = {lr:.6f}")

print("\n✓ Example 11 completed\n")

# ============================================================================
# EXAMPLE 12: Scheduler Comparison
# ============================================================================
print("EXAMPLE 12: Scheduler Comparison")
print("-" * 80)

schedulers = {
    'StepLR': StepLR(initial_lr=0.1, step_size=30, gamma=0.1),
    'ExponentialLR': ExponentialLR(initial_lr=0.1, gamma=0.95),
    'CosineAnnealingLR': CosineAnnealingLR(initial_lr=0.1, T_max=100, eta_min=0.001),
    'PolynomialLR': PolynomialLR(initial_lr=0.1, total_epochs=100, end_lr=0.001, power=2.0),
}

print("Comparing different schedulers over 100 epochs:")
print()

all_lrs = {}
for name, scheduler in schedulers.items():
    lrs = [scheduler.step(i) for i in range(100)]
    all_lrs[name] = lrs
    print(f"{name:20s}: Start={lrs[0]:.6f}, Mid={lrs[50]:.6f}, End={lrs[-1]:.6f}")

print("\n✓ Example 12 completed\n")

# ============================================================================
# EXAMPLE 13: Training Simulation
# ============================================================================
print("EXAMPLE 13: Training Simulation")
print("-" * 80)

scheduler = OneCycleLR(initial_lr=0.001, max_lr=0.1, total_steps=100)

print("Simulating training with One Cycle LR:")
print()

# Simulate training
for epoch in range(10):
    epoch_lrs = []
    for step in range(10):
        global_step = epoch * 10 + step
        lr = scheduler.step(global_step)
        epoch_lrs.append(lr)
    
    avg_lr = np.mean(epoch_lrs)
    print(f"Epoch {epoch + 1:2d}: Avg LR = {avg_lr:.6f}, Min = {min(epoch_lrs):.6f}, Max = {max(epoch_lrs):.6f}")

print("\n✓ Example 13 completed\n")

# ============================================================================
# EXAMPLE 14: Scheduler Selection Guide
# ============================================================================
print("EXAMPLE 14: Scheduler Selection Guide")
print("-" * 80)

print("When to use each scheduler:")
print()
print("1. StepLR:")
print("   - Image classification (ResNet, VGG)")
print("   - When you know good decay points")
print("   - Simple and effective")
print()
print("2. ExponentialLR:")
print("   - Smooth continuous decay needed")
print("   - Long training runs")
print("   - Reinforcement learning")
print()
print("3. CosineAnnealingLR:")
print("   - Deep networks")
print("   - Better than step decay")
print("   - Smooth convergence")
print()
print("4. CosineAnnealingWarmRestarts (SGDR):")
print("   - Escape local minima")
print("   - Ensemble training")
print("   - Snapshot ensembles")
print()
print("5. OneCycleLR:")
print("   - Fast training (super-convergence)")
print("   - Limited time/resources")
print("   - Modern best practice")
print()
print("6. CyclicLR:")
print("   - Finding optimal LR range")
print("   - Regularization effect")
print("   - Exploration during training")
print()
print("7. ReduceLROnPlateau:")
print("   - Unknown optimal schedule")
print("   - Adaptive to training dynamics")
print("   - Safe default choice")
print()
print("8. PolynomialLR:")
print("   - Semantic segmentation")
print("   - Object detection")
print("   - Smooth decay preferred")
print()
print("9. WarmupLR:")
print("   - Large batch training")
print("   - Transformers (BERT, GPT)")
print("   - Stabilize early training")
print()
print("10. MultiStepLR:")
print("    - Multiple known decay points")
print("    - Fine-grained control")
print("    - Standard in many papers")

print("\n✓ Example 14 completed\n")

# ============================================================================
# EXAMPLE 15: Performance Metrics
# ============================================================================
print("EXAMPLE 15: Performance Metrics")
print("-" * 80)

print("Scheduler characteristics:")
print()

schedulers_info = [
    ("StepLR", "Simple", "Low", "Discrete jumps"),
    ("ExponentialLR", "Simple", "Low", "Smooth decay"),
    ("CosineAnnealingLR", "Moderate", "Low", "Smooth, better than step"),
    ("SGDR", "Moderate", "Medium", "Periodic restarts"),
    ("OneCycleLR", "Complex", "Medium", "Fast convergence"),
    ("CyclicLR", "Moderate", "Medium", "Oscillating"),
    ("ReduceLROnPlateau", "Simple", "High", "Adaptive"),
    ("PolynomialLR", "Simple", "Low", "Smooth polynomial"),
    ("WarmupLR", "Simple", "Low", "Linear increase"),
    ("MultiStepLR", "Simple", "Low", "Multiple steps"),
]

print(f"{'Scheduler':<25} {'Complexity':<12} {'Overhead':<10} {'Behavior':<25}")
print("-" * 75)
for name, complexity, overhead, behavior in schedulers_info:
    print(f"{name:<25} {complexity:<12} {overhead:<10} {behavior:<25}")

print("\n✓ Example 15 completed\n")

# ============================================================================
# SUMMARY
# ============================================================================
print("=" * 80)
print("ALL EXAMPLES COMPLETED SUCCESSFULLY! ✓")
print("=" * 80)
print()
print("Summary of what we covered:")
print("1. ✓ StepLR - Image Classification")
print("2. ✓ ExponentialLR - Continuous Decay")
print("3. ✓ CosineAnnealingLR - Smooth Decay")
print("4. ✓ SGDR - Warm Restarts")
print("5. ✓ OneCycleLR - Super Convergence")
print("6. ✓ CyclicLR - Triangular Policy")
print("7. ✓ ReduceLROnPlateau - Adaptive")
print("8. ✓ PolynomialLR - Smooth Decay")
print("9. ✓ WarmupLR - Gradual Start")
print("10. ✓ MultiStepLR - Multiple Milestones")
print("11. ✓ Combining Schedulers")
print("12. ✓ Scheduler Comparison")
print("13. ✓ Training Simulation")
print("14. ✓ Scheduler Selection Guide")
print("15. ✓ Performance Metrics")
print()
print("You now have a complete understanding of learning rate schedulers!")
print()
print("Next steps:")
print("- Choose scheduler based on your task")
print("- Experiment with hyperparameters")
print("- Monitor training curves")
print("- Combine with other techniques (warmup, etc.)")
