"""
Comprehensive Examples: Learning Rate Schedulers

This file demonstrates all learning rate scheduling techniques
with practical examples and use cases.
"""

import numpy as np
from ilovetools.ml.lr_schedulers import (
    StepLRScheduler,
    ExponentialLRScheduler,
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    OneCycleLR,
    ReduceLROnPlateau,
    PolynomialLRScheduler,
    LinearWarmupScheduler,
    CyclicalLR,
    LRFinder,
    WarmupCosineScheduler,
    get_scheduler,
)

print("=" * 80)
print("LEARNING RATE SCHEDULERS - COMPREHENSIVE EXAMPLES")
print("=" * 80)
print()

# ============================================================================
# EXAMPLE 1: Step Decay Scheduler (ResNet-style)
# ============================================================================
print("EXAMPLE 1: Step Decay Scheduler (ResNet-style)")
print("-" * 80)

initial_lr = 0.1
step_size = 30
gamma = 0.1
total_epochs = 90

scheduler = StepLRScheduler(initial_lr, step_size, gamma)

print(f"Initial LR: {initial_lr}")
print(f"Step size: {step_size} epochs")
print(f"Gamma: {gamma}")
print()

print("Learning rate schedule:")
for epoch in [0, 29, 30, 59, 60, 89]:
    lr = scheduler.step(epoch)
    print(f"Epoch {epoch:3d}: LR = {lr:.6f}")

print("\n✓ Example 1 completed\n")

# ============================================================================
# EXAMPLE 2: Exponential Decay
# ============================================================================
print("EXAMPLE 2: Exponential Decay")
print("-" * 80)

initial_lr = 0.1
gamma = 0.95
total_epochs = 50

scheduler = ExponentialLRScheduler(initial_lr, gamma)

print(f"Initial LR: {initial_lr}")
print(f"Gamma: {gamma}")
print()

print("Learning rate schedule (every 10 epochs):")
for epoch in range(0, total_epochs, 10):
    lr = scheduler.step(epoch)
    print(f"Epoch {epoch:3d}: LR = {lr:.6f}")

print("\n✓ Example 2 completed\n")

# ============================================================================
# EXAMPLE 3: Cosine Annealing (Transformer-style)
# ============================================================================
print("EXAMPLE 3: Cosine Annealing (Transformer-style)")
print("-" * 80)

initial_lr = 0.1
T_max = 100
eta_min = 0.001

scheduler = CosineAnnealingLR(initial_lr, T_max, eta_min)

print(f"Initial LR: {initial_lr}")
print(f"T_max: {T_max} epochs")
print(f"Min LR: {eta_min}")
print()

print("Learning rate schedule:")
for epoch in [0, 25, 50, 75, 100]:
    lr = scheduler.step(epoch)
    print(f"Epoch {epoch:3d}: LR = {lr:.6f}")

print("\n✓ Example 3 completed\n")

# ============================================================================
# EXAMPLE 4: Cosine Annealing with Warm Restarts (SGDR)
# ============================================================================
print("EXAMPLE 4: Cosine Annealing with Warm Restarts (SGDR)")
print("-" * 80)

initial_lr = 0.1
T_0 = 10
T_mult = 2
eta_min = 0.001

scheduler = CosineAnnealingWarmRestarts(initial_lr, T_0, T_mult, eta_min)

print(f"Initial LR: {initial_lr}")
print(f"T_0: {T_0} epochs")
print(f"T_mult: {T_mult}")
print(f"Min LR: {eta_min}")
print()

print("Learning rate schedule (showing restarts):")
for epoch in range(35):
    lr = scheduler.step(epoch)
    if epoch in [0, 10, 30]:  # Restart points
        print(f"Epoch {epoch:3d}: LR = {lr:.6f} <- RESTART")
    elif epoch % 5 == 0:
        print(f"Epoch {epoch:3d}: LR = {lr:.6f}")

print("\n✓ Example 4 completed\n")

# ============================================================================
# EXAMPLE 5: One Cycle Policy (Super-Convergence)
# ============================================================================
print("EXAMPLE 5: One Cycle Policy (Super-Convergence)")
print("-" * 80)

max_lr = 0.1
total_steps = 1000
pct_start = 0.3

scheduler = OneCycleLR(max_lr, total_steps, pct_start=pct_start)

print(f"Max LR: {max_lr}")
print(f"Total steps: {total_steps}")
print(f"Warmup percentage: {pct_start * 100}%")
print()

print("Learning rate schedule:")
lrs = []
for step in range(total_steps):
    lr = scheduler.step()
    lrs.append(lr)

for step in [0, 100, 300, 500, 700, 999]:
    print(f"Step {step:4d}: LR = {lrs[step]:.6f}")

print(f"\nMax LR reached: {max(lrs):.6f}")
print(f"Final LR: {lrs[-1]:.6f}")

print("\n✓ Example 5 completed\n")

# ============================================================================
# EXAMPLE 6: Reduce on Plateau (Adaptive)
# ============================================================================
print("EXAMPLE 6: Reduce on Plateau (Adaptive)")
print("-" * 80)

initial_lr = 0.1
factor = 0.1
patience = 5

scheduler = ReduceLROnPlateau(initial_lr, mode='min', factor=factor, patience=patience)

print(f"Initial LR: {initial_lr}")
print(f"Factor: {factor}")
print(f"Patience: {patience} epochs")
print()

# Simulate validation losses
val_losses = [2.0, 1.8, 1.6, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4]

print("Validation loss and learning rate:")
for epoch, val_loss in enumerate(val_losses):
    lr = scheduler.step(val_loss)
    print(f"Epoch {epoch:2d}: Val Loss = {val_loss:.2f}, LR = {lr:.6f}")

print("\n✓ Example 6 completed\n")

# ============================================================================
# EXAMPLE 7: Polynomial Decay (BERT-style)
# ============================================================================
print("EXAMPLE 7: Polynomial Decay (BERT-style)")
print("-" * 80)

initial_lr = 0.1
total_steps = 1000
power = 1.0  # Linear decay
end_lr = 0.0

scheduler = PolynomialLRScheduler(initial_lr, total_steps, power, end_lr)

print(f"Initial LR: {initial_lr}")
print(f"Total steps: {total_steps}")
print(f"Power: {power}")
print(f"End LR: {end_lr}")
print()

print("Learning rate schedule:")
for step in [0, 200, 400, 600, 800, 999]:
    for _ in range(step - scheduler.current_step):
        scheduler.step()
    lr = scheduler.get_lr()
    print(f"Step {step:4d}: LR = {lr:.6f}")

print("\n✓ Example 7 completed\n")

# ============================================================================
# EXAMPLE 8: Linear Warmup
# ============================================================================
print("EXAMPLE 8: Linear Warmup")
print("-" * 80)

target_lr = 0.1
warmup_steps = 100

scheduler = LinearWarmupScheduler(target_lr, warmup_steps)

print(f"Target LR: {target_lr}")
print(f"Warmup steps: {warmup_steps}")
print()

print("Learning rate schedule:")
for step in [0, 25, 50, 75, 100, 150]:
    for _ in range(step - scheduler.current_step):
        scheduler.step()
    lr = scheduler.get_lr()
    print(f"Step {step:3d}: LR = {lr:.6f}")

print("\n✓ Example 8 completed\n")

# ============================================================================
# EXAMPLE 9: Cyclical Learning Rate
# ============================================================================
print("EXAMPLE 9: Cyclical Learning Rate")
print("-" * 80)

base_lr = 0.001
max_lr = 0.1
step_size = 100

scheduler = CyclicalLR(base_lr, max_lr, step_size, mode='triangular')

print(f"Base LR: {base_lr}")
print(f"Max LR: {max_lr}")
print(f"Step size: {step_size}")
print()

print("Learning rate schedule (2 cycles):")
lrs = []
for step in range(2 * 2 * step_size):
    lr = scheduler.step()
    lrs.append(lr)

for step in [0, 50, 100, 150, 200, 250, 300, 350, 399]:
    print(f"Step {step:3d}: LR = {lrs[step]:.6f}")

print("\n✓ Example 9 completed\n")

# ============================================================================
# EXAMPLE 10: Learning Rate Finder
# ============================================================================
print("EXAMPLE 10: Learning Rate Finder")
print("-" * 80)

start_lr = 1e-7
end_lr = 10
num_steps = 100

finder = LRFinder(start_lr, end_lr, num_steps)

print(f"Start LR: {start_lr}")
print(f"End LR: {end_lr}")
print(f"Num steps: {num_steps}")
print()

# Simulate training with loss curve
print("Simulating LR range test...")
for step in range(num_steps):
    # Simulate loss (decreases then increases)
    if step < 40:
        loss = 2.0 - step * 0.03
    else:
        loss = 0.8 + (step - 40) * 0.05
    
    lr = finder.step(loss)

suggested_lr = finder.suggest_lr()
print(f"Suggested learning rate: {suggested_lr:.6e}")

lr_history, loss_history = finder.plot_results()
print(f"Recorded {len(lr_history)} LR-loss pairs")

print("\n✓ Example 10 completed\n")

# ============================================================================
# EXAMPLE 11: Warmup + Cosine (BERT/GPT-style)
# ============================================================================
print("EXAMPLE 11: Warmup + Cosine (BERT/GPT-style)")
print("-" * 80)

max_lr = 0.1
warmup_steps = 100
total_steps = 1000

scheduler = WarmupCosineScheduler(max_lr, warmup_steps, total_steps)

print(f"Max LR: {max_lr}")
print(f"Warmup steps: {warmup_steps}")
print(f"Total steps: {total_steps}")
print()

print("Learning rate schedule:")
lrs = []
for step in range(total_steps):
    lr = scheduler.step()
    lrs.append(lr)

for step in [0, 50, 100, 300, 500, 700, 999]:
    print(f"Step {step:4d}: LR = {lrs[step]:.6f}")

print("\n✓ Example 11 completed\n")

# ============================================================================
# EXAMPLE 12: Complete Training Loop with Step Decay
# ============================================================================
print("EXAMPLE 12: Complete Training Loop with Step Decay")
print("-" * 80)

# Hyperparameters
initial_lr = 0.1
step_size = 30
gamma = 0.1
total_epochs = 90
batch_size = 32

scheduler = StepLRScheduler(initial_lr, step_size, gamma)

print("Simulating training with Step Decay...")
print()

# Simulate training
for epoch in range(0, total_epochs, 10):
    lr = scheduler.step(epoch)
    
    # Simulate training loss (decreases over time)
    train_loss = 2.0 * np.exp(-epoch / 30) + np.random.normal(0, 0.1)
    val_loss = 2.2 * np.exp(-epoch / 30) + np.random.normal(0, 0.15)
    
    print(f"Epoch {epoch:3d}: LR = {lr:.6f}, Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

print("\n✓ Example 12 completed\n")

# ============================================================================
# EXAMPLE 13: Comparing Different Schedulers
# ============================================================================
print("EXAMPLE 13: Comparing Different Schedulers")
print("-" * 80)

total_epochs = 100
initial_lr = 0.1

schedulers = {
    'Step Decay': StepLRScheduler(initial_lr, step_size=30, gamma=0.1),
    'Exponential': ExponentialLRScheduler(initial_lr, gamma=0.95),
    'Cosine': CosineAnnealingLR(initial_lr, T_max=total_epochs),
}

print("Comparing schedulers at key epochs:")
print()

for epoch in [0, 25, 50, 75, 99]:
    print(f"Epoch {epoch}:")
    for name, sched in schedulers.items():
        # Reset and step to epoch
        if hasattr(sched, 'current_epoch'):
            sched.current_epoch = 0
        for _ in range(epoch + 1):
            lr = sched.step()
        print(f"  {name:15s}: LR = {lr:.6f}")
    print()

print("✓ Example 13 completed\n")

# ============================================================================
# EXAMPLE 14: Using get_scheduler Factory
# ============================================================================
print("EXAMPLE 14: Using get_scheduler Factory")
print("-" * 80)

initial_lr = 0.1

# Create different schedulers using factory
schedulers_config = [
    ('step', {'step_size': 30, 'gamma': 0.1}),
    ('exponential', {'gamma': 0.95}),
    ('cosine', {'T_max': 100}),
    ('onecycle', {'total_steps': 1000}),
]

print("Creating schedulers using factory:")
for name, kwargs in schedulers_config:
    scheduler = get_scheduler(name, initial_lr, **kwargs)
    print(f"✓ Created {name} scheduler")
    print(f"  Initial LR: {scheduler.get_lr():.6f}")

print("\n✓ Example 14 completed\n")

# ============================================================================
# EXAMPLE 15: Real-World: Image Classification Training
# ============================================================================
print("EXAMPLE 15: Real-World: Image Classification Training")
print("-" * 80)

# Simulate ResNet training on ImageNet
print("Simulating ResNet-50 training on ImageNet...")
print()

initial_lr = 0.1
epochs = 90
batch_size = 256

scheduler = StepLRScheduler(initial_lr, step_size=30, gamma=0.1)

print("Training schedule:")
print(f"Initial LR: {initial_lr}")
print(f"Batch size: {batch_size}")
print(f"Total epochs: {epochs}")
print()

# Simulate training
best_acc = 0
for epoch in range(0, epochs, 10):
    lr = scheduler.step(epoch)
    
    # Simulate metrics
    train_loss = 4.0 * np.exp(-epoch / 25) + np.random.normal(0, 0.1)
    train_acc = min(95, 20 + epoch * 0.8 + np.random.normal(0, 2))
    val_acc = min(92, 18 + epoch * 0.75 + np.random.normal(0, 2))
    
    if val_acc > best_acc:
        best_acc = val_acc
        print(f"Epoch {epoch:3d}: LR = {lr:.6f}, Loss = {train_loss:.4f}, "
              f"Train Acc = {train_acc:.2f}%, Val Acc = {val_acc:.2f}% ★")
    else:
        print(f"Epoch {epoch:3d}: LR = {lr:.6f}, Loss = {train_loss:.4f}, "
              f"Train Acc = {train_acc:.2f}%, Val Acc = {val_acc:.2f}%")

print(f"\nBest validation accuracy: {best_acc:.2f}%")

print("\n✓ Example 15 completed\n")

# ============================================================================
# SUMMARY
# ============================================================================
print("=" * 80)
print("ALL EXAMPLES COMPLETED SUCCESSFULLY! ✓")
print("=" * 80)
print()
print("Summary of what we covered:")
print("1. ✓ Step Decay Scheduler (ResNet-style)")
print("2. ✓ Exponential Decay")
print("3. ✓ Cosine Annealing (Transformer-style)")
print("4. ✓ Cosine Annealing with Warm Restarts (SGDR)")
print("5. ✓ One Cycle Policy (Super-Convergence)")
print("6. ✓ Reduce on Plateau (Adaptive)")
print("7. ✓ Polynomial Decay (BERT-style)")
print("8. ✓ Linear Warmup")
print("9. ✓ Cyclical Learning Rate")
print("10. ✓ Learning Rate Finder")
print("11. ✓ Warmup + Cosine (BERT/GPT-style)")
print("12. ✓ Complete Training Loop")
print("13. ✓ Comparing Schedulers")
print("14. ✓ Using Factory Function")
print("15. ✓ Real-World Image Classification")
print()
print("You now have a complete understanding of learning rate scheduling!")
print()
print("Next steps:")
print("- Choose the right scheduler for your task")
print("- Experiment with different hyperparameters")
print("- Monitor training curves")
print("- Combine with other optimization techniques")
