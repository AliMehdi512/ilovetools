# Release Notes - v0.2.12

**Release Date:** December 18, 2024

## 🚀 NEW: Gradient Descent Optimization Module

Added comprehensive gradient descent optimization algorithms with 20+ functions!

### 📦 What's New

#### Basic Gradient Descent (4 functions)
1. **gradient_descent()** - Basic gradient descent update
2. **batch_gradient_descent()** - Full dataset per update
3. **stochastic_gradient_descent()** - One sample per update
4. **mini_batch_gradient_descent()** - Small batches (best of both worlds!)

#### Advanced Optimizers (8 functions)
5. **momentum_optimizer()** - Adds velocity to overcome oscillations
6. **nesterov_momentum()** - Look-ahead momentum
7. **adagrad_optimizer()** - Adaptive learning rates per parameter
8. **rmsprop_optimizer()** - Fixes AdaGrad's decay problem
9. **adam_optimizer()** - Most popular! Combines Momentum + RMSProp
10. **adamw_optimizer()** - Adam with weight decay
11. **nadam_optimizer()** - Nesterov + Adam
12. **adadelta_optimizer()** - No learning rate required

#### Learning Rate Schedules (5 functions)
13. **step_decay_schedule()** - Reduce by factor every N epochs
14. **exponential_decay_schedule()** - Exponential reduction
15. **cosine_annealing_schedule()** - Smooth periodic reduction
16. **linear_warmup_schedule()** - Gradual increase then decrease
17. **polynomial_decay_schedule()** - Polynomial decay

#### Utilities (5 functions)
18. **compute_gradient()** - Numerical gradient computation
19. **gradient_clipping()** - Prevent exploding gradients
20. **check_convergence()** - Monitor convergence
21. **line_search()** - Backtracking line search
22. **compute_learning_rate()** - Unified LR computation

## 💻 Installation

```bash
pip install --upgrade ilovetools
```

## ✅ Quick Start

```python
from ilovetools.ml import (
    gradient_descent,
    adam_optimizer,
    cosine_annealing_schedule,
    gradient_clipping
)
import numpy as np

# Basic gradient descent
params = np.array([1.0, 2.0, 3.0])
gradient = np.array([0.1, 0.2, 0.3])
new_params = gradient_descent(params, gradient, learning_rate=0.1)

# Adam optimizer
m = np.zeros(3)
v = np.zeros(3)
new_params, new_m, new_v = adam_optimizer(
    params, gradient, m, v, t=1, learning_rate=0.001
)

# Learning rate scheduling
lr = cosine_annealing_schedule(0.1, epoch=50, total_epochs=100)

# Gradient clipping
gradient = np.array([10.0, 20.0, 30.0])
clipped = gradient_clipping(gradient, max_norm=1.0)
```

## 🎯 Use Cases

✅ Train neural networks from scratch
✅ Implement custom optimizers
✅ Research new optimization algorithms
✅ Educational ML projects
✅ Fine-tune learning rates
✅ Prevent gradient explosions
✅ Monitor training convergence

## 📊 Complete Example

```python
from ilovetools.ml import mini_batch_gradient_descent
import numpy as np

# Generate synthetic data
np.random.seed(42)
X = np.random.randn(1000, 10)
true_params = np.random.randn(10)
y = X @ true_params + np.random.randn(1000) * 0.1

# Define gradient function
def gradient_fn(params, X_batch, y_batch):
    predictions = X_batch @ params
    return X_batch.T @ (predictions - y_batch) / len(y_batch)

# Train with mini-batch gradient descent
params = np.zeros(10)
final_params, loss_history = mini_batch_gradient_descent(
    params, X, y, gradient_fn,
    learning_rate=0.01,
    batch_size=32,
    epochs=100
)

print(f"Final loss: {loss_history[-1]:.6f}")
print(f"Parameter error: {np.linalg.norm(final_params - true_params):.6f}")
```

## 🔗 Links

- **PyPI:** https://pypi.org/project/ilovetools/
- **GitHub:** https://github.com/AliMehdi512/ilovetools
- **Gradient Descent Code:** https://github.com/AliMehdi512/ilovetools/blob/main/ilovetools/ml/gradient_descent.py
- **Tests:** https://github.com/AliMehdi512/ilovetools/blob/main/tests/test_gradient_descent.py

## 📈 Total ML Functions

- **Previous (v0.2.11):** ~155 functions
- **New (v0.2.12):** **175+ functions** (20 new gradient descent functions!)

## 🎓 Educational Content

Check out our LinkedIn post for a complete guide to gradient descent optimization:
https://www.linkedin.com/feed/update/urn:li:share:7407343818112626688

## 🔧 Technical Details

### Optimizer Comparison

| Optimizer | Best For | Learning Rate | Memory |
|-----------|----------|---------------|--------|
| SGD | Simple problems | Manual tuning | Low |
| Momentum | Oscillating loss | Manual tuning | Low |
| Adam | Most problems | Adaptive | Medium |
| AdamW | Transformers/NLP | Adaptive | Medium |
| Nadam | Fast convergence | Adaptive | Medium |
| RMSProp | RNNs | Adaptive | Low |

### Learning Rate Schedules

| Schedule | Use Case | Parameters |
|----------|----------|------------|
| Step Decay | Simple reduction | drop_rate, epochs_drop |
| Exponential | Smooth decay | decay_rate |
| Cosine Annealing | Periodic training | total_epochs |
| Linear Warmup | Transformers | warmup_epochs |
| Polynomial | Custom curves | power |

## 🙏 Thank You

Thank you for using ilovetools! We're committed to providing the best ML utilities for Python developers.

## 📝 Version History

- **v0.2.12** (Dec 18, 2024): ✅ Gradient descent optimization module
- **v0.2.11** (Dec 17, 2024): Previous release
- **v0.2.7.1** (Dec 16, 2024): Fixed Optional import bug

---

**Happy Optimizing! 🚀**
