# Release Notes - v0.2.13

**Release Date:** December 20, 2024

## 🛡️ NEW: Regularization Techniques Module

Added comprehensive regularization methods to prevent overfitting with 15+ functions!

### 📦 What's New

#### L1/L2 Regularization (6 functions)
1. **l1_regularization()** - Lasso penalty (sparse models)
2. **l2_regularization()** - Ridge penalty (smooth shrinkage)
3. **elastic_net_regularization()** - L1 + L2 combination
4. **l1_penalty()** - L1 gradient penalty
5. **l2_penalty()** - L2 gradient penalty
6. **elastic_net_penalty()** - Elastic Net gradient penalty

#### Dropout (3 functions)
7. **dropout()** - Standard dropout with mask
8. **dropout_mask()** - Generate dropout masks
9. **inverted_dropout()** - Inverted dropout (preferred)

#### Early Stopping (2 functions)
10. **reg_early_stopping_monitor()** - Monitor validation loss
11. **should_stop_early()** - Stateless early stopping check

#### Weight Constraints (3 functions)
12. **max_norm_constraint()** - Constrain weight norms
13. **unit_norm_constraint()** - Normalize to unit norm
14. **non_negative_constraint()** - Force non-negative weights

#### Utilities (2 functions)
15. **compute_regularization_loss()** - Unified regularization loss
16. **apply_weight_decay()** - Weight decay (L2 in optimizer)

## 💻 Installation

```bash
pip install --upgrade ilovetools
```

## ✅ Quick Start

```python
from ilovetools.ml import (
    l1_regularization,
    l2_regularization,
    dropout,
    reg_early_stopping_monitor,
    max_norm_constraint
)
import numpy as np

# L1 Regularization (Lasso)
weights = np.array([1.0, -2.0, 3.0, 0.5])
l1_loss = l1_regularization(weights, lambda_param=0.01)
print(f"L1 penalty: {l1_loss:.4f}")

# L2 Regularization (Ridge)
l2_loss = l2_regularization(weights, lambda_param=0.01)
print(f"L2 penalty: {l2_loss:.4f}")

# Dropout
X = np.random.randn(100, 50)
X_dropout, mask = dropout(X, dropout_rate=0.5, training=True)
print(f"Dropped neurons: {np.sum(mask == 0)}")

# Early Stopping
val_losses = [1.0, 0.8, 0.7, 0.71, 0.72, 0.73]
should_stop, epochs_no_improve, best = reg_early_stopping_monitor(
    val_losses, patience=3
)
print(f"Stop training: {should_stop}, Best loss: {best:.2f}")

# Max Norm Constraint
weights_large = np.random.randn(100, 50) * 10
constrained = max_norm_constraint(weights_large, max_norm=3.0)
norms = np.linalg.norm(constrained, axis=0)
print(f"Max norm after constraint: {norms.max():.2f}")
```

## 🎯 Use Cases

✅ Prevent overfitting in neural networks
✅ Feature selection with L1
✅ Handle multicollinearity with L2
✅ Deep learning with dropout
✅ Automatic early stopping
✅ Constrain weight magnitudes
✅ Improve model generalization

## 📊 Complete Example

```python
from ilovetools.ml import (
    l2_regularization,
    dropout,
    reg_early_stopping_monitor,
    apply_weight_decay
)
import numpy as np

# Training loop with regularization
np.random.seed(42)
X_train = np.random.randn(1000, 100)
y_train = np.random.randn(1000)
X_val = np.random.randn(200, 100)
y_val = np.random.randn(200)

weights = np.random.randn(100) * 0.01
learning_rate = 0.01
lambda_param = 0.01
dropout_rate = 0.3

val_losses = []
best_weights = weights.copy()

for epoch in range(100):
    # Apply dropout during training
    X_train_dropout, _ = dropout(
        X_train, dropout_rate=dropout_rate, training=True
    )
    
    # Forward pass
    predictions = X_train_dropout @ weights
    
    # Compute loss with L2 regularization
    mse_loss = np.mean((predictions - y_train) ** 2)
    reg_loss = l2_regularization(weights, lambda_param)
    total_loss = mse_loss + reg_loss
    
    # Backward pass (simplified)
    gradient = 2 * X_train_dropout.T @ (predictions - y_train) / len(y_train)
    
    # Update weights with weight decay
    weights = weights - learning_rate * gradient
    weights = apply_weight_decay(weights, learning_rate, lambda_param)
    
    # Validation (no dropout)
    val_pred = X_val @ weights
    val_loss = np.mean((val_pred - y_val) ** 2)
    val_losses.append(val_loss)
    
    # Early stopping check
    should_stop, _, best_val = reg_early_stopping_monitor(
        val_losses, patience=10
    )
    
    if should_stop:
        print(f"Early stopping at epoch {epoch}")
        print(f"Best validation loss: {best_val:.4f}")
        break

print(f"Final training loss: {total_loss:.4f}")
print(f"Final validation loss: {val_loss:.4f}")
```

## 🔗 Links

- **PyPI:** https://pypi.org/project/ilovetools/
- **GitHub:** https://github.com/AliMehdi512/ilovetools
- **Regularization Code:** https://github.com/AliMehdi512/ilovetools/blob/main/ilovetools/ml/regularization.py
- **Tests:** https://github.com/AliMehdi512/ilovetools/blob/main/tests/test_regularization.py

## 📈 Total ML Functions

- **Previous (v0.2.12):** ~175 functions
- **New (v0.2.13):** **190+ functions** (15 new regularization functions!)

## 🎓 Educational Content

Check out our LinkedIn post for a complete guide to regularization:
https://www.linkedin.com/feed/update/urn:li:share:7408011799968612354

## 🔧 Technical Details

### Regularization Comparison

| Method | Type | Effect | Best For |
|--------|------|--------|----------|
| L1 (Lasso) | Penalty | Sparse weights | Feature selection |
| L2 (Ridge) | Penalty | Smooth shrinkage | Multicollinearity |
| Elastic Net | Penalty | L1 + L2 | Grouped features |
| Dropout | Stochastic | Random disable | Deep networks |
| Early Stopping | Monitoring | Stop training | All models |
| Max Norm | Constraint | Limit magnitude | Deep learning |

### When to Use What

| Problem | Solution |
|---------|----------|
| Too many features | L1 regularization |
| Correlated features | L2 regularization |
| Overfitting neural net | Dropout + L2 |
| Training too long | Early stopping |
| Exploding weights | Max norm constraint |
| Need interpretability | L1 (sparse model) |

### Typical Hyperparameters

**L1/L2 Lambda:**
- Start: 0.01
- Range: [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]
- Find via cross-validation

**Dropout Rate:**
- Hidden layers: 0.2 - 0.5
- Input layer: 0.1 - 0.2
- Never on output layer

**Early Stopping:**
- Patience: 5 - 20 epochs
- Min delta: 0.0001 - 0.001
- Monitor: validation loss

**Max Norm:**
- Typical: 2.0 - 5.0
- CNNs: 3.0 - 4.0
- RNNs: 1.0 - 3.0

## 🙏 Thank You

Thank you for using ilovetools! We're committed to providing the best ML utilities for Python developers.

## 📝 Version History

- **v0.2.13** (Dec 20, 2024): ✅ Regularization techniques module
- **v0.2.12** (Dec 18, 2024): Gradient descent optimization
- **v0.2.11** (Dec 17, 2024): Previous release

---

**Stop Overfitting! 🛡️**
