# Release Notes - v0.2.14

**Release Date:** December 21, 2024

## 🎯 NEW: Loss Functions Module

Added comprehensive loss functions for regression, classification, and segmentation with 20+ functions!

### 📦 What's New

#### Regression Losses (7 functions)
1. **mean_squared_error_loss()** - MSE (most common)
2. **mean_absolute_error_loss()** - MAE (robust to outliers)
3. **root_mean_squared_error_loss()** - RMSE (same scale as target)
4. **huber_loss()** - Smooth MSE + MAE combination
5. **log_cosh_loss()** - Smooth and robust
6. **quantile_loss()** - Quantile regression
7. **mean_squared_logarithmic_error()** - MSLE (exponential growth)

#### Classification Losses (8 functions)
8. **binary_crossentropy_loss()** - Binary classification
9. **categorical_crossentropy_loss()** - Multi-class (one-hot)
10. **sparse_categorical_crossentropy_loss()** - Multi-class (integers)
11. **hinge_loss()** - SVM loss
12. **squared_hinge_loss()** - Smooth hinge
13. **categorical_hinge_loss()** - Multi-class hinge
14. **focal_loss()** - Handles class imbalance
15. **kullback_leibler_divergence()** - KL divergence

#### Segmentation Losses (5 functions)
16. **dice_loss()** - Overlap-based (medical imaging)
17. **dice_coefficient()** - Dice score
18. **iou_loss()** - Intersection over Union
19. **tversky_loss()** - Generalized Dice
20. **focal_tversky_loss()** - Focal + Tversky

#### Utilities (2 functions)
21. **combined_loss()** - Combine multiple losses
22. **weighted_loss()** - Apply sample weights

## 💻 Installation

```bash
pip install --upgrade ilovetools
```

## ✅ Quick Start

```python
from ilovetools.ml import (
    mean_squared_error_loss,
    binary_crossentropy_loss,
    focal_loss,
    dice_loss,
    combined_loss
)
import numpy as np

# Regression: MSE
y_true = np.array([1.0, 2.0, 3.0, 4.0])
y_pred = np.array([1.1, 2.2, 2.8, 4.3])
mse = mean_squared_error_loss(y_true, y_pred)
print(f"MSE: {mse:.4f}")

# Binary Classification: BCE
y_true = np.array([0, 1, 1, 0])
y_pred = np.array([0.1, 0.9, 0.8, 0.2])
bce = binary_crossentropy_loss(y_true, y_pred)
print(f"BCE: {bce:.4f}")

# Imbalanced Classification: Focal Loss
focal = focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0)
print(f"Focal: {focal:.4f}")

# Segmentation: Dice Loss
y_true_seg = np.array([1, 1, 0, 0, 1])
y_pred_seg = np.array([1, 1, 0, 1, 1])
dice = dice_loss(y_true_seg, y_pred_seg)
print(f"Dice: {dice:.4f}")

# Combined Loss
combined = combined_loss(
    y_true_seg, y_pred_seg,
    loss_functions=[dice_loss, binary_crossentropy_loss],
    weights=[0.7, 0.3]
)
print(f"Combined: {combined:.4f}")
```

## 🎯 Use Cases

✅ Train regression models (house prices, stocks)
✅ Binary classification (spam detection)
✅ Multi-class classification (image recognition)
✅ Imbalanced datasets (fraud detection)
✅ Medical image segmentation
✅ Object detection
✅ Custom multi-task learning

## 📊 Loss Function Selection Guide

### Regression
| Problem | Loss Function |
|---------|---------------|
| Normal data | MSE |
| Outliers present | MAE or Huber |
| Need smoothness | Huber or Log-Cosh |
| Exponential growth | MSLE |
| Quantile prediction | Quantile Loss |

### Classification
| Problem | Loss Function |
|---------|---------------|
| Binary | Binary Cross-Entropy |
| Multi-class | Categorical CE |
| Many classes | Sparse Categorical CE |
| Imbalanced | Focal Loss |
| SVM | Hinge Loss |

### Segmentation
| Problem | Loss Function |
|---------|---------------|
| Medical imaging | Dice Loss |
| Object detection | IoU Loss |
| Imbalanced masks | Focal + Dice |
| Custom FP/FN trade-off | Tversky Loss |

## 🔗 Links

- **PyPI:** https://pypi.org/project/ilovetools/
- **GitHub:** https://github.com/AliMehdi512/ilovetools
- **Loss Functions Code:** https://github.com/AliMehdi512/ilovetools/blob/main/ilovetools/ml/loss_functions.py
- **Tests:** https://github.com/AliMehdi512/ilovetools/blob/main/tests/test_loss_functions.py

## 📈 Total ML Functions

- **Previous (v0.2.13):** ~190 functions
- **New (v0.2.14):** **210+ functions** (20+ new loss functions!)

## 🎓 Educational Content

Check out our LinkedIn post for a complete guide to loss functions:
https://www.linkedin.com/feed/update/urn:li:share:7409441470656471042

## 🔧 Technical Details

### Loss Function Properties

All loss functions in this module satisfy:
- ✅ Differentiable (for gradient descent)
- ✅ Non-negative
- ✅ Zero at perfect prediction
- ✅ Increases with error
- ✅ Numerically stable

### Combining Losses

```python
# Example: Segmentation with Dice + BCE
from ilovetools.ml import combined_loss, dice_loss, binary_crossentropy_loss

total_loss = combined_loss(
    y_true, y_pred,
    loss_functions=[dice_loss, binary_crossentropy_loss],
    weights=[0.7, 0.3]  # 70% Dice, 30% BCE
)
```

### Sample Weighting

```python
# Example: Handle class imbalance
from ilovetools.ml import weighted_loss, mean_squared_error_loss

weights = np.array([1.0, 1.0, 2.0, 2.0])  # More weight on last two
loss = weighted_loss(y_true, y_pred, mean_squared_error_loss, weights)
```

## 🙏 Thank You

Thank you for using ilovetools! We're committed to providing the best ML utilities for Python developers.

## 📝 Version History

- **v0.2.14** (Dec 21, 2024): ✅ Loss functions module
- **v0.2.13** (Dec 20, 2024): Regularization techniques
- **v0.2.12** (Dec 18, 2024): Gradient descent optimization

---

**Train Better Models! 🎯**
