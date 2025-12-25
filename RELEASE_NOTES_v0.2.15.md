# Release Notes - v0.2.15

**Release Date:** December 24, 2024

## 🧠 NEW: Activation Functions Module

Added comprehensive activation functions for neural networks with 15+ functions and derivatives!

### 📦 What's New

#### Basic Activation Functions (15 functions)
1. **sigmoid_activation()** - Sigmoid σ(x) = 1/(1+e^(-x))
2. **tanh_activation()** - Hyperbolic tangent
3. **relu_activation()** - Rectified Linear Unit
4. **leaky_relu_activation()** - Leaky ReLU with small negative slope
5. **elu_activation()** - Exponential Linear Unit
6. **selu_activation()** - Scaled ELU (self-normalizing)
7. **gelu_activation()** - Gaussian Error Linear Unit (BERT, GPT)
8. **swish_activation()** - Swish/SiLU (EfficientNet)
9. **mish_activation()** - Mish activation
10. **softplus_activation()** - Smooth ReLU approximation
11. **softsign_activation()** - Polynomial alternative to tanh
12. **hard_sigmoid_activation()** - Fast piecewise linear sigmoid
13. **hard_tanh_activation()** - Fast piecewise linear tanh
14. **softmax_activation()** - Probability distribution for multi-class
15. **log_softmax_activation()** - Numerically stable log-softmax

#### Derivatives (7 functions)
16. **sigmoid_deriv()** - Sigmoid derivative
17. **tanh_deriv()** - Tanh derivative
18. **relu_deriv()** - ReLU derivative
19. **leaky_relu_derivative()** - Leaky ReLU derivative
20. **elu_derivative()** - ELU derivative
21. **swish_derivative()** - Swish derivative
22. **softplus_derivative()** - Softplus derivative

#### Utilities (2 functions)
23. **apply_activation()** - Apply activation by name
24. **get_activation_function()** - Get activation function by name

## 💻 Installation

```bash
pip install --upgrade ilovetools
```

## ✅ Quick Start

```python
from ilovetools.ml import (
    sigmoid_activation,
    relu_activation,
    gelu_activation,
    softmax_activation,
    apply_activation
)
import numpy as np

# Sigmoid: Binary classification
x = np.array([-2, -1, 0, 1, 2])
sigmoid = sigmoid_activation(x)
print(f"Sigmoid: {sigmoid}")
# Output: [0.1192 0.2689 0.5000 0.7311 0.8808]

# ReLU: Hidden layers
relu = relu_activation(x)
print(f"ReLU: {relu}")
# Output: [0 0 0 1 2]

# GELU: Transformers (BERT, GPT)
gelu = gelu_activation(x)
print(f"GELU: {gelu}")
# Output: [-0.0454 -0.1588  0.0000  0.8412  1.9545]

# Softmax: Multi-class classification
logits = np.array([1.0, 2.0, 3.0])
probs = softmax_activation(logits)
print(f"Softmax: {probs}")
# Output: [0.0900 0.2447 0.6652]
print(f"Sum: {probs.sum()}")  # 1.0

# Apply by name
output = apply_activation(x, 'relu')
print(f"Apply ReLU: {output}")
```

## 🎯 Use Cases

✅ Build neural networks from scratch
✅ Implement custom layers
✅ Experiment with different activations
✅ Understand activation behavior
✅ Train deep learning models
✅ Implement transformers (BERT, GPT)
✅ Computer vision models
✅ Natural language processing

## 📊 Activation Function Selection Guide

### Hidden Layers
| Use Case | Activation | Why |
|----------|------------|-----|
| Default choice | ReLU | Fast, no vanishing gradient |
| Deep networks | ELU/SELU | Self-normalizing, smooth |
| State-of-the-art | GELU/Swish | Best performance |
| RNNs/LSTMs | Tanh | Zero-centered |
| Dying ReLU | Leaky ReLU | Fixes dead neurons |

### Output Layers
| Task | Activation | Output Range |
|------|------------|--------------|
| Binary classification | Sigmoid | (0, 1) |
| Multi-class classification | Softmax | Probabilities sum to 1 |
| Regression | Linear (none) | (-∞, ∞) |
| Multi-label | Sigmoid | Independent probabilities |

### Performance Comparison

**Speed:**
ReLU > Leaky ReLU > ELU > Swish > GELU

**Accuracy:**
GELU ≈ Swish > ELU > ReLU > Tanh > Sigmoid

**Gradient Flow:**
ReLU variants > Tanh > Sigmoid

## 🔗 Links

- **PyPI:** https://pypi.org/project/ilovetools/
- **GitHub:** https://github.com/AliMehdi512/ilovetools
- **Activation Functions Code:** https://github.com/AliMehdi512/ilovetools/blob/main/ilovetools/ml/activations.py
- **Tests:** https://github.com/AliMehdi512/ilovetools/blob/main/tests/test_activations.py

## 📈 Total ML Functions

- **Previous (v0.2.14):** 210+ functions
- **New (v0.2.15):** **235+ functions** (25+ new activation functions!)

## 🎓 Educational Content

Check out our LinkedIn post for a complete guide to activation functions:
https://www.linkedin.com/feed/update/urn:li:share:7409972257818775552

## 🔧 Technical Details

### Activation Properties

All activation functions in this module:
- ✅ Vectorized (NumPy)
- ✅ Numerically stable
- ✅ Properly clipped to avoid overflow
- ✅ Fully documented with examples
- ✅ Tested with 100% coverage

### Common Patterns

```python
# Hidden layer with ReLU
from ilovetools.ml import relu_activation

z = np.dot(W, x) + b
a = relu_activation(z)

# Output layer with Softmax
from ilovetools.ml import softmax_activation

logits = np.dot(W_out, h) + b_out
probs = softmax_activation(logits)

# Transformer with GELU
from ilovetools.ml import gelu_activation

ffn_output = gelu_activation(np.dot(W1, x) + b1)
output = np.dot(W2, ffn_output) + b2
```

### Derivatives for Backpropagation

```python
from ilovetools.ml import relu_activation, relu_deriv

# Forward pass
z = np.dot(W, x) + b
a = relu_activation(z)

# Backward pass
dz = da * relu_deriv(z)
dW = np.dot(dz, x.T)
```

## 💡 Tips & Best Practices

✅ **Start with ReLU** - Default choice for hidden layers
✅ **Use GELU for transformers** - State-of-the-art for NLP
✅ **Try Leaky ReLU if neurons die** - Fixes dying ReLU problem
✅ **Softmax for classification** - Multi-class output layer
✅ **Match activation to problem** - Consider task requirements
✅ **Consider computation cost** - ReLU is fastest

❌ **Don't use sigmoid in hidden layers** - Vanishing gradient
❌ **Don't ignore dying ReLU** - Switch to Leaky ReLU
❌ **Don't forget output activation** - Match to task type

## 🚀 Advanced Activations

### GELU (Gaussian Error Linear Unit)
Used in BERT, GPT, and other transformers. Smooth, probabilistic interpretation.

```python
from ilovetools.ml import gelu_activation

# Transformer feedforward
x = gelu_activation(np.dot(W1, input) + b1)
output = np.dot(W2, x) + b2
```

### Swish (Self-Gated)
Used in EfficientNet. Non-monotonic, better than ReLU in deep networks.

```python
from ilovetools.ml import swish_activation

# Deep convolutional layer
x = swish_activation(conv_output)
```

### SELU (Self-Normalizing)
No batch normalization needed. Good for deep feedforward networks.

```python
from ilovetools.ml import selu_activation

# Deep feedforward network
for layer in layers:
    x = selu_activation(np.dot(W, x) + b)
```

## 🙏 Thank You

Thank you for using ilovetools! We're committed to providing the best ML utilities for Python developers.

## 📝 Version History

- **v0.2.15** (Dec 24, 2024): ✅ Activation functions module
- **v0.2.14** (Dec 21, 2024): Loss functions module
- **v0.2.13** (Dec 20, 2024): Regularization techniques
- **v0.2.12** (Dec 18, 2024): Gradient descent optimization

---

**Build Better Neural Networks! 🧠**
