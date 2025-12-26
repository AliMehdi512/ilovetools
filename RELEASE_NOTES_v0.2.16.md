# Release Notes - v0.2.16

**Release Date:** December 25, 2024

## 🚀 NEW: Advanced Optimizers Module

Added 11 state-of-the-art optimization algorithms for neural network training!

### 📦 What's New

#### Adam Variants (5 optimizers)
1. **adam_optimizer()** - Adaptive Moment Estimation (most popular)
2. **adamw_optimizer()** - Adam with decoupled weight decay (BERT, GPT)
3. **adamax_optimizer()** - Adam with infinity norm (stable with large gradients)
4. **nadam_optimizer()** - Nesterov-accelerated Adam
5. **amsgrad_optimizer()** - Adam with maximum of past squared gradients

#### RMSprop Variants (2 optimizers)
6. **rmsprop_optimizer()** - Root Mean Square Propagation (good for RNNs)
7. **rmsprop_momentum_optimizer()** - RMSprop with momentum

#### Modern Optimizers (4 optimizers)
8. **radam_optimizer()** - Rectified Adam (fixes warmup issues)
9. **lamb_optimizer()** - Layer-wise Adaptive Moments (large batch training)
10. **lookahead_optimizer()** - Maintains slow and fast weights
11. **adabelief_optimizer()** - Adapts based on gradient prediction belief

#### Utilities (2 functions)
12. **create_optimizer_state()** - Initialize optimizer state
13. **get_optimizer_function()** - Get optimizer by name

## 💻 Installation

```bash
pip install --upgrade ilovetools
```

## ✅ Quick Start

```python
from ilovetools.ml.optimizers import (
    adam_optimizer,
    adamw_optimizer,
    radam_optimizer,
    lamb_optimizer
)
import numpy as np

# Initialize parameters
params = np.array([1.0, 2.0, 3.0])
grads = np.array([0.1, 0.2, 0.3])

# Adam - Most popular optimizer
m = np.zeros_like(params)
v = np.zeros_like(params)
new_params, m, v = adam_optimizer(params, grads, m, v, t=1)
print(f"Adam: {params} -> {new_params}")

# AdamW - Better for transformers (BERT, GPT)
m = np.zeros_like(params)
v = np.zeros_like(params)
new_params, m, v = adamw_optimizer(
    params, grads, m, v, t=1, weight_decay=0.01
)
print(f"AdamW: {params} -> {new_params}")

# RAdam - No warmup needed
m = np.zeros_like(params)
v = np.zeros_like(params)
new_params, m, v = radam_optimizer(params, grads, m, v, t=1)
print(f"RAdam: {params} -> {new_params}")

# LAMB - Large batch training (BERT with 65k batch)
m = np.zeros_like(params)
v = np.zeros_like(params)
new_params, m, v = lamb_optimizer(params, grads, m, v, t=1)
print(f"LAMB: {params} -> {new_params}")
```

## 🎯 Optimizer Selection Guide

### By Use Case

| Use Case | Optimizer | Why |
|----------|-----------|-----|
| **Default choice** | Adam | Fast, reliable, works well |
| **Transformers (BERT, GPT)** | AdamW | Decoupled weight decay |
| **Large batch training** | LAMB | Layer-wise adaptation |
| **No warmup** | RAdam | Auto-adjusts early learning |
| **RNNs/LSTMs** | RMSprop | Good for recurrent networks |
| **Better generalization** | AdaBelief | Adapts to gradient belief |
| **Reduce variance** | Lookahead | Slow + fast weights |

### By Problem Type

**Computer Vision:**
- Standard: Adam or AdamW
- Large models: LAMB
- Fine-tuning: AdamW with low weight decay

**Natural Language Processing:**
- Transformers: AdamW (BERT, GPT standard)
- RNNs: RMSprop or Adam
- Large batch: LAMB

**Reinforcement Learning:**
- Policy gradients: Adam
- Value functions: RMSprop
- Actor-Critic: Adam for both

## 📊 Performance Comparison

### Convergence Speed
**Fastest:** Adam ≈ AdamW ≈ RAdam  
**Medium:** Nadam ≈ AdaBelief  
**Slower:** RMSprop ≈ AMSGrad

### Generalization
**Best:** AdamW ≈ AdaBelief ≈ Lookahead  
**Good:** RAdam ≈ LAMB  
**Standard:** Adam ≈ Nadam

### Memory Usage
**Lowest:** RMSprop (1 state)  
**Medium:** Adam variants (2 states)  
**Highest:** AMSGrad, Lookahead (3+ states)

## 🔧 Advanced Usage

### Training Loop Example

```python
from ilovetools.ml.optimizers import adam_optimizer, create_optimizer_state
import numpy as np

# Initialize
params = np.random.randn(100)
state = create_optimizer_state(params.shape, 'adam')

# Training loop
for epoch in range(100):
    # Forward pass (your model)
    loss, grads = compute_loss_and_gradients(params)
    
    # Update with Adam
    state['t'] += 1
    params, state['m'], state['v'] = adam_optimizer(
        params, grads, state['m'], state['v'], state['t'],
        learning_rate=0.001
    )
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")
```

### Combining with Lookahead

```python
from ilovetools.ml.optimizers import adam_optimizer, lookahead_optimizer

# Fast weights (Adam)
params = np.random.randn(100)
m = np.zeros_like(params)
v = np.zeros_like(params)

# Slow weights (Lookahead)
slow_params = params.copy()
k_counter = 0

for t in range(1, 1001):
    # Inner optimizer (Adam)
    grads = compute_gradients(params)
    params, m, v = adam_optimizer(params, grads, m, v, t)
    
    # Outer optimizer (Lookahead)
    params, slow_params, k_counter = lookahead_optimizer(
        params, slow_params, k_counter, k=5, alpha=0.5
    )
```

### AdamW for Transformers

```python
from ilovetools.ml.optimizers import adamw_optimizer

# BERT-style training
params = initialize_bert_params()
m = np.zeros_like(params)
v = np.zeros_like(params)

for t in range(1, num_steps + 1):
    grads = compute_gradients(params)
    
    # AdamW with typical BERT settings
    params, m, v = adamw_optimizer(
        params, grads, m, v, t,
        learning_rate=1e-4,
        weight_decay=0.01,  # Decoupled weight decay
        beta1=0.9,
        beta2=0.999
    )
```

### LAMB for Large Batch

```python
from ilovetools.ml.optimizers import lamb_optimizer

# Large batch training (like BERT with 65k batch)
params = initialize_model_params()
m = np.zeros_like(params)
v = np.zeros_like(params)

for t in range(1, num_steps + 1):
    # Large batch gradients
    grads = compute_large_batch_gradients(params, batch_size=65536)
    
    # LAMB with layer-wise adaptation
    params, m, v = lamb_optimizer(
        params, grads, m, v, t,
        learning_rate=0.00176,  # Scaled for large batch
        weight_decay=0.01
    )
```

## 💡 Pro Tips

✅ **Start with Adam** - Default choice for most problems  
✅ **Use AdamW for transformers** - Standard for BERT, GPT  
✅ **Try RAdam if no warmup** - Automatically adjusts early learning  
✅ **Use LAMB for large batches** - Enables batch sizes > 32k  
✅ **Combine with Lookahead** - Reduces variance, improves convergence  
✅ **Monitor gradient norms** - Helps diagnose training issues  

❌ **Don't use same LR for all optimizers** - Each has optimal range  
❌ **Don't forget weight decay** - Important for generalization  
❌ **Don't ignore warmup** - Except with RAdam  
❌ **Don't use Adam for large batch** - Use LAMB instead  

## 🔬 Technical Details

### Optimizer Properties

All optimizers in this module:
- ✅ Vectorized (NumPy)
- ✅ Numerically stable
- ✅ Bias correction (where applicable)
- ✅ Fully documented with examples
- ✅ Tested with 100% coverage

### State Management

```python
from ilovetools.ml.optimizers import create_optimizer_state, get_optimizer_function

# Create state for any optimizer
state = create_optimizer_state(params.shape, 'adam')
# Returns: {'m': zeros, 'v': zeros, 't': 0}

# Get optimizer function by name
opt_fn = get_optimizer_function('adamw')
# Returns: adamw_optimizer function
```

## 🔗 Links

- **PyPI:** https://pypi.org/project/ilovetools/
- **GitHub:** https://github.com/AliMehdi512/ilovetools
- **Optimizers Code:** https://github.com/AliMehdi512/ilovetools/blob/main/ilovetools/ml/optimizers.py
- **Tests:** https://github.com/AliMehdi512/ilovetools/blob/main/tests/test_optimizers.py
- **Verification:** https://github.com/AliMehdi512/ilovetools/blob/main/scripts/verify_optimizers.py

## 📈 Total ML Functions

- **Previous (v0.2.15):** 235+ functions
- **New (v0.2.16):** **248+ functions** (13+ new optimizer functions!)

## 🎓 Educational Content

Check out our LinkedIn posts:
- **Loss Functions Guide:** https://www.linkedin.com/feed/update/urn:li:share:7410204146366189569
- **Activation Functions:** https://www.linkedin.com/feed/update/urn:li:share:7409972257818775552

## 📚 Research Papers

These optimizers are based on:
- **Adam:** Kingma & Ba (2014)
- **AdamW:** Loshchilov & Hutter (2017)
- **RAdam:** Liu et al. (2019)
- **LAMB:** You et al. (2019)
- **Lookahead:** Zhang et al. (2019)
- **AdaBelief:** Zhuang et al. (2020)

## 🚀 What's Next

Coming in future releases:
- Learning rate schedulers
- Gradient accumulation
- Mixed precision training
- Distributed optimization

## 🙏 Thank You

Thank you for using ilovetools! We're committed to providing the best ML utilities for Python developers.

## 📝 Version History

- **v0.2.16** (Dec 25, 2024): ✅ Advanced optimizers module
- **v0.2.15** (Dec 25, 2024): Activation functions module
- **v0.2.14** (Dec 21, 2024): Loss functions module
- **v0.2.13** (Dec 20, 2024): Regularization techniques
- **v0.2.12** (Dec 18, 2024): Gradient descent optimization

---

**Train Faster, Generalize Better! 🚀**
