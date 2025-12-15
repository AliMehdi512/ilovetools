# Release Notes - v0.2.7

**Release Date:** December 15, 2024

## 🔧 Critical Fix

### Fixed Optional Import Issue

**Problem:** Version 0.2.6 had a missing `Optional` import in `config_converter.py` that caused:
```python
NameError: name 'Optional' is not defined
```

This prevented users from importing ML functions:
```python
from ilovetools.ml import sigmoid, relu, softmax  # ❌ Failed in v0.2.6
```

**Solution:** Added `Optional` to the typing imports on line 8:
```python
# Before (v0.2.6)
from typing import Any, Dict, List, Union

# After (v0.2.7)
from typing import Any, Dict, List, Union, Optional
```

**Impact:** All imports now work correctly without any manual fixes!

## 📦 Installation

### New Installation (Recommended)
```bash
pip install ilovetools==0.2.7
```

### Upgrade from v0.2.6
```bash
pip install --upgrade ilovetools
```

## ✅ Verification

Test that the fix works:
```python
from ilovetools.ml import sigmoid, relu, softmax
import numpy as np

x = np.array([-2, -1, 0, 1, 2])
print(sigmoid(x))  # ✅ Works!
print(relu(x))     # ✅ Works!
```

## 📝 What's Included

All features from v0.2.6 are included:

### Neural Network Utilities (35+ functions)
- ✅ Activation Functions (7): sigmoid, relu, leaky_relu, tanh, softmax, elu, swish
- ✅ Loss Functions (4): mse_loss, binary_crossentropy, categorical_crossentropy, huber_loss
- ✅ Weight Initialization (5): xavier_init, he_init, random_init, zeros_init, ones_init
- ✅ Layer Operations (4): dense_forward, dense_backward, dropout_forward, batch_norm_forward
- ✅ Optimizers (4): sgd_update, momentum_update, adam_update, rmsprop_update
- ✅ Utilities (5): one_hot_encode, shuffle_data, mini_batch_generator, calculate_accuracy, confusion_matrix_nn

## 🔗 Links

- **PyPI:** https://pypi.org/project/ilovetools/
- **GitHub:** https://github.com/AliMehdi512/ilovetools
- **Neural Network Code:** https://github.com/AliMehdi512/ilovetools/blob/main/ilovetools/ml/neural_network.py
- **Tests:** https://github.com/AliMehdi512/ilovetools/blob/main/tests/test_neural_network.py
- **Fix Commit:** https://github.com/AliMehdi512/ilovetools/commit/4dae40a23d811fbd5996fbb7eb87ffd1a45fc2c9

## 🙏 Thank You

Thank you for your patience! If you encountered the import error in v0.2.6, please upgrade to v0.2.7.

---

**Previous Version Issues:**
- v0.2.6: ❌ Missing Optional import (Fixed in v0.2.7)
- v0.2.7: ✅ All imports working correctly
