# Activation Functions - Import Fix

## Issue
Users were getting `ImportError` when trying to import activation functions with short names:

```python
from ilovetools.ml.activations import relu  # ImportError!
```

## Root Cause
Activation functions were only available with the `_activation` suffix:
- `relu_activation` ✅
- `relu` ❌

## Solution
Added convenient aliases at the end of `activations.py`:

```python
# Convenient aliases without _activation suffix
sigmoid = sigmoid_activation
tanh = tanh_activation
relu = relu_activation
leaky_relu = leaky_relu_activation
elu = elu_activation
selu = selu_activation
gelu = gelu_activation
swish = swish_activation
mish = mish_activation
softplus = softplus_activation
softsign = softsign_activation
hard_sigmoid = hard_sigmoid_activation
hard_tanh = hard_tanh_activation
softmax = softmax_activation
log_softmax = log_softmax_activation
```

## Now Both Work!

### Option 1: Short names (NEW)
```python
from ilovetools.ml.activations import relu, sigmoid, tanh

x = np.array([-2, -1, 0, 1, 2])
output = relu(x)  # Works!
```

### Option 2: Full names (STILL WORKS)
```python
from ilovetools.ml.activations import relu_activation

x = np.array([-2, -1, 0, 1, 2])
output = relu_activation(x)  # Also works!
```

## Available Aliases

All 15 activation functions now have short aliases:

1. `sigmoid` → `sigmoid_activation`
2. `tanh` → `tanh_activation`
3. `relu` → `relu_activation`
4. `leaky_relu` → `leaky_relu_activation`
5. `elu` → `elu_activation`
6. `selu` → `selu_activation`
7. `gelu` → `gelu_activation`
8. `swish` → `swish_activation`
9. `mish` → `mish_activation`
10. `softplus` → `softplus_activation`
11. `softsign` → `softsign_activation`
12. `hard_sigmoid` → `hard_sigmoid_activation`
13. `hard_tanh` → `hard_tanh_activation`
14. `softmax` → `softmax_activation`
15. `log_softmax` → `log_softmax_activation`

## Testing

Run the test script to verify:

```bash
python scripts/test_activation_aliases.py
```

Expected output:
```
✓ All activation aliases imported successfully
✓ relu([-2 -1  0  1  2]) = [0 0 0 1 2]
✓ sigmoid([-2 -1  0  1  2]) = [0.1192 0.2689 0.5000 0.7311 0.8808]
✓ tanh([-2 -1  0  1  2]) = [-0.9640 -0.7616  0.0000  0.7616  0.9640]
✓ softmax([-2 -1  0  1  2]) = [0.0117 0.0317 0.0861 0.2341 0.6364]
✓ gelu([-2 -1  0  1  2]) = [-0.0454 -0.1588  0.0000  0.8412  1.9546]
✓ swish([-2 -1  0  1  2]) = [-0.2384 -0.2689  0.0000  0.7311  1.7616]

✅ All activation function aliases work correctly!
```

## Backward Compatibility

✅ **100% backward compatible** - All existing code using `_activation` suffix continues to work!

## Updated in Version

This fix is included in **ilovetools v0.2.21+**

Install the latest version:
```bash
pip install --upgrade ilovetools
```
