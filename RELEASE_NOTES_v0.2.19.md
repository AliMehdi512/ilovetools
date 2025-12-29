# Release Notes - v0.2.19

**Release Date:** December 29, 2024

## 🎯 NEW: CNN Operations Module

Added complete Convolutional Neural Network operations - the foundation of computer vision!

### 📦 What's New

#### Convolution Operations (2 functions)
1. **conv2d()** - 2D convolution with stride, padding, dilation
2. **conv2d_fast()** - Fast convolution using im2col

#### Pooling Operations (4 functions)
3. **max_pool2d()** - Max pooling
4. **avg_pool2d()** - Average pooling
5. **global_avg_pool2d()** - Global average pooling
6. **global_max_pool2d()** - Global max pooling

#### Im2Col Transformations (2 functions)
7. **im2col()** - Image to column transformation
8. **col2im()** - Column to image transformation

#### Depthwise/Separable Convolutions (2 functions)
9. **depthwise_conv2d()** - Depthwise convolution (MobileNets)
10. **separable_conv2d()** - Separable convolution (efficient)

#### Utilities (1 function)
11. **calculate_output_size()** - Calculate output dimensions

#### Aliases (6 shortcuts)
12. **maxpool2d**, **avgpool2d**, **global_avgpool**, **global_maxpool**, **depthwise_conv**, **separable_conv**

## 💻 Installation

```bash
pip install --upgrade ilovetools
```

## ✅ Quick Start

```python
from ilovetools.ml.cnn import (
    conv2d,
    max_pool2d,
    avg_pool2d,
    global_avg_pool2d,
    depthwise_conv2d,
    separable_conv2d
)
import numpy as np

# 2D Convolution
input = np.random.randn(8, 3, 224, 224)  # (batch, channels, H, W)
kernel = np.random.randn(64, 3, 3, 3)  # (out_ch, in_ch, kH, kW)

output = conv2d(input, kernel, stride=1, padding='same')
print(f"Conv2D: {input.shape} -> {output.shape}")  # (8, 64, 224, 224)

# With stride 2
output = conv2d(input, kernel, stride=2, padding=1)
print(f"Stride 2: {input.shape} -> {output.shape}")  # (8, 64, 112, 112)

# Max Pooling
pool_input = np.random.randn(8, 64, 28, 28)
pool_output = max_pool2d(pool_input, pool_size=2, stride=2)
print(f"Max Pool: {pool_input.shape} -> {pool_output.shape}")  # (8, 64, 14, 14)

# Average Pooling
avg_output = avg_pool2d(pool_input, pool_size=2, stride=2)
print(f"Avg Pool: {pool_input.shape} -> {avg_output.shape}")  # (8, 64, 14, 14)

# Global Average Pooling
global_input = np.random.randn(8, 512, 7, 7)
global_output = global_avg_pool2d(global_input)
print(f"Global Avg: {global_input.shape} -> {global_output.shape}")  # (8, 512, 1, 1)

# Depthwise Convolution (MobileNets)
dw_input = np.random.randn(8, 32, 56, 56)
dw_kernel = np.random.randn(32, 1, 3, 3)
dw_output = depthwise_conv2d(dw_input, dw_kernel, stride=1, padding='same')
print(f"Depthwise: {dw_input.shape} -> {dw_output.shape}")  # (8, 32, 56, 56)

# Separable Convolution
pw_kernel = np.random.randn(64, 32, 1, 1)
sep_output = separable_conv2d(dw_input, dw_kernel, pw_kernel)
print(f"Separable: {dw_input.shape} -> {sep_output.shape}")  # (8, 64, 56, 56)
```

## 🔧 Advanced Usage

### Building a CNN Block

```python
from ilovetools.ml.cnn import conv2d, max_pool2d
from ilovetools.ml.normalization import batch_normalization
from ilovetools.ml.activations import relu

# Input
x = np.random.randn(32, 3, 224, 224)

# Conv -> BN -> ReLU -> Pool
conv_out = conv2d(x, kernel, stride=1, padding='same')
bn_out, _, _ = batch_normalization(conv_out, gamma, beta, training=True)
relu_out = relu(bn_out)
pool_out = max_pool2d(relu_out, pool_size=2, stride=2)

print(f"CNN Block: {x.shape} -> {pool_out.shape}")
```

### Edge Detection

```python
# Sobel edge detection
input = np.random.randn(1, 1, 28, 28)

# Horizontal edges
h_kernel = np.array([[[[-1, -1, -1],
                       [ 0,  0,  0],
                       [ 1,  1,  1]]]])

edges = conv2d(input, h_kernel, stride=1, padding=0)
```

### MobileNet-Style Block

```python
# Depthwise separable convolution
x = np.random.randn(8, 32, 56, 56)

# Depthwise
dw_kernel = np.random.randn(32, 1, 3, 3)
dw_out = depthwise_conv2d(x, dw_kernel, stride=1, padding='same')

# Pointwise
pw_kernel = np.random.randn(64, 32, 1, 1)
pw_out = conv2d(dw_out, pw_kernel, stride=1, padding=0)

print(f"MobileNet block: {x.shape} -> {pw_out.shape}")
```

## 💡 Pro Tips

✅ **Use 'same' padding** - Preserves spatial dimensions  
✅ **Stack multiple conv layers** - Deeper networks learn better  
✅ **Use small kernels (3×3)** - More efficient than large kernels  
✅ **Add pooling layers** - Reduces computation and overfitting  
✅ **Use depthwise separable** - 8-9x fewer parameters  
✅ **Global pooling before FC** - Reduces parameters dramatically  

❌ **Don't use large kernels only** - Stack small ones instead  
❌ **Don't skip pooling** - Spatial reduction is important  
❌ **Don't forget padding** - Or dimensions shrink quickly  
❌ **Don't ignore stride** - Controls output size  

## 📊 Complexity Analysis

### Convolution
- **Time:** O(batch × out_ch × in_ch × kH × kW × out_H × out_W)
- **Space:** O(batch × out_ch × out_H × out_W)
- **Parameters:** out_ch × in_ch × kH × kW

### Depthwise Separable
- **Parameters:** in_ch × kH × kW + in_ch × out_ch
- **Reduction:** ~8-9x fewer than standard convolution

### Pooling
- **Time:** O(batch × channels × out_H × out_W × pool_H × pool_W)
- **Space:** O(batch × channels × out_H × out_W)
- **Parameters:** 0 (no learnable parameters)

## 🔗 Links

- **PyPI:** https://pypi.org/project/ilovetools/
- **GitHub:** https://github.com/AliMehdi512/ilovetools
- **CNN Code:** https://github.com/AliMehdi512/ilovetools/blob/main/ilovetools/ml/cnn.py
- **Tests:** https://github.com/AliMehdi512/ilovetools/blob/main/tests/test_cnn.py
- **Verification:** https://github.com/AliMehdi512/ilovetools/blob/main/scripts/verify_cnn.py

## 📈 Total ML Functions

- **Previous (v0.2.18):** 283+ functions
- **New (v0.2.19):** **301+ functions** (18+ new CNN functions!)

## 🎓 Educational Content

Check out our LinkedIn posts:
- **CNNs Guide:** https://www.linkedin.com/feed/update/urn:li:share:7411263336107036672
- **Attention Mechanisms:** https://www.linkedin.com/feed/update/urn:li:share:7410931572444385280
- **Normalization:** https://www.linkedin.com/feed/update/urn:li:share:7410522180330983424

## 📚 Research Papers

- **LeNet-5:** LeCun et al. (1998)
- **AlexNet:** Krizhevsky et al. (2012)
- **VGGNet:** Simonyan & Zisserman (2014)
- **ResNet:** He et al. (2015)
- **MobileNets:** Howard et al. (2017)
- **EfficientNet:** Tan & Le (2019)

## 🚀 What's Next

Coming in future releases:
- Transposed convolutions
- Grouped convolutions
- Deformable convolutions
- 3D convolutions

## 📝 Version History

- **v0.2.19** (Dec 29, 2024): ✅ CNN operations module
- **v0.2.18** (Dec 28, 2024): Attention mechanisms module
- **v0.2.17** (Dec 27, 2024): Normalization techniques module
- **v0.2.16** (Dec 25, 2024): Advanced optimizers module

---

**Convolve, Pool, Classify! 🎯**
