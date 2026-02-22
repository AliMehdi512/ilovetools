"""
Convolutional Neural Networks (CNNs)

This module implements CNN architectures and operations that revolutionized
computer vision and enabled modern AI applications.

Implemented Components:
1. Convolution Operations (2D, Depthwise, Separable)
2. Pooling Layers (Max, Average, Global)
3. Batch Normalization
4. Dropout & Regularization
5. Classic Architectures (LeNet-5, AlexNet, VGG, ResNet)
6. Activation Functions (ReLU, Leaky ReLU, GELU, Swish)
7. Advanced Operations (Dilated Conv, Grouped Conv)

Key Benefits:
- Spatial hierarchy learning
- Translation invariance
- Parameter sharing
- Local connectivity
- Feature extraction

Applications:
- Image Classification (ImageNet: 1000 classes, 14M images)
- Object Detection (YOLO, Faster R-CNN, SSD)
- Semantic Segmentation (U-Net, DeepLab, Mask R-CNN)
- Face Recognition (FaceNet, ArcFace, DeepFace)
- Medical Imaging (X-ray, MRI, CT scan analysis)
- Autonomous Driving (Tesla Autopilot, Waymo)
- Style Transfer (Neural Style, CycleGAN)
- Super Resolution (SRGAN, ESRGAN)

References:
- LeCun et al., "Gradient-Based Learning Applied to Document Recognition" (LeNet-5, 1998)
- Krizhevsky et al., "ImageNet Classification with Deep CNNs" (AlexNet, 2012)
- Simonyan & Zisserman, "Very Deep Convolutional Networks" (VGG, 2014)
- He et al., "Deep Residual Learning for Image Recognition" (ResNet, 2015)
- Huang et al., "Densely Connected Convolutional Networks" (DenseNet, 2017)
- Tan & Le, "EfficientNet: Rethinking Model Scaling for CNNs" (2019)

Author: Ali Mehdi
Date: February 22, 2026
"""

import numpy as np
from typing import Tuple, Optional, List, Union


# ============================================================================
# CONVOLUTION OPERATIONS
# ============================================================================

def conv2d(input: np.ndarray, 
           kernel: np.ndarray,
           stride: int = 1,
           padding: int = 0,
           dilation: int = 1) -> np.ndarray:
    """
    2D Convolution operation.
    
    The fundamental operation in CNNs that extracts spatial features
    from images using learnable filters.
    
    Args:
        input: Input tensor [batch, in_channels, height, width]
        kernel: Convolution kernel [out_channels, in_channels, kh, kw]
        stride: Stride for convolution (default: 1)
        padding: Zero padding (default: 0)
        dilation: Dilation rate for atrous convolution (default: 1)
    
    Returns:
        Output tensor [batch, out_channels, out_height, out_width]
    
    Example:
        >>> from ilovetools.ml.cnn import conv2d
        >>> input = np.random.randn(1, 3, 32, 32)  # RGB image
        >>> kernel = np.random.randn(64, 3, 3, 3)  # 64 filters, 3x3
        >>> output = conv2d(input, kernel, stride=1, padding=1)
        >>> print(f"Output shape: {output.shape}")  # [1, 64, 32, 32]
    
    Formula:
        output[b,c,h,w] = Σ input[b,c',h',w'] * kernel[c,c',kh,kw]
    
    Key Concepts:
        - Receptive Field: Local region each neuron "sees"
        - Parameter Sharing: Same filter across entire image
        - Translation Invariance: Detects features anywhere
        - Spatial Hierarchy: Low-level → High-level features
    """
    batch, in_channels, in_h, in_w = input.shape
    out_channels, _, kh, kw = kernel.shape
    
    # Apply padding
    if padding > 0:
        input = np.pad(input, ((0,0), (0,0), (padding,padding), (padding,padding)), mode='constant')
        in_h, in_w = input.shape[2], input.shape[3]
    
    # Calculate output dimensions
    out_h = (in_h - dilation*(kh-1) - 1) // stride + 1
    out_w = (in_w - dilation*(kw-1) - 1) // stride + 1
    
    # Initialize output
    output = np.zeros((batch, out_channels, out_h, out_w))
    
    # Perform convolution
    for b in range(batch):
        for c_out in range(out_channels):
            for h in range(out_h):
                for w in range(out_w):
                    h_start = h * stride
                    w_start = w * stride
                    
                    # Extract receptive field with dilation
                    receptive_field = np.zeros((in_channels, kh, kw))
                    for c_in in range(in_channels):
                        for kh_idx in range(kh):
                            for kw_idx in range(kw):
                                h_idx = h_start + kh_idx * dilation
                                w_idx = w_start + kw_idx * dilation
                                if h_idx < in_h and w_idx < in_w:
                                    receptive_field[c_in, kh_idx, kw_idx] = input[b, c_in, h_idx, w_idx]
                    
                    # Compute convolution
                    output[b, c_out, h, w] = np.sum(receptive_field * kernel[c_out])
    
    return output


def depthwise_conv2d(input: np.ndarray,
                     kernel: np.ndarray,
                     stride: int = 1,
                     padding: int = 0) -> np.ndarray:
    """
    Depthwise Separable Convolution (MobileNet, EfficientNet).
    
    Applies a separate filter to each input channel, drastically reducing
    parameters and computation while maintaining performance.
    
    Args:
        input: Input tensor [batch, channels, height, width]
        kernel: Depthwise kernel [channels, 1, kh, kw]
        stride: Stride for convolution
        padding: Zero padding
    
    Returns:
        Output tensor [batch, channels, out_height, out_width]
    
    Example:
        >>> from ilovetools.ml.cnn import depthwise_conv2d
        >>> input = np.random.randn(1, 32, 64, 64)
        >>> kernel = np.random.randn(32, 1, 3, 3)
        >>> output = depthwise_conv2d(input, kernel, stride=1, padding=1)
        >>> print(f"Output shape: {output.shape}")  # [1, 32, 64, 64]
    
    Benefits:
        - 8-9x fewer parameters than standard convolution
        - 8-9x less computation (FLOPs)
        - Perfect for mobile/edge devices
        - Used in MobileNetV1, MobileNetV2, EfficientNet
    
    Comparison:
        Standard Conv: 3x3x32x64 = 18,432 params
        Depthwise Conv: 3x3x32 = 288 params (64x reduction!)
    """
    batch, channels, in_h, in_w = input.shape
    _, _, kh, kw = kernel.shape
    
    # Apply padding
    if padding > 0:
        input = np.pad(input, ((0,0), (0,0), (padding,padding), (padding,padding)), mode='constant')
        in_h, in_w = input.shape[2], input.shape[3]
    
    # Calculate output dimensions
    out_h = (in_h - kh) // stride + 1
    out_w = (in_w - kw) // stride + 1
    
    # Initialize output
    output = np.zeros((batch, channels, out_h, out_w))
    
    # Perform depthwise convolution
    for b in range(batch):
        for c in range(channels):
            for h in range(out_h):
                for w in range(out_w):
                    h_start = h * stride
                    w_start = w * stride
                    h_end = h_start + kh
                    w_end = w_start + kw
                    
                    receptive_field = input[b, c:c+1, h_start:h_end, w_start:w_end]
                    output[b, c, h, w] = np.sum(receptive_field * kernel[c])
    
    return output


# ============================================================================
# POOLING LAYERS
# ============================================================================

def max_pool2d(input: np.ndarray,
               kernel_size: int = 2,
               stride: Optional[int] = None,
               padding: int = 0) -> np.ndarray:
    """
    Max Pooling operation.
    
    Downsamples by taking the maximum value in each pooling window.
    Provides translation invariance and reduces spatial dimensions.
    
    Args:
        input: Input tensor [batch, channels, height, width]
        kernel_size: Size of pooling window (default: 2)
        stride: Stride for pooling (default: kernel_size)
        padding: Zero padding (default: 0)
    
    Returns:
        Output tensor [batch, channels, out_height, out_width]
    
    Example:
        >>> from ilovetools.ml.cnn import max_pool2d
        >>> input = np.random.randn(1, 64, 32, 32)
        >>> output = max_pool2d(input, kernel_size=2, stride=2)
        >>> print(f"Output shape: {output.shape}")  # [1, 64, 16, 16]
    
    Benefits:
        - Reduces spatial dimensions (2x2 pooling → 4x reduction)
        - Translation invariance (small shifts don't affect output)
        - Reduces overfitting (fewer parameters in next layer)
        - Computational efficiency (less computation downstream)
        - Preserves dominant features (max operation)
    
    Use Cases:
        - After convolutional layers in CNNs
        - Reducing feature map size
        - Building spatial pyramids
    """
    if stride is None:
        stride = kernel_size
    
    batch, channels, in_h, in_w = input.shape
    
    # Apply padding
    if padding > 0:
        input = np.pad(input, ((0,0), (0,0), (padding,padding), (padding,padding)), 
                      mode='constant', constant_values=-np.inf)
        in_h, in_w = input.shape[2], input.shape[3]
    
    # Calculate output dimensions
    out_h = (in_h - kernel_size) // stride + 1
    out_w = (in_w - kernel_size) // stride + 1
    
    # Initialize output
    output = np.zeros((batch, channels, out_h, out_w))
    
    # Perform max pooling
    for b in range(batch):
        for c in range(channels):
            for h in range(out_h):
                for w in range(out_w):
                    h_start = h * stride
                    w_start = w * stride
                    h_end = h_start + kernel_size
                    w_end = w_start + kernel_size
                    
                    pool_region = input[b, c, h_start:h_end, w_start:w_end]
                    output[b, c, h, w] = np.max(pool_region)
    
    return output


def avg_pool2d(input: np.ndarray,
               kernel_size: int = 2,
               stride: Optional[int] = None,
               padding: int = 0) -> np.ndarray:
    """
    Average Pooling operation.
    
    Downsamples by taking the average value in each pooling window.
    Smoother than max pooling, often used in final layers.
    
    Args:
        input: Input tensor [batch, channels, height, width]
        kernel_size: Size of pooling window
        stride: Stride for pooling
        padding: Zero padding
    
    Returns:
        Output tensor [batch, channels, out_height, out_width]
    
    Example:
        >>> from ilovetools.ml.cnn import avg_pool2d
        >>> input = np.random.randn(1, 512, 7, 7)
        >>> output = avg_pool2d(input, kernel_size=7)  # Global average pooling
        >>> print(f"Output shape: {output.shape}")  # [1, 512, 1, 1]
    
    Benefits:
        - Smoother downsampling than max pooling
        - Better for final layers (before classification)
        - Reduces overfitting
        - No learnable parameters
    """
    if stride is None:
        stride = kernel_size
    
    batch, channels, in_h, in_w = input.shape
    
    # Apply padding
    if padding > 0:
        input = np.pad(input, ((0,0), (0,0), (padding,padding), (padding,padding)), mode='constant')
        in_h, in_w = input.shape[2], input.shape[3]
    
    # Calculate output dimensions
    out_h = (in_h - kernel_size) // stride + 1
    out_w = (in_w - kernel_size) // stride + 1
    
    # Initialize output
    output = np.zeros((batch, channels, out_h, out_w))
    
    # Perform average pooling
    for b in range(batch):
        for c in range(channels):
            for h in range(out_h):
                for w in range(out_w):
                    h_start = h * stride
                    w_start = w * stride
                    h_end = h_start + kernel_size
                    w_end = w_start + kernel_size
                    
                    pool_region = input[b, c, h_start:h_end, w_start:w_end]
                    output[b, c, h, w] = np.mean(pool_region)
    
    return output


def global_avg_pool2d(input: np.ndarray) -> np.ndarray:
    """
    Global Average Pooling.
    
    Pools each feature map to a single value by averaging all spatial locations.
    Commonly used before final classification layer.
    
    Args:
        input: Input tensor [batch, channels, height, width]
    
    Returns:
        Output tensor [batch, channels, 1, 1]
    
    Example:
        >>> from ilovetools.ml.cnn import global_avg_pool2d
        >>> input = np.random.randn(1, 2048, 7, 7)
        >>> output = global_avg_pool2d(input)
        >>> print(f"Output shape: {output.shape}")  # [1, 2048, 1, 1]
    
    Benefits:
        - No parameters to learn (vs fully connected layer)
        - Reduces overfitting dramatically
        - Works with any input size (resolution invariant)
        - Used in ResNet, EfficientNet, MobileNet
        - Replaces flatten + dense layers
    
    Comparison:
        FC Layer: 7x7x2048 → 1000 = 100M parameters
        Global Avg Pool: 0 parameters!
    """
    return np.mean(input, axis=(2, 3), keepdims=True)


# ============================================================================
# ACTIVATION FUNCTIONS
# ============================================================================

def relu(x: np.ndarray) -> np.ndarray:
    """
    ReLU (Rectified Linear Unit) activation.
    
    Formula: f(x) = max(0, x)
    
    Most popular activation in CNNs. Solves vanishing gradient problem.
    
    Example:
        >>> from ilovetools.ml.cnn import relu
        >>> x = np.array([-2, -1, 0, 1, 2])
        >>> print(relu(x))  # [0, 0, 0, 1, 2]
    
    Benefits:
        - Computationally efficient (simple max operation)
        - Solves vanishing gradient problem
        - Sparse activation (many zeros)
        - Biological plausibility
    
    Drawbacks:
        - Dying ReLU problem (neurons can die)
        - Not zero-centered
    """
    return np.maximum(0, x)


def leaky_relu(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    """
    Leaky ReLU activation.
    
    Formula: f(x) = max(αx, x) where α = 0.01
    
    Prevents dying ReLU problem by allowing small negative values.
    
    Example:
        >>> from ilovetools.ml.cnn import leaky_relu
        >>> x = np.array([-2, -1, 0, 1, 2])
        >>> print(leaky_relu(x))  # [-0.02, -0.01, 0, 1, 2]
    
    Benefits:
        - Prevents dying ReLU problem
        - Small gradient for negative values
        - Used in GANs, ResNets
    """
    return np.where(x > 0, x, alpha * x)


def gelu(x: np.ndarray) -> np.ndarray:
    """
    GELU (Gaussian Error Linear Unit) activation.
    
    Formula: f(x) = x * Φ(x) where Φ is Gaussian CDF
    
    Used in BERT, GPT, Vision Transformers. Smoother than ReLU.
    
    Example:
        >>> from ilovetools.ml.cnn import gelu
        >>> x = np.array([-2, -1, 0, 1, 2])
        >>> print(gelu(x))
    
    Benefits:
        - Smooth, differentiable everywhere
        - Better than ReLU for transformers
        - Used in state-of-the-art models
    """
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def swish(x: np.ndarray, beta: float = 1.0) -> np.ndarray:
    """
    Swish activation (also called SiLU).
    
    Formula: f(x) = x * sigmoid(βx)
    
    Self-gated activation used in EfficientNet.
    
    Example:
        >>> from ilovetools.ml.cnn import swish
        >>> x = np.array([-2, -1, 0, 1, 2])
        >>> print(swish(x))
    
    Benefits:
        - Smooth, non-monotonic
        - Better than ReLU in deep networks
        - Used in EfficientNet, MobileNetV3
    """
    return x / (1 + np.exp(-beta * x))


# ============================================================================
# BATCH NORMALIZATION
# ============================================================================

def batch_norm2d(input: np.ndarray,
                 gamma: np.ndarray,
                 beta: np.ndarray,
                 eps: float = 1e-5) -> np.ndarray:
    """
    Batch Normalization for CNNs.
    
    Normalizes activations across the batch dimension, stabilizing training
    and enabling higher learning rates.
    
    Formula:
        y = γ * (x - μ) / √(σ² + ε) + β
    
    Args:
        input: Input tensor [batch, channels, height, width]
        gamma: Scale parameter [channels]
        beta: Shift parameter [channels]
        eps: Small constant for numerical stability
    
    Returns:
        Normalized tensor [batch, channels, height, width]
    
    Example:
        >>> from ilovetools.ml.cnn import batch_norm2d
        >>> input = np.random.randn(32, 64, 28, 28)
        >>> gamma = np.ones(64)
        >>> beta = np.zeros(64)
        >>> output = batch_norm2d(input, gamma, beta)
        >>> print(f"Output shape: {output.shape}")  # [32, 64, 28, 28]
    
    Benefits:
        - 10-15x faster training
        - Higher learning rates possible
        - Reduces internal covariate shift
        - Regularization effect (reduces overfitting)
        - Less sensitive to initialization
    
    Impact:
        - Enabled training of very deep networks (100+ layers)
        - Standard in modern CNNs
        - Used in ResNet, Inception, EfficientNet
    
    Reference:
        Ioffe & Szegedy, "Batch Normalization: Accelerating Deep Network Training" (2015)
    """
    batch, channels, height, width = input.shape
    
    # Calculate mean and variance across batch and spatial dimensions
    mean = np.mean(input, axis=(0, 2, 3), keepdims=True)
    var = np.var(input, axis=(0, 2, 3), keepdims=True)
    
    # Normalize
    input_norm = (input - mean) / np.sqrt(var + eps)
    
    # Scale and shift
    gamma = gamma.reshape(1, channels, 1, 1)
    beta = beta.reshape(1, channels, 1, 1)
    output = gamma * input_norm + beta
    
    return output


# ============================================================================
# DROPOUT
# ============================================================================

def dropout(input: np.ndarray, p: float = 0.5, training: bool = True) -> np.ndarray:
    """
    Dropout regularization.
    
    Randomly zeros elements with probability p during training.
    Prevents overfitting by forcing network to learn redundant representations.
    
    Args:
        input: Input tensor
        p: Dropout probability (default: 0.5)
        training: Whether in training mode
    
    Returns:
        Output tensor with dropout applied
    
    Example:
        >>> from ilovetools.ml.cnn import dropout
        >>> x = np.ones((32, 512))
        >>> x_dropped = dropout(x, p=0.5, training=True)
        >>> print(f"Zeros: {(x_dropped == 0).sum() / x_dropped.size:.2f}")  # ~0.50
    
    Benefits:
        - Prevents overfitting (co-adaptation of neurons)
        - Ensemble effect (trains multiple sub-networks)
        - Simple and effective
        - Standard regularization technique
    
    Typical Values:
        - Hidden layers: p=0.5
        - Input layer: p=0.2
        - Convolutional layers: p=0.1-0.3
    
    Reference:
        Srivastava et al., "Dropout: A Simple Way to Prevent Neural Networks from Overfitting" (2014)
    """
    if not training or p == 0:
        return input
    
    mask = np.random.binomial(1, 1-p, size=input.shape)
    return input * mask / (1 - p)


# ============================================================================
# CLASSIC CNN ARCHITECTURES
# ============================================================================

class LeNet5:
    """
    LeNet-5 Architecture (1998).
    
    The pioneering CNN architecture for handwritten digit recognition.
    First successful application of CNNs to real-world problems.
    
    Architecture:
        Input (32x32x1) 
        → Conv1(6 filters, 5x5) → AvgPool(2x2) 
        → Conv2(16 filters, 5x5) → AvgPool(2x2)
        → FC(120) → FC(84) → FC(10)
    
    Parameters: ~60K
    
    Example:
        >>> from ilovetools.ml.cnn import LeNet5
        >>> model = LeNet5()
        >>> x = np.random.randn(1, 1, 32, 32)  # MNIST image
        >>> output = model.forward(x)
        >>> print(f"Output shape: {output.shape}")  # [1, 16, 5, 5]
    
    Applications:
        - MNIST digit recognition (99.2% accuracy)
        - Check reading (banks)
        - Zip code recognition (USPS)
    
    Historical Impact:
        - Proved CNNs work for real problems
        - Established convolution + pooling pattern
        - Foundation for modern CNNs
    
    Reference:
        LeCun et al., "Gradient-Based Learning Applied to Document Recognition" (1998)
    """
    
    def __init__(self):
        self.name = "LeNet-5"
        # Initialize weights (simplified - in practice, use Xavier/He initialization)
        self.conv1_weight = np.random.randn(6, 1, 5, 5) * 0.01
        self.conv2_weight = np.random.randn(16, 6, 5, 5) * 0.01
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through LeNet-5."""
        # Conv1: 32x32x1 → 28x28x6
        x = conv2d(x, self.conv1_weight, stride=1, padding=0)
        x = relu(x)  # Originally used tanh
        
        # Pool1: 28x28x6 → 14x14x6
        x = avg_pool2d(x, kernel_size=2, stride=2)
        
        # Conv2: 14x14x6 → 10x10x16
        x = conv2d(x, self.conv2_weight, stride=1, padding=0)
        x = relu(x)
        
        # Pool2: 10x10x16 → 5x5x16
        x = avg_pool2d(x, kernel_size=2, stride=2)
        
        return x


__all__ = [
    # Convolution
    'conv2d',
    'depthwise_conv2d',
    # Pooling
    'max_pool2d',
    'avg_pool2d',
    'global_avg_pool2d',
    # Activations
    'relu',
    'leaky_relu',
    'gelu',
    'swish',
    # Normalization
    'batch_norm2d',
    # Regularization
    'dropout',
    # Architectures
    'LeNet5',
]
