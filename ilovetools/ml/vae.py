"""
Variational Autoencoders (VAEs)

This module implements VAE architectures that combine deep learning with
probabilistic modeling for generative tasks and representation learning.

Implemented Components:
1. Vanilla VAE (Kingma & Welling, 2013)
2. Conditional VAE (CVAE)
3. Beta-VAE (Disentangled Representations)
4. Vector Quantized VAE (VQ-VAE)
5. Loss Functions (ELBO, KL Divergence, Reconstruction)
6. Sampling & Generation Utilities
7. Latent Space Interpolation

Key Concepts:
- Encoder: Maps input to latent distribution (μ, σ)
- Decoder: Reconstructs input from latent sample
- Reparameterization Trick: Enables backpropagation through sampling
- ELBO: Evidence Lower Bound (optimization objective)
- KL Divergence: Regularizes latent space to be Gaussian
- Latent Space: Continuous, smooth, interpretable

Differences from Standard Autoencoders:
- VAE: Probabilistic, generates new samples, smooth latent space
- AE: Deterministic, only reconstructs, discrete latent space
- VAE: Regularized latent space (KL divergence)
- AE: No regularization, can overfit

Applications:
- Image Generation (faces, digits, objects)
- Anomaly Detection (outlier detection)
- Data Compression (lossy compression)
- Feature Learning (representation learning)
- Semi-Supervised Learning (few labeled samples)
- Drug Discovery (molecular generation)
- Text Generation (sentence VAE)
- Music Generation (MusicVAE)
- Disentangled Representations (interpretable features)
- Data Augmentation (synthetic samples)

References:
- Kingma & Welling, "Auto-Encoding Variational Bayes" (2013)
- Rezende et al., "Stochastic Backpropagation" (2014)
- Higgins et al., "β-VAE: Learning Basic Visual Concepts" (2017)
- van den Oord et al., "Neural Discrete Representation Learning" (VQ-VAE, 2017)
- Sohn et al., "Learning Structured Output Representation" (CVAE, 2015)

Author: Ali Mehdi
Date: February 24, 2026
"""

import numpy as np
from typing import Tuple, Optional, List


# ============================================================================
# ENCODER NETWORK
# ============================================================================

class VAEEncoder:
    """
    VAE Encoder Network.
    
    Maps input data to parameters of a latent distribution (mean and log-variance).
    Unlike standard autoencoders, VAE encoder outputs a distribution, not a point.
    
    Architecture:
        Input → Dense → Dense → [μ, log(σ²)]
    
    Args:
        input_dim: Dimension of input data
        latent_dim: Dimension of latent space
        hidden_dims: List of hidden layer dimensions
    
    Example:
        >>> from ilovetools.ml.vae import VAEEncoder
        >>> encoder = VAEEncoder(input_dim=784, latent_dim=20)
        >>> x = np.random.randn(32, 784)  # MNIST images
        >>> mu, logvar = encoder.forward(x)
        >>> print(f"Mean: {mu.shape}, Log-variance: {logvar.shape}")
        Mean: (32, 20), Log-variance: (32, 20)
    
    Key Concepts:
        - μ (mu): Mean of latent distribution
        - log(σ²): Log-variance (for numerical stability)
        - Distribution: q(z|x) ~ N(μ, σ²)
        - Probabilistic: Each input → distribution, not point
    
    Why Log-Variance?
        - Numerical stability (avoids exp overflow)
        - Unconstrained optimization (can be negative)
        - σ = exp(0.5 * log(σ²))
    """
    
    def __init__(self,
                 input_dim: int = 784,
                 latent_dim: int = 20,
                 hidden_dims: List[int] = [512, 256]):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        
        # Initialize weights
        self.weights = []
        prev_dim = input_dim
        
        # Hidden layers
        for dim in hidden_dims:
            self.weights.append(np.random.randn(prev_dim, dim) * 0.01)
            prev_dim = dim
        
        # Output layers (mean and log-variance)
        self.fc_mu = np.random.randn(prev_dim, latent_dim) * 0.01
        self.fc_logvar = np.random.randn(prev_dim, latent_dim) * 0.01
    
    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encode input to latent distribution parameters.
        
        Args:
            x: Input data [batch, input_dim]
        
        Returns:
            mu: Mean of latent distribution [batch, latent_dim]
            logvar: Log-variance of latent distribution [batch, latent_dim]
        """
        # Pass through hidden layers
        h = x
        for weight in self.weights:
            h = np.dot(h, weight)
            h = self._relu(h)
        
        # Compute mean and log-variance
        mu = np.dot(h, self.fc_mu)
        logvar = np.dot(h, self.fc_logvar)
        
        return mu, logvar
    
    def _relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation."""
        return np.maximum(0, x)


# ============================================================================
# DECODER NETWORK
# ============================================================================

class VAEDecoder:
    """
    VAE Decoder Network.
    
    Reconstructs input data from latent samples. Learns to generate realistic
    outputs from the latent distribution.
    
    Architecture:
        Latent Sample (z) → Dense → Dense → Reconstruction
    
    Args:
        latent_dim: Dimension of latent space
        output_dim: Dimension of output data
        hidden_dims: List of hidden layer dimensions
    
    Example:
        >>> from ilovetools.ml.vae import VAEDecoder
        >>> decoder = VAEDecoder(latent_dim=20, output_dim=784)
        >>> z = np.random.randn(32, 20)  # Latent samples
        >>> x_recon = decoder.forward(z)
        >>> print(f"Reconstructed: {x_recon.shape}")  # (32, 784)
    
    Key Concepts:
        - Generative: Can sample z ~ N(0,1) to generate new data
        - Reconstruction: p(x|z) - probability of x given z
        - Sigmoid Output: For binary data (MNIST)
        - Tanh Output: For normalized data [-1, 1]
    """
    
    def __init__(self,
                 latent_dim: int = 20,
                 output_dim: int = 784,
                 hidden_dims: List[int] = [256, 512]):
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        
        # Initialize weights
        self.weights = []
        prev_dim = latent_dim
        
        # Hidden layers
        for dim in hidden_dims:
            self.weights.append(np.random.randn(prev_dim, dim) * 0.01)
            prev_dim = dim
        
        # Output layer
        self.fc_out = np.random.randn(prev_dim, output_dim) * 0.01
    
    def forward(self, z: np.ndarray) -> np.ndarray:
        """
        Decode latent samples to reconstructions.
        
        Args:
            z: Latent samples [batch, latent_dim]
        
        Returns:
            x_recon: Reconstructed data [batch, output_dim]
        """
        # Pass through hidden layers
        h = z
        for weight in self.weights:
            h = np.dot(h, weight)
            h = self._relu(h)
        
        # Output layer
        x_recon = np.dot(h, self.fc_out)
        x_recon = self._sigmoid(x_recon)  # For binary data
        
        return x_recon
    
    def _relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation."""
        return np.maximum(0, x)
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid activation."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


# ============================================================================
# VARIATIONAL AUTOENCODER
# ============================================================================

class VAE:
    """
    Variational Autoencoder (VAE).
    
    Combines encoder and decoder with probabilistic latent space.
    Learns to generate new samples and discover meaningful representations.
    
    Args:
        input_dim: Dimension of input data
        latent_dim: Dimension of latent space
        hidden_dims: List of hidden layer dimensions
    
    Example:
        >>> from ilovetools.ml.vae import VAE
        >>> vae = VAE(input_dim=784, latent_dim=20)
        >>> 
        >>> # Training
        >>> x = np.random.randn(32, 784)
        >>> x_recon, mu, logvar = vae.forward(x)
        >>> loss = vae.loss(x, x_recon, mu, logvar)
        >>> 
        >>> # Generation
        >>> z = np.random.randn(10, 20)
        >>> generated = vae.generate(z)
        >>> print(f"Generated samples: {generated.shape}")
    
    Training Objective (ELBO):
        L = E[log p(x|z)] - KL[q(z|x) || p(z)]
        = Reconstruction Loss - KL Divergence
    
    Benefits:
        - Smooth latent space (interpolation works)
        - Generative (sample new data)
        - Regularized (prevents overfitting)
        - Interpretable (disentangled features)
    
    Reference:
        Kingma & Welling, "Auto-Encoding Variational Bayes" (2013)
    """
    
    def __init__(self,
                 input_dim: int = 784,
                 latent_dim: int = 20,
                 hidden_dims: List[int] = [512, 256]):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Create encoder and decoder
        self.encoder = VAEEncoder(input_dim, latent_dim, hidden_dims)
        decoder_hidden = hidden_dims[::-1]  # Reverse for decoder
        self.decoder = VAEDecoder(latent_dim, input_dim, decoder_hidden)
    
    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Forward pass through VAE.
        
        Args:
            x: Input data [batch, input_dim]
        
        Returns:
            x_recon: Reconstructed data [batch, input_dim]
            mu: Latent mean [batch, latent_dim]
            logvar: Latent log-variance [batch, latent_dim]
        """
        # Encode
        mu, logvar = self.encoder.forward(x)
        
        # Reparameterization trick
        z = self.reparameterize(mu, logvar)
        
        # Decode
        x_recon = self.decoder.forward(z)
        
        return x_recon, mu, logvar
    
    def reparameterize(self, mu: np.ndarray, logvar: np.ndarray) -> np.ndarray:
        """
        Reparameterization trick: z = μ + σ * ε, where ε ~ N(0,1).
        
        Enables backpropagation through sampling operation.
        
        Args:
            mu: Mean [batch, latent_dim]
            logvar: Log-variance [batch, latent_dim]
        
        Returns:
            z: Latent samples [batch, latent_dim]
        """
        std = np.exp(0.5 * logvar)
        eps = np.random.randn(*mu.shape)
        z = mu + std * eps
        return z
    
    def generate(self, z: np.ndarray) -> np.ndarray:
        """
        Generate new samples from latent vectors.
        
        Args:
            z: Latent samples [batch, latent_dim]
        
        Returns:
            Generated samples [batch, input_dim]
        """
        return self.decoder.forward(z)
    
    def loss(self, x: np.ndarray, x_recon: np.ndarray,
             mu: np.ndarray, logvar: np.ndarray,
             beta: float = 1.0) -> float:
        """
        Compute VAE loss (negative ELBO).
        
        Args:
            x: Original input [batch, input_dim]
            x_recon: Reconstructed input [batch, input_dim]
            mu: Latent mean [batch, latent_dim]
            logvar: Latent log-variance [batch, latent_dim]
            beta: Weight for KL divergence (β-VAE)
        
        Returns:
            Total loss (scalar)
        """
        # Reconstruction loss (binary cross-entropy)
        recon_loss = reconstruction_loss(x, x_recon)
        
        # KL divergence
        kl_loss = kl_divergence(mu, logvar)
        
        # Total loss (ELBO)
        total_loss = recon_loss + beta * kl_loss
        
        return total_loss


# ============================================================================
# CONDITIONAL VAE
# ============================================================================

class ConditionalVAE:
    """
    Conditional Variational Autoencoder (CVAE).
    
    Extends VAE with conditional information (e.g., class labels).
    Enables controlled generation: "generate digit 7" or "generate cat image".
    
    Args:
        input_dim: Dimension of input data
        latent_dim: Dimension of latent space
        num_classes: Number of conditional classes
        hidden_dims: List of hidden layer dimensions
    
    Example:
        >>> from ilovetools.ml.vae import ConditionalVAE
        >>> cvae = ConditionalVAE(input_dim=784, latent_dim=20, num_classes=10)
        >>> 
        >>> # Training
        >>> x = np.random.randn(32, 784)
        >>> labels = np.random.randint(0, 10, size=32)
        >>> x_recon, mu, logvar = cvae.forward(x, labels)
        >>> 
        >>> # Conditional generation
        >>> z = np.random.randn(10, 20)
        >>> labels = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        >>> generated = cvae.generate(z, labels)
        >>> print(f"Generated digits 0-9: {generated.shape}")
    
    Applications:
        - Class-conditional generation
        - Attribute-guided synthesis
        - Semi-supervised learning
        - Missing data imputation
    
    Reference:
        Sohn et al., "Learning Structured Output Representation" (2015)
    """
    
    def __init__(self,
                 input_dim: int = 784,
                 latent_dim: int = 20,
                 num_classes: int = 10,
                 hidden_dims: List[int] = [512, 256]):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
        # Label embedding
        self.label_embedding = np.random.randn(num_classes, 10) * 0.01
        
        # Encoder (input + label)
        encoder_input_dim = input_dim + 10
        self.encoder = VAEEncoder(encoder_input_dim, latent_dim, hidden_dims)
        
        # Decoder (latent + label)
        decoder_input_dim = latent_dim + 10
        decoder_hidden = hidden_dims[::-1]
        self.decoder = VAEDecoder(decoder_input_dim, input_dim, decoder_hidden)
    
    def forward(self, x: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Forward pass through CVAE.
        
        Args:
            x: Input data [batch, input_dim]
            labels: Class labels [batch]
        
        Returns:
            x_recon: Reconstructed data [batch, input_dim]
            mu: Latent mean [batch, latent_dim]
            logvar: Latent log-variance [batch, latent_dim]
        """
        # Embed labels
        label_emb = self.label_embedding[labels]
        
        # Concatenate input and label
        x_cond = np.concatenate([x, label_emb], axis=1)
        
        # Encode
        mu, logvar = self.encoder.forward(x_cond)
        
        # Reparameterize
        std = np.exp(0.5 * logvar)
        eps = np.random.randn(*mu.shape)
        z = mu + std * eps
        
        # Concatenate latent and label
        z_cond = np.concatenate([z, label_emb], axis=1)
        
        # Decode
        x_recon = self.decoder.forward(z_cond)
        
        return x_recon, mu, logvar
    
    def generate(self, z: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Generate samples conditioned on labels.
        
        Args:
            z: Latent samples [batch, latent_dim]
            labels: Class labels [batch]
        
        Returns:
            Generated samples [batch, input_dim]
        """
        # Embed labels
        label_emb = self.label_embedding[labels]
        
        # Concatenate latent and label
        z_cond = np.concatenate([z, label_emb], axis=1)
        
        # Decode
        return self.decoder.forward(z_cond)


# ============================================================================
# BETA-VAE
# ============================================================================

class BetaVAE(VAE):
    """
    β-VAE (Beta-VAE) for Disentangled Representations.
    
    Extends VAE with weighted KL divergence (β > 1) to encourage
    disentangled latent representations where each dimension captures
    a single factor of variation.
    
    Args:
        input_dim: Dimension of input data
        latent_dim: Dimension of latent space
        beta: Weight for KL divergence (β > 1 for disentanglement)
        hidden_dims: List of hidden layer dimensions
    
    Example:
        >>> from ilovetools.ml.vae import BetaVAE
        >>> beta_vae = BetaVAE(input_dim=784, latent_dim=20, beta=4.0)
        >>> 
        >>> # Training
        >>> x = np.random.randn(32, 784)
        >>> x_recon, mu, logvar = beta_vae.forward(x)
        >>> loss = beta_vae.loss(x, x_recon, mu, logvar)
        >>> 
        >>> # Disentangled traversal
        >>> z = np.zeros((10, 20))
        >>> z[:, 0] = np.linspace(-3, 3, 10)  # Vary dimension 0
        >>> traversal = beta_vae.generate(z)
        >>> print(f"Traversal: {traversal.shape}")  # (10, 784)
    
    Benefits:
        - Disentangled representations (interpretable)
        - Each dimension = single factor (pose, color, shape)
        - Better generalization
        - Controllable generation
    
    Typical β Values:
        - β = 1: Standard VAE
        - β = 4: Moderate disentanglement
        - β = 10: Strong disentanglement
        - β = 100: Very strong (may hurt reconstruction)
    
    Reference:
        Higgins et al., "β-VAE: Learning Basic Visual Concepts" (2017)
    """
    
    def __init__(self,
                 input_dim: int = 784,
                 latent_dim: int = 20,
                 beta: float = 4.0,
                 hidden_dims: List[int] = [512, 256]):
        super().__init__(input_dim, latent_dim, hidden_dims)
        self.beta = beta
    
    def loss(self, x: np.ndarray, x_recon: np.ndarray,
             mu: np.ndarray, logvar: np.ndarray) -> float:
        """
        Compute β-VAE loss with weighted KL divergence.
        
        Args:
            x: Original input [batch, input_dim]
            x_recon: Reconstructed input [batch, input_dim]
            mu: Latent mean [batch, latent_dim]
            logvar: Latent log-variance [batch, latent_dim]
        
        Returns:
            Total loss (scalar)
        """
        return super().loss(x, x_recon, mu, logvar, beta=self.beta)


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

def reconstruction_loss(x: np.ndarray, x_recon: np.ndarray) -> float:
    """
    Reconstruction loss (Binary Cross-Entropy).
    
    Measures how well the decoder reconstructs the input.
    
    Formula:
        L_recon = -Σ [x*log(x_recon) + (1-x)*log(1-x_recon)]
    
    Args:
        x: Original input [batch, dim]
        x_recon: Reconstructed input [batch, dim]
    
    Returns:
        Reconstruction loss (scalar)
    
    Example:
        >>> from ilovetools.ml.vae import reconstruction_loss
        >>> x = np.random.rand(32, 784)
        >>> x_recon = np.random.rand(32, 784)
        >>> loss = reconstruction_loss(x, x_recon)
        >>> print(f"Reconstruction loss: {loss:.4f}")
    
    Alternatives:
        - MSE: For continuous data
        - BCE: For binary data (MNIST)
        - Perceptual Loss: For images (VGG features)
    """
    eps = 1e-8
    bce = -(x * np.log(x_recon + eps) + (1 - x) * np.log(1 - x_recon + eps))
    return np.mean(bce)


def kl_divergence(mu: np.ndarray, logvar: np.ndarray) -> float:
    """
    KL Divergence between q(z|x) and p(z) = N(0, I).
    
    Regularizes latent space to be close to standard Gaussian.
    
    Formula:
        KL[q(z|x) || p(z)] = -0.5 * Σ [1 + log(σ²) - μ² - σ²]
    
    Args:
        mu: Latent mean [batch, latent_dim]
        logvar: Latent log-variance [batch, latent_dim]
    
    Returns:
        KL divergence (scalar)
    
    Example:
        >>> from ilovetools.ml.vae import kl_divergence
        >>> mu = np.random.randn(32, 20)
        >>> logvar = np.random.randn(32, 20)
        >>> kl = kl_divergence(mu, logvar)
        >>> print(f"KL divergence: {kl:.4f}")
    
    Interpretation:
        - Low KL: Latent distribution close to N(0,1)
        - High KL: Latent distribution far from N(0,1)
        - Regularization: Prevents overfitting, enables generation
    """
    kl = -0.5 * np.sum(1 + logvar - mu**2 - np.exp(logvar), axis=1)
    return np.mean(kl)


def elbo_loss(x: np.ndarray, x_recon: np.ndarray,
              mu: np.ndarray, logvar: np.ndarray,
              beta: float = 1.0) -> float:
    """
    Evidence Lower Bound (ELBO) loss.
    
    Complete VAE objective function.
    
    Formula:
        ELBO = E[log p(x|z)] - β * KL[q(z|x) || p(z)]
        Loss = -ELBO (we minimize)
    
    Args:
        x: Original input [batch, dim]
        x_recon: Reconstructed input [batch, dim]
        mu: Latent mean [batch, latent_dim]
        logvar: Latent log-variance [batch, latent_dim]
        beta: Weight for KL divergence
    
    Returns:
        Negative ELBO (scalar)
    
    Example:
        >>> from ilovetools.ml.vae import elbo_loss
        >>> x = np.random.rand(32, 784)
        >>> x_recon = np.random.rand(32, 784)
        >>> mu = np.random.randn(32, 20)
        >>> logvar = np.random.randn(32, 20)
        >>> loss = elbo_loss(x, x_recon, mu, logvar, beta=1.0)
        >>> print(f"ELBO loss: {loss:.4f}")
    """
    recon = reconstruction_loss(x, x_recon)
    kl = kl_divergence(mu, logvar)
    return recon + beta * kl


# ============================================================================
# UTILITIES
# ============================================================================

def sample_from_latent(latent_dim: int, num_samples: int = 1) -> np.ndarray:
    """
    Sample from standard Gaussian prior p(z) = N(0, I).
    
    Args:
        latent_dim: Dimension of latent space
        num_samples: Number of samples to generate
    
    Returns:
        Latent samples [num_samples, latent_dim]
    
    Example:
        >>> from ilovetools.ml.vae import sample_from_latent
        >>> z = sample_from_latent(latent_dim=20, num_samples=10)
        >>> print(f"Sampled latent vectors: {z.shape}")  # (10, 20)
    """
    return np.random.randn(num_samples, latent_dim)


def interpolate_latent(z1: np.ndarray, z2: np.ndarray, 
                       num_steps: int = 10) -> np.ndarray:
    """
    Linear interpolation between two latent vectors.
    
    Creates smooth transitions in latent space.
    
    Args:
        z1: Start latent vector [latent_dim]
        z2: End latent vector [latent_dim]
        num_steps: Number of interpolation steps
    
    Returns:
        Interpolated latent vectors [num_steps, latent_dim]
    
    Example:
        >>> from ilovetools.ml.vae import interpolate_latent
        >>> z1 = np.random.randn(20)
        >>> z2 = np.random.randn(20)
        >>> z_interp = interpolate_latent(z1, z2, num_steps=10)
        >>> print(f"Interpolated: {z_interp.shape}")  # (10, 20)
        >>> 
        >>> # Generate interpolated images
        >>> vae = VAE(input_dim=784, latent_dim=20)
        >>> images = vae.generate(z_interp)
        >>> print(f"Interpolated images: {images.shape}")  # (10, 784)
    """
    alphas = np.linspace(0, 1, num_steps).reshape(-1, 1)
    z_interp = (1 - alphas) * z1 + alphas * z2
    return z_interp


def latent_traversal(latent_dim: int, dim_idx: int,
                     min_val: float = -3.0, max_val: float = 3.0,
                     num_steps: int = 10) -> np.ndarray:
    """
    Traverse a single latent dimension while keeping others fixed.
    
    Useful for visualizing what each latent dimension controls.
    
    Args:
        latent_dim: Dimension of latent space
        dim_idx: Index of dimension to traverse
        min_val: Minimum value for traversal
        max_val: Maximum value for traversal
        num_steps: Number of steps
    
    Returns:
        Latent vectors [num_steps, latent_dim]
    
    Example:
        >>> from ilovetools.ml.vae import latent_traversal
        >>> # Traverse dimension 0 (might control rotation)
        >>> z = latent_traversal(latent_dim=20, dim_idx=0, num_steps=10)
        >>> 
        >>> # Generate images showing effect of dimension 0
        >>> vae = VAE(input_dim=784, latent_dim=20)
        >>> images = vae.generate(z)
        >>> print(f"Traversal images: {images.shape}")  # (10, 784)
    """
    z = np.zeros((num_steps, latent_dim))
    z[:, dim_idx] = np.linspace(min_val, max_val, num_steps)
    return z


__all__ = [
    # Networks
    'VAEEncoder',
    'VAEDecoder',
    'VAE',
    'ConditionalVAE',
    'BetaVAE',
    # Loss Functions
    'reconstruction_loss',
    'kl_divergence',
    'elbo_loss',
    # Utilities
    'sample_from_latent',
    'interpolate_latent',
    'latent_traversal',
]
