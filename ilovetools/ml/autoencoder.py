"""
Autoencoder Architectures Suite

This module implements various autoencoder architectures for unsupervised learning.
Autoencoders learn to compress data into a latent space and reconstruct it, enabling
dimensionality reduction, anomaly detection, and feature learning.

Implemented Autoencoder Types:
1. VanillaAutoencoder - Basic autoencoder with encoder-decoder
2. DenoisingAutoencoder - Learns to denoise corrupted inputs
3. SparseAutoencoder - Enforces sparsity in latent representations
4. ContractiveAutoencoder - Robust to small input perturbations
5. VAE - Variational Autoencoder (probabilistic, generative)

Key Benefits:
- Unsupervised feature learning
- Dimensionality reduction (alternative to PCA)
- Anomaly detection (reconstruction error)
- Image compression and denoising
- Generative modeling (VAE)

References:
- Autoencoder: Hinton & Salakhutdinov, "Reducing the Dimensionality of Data with Neural Networks" (2006)
- Denoising: Vincent et al., "Extracting and Composing Robust Features with Denoising Autoencoders" (2008)
- Sparse: Ng, "Sparse Autoencoder" (2011)
- Contractive: Rifai et al., "Contractive Auto-Encoders" (2011)
- VAE: Kingma & Welling, "Auto-Encoding Variational Bayes" (2014)

Author: Ali Mehdi
Date: January 31, 2026
"""

import numpy as np
from typing import Optional, Tuple


# ============================================================================
# VANILLA AUTOENCODER
# ============================================================================

class VanillaAutoencoder:
    """
    Vanilla Autoencoder.
    
    Basic autoencoder with encoder-decoder architecture. Learns to compress
    input into latent space and reconstruct it.
    
    Architecture:
        Encoder: input → hidden → latent
        Decoder: latent → hidden → output
    
    Loss:
        L = MSE(x, x̂) = ||x - x̂||²
    
    Args:
        input_dim: Input dimension
        latent_dim: Latent space dimension (bottleneck)
        hidden_dims: List of hidden layer dimensions (default: [512, 256])
        activation: Activation function ('relu', 'sigmoid', 'tanh') (default: 'relu')
    
    Example:
        >>> ae = VanillaAutoencoder(input_dim=784, latent_dim=32)
        >>> x = np.random.randn(100, 784)  # 100 samples
        >>> encoded = ae.encode(x)
        >>> decoded = ae.decode(encoded)
        >>> print(encoded.shape)  # (100, 32)
        >>> print(decoded.shape)  # (100, 784)
    
    Use Case:
        Dimensionality reduction, feature learning, image compression
    
    Reference:
        Hinton & Salakhutdinov, "Reducing the Dimensionality of Data" (2006)
    """
    
    def __init__(self, input_dim: int, latent_dim: int,
                 hidden_dims: Optional[list] = None,
                 activation: str = 'relu'):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims or [512, 256]
        self.activation = activation
        
        # Initialize encoder weights
        self.encoder_weights = []
        prev_dim = input_dim
        
        for hidden_dim in self.hidden_dims:
            w = np.random.randn(prev_dim, hidden_dim) * np.sqrt(2.0 / prev_dim)
            b = np.zeros(hidden_dim)
            self.encoder_weights.append((w, b))
            prev_dim = hidden_dim
        
        # Encoder to latent
        w = np.random.randn(prev_dim, latent_dim) * np.sqrt(2.0 / prev_dim)
        b = np.zeros(latent_dim)
        self.encoder_weights.append((w, b))
        
        # Initialize decoder weights (symmetric)
        self.decoder_weights = []
        prev_dim = latent_dim
        
        for hidden_dim in reversed(self.hidden_dims):
            w = np.random.randn(prev_dim, hidden_dim) * np.sqrt(2.0 / prev_dim)
            b = np.zeros(hidden_dim)
            self.decoder_weights.append((w, b))
            prev_dim = hidden_dim
        
        # Decoder to output
        w = np.random.randn(prev_dim, input_dim) * np.sqrt(2.0 / prev_dim)
        b = np.zeros(input_dim)
        self.decoder_weights.append((w, b))
    
    def activate(self, x: np.ndarray) -> np.ndarray:
        """Apply activation function."""
        if self.activation == 'relu':
            return np.maximum(0, x)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        elif self.activation == 'tanh':
            return np.tanh(x)
        else:
            return x
    
    def encode(self, x: np.ndarray) -> np.ndarray:
        """
        Encode input to latent space.
        
        Args:
            x: Input data (batch_size, input_dim)
        
        Returns:
            Latent representation (batch_size, latent_dim)
        """
        h = x
        
        for i, (w, b) in enumerate(self.encoder_weights):
            h = h @ w + b
            
            # Apply activation (except last layer)
            if i < len(self.encoder_weights) - 1:
                h = self.activate(h)
        
        return h
    
    def decode(self, z: np.ndarray) -> np.ndarray:
        """
        Decode latent representation to output.
        
        Args:
            z: Latent representation (batch_size, latent_dim)
        
        Returns:
            Reconstructed output (batch_size, input_dim)
        """
        h = z
        
        for i, (w, b) in enumerate(self.decoder_weights):
            h = h @ w + b
            
            # Apply activation (except last layer)
            if i < len(self.decoder_weights) - 1:
                h = self.activate(h)
        
        # Sigmoid activation for output (0-1 range)
        h = 1 / (1 + np.exp(-np.clip(h, -500, 500)))
        
        return h
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass (encode + decode).
        
        Args:
            x: Input data (batch_size, input_dim)
        
        Returns:
            Reconstructed output (batch_size, input_dim)
        """
        z = self.encode(x)
        x_reconstructed = self.decode(z)
        return x_reconstructed
    
    def reconstruction_loss(self, x: np.ndarray, x_reconstructed: np.ndarray) -> float:
        """
        Compute reconstruction loss (MSE).
        
        Args:
            x: Original input
            x_reconstructed: Reconstructed output
        
        Returns:
            Mean squared error
        """
        return np.mean((x - x_reconstructed) ** 2)
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


# ============================================================================
# DENOISING AUTOENCODER
# ============================================================================

class DenoisingAutoencoder(VanillaAutoencoder):
    """
    Denoising Autoencoder.
    
    Learns to reconstruct clean data from corrupted inputs. Adds noise during
    training to make the model robust.
    
    Formula:
        x̃ = corrupt(x)
        L = MSE(x, decode(encode(x̃)))
    
    Args:
        input_dim: Input dimension
        latent_dim: Latent space dimension
        hidden_dims: List of hidden layer dimensions
        noise_factor: Noise level (0.0 to 1.0) (default: 0.3)
        noise_type: Type of noise ('gaussian', 'masking') (default: 'gaussian')
    
    Example:
        >>> dae = DenoisingAutoencoder(input_dim=784, latent_dim=32, noise_factor=0.3)
        >>> x = np.random.randn(100, 784)
        >>> x_noisy = dae.add_noise(x)
        >>> x_reconstructed = dae.forward(x_noisy)
        >>> print(x_reconstructed.shape)  # (100, 784)
    
    Use Case:
        Image denoising, robust feature learning, data cleaning
    
    Reference:
        Vincent et al., "Extracting and Composing Robust Features" (2008)
    """
    
    def __init__(self, input_dim: int, latent_dim: int,
                 hidden_dims: Optional[list] = None,
                 noise_factor: float = 0.3,
                 noise_type: str = 'gaussian'):
        super().__init__(input_dim, latent_dim, hidden_dims)
        self.noise_factor = noise_factor
        self.noise_type = noise_type
    
    def add_noise(self, x: np.ndarray) -> np.ndarray:
        """
        Add noise to input.
        
        Args:
            x: Clean input
        
        Returns:
            Noisy input
        """
        if self.noise_type == 'gaussian':
            # Gaussian noise
            noise = np.random.randn(*x.shape) * self.noise_factor
            x_noisy = x + noise
        elif self.noise_type == 'masking':
            # Random masking
            mask = np.random.binomial(1, 1 - self.noise_factor, size=x.shape)
            x_noisy = x * mask
        else:
            x_noisy = x
        
        return np.clip(x_noisy, 0, 1)
    
    def forward(self, x: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """
        Forward pass with optional noise.
        
        Args:
            x: Input data
            add_noise: Whether to add noise (training mode)
        
        Returns:
            Reconstructed clean output
        """
        if add_noise:
            x_noisy = self.add_noise(x)
        else:
            x_noisy = x
        
        return super().forward(x_noisy)


# ============================================================================
# SPARSE AUTOENCODER
# ============================================================================

class SparseAutoencoder(VanillaAutoencoder):
    """
    Sparse Autoencoder.
    
    Enforces sparsity in latent representations, encouraging only a few neurons
    to be active. Uses L1 regularization or KL divergence.
    
    Loss:
        L = MSE(x, x̂) + λ * Σ|z_i|  (L1 regularization)
        or
        L = MSE(x, x̂) + β * KL(ρ || ρ̂)  (KL divergence)
    
    Args:
        input_dim: Input dimension
        latent_dim: Latent space dimension
        hidden_dims: List of hidden layer dimensions
        sparsity_weight: Weight for sparsity penalty (default: 0.001)
        sparsity_type: Type of sparsity ('l1', 'kl') (default: 'l1')
        target_sparsity: Target average activation (for KL) (default: 0.05)
    
    Example:
        >>> sae = SparseAutoencoder(input_dim=784, latent_dim=64, sparsity_weight=0.001)
        >>> x = np.random.randn(100, 784)
        >>> encoded = sae.encode(x)
        >>> loss = sae.total_loss(x, sae.forward(x), encoded)
        >>> print(f"Total loss: {loss:.4f}")
    
    Use Case:
        Feature learning, interpretable representations, classification pretraining
    
    Reference:
        Ng, "Sparse Autoencoder" (2011)
    """
    
    def __init__(self, input_dim: int, latent_dim: int,
                 hidden_dims: Optional[list] = None,
                 sparsity_weight: float = 0.001,
                 sparsity_type: str = 'l1',
                 target_sparsity: float = 0.05):
        super().__init__(input_dim, latent_dim, hidden_dims)
        self.sparsity_weight = sparsity_weight
        self.sparsity_type = sparsity_type
        self.target_sparsity = target_sparsity
    
    def sparsity_penalty(self, z: np.ndarray) -> float:
        """
        Compute sparsity penalty.
        
        Args:
            z: Latent representation
        
        Returns:
            Sparsity penalty
        """
        if self.sparsity_type == 'l1':
            # L1 regularization
            return np.mean(np.abs(z))
        
        elif self.sparsity_type == 'kl':
            # KL divergence
            rho_hat = np.mean(z, axis=0)  # Average activation
            rho = self.target_sparsity
            
            # KL(rho || rho_hat)
            kl = rho * np.log(rho / (rho_hat + 1e-8)) + \
                 (1 - rho) * np.log((1 - rho) / (1 - rho_hat + 1e-8))
            
            return np.sum(kl)
        
        return 0.0
    
    def total_loss(self, x: np.ndarray, x_reconstructed: np.ndarray, 
                   z: np.ndarray) -> float:
        """
        Compute total loss (reconstruction + sparsity).
        
        Args:
            x: Original input
            x_reconstructed: Reconstructed output
            z: Latent representation
        
        Returns:
            Total loss
        """
        reconstruction = self.reconstruction_loss(x, x_reconstructed)
        sparsity = self.sparsity_penalty(z)
        
        return reconstruction + self.sparsity_weight * sparsity


# ============================================================================
# CONTRACTIVE AUTOENCODER
# ============================================================================

class ContractiveAutoencoder(VanillaAutoencoder):
    """
    Contractive Autoencoder.
    
    Learns representations robust to small input perturbations by penalizing
    the Frobenius norm of the Jacobian.
    
    Loss:
        L = MSE(x, x̂) + λ * ||J_f(x)||²_F
        
        where J_f(x) is the Jacobian of the encoder
    
    Args:
        input_dim: Input dimension
        latent_dim: Latent space dimension
        hidden_dims: List of hidden layer dimensions
        contractive_weight: Weight for contractive penalty (default: 0.1)
    
    Example:
        >>> cae = ContractiveAutoencoder(input_dim=784, latent_dim=32)
        >>> x = np.random.randn(100, 784)
        >>> encoded = cae.encode(x)
        >>> penalty = cae.contractive_penalty(x, encoded)
        >>> print(f"Contractive penalty: {penalty:.4f}")
    
    Use Case:
        Robust feature learning, invariant representations, classification
    
    Reference:
        Rifai et al., "Contractive Auto-Encoders" (2011)
    """
    
    def __init__(self, input_dim: int, latent_dim: int,
                 hidden_dims: Optional[list] = None,
                 contractive_weight: float = 0.1):
        super().__init__(input_dim, latent_dim, hidden_dims)
        self.contractive_weight = contractive_weight
    
    def contractive_penalty(self, x: np.ndarray, z: np.ndarray) -> float:
        """
        Compute contractive penalty (Frobenius norm of Jacobian).
        
        Args:
            x: Input
            z: Latent representation
        
        Returns:
            Contractive penalty
        """
        # Approximate Jacobian using finite differences
        epsilon = 1e-4
        batch_size = x.shape[0]
        
        penalty = 0.0
        
        for i in range(min(10, self.input_dim)):  # Sample dimensions for efficiency
            # Perturb input
            x_perturbed = x.copy()
            x_perturbed[:, i] += epsilon
            
            # Encode perturbed input
            z_perturbed = self.encode(x_perturbed)
            
            # Compute gradient
            grad = (z_perturbed - z) / epsilon
            
            # Add to penalty
            penalty += np.sum(grad ** 2)
        
        return penalty / batch_size
    
    def total_loss(self, x: np.ndarray, x_reconstructed: np.ndarray,
                   z: np.ndarray) -> float:
        """
        Compute total loss (reconstruction + contractive).
        
        Args:
            x: Original input
            x_reconstructed: Reconstructed output
            z: Latent representation
        
        Returns:
            Total loss
        """
        reconstruction = self.reconstruction_loss(x, x_reconstructed)
        contractive = self.contractive_penalty(x, z)
        
        return reconstruction + self.contractive_weight * contractive


# ============================================================================
# VARIATIONAL AUTOENCODER (VAE)
# ============================================================================

class VAE:
    """
    Variational Autoencoder.
    
    Probabilistic autoencoder that learns a distribution over latent space.
    Enables generative modeling by sampling from the latent distribution.
    
    Formula:
        Encoder: q(z|x) = N(μ(x), σ²(x))
        Decoder: p(x|z)
        Loss: L = -E[log p(x|z)] + KL(q(z|x) || p(z))
             = Reconstruction Loss + KL Divergence
    
    Reparameterization Trick:
        z = μ + σ * ε, where ε ~ N(0, 1)
    
    Args:
        input_dim: Input dimension
        latent_dim: Latent space dimension
        hidden_dims: List of hidden layer dimensions (default: [512, 256])
    
    Example:
        >>> vae = VAE(input_dim=784, latent_dim=32)
        >>> x = np.random.randn(100, 784)
        >>> x_reconstructed, mu, logvar = vae.forward(x)
        >>> loss = vae.loss(x, x_reconstructed, mu, logvar)
        >>> print(f"VAE loss: {loss:.4f}")
        >>> # Generate new samples
        >>> z_sample = np.random.randn(10, 32)
        >>> generated = vae.decode(z_sample)
    
    Use Case:
        Generative modeling, image generation, latent space interpolation
    
    Reference:
        Kingma & Welling, "Auto-Encoding Variational Bayes" (2014)
    """
    
    def __init__(self, input_dim: int, latent_dim: int,
                 hidden_dims: Optional[list] = None):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims or [512, 256]
        
        # Initialize encoder weights (to mu and logvar)
        self.encoder_weights = []
        prev_dim = input_dim
        
        for hidden_dim in self.hidden_dims:
            w = np.random.randn(prev_dim, hidden_dim) * np.sqrt(2.0 / prev_dim)
            b = np.zeros(hidden_dim)
            self.encoder_weights.append((w, b))
            prev_dim = hidden_dim
        
        # Encoder to mu
        self.w_mu = np.random.randn(prev_dim, latent_dim) * np.sqrt(2.0 / prev_dim)
        self.b_mu = np.zeros(latent_dim)
        
        # Encoder to logvar
        self.w_logvar = np.random.randn(prev_dim, latent_dim) * np.sqrt(2.0 / prev_dim)
        self.b_logvar = np.zeros(latent_dim)
        
        # Initialize decoder weights
        self.decoder_weights = []
        prev_dim = latent_dim
        
        for hidden_dim in reversed(self.hidden_dims):
            w = np.random.randn(prev_dim, hidden_dim) * np.sqrt(2.0 / prev_dim)
            b = np.zeros(hidden_dim)
            self.decoder_weights.append((w, b))
            prev_dim = hidden_dim
        
        # Decoder to output
        self.w_out = np.random.randn(prev_dim, input_dim) * np.sqrt(2.0 / prev_dim)
        self.b_out = np.zeros(input_dim)
    
    def encode(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encode input to latent distribution parameters.
        
        Args:
            x: Input data (batch_size, input_dim)
        
        Returns:
            Tuple of (mu, logvar)
        """
        h = x
        
        for w, b in self.encoder_weights:
            h = h @ w + b
            h = np.maximum(0, h)  # ReLU
        
        # Compute mu and logvar
        mu = h @ self.w_mu + self.b_mu
        logvar = h @ self.w_logvar + self.b_logvar
        
        return mu, logvar
    
    def reparameterize(self, mu: np.ndarray, logvar: np.ndarray) -> np.ndarray:
        """
        Reparameterization trick: z = μ + σ * ε
        
        Args:
            mu: Mean
            logvar: Log variance
        
        Returns:
            Sampled latent vector
        """
        std = np.exp(0.5 * logvar)
        epsilon = np.random.randn(*mu.shape)
        z = mu + std * epsilon
        return z
    
    def decode(self, z: np.ndarray) -> np.ndarray:
        """
        Decode latent representation to output.
        
        Args:
            z: Latent representation (batch_size, latent_dim)
        
        Returns:
            Reconstructed output (batch_size, input_dim)
        """
        h = z
        
        for w, b in self.decoder_weights:
            h = h @ w + b
            h = np.maximum(0, h)  # ReLU
        
        # Output layer with sigmoid
        h = h @ self.w_out + self.b_out
        h = 1 / (1 + np.exp(-np.clip(h, -500, 500)))
        
        return h
    
    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Forward pass.
        
        Args:
            x: Input data
        
        Returns:
            Tuple of (reconstructed, mu, logvar)
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_reconstructed = self.decode(z)
        
        return x_reconstructed, mu, logvar
    
    def loss(self, x: np.ndarray, x_reconstructed: np.ndarray,
             mu: np.ndarray, logvar: np.ndarray) -> float:
        """
        Compute VAE loss (ELBO).
        
        Loss = Reconstruction Loss + KL Divergence
        
        Args:
            x: Original input
            x_reconstructed: Reconstructed output
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
        
        Returns:
            Total VAE loss
        """
        # Reconstruction loss (binary cross-entropy)
        reconstruction = -np.sum(
            x * np.log(x_reconstructed + 1e-8) + 
            (1 - x) * np.log(1 - x_reconstructed + 1e-8)
        ) / x.shape[0]
        
        # KL divergence: KL(q(z|x) || N(0, 1))
        kl_divergence = -0.5 * np.sum(1 + logvar - mu**2 - np.exp(logvar)) / x.shape[0]
        
        return reconstruction + kl_divergence
    
    def __call__(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.forward(x)


__all__ = [
    'VanillaAutoencoder',
    'DenoisingAutoencoder',
    'SparseAutoencoder',
    'ContractiveAutoencoder',
    'VAE',
]
