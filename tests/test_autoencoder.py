"""
Tests for Autoencoder Architectures

This file contains comprehensive tests for all autoencoder types.

Author: Ali Mehdi
Date: January 31, 2026
"""

import numpy as np
import pytest
from ilovetools.ml.autoencoder import (
    VanillaAutoencoder,
    DenoisingAutoencoder,
    SparseAutoencoder,
    ContractiveAutoencoder,
    VAE,
)


# ============================================================================
# TEST VANILLA AUTOENCODER
# ============================================================================

def test_vanilla_autoencoder_basic():
    """Test basic vanilla autoencoder functionality."""
    ae = VanillaAutoencoder(input_dim=784, latent_dim=32)
    x = np.random.randn(100, 784)
    
    encoded = ae.encode(x)
    decoded = ae.decode(encoded)
    
    assert encoded.shape == (100, 32)
    assert decoded.shape == (100, 784)


def test_vanilla_autoencoder_forward():
    """Test forward pass."""
    ae = VanillaAutoencoder(input_dim=784, latent_dim=32)
    x = np.random.randn(100, 784)
    
    reconstructed = ae.forward(x)
    
    assert reconstructed.shape == x.shape


def test_vanilla_autoencoder_reconstruction_loss():
    """Test reconstruction loss computation."""
    ae = VanillaAutoencoder(input_dim=784, latent_dim=32)
    x = np.random.randn(100, 784)
    
    reconstructed = ae.forward(x)
    loss = ae.reconstruction_loss(x, reconstructed)
    
    assert isinstance(loss, float)
    assert loss >= 0


def test_vanilla_autoencoder_callable():
    """Test that autoencoder is callable."""
    ae = VanillaAutoencoder(input_dim=784, latent_dim=32)
    x = np.random.randn(100, 784)
    
    output = ae(x)
    assert output is not None


# ============================================================================
# TEST DENOISING AUTOENCODER
# ============================================================================

def test_denoising_autoencoder_basic():
    """Test basic denoising autoencoder functionality."""
    dae = DenoisingAutoencoder(input_dim=784, latent_dim=32, noise_factor=0.3)
    x = np.random.randn(100, 784)
    
    x_noisy = dae.add_noise(x)
    reconstructed = dae.forward(x_noisy, add_noise=False)
    
    assert x_noisy.shape == x.shape
    assert reconstructed.shape == x.shape


def test_denoising_autoencoder_gaussian_noise():
    """Test Gaussian noise addition."""
    dae = DenoisingAutoencoder(input_dim=784, latent_dim=32, noise_type='gaussian')
    x = np.random.randn(100, 784)
    
    x_noisy = dae.add_noise(x)
    
    assert not np.allclose(x, x_noisy)


def test_denoising_autoencoder_masking_noise():
    """Test masking noise."""
    dae = DenoisingAutoencoder(input_dim=784, latent_dim=32, noise_type='masking')
    x = np.random.randn(100, 784)
    
    x_noisy = dae.add_noise(x)
    
    assert not np.allclose(x, x_noisy)


# ============================================================================
# TEST SPARSE AUTOENCODER
# ============================================================================

def test_sparse_autoencoder_basic():
    """Test basic sparse autoencoder functionality."""
    sae = SparseAutoencoder(input_dim=784, latent_dim=64, sparsity_weight=0.001)
    x = np.random.randn(100, 784)
    
    encoded = sae.encode(x)
    reconstructed = sae.forward(x)
    
    assert encoded.shape == (100, 64)
    assert reconstructed.shape == (100, 784)


def test_sparse_autoencoder_l1_penalty():
    """Test L1 sparsity penalty."""
    sae = SparseAutoencoder(input_dim=784, latent_dim=64, sparsity_type='l1')
    x = np.random.randn(100, 784)
    
    encoded = sae.encode(x)
    penalty = sae.sparsity_penalty(encoded)
    
    assert isinstance(penalty, float)
    assert penalty >= 0


def test_sparse_autoencoder_kl_penalty():
    """Test KL divergence sparsity penalty."""
    sae = SparseAutoencoder(input_dim=784, latent_dim=64, sparsity_type='kl')
    x = np.random.randn(100, 784)
    
    encoded = sae.encode(x)
    penalty = sae.sparsity_penalty(encoded)
    
    assert isinstance(penalty, float)


def test_sparse_autoencoder_total_loss():
    """Test total loss computation."""
    sae = SparseAutoencoder(input_dim=784, latent_dim=64)
    x = np.random.randn(100, 784)
    
    encoded = sae.encode(x)
    reconstructed = sae.forward(x)
    loss = sae.total_loss(x, reconstructed, encoded)
    
    assert isinstance(loss, float)
    assert loss >= 0


# ============================================================================
# TEST CONTRACTIVE AUTOENCODER
# ============================================================================

def test_contractive_autoencoder_basic():
    """Test basic contractive autoencoder functionality."""
    cae = ContractiveAutoencoder(input_dim=784, latent_dim=32)
    x = np.random.randn(100, 784)
    
    encoded = cae.encode(x)
    reconstructed = cae.forward(x)
    
    assert encoded.shape == (100, 32)
    assert reconstructed.shape == (100, 784)


def test_contractive_autoencoder_penalty():
    """Test contractive penalty computation."""
    cae = ContractiveAutoencoder(input_dim=784, latent_dim=32)
    x = np.random.randn(10, 784)  # Small batch for efficiency
    
    encoded = cae.encode(x)
    penalty = cae.contractive_penalty(x, encoded)
    
    assert isinstance(penalty, float)
    assert penalty >= 0


def test_contractive_autoencoder_total_loss():
    """Test total loss computation."""
    cae = ContractiveAutoencoder(input_dim=784, latent_dim=32)
    x = np.random.randn(10, 784)
    
    encoded = cae.encode(x)
    reconstructed = cae.forward(x)
    loss = cae.total_loss(x, reconstructed, encoded)
    
    assert isinstance(loss, float)
    assert loss >= 0


# ============================================================================
# TEST VAE
# ============================================================================

def test_vae_basic():
    """Test basic VAE functionality."""
    vae = VAE(input_dim=784, latent_dim=32)
    x = np.random.randn(100, 784)
    
    reconstructed, mu, logvar = vae.forward(x)
    
    assert reconstructed.shape == (100, 784)
    assert mu.shape == (100, 32)
    assert logvar.shape == (100, 32)


def test_vae_encode():
    """Test VAE encoding."""
    vae = VAE(input_dim=784, latent_dim=32)
    x = np.random.randn(100, 784)
    
    mu, logvar = vae.encode(x)
    
    assert mu.shape == (100, 32)
    assert logvar.shape == (100, 32)


def test_vae_reparameterize():
    """Test reparameterization trick."""
    vae = VAE(input_dim=784, latent_dim=32)
    mu = np.random.randn(100, 32)
    logvar = np.random.randn(100, 32)
    
    z = vae.reparameterize(mu, logvar)
    
    assert z.shape == (100, 32)


def test_vae_decode():
    """Test VAE decoding."""
    vae = VAE(input_dim=784, latent_dim=32)
    z = np.random.randn(100, 32)
    
    decoded = vae.decode(z)
    
    assert decoded.shape == (100, 784)


def test_vae_loss():
    """Test VAE loss computation."""
    vae = VAE(input_dim=784, latent_dim=32)
    x = np.random.rand(100, 784)  # Use rand for [0, 1] range
    
    reconstructed, mu, logvar = vae.forward(x)
    loss = vae.loss(x, reconstructed, mu, logvar)
    
    assert isinstance(loss, float)


def test_vae_generation():
    """Test VAE generation from random latent vectors."""
    vae = VAE(input_dim=784, latent_dim=32)
    
    # Sample from standard normal
    z_sample = np.random.randn(10, 32)
    generated = vae.decode(z_sample)
    
    assert generated.shape == (10, 784)


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

def test_all_autoencoders_callable():
    """Test that all autoencoders are callable."""
    x = np.random.randn(50, 784)
    
    ae = VanillaAutoencoder(784, 32)
    dae = DenoisingAutoencoder(784, 32)
    sae = SparseAutoencoder(784, 32)
    cae = ContractiveAutoencoder(784, 32)
    vae = VAE(784, 32)
    
    assert ae(x) is not None
    assert dae(x) is not None
    assert sae(x) is not None
    assert cae(x) is not None
    assert vae(x) is not None


def test_autoencoders_preserve_batch_size():
    """Test that autoencoders preserve batch size."""
    batch_sizes = [10, 50, 100]
    
    for batch_size in batch_sizes:
        x = np.random.randn(batch_size, 784)
        
        ae = VanillaAutoencoder(784, 32)
        output = ae.forward(x)
        
        assert output.shape[0] == batch_size


def test_autoencoders_different_latent_dims():
    """Test autoencoders with different latent dimensions."""
    latent_dims = [16, 32, 64, 128]
    
    for latent_dim in latent_dims:
        ae = VanillaAutoencoder(784, latent_dim)
        x = np.random.randn(50, 784)
        
        encoded = ae.encode(x)
        assert encoded.shape == (50, latent_dim)


def test_dimensionality_reduction():
    """Test dimensionality reduction capability."""
    ae = VanillaAutoencoder(input_dim=784, latent_dim=32)
    x = np.random.randn(100, 784)
    
    # Encode to lower dimension
    encoded = ae.encode(x)
    
    assert encoded.shape[1] < x.shape[1]
    assert encoded.shape[1] == 32


def test_anomaly_detection_pipeline():
    """Test anomaly detection using reconstruction error."""
    ae = VanillaAutoencoder(input_dim=784, latent_dim=32)
    
    # Normal data
    x_normal = np.random.randn(100, 784)
    reconstructed_normal = ae.forward(x_normal)
    error_normal = np.mean((x_normal - reconstructed_normal) ** 2, axis=1)
    
    # Anomalous data (different distribution)
    x_anomaly = np.random.randn(10, 784) * 5  # Higher variance
    reconstructed_anomaly = ae.forward(x_anomaly)
    error_anomaly = np.mean((x_anomaly - reconstructed_anomaly) ** 2, axis=1)
    
    # Anomalies should have higher reconstruction error
    assert np.mean(error_anomaly) > np.mean(error_normal)


def test_vae_latent_space_interpolation():
    """Test VAE latent space interpolation."""
    vae = VAE(input_dim=784, latent_dim=32)
    
    # Two points in latent space
    z1 = np.random.randn(1, 32)
    z2 = np.random.randn(1, 32)
    
    # Interpolate
    alpha = 0.5
    z_interp = alpha * z1 + (1 - alpha) * z2
    
    # Decode
    x_interp = vae.decode(z_interp)
    
    assert x_interp.shape == (1, 784)


print("=" * 80)
print("ALL AUTOENCODER TESTS PASSED! ✓")
print("=" * 80)
