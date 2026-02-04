"""
Comprehensive Examples: Autoencoder Architectures

This file demonstrates all autoencoder types with practical examples and use cases.

Author: Ali Mehdi
Date: January 31, 2026
"""

import numpy as np
from ilovetools.ml.autoencoder import (
    VanillaAutoencoder,
    DenoisingAutoencoder,
    SparseAutoencoder,
    ContractiveAutoencoder,
    VAE,
)

print("=" * 80)
print("AUTOENCODER ARCHITECTURES - COMPREHENSIVE EXAMPLES")
print("=" * 80)
print()

# ============================================================================
# EXAMPLE 1: Vanilla Autoencoder - Dimensionality Reduction
# ============================================================================
print("EXAMPLE 1: Vanilla Autoencoder - Dimensionality Reduction (MNIST)")
print("-" * 80)

# Simulate MNIST data (28x28 = 784 pixels)
input_dim = 784
latent_dim = 32
num_samples = 1000

ae = VanillaAutoencoder(input_dim=input_dim, latent_dim=latent_dim, hidden_dims=[512, 256])

print("Dimensionality reduction:")
print(f"Input dimension: {input_dim} (28×28 images)")
print(f"Latent dimension: {latent_dim}")
print(f"Compression ratio: {input_dim / latent_dim:.1f}x")
print()

# Simulate images
images = np.random.rand(num_samples, input_dim)

print(f"Original images: {images.shape}")

# Encode to latent space
encoded = ae.encode(images)
print(f"Encoded (latent): {encoded.shape}")
print(f"Reduced from {input_dim} to {latent_dim} dimensions!")
print()

# Decode back
reconstructed = ae.decode(encoded)
print(f"Reconstructed: {reconstructed.shape}")

# Reconstruction error
loss = ae.reconstruction_loss(images, reconstructed)
print(f"Reconstruction loss: {loss:.6f}")

print("\n✓ Example 1 completed\n")

# ============================================================================
# EXAMPLE 2: Denoising Autoencoder - Image Denoising
# ============================================================================
print("EXAMPLE 2: Denoising Autoencoder - Image Denoising")
print("-" * 80)

dae = DenoisingAutoencoder(
    input_dim=784,
    latent_dim=64,
    noise_factor=0.3,
    noise_type='gaussian'
)

print("Image denoising:")
print(f"Noise factor: 0.3")
print(f"Noise type: Gaussian")
print()

# Clean images
clean_images = np.random.rand(100, 784)
print(f"Clean images: {clean_images.shape}")

# Add noise
noisy_images = dae.add_noise(clean_images)
print(f"Noisy images: {noisy_images.shape}")

# Denoise
denoised = dae.forward(noisy_images, add_noise=False)
print(f"Denoised images: {denoised.shape}")
print()

# Compare errors
noise_error = np.mean((clean_images - noisy_images) ** 2)
denoise_error = np.mean((clean_images - denoised) ** 2)

print(f"Noise error: {noise_error:.6f}")
print(f"Denoising error: {denoise_error:.6f}")
print(f"Improvement: {(noise_error - denoise_error) / noise_error * 100:.1f}%")

print("\n✓ Example 2 completed\n")

# ============================================================================
# EXAMPLE 3: Sparse Autoencoder - Feature Learning
# ============================================================================
print("EXAMPLE 3: Sparse Autoencoder - Feature Learning")
print("-" * 80)

sae = SparseAutoencoder(
    input_dim=784,
    latent_dim=128,
    sparsity_weight=0.001,
    sparsity_type='l1'
)

print("Sparse feature learning:")
print(f"Latent dimension: 128")
print(f"Sparsity type: L1 regularization")
print(f"Sparsity weight: 0.001")
print()

# Input data
x = np.random.rand(200, 784)

# Encode
encoded = sae.encode(x)
print(f"Encoded features: {encoded.shape}")

# Check sparsity
active_neurons = np.mean(np.abs(encoded) > 0.1, axis=0)
print(f"Average active neurons: {np.mean(active_neurons) * 100:.1f}%")
print()

# Reconstruct
reconstructed = sae.forward(x)

# Total loss
loss = sae.total_loss(x, reconstructed, encoded)
print(f"Total loss (reconstruction + sparsity): {loss:.6f}")

print("\n✓ Example 3 completed\n")

# ============================================================================
# EXAMPLE 4: Contractive Autoencoder - Robust Features
# ============================================================================
print("EXAMPLE 4: Contractive Autoencoder - Robust Features")
print("-" * 80)

cae = ContractiveAutoencoder(
    input_dim=784,
    latent_dim=64,
    contractive_weight=0.1
)

print("Robust feature learning:")
print(f"Latent dimension: 64")
print(f"Contractive weight: 0.1")
print()

# Input data
x = np.random.rand(50, 784)

# Encode
encoded = cae.encode(x)
print(f"Encoded features: {encoded.shape}")

# Perturb input slightly
x_perturbed = x + np.random.randn(*x.shape) * 0.01
encoded_perturbed = cae.encode(x_perturbed)

# Check robustness
perturbation_effect = np.mean(np.abs(encoded - encoded_perturbed))
print(f"Perturbation effect on latent: {perturbation_effect:.6f}")
print("(Lower is better - more robust)")
print()

# Contractive penalty
penalty = cae.contractive_penalty(x, encoded)
print(f"Contractive penalty: {penalty:.6f}")

print("\n✓ Example 4 completed\n")

# ============================================================================
# EXAMPLE 5: VAE - Generative Modeling
# ============================================================================
print("EXAMPLE 5: VAE - Generative Modeling")
print("-" * 80)

vae = VAE(input_dim=784, latent_dim=32, hidden_dims=[512, 256])

print("Variational Autoencoder:")
print(f"Input dimension: 784")
print(f"Latent dimension: 32")
print(f"Probabilistic latent space")
print()

# Training data
x_train = np.random.rand(500, 784)

# Forward pass
reconstructed, mu, logvar = vae.forward(x_train)

print(f"Reconstructed: {reconstructed.shape}")
print(f"Mean (μ): {mu.shape}")
print(f"Log variance (log σ²): {logvar.shape}")
print()

# VAE loss
loss = vae.loss(x_train, reconstructed, mu, logvar)
print(f"VAE loss (ELBO): {loss:.6f}")
print()

# Generate new samples
print("Generating new samples:")
z_sample = np.random.randn(10, 32)  # Sample from N(0, 1)
generated = vae.decode(z_sample)

print(f"Generated samples: {generated.shape}")
print("✓ 10 new images generated from random latent vectors!")

print("\n✓ Example 5 completed\n")

# ============================================================================
# EXAMPLE 6: Anomaly Detection
# ============================================================================
print("EXAMPLE 6: Anomaly Detection - Fraud Detection")
print("-" * 80)

ae = VanillaAutoencoder(input_dim=30, latent_dim=10)

print("Fraud detection using reconstruction error:")
print(f"Features: 30 (transaction features)")
print(f"Latent dimension: 10")
print()

# Normal transactions
normal_transactions = np.random.randn(1000, 30)
reconstructed_normal = ae.forward(normal_transactions)
errors_normal = np.mean((normal_transactions - reconstructed_normal) ** 2, axis=1)

# Fraudulent transactions (different distribution)
fraud_transactions = np.random.randn(50, 30) * 3  # Higher variance
reconstructed_fraud = ae.forward(fraud_transactions)
errors_fraud = np.mean((fraud_transactions - reconstructed_fraud) ** 2, axis=1)

print(f"Normal transactions: {len(normal_transactions)}")
print(f"Fraudulent transactions: {len(fraud_transactions)}")
print()

print(f"Average reconstruction error (normal): {np.mean(errors_normal):.6f}")
print(f"Average reconstruction error (fraud): {np.mean(errors_fraud):.6f}")
print()

# Set threshold
threshold = np.percentile(errors_normal, 95)
print(f"Anomaly threshold (95th percentile): {threshold:.6f}")

# Detect anomalies
detected_fraud = np.sum(errors_fraud > threshold)
print(f"Detected fraudulent transactions: {detected_fraud}/{len(fraud_transactions)}")
print(f"Detection rate: {detected_fraud / len(fraud_transactions) * 100:.1f}%")

print("\n✓ Example 6 completed\n")

# ============================================================================
# EXAMPLE 7: Image Compression
# ============================================================================
print("EXAMPLE 7: Image Compression")
print("-" * 80)

# High compression ratio
ae_compress = VanillaAutoencoder(input_dim=784, latent_dim=16)

print("Lossy image compression:")
print(f"Original size: 784 values")
print(f"Compressed size: 16 values")
print(f"Compression ratio: {784 / 16:.1f}x")
print()

# Images
images = np.random.rand(100, 784)

# Compress
compressed = ae_compress.encode(images)
print(f"Compressed: {compressed.shape}")
print(f"Storage reduction: {(1 - 16/784) * 100:.1f}%")
print()

# Decompress
decompressed = ae_compress.decode(compressed)

# Quality
quality = 1 - np.mean((images - decompressed) ** 2)
print(f"Reconstruction quality: {quality * 100:.1f}%")

print("\n✓ Example 7 completed\n")

# ============================================================================
# EXAMPLE 8: VAE Latent Space Interpolation
# ============================================================================
print("EXAMPLE 8: VAE Latent Space Interpolation")
print("-" * 80)

vae = VAE(input_dim=784, latent_dim=32)

print("Latent space interpolation:")
print()

# Two images
img1 = np.random.rand(1, 784)
img2 = np.random.rand(1, 784)

# Encode to latent space
mu1, logvar1 = vae.encode(img1)
mu2, logvar2 = vae.encode(img2)

print(f"Image 1 latent: {mu1.shape}")
print(f"Image 2 latent: {mu2.shape}")
print()

# Interpolate
num_steps = 5
print(f"Interpolating {num_steps} steps between images:")

for i, alpha in enumerate(np.linspace(0, 1, num_steps)):
    z_interp = alpha * mu1 + (1 - alpha) * mu2
    img_interp = vae.decode(z_interp)
    print(f"  Step {i+1}: α={alpha:.2f}, shape={img_interp.shape}")

print()
print("✓ Smooth interpolation in latent space!")

print("\n✓ Example 8 completed\n")

# ============================================================================
# EXAMPLE 9: Comparing Autoencoder Types
# ============================================================================
print("EXAMPLE 9: Comparing Autoencoder Types")
print("-" * 80)

x = np.random.rand(100, 784)

print(f"Input data: {x.shape}")
print()

# Vanilla
ae = VanillaAutoencoder(784, 32)
recon_ae = ae.forward(x)
loss_ae = ae.reconstruction_loss(x, recon_ae)

# Denoising
dae = DenoisingAutoencoder(784, 32)
recon_dae = dae.forward(x, add_noise=False)
loss_dae = dae.reconstruction_loss(x, recon_dae)

# Sparse
sae = SparseAutoencoder(784, 32)
encoded_sae = sae.encode(x)
recon_sae = sae.forward(x)
loss_sae = sae.total_loss(x, recon_sae, encoded_sae)

# VAE
vae = VAE(784, 32)
recon_vae, mu, logvar = vae.forward(x)
loss_vae = vae.loss(x, recon_vae, mu, logvar)

print("Reconstruction losses:")
print(f"Vanilla AE: {loss_ae:.6f}")
print(f"Denoising AE: {loss_dae:.6f}")
print(f"Sparse AE: {loss_sae:.6f}")
print(f"VAE: {loss_vae:.6f}")
print()

print("When to use:")
print("✓ Vanilla: General dimensionality reduction")
print("✓ Denoising: Noisy data, robust features")
print("✓ Sparse: Interpretable features, classification")
print("✓ VAE: Generative modeling, sampling")

print("\n✓ Example 9 completed\n")

# ============================================================================
# EXAMPLE 10: Pretraining for Classification
# ============================================================================
print("EXAMPLE 10: Unsupervised Pretraining for Classification")
print("-" * 80)

# Pretrain autoencoder
ae_pretrain = VanillaAutoencoder(input_dim=784, latent_dim=128)

print("Unsupervised pretraining:")
print(f"Input: 784 features")
print(f"Latent: 128 features")
print()

# Unlabeled data
x_unlabeled = np.random.rand(5000, 784)

print(f"Unlabeled data: {x_unlabeled.shape}")

# Pretrain (learn features)
encoded = ae_pretrain.encode(x_unlabeled)

print(f"Learned features: {encoded.shape}")
print()

print("Use learned encoder for classification:")
print("1. Freeze encoder weights")
print("2. Add classification layer on top")
print("3. Fine-tune on labeled data")
print()

print("Benefits:")
print("✓ Better initialization")
print("✓ Fewer labeled samples needed")
print("✓ Better generalization")

print("\n✓ Example 10 completed\n")

# ============================================================================
# SUMMARY
# ============================================================================
print("=" * 80)
print("ALL EXAMPLES COMPLETED SUCCESSFULLY! ✓")
print("=" * 80)
print()
print("Summary of what we covered:")
print("1. ✓ Vanilla Autoencoder - Dimensionality Reduction")
print("2. ✓ Denoising Autoencoder - Image Denoising")
print("3. ✓ Sparse Autoencoder - Feature Learning")
print("4. ✓ Contractive Autoencoder - Robust Features")
print("5. ✓ VAE - Generative Modeling")
print("6. ✓ Anomaly Detection - Fraud Detection")
print("7. ✓ Image Compression")
print("8. ✓ VAE Latent Space Interpolation")
print("9. ✓ Comparing Autoencoder Types")
print("10. ✓ Unsupervised Pretraining")
print()
print("You now have a complete understanding of autoencoder architectures!")
print()
print("Next steps:")
print("- Use Vanilla AE for dimensionality reduction")
print("- Use Denoising AE for noisy data")
print("- Use Sparse AE for interpretable features")
print("- Use Contractive AE for robust representations")
print("- Use VAE for generative modeling")
print("- Apply to anomaly detection, compression, pretraining")
