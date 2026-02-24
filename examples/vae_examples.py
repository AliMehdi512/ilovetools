"""
Comprehensive Examples: Variational Autoencoders (VAEs)

This file demonstrates all VAE components with practical examples
and real-world applications.

Author: Ali Mehdi
Date: February 24, 2026
"""

import numpy as np
from ilovetools.ml.vae import (
    VAE,
    ConditionalVAE,
    BetaVAE,
    VAEEncoder,
    VAEDecoder,
    reconstruction_loss,
    kl_divergence,
    elbo_loss,
    sample_from_latent,
    interpolate_latent,
    latent_traversal,
)

print("=" * 80)
print("VARIATIONAL AUTOENCODERS (VAEs) - COMPREHENSIVE EXAMPLES")
print("=" * 80)
print()

# ============================================================================
# EXAMPLE 1: Basic VAE
# ============================================================================
print("EXAMPLE 1: Basic VAE")
print("-" * 80)

# Create VAE
vae = VAE(input_dim=784, latent_dim=20, hidden_dims=[512, 256])
print(f"VAE created:")
print(f"  Input dimension: 784 (28×28 MNIST images)")
print(f"  Latent dimension: 20")
print(f"  Hidden layers: [512, 256]")
print()

# Forward pass
x = np.random.rand(32, 784)  # Batch of MNIST images
x_recon, mu, logvar = vae.forward(x)

print(f"Forward pass:")
print(f"  Input: {x.shape}")
print(f"  Reconstruction: {x_recon.shape}")
print(f"  Latent mean (μ): {mu.shape}")
print(f"  Latent log-variance (log σ²): {logvar.shape}")
print()

print("Key Concepts:")
print("  • Encoder: x → (μ, log σ²)")
print("  • Reparameterization: z = μ + σ * ε, ε ~ N(0,1)")
print("  • Decoder: z → x_recon")
print("  • Probabilistic: Each input → distribution, not point")

print("\n✓ Example 1 completed\n")

# ============================================================================
# EXAMPLE 2: Reparameterization Trick
# ============================================================================
print("EXAMPLE 2: Reparameterization Trick")
print("-" * 80)

mu_example = np.array([[0.5, -0.3, 1.2]])
logvar_example = np.array([[-0.5, 0.2, -1.0]])

print(f"Latent distribution parameters:")
print(f"  μ = {mu_example[0]}")
print(f"  log σ² = {logvar_example[0]}")
print()

# Sample using reparameterization
z_samples = []
for i in range(5):
    z = vae.reparameterize(mu_example, logvar_example)
    z_samples.append(z[0])

print(f"5 samples from q(z|x):")
for i, z in enumerate(z_samples):
    print(f"  Sample {i+1}: {z}")
print()

print("Why Reparameterization?")
print("  ✓ Enables backpropagation through sampling")
print("  ✓ Gradient flows through μ and σ")
print("  ✓ Randomness moved to ε ~ N(0,1)")
print()

print("Formula:")
print("  z = μ + σ * ε")
print("  where σ = exp(0.5 * log σ²)")
print("  and ε ~ N(0, 1)")

print("\n✓ Example 2 completed\n")

# ============================================================================
# EXAMPLE 3: Loss Functions
# ============================================================================
print("EXAMPLE 3: Loss Functions (ELBO)")
print("-" * 80)

# Compute losses
recon_loss = reconstruction_loss(x, x_recon)
kl_loss = kl_divergence(mu, logvar)
total_loss = elbo_loss(x, x_recon, mu, logvar, beta=1.0)

print(f"Loss components:")
print(f"  Reconstruction loss: {recon_loss:.4f}")
print(f"  KL divergence: {kl_loss:.4f}")
print(f"  Total loss (ELBO): {total_loss:.4f}")
print()

print("ELBO Formula:")
print("  ELBO = E[log p(x|z)] - KL[q(z|x) || p(z)]")
print("  Loss = -ELBO (we minimize)")
print()

print("Interpretation:")
print("  • Reconstruction: How well decoder reconstructs input")
print("  • KL Divergence: How close latent is to N(0,1)")
print("  • Trade-off: Good reconstruction vs regularized latent")
print()

print("Training Dynamics:")
print("  • Early: High reconstruction loss, low KL")
print("  • Middle: Both losses decrease")
print("  • Late: Balance between reconstruction and KL")

print("\n✓ Example 3 completed\n")

# ============================================================================
# EXAMPLE 4: Generation from Prior
# ============================================================================
print("EXAMPLE 4: Generation from Prior")
print("-" * 80)

# Sample from prior p(z) = N(0, I)
z_prior = sample_from_latent(latent_dim=20, num_samples=10)
print(f"Sampled from prior: {z_prior.shape}")
print(f"  Mean: {z_prior.mean():.3f} (should be ~0)")
print(f"  Std: {z_prior.std():.3f} (should be ~1)")
print()

# Generate new samples
generated = vae.generate(z_prior)
print(f"Generated samples: {generated.shape}")
print(f"  10 new MNIST-like images")
print()

print("Generation Process:")
print("  1. Sample z ~ N(0, I)")
print("  2. Decode: x = decoder(z)")
print("  3. Result: New synthetic image")
print()

print("Benefits:")
print("  ✓ Generate unlimited new samples")
print("  ✓ Smooth latent space (interpolation works)")
print("  ✓ Controllable (traverse latent dimensions)")

print("\n✓ Example 4 completed\n")

# ============================================================================
# EXAMPLE 5: Latent Space Interpolation
# ============================================================================
print("EXAMPLE 5: Latent Space Interpolation")
print("-" * 80)

# Two random latent vectors
z1 = np.random.randn(20)
z2 = np.random.randn(20)

# Interpolate
z_interp = interpolate_latent(z1, z2, num_steps=10)
print(f"Interpolation:")
print(f"  Start: z1 shape {z1.shape}")
print(f"  End: z2 shape {z2.shape}")
print(f"  Interpolated: {z_interp.shape} (10 steps)")
print()

# Generate interpolated images
images_interp = vae.generate(z_interp)
print(f"Interpolated images: {images_interp.shape}")
print()

print("Applications:")
print("  • Smooth transitions between images")
print("  • Morphing animations")
print("  • Exploring latent space")
print("  • Understanding learned representations")
print()

print("Example Use Cases:")
print("  • Face morphing: Person A → Person B")
print("  • Digit morphing: '3' → '8'")
print("  • Style interpolation: Realistic → Artistic")

print("\n✓ Example 5 completed\n")

# ============================================================================
# EXAMPLE 6: Latent Dimension Traversal
# ============================================================================
print("EXAMPLE 6: Latent Dimension Traversal")
print("-" * 80)

# Traverse dimension 0
z_trav_0 = latent_traversal(latent_dim=20, dim_idx=0, num_steps=10)
print(f"Traversing dimension 0:")
print(f"  Shape: {z_trav_0.shape}")
print(f"  Dimension 0 values: {z_trav_0[:, 0]}")
print(f"  Other dimensions: all zeros")
print()

# Generate images showing effect of dimension 0
images_trav = vae.generate(z_trav_0)
print(f"Traversal images: {images_trav.shape}")
print()

print("What Each Dimension Might Control:")
print("  • Dimension 0: Rotation angle")
print("  • Dimension 1: Thickness/stroke width")
print("  • Dimension 2: Slant/tilt")
print("  • Dimension 3: Size/scale")
print("  • Dimension 4: Position")
print()

print("Benefits:")
print("  ✓ Interpretable representations")
print("  ✓ Controllable generation")
print("  ✓ Understanding what model learned")

print("\n✓ Example 6 completed\n")

# ============================================================================
# EXAMPLE 7: Conditional VAE (CVAE)
# ============================================================================
print("EXAMPLE 7: Conditional VAE (CVAE)")
print("-" * 80)

# Create CVAE
cvae = ConditionalVAE(input_dim=784, latent_dim=20, num_classes=10)
print(f"Conditional VAE:")
print(f"  Input dimension: 784")
print(f"  Latent dimension: 20")
print(f"  Number of classes: 10 (digits 0-9)")
print()

# Training
x_cvae = np.random.rand(32, 784)
labels = np.random.randint(0, 10, size=32)
x_recon_cvae, mu_cvae, logvar_cvae = cvae.forward(x_cvae, labels)

print(f"Training:")
print(f"  Input: {x_cvae.shape}")
print(f"  Labels: {labels.shape}")
print(f"  Reconstruction: {x_recon_cvae.shape}")
print()

# Conditional generation
z_cvae = np.random.randn(10, 20)
labels_gen = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
generated_cvae = cvae.generate(z_cvae, labels_gen)

print(f"Conditional generation:")
print(f"  Generated one of each digit (0-9)")
print(f"  Shape: {generated_cvae.shape}")
print()

print("Applications:")
print("  • Class-conditional generation: 'Generate digit 7'")
print("  • Attribute-guided synthesis: 'Generate smiling face'")
print("  • Semi-supervised learning: Few labeled samples")
print("  • Missing data imputation: Fill in missing values")

print("\n✓ Example 7 completed\n")

# ============================================================================
# EXAMPLE 8: Beta-VAE (Disentangled Representations)
# ============================================================================
print("EXAMPLE 8: Beta-VAE (Disentangled Representations)")
print("-" * 80)

# Create β-VAE with different β values
beta_values = [1.0, 4.0, 10.0]

for beta in beta_values:
    beta_vae = BetaVAE(input_dim=784, latent_dim=20, beta=beta)
    x_beta = np.random.rand(32, 784)
    x_recon_beta, mu_beta, logvar_beta = beta_vae.forward(x_beta)
    loss_beta = beta_vae.loss(x_beta, x_recon_beta, mu_beta, logvar_beta)
    
    print(f"β = {beta}:")
    print(f"  Total loss: {loss_beta:.4f}")
    print(f"  Reconstruction: {reconstruction_loss(x_beta, x_recon_beta):.4f}")
    print(f"  KL (weighted): {beta * kl_divergence(mu_beta, logvar_beta):.4f}")
    print()

print("Effect of β:")
print("  • β = 1: Standard VAE")
print("  • β > 1: Encourages disentanglement")
print("  • β = 4: Moderate disentanglement (good balance)")
print("  • β = 10: Strong disentanglement")
print("  • β = 100: Very strong (may hurt reconstruction)")
print()

print("Disentanglement Benefits:")
print("  ✓ Each dimension = single factor of variation")
print("  ✓ Interpretable latent space")
print("  ✓ Controllable generation")
print("  ✓ Better generalization")

print("\n✓ Example 8 completed\n")

# ============================================================================
# EXAMPLE 9: VAE vs Standard Autoencoder
# ============================================================================
print("EXAMPLE 9: VAE vs Standard Autoencoder")
print("-" * 80)

print("Standard Autoencoder:")
print("  • Deterministic: x → z → x_recon")
print("  • Point in latent space")
print("  • No regularization")
print("  • Can overfit")
print("  • Discrete latent space (gaps)")
print("  • Only reconstructs, doesn't generate")
print()

print("Variational Autoencoder:")
print("  • Probabilistic: x → (μ, σ) → z → x_recon")
print("  • Distribution in latent space")
print("  • KL regularization")
print("  • Prevents overfitting")
print("  • Continuous latent space (smooth)")
print("  • Generates new samples")
print()

print("Key Differences:")
print("  1. VAE learns distributions, AE learns points")
print("  2. VAE can generate, AE only reconstructs")
print("  3. VAE has smooth latent space, AE has gaps")
print("  4. VAE is regularized, AE is not")

print("\n✓ Example 9 completed\n")

# ============================================================================
# EXAMPLE 10: Real-World Applications
# ============================================================================
print("EXAMPLE 10: Real-World Applications")
print("-" * 80)

print("Image Generation:")
print("  • CelebA faces: 64×64 photorealistic faces")
print("  • MNIST digits: Handwritten digit generation")
print("  • Fashion-MNIST: Clothing item generation")
print("  • Applications: Data augmentation, creative design")
print()

print("Anomaly Detection:")
print("  • Reconstruction error for outliers")
print("  • Fraud detection: Credit card transactions")
print("  • Manufacturing: Defect detection")
print("  • Healthcare: Disease detection from medical images")
print()

print("Semi-Supervised Learning:")
print("  • Few labeled samples + many unlabeled")
print("  • CVAE for classification with limited labels")
print("  • Applications: Medical imaging, rare events")
print()

print("Drug Discovery:")
print("  • Molecular generation: New drug candidates")
print("  • Chemical property optimization")
print("  • Protein structure prediction")
print("  • Applications: Pharmaceutical research")
print()

print("Text Generation:")
print("  • Sentence VAE: Generate coherent sentences")
print("  • Dialogue generation: Chatbot responses")
print("  • Text style transfer: Formal ↔ Casual")
print("  • Applications: NLP, content creation")
print()

print("Music Generation:")
print("  • MusicVAE: Generate melodies")
print("  • Rhythm generation: Drum patterns")
print("  • Interpolation: Smooth transitions")
print("  • Applications: Music production, AI composition")
print()

print("Data Compression:")
print("  • Lossy compression with learned representations")
print("  • Better than traditional codecs for specific domains")
print("  • Applications: Image/video compression")

print("\n✓ Example 10 completed\n")

# ============================================================================
# EXAMPLE 11: Training Tips
# ============================================================================
print("EXAMPLE 11: Training Tips")
print("-" * 80)

print("Hyperparameters:")
print("  • Latent dimension: 10-512 (20-50 typical)")
print("  • Learning rate: 1e-4 to 1e-3")
print("  • Batch size: 64-256")
print("  • β (Beta-VAE): 1-10 (4 is good default)")
print()

print("Architecture:")
print("  • Encoder: [input → 512 → 256 → latent]")
print("  • Decoder: [latent → 256 → 512 → output]")
print("  • Activation: ReLU (hidden), Sigmoid (output)")
print()

print("Training Tricks:")
print("  1. KL Annealing: Gradually increase KL weight")
print("  2. Free Bits: Allow minimum KL per dimension")
print("  3. Warm-up: Start with low β, increase gradually")
print("  4. Batch Normalization: Stabilizes training")
print("  5. Learning Rate Scheduling: Reduce on plateau")
print()

print("Common Issues:")
print("  • KL Collapse: KL → 0, latent not used")
print("    Solution: KL annealing, free bits")
print("  • Posterior Collapse: Decoder ignores latent")
print("    Solution: Stronger decoder, weaker encoder")
print("  • Blurry Reconstructions: MSE loss")
print("    Solution: Perceptual loss, adversarial training")

print("\n✓ Example 11 completed\n")

# ============================================================================
# EXAMPLE 12: VAE Evolution Timeline
# ============================================================================
print("EXAMPLE 12: VAE Evolution Timeline")
print("-" * 80)

print("2013 - Original VAE (Kingma & Welling):")
print("  • First variational autoencoder")
print("  • ELBO optimization")
print("  • Reparameterization trick")
print()

print("2015 - Conditional VAE (Sohn et al.):")
print("  • Class-conditional generation")
print("  • Semi-supervised learning")
print()

print("2016 - Importance Weighted VAE (Burda et al.):")
print("  • Tighter ELBO bound")
print("  • Better log-likelihood")
print()

print("2017 - β-VAE (Higgins et al.):")
print("  • Disentangled representations")
print("  • Interpretable latent dimensions")
print()

print("2017 - VQ-VAE (van den Oord et al.):")
print("  • Discrete latent space")
print("  • Vector quantization")
print("  • High-quality image generation")
print()

print("2018 - WAE (Wasserstein Autoencoder):")
print("  • Wasserstein distance")
print("  • Alternative to KL divergence")
print()

print("2020 - VQ-VAE-2 (Razavi et al.):")
print("  • Hierarchical VQ-VAE")
print("  • 1024×1024 image generation")
print("  • State-of-the-art quality")

print("\n✓ Example 12 completed\n")

# ============================================================================
# SUMMARY
# ============================================================================
print("=" * 80)
print("ALL EXAMPLES COMPLETED SUCCESSFULLY! ✓")
print("=" * 80)
print()
print("Summary of what we covered:")
print("1. ✓ Basic VAE")
print("2. ✓ Reparameterization Trick")
print("3. ✓ Loss Functions (ELBO)")
print("4. ✓ Generation from Prior")
print("5. ✓ Latent Space Interpolation")
print("6. ✓ Latent Dimension Traversal")
print("7. ✓ Conditional VAE")
print("8. ✓ Beta-VAE (Disentanglement)")
print("9. ✓ VAE vs Standard Autoencoder")
print("10. ✓ Real-World Applications")
print("11. ✓ Training Tips")
print("12. ✓ VAE Evolution Timeline")
print()
print("You now have a complete understanding of VAEs!")
print()
print("Next steps:")
print("- Train VAE on MNIST, CelebA")
print("- Implement VQ-VAE for discrete latents")
print("- Build anomaly detection system")
print("- Explore disentanglement with β-VAE")
print("- Apply to your own generative tasks")
print()
print("GitHub: https://github.com/AliMehdi512/ilovetools")
print("Install: pip install ilovetools")
