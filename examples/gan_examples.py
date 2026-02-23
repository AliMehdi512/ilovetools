"""
Comprehensive Examples: Generative Adversarial Networks (GANs)

This file demonstrates all GAN components with practical examples
and real-world applications.

Author: Ali Mehdi
Date: February 23, 2026
"""

import numpy as np
from ilovetools.ml.gan import (
    Generator,
    DCGANGenerator,
    ConditionalGenerator,
    Discriminator,
    minimax_loss_discriminator,
    minimax_loss_generator,
    wasserstein_loss_discriminator,
    wasserstein_loss_generator,
    gradient_penalty,
    detect_mode_collapse,
    sample_latent_vectors,
)

print("=" * 80)
print("GENERATIVE ADVERSARIAL NETWORKS (GANs) - COMPREHENSIVE EXAMPLES")
print("=" * 80)
print()

# ============================================================================
# EXAMPLE 1: Basic GAN Generator
# ============================================================================
print("EXAMPLE 1: Basic GAN Generator")
print("-" * 80)

# Create generator
gen = Generator(latent_dim=100, output_shape=(3, 64, 64))
print(f"Generator created:")
print(f"  Latent dimension: 100")
print(f"  Output shape: (3, 64, 64) - RGB 64x64 images")
print()

# Sample random noise
z = np.random.randn(32, 100)
print(f"Sampled noise: {z.shape}")
print(f"  32 random vectors in 100-dimensional latent space")
print()

# Generate fake images
fake_images = gen.forward(z)
print(f"Generated images: {fake_images.shape}")
print(f"  32 fake RGB images, 64x64 pixels")
print(f"  Value range: [{fake_images.min():.2f}, {fake_images.max():.2f}]")
print()

print("Key Concepts:")
print("  • Latent Space: High-dimensional space of random noise")
print("  • Generator: Transforms noise → realistic images")
print("  • Tanh Output: Values in [-1, 1] range")
print("  • Deterministic: Same noise → same image")

print("\n✓ Example 1 completed\n")

# ============================================================================
# EXAMPLE 2: Discriminator Network
# ============================================================================
print("EXAMPLE 2: Discriminator Network")
print("-" * 80)

# Create discriminator
disc = Discriminator(input_shape=(3, 64, 64))
print(f"Discriminator created:")
print(f"  Input shape: (3, 64, 64)")
print(f"  Output: Single probability [0, 1]")
print()

# Real images (from dataset)
real_images = np.random.randn(32, 3, 64, 64)
real_preds = disc.forward(real_images)
print(f"Real images predictions: {real_preds.shape}")
print(f"  Mean probability: {real_preds.mean():.3f}")
print(f"  (Should be close to 1.0 for real images)")
print()

# Fake images (from generator)
fake_preds = disc.forward(fake_images)
print(f"Fake images predictions: {fake_preds.shape}")
print(f"  Mean probability: {fake_preds.mean():.3f}")
print(f"  (Should be close to 0.0 for fake images)")
print()

print("Training Dynamics:")
print("  • Discriminator learns: Real → 1, Fake → 0")
print("  • Generator learns: Fool discriminator (Fake → 1)")
print("  • Adversarial game: Two networks compete")
print("  • Nash Equilibrium: Both networks optimal")

print("\n✓ Example 2 completed\n")

# ============================================================================
# EXAMPLE 3: Minimax Loss
# ============================================================================
print("EXAMPLE 3: Minimax Loss (Original GAN)")
print("-" * 80)

# Discriminator loss
d_loss = minimax_loss_discriminator(real_preds, fake_preds)
print(f"Discriminator loss: {d_loss:.4f}")
print()

print("Formula:")
print("  L_D = -E[log D(x)] - E[log(1 - D(G(z)))]")
print()

print("Interpretation:")
print("  • Wants to maximize log D(x) for real images")
print("  • Wants to maximize log(1 - D(G(z))) for fake images")
print("  • Lower loss = better discrimination")
print()

# Generator loss
g_loss = minimax_loss_generator(fake_preds)
print(f"Generator loss: {g_loss:.4f}")
print()

print("Formula:")
print("  L_G = -E[log D(G(z))]")
print()

print("Interpretation:")
print("  • Wants to maximize D(G(z)) (fool discriminator)")
print("  • Lower loss = better generation")

print("\n✓ Example 3 completed\n")

# ============================================================================
# EXAMPLE 4: DCGAN (Deep Convolutional GAN)
# ============================================================================
print("EXAMPLE 4: DCGAN (Deep Convolutional GAN)")
print("-" * 80)

# Create DCGAN generator
dcgan_gen = DCGANGenerator(latent_dim=100, output_channels=3)
print("DCGAN Generator Architecture:")
print("  100D noise")
print("  → Dense → 4×4×512")
print("  → UpConv → 8×8×256")
print("  → UpConv → 16×16×128")
print("  → UpConv → 32×32×64")
print("  → UpConv → 64×64×3")
print()

# Generate images
z_dcgan = np.random.randn(16, 100)
dcgan_images = dcgan_gen.forward(z_dcgan)
print(f"Generated images: {dcgan_images.shape}")
print()

print("DCGAN Guidelines (Radford et al., 2015):")
print("  1. ✓ Replace pooling with strided convolutions")
print("  2. ✓ Use batch normalization in both G and D")
print("  3. ✓ Remove fully connected hidden layers")
print("  4. ✓ Use ReLU in generator (except output: Tanh)")
print("  5. ✓ Use LeakyReLU in discriminator")
print()

print("Benefits:")
print("  • Better image quality than vanilla GAN")
print("  • More stable training")
print("  • Learns hierarchical features")
print("  • Widely used baseline")

print("\n✓ Example 4 completed\n")

# ============================================================================
# EXAMPLE 5: Conditional GAN (cGAN)
# ============================================================================
print("EXAMPLE 5: Conditional GAN (cGAN)")
print("-" * 80)

# Create conditional generator
cgan_gen = ConditionalGenerator(latent_dim=100, num_classes=10)
print("Conditional GAN:")
print("  Latent dimension: 100")
print("  Number of classes: 10 (e.g., MNIST digits 0-9)")
print()

# Generate class-specific images
z_cond = np.random.randn(10, 100)
labels = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])  # One of each class
cond_images = cgan_gen.forward(z_cond, labels)
print(f"Generated images: {cond_images.shape}")
print(f"  10 images, one for each class (0-9)")
print()

print("Applications:")
print("  • Class-conditional generation (generate specific digit)")
print("  • Text-to-image synthesis (generate from description)")
print("  • Attribute-guided generation (age, gender, style)")
print("  • Image-to-image translation (day→night, summer→winter)")
print()

print("Example Use Cases:")
print("  • Generate '7' digit: cgan_gen.forward(z, label=7)")
print("  • Generate 'cat' image: cgan_gen.forward(z, label='cat')")
print("  • Generate 'smiling face': cgan_gen.forward(z, attr='smile')")

print("\n✓ Example 5 completed\n")

# ============================================================================
# EXAMPLE 6: Wasserstein GAN (WGAN)
# ============================================================================
print("EXAMPLE 6: Wasserstein GAN (WGAN)")
print("-" * 80)

# Simulate critic scores (WGAN uses critic instead of discriminator)
real_scores = np.random.randn(32, 1) + 5.0  # High scores for real
fake_scores = np.random.randn(32, 1) - 3.0  # Low scores for fake

# Wasserstein loss
w_d_loss = wasserstein_loss_discriminator(real_scores, fake_scores)
w_g_loss = wasserstein_loss_generator(fake_scores)

print(f"Wasserstein Discriminator Loss: {w_d_loss:.4f}")
print(f"Wasserstein Generator Loss: {w_g_loss:.4f}")
print()

print("Formula:")
print("  L_D = -E[D(x)] + E[D(G(z))]")
print("  L_G = -E[D(G(z))]")
print()

print("Benefits over Original GAN:")
print("  ✓ More stable training (no mode collapse)")
print("  ✓ Meaningful loss metric (correlates with quality)")
print("  ✓ No saturation (gradients always flow)")
print("  ✓ Works with any architecture")
print()

print("Key Difference:")
print("  • Original GAN: Discriminator outputs probability [0, 1]")
print("  • WGAN: Critic outputs unbounded score [-∞, +∞]")
print("  • Measures Wasserstein distance (Earth Mover's Distance)")

print("\n✓ Example 6 completed\n")

# ============================================================================
# EXAMPLE 7: Gradient Penalty (WGAN-GP)
# ============================================================================
print("EXAMPLE 7: Gradient Penalty (WGAN-GP)")
print("-" * 80)

# Create discriminator
disc_gp = Discriminator(input_shape=(3, 64, 64))

# Real and fake data
real_data = np.random.randn(32, 3, 64, 64)
fake_data = np.random.randn(32, 3, 64, 64)

# Compute gradient penalty
gp = gradient_penalty(disc_gp, real_data, fake_data, lambda_gp=10.0)
print(f"Gradient Penalty: {gp:.4f}")
print()

print("Formula:")
print("  GP = λ × E[(||∇D(x̂)||₂ - 1)²]")
print("  where x̂ = εx + (1-ε)G(z), ε ~ U(0,1)")
print()

print("Purpose:")
print("  • Enforces Lipschitz constraint (gradient norm ≈ 1)")
print("  • Replaces weight clipping in original WGAN")
print("  • More stable training")
print("  • Better gradient flow")
print()

print("Typical λ values:")
print("  • λ = 10 (standard)")
print("  • λ = 1 (lighter regularization)")
print("  • λ = 100 (stronger regularization)")

print("\n✓ Example 7 completed\n")

# ============================================================================
# EXAMPLE 8: Mode Collapse Detection
# ============================================================================
print("EXAMPLE 8: Mode Collapse Detection")
print("-" * 80)

# Good diversity
diverse_samples = np.random.randn(100, 3, 64, 64)
is_collapsed_diverse = detect_mode_collapse(diverse_samples, threshold=0.1)
print(f"Diverse samples - Mode collapse: {is_collapsed_diverse}")
print(f"  Variance: {np.var(diverse_samples):.4f}")
print()

# Mode collapse (all similar)
collapsed_samples = np.ones((100, 3, 64, 64)) + np.random.randn(100, 3, 64, 64) * 0.01
is_collapsed = detect_mode_collapse(collapsed_samples, threshold=0.1)
print(f"Collapsed samples - Mode collapse: {is_collapsed}")
print(f"  Variance: {np.var(collapsed_samples):.4f}")
print()

print("Mode Collapse Symptoms:")
print("  • Generator produces limited variety")
print("  • All outputs look similar")
print("  • Ignores parts of data distribution")
print("  • Low variance across samples")
print()

print("Solutions:")
print("  • Use Wasserstein GAN (WGAN)")
print("  • Add diversity regularization")
print("  • Unrolled GAN optimization")
print("  • Minibatch discrimination")
print("  • Feature matching")

print("\n✓ Example 8 completed\n")

# ============================================================================
# EXAMPLE 9: Latent Space Sampling
# ============================================================================
print("EXAMPLE 9: Latent Space Sampling")
print("-" * 80)

# Normal distribution
z_normal = sample_latent_vectors(32, 100, distribution='normal')
print(f"Normal distribution: {z_normal.shape}")
print(f"  Mean: {z_normal.mean():.3f} (should be ~0)")
print(f"  Std: {z_normal.std():.3f} (should be ~1)")
print()

# Uniform distribution
z_uniform = sample_latent_vectors(32, 100, distribution='uniform')
print(f"Uniform distribution: {z_uniform.shape}")
print(f"  Min: {z_uniform.min():.3f} (should be ~-1)")
print(f"  Max: {z_uniform.max():.3f} (should be ~1)")
print()

print("Distribution Choice:")
print("  • Normal (Gaussian): Most common, smooth interpolation")
print("  • Uniform: Bounded range, easier to sample")
print("  • Spherical: Used in some advanced GANs")
print()

print("Latent Space Properties:")
print("  • Smooth: Small changes → small output changes")
print("  • Continuous: Interpolation creates smooth transitions")
print("  • Disentangled: Each dimension controls specific feature")

print("\n✓ Example 9 completed\n")

# ============================================================================
# EXAMPLE 10: GAN Training Loop (Pseudocode)
# ============================================================================
print("EXAMPLE 10: GAN Training Loop")
print("-" * 80)

print("Training Algorithm:")
print()
print("for epoch in range(num_epochs):")
print("    for batch in dataloader:")
print("        # 1. Train Discriminator")
print("        real_images = batch")
print("        z = sample_latent_vectors(batch_size, latent_dim)")
print("        fake_images = generator(z)")
print("        ")
print("        real_preds = discriminator(real_images)")
print("        fake_preds = discriminator(fake_images)")
print("        ")
print("        d_loss = minimax_loss_discriminator(real_preds, fake_preds)")
print("        d_loss.backward()")
print("        optimizer_d.step()")
print("        ")
print("        # 2. Train Generator")
print("        z = sample_latent_vectors(batch_size, latent_dim)")
print("        fake_images = generator(z)")
print("        fake_preds = discriminator(fake_images)")
print("        ")
print("        g_loss = minimax_loss_generator(fake_preds)")
print("        g_loss.backward()")
print("        optimizer_g.step()")
print()

print("Training Tips:")
print("  • Train D more than G (k=5 D steps per 1 G step)")
print("  • Use label smoothing (real=0.9 instead of 1.0)")
print("  • Add noise to discriminator inputs")
print("  • Use spectral normalization")
print("  • Monitor FID score for quality")

print("\n✓ Example 10 completed\n")

# ============================================================================
# EXAMPLE 11: Real-World Applications
# ============================================================================
print("EXAMPLE 11: Real-World Applications")
print("-" * 80)

print("Image Generation:")
print("  • StyleGAN: Photorealistic faces (1024×1024)")
print("  • BigGAN: High-resolution ImageNet (512×512)")
print("  • ProGAN: Progressive growing for quality")
print("  • Applications: Art, design, content creation")
print()

print("Image-to-Image Translation:")
print("  • Pix2Pix: Paired translation (edges→photo)")
print("  • CycleGAN: Unpaired translation (horse→zebra)")
print("  • StarGAN: Multi-domain translation")
print("  • Applications: Photo editing, style transfer")
print()

print("Super Resolution:")
print("  • SRGAN: 4x upscaling with perceptual loss")
print("  • ESRGAN: Enhanced SRGAN (8x upscaling)")
print("  • Applications: Photo enhancement, video upscaling")
print()

print("Deepfakes & Face Swapping:")
print("  • FaceSwap: Real-time face replacement")
print("  • DeepFaceLab: High-quality face swapping")
print("  • First Order Motion: Animate portraits")
print("  • Applications: Entertainment, VFX, avatars")
print()

print("Text-to-Image:")
print("  • AttnGAN: Attention-based text-to-image")
print("  • StackGAN: Stacked generation (low→high res)")
print("  • DALL-E precursor: Conditional generation")
print("  • Applications: Creative design, advertising")
print()

print("Data Augmentation:")
print("  • Generate synthetic training data")
print("  • Balance imbalanced datasets")
print("  • Privacy-preserving synthetic data")
print("  • Applications: Medical imaging, rare events")
print()

print("Drug Discovery:")
print("  • Molecular generation (new drug candidates)")
print("  • Protein structure prediction")
print("  • Chemical property optimization")
print("  • Applications: Pharmaceutical research")

print("\n✓ Example 11 completed\n")

# ============================================================================
# EXAMPLE 12: GAN Evolution Timeline
# ============================================================================
print("EXAMPLE 12: GAN Evolution Timeline")
print("-" * 80)

print("2014 - Original GAN (Goodfellow et al.):")
print("  • First adversarial training framework")
print("  • Minimax game theory")
print("  • Generated 28×28 MNIST digits")
print()

print("2015 - DCGAN (Radford et al.):")
print("  • Convolutional architecture")
print("  • Stable training guidelines")
print("  • 64×64 bedroom images")
print()

print("2016 - Conditional GAN (Mirza & Osindero):")
print("  • Class-conditional generation")
print("  • Controlled synthesis")
print()

print("2017 - Wasserstein GAN (Arjovsky et al.):")
print("  • Wasserstein distance")
print("  • Stable training")
print("  • Meaningful loss metric")
print()

print("2017 - Progressive GAN (Karras et al.):")
print("  • Progressive growing (4×4 → 1024×1024)")
print("  • High-resolution faces")
print()

print("2018 - StyleGAN (Karras et al.):")
print("  • Style-based generator")
print("  • Unprecedented quality")
print("  • Disentangled latent space")
print()

print("2018 - BigGAN (Brock et al.):")
print("  • Large-scale training")
print("  • 512×512 ImageNet")
print("  • Class-conditional synthesis")
print()

print("2020 - StyleGAN2 (Karras et al.):")
print("  • Improved quality")
print("  • Removed artifacts")
print("  • State-of-the-art faces")

print("\n✓ Example 12 completed\n")

# ============================================================================
# SUMMARY
# ============================================================================
print("=" * 80)
print("ALL EXAMPLES COMPLETED SUCCESSFULLY! ✓")
print("=" * 80)
print()
print("Summary of what we covered:")
print("1. ✓ Basic GAN Generator")
print("2. ✓ Discriminator Network")
print("3. ✓ Minimax Loss")
print("4. ✓ DCGAN Architecture")
print("5. ✓ Conditional GAN")
print("6. ✓ Wasserstein GAN")
print("7. ✓ Gradient Penalty")
print("8. ✓ Mode Collapse Detection")
print("9. ✓ Latent Space Sampling")
print("10. ✓ Training Loop")
print("11. ✓ Real-World Applications")
print("12. ✓ GAN Evolution Timeline")
print()
print("You now have a complete understanding of GANs!")
print()
print("Next steps:")
print("- Implement StyleGAN, BigGAN")
print("- Train on CelebA, ImageNet")
print("- Build image-to-image translation (Pix2Pix, CycleGAN)")
print("- Explore super resolution (SRGAN)")
print("- Apply to your own generative tasks")
print()
print("GitHub: https://github.com/AliMehdi512/ilovetools")
print("Install: pip install ilovetools")
