"""
Comprehensive Examples: Normalizing Flows

This file demonstrates all normalizing flow components with practical examples
and real-world applications.

Author: Ali Mehdi
Date: February 27, 2026
"""

import numpy as np
from ilovetools.ml.normalizing_flows import (
    NormalizingFlow,
    PlanarFlow,
    CouplingLayer,
    flow_nll_loss,
    flow_bits_per_dim,
    check_invertibility,
    visualize_flow_transformation,
)

print("=" * 80)
print("NORMALIZING FLOWS - COMPREHENSIVE EXAMPLES")
print("=" * 80)
print()

# ============================================================================
# EXAMPLE 1: Planar Flow
# ============================================================================
print("EXAMPLE 1: Planar Flow (Simple Transformation)")
print("-" * 80)

# Create planar flow
planar = PlanarFlow(dim=10)
print(f"Planar Flow:")
print(f"  Dimension: 10")
print(f"  Parameters: u, w, b")
print()

# Forward transformation
z = np.random.randn(32, 10)
x, log_det = planar.forward(z)

print(f"Forward transformation:")
print(f"  Input z: {z.shape}")
print(f"  Output x: {x.shape}")
print(f"  Log-det: {log_det.shape}")
print(f"  Log-det mean: {log_det.mean():.4f}")
print()

# Inverse transformation
z_recon = planar.inverse(x)
error = np.abs(z - z_recon).max()

print(f"Inverse transformation:")
print(f"  Reconstruction error: {error:.6f}")
print()

print("Transformation Formula:")
print("  f(z) = z + u * tanh(w^T z + b)")
print()

print("Key Concepts:")
print("  • Planar: Transformation in hyperplane")
print("  • Invertible: Can recover original input")
print("  • Log-det: Tracks volume change")
print("  • Simple: Good for variational inference")

print("\n✓ Example 1 completed\n")

# ============================================================================
# EXAMPLE 2: Coupling Layer (RealNVP)
# ============================================================================
print("EXAMPLE 2: Coupling Layer (RealNVP)")
print("-" * 80)

# Create coupling layer
coupling = CouplingLayer(dim=10, hidden_dim=64)
print(f"Coupling Layer:")
print(f"  Dimension: 10")
print(f"  Hidden dimension: 64")
print(f"  Split dimension: 5")
print()

# Forward transformation
z = np.random.randn(32, 10)
x, log_det = coupling.forward(z)

print(f"Forward transformation:")
print(f"  Input z: {z.shape}")
print(f"  Output x: {x.shape}")
print(f"  Log-det: {log_det.shape}")
print(f"  Log-det mean: {log_det.mean():.4f}")
print()

# Exact inverse
z_recon = coupling.inverse(x)
error = np.abs(z - z_recon).max()

print(f"Exact inverse:")
print(f"  Reconstruction error: {error:.10f}")
print(f"  (Should be near machine precision)")
print()

print("Coupling Transformation:")
print("  x1, x2 = split(z)")
print("  y1 = x1")
print("  y2 = x2 * exp(s(x1)) + t(x1)")
print()

print("Benefits:")
print("  ✓ Exact inverse (no iterations)")
print("  ✓ Efficient O(1) computation")
print("  ✓ Triangular Jacobian (easy determinant)")
print("  ✓ Expressive transformations")

print("\n✓ Example 2 completed\n")

# ============================================================================
# EXAMPLE 3: Normalizing Flow Model
# ============================================================================
print("EXAMPLE 3: Normalizing Flow Model (Stacked Layers)")
print("-" * 80)

# Create flow model
flow = NormalizingFlow(dim=10, num_flows=8, flow_type='coupling', hidden_dim=64)
print(f"Normalizing Flow:")
print(f"  Dimension: 10")
print(f"  Number of flows: 8")
print(f"  Flow type: coupling")
print(f"  Hidden dimension: 64")
print()

# Forward pass
x = np.random.randn(32, 10)
z, log_det_total = flow.forward(x)

print(f"Forward pass (data → latent):")
print(f"  Data x: {x.shape}")
print(f"  Latent z: {z.shape}")
print(f"  Total log-det: {log_det_total.shape}")
print(f"  Log-det mean: {log_det_total.mean():.4f}")
print()

# Inverse pass
x_recon = flow.inverse(z)
error = np.abs(x - x_recon).max()

print(f"Inverse pass (latent → data):")
print(f"  Reconstruction error: {error:.10f}")
print()

print("Architecture:")
print("  Input → Flow1 → Flow2 → ... → Flow8 → Latent")
print("  Each flow: Coupling layer with scale & translation networks")

print("\n✓ Example 3 completed\n")

# ============================================================================
# EXAMPLE 4: Log Probability Computation
# ============================================================================
print("EXAMPLE 4: Exact Log Probability")
print("-" * 80)

# Compute log probability
x = np.random.randn(32, 10)
log_prob = flow.log_prob(x)

print(f"Log probability computation:")
print(f"  Data: {x.shape}")
print(f"  Log p(x): {log_prob.shape}")
print(f"  Mean log p(x): {log_prob.mean():.4f}")
print(f"  Min log p(x): {log_prob.min():.4f}")
print(f"  Max log p(x): {log_prob.max():.4f}")
print()

print("Formula:")
print("  log p(x) = log p(z) + log|det(∂f/∂x)|")
print("  where z = f(x) and p(z) = N(0, I)")
print()

print("Key Advantage:")
print("  • Exact likelihood (no approximation)")
print("  • Unlike VAE (ELBO bound)")
print("  • Unlike GAN (no likelihood)")
print("  • Tractable and efficient")

print("\n✓ Example 4 completed\n")

# ============================================================================
# EXAMPLE 5: Sampling from Flow
# ============================================================================
print("EXAMPLE 5: Sampling from Flow")
print("-" * 80)

# Generate samples
num_samples = 100
samples = flow.sample(num_samples=num_samples)

print(f"Sampling process:")
print(f"  1. Sample z ~ N(0, I)")
print(f"  2. Transform x = f⁻¹(z)")
print(f"  3. Result: {samples.shape}")
print()

print(f"Generated samples:")
print(f"  Shape: {samples.shape}")
print(f"  Mean: {samples.mean():.4f}")
print(f"  Std: {samples.std():.4f}")
print()

print("Sampling Benefits:")
print("  ✓ Single forward pass (fast)")
print("  ✓ No iterative denoising (unlike diffusion)")
print("  ✓ Exact inverse transformation")
print("  ✓ Efficient generation")

print("\n✓ Example 5 completed\n")

# ============================================================================
# EXAMPLE 6: Training with NLL Loss
# ============================================================================
print("EXAMPLE 6: Training with Negative Log-Likelihood")
print("-" * 80)

# Training data
x_train = np.random.randn(128, 10)

# Compute loss
loss = flow_nll_loss(x_train, flow)

print(f"Training:")
print(f"  Data: {x_train.shape}")
print(f"  NLL loss: {loss:.4f}")
print()

print("Training Objective:")
print("  Maximize: E[log p(x)]")
print("  Minimize: -E[log p(x)] (NLL)")
print()

print("Loss Formula:")
print("  L = -1/N Σ log p(x_i)")
print("  = -1/N Σ [log p(z_i) + log|det(∂f/∂x_i)|]")
print()

print("Training Process:")
print("  1. Forward pass: x → z")
print("  2. Compute log p(z) (Gaussian)")
print("  3. Add log-det Jacobian")
print("  4. Maximize total log p(x)")
print("  5. Backprop through invertible layers")

print("\n✓ Example 6 completed\n")

# ============================================================================
# EXAMPLE 7: Bits Per Dimension
# ============================================================================
print("EXAMPLE 7: Bits Per Dimension Metric")
print("-" * 80)

# Compute BPD
x_test = np.random.randn(64, 10)
bpd = flow_bits_per_dim(x_test, flow)

print(f"Bits per dimension:")
print(f"  Test data: {x_test.shape}")
print(f"  BPD: {bpd:.4f}")
print()

print("BPD Formula:")
print("  BPD = -log₂ p(x) / D")
print("  where D is the dimension")
print()

print("Interpretation:")
print("  • Lower BPD = Better model")
print("  • Common metric for image models")
print("  • CIFAR-10: ~3.0 BPD (good)")
print("  • ImageNet: ~3.5 BPD (good)")
print()

print("Comparison:")
print("  • Glow (CIFAR-10): 3.35 BPD")
print("  • RealNVP (CIFAR-10): 3.49 BPD")
print("  • PixelCNN++ (CIFAR-10): 2.92 BPD")

print("\n✓ Example 7 completed\n")

# ============================================================================
# EXAMPLE 8: Invertibility Check
# ============================================================================
print("EXAMPLE 8: Invertibility Verification")
print("-" * 80)

# Check invertibility
x_check = np.random.randn(32, 10)
is_invertible = check_invertibility(flow, x_check, tolerance=1e-5)

print(f"Invertibility check:")
print(f"  Test data: {x_check.shape}")
print(f"  Tolerance: 1e-5")
print(f"  Is invertible: {is_invertible}")
print()

# Detailed error analysis
z, _ = flow.forward(x_check)
x_recon = flow.inverse(z)
errors = np.abs(x_check - x_recon)

print(f"Reconstruction errors:")
print(f"  Max error: {errors.max():.10f}")
print(f"  Mean error: {errors.mean():.10f}")
print(f"  Std error: {errors.std():.10f}")
print()

print("Why Invertibility Matters:")
print("  • Exact sampling: z → x")
print("  • Exact likelihood: x → z → log p(x)")
print("  • Bidirectional: Can go both ways")
print("  • No approximation errors")

print("\n✓ Example 8 completed\n")

# ============================================================================
# EXAMPLE 9: Flow Transformation Visualization
# ============================================================================
print("EXAMPLE 9: Flow Transformation Visualization")
print("-" * 80)

# Visualize transformation
x_vis = np.random.randn(100, 10)
z_vis, log_det_vis = visualize_flow_transformation(flow, x_vis)

print(f"Transformation visualization:")
print(f"  Data x: {x_vis.shape}")
print(f"  Latent z: {z_vis.shape}")
print(f"  Log-det: {log_det_vis.shape}")
print()

print(f"Data statistics:")
print(f"  x mean: {x_vis.mean():.4f}, std: {x_vis.std():.4f}")
print(f"  z mean: {z_vis.mean():.4f}, std: {z_vis.std():.4f}")
print()

print(f"Log-det statistics:")
print(f"  Mean: {log_det_vis.mean():.4f}")
print(f"  Std: {log_det_vis.std():.4f}")
print(f"  Range: [{log_det_vis.min():.4f}, {log_det_vis.max():.4f}]")
print()

print("Interpretation:")
print("  • Positive log-det: Volume expansion")
print("  • Negative log-det: Volume contraction")
print("  • Zero log-det: Volume preserving")

print("\n✓ Example 9 completed\n")

# ============================================================================
# EXAMPLE 10: Real-World Applications
# ============================================================================
print("EXAMPLE 10: Real-World Applications")
print("-" * 80)

print("Image Generation:")
print("  • Glow: 256×256 high-quality faces")
print("  • RealNVP: CIFAR-10, ImageNet")
print("  • Flow++: State-of-the-art on CIFAR-10")
print("  • Applications: Photo synthesis, editing")
print()

print("Audio Generation:")
print("  • WaveGlow: Real-time speech synthesis")
print("  • FloWaveNet: High-quality audio")
print("  • Parallel WaveGAN: Fast vocoder")
print("  • Applications: TTS, music generation")
print()

print("Density Estimation:")
print("  • Exact likelihood computation")
print("  • Anomaly detection (low likelihood = anomaly)")
print("  • Out-of-distribution detection")
print("  • Applications: Fraud detection, QA")
print()

print("Variational Inference:")
print("  • Flexible posterior approximations")
print("  • Better than Gaussian VAE")
print("  • Normalizing flows as priors")
print("  • Applications: Bayesian deep learning")
print()

print("Data Compression:")
print("  • Lossless compression via likelihood")
print("  • Better than traditional codecs")
print("  • Learned compression")
print("  • Applications: Image/video compression")
print()

print("Molecular Generation:")
print("  • Drug discovery: Generate molecules")
print("  • Chemical property optimization")
print("  • Invertible molecular representations")
print("  • Applications: Pharmaceutical research")
print()

print("Dequantization:")
print("  • Convert discrete to continuous")
print("  • Add uniform noise then model")
print("  • Better likelihood estimates")
print("  • Applications: Image modeling")

print("\n✓ Example 10 completed\n")

# ============================================================================
# EXAMPLE 11: Flow vs Other Generative Models
# ============================================================================
print("EXAMPLE 11: Normalizing Flows vs Other Models")
print("-" * 80)

print("Exact Likelihood:")
print("  Flows: ★★★★★ (Exact)")
print("  VAE: ★★★☆☆ (ELBO approximation)")
print("  GAN: ☆☆☆☆☆ (No likelihood)")
print("  Diffusion: ★★★★☆ (Tractable but complex)")
print()

print("Sample Quality:")
print("  Flows: ★★★★☆ (High quality)")
print("  VAE: ★★★☆☆ (Blurry)")
print("  GAN: ★★★★★ (Photorealistic)")
print("  Diffusion: ★★★★★ (Photorealistic)")
print()

print("Sampling Speed:")
print("  Flows: ★★★★★ (Single pass)")
print("  VAE: ★★★★★ (Single pass)")
print("  GAN: ★★★★★ (Single pass)")
print("  Diffusion: ★★☆☆☆ (1000 steps)")
print()

print("Training Stability:")
print("  Flows: ★★★★☆ (Stable)")
print("  VAE: ★★★★☆ (Stable)")
print("  GAN: ★★☆☆☆ (Unstable)")
print("  Diffusion: ★★★★★ (Very stable)")
print()

print("Invertibility:")
print("  Flows: ★★★★★ (Exactly invertible)")
print("  VAE: ★★☆☆☆ (Approximate)")
print("  GAN: ☆☆☆☆☆ (Not invertible)")
print("  Diffusion: ★★★☆☆ (Iterative inverse)")
print()

print("Best Use Cases:")
print("  Flows: Density estimation, exact likelihood")
print("  VAE: Representation learning, anomaly detection")
print("  GAN: High-quality image generation")
print("  Diffusion: Text-to-image, photorealistic generation")

print("\n✓ Example 11 completed\n")

# ============================================================================
# EXAMPLE 12: Normalizing Flows Evolution
# ============================================================================
print("EXAMPLE 12: Normalizing Flows Evolution Timeline")
print("-" * 80)

print("2015 - Variational Inference with Flows (Rezende & Mohamed):")
print("  • First normalizing flows paper")
print("  • Planar and radial flows")
print("  • Flexible variational posteriors")
print()

print("2016 - RealNVP (Dinh et al.):")
print("  • Affine coupling layers")
print("  • Exact inverse computation")
print("  • High-quality image generation")
print("  • Breakthrough for scalability")
print()

print("2017 - MAF (Papamakarios et al.):")
print("  • Masked Autoregressive Flow")
print("  • Autoregressive transformations")
print("  • Better density estimation")
print()

print("2018 - Glow (Kingma & Dhariwal):")
print("  • Invertible 1×1 convolutions")
print("  • 256×256 face generation")
print("  • Actnorm (activation normalization)")
print("  • State-of-the-art quality")
print()

print("2018 - FFJORD (Grathwohl et al.):")
print("  • Continuous normalizing flows")
print("  • Neural ODEs")
print("  • Free-form Jacobian")
print()

print("2019 - Flow++ (Ho et al.):")
print("  • Improved coupling layers")
print("  • Variational dequantization")
print("  • Best CIFAR-10 results")
print()

print("2020 - SurVAE Flows (Nielsen et al.):")
print("  • Surjective flows")
print("  • Stochastic transformations")
print("  • Unified framework")

print("\n✓ Example 12 completed\n")

# ============================================================================
# SUMMARY
# ============================================================================
print("=" * 80)
print("ALL EXAMPLES COMPLETED SUCCESSFULLY! ✓")
print("=" * 80)
print()
print("Summary of what we covered:")
print("1. ✓ Planar Flow")
print("2. ✓ Coupling Layer (RealNVP)")
print("3. ✓ Normalizing Flow Model")
print("4. ✓ Exact Log Probability")
print("5. ✓ Sampling from Flow")
print("6. ✓ Training with NLL")
print("7. ✓ Bits Per Dimension")
print("8. ✓ Invertibility Check")
print("9. ✓ Flow Transformation Visualization")
print("10. ✓ Real-World Applications")
print("11. ✓ Flows vs Other Models")
print("12. ✓ Evolution Timeline")
print()
print("You now have a complete understanding of Normalizing Flows!")
print()
print("Next steps:")
print("- Implement Glow architecture")
print("- Train on CIFAR-10, CelebA")
print("- Add MAF (Masked Autoregressive Flow)")
print("- Build continuous flows (FFJORD)")
print("- Apply to density estimation tasks")
print()
print("GitHub: https://github.com/AliMehdi512/ilovetools")
print("Install: pip install ilovetools")
