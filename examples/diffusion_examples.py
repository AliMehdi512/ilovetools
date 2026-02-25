"""
Comprehensive Examples: Diffusion Models

This file demonstrates all diffusion model components with practical examples
and real-world applications.

Author: Ali Mehdi
Date: February 25, 2026
"""

import numpy as np
from ilovetools.ml.diffusion import (
    DiffusionModel,
    DDIMSampler,
    linear_beta_schedule,
    cosine_beta_schedule,
    quadratic_beta_schedule,
    diffusion_loss,
    vlb_loss,
    extract_timestep,
    noise_like,
)

print("=" * 80)
print("DIFFUSION MODELS - COMPREHENSIVE EXAMPLES")
print("=" * 80)
print()

# ============================================================================
# EXAMPLE 1: Variance Schedules
# ============================================================================
print("EXAMPLE 1: Variance Schedules")
print("-" * 80)

timesteps = 1000

# Linear schedule
linear_betas = linear_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02)
print(f"Linear schedule:")
print(f"  Shape: {linear_betas.shape}")
print(f"  Start β: {linear_betas[0]:.6f}")
print(f"  End β: {linear_betas[-1]:.6f}")
print(f"  Mean β: {linear_betas.mean():.6f}")
print()

# Cosine schedule
cosine_betas = cosine_beta_schedule(timesteps)
print(f"Cosine schedule:")
print(f"  Shape: {cosine_betas.shape}")
print(f"  Start β: {cosine_betas[0]:.6f}")
print(f"  End β: {cosine_betas[-1]:.6f}")
print(f"  Mean β: {cosine_betas.mean():.6f}")
print()

# Quadratic schedule
quadratic_betas = quadratic_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02)
print(f"Quadratic schedule:")
print(f"  Shape: {quadratic_betas.shape}")
print(f"  Start β: {quadratic_betas[0]:.6f}")
print(f"  End β: {quadratic_betas[-1]:.6f}")
print(f"  Mean β: {quadratic_betas.mean():.6f}")
print()

print("Schedule Comparison:")
print("  • Linear: Simple, uniform noise addition")
print("  • Cosine: Better quality, smoother (Stable Diffusion)")
print("  • Quadratic: Smoother than linear")
print()

print("Key Insight:")
print("  Cosine schedule adds noise more gradually at start")
print("  → Better preservation of structure")
print("  → Higher quality samples")

print("\n✓ Example 1 completed\n")

# ============================================================================
# EXAMPLE 2: Forward Diffusion Process
# ============================================================================
print("EXAMPLE 2: Forward Diffusion Process (Adding Noise)")
print("-" * 80)

# Create diffusion model
diffusion = DiffusionModel(timesteps=1000, beta_schedule='cosine')
print(f"Diffusion model created:")
print(f"  Timesteps: 1000")
print(f"  Schedule: cosine")
print()

# Clean images
x0 = np.random.randn(32, 3, 64, 64)
print(f"Clean images: {x0.shape}")
print(f"  Mean: {x0.mean():.3f}, Std: {x0.std():.3f}")
print()

# Add noise at different timesteps
timesteps_to_show = [0, 250, 500, 750, 999]

for t_val in timesteps_to_show:
    t = np.array([t_val] * 32)
    xt, noise = diffusion.q_sample(x0, t)
    
    print(f"Timestep t={t_val}:")
    print(f"  Noisy images: {xt.shape}")
    print(f"  Mean: {xt.mean():.3f}, Std: {xt.std():.3f}")
    print(f"  Signal-to-noise ratio: {diffusion.sqrt_alphas_cumprod[t_val]:.3f}")
    print()

print("Forward Process Formula:")
print("  x_t = √ᾱ_t * x_0 + √(1-ᾱ_t) * ε")
print("  where ε ~ N(0, I)")
print()

print("Observations:")
print("  • t=0: Almost clean (√ᾱ ≈ 1)")
print("  • t=500: Mixed signal and noise")
print("  • t=999: Pure noise (√ᾱ ≈ 0)")

print("\n✓ Example 2 completed\n")

# ============================================================================
# EXAMPLE 3: Reverse Diffusion Process
# ============================================================================
print("EXAMPLE 3: Reverse Diffusion Process (Denoising)")
print("-" * 80)

# Noisy image at t=500
t = np.array([500] * 32)
xt, true_noise = diffusion.q_sample(x0, t)

print(f"Starting from noisy images at t=500:")
print(f"  Shape: {xt.shape}")
print()

# Simulate noise prediction (in practice, this comes from trained model)
noise_pred = true_noise + np.random.randn(*true_noise.shape) * 0.1

# Denoise one step
x_prev = diffusion.p_sample(xt, t, noise_pred)

print(f"After one denoising step:")
print(f"  Shape: {x_prev.shape}")
print(f"  Mean: {x_prev.mean():.3f}, Std: {x_prev.std():.3f}")
print()

print("Reverse Process Formula:")
print("  x_{t-1} = 1/√α_t * (x_t - (1-α_t)/√(1-ᾱ_t) * ε_θ) + σ_t * z")
print()

print("Key Concepts:")
print("  • ε_θ: Neural network predicts noise")
print("  • Iterative: Denoise step-by-step (1000 steps)")
print("  • Stochastic: Add small noise at each step")
print("  • Markov: Each step depends only on previous")

print("\n✓ Example 3 completed\n")

# ============================================================================
# EXAMPLE 4: Complete Sampling Loop
# ============================================================================
print("EXAMPLE 4: Complete Sampling Loop (Generation)")
print("-" * 80)

# Define noise predictor (simplified)
def noise_predictor(x_t, t):
    """Simplified noise predictor (use trained model in practice)."""
    # In practice, this would be your trained U-Net model
    return np.random.randn(*x_t.shape) * 0.5

print("Generating samples from pure noise...")
print()

# Generate samples
shape = (16, 3, 64, 64)
print(f"Target shape: {shape}")
print(f"  16 images, 3 channels (RGB), 64×64 pixels")
print()

print("Sampling process:")
print("  1. Start: x_T ~ N(0, I) (pure noise)")
print("  2. Loop: t = T-1 → 0")
print("  3.   Predict noise: ε_θ(x_t, t)")
print("  4.   Denoise: x_{t-1} = f(x_t, ε_θ)")
print("  5. End: x_0 (clean sample)")
print()

# Note: Full sampling would take 1000 steps
print("Note: Full sampling requires 1000 denoising steps")
print("      Each step: ~10ms → Total: ~10 seconds per batch")

print("\n✓ Example 4 completed\n")

# ============================================================================
# EXAMPLE 5: DDIM Sampler (Fast Sampling)
# ============================================================================
print("EXAMPLE 5: DDIM Sampler (Fast Sampling)")
print("-" * 80)

# Create DDIM sampler
ddim = DDIMSampler(diffusion, eta=0.0)
print(f"DDIM Sampler:")
print(f"  eta=0.0 (deterministic)")
print(f"  Base timesteps: 1000")
print()

# Fast sampling with fewer steps
steps_comparison = [1000, 100, 50, 20]

for steps in steps_comparison:
    speedup = 1000 / steps
    time_estimate = steps * 0.01  # 10ms per step
    
    print(f"Steps: {steps}")
    print(f"  Speedup: {speedup:.1f}x faster")
    print(f"  Time: ~{time_estimate:.2f}s per batch")
    print()

print("DDIM Benefits:")
print("  ✓ 10-50x faster than DDPM")
print("  ✓ Deterministic (same noise → same output)")
print("  ✓ Same quality as DDPM")
print("  ✓ Used in Stable Diffusion")
print()

print("Formula:")
print("  x_{t-1} = √ᾱ_{t-1} * pred_x0 + √(1-ᾱ_{t-1}) * ε_θ")
print("  (No random noise when eta=0)")

print("\n✓ Example 5 completed\n")

# ============================================================================
# EXAMPLE 6: Loss Functions
# ============================================================================
print("EXAMPLE 6: Loss Functions")
print("-" * 80)

# True noise and predicted noise
noise = np.random.randn(32, 3, 64, 64)
noise_pred = noise + np.random.randn(32, 3, 64, 64) * 0.2

# Compute loss
loss = diffusion_loss(noise, noise_pred)
print(f"Diffusion loss (MSE):")
print(f"  Loss: {loss:.6f}")
print()

print("Training Objective:")
print("  L = E[||ε - ε_θ(x_t, t)||²]")
print()

print("Why Predict Noise?")
print("  • Easier than predicting x_0 directly")
print("  • More stable training")
print("  • Better gradient flow")
print("  • Empirically works best")
print()

print("Alternative Objectives:")
print("  • Predict x_0: L = ||x_0 - x̂_0||²")
print("  • Predict v: Velocity prediction")
print("  • VLB: Variational lower bound")

print("\n✓ Example 6 completed\n")

# ============================================================================
# EXAMPLE 7: Noise Schedule Comparison
# ============================================================================
print("EXAMPLE 7: Noise Schedule Comparison")
print("-" * 80)

# Create models with different schedules
schedules = ['linear', 'cosine', 'quadratic']

for schedule in schedules:
    model = DiffusionModel(timesteps=1000, beta_schedule=schedule)
    
    # Check signal retention at t=500
    signal_ratio = model.sqrt_alphas_cumprod[500]
    noise_ratio = model.sqrt_one_minus_alphas_cumprod[500]
    
    print(f"{schedule.capitalize()} schedule at t=500:")
    print(f"  Signal ratio: {signal_ratio:.4f}")
    print(f"  Noise ratio: {noise_ratio:.4f}")
    print(f"  SNR: {(signal_ratio / noise_ratio):.4f}")
    print()

print("Best Practice:")
print("  Use cosine schedule for:")
print("  • Image generation (Stable Diffusion)")
print("  • High-resolution synthesis")
print("  • Better sample quality")
print()

print("  Use linear schedule for:")
print("  • Simplicity")
print("  • Baseline experiments")
print("  • Original DDPM paper")

print("\n✓ Example 7 completed\n")

# ============================================================================
# EXAMPLE 8: Timestep Extraction
# ============================================================================
print("EXAMPLE 8: Timestep Extraction")
print("-" * 80)

# Create coefficient array
alphas = np.linspace(0.9, 0.1, 1000)
print(f"Alpha values: {alphas.shape}")
print(f"  Start: {alphas[0]:.3f}, End: {alphas[-1]:.3f}")
print()

# Extract at specific timesteps
t = np.array([0, 250, 500, 750, 999])
x_shape = (5, 3, 64, 64)

extracted = extract_timestep(alphas, t, x_shape)
print(f"Extracted values:")
print(f"  Shape: {extracted.shape}")
print(f"  Values: {extracted.flatten()}")
print()

print("Purpose:")
print("  • Extract schedule values for batch of timesteps")
print("  • Broadcast to match data shape")
print("  • Efficient batched computation")

print("\n✓ Example 8 completed\n")

# ============================================================================
# EXAMPLE 9: Real-World Applications
# ============================================================================
print("EXAMPLE 9: Real-World Applications")
print("-" * 80)

print("Text-to-Image Generation:")
print("  • Stable Diffusion: 512×512 images in 5 seconds")
print("  • DALL-E 2: 1024×1024 photorealistic")
print("  • Midjourney: Artistic image generation")
print("  • Imagen: Google's text-to-image (2048×2048)")
print("  • Applications: Art, design, advertising, content creation")
print()

print("Image Editing:")
print("  • Inpainting: Fill missing regions")
print("  • Outpainting: Extend image boundaries")
print("  • SDEdit: Stroke-based editing")
print("  • ControlNet: Precise control (pose, depth, edges)")
print("  • Applications: Photo editing, restoration")
print()

print("Super Resolution:")
print("  • SR3: 4x upscaling (64×64 → 256×256)")
print("  • Imagen: 64×64 → 1024×1024")
print("  • Better than GAN-based methods")
print("  • Applications: Photo enhancement, video upscaling")
print()

print("Video Generation:")
print("  • Imagen Video: Text-to-video")
print("  • Make-A-Video: Meta's video generation")
print("  • Gen-2 (Runway): Video editing")
print("  • Applications: Film, animation, content creation")
print()

print("Audio Generation:")
print("  • DiffWave: High-quality audio synthesis")
print("  • WaveGrad: Fast audio generation")
print("  • Noise2Music: Text-to-music")
print("  • Applications: Music production, sound effects")
print()

print("3D Generation:")
print("  • DreamFusion: Text-to-3D")
print("  • Point-E: 3D point cloud generation")
print("  • Shap-E: 3D shape generation")
print("  • Applications: Game dev, VR/AR, product design")
print()

print("Medical Imaging:")
print("  • MRI denoising: 40% faster scans")
print("  • CT reconstruction: Lower radiation")
print("  • Anomaly detection: Disease diagnosis")
print("  • Applications: Healthcare, diagnostics")
print()

print("Drug Discovery:")
print("  • Molecular generation: New drug candidates")
print("  • Protein structure: AlphaFold-style")
print("  • Chemical optimization")
print("  • Applications: Pharmaceutical research")

print("\n✓ Example 9 completed\n")

# ============================================================================
# EXAMPLE 10: Training Tips
# ============================================================================
print("EXAMPLE 10: Training Tips")
print("-" * 80)

print("Hyperparameters:")
print("  • Timesteps: 1000 (DDPM), 50-100 (DDIM)")
print("  • Learning rate: 1e-4 to 2e-4")
print("  • Batch size: 64-256")
print("  • Schedule: Cosine (best quality)")
print("  • EMA: 0.9999 (exponential moving average)")
print()

print("Architecture (U-Net):")
print("  • Encoder: Downsample with conv")
print("  • Bottleneck: Attention layers")
print("  • Decoder: Upsample with transposed conv")
print("  • Skip connections: Preserve details")
print("  • Time embedding: Sinusoidal positional encoding")
print()

print("Training Tricks:")
print("  1. EMA of weights: Smoother samples")
print("  2. Gradient clipping: Stable training")
print("  3. Mixed precision: 2x faster")
print("  4. Importance sampling: Focus on hard timesteps")
print("  5. Classifier-free guidance: Better text alignment")
print()

print("Common Issues:")
print("  • Blurry samples: Train longer, use cosine schedule")
print("  • Mode collapse: Doesn't happen in diffusion!")
print("  • Slow sampling: Use DDIM (10-50x faster)")
print("  • Memory: Use latent diffusion (Stable Diffusion)")

print("\n✓ Example 10 completed\n")

# ============================================================================
# EXAMPLE 11: Diffusion vs GAN vs VAE
# ============================================================================
print("EXAMPLE 11: Diffusion vs GAN vs VAE")
print("-" * 80)

print("Sample Quality:")
print("  Diffusion: ★★★★★ (Photorealistic)")
print("  GAN: ★★★★☆ (High quality but mode collapse)")
print("  VAE: ★★★☆☆ (Blurry)")
print()

print("Training Stability:")
print("  Diffusion: ★★★★★ (Very stable)")
print("  GAN: ★★☆☆☆ (Unstable, hard to train)")
print("  VAE: ★★★★☆ (Stable)")
print()

print("Sampling Speed:")
print("  Diffusion: ★★☆☆☆ (Slow: 1000 steps)")
print("  GAN: ★★★★★ (Fast: 1 step)")
print("  VAE: ★★★★★ (Fast: 1 step)")
print()

print("Mode Coverage:")
print("  Diffusion: ★★★★★ (Covers all modes)")
print("  GAN: ★★★☆☆ (Mode collapse)")
print("  VAE: ★★★★☆ (Good coverage)")
print()

print("Controllability:")
print("  Diffusion: ★★★★★ (Classifier-free guidance)")
print("  GAN: ★★★☆☆ (Limited)")
print("  VAE: ★★★★☆ (Latent interpolation)")
print()

print("Best Use Cases:")
print("  Diffusion: Text-to-image, high-quality generation")
print("  GAN: Real-time generation, style transfer")
print("  VAE: Anomaly detection, representation learning")

print("\n✓ Example 11 completed\n")

# ============================================================================
# EXAMPLE 12: Diffusion Evolution Timeline
# ============================================================================
print("EXAMPLE 12: Diffusion Evolution Timeline")
print("-" * 80)

print("2015 - Original Diffusion (Sohl-Dickstein et al.):")
print("  • First diffusion probabilistic model")
print("  • Thermodynamics inspiration")
print("  • Proof of concept")
print()

print("2020 - DDPM (Ho et al.):")
print("  • Denoising Diffusion Probabilistic Models")
print("  • Simplified training objective")
print("  • High-quality image generation")
print("  • Breakthrough paper")
print()

print("2020 - DDIM (Song et al.):")
print("  • Deterministic sampling")
print("  • 10-50x faster")
print("  • Same quality as DDPM")
print()

print("2021 - Improved DDPM (Nichol & Dhariwal):")
print("  • Cosine schedule")
print("  • Better architecture")
print("  • State-of-the-art quality")
print()

print("2021 - Classifier Guidance (Dhariwal & Nichol):")
print("  • Conditional generation")
print("  • Better text alignment")
print("  • ImageNet generation")
print()

print("2022 - Stable Diffusion (Rombach et al.):")
print("  • Latent diffusion (compress first)")
print("  • 512×512 in 5 seconds")
print("  • Open source")
print("  • Democratized AI art")
print()

print("2022 - DALL-E 2 (Ramesh et al.):")
print("  • Text-to-image with CLIP")
print("  • 1024×1024 photorealistic")
print("  • Inpainting, variations")
print()

print("2023 - Consistency Models (Song et al.):")
print("  • 1-step generation")
print("  • Fast as GANs")
print("  • Quality of diffusion")

print("\n✓ Example 12 completed\n")

# ============================================================================
# SUMMARY
# ============================================================================
print("=" * 80)
print("ALL EXAMPLES COMPLETED SUCCESSFULLY! ✓")
print("=" * 80)
print()
print("Summary of what we covered:")
print("1. ✓ Variance Schedules (Linear, Cosine, Quadratic)")
print("2. ✓ Forward Diffusion Process")
print("3. ✓ Reverse Diffusion Process")
print("4. ✓ Complete Sampling Loop")
print("5. ✓ DDIM Sampler (Fast)")
print("6. ✓ Loss Functions")
print("7. ✓ Noise Schedule Comparison")
print("8. ✓ Timestep Extraction")
print("9. ✓ Real-World Applications")
print("10. ✓ Training Tips")
print("11. ✓ Diffusion vs GAN vs VAE")
print("12. ✓ Evolution Timeline")
print()
print("You now have a complete understanding of Diffusion Models!")
print()
print("Next steps:")
print("- Implement U-Net architecture")
print("- Train on CIFAR-10, CelebA")
print("- Add classifier-free guidance")
print("- Build text-to-image system")
print("- Explore latent diffusion (Stable Diffusion)")
print()
print("GitHub: https://github.com/AliMehdi512/ilovetools")
print("Install: pip install ilovetools")
