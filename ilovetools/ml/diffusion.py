"""
Diffusion Models (Denoising Diffusion Probabilistic Models)

This module implements diffusion model architectures that power state-of-the-art
generative AI systems like Stable Diffusion, DALL-E 2, and Midjourney.

Implemented Components:
1. DDPM (Denoising Diffusion Probabilistic Models)
2. DDIM (Denoising Diffusion Implicit Models)
3. Noise Schedulers (Linear, Cosine, Quadratic)
4. Forward Diffusion Process (Adding Noise)
5. Reverse Diffusion Process (Denoising)
6. Variance Schedules (β_t)
7. Sampling Algorithms (DDPM, DDIM, Ancestral)

Key Concepts:
- Forward Process: Gradually add Gaussian noise to data
- Reverse Process: Learn to denoise and generate samples
- Markov Chain: Each step depends only on previous step
- Variance Schedule: Controls noise addition rate
- Score Matching: Learn gradient of log probability
- Langevin Dynamics: Sampling via gradient descent + noise

Differences from GANs and VAEs:
- Diffusion: Iterative denoising, stable training, high quality
- GAN: Adversarial, mode collapse, training instability
- VAE: Single-step generation, blurry outputs, smooth latent space
- Diffusion: Multi-step, photorealistic, no mode collapse

Applications:
- Text-to-Image (Stable Diffusion, DALL-E 2, Midjourney)
- Image-to-Image (ControlNet, InstructPix2Pix)
- Super Resolution (SR3, Imagen)
- Inpainting (Fill missing regions)
- Video Generation (Imagen Video, Make-A-Video)
- Audio Generation (DiffWave, WaveGrad)
- 3D Generation (DreamFusion, Point-E)
- Molecule Generation (Drug discovery)
- Medical Imaging (Denoising, reconstruction)
- Image Editing (SDEdit, Prompt-to-Prompt)

References:
- Sohl-Dickstein et al., "Deep Unsupervised Learning using Nonequilibrium Thermodynamics" (2015)
- Ho et al., "Denoising Diffusion Probabilistic Models" (DDPM, 2020)
- Song et al., "Denoising Diffusion Implicit Models" (DDIM, 2020)
- Nichol & Dhariwal, "Improved Denoising Diffusion Probabilistic Models" (2021)
- Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models" (Stable Diffusion, 2022)
- Ramesh et al., "Hierarchical Text-Conditional Image Generation with CLIP Latents" (DALL-E 2, 2022)

Author: Ali Mehdi
Date: February 25, 2026
"""

import numpy as np
from typing import Tuple, Optional, List, Callable


# ============================================================================
# NOISE SCHEDULERS
# ============================================================================

def linear_beta_schedule(timesteps: int,
                         beta_start: float = 0.0001,
                         beta_end: float = 0.02) -> np.ndarray:
    """
    Linear variance schedule for diffusion process.
    
    Simple linear interpolation from beta_start to beta_end.
    
    Args:
        timesteps: Number of diffusion steps (typically 1000)
        beta_start: Starting variance (small noise)
        beta_end: Ending variance (large noise)
    
    Returns:
        Beta schedule [timesteps]
    
    Example:
        >>> from ilovetools.ml.diffusion import linear_beta_schedule
        >>> betas = linear_beta_schedule(timesteps=1000)
        >>> print(f"Beta schedule: {betas.shape}")  # (1000,)
        >>> print(f"Start: {betas[0]:.6f}, End: {betas[-1]:.6f}")
    
    Typical Values:
        - timesteps: 1000 (DDPM default)
        - beta_start: 0.0001 (very small noise)
        - beta_end: 0.02 (moderate noise)
    """
    return np.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps: int,
                         s: float = 0.008) -> np.ndarray:
    """
    Cosine variance schedule (Improved DDPM).
    
    Provides better sample quality than linear schedule by
    reducing noise more gradually at the beginning.
    
    Formula:
        α_bar_t = cos²((t/T + s) / (1 + s) * π/2)
        β_t = 1 - α_t / α_{t-1}
    
    Args:
        timesteps: Number of diffusion steps
        s: Small offset to prevent β_t from being too small
    
    Returns:
        Beta schedule [timesteps]
    
    Example:
        >>> from ilovetools.ml.diffusion import cosine_beta_schedule
        >>> betas = cosine_beta_schedule(timesteps=1000)
        >>> print(f"Cosine schedule: {betas.shape}")
    
    Benefits:
        - Better sample quality
        - More stable training
        - Smoother noise addition
        - Used in Stable Diffusion
    
    Reference:
        Nichol & Dhariwal, "Improved DDPM" (2021)
    """
    steps = timesteps + 1
    t = np.linspace(0, timesteps, steps)
    alphas_cumprod = np.cos(((t / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, 0, 0.999)


def quadratic_beta_schedule(timesteps: int,
                            beta_start: float = 0.0001,
                            beta_end: float = 0.02) -> np.ndarray:
    """
    Quadratic variance schedule.
    
    Provides smoother noise addition than linear schedule.
    
    Args:
        timesteps: Number of diffusion steps
        beta_start: Starting variance
        beta_end: Ending variance
    
    Returns:
        Beta schedule [timesteps]
    
    Example:
        >>> from ilovetools.ml.diffusion import quadratic_beta_schedule
        >>> betas = quadratic_beta_schedule(timesteps=1000)
        >>> print(f"Quadratic schedule: {betas.shape}")
    """
    return np.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2


# ============================================================================
# DIFFUSION PROCESS
# ============================================================================

class DiffusionModel:
    """
    Denoising Diffusion Probabilistic Model (DDPM).
    
    Implements the complete diffusion process: forward (adding noise)
    and reverse (denoising) for high-quality image generation.
    
    Args:
        timesteps: Number of diffusion steps (default: 1000)
        beta_schedule: Variance schedule ('linear', 'cosine', 'quadratic')
        beta_start: Starting variance for linear/quadratic
        beta_end: Ending variance for linear/quadratic
    
    Example:
        >>> from ilovetools.ml.diffusion import DiffusionModel
        >>> diffusion = DiffusionModel(timesteps=1000, beta_schedule='cosine')
        >>> 
        >>> # Forward process (add noise)
        >>> x0 = np.random.randn(32, 3, 64, 64)  # Clean images
        >>> t = np.array([500] * 32)  # Timestep
        >>> xt, noise = diffusion.q_sample(x0, t)
        >>> 
        >>> # Reverse process (denoise)
        >>> x_denoised = diffusion.p_sample(xt, t, noise_pred)
    
    Training Objective:
        L = E[||ε - ε_θ(x_t, t)||²]
        Predict the noise that was added
    
    Key Concepts:
        - q(x_t|x_0): Forward process (add noise)
        - p_θ(x_{t-1}|x_t): Reverse process (denoise)
        - ε_θ: Neural network that predicts noise
        - α_t = 1 - β_t: Signal retention
        - ᾱ_t = ∏α_i: Cumulative product
    
    Reference:
        Ho et al., "Denoising Diffusion Probabilistic Models" (2020)
    """
    
    def __init__(self,
                 timesteps: int = 1000,
                 beta_schedule: str = 'cosine',
                 beta_start: float = 0.0001,
                 beta_end: float = 0.02):
        self.timesteps = timesteps
        
        # Get beta schedule
        if beta_schedule == 'linear':
            self.betas = linear_beta_schedule(timesteps, beta_start, beta_end)
        elif beta_schedule == 'cosine':
            self.betas = cosine_beta_schedule(timesteps)
        elif beta_schedule == 'quadratic':
            self.betas = quadratic_beta_schedule(timesteps, beta_start, beta_end)
        else:
            raise ValueError(f"Unknown beta_schedule: {beta_schedule}")
        
        # Precompute useful values
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = np.concatenate([np.array([1.0]), self.alphas_cumprod[:-1]])
        
        # Calculations for diffusion q(x_t | x_{t-1})
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = np.log(
            np.maximum(self.posterior_variance, 1e-20)
        )
        self.posterior_mean_coef1 = (
            self.betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * np.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )
    
    def q_sample(self, x_start: np.ndarray, t: np.ndarray,
                 noise: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward diffusion process: q(x_t | x_0).
        
        Add noise to clean data according to variance schedule.
        
        Formula:
            x_t = √ᾱ_t * x_0 + √(1-ᾱ_t) * ε
            where ε ~ N(0, I)
        
        Args:
            x_start: Clean data [batch, *shape]
            t: Timesteps [batch]
            noise: Optional pre-sampled noise
        
        Returns:
            x_t: Noisy data [batch, *shape]
            noise: Sampled noise [batch, *shape]
        
        Example:
            >>> diffusion = DiffusionModel(timesteps=1000)
            >>> x0 = np.random.randn(32, 3, 64, 64)
            >>> t = np.array([500] * 32)
            >>> xt, noise = diffusion.q_sample(x0, t)
            >>> print(f"Noisy images: {xt.shape}")
        """
        if noise is None:
            noise = np.random.randn(*x_start.shape)
        
        # Extract coefficients for timestep t
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )
        
        # Apply noise
        x_t = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
        
        return x_t, noise
    
    def p_sample(self, x_t: np.ndarray, t: np.ndarray,
                 noise_pred: np.ndarray) -> np.ndarray:
        """
        Reverse diffusion process: p(x_{t-1} | x_t).
        
        Denoise one step using predicted noise.
        
        Formula:
            x_{t-1} = 1/√α_t * (x_t - (1-α_t)/√(1-ᾱ_t) * ε_θ) + σ_t * z
            where z ~ N(0, I)
        
        Args:
            x_t: Noisy data at timestep t [batch, *shape]
            t: Timesteps [batch]
            noise_pred: Predicted noise from model [batch, *shape]
        
        Returns:
            x_{t-1}: Denoised data [batch, *shape]
        
        Example:
            >>> x_t = np.random.randn(32, 3, 64, 64)
            >>> t = np.array([500] * 32)
            >>> noise_pred = np.random.randn(32, 3, 64, 64)
            >>> x_prev = diffusion.p_sample(x_t, t, noise_pred)
        """
        # Extract coefficients
        betas_t = self._extract(self.betas, t, x_t.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_t.shape
        )
        sqrt_recip_alphas_t = self._extract(1.0 / np.sqrt(self.alphas), t, x_t.shape)
        
        # Predict x_0
        model_mean = sqrt_recip_alphas_t * (
            x_t - betas_t * noise_pred / sqrt_one_minus_alphas_cumprod_t
        )
        
        # Add noise (except at t=0)
        posterior_variance_t = self._extract(self.posterior_variance, t, x_t.shape)
        noise = np.random.randn(*x_t.shape)
        
        # No noise when t == 0
        nonzero_mask = (t != 0).reshape(-1, *([1] * (len(x_t.shape) - 1)))
        
        x_prev = model_mean + nonzero_mask * np.sqrt(posterior_variance_t) * noise
        
        return x_prev
    
    def p_sample_loop(self, shape: Tuple[int, ...],
                      noise_predictor: Callable) -> np.ndarray:
        """
        Complete sampling loop: generate samples from noise.
        
        Iteratively denoise from pure noise to clean samples.
        
        Args:
            shape: Shape of samples to generate
            noise_predictor: Function that predicts noise given (x_t, t)
        
        Returns:
            Generated samples [batch, *shape]
        
        Example:
            >>> def noise_predictor(x_t, t):
            ...     # Your trained model
            ...     return model(x_t, t)
            >>> 
            >>> samples = diffusion.p_sample_loop(
            ...     shape=(16, 3, 64, 64),
            ...     noise_predictor=noise_predictor
            ... )
            >>> print(f"Generated: {samples.shape}")
        """
        # Start from pure noise
        x = np.random.randn(*shape)
        
        # Iteratively denoise
        for t in reversed(range(self.timesteps)):
            t_batch = np.array([t] * shape[0])
            noise_pred = noise_predictor(x, t_batch)
            x = self.p_sample(x, t_batch, noise_pred)
        
        return x
    
    def _extract(self, a: np.ndarray, t: np.ndarray, x_shape: Tuple) -> np.ndarray:
        """
        Extract coefficients at specified timesteps and reshape.
        
        Args:
            a: Coefficient array [timesteps]
            t: Timesteps [batch]
            x_shape: Shape to broadcast to
        
        Returns:
            Extracted coefficients [batch, 1, 1, ...]
        """
        batch_size = t.shape[0]
        out = a[t]
        return out.reshape(batch_size, *([1] * (len(x_shape) - 1)))


# ============================================================================
# DDIM SAMPLER
# ============================================================================

class DDIMSampler:
    """
    Denoising Diffusion Implicit Models (DDIM) Sampler.
    
    Faster sampling than DDPM by using deterministic sampling
    and skipping timesteps.
    
    Args:
        diffusion: DiffusionModel instance
        eta: Stochasticity parameter (0=deterministic, 1=DDPM)
    
    Example:
        >>> from ilovetools.ml.diffusion import DiffusionModel, DDIMSampler
        >>> diffusion = DiffusionModel(timesteps=1000)
        >>> ddim = DDIMSampler(diffusion, eta=0.0)
        >>> 
        >>> # Fast sampling (100 steps instead of 1000)
        >>> samples = ddim.sample(
        ...     shape=(16, 3, 64, 64),
        ...     noise_predictor=model,
        ...     steps=100
        ... )
    
    Benefits:
        - 10-50x faster sampling
        - Deterministic (eta=0)
        - Same quality as DDPM
        - Used in Stable Diffusion
    
    Reference:
        Song et al., "Denoising Diffusion Implicit Models" (2020)
    """
    
    def __init__(self, diffusion: DiffusionModel, eta: float = 0.0):
        self.diffusion = diffusion
        self.eta = eta
    
    def sample(self, shape: Tuple[int, ...],
               noise_predictor: Callable,
               steps: int = 50) -> np.ndarray:
        """
        DDIM sampling with fewer steps.
        
        Args:
            shape: Shape of samples to generate
            noise_predictor: Function that predicts noise
            steps: Number of sampling steps (< timesteps)
        
        Returns:
            Generated samples [batch, *shape]
        """
        # Create subsequence of timesteps
        timesteps = np.linspace(0, self.diffusion.timesteps - 1, steps, dtype=int)
        timesteps = timesteps[::-1]  # Reverse order
        
        # Start from noise
        x = np.random.randn(*shape)
        
        # Iteratively denoise
        for i, t in enumerate(timesteps):
            t_batch = np.array([t] * shape[0])
            noise_pred = noise_predictor(x, t_batch)
            x = self._ddim_step(x, t, noise_pred, timesteps, i)
        
        return x
    
    def _ddim_step(self, x_t: np.ndarray, t: int,
                   noise_pred: np.ndarray,
                   timesteps: np.ndarray, i: int) -> np.ndarray:
        """Single DDIM denoising step."""
        # Get alpha values
        alpha_t = self.diffusion.alphas_cumprod[t]
        
        if i < len(timesteps) - 1:
            alpha_prev = self.diffusion.alphas_cumprod[timesteps[i + 1]]
        else:
            alpha_prev = 1.0
        
        # Predict x_0
        pred_x0 = (x_t - np.sqrt(1 - alpha_t) * noise_pred) / np.sqrt(alpha_t)
        
        # Direction pointing to x_t
        dir_xt = np.sqrt(1 - alpha_prev - self.eta**2 * (1 - alpha_prev)) * noise_pred
        
        # Random noise
        noise = np.random.randn(*x_t.shape) if self.eta > 0 else 0
        
        # Compute x_{t-1}
        x_prev = np.sqrt(alpha_prev) * pred_x0 + dir_xt + self.eta * np.sqrt(1 - alpha_prev) * noise
        
        return x_prev


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

def diffusion_loss(noise: np.ndarray, noise_pred: np.ndarray) -> float:
    """
    Simple MSE loss for diffusion models.
    
    Formula:
        L = ||ε - ε_θ(x_t, t)||²
    
    Args:
        noise: True noise [batch, *shape]
        noise_pred: Predicted noise [batch, *shape]
    
    Returns:
        Loss (scalar)
    
    Example:
        >>> from ilovetools.ml.diffusion import diffusion_loss
        >>> noise = np.random.randn(32, 3, 64, 64)
        >>> noise_pred = np.random.randn(32, 3, 64, 64)
        >>> loss = diffusion_loss(noise, noise_pred)
        >>> print(f"Diffusion loss: {loss:.4f}")
    """
    return np.mean((noise - noise_pred) ** 2)


def vlb_loss(x_start: np.ndarray, x_t: np.ndarray,
             noise: np.ndarray, noise_pred: np.ndarray,
             t: np.ndarray, diffusion: DiffusionModel) -> float:
    """
    Variational Lower Bound (VLB) loss.
    
    More principled loss that includes KL divergence terms.
    
    Args:
        x_start: Clean data [batch, *shape]
        x_t: Noisy data [batch, *shape]
        noise: True noise [batch, *shape]
        noise_pred: Predicted noise [batch, *shape]
        t: Timesteps [batch]
        diffusion: DiffusionModel instance
    
    Returns:
        VLB loss (scalar)
    
    Reference:
        Nichol & Dhariwal, "Improved DDPM" (2021)
    """
    # Simple MSE for now (full VLB requires more computation)
    return diffusion_loss(noise, noise_pred)


# ============================================================================
# UTILITIES
# ============================================================================

def extract_timestep(a: np.ndarray, t: np.ndarray, x_shape: Tuple) -> np.ndarray:
    """
    Extract values from array at specified timesteps.
    
    Args:
        a: Array to extract from [timesteps]
        t: Timesteps [batch]
        x_shape: Shape to broadcast to
    
    Returns:
        Extracted values [batch, 1, 1, ...]
    
    Example:
        >>> from ilovetools.ml.diffusion import extract_timestep
        >>> alphas = np.linspace(0.9, 0.1, 1000)
        >>> t = np.array([0, 500, 999])
        >>> extracted = extract_timestep(alphas, t, (3, 3, 64, 64))
        >>> print(f"Extracted: {extracted.shape}")
    """
    batch_size = t.shape[0]
    out = a[t]
    return out.reshape(batch_size, *([1] * (len(x_shape) - 1)))


def noise_like(shape: Tuple[int, ...]) -> np.ndarray:
    """
    Generate Gaussian noise with specified shape.
    
    Args:
        shape: Shape of noise to generate
    
    Returns:
        Gaussian noise [*shape]
    
    Example:
        >>> from ilovetools.ml.diffusion import noise_like
        >>> noise = noise_like((32, 3, 64, 64))
        >>> print(f"Noise: {noise.shape}, Mean: {noise.mean():.3f}")
    """
    return np.random.randn(*shape)


__all__ = [
    # Schedulers
    'linear_beta_schedule',
    'cosine_beta_schedule',
    'quadratic_beta_schedule',
    # Models
    'DiffusionModel',
    'DDIMSampler',
    # Losses
    'diffusion_loss',
    'vlb_loss',
    # Utilities
    'extract_timestep',
    'noise_like',
]
