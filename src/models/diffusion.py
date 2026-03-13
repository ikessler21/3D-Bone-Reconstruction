"""
DDPM / DDIM latent diffusion model for z_global space.

SinusoidalPositionEmbeddings — timestep positional encodings
DDPMDenoiser                 — U-Net-like MLP with cross-attention to conditioning
DDPM                         — forward/reverse diffusion process
DDIMSampler                  — fast deterministic 20-step sampler
LatentDiffusionModel         — combines denoiser + DDIM sampler
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor


# ---------------------------------------------------------------------------
# Sinusoidal timestep embeddings
# ---------------------------------------------------------------------------

class SinusoidalPositionEmbeddings(nn.Module):
    """
    Classic sinusoidal embeddings for diffusion timesteps.

    Maps integer timestep t → ℝ^dim.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: Tensor) -> Tensor:
        """
        Parameters
        ----------
        t : Tensor[B]   integer timesteps in [0, T-1]

        Returns
        -------
        Tensor[B, dim]
        """
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)  # [B, half_dim]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)   # [B, dim]
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


# ---------------------------------------------------------------------------
# Residual block with AdaLN and cross-attention
# ---------------------------------------------------------------------------

class AdaLNResidualBlock(nn.Module):
    """
    Residual block with Adaptive Layer Norm (AdaLN) conditioning on timestep.

    Architecture:
    x → LayerNorm (modulated by t_emb) → Linear → SiLU → Linear → residual

    Parameters
    ----------
    d_model   : int   feature dimension
    t_dim     : int   timestep embedding dimension
    """

    def __init__(self, d_model: int, t_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.t_proj = nn.Linear(t_dim, d_model * 2)   # scale + shift
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.act = nn.SiLU()
        self.linear2 = nn.Linear(d_model * 4, d_model)

    def forward(self, x: Tensor, t_emb: Tensor) -> Tensor:
        """
        x     : Tensor[B, d_model]
        t_emb : Tensor[B, t_dim]
        """
        scale_shift = self.t_proj(t_emb)               # [B, 2*d_model]
        scale, shift = scale_shift.chunk(2, dim=-1)     # each [B, d_model]
        normed = self.norm(x) * (1 + scale) + shift
        out = self.linear2(self.act(self.linear1(normed)))
        return x + out


class CrossAttentionLayer(nn.Module):
    """Single cross-attention layer: query from z_global, key/value from conditioning."""

    def __init__(self, d_model: int, cond_dim: int, n_heads: int = 4) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(cond_dim)
        # Project k/v to d_model if dims differ
        self.kv_proj = nn.Linear(cond_dim, d_model) if cond_dim != d_model else nn.Identity()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, x: Tensor, context: Tensor) -> Tensor:
        """
        x       : Tensor[B, d_model]        — query (z_global, treated as seq_len=1)
        context : Tensor[B, L, cond_dim]    — key/value (conditioning sequence)
        """
        q = self.norm_q(x).unsqueeze(1)         # [B, 1, d_model]
        kv = self.kv_proj(self.norm_kv(context)) # [B, L, d_model]
        attended, _ = self.attn(q, kv, kv)
        out = x + attended.squeeze(1)
        out = out + self.ff(self.norm2(out))
        return out


# ---------------------------------------------------------------------------
# DDPM Denoiser
# ---------------------------------------------------------------------------

class DDPMDenoiser(nn.Module):
    """
    U-Net-like MLP operating on z_global ∈ ℝ^256.

    Architecture (4 residual blocks, cross-attention every 2 blocks):

    z_global [B,D] → input_proj
        → ResBlock(D, t_dim)           [block 1]
        → ResBlock(D, t_dim)           [block 2]
        → CrossAttention(D, cond_dim)  ← conditioning
        → ResBlock(D, t_dim)           [block 3]
        → ResBlock(D, t_dim)           [block 4]
        → CrossAttention(D, cond_dim)
        → output_proj → ε_pred [B, D]

    Parameters
    ----------
    z_dim    : int    latent dimension
    t_dim    : int    timestep embedding dimension
    cond_dim : int    conditioning sequence token dimension
    cond_seq_len : int  conditioning sequence length (or 1 for global conditioning)
    """

    def __init__(
        self,
        z_dim: int = 256,
        t_dim: int = 128,
        cond_dim: int = 256,
        n_heads: int = 4,
    ) -> None:
        super().__init__()
        self.z_dim = z_dim

        # Timestep embedding: integer → sinusoidal → MLP
        self.t_emb = nn.Sequential(
            SinusoidalPositionEmbeddings(t_dim),
            nn.Linear(t_dim, t_dim * 4),
            nn.SiLU(),
            nn.Linear(t_dim * 4, t_dim),
        )

        self.input_proj = nn.Linear(z_dim, z_dim)

        # 4 residual blocks
        self.block1 = AdaLNResidualBlock(z_dim, t_dim)
        self.block2 = AdaLNResidualBlock(z_dim, t_dim)
        self.cross_attn1 = CrossAttentionLayer(z_dim, cond_dim, n_heads)
        self.block3 = AdaLNResidualBlock(z_dim, t_dim)
        self.block4 = AdaLNResidualBlock(z_dim, t_dim)
        self.cross_attn2 = CrossAttentionLayer(z_dim, cond_dim, n_heads)

        self.output_norm = nn.LayerNorm(z_dim)
        self.output_proj = nn.Linear(z_dim, z_dim)

    def forward(
        self,
        z_t: Tensor,               # [B, z_dim]   noisy latent
        t: Tensor,                 # [B]           integer timestep
        conditioning: Optional[Tensor] = None,  # [B, L, cond_dim] or [B, cond_dim]
    ) -> Tensor:
        """
        Returns predicted noise ε_θ(z_t, t, c) ∈ ℝ^[B, z_dim].
        """
        # Build conditioning context
        if conditioning is None:
            # Null conditioning: zero sequence of length 1
            context = torch.zeros(z_t.shape[0], 1, self.cross_attn1.kv_proj.in_features
                                  if hasattr(self.cross_attn1.kv_proj, 'in_features')
                                  else z_t.shape[-1],
                                  device=z_t.device)
        elif conditioning.dim() == 2:
            context = conditioning.unsqueeze(1)  # [B, 1, cond_dim]
        else:
            context = conditioning               # [B, L, cond_dim]

        # Timestep embedding
        t_emb = self.t_emb(t)   # [B, t_dim]

        # Forward
        h = self.input_proj(z_t)
        h = self.block1(h, t_emb)
        h = self.block2(h, t_emb)
        h = self.cross_attn1(h, context)
        h = self.block3(h, t_emb)
        h = self.block4(h, t_emb)
        h = self.cross_attn2(h, context)
        h = self.output_proj(self.output_norm(h))
        return h


# ---------------------------------------------------------------------------
# DDPM (noise schedule + forward/reverse process)
# ---------------------------------------------------------------------------

class DDPM(nn.Module):
    """
    Denoising Diffusion Probabilistic Model over z_global latent space.

    Uses a linear beta schedule with T=100 steps.

    Parameters
    ----------
    denoiser  : DDPMDenoiser
    T         : int    total diffusion steps
    beta_start: float  starting beta
    beta_end  : float  ending beta
    """

    def __init__(
        self,
        denoiser: DDPMDenoiser,
        T: int = 100,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ) -> None:
        super().__init__()
        self.denoiser = denoiser
        self.T = T

        # Linear beta schedule
        betas = torch.linspace(beta_start, beta_end, T)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # Register buffers (non-trainable tensors moved with the module)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", alphas_cumprod.sqrt())
        self.register_buffer("sqrt_one_minus_alphas_cumprod", (1 - alphas_cumprod).sqrt())
        self.register_buffer(
            "posterior_variance",
            betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod),
        )

    def forward_process(self, z0: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Sample z_t from q(z_t | z_0) = N(sqrt_ā_t * z0, (1-ā_t)*I).

        Parameters
        ----------
        z0 : Tensor[B, D]   clean latent
        t  : Tensor[B]      integer timestep in [0, T-1]

        Returns
        -------
        z_t    : Tensor[B, D]   noisy latent
        epsilon: Tensor[B, D]   noise added
        """
        eps = torch.randn_like(z0)
        sqrt_alpha = self.sqrt_alphas_cumprod[t].unsqueeze(-1)          # [B, 1]
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)  # [B, 1]
        z_t = sqrt_alpha * z0 + sqrt_one_minus * eps
        return z_t, eps

    def reverse_step(
        self,
        z_t: Tensor,
        t: Tensor,
        eps_pred: Tensor,
    ) -> Tensor:
        """
        Compute z_{t-1} from z_t and predicted noise (DDPM posterior).

        Parameters
        ----------
        z_t      : Tensor[B, D]
        t        : Tensor[B]     integer timestep
        eps_pred : Tensor[B, D]  predicted noise

        Returns
        -------
        z_{t-1}  : Tensor[B, D]
        """
        betas_t = self.betas[t].unsqueeze(-1)
        sqrt_one_minus_t = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)
        sqrt_recip_alphas = (1.0 / self.alphas[t].sqrt()).unsqueeze(-1)

        # Predicted mean
        mu = sqrt_recip_alphas * (z_t - betas_t / sqrt_one_minus_t * eps_pred)

        # Add posterior variance noise (only for t > 0)
        posterior_var = self.posterior_variance[t].unsqueeze(-1)
        noise = torch.randn_like(z_t)
        # Mask: don't add noise at t=0
        mask = (t > 0).float().unsqueeze(-1)
        return mu + mask * posterior_var.sqrt() * noise

    def compute_loss(
        self,
        z0: Tensor,
        conditioning: Optional[Tensor] = None,
    ) -> Tensor:
        """
        DDPM training loss: E[||ε - ε_θ(z_t, t, c)||²].

        Parameters
        ----------
        z0           : Tensor[B, D]
        conditioning : Tensor[B, L, D] or [B, D] or None

        Returns
        -------
        loss : scalar Tensor
        """
        B = z0.shape[0]
        t = torch.randint(0, self.T, (B,), device=z0.device)
        z_t, eps = self.forward_process(z0, t)
        eps_pred = self.denoiser(z_t, t, conditioning)
        return F.mse_loss(eps_pred, eps)


# ---------------------------------------------------------------------------
# DDIM Sampler
# ---------------------------------------------------------------------------

class DDIMSampler:
    """
    Deterministic DDIM sampling from a trained DDPM.

    Implements the DDIM update rule (Song et al., ICLR 2021):
        z_{t-1} = sqrt(ā_{t-1}) * z0_pred + sqrt(1-ā_{t-1}) * direction

    where:
        z0_pred = (z_t - sqrt(1-ā_t) * ε_θ) / sqrt(ā_t)
        direction = sqrt(1-ā_{t-1}-σ²) * ε_θ  (with σ=0 for deterministic DDIM)

    Parameters
    ----------
    ddpm        : DDPM   trained DDPM model
    num_steps   : int    number of DDIM steps (default 20)
    """

    def __init__(self, ddpm: DDPM, num_steps: int = 20) -> None:
        self.ddpm = ddpm
        self.num_steps = num_steps

        # Sub-sequence of timesteps (uniform spacing)
        T = ddpm.T
        step = T // num_steps
        self.timesteps = list(range(0, T, step))[::-1]   # e.g. [99, 94, ..., 4, 0]
        # Ensure we cover t=T-1 down to t=0
        if self.timesteps[0] < T - 1:
            self.timesteps.insert(0, T - 1)
        self.timesteps = self.timesteps[:num_steps]

    @torch.no_grad()
    def sample(
        self,
        shape: Tuple[int, ...],
        conditioning: Optional[Tensor] = None,
        guidance_scale: float = 3.0,
        device: Optional[torch.device] = None,
    ) -> Tensor:
        """
        Sample z0 from p_θ(z0 | c) using DDIM.

        Parameters
        ----------
        shape           : (B, D) — shape of the latent to sample
        conditioning    : Tensor[B, ...] or None
        guidance_scale  : float   classifier-free guidance scale
        device          : torch.device

        Returns
        -------
        z0 : Tensor[shape]   denoised latent
        """
        if device is None:
            device = next(self.ddpm.parameters()).device

        z_t = torch.randn(*shape, device=device)

        use_cfg = (conditioning is not None) and (guidance_scale != 1.0)

        for i, t_val in enumerate(self.timesteps):
            t = torch.full((shape[0],), t_val, dtype=torch.long, device=device)

            # Predict noise
            eps_cond = self.ddpm.denoiser(z_t, t, conditioning)

            if use_cfg:
                eps_uncond = self.ddpm.denoiser(z_t, t, None)
                eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
            else:
                eps = eps_cond

            # Determine t_{t-1}
            t_prev_val = self.timesteps[i + 1] if i + 1 < len(self.timesteps) else -1

            # DDIM update
            alpha_t = self.ddpm.alphas_cumprod[t_val]
            alpha_prev = (
                self.ddpm.alphas_cumprod[t_prev_val]
                if t_prev_val >= 0
                else torch.ones(1, device=device)
            )

            # Predicted z0
            z0_pred = (z_t - (1 - alpha_t).sqrt() * eps) / alpha_t.sqrt().clamp(min=1e-8)
            # Clamp to avoid extreme values
            z0_pred = z0_pred.clamp(-5, 5)

            # DDIM direction (σ=0 for deterministic)
            direction = (1 - alpha_prev).sqrt() * eps
            z_t = alpha_prev.sqrt() * z0_pred + direction

        return z_t


# ---------------------------------------------------------------------------
# Latent Diffusion Model
# ---------------------------------------------------------------------------

class LatentDiffusionModel(nn.Module):
    """
    Wraps DDPMDenoiser + DDPM + DDIMSampler into a single module.

    Parameters
    ----------
    config : dict   diffusion configuration block
    """

    def __init__(self, config: dict) -> None:
        super().__init__()
        z_dim = config.get("z_global_dim", 256)
        t_dim = config.get("t_emb_dim", 128)
        cond_dim = config.get("cond_dim", 256)
        T = config.get("T", 100)
        ddim_steps = config.get("ddim_steps", 20)
        beta_start = config.get("beta_start", 1e-4)
        beta_end = config.get("beta_end", 0.02)

        self.denoiser = DDPMDenoiser(
            z_dim=z_dim,
            t_dim=t_dim,
            cond_dim=cond_dim,
        )
        self.ddpm = DDPM(
            denoiser=self.denoiser,
            T=T,
            beta_start=beta_start,
            beta_end=beta_end,
        )
        self.sampler = DDIMSampler(self.ddpm, num_steps=ddim_steps)
        self.z_dim = z_dim

    def compute_loss(
        self,
        z0: Tensor,
        conditioning: Optional[Tensor] = None,
    ) -> Tensor:
        """Training loss: DDPM ELBO (ε-matching)."""
        return self.ddpm.compute_loss(z0, conditioning)

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        conditioning: Optional[Tensor] = None,
        guidance_scale: float = 3.0,
        device: Optional[torch.device] = None,
    ) -> Tensor:
        """
        Sample `batch_size` latents from the diffusion prior.

        Returns Tensor[batch_size, z_dim].
        """
        shape = (batch_size, self.z_dim)
        return self.sampler.sample(
            shape, conditioning=conditioning,
            guidance_scale=guidance_scale, device=device,
        )
