"""
Hierarchical VAE — Stage 1 of RELIC.

HierarchicalVAE — encoder → (z_global, z_local) with reparameterisation
VAELoss         — Chamfer L1 + beta-annealed KL divergence
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from src.models.encoder import GeometryAwareEncoder
from src.models.decoder import HierarchicalTransformerDecoder
from src.models.symmetry import ConfidenceGatedSymmetryModule


# ---------------------------------------------------------------------------
# Hierarchical VAE
# ---------------------------------------------------------------------------

class HierarchicalVAE(nn.Module):
    """
    Stage 1: Hierarchical VAE for bone point cloud reconstruction.

    Encoder: GeometryAwareEncoder → z_global_mu, z_global_logvar ∈ ℝ^[B,256]
             also produces z_local ∈ ℝ^[B,K,d]
    Decoder: HierarchicalTransformerDecoder → complete point cloud [B,M,3]

    Training loss: Chamfer-L1 + β * KL (β annealed with KL-annealing schedule)

    Parameters
    ----------
    config : dict
        VAE configuration block (see experiments/relic_full.yaml).
    """

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.z_global_dim = config.get("z_global_dim", 256)
        self.z_local_dim = config.get("z_local_dim", 64)
        self.k_local = config.get("k_local", 128)
        self.n_points_out = config.get("n_points_out", 4096)
        self.use_symmetry = config.get("use_symmetry", True)
        self.symmetry_threshold = config.get("confidence_threshold", 0.25)

        d_model = config.get("d_model", 256)
        n_proxies = config.get("n_proxies", 256)
        k_local_enc = config.get("k_local_enc", 16)

        # Encoder backbone
        self.encoder = GeometryAwareEncoder(
            d_model=d_model,
            out_dim=d_model,
            n_proxies=n_proxies,
            k_local=k_local_enc,
            n_proxy_layers=config.get("n_proxy_layers", 2),
            n_global_layers=config.get("n_global_layers", 4),
        )

        # Symmetry module (optional)
        if self.use_symmetry:
            self.symmetry_module = ConfidenceGatedSymmetryModule(
                d_model=d_model,
                confidence_threshold=self.symmetry_threshold,
                n_ransac=config.get("n_ransac", 100),
            )
        else:
            self.symmetry_module = None

        # VAE head: map encoder output → mu + logvar
        self.mu_head = nn.Linear(d_model, self.z_global_dim)
        self.logvar_head = nn.Linear(d_model, self.z_global_dim)

        # Project z_local dim from d_model to z_local_dim
        self.local_proj = nn.Linear(d_model, self.z_local_dim)

        # Decoder
        self.decoder = HierarchicalTransformerDecoder(
            z_global_dim=self.z_global_dim,
            z_local_dim=self.z_local_dim,
            k_local=n_proxies,
            cond_dim=config.get("cond_dim", 256),
            n_points_out=self.n_points_out,
        )

    def encode(
        self,
        partial_points: Tensor,
        conditioning: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, dict]:
        """
        Encode partial_points to (z_global, z_local, mu, logvar, sym_info).

        Parameters
        ----------
        partial_points : Tensor[B, N, 3]
        conditioning   : Tensor[B, C] or None

        Returns
        -------
        z_global  : Tensor[B, z_global_dim]   reparameterised global latent
        z_local   : Tensor[B, K, z_local_dim]  projected local features
        mu        : Tensor[B, z_global_dim]
        logvar    : Tensor[B, z_global_dim]
        sym_info  : dict
        """
        z_enc, proxy_feats, proxy_coords = self.encoder(partial_points)

        sym_info = {}
        if self.symmetry_module is not None:
            z_enc, proxy_feats, sym_info = self.symmetry_module(
                partial_points, z_enc, proxy_feats, proxy_coords
            )

        mu = self.mu_head(z_enc)           # [B, z_global_dim]
        logvar = self.logvar_head(z_enc)   # [B, z_global_dim]

        z_global = self.reparameterize(mu, logvar)   # [B, z_global_dim]
        z_local = self.local_proj(proxy_feats)        # [B, K, z_local_dim]

        return z_global, z_local, mu, logvar, sym_info

    @staticmethod
    def reparameterize(mu: Tensor, logvar: Tensor) -> Tensor:
        """Reparameterisation trick: z = mu + eps * sigma."""
        if not mu.requires_grad and not logvar.requires_grad:
            # Deterministic at inference if desired — still sample to get diversity
            pass
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(
        self,
        z_global: Tensor,
        z_local: Tensor,
        conditioning: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Decode (z_global, z_local) → complete point cloud [B, M, 3].
        """
        return self.decoder(z_global, z_local, conditioning)

    def forward(
        self,
        partial: Tensor,
        full: Optional[Tensor] = None,
        conditioning: Optional[Tensor] = None,
    ) -> dict:
        """
        Forward pass.

        Parameters
        ----------
        partial      : Tensor[B, N, 3]
        full         : Tensor[B, M, 3] or None (for inference)
        conditioning : Tensor[B, C] or None

        Returns
        -------
        dict with keys: pred, mu, logvar, sym_info
        """
        z_global, z_local, mu, logvar, sym_info = self.encode(partial, conditioning)
        pred = self.decode(z_global, z_local, conditioning)
        return {
            "pred": pred,
            "mu": mu,
            "logvar": logvar,
            "z_global": z_global,
            "z_local": z_local,
            "sym_info": sym_info,
        }


# ---------------------------------------------------------------------------
# VAE Loss
# ---------------------------------------------------------------------------

class VAELoss(nn.Module):
    """
    Combined VAE loss: Chamfer-L1 + β * KL.

    Supports β annealing via a monotonic schedule (warm-up from 0 to beta_max).

    Parameters
    ----------
    beta_max      : float   max KL weight
    beta_warmup   : int     number of training steps to ramp beta from 0 to beta_max
    cd_weight     : float   Chamfer-L1 weight
    fidelity_weight: float  partial fidelity loss weight
    """

    def __init__(
        self,
        beta_max: float = 0.01,
        beta_warmup: int = 5000,
        cd_weight: float = 1.0,
        fidelity_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.beta_max = beta_max
        self.beta_warmup = beta_warmup
        self.cd_weight = cd_weight
        self.fidelity_weight = fidelity_weight
        self._step = 0

    @property
    def beta(self) -> float:
        """Current KL weight (linearly annealed)."""
        return self.beta_max * min(1.0, self._step / max(self.beta_warmup, 1))

    def step(self) -> None:
        """Increment training step counter for annealing."""
        self._step += 1

    def forward(
        self,
        pred: Tensor,          # [B, M, 3]
        full: Tensor,          # [B, M, 3]
        partial: Tensor,       # [B, N, 3]
        mu: Tensor,            # [B, D]
        logvar: Tensor,        # [B, D]
    ) -> dict:
        """
        Returns
        -------
        dict with total loss and component losses.
        """
        from src.training.losses import chamfer_distance_l1, fidelity_loss, kl_divergence

        cd = chamfer_distance_l1(pred, full)
        kl = kl_divergence(mu, logvar)
        fidel = fidelity_loss(partial, pred)

        total = (
            self.cd_weight * cd
            + self.beta * kl
            + self.fidelity_weight * fidel
        )

        return {
            "loss": total,
            "cd_l1": cd.detach(),
            "kl": kl.detach(),
            "fidelity": fidel.detach(),
            "beta": self.beta,
        }
