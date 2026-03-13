"""
SeedFormer-style Hierarchical Transformer Decoder.

SeedPoint                  — generates coarse seed points from z_global
FeaturePropagation         — interpolates features from coarse to fine
HierarchicalTransformerDecoder — full 3-stage decoder: 128 → 512 → 4096 points
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """Simple MLP with GELU activations."""

    def __init__(self, dims: list[int], last_activation: bool = False) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2 or last_activation:
                layers.append(nn.GELU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# SeedPoint
# ---------------------------------------------------------------------------

class SeedPoint(nn.Module):
    """
    Generates K coarse seed points from z_global ∈ ℝ^[B, D].

    Uses an MLP to produce K × 3 coordinates.

    Parameters
    ----------
    z_dim    : int    z_global dimension
    n_seeds  : int    number of seed points to generate (K)
    """

    def __init__(self, z_dim: int = 256, n_seeds: int = 128) -> None:
        super().__init__()
        self.n_seeds = n_seeds
        self.mlp = MLP([z_dim, 512, 1024, n_seeds * 3])

    def forward(self, z_global: Tensor) -> Tensor:
        """
        Parameters
        ----------
        z_global : Tensor[B, D]

        Returns
        -------
        seeds : Tensor[B, n_seeds, 3]
        """
        B = z_global.shape[0]
        out = self.mlp(z_global)          # [B, n_seeds * 3]
        return out.reshape(B, self.n_seeds, 3)


# ---------------------------------------------------------------------------
# Feature Propagation (distance-weighted interpolation)
# ---------------------------------------------------------------------------

class FeaturePropagation(nn.Module):
    """
    Propagates features from a coarse set to a fine set by distance-weighted
    interpolation (as in PointNet++).

    Parameters
    ----------
    in_channels  : int
    out_channels : int
    k            : int   number of nearest neighbours for interpolation
    """

    def __init__(self, in_channels: int, out_channels: int, k: int = 3) -> None:
        super().__init__()
        self.k = k
        self.mlp = MLP([in_channels, out_channels, out_channels])

    def forward(
        self,
        fine_coords: Tensor,    # [B, M, 3]  fine set coordinates
        coarse_coords: Tensor,  # [B, K, 3]  coarse set coordinates
        coarse_feats: Tensor,   # [B, K, C]  coarse set features
    ) -> Tensor:
        """
        Returns interpolated features at fine_coords: Tensor[B, M, out_channels].
        """
        B, M, _ = fine_coords.shape
        _, K, C = coarse_feats.shape
        k = min(self.k, K)

        # Compute pairwise distances: [B, M, K]
        dist = torch.cdist(fine_coords, coarse_coords)
        knn_dists, knn_idx = dist.topk(k, dim=-1, largest=False)  # [B, M, k]

        # Inverse-distance weights
        weights = 1.0 / (knn_dists + 1e-8)
        weights = weights / weights.sum(dim=-1, keepdim=True)  # [B, M, k]

        # Gather coarse features at k-NN indices: [B, M, k, C]
        idx_exp = knn_idx.unsqueeze(-1).expand(-1, -1, -1, C)
        gathered = torch.gather(
            coarse_feats.unsqueeze(1).expand(-1, M, -1, -1),
            2,
            idx_exp,
        )  # [B, M, k, C]

        # Weighted sum
        interpolated = (gathered * weights.unsqueeze(-1)).sum(dim=2)  # [B, M, C]
        return self.mlp(interpolated)                                   # [B, M, out_channels]


# ---------------------------------------------------------------------------
# Cross-attention block
# ---------------------------------------------------------------------------

class CrossAttentionBlock(nn.Module):
    """Transformer cross-attention: query from seed set, key/value from z_local."""

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, query: Tensor, key_value: Tensor) -> Tensor:
        """
        query     : Tensor[B, Q, d_model]
        key_value : Tensor[B, KV, d_model]
        """
        q = self.norm_q(query)
        kv = self.norm_kv(key_value)
        attended, _ = self.attn(q, kv, kv)
        out = query + attended
        out = out + self.ff(self.norm2(out))
        return out


# ---------------------------------------------------------------------------
# Hierarchical Transformer Decoder
# ---------------------------------------------------------------------------

class HierarchicalTransformerDecoder(nn.Module):
    """
    Three-stage hierarchical decoder.

    Stage 1: z_global (256) → MLP → 128 seed points × 3
    Stage 2: 128 seeds → transformer cross-attention on z_local → 512 points
    Stage 3: 512 points → FoldingNet 2-D grid + local refinement → 4096 points

    Parameters
    ----------
    z_global_dim : int    global latent dimension
    z_local_dim  : int    per-proxy local feature dimension
    k_local      : int    number of proxy tokens (K)
    cond_dim     : int    optional conditioning dimension (injected via cross-attn)
    n_points_out : int    final output resolution (default 4096)
    d_model      : int    internal feature dimension
    """

    def __init__(
        self,
        z_global_dim: int = 256,
        z_local_dim: int = 64,
        k_local: int = 256,
        cond_dim: int = 256,
        n_points_out: int = 4096,
        d_model: int = 256,
    ) -> None:
        super().__init__()
        self.n_points_out = n_points_out
        self.d_model = d_model

        # --- Stage 1: coarse seed points ---
        self.seed_gen = SeedPoint(z_dim=z_global_dim, n_seeds=128)

        # Embed seed coordinates to d_model
        self.seed_embed = MLP([3, d_model // 2, d_model])

        # Project z_local to d_model if needed
        self.local_proj = nn.Linear(z_local_dim, d_model)

        # Optional conditioning injection
        self.cond_proj = nn.Linear(cond_dim, d_model) if cond_dim > 0 else None

        # Stage 1→2 cross-attention (128 seeds attend over K local features)
        self.stage2_attn = nn.ModuleList([
            CrossAttentionBlock(d_model, n_heads=4)
            for _ in range(2)
        ])

        # Stage 2: upsample from 128 → 512
        # Each seed generates 4 fine points via FeaturePropagation + MLP
        self.stage2_upsample = nn.Linear(d_model, 4 * d_model)
        self.stage2_refine = MLP([d_model, d_model, d_model])
        self.stage2_out = MLP([d_model, d_model // 2, 3])

        # Stage 3: upsample from 512 → 4096 via 2-D grid folding
        # Grid size sqrt(4096/512) * 512 = 8 per seed → 512 * 8 = 4096
        self.fold_grid_size = 8   # each stage-2 point expands to 8 fine points
        self.fold_input_dim = d_model + 2  # feature + 2D grid coords
        self.fold_mlp1 = MLP([self.fold_input_dim, d_model, d_model // 2, 3])
        self.fold_refine = MLP([d_model + 3, d_model // 2, 3])

        # Grid for folding: (fold_grid_size, ) uniform in [-0.05, 0.05]
        # Built lazily in forward
        self._grid: Optional[Tensor] = None

    def _get_grid(self, device) -> Tensor:
        """Generate a 1D grid of fold_grid_size points."""
        if self._grid is None or self._grid.device != device:
            self._grid = torch.linspace(-0.05, 0.05, self.fold_grid_size, device=device)
        return self._grid

    def forward(
        self,
        z_global: Tensor,             # [B, z_global_dim]
        z_local: Tensor,              # [B, K, z_local_dim]
        conditioning: Optional[Tensor] = None,  # [B, cond_dim] or None
    ) -> Tensor:
        """
        Returns
        -------
        complete_points : Tensor[B, n_points_out, 3]
        """
        B = z_global.shape[0]
        device = z_global.device

        # --- Stage 1: generate 128 coarse seeds ---
        seeds_xyz = self.seed_gen(z_global)       # [B, 128, 3]
        seed_feats = self.seed_embed(seeds_xyz)   # [B, 128, d_model]

        # Project z_local
        local_feats = self.local_proj(z_local)    # [B, K, d_model]

        # Inject conditioning into local features via addition
        if conditioning is not None and self.cond_proj is not None:
            cond_feat = self.cond_proj(conditioning)   # [B, d_model]
            local_feats = local_feats + cond_feat.unsqueeze(1)

        # Stage 1→2: cross-attention on z_local
        for attn_layer in self.stage2_attn:
            seed_feats = attn_layer(seed_feats, local_feats)  # [B, 128, d_model]

        # --- Stage 2: 128 → 512 points ---
        # Each seed generates 4 children
        stage2_feats = self.stage2_upsample(seed_feats)        # [B, 128, 4*d_model]
        stage2_feats = stage2_feats.reshape(B, 512, self.d_model)
        stage2_feats = self.stage2_refine(stage2_feats)        # [B, 512, d_model]
        stage2_xyz = seeds_xyz.repeat_interleave(4, dim=1)     # [B, 512, 3]
        stage2_xyz = stage2_xyz + self.stage2_out(stage2_feats)  # [B, 512, 3]

        # --- Stage 3: 512 → 4096 via FoldingNet-style 2D grid ---
        # For each of the 512 points, generate fold_grid_size fine points
        # Grid offsets in [x, y] directions
        grid = self._get_grid(device)  # [fold_grid_size]

        # Expand: [B, 512, fold_grid_size, d_model + 2]
        feat_exp = stage2_feats.unsqueeze(2).expand(-1, -1, self.fold_grid_size, -1)
        xyz_exp = stage2_xyz.unsqueeze(2).expand(-1, -1, self.fold_grid_size, -1)

        # 1D grid offset added to x coordinate for variety
        grid_exp = grid.view(1, 1, -1, 1).expand(B, 512, -1, 1)
        # Repeat for second dimension (use different stride)
        grid2 = torch.linspace(-0.05, 0.05, self.fold_grid_size, device=device)
        grid2_exp = grid2.flip(0).view(1, 1, -1, 1).expand(B, 512, -1, 1)

        fold_in = torch.cat([feat_exp, grid_exp, grid2_exp], dim=-1)  # [B, 512, G, d+2]
        fold_in_flat = fold_in.reshape(B, 512 * self.fold_grid_size, self.fold_input_dim)

        fold_offsets = self.fold_mlp1(fold_in_flat)  # [B, 4096, 3]

        # Add parent offset
        fine_xyz = xyz_exp.reshape(B, 512 * self.fold_grid_size, 3) + fold_offsets

        # Local refinement: attend back to stage2 features via feature propagation
        # Simple residual refinement using stage2 features
        parent_feats = stage2_feats.unsqueeze(2).expand(-1, -1, self.fold_grid_size, -1)
        parent_feats_flat = parent_feats.reshape(B, 4096, self.d_model)
        refine_input = torch.cat([parent_feats_flat, fine_xyz], dim=-1)  # [B, 4096, d+3]
        fine_xyz = fine_xyz + self.fold_refine(refine_input)             # [B, 4096, 3]

        # Trim or pad to exactly n_points_out
        if fine_xyz.shape[1] > self.n_points_out:
            fine_xyz = fine_xyz[:, :self.n_points_out, :]
        elif fine_xyz.shape[1] < self.n_points_out:
            pad_n = self.n_points_out - fine_xyz.shape[1]
            pad = fine_xyz[:, :pad_n, :]
            fine_xyz = torch.cat([fine_xyz, pad], dim=1)

        return fine_xyz
