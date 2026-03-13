"""
Geometry-Aware Encoder (AdaPoinTr-style).

PointProxy          — learnable proxy tokens via FPS + local attention
LocalAttentionBlock — self-attention within k-NN neighbourhoods
GeometryAwareEncoder— full encoder: N×3 → z_global ∈ ℝ^C + z_local ∈ ℝ^(K×d)
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor

from src.utils.geometry import farthest_point_sample, knn


# ---------------------------------------------------------------------------
# Mini-PointNet for local feature extraction
# ---------------------------------------------------------------------------

class SharedMLP(nn.Module):
    """1-D convolution-based shared MLP operating on [B, C, N]."""

    def __init__(self, in_channels: int, out_channels: int, bn: bool = True) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Conv1d(in_channels, out_channels, 1)]
        if bn:
            layers.append(nn.BatchNorm1d(out_channels))
        layers.append(nn.GELU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Local Attention Block
# ---------------------------------------------------------------------------

class LocalAttentionBlock(nn.Module):
    """
    Self-attention within local k-NN neighbourhoods.

    For each proxy point, gathers its k nearest neighbours from the full
    point set and performs multi-head self-attention within that local window.

    Parameters
    ----------
    d_model  : int   feature dimension
    n_heads  : int   number of attention heads
    k        : int   neighbourhood size
    dropout  : float
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 4,
        k: int = 16,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.k = k

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(
        self,
        proxy_feats: Tensor,   # [B, K, d_model]
        proxy_coords: Tensor,  # [B, K, 3]
        point_feats: Tensor,   # [B, N, d_model]
        point_coords: Tensor,  # [B, N, 3]
    ) -> Tensor:
        """
        Returns updated proxy_feats [B, K, d_model].
        """
        B, K, _ = proxy_feats.shape
        k = min(self.k, point_coords.shape[1])

        # Find k nearest neighbours in full point set for each proxy
        _, nn_idx = knn(proxy_coords, point_coords, k)  # [B, K, k]

        # Gather neighbour features
        # nn_idx: [B, K, k]
        nn_idx_flat = nn_idx.reshape(B, -1)  # [B, K*k]
        gathered = torch.gather(
            point_feats,
            1,
            nn_idx_flat.unsqueeze(-1).expand(-1, -1, self.d_model),
        ).reshape(B, K, k, self.d_model)  # [B, K, k, d]

        # For each proxy, attend over its local neighbourhood
        # Reshape: treat K proxies independently → [B*K, k, d]
        bk = B * K
        q = proxy_feats.reshape(bk, 1, self.d_model)
        kv = gathered.reshape(bk, k, self.d_model)

        attended, _ = self.attn(
            self.norm1(q),
            self.norm1(kv),
            self.norm1(kv),
        )
        out = proxy_feats.reshape(bk, 1, self.d_model) + attended  # residual
        out = out + self.ff(self.norm2(out))
        return out.reshape(B, K, self.d_model)


# ---------------------------------------------------------------------------
# Point Proxy
# ---------------------------------------------------------------------------

class PointProxy(nn.Module):
    """
    Learnable point proxy tokens (AdaPoinTr-inspired).

    1. Apply FPS to select K proxy centroids.
    2. For each proxy, aggregate local features from the k nearest points
       using a shared MLP + max pooling.
    3. Refine proxy features with LocalAttentionBlock.

    Parameters
    ----------
    in_dim    : int  input point feature dim (3 for raw XYZ, or higher if pre-lifted)
    d_model   : int  proxy feature dimension
    n_proxies : int  number of proxy tokens (K)
    k_local   : int  local neighbourhood size for initial feature extraction
    n_layers  : int  number of local attention layers
    """

    def __init__(
        self,
        in_dim: int = 3,
        d_model: int = 256,
        n_proxies: int = 256,
        k_local: int = 16,
        n_layers: int = 2,
        n_heads: int = 4,
    ) -> None:
        super().__init__()
        self.n_proxies = n_proxies
        self.k_local = k_local
        self.d_model = d_model

        # Lift raw XYZ to d_model via shared MLP
        self.lift_mlp = nn.Sequential(
            SharedMLP(in_dim, 64),
            SharedMLP(64, 128),
            SharedMLP(128, d_model),
        )

        # Local attention blocks
        self.local_attn_layers = nn.ModuleList([
            LocalAttentionBlock(d_model=d_model, n_heads=n_heads, k=k_local)
            for _ in range(n_layers)
        ])

    def forward(self, points: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Parameters
        ----------
        points : Tensor[B, N, 3]

        Returns
        -------
        proxy_feats  : Tensor[B, K, d_model]
        proxy_coords : Tensor[B, K, 3]
        point_feats  : Tensor[B, N, d_model]
        """
        B, N, _ = points.shape

        # Lift point features
        # lift_mlp expects [B, C, N]
        point_feats = self.lift_mlp(points.permute(0, 2, 1))   # [B, d_model, N]
        point_feats = point_feats.permute(0, 2, 1)              # [B, N, d_model]

        # FPS to get proxy centroids
        proxy_coords_list = []
        for b in range(B):
            sampled = farthest_point_sample(points[b], self.n_proxies)  # [K, 3]
            proxy_coords_list.append(sampled)
        proxy_coords = torch.stack(proxy_coords_list, dim=0)  # [B, K, 3]

        # Initialise proxy features: aggregate k-NN from lifted features
        k = min(self.k_local, N)
        _, nn_idx = knn(proxy_coords, points, k)  # [B, K, k]
        nn_idx_flat = nn_idx.reshape(B, -1)       # [B, K*k]
        gathered = torch.gather(
            point_feats,
            1,
            nn_idx_flat.unsqueeze(-1).expand(-1, -1, self.d_model),
        ).reshape(B, self.n_proxies, k, self.d_model)
        proxy_feats = gathered.max(dim=2).values  # [B, K, d_model] — max pooling

        # Refine with local attention
        for attn_layer in self.local_attn_layers:
            proxy_feats = attn_layer(proxy_feats, proxy_coords, point_feats, points)

        return proxy_feats, proxy_coords, point_feats


# ---------------------------------------------------------------------------
# Geometry-Aware Encoder
# ---------------------------------------------------------------------------

class GeometryAwareEncoder(nn.Module):
    """
    Full point cloud encoder.

    Input  : Tensor[B, N, 3]
    Output : z_enc     ∈ ℝ^[B, C]       global feature via global pooling
             z_local   ∈ ℝ^[B, K, d]    per-proxy local features
             proxy_coords ∈ ℝ^[B, K, 3] proxy centroid coordinates

    Architecture:
    1. PointProxy: FPS → local attention → proxy features [B, K, d_model]
    2. Transformer self-attention across all K proxies
    3. Global pooling (mean + max) → z_enc ∈ ℝ^[B, 2*d_model] → Linear → ℝ^[B, C]

    Parameters
    ----------
    n_points   : int  expected number of input points (not strictly required)
    d_model    : int  internal feature dimension
    out_dim    : int  C — dimension of z_enc
    n_proxies  : int  K — number of proxy tokens
    k_local    : int  local neighbourhood size
    n_proxy_layers : int  local attention layers in PointProxy
    n_global_layers: int  global transformer layers across proxies
    """

    def __init__(
        self,
        n_points: int = 2048,
        d_model: int = 256,
        out_dim: int = 256,
        n_proxies: int = 256,
        k_local: int = 16,
        n_proxy_layers: int = 2,
        n_global_layers: int = 4,
        n_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.out_dim = out_dim
        self.n_proxies = n_proxies

        self.point_proxy = PointProxy(
            in_dim=3,
            d_model=d_model,
            n_proxies=n_proxies,
            k_local=k_local,
            n_layers=n_proxy_layers,
            n_heads=n_heads,
        )

        # Global self-attention encoder (Transformer over K proxy tokens)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.global_transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_global_layers, enable_nested_tensor=False
        )

        # Projection from pooled features to z_enc
        self.proj = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, out_dim),
        )

    def forward(self, points: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Parameters
        ----------
        points : Tensor[B, N, 3]

        Returns
        -------
        z_enc        : Tensor[B, out_dim]   global encoding
        z_local      : Tensor[B, K, d_model] per-proxy features (for decoder)
        proxy_coords : Tensor[B, K, 3]
        """
        # Step 1: build proxy features
        proxy_feats, proxy_coords, _ = self.point_proxy(points)  # [B, K, d]

        # Step 2: global self-attention across proxies
        proxy_feats = self.global_transformer(proxy_feats)  # [B, K, d]

        # Step 3: global pooling
        mean_feat = proxy_feats.mean(dim=1)  # [B, d]
        max_feat = proxy_feats.max(dim=1).values  # [B, d]
        pooled = torch.cat([mean_feat, max_feat], dim=-1)  # [B, 2*d]

        z_enc = self.proj(pooled)  # [B, out_dim]

        return z_enc, proxy_feats, proxy_coords
