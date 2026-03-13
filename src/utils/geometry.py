"""
Geometry utilities for point cloud processing.

poisson_disk_sample     — Poisson disk sampling via trimesh
area_weighted_sample    — area-weighted mesh sampling
pca_normalize           — normalise point cloud to unit sphere
pca_aspect_ratio        — PCA-based shape elongation ratio
farthest_point_sample   — iterative FPS in PyTorch
knn                     — k-nearest-neighbour via torch.cdist
compute_normals         — PCA-based normal estimation
landmark_region_mask    — boolean mask for a spherical landmark region
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import trimesh
from torch import Tensor


# ---------------------------------------------------------------------------
# Mesh-to-point-cloud sampling
# ---------------------------------------------------------------------------

def poisson_disk_sample(mesh: trimesh.Trimesh, n_points: int = 4096) -> Tensor:
    """
    Sample `n_points` from `mesh` using Poisson disk (blue-noise) sampling.

    Falls back to area-weighted sampling if the mesh is degenerate.
    """
    try:
        pts, _ = trimesh.sample.sample_surface_even(mesh, n_points)
    except Exception:
        pts, _ = trimesh.sample.sample_surface(mesh, n_points)
    pts = np.array(pts, dtype=np.float32)
    # Pad / trim to exactly n_points
    if pts.shape[0] < n_points:
        idx = np.random.choice(pts.shape[0], n_points - pts.shape[0])
        pts = np.concatenate([pts, pts[idx]], axis=0)
    pts = pts[:n_points]
    return torch.from_numpy(pts)


def area_weighted_sample(mesh: trimesh.Trimesh, n_points: int = 4096) -> Tensor:
    """
    Sample `n_points` from `mesh` with probability proportional to face area.
    """
    pts, _ = trimesh.sample.sample_surface(mesh, n_points)
    pts = np.array(pts, dtype=np.float32)
    if pts.shape[0] < n_points:
        idx = np.random.choice(pts.shape[0], n_points - pts.shape[0])
        pts = np.concatenate([pts, pts[idx]], axis=0)
    pts = pts[:n_points]
    return torch.from_numpy(pts)


# ---------------------------------------------------------------------------
# PCA normalisation
# ---------------------------------------------------------------------------

def pca_normalize(points: Tensor) -> Tuple[Tensor, Dict]:
    """
    Centre the point cloud and scale to fit the unit sphere.

    Returns
    -------
    normalised : Tensor[N, 3]
    transform  : dict with keys 'centroid' (Tensor[3]) and 'scale' (float)
    """
    centroid = points.mean(dim=0)
    pts = points - centroid
    scale = float(torch.norm(pts, dim=-1).max().item())
    if scale < 1e-12:
        scale = 1.0
    pts = pts / scale
    return pts, {"centroid": centroid, "scale": scale}


def pca_aspect_ratio(points: Tensor) -> float:
    """
    Compute the ratio of the largest to smallest PCA eigenvalue.

    High ratios indicate elongated / taphonomically compressed shapes.
    Specimens with ratio > ~1.2× taxon mean are considered deformed.
    """
    pts = points - points.mean(dim=0, keepdim=True)
    cov = (pts.T @ pts) / (pts.shape[0] - 1)
    eigenvalues = torch.linalg.eigvalsh(cov)  # ascending order
    # Clamp to avoid division by zero for degenerate clouds
    min_eig = eigenvalues[0].abs().clamp(min=1e-12)
    max_eig = eigenvalues[-1].abs().clamp(min=1e-12)
    return float(max_eig / min_eig)


# ---------------------------------------------------------------------------
# Farthest Point Sampling
# ---------------------------------------------------------------------------

def farthest_point_sample(points: Tensor, n_samples: int) -> Tensor:
    """
    Iterative farthest point sampling.

    Parameters
    ----------
    points   : Tensor[N, 3]
    n_samples: int

    Returns
    -------
    sampled  : Tensor[n_samples, 3]
    """
    device = points.device
    N = points.shape[0]
    n_samples = min(n_samples, N)

    selected = torch.zeros(n_samples, dtype=torch.long, device=device)
    # Distances from each point to the current selected set
    dists = torch.full((N,), float("inf"), device=device)

    # Start from a random point
    selected[0] = torch.randint(0, N, (1,), device=device)
    for i in range(1, n_samples):
        last = selected[i - 1]
        d = torch.sum((points - points[last].unsqueeze(0)) ** 2, dim=-1)
        dists = torch.minimum(dists, d)
        selected[i] = dists.argmax()

    return points[selected]


# ---------------------------------------------------------------------------
# k-Nearest Neighbours
# ---------------------------------------------------------------------------

def knn(query: Tensor, key: Tensor, k: int) -> Tuple[Tensor, Tensor]:
    """
    Batched or unbatched k-nearest-neighbour search using torch.cdist.

    Parameters
    ----------
    query : Tensor[..., Q, 3]  — query points
    key   : Tensor[..., K, 3]  — database points
    k     : int

    Returns
    -------
    distances : Tensor[..., Q, k]  — Euclidean distances
    indices   : Tensor[..., Q, k]  — indices into `key`
    """
    dist_mat = torch.cdist(query, key)  # [..., Q, K]
    k_actual = min(k, key.shape[-2])
    distances, indices = dist_mat.topk(k_actual, dim=-1, largest=False)
    return distances, indices


# ---------------------------------------------------------------------------
# Normal estimation
# ---------------------------------------------------------------------------

def compute_normals(points: Tensor, k: int = 10) -> Tensor:
    """
    Estimate per-point normals via PCA of the local k-neighbourhood.

    Parameters
    ----------
    points : Tensor[N, 3]
    k      : int

    Returns
    -------
    normals : Tensor[N, 3]  (unit length)
    """
    N = points.shape[0]
    k = min(k, N - 1)

    _, indices = knn(points.unsqueeze(0), points.unsqueeze(0), k + 1)
    indices = indices[0]  # [N, k+1]

    normals = torch.zeros_like(points)
    for i in range(N):
        nbrs = points[indices[i]]  # [k+1, 3]
        centered = nbrs - nbrs.mean(dim=0, keepdim=True)
        cov = (centered.T @ centered) / (k)
        try:
            _, _, Vt = torch.linalg.svd(cov, full_matrices=False)
            n = Vt[-1]  # smallest singular vector → normal
        except RuntimeError:
            n = torch.zeros(3, device=points.device)
            n[2] = 1.0
        normals[i] = n / (torch.norm(n) + 1e-12)

    return normals


# ---------------------------------------------------------------------------
# Landmark region mask
# ---------------------------------------------------------------------------

def landmark_region_mask(
    points: Tensor,
    landmark_center: Tensor,
    radius: float,
) -> Tensor:
    """
    Returns a boolean mask [N] that is True for points within `radius`
    of `landmark_center`.
    """
    dists = torch.norm(points - landmark_center.unsqueeze(0), dim=-1)
    return dists <= radius
