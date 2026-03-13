"""
Evaluation metrics for RELIC / PaleoComplete benchmark.

compute_cd_l1                — Chamfer Distance L1
compute_cd_l2                — Chamfer Distance L2
compute_fscore               — F-Score at threshold
compute_normal_consistency   — normal cosine alignment
compute_bse                  — Bilateral Symmetry Error
compute_diversity            — MMD and COV across k samples
compute_uncertainty_calibration — Spearman ρ
MetricsTracker               — accumulates metrics across batches
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.stats import spearmanr
from torch import Tensor


# ---------------------------------------------------------------------------
# Tensor → numpy helpers
# ---------------------------------------------------------------------------

def _t(x) -> np.ndarray:
    if isinstance(x, Tensor):
        return x.detach().cpu().float().numpy()
    return np.asarray(x, dtype=np.float32)


def _cdist_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise Euclidean distance matrix [N, M]."""
    diff = a[:, None, :] - b[None, :, :]   # [N, M, 3]
    return np.sqrt((diff ** 2).sum(axis=-1))


# ---------------------------------------------------------------------------
# Chamfer Distance L1
# ---------------------------------------------------------------------------

def compute_cd_l1(pred: np.ndarray | Tensor, gt: np.ndarray | Tensor) -> float:
    """
    Chamfer Distance L1 for a single point pair (no batch dimension).

    Parameters
    ----------
    pred : [N, 3]
    gt   : [M, 3]

    Returns float
    """
    p = _t(pred)
    g = _t(gt)
    dist = _cdist_np(p, g)
    cd = dist.min(axis=1).mean() + dist.min(axis=0).mean()
    return float(cd)


# ---------------------------------------------------------------------------
# Chamfer Distance L2
# ---------------------------------------------------------------------------

def compute_cd_l2(pred: np.ndarray | Tensor, gt: np.ndarray | Tensor) -> float:
    """Chamfer Distance L2 (squared distances)."""
    p = _t(pred)
    g = _t(gt)
    dist_sq = _cdist_np(p, g) ** 2
    cd = dist_sq.min(axis=1).mean() + dist_sq.min(axis=0).mean()
    return float(cd)


# ---------------------------------------------------------------------------
# F-Score
# ---------------------------------------------------------------------------

def compute_fscore(
    pred: np.ndarray | Tensor,
    gt: np.ndarray | Tensor,
    threshold: float = 0.01,
) -> float:
    """F-Score at a distance threshold (single sample, no batch)."""
    p = _t(pred)
    g = _t(gt)
    dist = _cdist_np(p, g)
    precision = (dist.min(axis=1) < threshold).mean()
    recall = (dist.min(axis=0) < threshold).mean()
    f = 2 * precision * recall / (precision + recall + 1e-8)
    return float(f)


# ---------------------------------------------------------------------------
# Normal Consistency
# ---------------------------------------------------------------------------

def compute_normal_consistency(
    pred: np.ndarray | Tensor,
    gt: np.ndarray | Tensor,
    pred_normals: np.ndarray | Tensor,
    gt_normals: np.ndarray | Tensor,
) -> float:
    """
    Normal Consistency: mean |cos(n_pred, n_gt)| matched by nearest-neighbour.

    For each predicted point, find the nearest GT point, then measure
    the absolute cosine similarity of their normals.

    Parameters
    ----------
    pred         : [N, 3]
    gt           : [M, 3]
    pred_normals : [N, 3]
    gt_normals   : [M, 3]

    Returns float in [0, 1] (higher is better)
    """
    p = _t(pred)
    g = _t(gt)
    pn = _t(pred_normals)
    gn = _t(gt_normals)

    # Normalise
    pn = pn / (np.linalg.norm(pn, axis=-1, keepdims=True) + 1e-12)
    gn = gn / (np.linalg.norm(gn, axis=-1, keepdims=True) + 1e-12)

    dist = _cdist_np(p, g)
    nn_idx = dist.argmin(axis=1)   # [N] — index into gt for each pred point

    cos_sim = np.abs((pn * gn[nn_idx]).sum(axis=-1))   # [N]
    return float(cos_sim.mean())


# ---------------------------------------------------------------------------
# Bilateral Symmetry Error
# ---------------------------------------------------------------------------

def compute_bse(
    pred: np.ndarray | Tensor,
    plane_normal: np.ndarray | Tensor,
    plane_offset: float,
    confidence: float,
    threshold: float = 0.25,
) -> float:
    """
    Bilateral Symmetry Error (BSE).

    Returns confidence * CD(pred, reflect(pred, plane)).
    Returns 0 if confidence < threshold.

    Parameters
    ----------
    pred          : [N, 3]
    plane_normal  : [3]
    plane_offset  : float
    confidence    : float
    threshold     : float

    Returns float
    """
    if confidence < threshold:
        return 0.0

    p = _t(pred)
    n = _t(plane_normal)
    n = n / (np.linalg.norm(n) + 1e-12)

    # Reflect
    dist_to_plane = p @ n - plane_offset
    reflected = p - 2 * dist_to_plane[:, None] * n[None, :]

    dist = _cdist_np(p, reflected)
    cd = dist.min(axis=1).mean() + dist.min(axis=0).mean()
    return float(confidence * cd * 0.5)


# ---------------------------------------------------------------------------
# Diversity: MMD and COV
# ---------------------------------------------------------------------------

def compute_diversity(samples: List[np.ndarray | Tensor]) -> dict:
    """
    Compute diversity metrics across k completions of a single partial input.

    MMD (Minimum Matching Distance): for each sample, find the nearest sample
    in the set; report mean pairwise CD between nearest neighbours.

    COV (Coverage): fraction of samples that are the nearest neighbour of at
    least one other sample (a proxy for mode coverage).

    Parameters
    ----------
    samples : list of k arrays, each [N, 3]

    Returns
    -------
    dict with keys "mmd", "cov"
    """
    k = len(samples)
    if k < 2:
        return {"mmd": 0.0, "cov": 1.0}

    pts = [_t(s) for s in samples]

    # Pairwise CD matrix [k, k]
    cd_matrix = np.zeros((k, k), dtype=np.float32)
    for i in range(k):
        for j in range(i + 1, k):
            cd = compute_cd_l1(pts[i], pts[j])
            cd_matrix[i, j] = cd
            cd_matrix[j, i] = cd

    # MMD: for each sample, find nearest other sample
    np.fill_diagonal(cd_matrix, np.inf)
    min_dists = cd_matrix.min(axis=1)   # [k]
    mmd = float(min_dists.mean())

    # COV: fraction of samples that are nearest neighbour of at least one other
    nn_of = cd_matrix.argmin(axis=1)    # [k] — for each i, which j is nearest?
    covered = set(nn_of.tolist())
    cov = float(len(covered) / k)

    return {"mmd": mmd, "cov": cov}


# ---------------------------------------------------------------------------
# Uncertainty Calibration
# ---------------------------------------------------------------------------

def compute_uncertainty_calibration(
    samples: List[np.ndarray | Tensor],   # k arrays [N, 3]
    gt: np.ndarray | Tensor,              # [N, 3]
) -> dict:
    """
    Compute Spearman ρ between per-point uncertainty (std across samples)
    and per-point reconstruction error (dist to GT).

    A high ρ validates that the uncertainty heatmap is scientifically meaningful.

    Parameters
    ----------
    samples : list of k arrays [N, 3]  — multiple completions
    gt      : [M, 3]                   — ground truth

    Returns
    -------
    dict with keys "spearman_rho", "p_value"
    """
    if len(samples) < 2:
        return {"spearman_rho": 0.0, "p_value": 1.0}

    pts = np.stack([_t(s) for s in samples], axis=0)  # [k, N, 3]
    gt_np = _t(gt)

    # Per-point std across samples (magnitude of std vector)
    per_point_std = pts.std(axis=0)                   # [N, 3]
    uncertainty = np.linalg.norm(per_point_std, axis=-1)  # [N]

    # Per-point reconstruction error: distance from each point to nearest GT point
    ref = pts[0]   # [N, 3]
    dist_to_gt = _cdist_np(ref, gt_np).min(axis=1)    # [N]

    rho, p_val = spearmanr(uncertainty, dist_to_gt)
    return {"spearman_rho": float(rho), "p_value": float(p_val)}


# ---------------------------------------------------------------------------
# MetricsTracker
# ---------------------------------------------------------------------------

class MetricsTracker:
    """
    Accumulates metrics across batches and reports mean ± std.

    Usage
    -----
    tracker = MetricsTracker()
    for batch in ...:
        tracker.update({"cd_l1": 0.05, "fscore": 0.9})
    summary = tracker.summary()
    """

    def __init__(self) -> None:
        self._data: Dict[str, List[float]] = {}

    def update(self, metrics: Dict[str, float]) -> None:
        """Add a dict of scalar metrics."""
        for k, v in metrics.items():
            self._data.setdefault(k, []).append(float(v))

    def reset(self) -> None:
        self._data.clear()

    def summary(self) -> Dict[str, Dict[str, float]]:
        """
        Returns a dict mapping metric_name → {"mean": float, "std": float}.
        """
        out: Dict[str, Dict[str, float]] = {}
        for k, vals in self._data.items():
            arr = np.array(vals, dtype=np.float64)
            out[k] = {"mean": float(arr.mean()), "std": float(arr.std())}
        return out

    def mean(self, key: str) -> float:
        vals = self._data.get(key, [])
        return float(np.mean(vals)) if vals else 0.0
