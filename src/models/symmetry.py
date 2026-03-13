"""
Confidence-Gated Bilateral Symmetry Module.

RANSACSymmetryDetector     — finds the best bilateral symmetry plane
reflect_points             — efficient point reflection across a plane
SymmetryLoss               — confidence-gated CD symmetry loss
ConfidenceGatedSymmetryModule — injects equivariant features into encoder output
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# Point reflection
# ---------------------------------------------------------------------------

def reflect_points(
    points: Tensor,
    normal: Tensor,
    offset: float,
) -> Tensor:
    """
    Reflect `points` across the plane defined by (normal, offset).

    Plane equation: normal · x = offset
    Reflection formula: x' = x - 2 * (normal · x - offset) * normal

    Parameters
    ----------
    points : Tensor[..., N, 3]
    normal : Tensor[3]    unit normal
    offset : float        plane offset

    Returns
    -------
    reflected : Tensor[..., N, 3]
    """
    n = normal / (torch.norm(normal) + 1e-12)
    # signed distance from plane
    dist = (points @ n) - offset   # [..., N]
    return points - 2 * dist.unsqueeze(-1) * n


# ---------------------------------------------------------------------------
# RANSAC Symmetry Detector
# ---------------------------------------------------------------------------

class RANSACSymmetryDetector:
    """
    Estimates the best bilateral symmetry plane of a point cloud via RANSAC.

    Algorithm
    ---------
    1. Sample `n_iterations` triplets of points.
    2. For each triplet, compute the unique plane perpendicular to the line
       connecting two points and passing through their midpoint (mirror plane
       candidate).
    3. Reflect all points across the candidate plane and count the fraction
       that have a nearest-neighbour match within `match_threshold`.
    4. Return the plane with the highest inlier fraction as confidence.

    Parameters
    ----------
    n_iterations    : int    number of RANSAC iterations (candidate planes)
    match_threshold : float  distance threshold for a "mirror match" (in data units)
    min_points      : int    minimum points required; returns low confidence otherwise
    """

    def __init__(
        self,
        n_iterations: int = 100,
        match_threshold: float = 0.05,
        min_points: int = 32,
    ) -> None:
        self.n_iterations = n_iterations
        self.match_threshold = match_threshold
        self.min_points = min_points

    @torch.no_grad()
    def detect(
        self, points: Tensor
    ) -> Tuple[Tensor, float, float]:
        """
        Parameters
        ----------
        points : Tensor[N, 3]  (single point cloud, no batch dim)

        Returns
        -------
        plane_normal  : Tensor[3]   unit normal of best symmetry plane
        plane_offset  : float
        confidence    : float in [0, 1]
        """
        device = points.device
        N = points.shape[0]

        if N < self.min_points:
            fallback_normal = torch.tensor([0.0, 1.0, 0.0], device=device)
            return fallback_normal, 0.0, 0.0

        best_normal = torch.tensor([0.0, 1.0, 0.0], device=device)
        best_offset = 0.0
        best_confidence = 0.0

        for _ in range(self.n_iterations):
            # Sample 2 points; the mirror plane is perpendicular to the
            # line joining them and passes through their midpoint.
            idx = torch.randint(0, N, (2,), device=device)
            p1, p2 = points[idx[0]], points[idx[1]]

            direction = p2 - p1
            norm_d = torch.norm(direction)
            if norm_d < 1e-8:
                continue

            normal = direction / norm_d
            midpoint = (p1 + p2) * 0.5
            offset = float((normal * midpoint).sum().item())

            # Reflect all points
            reflected = reflect_points(points, normal, offset)

            # Count fraction with a mirror match
            dist_mat = torch.cdist(reflected, points)     # [N, N]
            min_dists = dist_mat.min(dim=1).values         # [N]
            inlier_frac = float((min_dists < self.match_threshold).float().mean().item())

            if inlier_frac > best_confidence:
                best_confidence = inlier_frac
                best_normal = normal.clone()
                best_offset = offset

        return best_normal, best_offset, best_confidence

    def detect_batch(
        self, points: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Run detection for each item in a batch.

        Parameters
        ----------
        points : Tensor[B, N, 3]

        Returns
        -------
        normals     : Tensor[B, 3]
        offsets     : Tensor[B]
        confidences : Tensor[B]
        """
        B = points.shape[0]
        normals = []
        offsets = []
        confidences = []
        for b in range(B):
            n, o, c = self.detect(points[b])
            normals.append(n)
            offsets.append(o)
            confidences.append(c)
        return (
            torch.stack(normals, dim=0),
            torch.tensor(offsets, device=points.device),
            torch.tensor(confidences, device=points.device),
        )


# ---------------------------------------------------------------------------
# Symmetry Loss
# ---------------------------------------------------------------------------

class SymmetryLoss(nn.Module):
    """
    Confidence-gated bilateral symmetry loss.

    L_sym = confidence * CD(pred, reflect(pred, plane))

    When confidence < threshold, returns 0 (no loss gradient).
    Uses CD-L1 internally.

    Parameters
    ----------
    confidence_threshold : float
        Below this, the loss is zero.
    """

    def __init__(self, confidence_threshold: float = 0.25) -> None:
        super().__init__()
        self.confidence_threshold = confidence_threshold

    def forward(
        self,
        pred: Tensor,          # [B, N, 3]
        plane_normals: Tensor, # [B, 3]
        plane_offsets: Tensor, # [B]
        confidences: Tensor,   # [B]
    ) -> Tensor:
        """
        Returns a scalar symmetry loss (mean across batch).
        """
        B = pred.shape[0]
        losses = []
        for b in range(B):
            conf = confidences[b]
            if conf < self.confidence_threshold:
                losses.append(torch.zeros(1, device=pred.device).squeeze())
                continue

            n = plane_normals[b]
            o = float(plane_offsets[b].item())
            reflected = reflect_points(pred[b], n, o)  # [N, 3]

            # CD-L1 between pred and its reflection
            dist_ab = torch.cdist(pred[b].unsqueeze(0), reflected.unsqueeze(0))[0]  # [N, N]
            dist_ba = dist_ab.T  # [N, N]

            loss_ab = dist_ab.min(dim=1).values.mean()
            loss_ba = dist_ba.min(dim=1).values.mean()
            cd = (loss_ab + loss_ba) * 0.5

            losses.append(conf * cd)

        return torch.stack(losses).mean()


# ---------------------------------------------------------------------------
# Confidence-Gated Symmetry Module
# ---------------------------------------------------------------------------

class ConfidenceGatedSymmetryModule(nn.Module):
    """
    Wraps RANSAC symmetry detection and injects equivariant symmetry features
    into the encoder output.

    Equivariant features: for each point, compute the signed distance to the
    detected symmetry plane. This provides the encoder with an orientation-
    consistent spatial feature.

    Parameters
    ----------
    d_model              : int    encoder feature dimension
    confidence_threshold : float  gating threshold
    n_ransac             : int    RANSAC iterations
    """

    def __init__(
        self,
        d_model: int = 256,
        confidence_threshold: float = 0.25,
        n_ransac: int = 100,
        match_threshold: float = 0.05,
    ) -> None:
        super().__init__()
        self.confidence_threshold = confidence_threshold
        self.detector = RANSACSymmetryDetector(
            n_iterations=n_ransac,
            match_threshold=match_threshold,
        )
        # Linear layer to inject symmetry features into encoder output
        # Adds 1 scalar (signed dist) + 3 (normal) = 4 extra dims → project to d_model
        self.sym_proj = nn.Linear(d_model + 4, d_model)
        self.sym_loss = SymmetryLoss(confidence_threshold=confidence_threshold)

    def forward(
        self,
        points: Tensor,      # [B, N, 3]
        z_enc: Tensor,       # [B, d_model]
        z_local: Tensor,     # [B, K, d_model]
        proxy_coords: Tensor, # [B, K, 3]
    ) -> Tuple[Tensor, Tensor, dict]:
        """
        Detect symmetry planes, augment encoder features with symmetry info.

        Returns
        -------
        z_enc_aug  : Tensor[B, d_model]   symmetry-augmented global encoding
        z_local_aug: Tensor[B, K, d_model] (unchanged in this implementation)
        sym_info   : dict with normals, offsets, confidences
        """
        normals, offsets, confidences = self.detector.detect_batch(points)

        # Compute per-sample signed distances from proxy centroids to symmetry plane
        # Then augment z_enc with symmetry features
        B = z_enc.shape[0]
        sym_feats = []
        for b in range(B):
            n = normals[b]
            o = offsets[b]
            conf = confidences[b]
            # Signed distance from centroid to plane
            centroid = points[b].mean(dim=0)
            signed_dist = float((n * centroid).sum().item()) - float(o.item())
            # Feature vector: [normal (3), signed_dist (1)]
            feat = torch.cat([n, torch.tensor([signed_dist], device=n.device)])  # [4]
            sym_feats.append(feat)

        sym_feats = torch.stack(sym_feats, dim=0)  # [B, 4]

        # Concatenate with z_enc and project back
        z_aug = torch.cat([z_enc, sym_feats], dim=-1)  # [B, d_model+4]
        z_enc_aug = self.sym_proj(z_aug)               # [B, d_model]

        sym_info = {
            "plane_normals": normals,
            "plane_offsets": offsets,
            "confidences": confidences,
        }

        return z_enc_aug, z_local, sym_info
