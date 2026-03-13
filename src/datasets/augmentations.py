"""
Data augmentation pipeline for fossil bone point clouds.

Includes:
- LandmarkShardGenerator : creates partial shards at L1/L2/L3 levels
- TPSCompression         : thin-plate spline axis-aligned compression
- AffineShear            : random shear matrix
- CT artifact transforms : ring, matrix contamination, mineralization, erosion, dropout
- BoneAugmentation       : full training pipeline
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy.interpolate import RBFInterpolator
from torch import Tensor


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _to_numpy(t: Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def _to_tensor(a: np.ndarray, device=None) -> Tensor:
    t = torch.from_numpy(a.astype(np.float32))
    if device is not None:
        t = t.to(device)
    return t


# ---------------------------------------------------------------------------
# LandmarkShardGenerator
# ---------------------------------------------------------------------------

class LandmarkShardGenerator:
    """
    Creates partial point cloud shards at L1/L2/L3 completeness levels
    by removing regions around randomly chosen missing landmarks.

    Parameters
    ----------
    landmark_centers : dict[str, Tensor[3]]
        Mapping from landmark name to 3-D centroid (unit-sphere coords).
    landmark_radius : float
        Radius (in normalised units) around each landmark to remove.
    """

    LEVEL_MISSING: Dict[str, Tuple[int, int]] = {
        "L1": (1, 2),
        "L2": (3, 3),
        "L3": (4, 6),
    }

    def __init__(
        self,
        landmark_centers: Dict[str, Tensor],
        landmark_radius: float = 0.15,
        seed: Optional[int] = None,
    ) -> None:
        self.landmark_centers = landmark_centers
        self.landmark_radius = landmark_radius
        self.rng = random.Random(seed)

    def generate(
        self,
        full_points: Tensor,
        level: str = "L2",
    ) -> Tuple[Tensor, List[str]]:
        """
        Remove landmark regions from `full_points`.

        Returns
        -------
        partial : Tensor[K, 3]   — remaining points
        removed_landmarks : list[str] — names of removed landmarks
        """
        if level not in self.LEVEL_MISSING:
            raise ValueError(f"level must be one of {list(self.LEVEL_MISSING)}, got '{level}'")

        lo, hi = self.LEVEL_MISSING[level]
        n_remove = self.rng.randint(lo, hi)
        landmark_names = list(self.landmark_centers.keys())
        to_remove = self.rng.sample(landmark_names, min(n_remove, len(landmark_names)))

        keep_mask = torch.ones(full_points.shape[0], dtype=torch.bool)
        for lm_name in to_remove:
            center = self.landmark_centers[lm_name].to(full_points.device)
            dists = torch.norm(full_points - center.unsqueeze(0), dim=-1)
            keep_mask &= (dists > self.landmark_radius)

        return full_points[keep_mask], to_remove


# ---------------------------------------------------------------------------
# TPSCompression
# ---------------------------------------------------------------------------

class TPSCompression:
    """
    Thin-plate spline warping simulating mild taphonomic compression.

    Applies a smooth displacement field along a randomly chosen axis,
    compressing the bone shape by a factor in `ratio_range`.

    Parameters
    ----------
    p : float
        Probability of applying this transform.
    ratio_range : tuple[float, float]
        Range for the compression ratio along the chosen axis.
    n_control : int
        Number of TPS control points.
    """

    def __init__(
        self,
        p: float = 0.3,
        ratio_range: Tuple[float, float] = (0.7, 1.0),
        n_control: int = 16,
    ) -> None:
        self.p = p
        self.ratio_range = ratio_range
        self.n_control = n_control

    def __call__(self, points: Tensor) -> Tensor:
        if random.random() > self.p:
            return points

        pts_np = _to_numpy(points)  # [N, 3]

        # Random compression axis (0=x, 1=y, 2=z)
        axis = random.randint(0, 2)
        ratio = random.uniform(*self.ratio_range)

        # Build control points: sample n_control points from the cloud
        n = pts_np.shape[0]
        idx = np.random.choice(n, min(self.n_control, n), replace=False)
        ctrl = pts_np[idx].copy()

        # Target: compress along chosen axis
        target = ctrl.copy()
        target[:, axis] *= ratio

        # Fit RBF interpolator (TPS kernel d²log(d) in 3D is not directly
        # available, but thin_plate_spline works in 2D; we use multiquadric
        # as a smooth approximation for 3D)
        try:
            rbf = RBFInterpolator(ctrl, target, kernel="thin_plate_spline", degree=1)
            warped = rbf(pts_np)
        except Exception:
            # Fall back to simple linear compression if RBF fails
            warped = pts_np.copy()
            warped[:, axis] *= ratio

        return _to_tensor(warped.astype(np.float32), device=points.device)


# ---------------------------------------------------------------------------
# AffineShear
# ---------------------------------------------------------------------------

class AffineShear:
    """
    Random shear matrix applied to the point cloud.

    Simulates block displacement / simple shear from taphonomic stress.

    Parameters
    ----------
    p : float
        Probability of applying.
    max_shear : float
        Maximum off-diagonal shear coefficient (e.g. 0.15).
    """

    def __init__(self, p: float = 0.3, max_shear: float = 0.15) -> None:
        self.p = p
        self.max_shear = max_shear

    def __call__(self, points: Tensor) -> Tensor:
        if random.random() > self.p:
            return points

        s = self.max_shear
        # Build a 3x3 shear matrix: randomly choose shear plane
        shear_mat = np.eye(3, dtype=np.float32)
        # Apply shear in one off-diagonal pair
        axes = random.choice([(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)])
        shear_mat[axes[0], axes[1]] = random.uniform(-s, s)

        mat = torch.from_numpy(shear_mat).to(points.device)
        return points @ mat.T


# ---------------------------------------------------------------------------
# CT Artifact Augmentations
# ---------------------------------------------------------------------------

class RingArtifact:
    """
    Simulates CT ring artifacts: concentric cylindrical noise bands.

    Points near ring boundaries receive additive Gaussian noise.

    Parameters
    ----------
    p : float
    n_rings : int
        Number of ring noise bands.
    noise_std : float
        Standard deviation of ring noise.
    """

    def __init__(self, p: float = 0.3, n_rings: int = 3, noise_std: float = 0.02) -> None:
        self.p = p
        self.n_rings = n_rings
        self.noise_std = noise_std

    def __call__(self, points: Tensor) -> Tensor:
        if random.random() > self.p:
            return points

        pts = points.clone()
        center = pts.mean(dim=0, keepdim=True)
        pts_c = pts - center

        # Radius from random axis center
        axis = random.randint(0, 2)
        r_axes = [i for i in range(3) if i != axis]
        radii = torch.norm(pts_c[:, r_axes], dim=-1)  # [N]

        r_max = radii.max().item()
        ring_radii = torch.linspace(0, float(r_max), self.n_rings + 2)[1:-1]

        for ring_r in ring_radii:
            band = (torch.abs(radii - ring_r) < (float(r_max) * 0.05))
            noise = torch.randn(band.sum().item(), 3, device=pts.device) * self.noise_std
            pts[band] = pts[band] + noise

        return pts


class MatrixContamination:
    """
    Simulates rock matrix occlusion by deleting points in random bounding boxes.

    Parameters
    ----------
    p : float
    fraction : float
        Expected fraction of points to remove.
    n_boxes : int
        Number of contamination boxes.
    """

    def __init__(self, p: float = 0.3, fraction: float = 0.1, n_boxes: int = 3) -> None:
        self.p = p
        self.fraction = fraction
        self.n_boxes = n_boxes

    def __call__(self, points: Tensor) -> Tensor:
        if random.random() > self.p:
            return points

        pts = points
        keep = torch.ones(pts.shape[0], dtype=torch.bool, device=pts.device)

        pt_min = pts.min(dim=0).values
        pt_max = pts.max(dim=0).values
        extent = pt_max - pt_min

        for _ in range(self.n_boxes):
            # Random box size proportional to desired fraction
            box_size = extent * (self.fraction ** (1 / 3)) * random.uniform(0.5, 1.5)
            box_min_coeff = torch.rand(3, device=pts.device)
            box_min = pt_min + box_min_coeff * (extent - box_size).clamp(min=0)
            box_max = box_min + box_size

            in_box = (
                (pts[:, 0] >= box_min[0]) & (pts[:, 0] <= box_max[0]) &
                (pts[:, 1] >= box_min[1]) & (pts[:, 1] <= box_max[1]) &
                (pts[:, 2] >= box_min[2]) & (pts[:, 2] <= box_max[2])
            )
            keep &= ~in_box

        result = pts[keep]
        # Avoid returning empty cloud
        if result.shape[0] < 64:
            return points
        return result


class MineralizationInfilling:
    """
    Simulates mineralization by adding random interior points (pseudo-infill).

    Parameters
    ----------
    p : float
    fraction : float
        Fraction of N to add as infill points.
    """

    def __init__(self, p: float = 0.2, fraction: float = 0.05) -> None:
        self.p = p
        self.fraction = fraction

    def __call__(self, points: Tensor) -> Tensor:
        if random.random() > self.p:
            return points

        n_add = max(1, int(points.shape[0] * self.fraction))
        pt_min = points.min(dim=0).values
        pt_max = points.max(dim=0).values

        # Sample uniformly inside bounding box (interior approximation)
        infill = torch.rand(n_add, 3, device=points.device) * (pt_max - pt_min) + pt_min
        return torch.cat([points, infill], dim=0)


class WeatheringErosion:
    """
    Deletes surface-exposed points by estimating surface normals via local PCA
    and removing points with outward-pointing normals facing a random "erosion" direction.

    Parameters
    ----------
    p : float
    fraction : float
        Fraction of points to erode.
    k_normal : int
        Neighbourhood size for normal estimation.
    """

    def __init__(self, p: float = 0.3, fraction: float = 0.1, k_normal: int = 10) -> None:
        self.p = p
        self.fraction = fraction
        self.k_normal = k_normal

    def __call__(self, points: Tensor) -> Tensor:
        if random.random() > self.p:
            return points

        pts_np = _to_numpy(points)  # [N, 3]
        n = pts_np.shape[0]
        k = min(self.k_normal, n - 1)

        # Estimate per-point normals via PCA of k-NN
        from src.utils.geometry import knn, compute_normals  # lazy import to avoid circular
        try:
            normals_t = compute_normals(points, k=k)  # [N, 3]
            normals = _to_numpy(normals_t)
        except Exception:
            # Fallback: skip this transform
            return points

        # Random erosion direction
        erosion_dir = np.random.randn(3)
        erosion_dir /= np.linalg.norm(erosion_dir) + 1e-12

        # Points with normal aligned to erosion direction are "exposed"
        alignment = normals @ erosion_dir  # [N]
        n_remove = int(n * self.fraction)
        exposed_idx = np.argsort(alignment)[-n_remove:]

        keep_mask = np.ones(n, dtype=bool)
        keep_mask[exposed_idx] = False
        result = points[torch.from_numpy(keep_mask)]

        if result.shape[0] < 64:
            return points
        return result


class ResolutionDropout:
    """
    Random subsampling at variable densities (simulates resolution variation).

    Parameters
    ----------
    p : float
    min_ratio : float
        Minimum fraction of points to keep.
    """

    def __init__(self, p: float = 0.3, min_ratio: float = 0.5) -> None:
        self.p = p
        self.min_ratio = min_ratio

    def __call__(self, points: Tensor) -> Tensor:
        if random.random() > self.p:
            return points

        ratio = random.uniform(self.min_ratio, 1.0)
        n_keep = max(64, int(points.shape[0] * ratio))
        idx = torch.randperm(points.shape[0], device=points.device)[:n_keep]
        return points[idx]


# ---------------------------------------------------------------------------
# Composed pipelines
# ---------------------------------------------------------------------------

class CTArtifactAugmentation:
    """
    Compose all CT artifact transforms in sequence.

    Applies: RingArtifact → MatrixContamination → MineralizationInfilling
             → WeatheringErosion → ResolutionDropout
    """

    def __init__(
        self,
        ring_p: float = 0.3,
        matrix_p: float = 0.3,
        mineral_p: float = 0.2,
        erosion_p: float = 0.3,
        dropout_p: float = 0.3,
    ) -> None:
        self.transforms = [
            RingArtifact(p=ring_p),
            MatrixContamination(p=matrix_p),
            MineralizationInfilling(p=mineral_p),
            WeatheringErosion(p=erosion_p),
            ResolutionDropout(p=dropout_p),
        ]

    def __call__(self, points: Tensor) -> Tensor:
        for t in self.transforms:
            points = t(points)
        return points


# Alias used in domain_adaptation
CTArtifactPipeline = CTArtifactAugmentation


class BoneAugmentation:
    """
    Full augmentation pipeline for training.

    Applies: TPS compression → AffineShear → CTArtifactAugmentation
    Also applies random rotation and small additive jitter.
    """

    def __init__(
        self,
        tps_p: float = 0.3,
        shear_p: float = 0.3,
        ct_p: float = 0.5,
        jitter_std: float = 0.005,
    ) -> None:
        self.tps = TPSCompression(p=tps_p)
        self.shear = AffineShear(p=shear_p)
        self.ct = CTArtifactAugmentation(
            ring_p=ct_p * 0.6,
            matrix_p=ct_p * 0.6,
            mineral_p=ct_p * 0.4,
            erosion_p=ct_p * 0.6,
            dropout_p=ct_p * 0.6,
        )
        self.jitter_std = jitter_std

    def __call__(self, points: Tensor) -> Tensor:
        # Random rotation (SO(3))
        theta = random.uniform(0, 2 * np.pi)
        axis = random.randint(0, 2)
        rot_mat = self._rot_mat(theta, axis, points.device)
        points = points @ rot_mat.T

        # TPS + shear
        points = self.tps(points)
        points = self.shear(points)

        # CT artifacts
        points = self.ct(points)

        # Jitter
        if self.jitter_std > 0:
            points = points + torch.randn_like(points) * self.jitter_std

        return points

    @staticmethod
    def _rot_mat(theta: float, axis: int, device) -> Tensor:
        c, s = np.cos(theta), np.sin(theta)
        if axis == 0:
            R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)
        elif axis == 1:
            R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
        else:
            R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
        return torch.from_numpy(R).to(device)
