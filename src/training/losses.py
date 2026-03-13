"""
Loss functions for RELIC training.

chamfer_distance_l1     — batched CD L1
chamfer_distance_l2     — batched CD L2
fscore                  — F-Score at threshold
normal_consistency_loss — cosine normal alignment
bse_loss                — confidence-gated bilateral symmetry error
fidelity_loss           — partial-to-complete consistency
kl_divergence           — KL(q || p) for diagonal Gaussian
RELICLoss               — composite loss with configurable weights
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Chamfer Distance
# ---------------------------------------------------------------------------

def chamfer_distance_l1(p1: Tensor, p2: Tensor) -> Tensor:
    """
    Batched Chamfer Distance L1 between two point sets.

    CD_L1(P, Q) = (1/|P|) Σ_{p∈P} min_{q∈Q} ||p-q||_1
                + (1/|Q|) Σ_{q∈Q} min_{p∈P} ||p-q||_1

    Uses L2 distance (torch.cdist) then takes L1 via Manhattan approximation.
    Standard practice: use Euclidean distance, but scale is "L1" in literature
    meaning the final sum uses |·| not |·|². We use L2 distance here for
    numerical stability, consistent with most point cloud completion papers.

    Parameters
    ----------
    p1 : Tensor[B, N, 3]
    p2 : Tensor[B, M, 3]

    Returns
    -------
    Tensor scalar   mean CD-L1 over batch
    """
    # [B, N, M]
    dist = torch.cdist(p1, p2, p=2)
    # p1 → p2: nearest neighbour distances
    nn_p1_to_p2 = dist.min(dim=2).values  # [B, N]
    # p2 → p1
    nn_p2_to_p1 = dist.min(dim=1).values  # [B, M]

    cd = nn_p1_to_p2.mean(dim=1) + nn_p2_to_p1.mean(dim=1)  # [B]
    return cd.mean()


def chamfer_distance_l2(p1: Tensor, p2: Tensor) -> Tensor:
    """
    Batched Chamfer Distance L2 (squared distances).

    Parameters
    ----------
    p1 : Tensor[B, N, 3]
    p2 : Tensor[B, M, 3]

    Returns
    -------
    Tensor scalar
    """
    dist_sq = torch.cdist(p1, p2, p=2) ** 2   # [B, N, M]
    nn_p1 = dist_sq.min(dim=2).values           # [B, N]
    nn_p2 = dist_sq.min(dim=1).values           # [B, M]
    cd = nn_p1.mean(dim=1) + nn_p2.mean(dim=1)  # [B]
    return cd.mean()


# ---------------------------------------------------------------------------
# F-Score
# ---------------------------------------------------------------------------

def fscore(pred: Tensor, gt: Tensor, threshold: float = 0.01) -> Tensor:
    """
    F-Score at a distance threshold.

    Precision: fraction of pred points within threshold of any gt point.
    Recall:    fraction of gt points within threshold of any pred point.
    F = 2 * P * R / (P + R + eps)

    Parameters
    ----------
    pred      : Tensor[B, N, 3]
    gt        : Tensor[B, M, 3]
    threshold : float

    Returns
    -------
    Tensor scalar   mean F-score over batch
    """
    dist = torch.cdist(pred, gt, p=2)  # [B, N, M]
    # Precision: pred → gt
    precision = (dist.min(dim=2).values < threshold).float().mean(dim=1)  # [B]
    # Recall: gt → pred
    recall = (dist.min(dim=1).values < threshold).float().mean(dim=1)     # [B]

    f = 2 * precision * recall / (precision + recall + 1e-8)
    return f.mean()


# ---------------------------------------------------------------------------
# Normal Consistency
# ---------------------------------------------------------------------------

def normal_consistency_loss(
    pred_normals: Tensor,
    gt_normals: Tensor,
) -> Tensor:
    """
    Normal consistency loss: 1 - |cos(n_pred, n_gt)|.

    Expects normals to be unit-length.

    Parameters
    ----------
    pred_normals : Tensor[B, N, 3]
    gt_normals   : Tensor[B, N, 3]

    Returns
    -------
    Tensor scalar
    """
    cos_sim = F.cosine_similarity(pred_normals, gt_normals, dim=-1)  # [B, N]
    return (1 - cos_sim.abs()).mean()


# ---------------------------------------------------------------------------
# Bilateral Symmetry Error (BSE)
# ---------------------------------------------------------------------------

def bse_loss(
    pred: Tensor,
    plane_normals: Tensor,
    plane_offsets: Tensor,
    confidences: Tensor,
    threshold: float = 0.25,
) -> Tensor:
    """
    Confidence-gated bilateral symmetry error.

    BSE = confidence * CD(pred, reflect(pred, plane))
    Returns 0 when confidence < threshold.

    Parameters
    ----------
    pred          : Tensor[B, N, 3]
    plane_normals : Tensor[B, 3]
    plane_offsets : Tensor[B]
    confidences   : Tensor[B]
    threshold     : float

    Returns
    -------
    Tensor scalar
    """
    from src.models.symmetry import reflect_points

    B = pred.shape[0]
    losses = []
    for b in range(B):
        conf = confidences[b]
        if conf < threshold:
            losses.append(torch.zeros(1, device=pred.device).squeeze())
            continue
        n = plane_normals[b]
        o = float(plane_offsets[b].item())
        reflected = reflect_points(pred[b], n, o)

        dist = torch.cdist(pred[b].unsqueeze(0), reflected.unsqueeze(0))[0]
        cd_l1 = dist.min(dim=1).values.mean() + dist.min(dim=0).values.mean()
        losses.append(conf * cd_l1 * 0.5)

    return torch.stack(losses).mean()


# ---------------------------------------------------------------------------
# Fidelity Loss
# ---------------------------------------------------------------------------

def fidelity_loss(partial: Tensor, complete: Tensor) -> Tensor:
    """
    Mean distance from partial cloud to nearest point in complete cloud.

    Ensures the completed shape is consistent with the observed fragment.

    Parameters
    ----------
    partial  : Tensor[B, N, 3]
    complete : Tensor[B, M, 3]

    Returns
    -------
    Tensor scalar
    """
    dist = torch.cdist(partial, complete, p=2)    # [B, N, M]
    min_dists = dist.min(dim=2).values             # [B, N]
    return min_dists.mean()


# ---------------------------------------------------------------------------
# KL Divergence
# ---------------------------------------------------------------------------

def kl_divergence(mu: Tensor, logvar: Tensor) -> Tensor:
    """
    KL divergence KL(q(z|x) || p(z)) for a diagonal Gaussian encoder.

    KL = -0.5 * sum(1 + logvar - mu² - exp(logvar))

    Parameters
    ----------
    mu     : Tensor[B, D]
    logvar : Tensor[B, D]

    Returns
    -------
    Tensor scalar   mean KL over batch
    """
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    return kl.sum(dim=-1).mean()


# ---------------------------------------------------------------------------
# RELIC Composite Loss
# ---------------------------------------------------------------------------

class RELICLoss(nn.Module):
    """
    Composite loss combining all RELIC training objectives.

    Default weights from the paper:
        cd_l1     : 1.0
        normal_con: 0.1
        symmetry  : 0.1
        kl        : 0.01 (annealed)
        diffusion : 1.0
        fidelity  : 0.5

    Parameters
    ----------
    weights : dict   override any default weight
    beta_max : float max KL weight
    beta_warmup : int training steps to warm up beta
    symmetry_threshold : float confidence gate for symmetry loss
    """

    def __init__(
        self,
        weights: Optional[dict] = None,
        beta_max: float = 0.01,
        beta_warmup: int = 5000,
        symmetry_threshold: float = 0.25,
    ) -> None:
        super().__init__()
        default_weights = {
            "cd_l1": 1.0,
            "normal_con": 0.1,
            "symmetry": 0.1,
            "kl": beta_max,
            "diffusion": 1.0,
            "fidelity": 0.5,
        }
        if weights:
            default_weights.update(weights)
        self.weights = default_weights
        self.beta_max = beta_max
        self.beta_warmup = beta_warmup
        self.symmetry_threshold = symmetry_threshold
        self._step = 0

    @property
    def kl_beta(self) -> float:
        return self.beta_max * min(1.0, self._step / max(self.beta_warmup, 1))

    def step(self) -> None:
        self._step += 1

    def forward(
        self,
        pred: Tensor,
        full: Tensor,
        partial: Tensor,
        mu: Tensor,
        logvar: Tensor,
        pred_normals: Optional[Tensor] = None,
        gt_normals: Optional[Tensor] = None,
        plane_normals: Optional[Tensor] = None,
        plane_offsets: Optional[Tensor] = None,
        sym_confidences: Optional[Tensor] = None,
        diffusion_loss: Optional[Tensor] = None,
    ) -> dict:
        """
        Compute all losses.

        Returns
        -------
        dict with 'loss' (total), plus each component.
        """
        losses: dict = {}

        # Chamfer L1 (primary)
        cd = chamfer_distance_l1(pred, full)
        losses["cd_l1"] = cd

        # KL
        kl = kl_divergence(mu, logvar)
        losses["kl"] = kl

        # Fidelity
        fidel = fidelity_loss(partial, pred)
        losses["fidelity"] = fidel

        # Normal consistency (optional)
        if pred_normals is not None and gt_normals is not None:
            nc = normal_consistency_loss(pred_normals, gt_normals)
            losses["normal_con"] = nc
        else:
            losses["normal_con"] = torch.zeros(1, device=pred.device).squeeze()

        # Symmetry (optional)
        if (plane_normals is not None and plane_offsets is not None
                and sym_confidences is not None):
            sym = bse_loss(pred, plane_normals, plane_offsets, sym_confidences,
                           threshold=self.symmetry_threshold)
            losses["symmetry"] = sym
        else:
            losses["symmetry"] = torch.zeros(1, device=pred.device).squeeze()

        # Diffusion loss (passed in from DDPM training step)
        if diffusion_loss is not None:
            losses["diffusion"] = diffusion_loss
        else:
            losses["diffusion"] = torch.zeros(1, device=pred.device).squeeze()

        # Total
        total = (
            self.weights["cd_l1"] * losses["cd_l1"]
            + self.kl_beta * losses["kl"]
            + self.weights["fidelity"] * losses["fidelity"]
            + self.weights["normal_con"] * losses["normal_con"]
            + self.weights["symmetry"] * losses["symmetry"]
            + self.weights["diffusion"] * losses["diffusion"]
        )
        losses["loss"] = total

        return losses
