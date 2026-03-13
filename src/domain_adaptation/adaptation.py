"""
Domain adaptation for RELIC — Phase 2 (adversarial GRL) and Phase 3 (masked autoencoding).

GradientReversalLayer       — autograd gradient reversal for domain-adversarial training
DomainClassifier            — MLP on z_global → domain logit
AdversarialAdaptation       — Phase 2: GRL adversarial encoder alignment
MaskedAutoencodingAdaptation— Phase 3: self-supervised masked autoencoding
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import Function
from torch.optim import AdamW
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Gradient Reversal Layer
# ---------------------------------------------------------------------------

class _GRLFunction(Function):
    """
    Autograd function for gradient reversal.

    Forward: identity
    Backward: multiply gradient by -lambda
    """

    @staticmethod
    def forward(ctx, x: Tensor, lam: float) -> Tensor:
        ctx.save_for_backward(torch.tensor(lam))
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        (lam,) = ctx.saved_tensors
        return -lam * grad_output, None


class GradientReversalLayer(nn.Module):
    """
    Gradient Reversal Layer (Ganin et al., JMLR 2016).

    During the backward pass, gradients are scaled by -lambda, causing the
    encoder to learn domain-invariant representations.

    Parameters
    ----------
    lam : float   gradient reversal coefficient (default 1.0)
    """

    def __init__(self, lam: float = 1.0) -> None:
        super().__init__()
        self.lam = lam

    def forward(self, x: Tensor) -> Tensor:
        return _GRLFunction.apply(x, self.lam)

    def set_lambda(self, lam: float) -> None:
        self.lam = lam


# ---------------------------------------------------------------------------
# Domain Classifier
# ---------------------------------------------------------------------------

class DomainClassifier(nn.Module):
    """
    MLP domain classifier operating on z_global.

    Predicts whether a sample is from the real scan domain (1) or
    the synthetic / proxy domain (0).

    Parameters
    ----------
    z_dim    : int   z_global dimension (input)
    hidden_dim: int  hidden layer size
    """

    def __init__(self, z_dim: int = 256, hidden_dim: int = 256) -> None:
        super().__init__()
        self.grl = GradientReversalLayer(lam=1.0)
        self.classifier = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, z_global: Tensor) -> Tensor:
        """
        Parameters
        ----------
        z_global : Tensor[B, z_dim]

        Returns
        -------
        Tensor[B, 1]   domain logit (pre-sigmoid)
        """
        z_reversed = self.grl(z_global)
        return self.classifier(z_reversed)


# ---------------------------------------------------------------------------
# Phase 2: Adversarial Adaptation
# ---------------------------------------------------------------------------

class AdversarialAdaptation:
    """
    Phase 2 domain adaptation: adversarial encoder alignment via GRL.

    Algorithm:
    1. Freeze decoder.
    2. Add DomainClassifier on top of encoder z_global output.
    3. Train encoder + domain_classifier with binary CE loss.
       The GRL reverses gradients to the encoder, making it domain-invariant.
    4. Real scan label = 1; synthetic label = 0.

    Parameters
    ----------
    model           : RELIC model (with .vae.encoder attribute)
    config          : training configuration dict
    real_loader     : DataLoader of real (unpaired) MorphoSource scans
    fake_loader     : DataLoader of synthetic point clouds
    device          : torch device
    """

    def __init__(
        self,
        model: nn.Module,
        config: dict,
        real_loader: DataLoader,
        fake_loader: DataLoader,
        device: torch.device,
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.config = config

        z_dim = config.get("vae", {}).get("z_global_dim", 256)
        self.domain_classifier = DomainClassifier(z_dim=z_dim).to(device)

        # Freeze decoder; only encoder and domain classifier are trained
        for name, param in model.named_parameters():
            if "vae.decoder" in name or "vae.mu_head" not in name and "vae.logvar_head" not in name:
                if "vae.encoder" not in name and "vae.symmetry_module" not in name:
                    param.requires_grad = False

        da_cfg = config.get("domain_adaptation", {})
        lr = float(da_cfg.get("lr", 1e-4))
        trainable = (
            list(model.vae.encoder.parameters())
            + list(self.domain_classifier.parameters())
        )
        self.optimizer = AdamW(trainable, lr=lr, weight_decay=1e-4)
        self.epochs = int(da_cfg.get("phase2_epochs", 12))
        self.real_loader = real_loader
        self.fake_loader = fake_loader

        self.lam_schedule = da_cfg.get("grl_lambda_max", 1.0)

    def _compute_lambda(self, step: int, total: int) -> float:
        """
        Ramp GRL lambda from 0 to lam_max using the Ganin et al. schedule:
        λ = 2 / (1 + exp(-10 * p)) - 1  where p = step / total
        """
        import math
        p = step / max(total, 1)
        return float(self.lam_schedule * (2 / (1 + math.exp(-10 * p)) - 1))

    def train_phase2(self) -> None:
        """Run Phase 2 adversarial domain adaptation for self.epochs epochs."""
        logger.info("Starting Phase 2 adversarial adaptation (%d epochs)", self.epochs)
        self.model.train()
        self.domain_classifier.train()

        total_steps = self.epochs * min(len(self.real_loader), len(self.fake_loader))
        step = 0

        for epoch in range(self.epochs):
            real_iter = iter(self.real_loader)
            fake_iter = iter(self.fake_loader)
            epoch_loss = 0.0
            n_batches = 0

            for real_batch, fake_batch in zip(real_iter, fake_iter):
                real_pts = real_batch["partial"].to(self.device)
                fake_pts = fake_batch["partial"].to(self.device)

                # Update GRL lambda
                lam = self._compute_lambda(step, total_steps)
                self.domain_classifier.grl.set_lambda(lam)

                self.optimizer.zero_grad(set_to_none=True)

                # Encode real scan
                z_real, _, _, _, _ = self.model.vae.encode(real_pts, None)
                logit_real = self.domain_classifier(z_real)
                label_real = torch.ones(real_pts.shape[0], 1, device=self.device)
                loss_real = F.binary_cross_entropy_with_logits(logit_real, label_real)

                # Encode synthetic
                z_fake, _, _, _, _ = self.model.vae.encode(fake_pts, None)
                logit_fake = self.domain_classifier(z_fake)
                label_fake = torch.zeros(fake_pts.shape[0], 1, device=self.device)
                loss_fake = F.binary_cross_entropy_with_logits(logit_fake, label_fake)

                loss = loss_real + loss_fake
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.model.vae.encoder.parameters())
                    + list(self.domain_classifier.parameters()),
                    1.0,
                )
                self.optimizer.step()

                epoch_loss += float(loss.item())
                n_batches += 1
                step += 1

            logger.info(
                "[Phase 2 Epoch %d/%d] loss=%.4f",
                epoch + 1, self.epochs, epoch_loss / max(n_batches, 1),
            )

        logger.info("Phase 2 adversarial adaptation complete.")


# ---------------------------------------------------------------------------
# Phase 3: Masked Autoencoding Adaptation
# ---------------------------------------------------------------------------

class MaskedAutoencodingAdaptation:
    """
    Phase 3 self-supervised domain adaptation via masked autoencoding.

    Algorithm:
    1. Randomly mask 40% of real scan points.
    2. Run masked partial through the full RELIC model.
    3. Reconstruction loss: CD between output and unmasked input.
    4. Optional consistency loss: CD between RELIC(mask_A(x)) and RELIC(mask_B(x))
       when multiple scans of the same specimen are available.
    5. No ground-truth complete shape required.

    Parameters
    ----------
    model        : RELIC model
    config       : training configuration dict
    real_loader  : DataLoader of real (unpaired) point clouds
                   If specimens have multiple scans, metadata["specimen_id"]
                   should match across samples for consistency loss.
    device       : torch device
    """

    def __init__(
        self,
        model: nn.Module,
        config: dict,
        real_loader: DataLoader,
        device: torch.device,
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.config = config

        da_cfg = config.get("domain_adaptation", {})
        self.mask_ratio = float(da_cfg.get("mask_ratio", 0.4))
        self.epochs = int(da_cfg.get("phase3_epochs", 7))
        self.consistency_weight = float(da_cfg.get("consistency_weight", 1.0))
        self.real_loader = real_loader

        lr = float(da_cfg.get("lr", 5e-5))
        self.optimizer = AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr, weight_decay=1e-4,
        )

    @staticmethod
    def _mask_points(points: Tensor, mask_ratio: float) -> Tensor:
        """Randomly remove `mask_ratio` fraction of points."""
        N = points.shape[1]   # [B, N, 3]
        n_keep = max(int(N * (1 - mask_ratio)), 64)
        idx = torch.randperm(N, device=points.device)[:n_keep]
        return points[:, idx, :]

    def train_phase3(self) -> None:
        """Run Phase 3 masked autoencoding for self.epochs epochs."""
        logger.info("Starting Phase 3 masked autoencoding (%d epochs)", self.epochs)
        self.model.train()

        from src.training.losses import chamfer_distance_l1

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            n_batches = 0

            for batch in self.real_loader:
                real_pts = batch["partial"].to(self.device)   # [B, N, 3]
                meta = batch.get("metadata", {})

                self.optimizer.zero_grad(set_to_none=True)

                # Apply mask A
                masked_A = self._mask_points(real_pts, self.mask_ratio)
                out_A = self.model(masked_A, mode="inference")
                pred_A = out_A["pred"]

                # Reconstruction loss: CD between pred and unmasked input
                # Use the unmasked real_pts as pseudo-ground-truth
                recon_loss = chamfer_distance_l1(pred_A, real_pts)
                total_loss = recon_loss

                # Optional consistency loss (mask B → should produce similar result)
                specimen_ids = meta.get("specimen_id", None)
                if specimen_ids is not None:
                    # Find specimens with duplicate IDs in this batch
                    id_list = specimen_ids if isinstance(specimen_ids, list) else specimen_ids.tolist()
                    from collections import Counter
                    dup_ids = {k for k, v in Counter(id_list).items() if v > 1}
                    if dup_ids:
                        masked_B = self._mask_points(real_pts, self.mask_ratio)
                        out_B = self.model(masked_B, mode="inference")
                        pred_B = out_B["pred"]
                        consistency_loss = chamfer_distance_l1(pred_A.detach(), pred_B)
                        total_loss = total_loss + self.consistency_weight * consistency_loss

                total_loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                epoch_loss += float(total_loss.item())
                n_batches += 1

            logger.info(
                "[Phase 3 Epoch %d/%d] loss=%.4f",
                epoch + 1, self.epochs, epoch_loss / max(n_batches, 1),
            )

        logger.info("Phase 3 masked autoencoding adaptation complete.")
