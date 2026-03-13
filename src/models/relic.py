"""
RELIC — Reconstruction of Extinct Life via Informed Completion.

Full two-stage model:
  Stage 1: HierarchicalVAE (encoder + decoder)
  Stage 2: LatentDiffusionModel (DDPM in z_global space)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from src.models.vae import HierarchicalVAE
from src.models.diffusion import LatentDiffusionModel
from src.models.conditioning import TaxonomyEncoder
from src.models.symmetry import SymmetryLoss


class RELIC(nn.Module):
    """
    Full RELIC model.

    Parameters
    ----------
    config : dict
        Full experiment configuration (see experiments/relic_full.yaml).
        Expected structure:
            model.vae.*
            model.diffusion.*
            model.conditioning.*
            model.symmetry.*
    """

    def __init__(self, config: dict) -> None:
        super().__init__()
        model_cfg = config
        vae_cfg = model_cfg.get("vae", {})
        diff_cfg = model_cfg.get("diffusion", {})
        cond_cfg = model_cfg.get("conditioning", {})
        sym_cfg = model_cfg.get("symmetry", {})

        z_global_dim = vae_cfg.get("z_global_dim", 256)
        cond_dim = cond_cfg.get("out_dim", 256)

        # Ensure decoder and diffusion know about z_global_dim and cond_dim
        vae_cfg.setdefault("z_global_dim", z_global_dim)
        vae_cfg.setdefault("cond_dim", cond_dim)
        diff_cfg.setdefault("z_global_dim", z_global_dim)
        diff_cfg.setdefault("cond_dim", cond_dim)

        # Sub-modules
        self.vae = HierarchicalVAE(vae_cfg)

        self.diffusion = LatentDiffusionModel(diff_cfg)

        # Conditioning encoder
        phylo_enabled = cond_cfg.get("phylo_embedding", "poincare") != "disabled"
        morpho_enabled = cond_cfg.get("morpho_encoder", "biobert") != "disabled"
        image_enabled = cond_cfg.get("image_encoder", "clip") != "disabled"

        self.taxonomy_encoder = TaxonomyEncoder(
            n_taxa=cond_cfg.get("n_taxa", 1024),
            out_dim=cond_dim,
            taxa_list=cond_cfg.get("taxa_list", None),
            phylo_dim=cond_cfg.get("phylo_dim", 64),
            morpho_dim=cond_cfg.get("morpho_dim", 256),
            image_dim=cond_cfg.get("image_dim", 256),
        )

        # Symmetry loss
        sym_enabled = sym_cfg.get("enabled", True)
        sym_threshold = sym_cfg.get("confidence_threshold", 0.25)
        self.sym_loss_fn = SymmetryLoss(confidence_threshold=sym_threshold) if sym_enabled else None

        self.z_global_dim = z_global_dim
        self.guidance_scale = diff_cfg.get("guidance_scale", 3.0)
        self.ddim_steps = diff_cfg.get("ddim_steps", 20)

    # ------------------------------------------------------------------
    # Core building blocks
    # ------------------------------------------------------------------

    def encode(
        self,
        partial_points: Tensor,
        conditioning: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, dict]:
        """
        Encode partial_points through the VAE encoder.

        Returns
        -------
        z_global    : Tensor[B, z_global_dim]
        z_local     : Tensor[B, K, z_local_dim]
        sym_info    : dict
        """
        z_global, z_local, mu, logvar, sym_info = self.vae.encode(
            partial_points, conditioning
        )
        return z_global, z_local, sym_info

    def decode(
        self,
        z_global: Tensor,
        z_local: Tensor,
        conditioning: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Decode (z_global, z_local) → complete point cloud [B, M, 3].
        """
        return self.vae.decode(z_global, z_local, conditioning)

    def get_conditioning(
        self,
        taxon_ids: Optional[Tensor] = None,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        pixel_values: Optional[Tensor] = None,
        use_null: bool = False,
    ) -> Optional[Tensor]:
        """Encode conditioning inputs → Tensor[B, cond_dim]."""
        if all(x is None for x in [taxon_ids, input_ids, pixel_values]) and not use_null:
            return None
        return self.taxonomy_encoder(
            taxon_ids=taxon_ids,
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            use_null_conditioning=use_null,
        )

    # ------------------------------------------------------------------
    # Training forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        partial: Tensor,
        full: Optional[Tensor] = None,
        conditioning: Optional[Dict] = None,
        mode: str = "train",
    ) -> dict:
        """
        Unified forward pass.

        Parameters
        ----------
        partial      : Tensor[B, N, 3]
        full         : Tensor[B, M, 3] or None
        conditioning : dict with optional keys:
                         taxon_ids, input_ids, attention_mask, pixel_values
        mode         : "train" | "inference"

        Returns
        -------
        train mode:   dict with losses
        inference mode: dict with pred (best completion)
        """
        # Build conditioning tensor
        cond_tensor = None
        if conditioning is not None:
            cond_tensor = self.get_conditioning(
                taxon_ids=conditioning.get("taxon_ids"),
                input_ids=conditioning.get("input_ids"),
                attention_mask=conditioning.get("attention_mask"),
                pixel_values=conditioning.get("pixel_values"),
            )

        if mode == "train":
            return self._train_forward(partial, full, cond_tensor)
        else:
            return self._inference_forward(partial, cond_tensor)

    def _train_forward(
        self,
        partial: Tensor,
        full: Optional[Tensor],
        cond_tensor: Optional[Tensor],
    ) -> dict:
        """Training forward: returns losses dict."""
        from src.training.losses import (
            chamfer_distance_l1, chamfer_distance_l2, fidelity_loss,
            kl_divergence, bse_loss,
        )

        # VAE forward
        vae_out = self.vae(partial, full, cond_tensor)
        pred = vae_out["pred"]           # [B, M, 3]
        mu = vae_out["mu"]
        logvar = vae_out["logvar"]
        z_global = vae_out["z_global"]
        sym_info = vae_out["sym_info"]

        losses = {}

        # VAE reconstruction losses
        if full is not None:
            cd_l1 = chamfer_distance_l1(pred, full)
            cd_l2 = chamfer_distance_l2(pred, full)
            losses["cd_l1"] = cd_l1
            losses["cd_l2"] = cd_l2
        else:
            cd_l1 = torch.zeros(1, device=partial.device).squeeze()
            losses["cd_l1"] = cd_l1

        # KL divergence
        kl = kl_divergence(mu, logvar)
        losses["kl"] = kl

        # Fidelity loss
        fidel = fidelity_loss(partial, pred)
        losses["fidelity"] = fidel

        # Symmetry loss
        if self.sym_loss_fn is not None and sym_info:
            sym_l = self.sym_loss_fn(
                pred,
                sym_info["plane_normals"],
                sym_info["plane_offsets"],
                sym_info["confidences"],
            )
            losses["symmetry"] = sym_l
        else:
            losses["symmetry"] = torch.zeros(1, device=partial.device).squeeze()

        # Diffusion loss (z_global)
        diff_loss = self.diffusion.compute_loss(z_global.detach(), cond_tensor)
        losses["diffusion"] = diff_loss

        # Combined total loss
        losses["loss"] = (
            1.0 * losses["cd_l1"]
            + 0.01 * losses["kl"]
            + 0.5 * losses["fidelity"]
            + 0.1 * losses["symmetry"]
            + 1.0 * losses["diffusion"]
        )
        losses["pred"] = pred

        return losses

    def _inference_forward(
        self,
        partial: Tensor,
        cond_tensor: Optional[Tensor],
    ) -> dict:
        """Inference forward: single deterministic sample."""
        B = partial.shape[0]
        device = partial.device

        # Encode partial to get z_local and sym_info
        z_global_enc, z_local, sym_info = self.encode(partial, cond_tensor)

        # Sample z_global from diffusion
        z_global_sampled = self.diffusion.sample(
            batch_size=B,
            conditioning=cond_tensor,
            guidance_scale=self.guidance_scale,
            device=device,
        )

        pred = self.decode(z_global_sampled, z_local, cond_tensor)
        return {"pred": pred, "sym_info": sym_info}

    # ------------------------------------------------------------------
    # Multi-sample inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        partial: Tensor,
        conditioning: Optional[Dict] = None,
        n_samples: int = 5,
    ) -> Tensor:
        """
        Generate n_samples diverse completions for each item in the batch.

        Parameters
        ----------
        partial   : Tensor[B, N, 3]
        conditioning : dict or None
        n_samples : int

        Returns
        -------
        Tensor[B, n_samples, M, 3]
        """
        B = partial.shape[0]
        device = partial.device

        cond_tensor = None
        if conditioning is not None:
            cond_tensor = self.get_conditioning(**conditioning)

        # Encode once to get z_local
        _, z_local, sym_info = self.encode(partial, cond_tensor)

        # Tile for n_samples
        partial_tiled = partial.unsqueeze(1).expand(-1, n_samples, -1, -1).reshape(
            B * n_samples, partial.shape[1], 3
        )
        z_local_tiled = z_local.unsqueeze(1).expand(-1, n_samples, -1, -1).reshape(
            B * n_samples, z_local.shape[1], z_local.shape[2]
        )
        cond_tiled = None
        if cond_tensor is not None:
            cond_tiled = cond_tensor.unsqueeze(1).expand(-1, n_samples, -1).reshape(
                B * n_samples, cond_tensor.shape[-1]
            )

        # Sample n_samples z_globals via DDIM
        z_globals = self.diffusion.sample(
            batch_size=B * n_samples,
            conditioning=cond_tiled,
            guidance_scale=self.guidance_scale,
            device=device,
        )

        completions = self.decode(z_globals, z_local_tiled, cond_tiled)
        M = completions.shape[1]
        return completions.reshape(B, n_samples, M, 3)

    def get_uncertainty_map(
        self,
        samples: Tensor,
    ) -> Tensor:
        """
        Compute per-point uncertainty as the std of nearest-neighbour distances
        across diffusion samples.

        Parameters
        ----------
        samples : Tensor[n_samples, M, 3]  (for a single specimen)

        Returns
        -------
        uncertainty : Tensor[M]   per-point std across samples
        """
        # Per-point std across sample dimension
        # Each point in the reference (sample[0]) → measure dist to nearest in each sample
        ref = samples[0]     # [M, 3]
        n_samples = samples.shape[0]

        per_point_dists = []
        for s in range(1, n_samples):
            # Distance from each ref point to nearest point in sample s
            dists = torch.cdist(ref.unsqueeze(0), samples[s].unsqueeze(0))[0]  # [M, M]
            min_dists = dists.min(dim=-1).values  # [M]
            per_point_dists.append(min_dists)

        if not per_point_dists:
            return torch.zeros(ref.shape[0], device=ref.device)

        per_point_dists = torch.stack(per_point_dists, dim=0)  # [n_samples-1, M]
        return per_point_dists.std(dim=0)                        # [M]
