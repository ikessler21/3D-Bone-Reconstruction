"""
Training loop for RELIC.

Trainer          — base trainer (VAE + diffusion joint or sequential)
BonePretrainer   — Stage 1 VAE-only pre-training on proxy bone datasets
DiffusionTrainer — Stage 2 diffusion-only training with frozen VAE encoder
"""

from __future__ import annotations

import argparse
import heapq
import logging
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch import Tensor
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """
    Full training loop for RELIC.

    Parameters
    ----------
    model        : nn.Module  (RELIC or sub-module)
    config       : dict       experiment configuration
    train_loader : DataLoader
    val_loader   : DataLoader
    device       : torch.device
    """

    def __init__(
        self,
        model: nn.Module,
        config: dict,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
    ) -> None:
        self.model = model.to(device)
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        train_cfg = config.get("training", config)

        # Optimiser
        lr = float(train_cfg.get("lr", 1e-4))
        weight_decay = float(train_cfg.get("weight_decay", 1e-4))
        self.optimizer = AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr,
            weight_decay=weight_decay,
        )

        # LR scheduler: cosine with warmup
        self.num_epochs = int(train_cfg.get("epochs", 50))
        warmup_epochs = int(train_cfg.get("warmup_epochs", 5))
        self.warmup_steps = warmup_epochs * max(1, len(train_loader))
        total_steps = self.num_epochs * max(1, len(train_loader))
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, total_steps - self.warmup_steps),
            eta_min=float(train_cfg.get("lr_min", 1e-6)),
        )

        # Mixed precision
        self.use_amp = train_cfg.get("mixed_precision", True) and device.type == "cuda"
        self.scaler = GradScaler(enabled=self.use_amp)

        # Gradient accumulation
        self.grad_accum_steps = int(train_cfg.get("grad_accum_steps", 1))
        self.clip_grad_norm = float(train_cfg.get("clip_grad_norm", 1.0))

        # Checkpointing
        self.ckpt_dir = Path(train_cfg.get("checkpoint_dir", "checkpoints"))
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.save_every = int(train_cfg.get("save_every", 5))
        self.top_k = int(train_cfg.get("top_k_checkpoints", 3))
        self._best_ckpts: List[tuple] = []   # (val_cd, epoch, path) — min-heap

        # WandB
        self.use_wandb = train_cfg.get("use_wandb", False)
        if self.use_wandb:
            try:
                import wandb
                wandb.init(
                    project=train_cfg.get("wandb_project", "relic-fossil"),
                    config=config,
                    name=train_cfg.get("run_name", None),
                )
                self.wandb = wandb
            except Exception as exc:
                logger.warning("WandB init failed: %s. Disabling.", exc)
                self.use_wandb = False

        # Loss function
        from src.training.losses import RELICLoss
        self.loss_fn = RELICLoss(
            weights=train_cfg.get("loss_weights", None),
            beta_max=float(train_cfg.get("beta_max", 0.01)),
            beta_warmup=int(train_cfg.get("beta_warmup", 5000)),
        )

        self.global_step = 0

    # ------------------------------------------------------------------
    # LR warmup
    # ------------------------------------------------------------------

    def _warmup_lr(self) -> None:
        if self.global_step < self.warmup_steps:
            lr_scale = (self.global_step + 1) / max(self.warmup_steps, 1)
            base_lr = self.config.get("training", self.config).get("lr", 1e-4)
            for pg in self.optimizer.param_groups:
                pg["lr"] = float(base_lr) * lr_scale

    # ------------------------------------------------------------------
    # Train epoch
    # ------------------------------------------------------------------

    def train_epoch(self) -> dict:
        """Run one training epoch. Returns dict of mean metrics."""
        self.model.train()
        total_metrics: Dict[str, float] = {}
        n_batches = 0

        self.optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(self.train_loader):
            partial = batch["partial"].to(self.device, non_blocking=True)
            full = batch["full"].to(self.device, non_blocking=True)
            meta = batch.get("metadata", {})

            # Build conditioning dict if metadata available
            conditioning = self._build_conditioning(meta)

            with autocast(enabled=self.use_amp):
                out = self.model(
                    partial, full, conditioning=conditioning, mode="train"
                )
                loss = out["loss"] / self.grad_accum_steps

            self.scaler.scale(loss).backward()

            if (batch_idx + 1) % self.grad_accum_steps == 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

                self._warmup_lr()
                if self.global_step >= self.warmup_steps:
                    self.scheduler.step()
                self.loss_fn.step()
                self.global_step += 1

            # Accumulate metrics
            for k, v in out.items():
                if k == "pred":
                    continue
                if isinstance(v, Tensor):
                    v = float(v.detach().item())
                total_metrics[k] = total_metrics.get(k, 0.0) + v
            n_batches += 1

        mean_metrics = {k: v / max(n_batches, 1) for k, v in total_metrics.items()}

        if self.use_wandb:
            self.wandb.log(
                {f"train/{k}": v for k, v in mean_metrics.items()},
                step=self.global_step,
            )

        return mean_metrics

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def validate(self) -> dict:
        """Validation loop. Returns dict of mean metrics."""
        self.model.eval()
        total_metrics: Dict[str, float] = {}
        n_batches = 0

        for batch in self.val_loader:
            partial = batch["partial"].to(self.device, non_blocking=True)
            full = batch["full"].to(self.device, non_blocking=True)
            meta = batch.get("metadata", {})
            conditioning = self._build_conditioning(meta)

            with autocast(enabled=self.use_amp):
                out = self.model(
                    partial, full, conditioning=conditioning, mode="train"
                )

            for k, v in out.items():
                if k == "pred":
                    continue
                if isinstance(v, Tensor):
                    v = float(v.item())
                total_metrics[k] = total_metrics.get(k, 0.0) + v
            n_batches += 1

        mean_metrics = {k: v / max(n_batches, 1) for k, v in total_metrics.items()}

        if self.use_wandb:
            self.wandb.log(
                {f"val/{k}": v for k, v in mean_metrics.items()},
                step=self.global_step,
            )

        return mean_metrics

    # ------------------------------------------------------------------
    # Full training loop
    # ------------------------------------------------------------------

    def train(self, num_epochs: Optional[int] = None) -> None:
        """Run full training for num_epochs (or self.num_epochs)."""
        n = num_epochs or self.num_epochs
        logger.info("Starting training for %d epochs", n)

        for epoch in range(n):
            t0 = time.time()
            train_metrics = self.train_epoch()
            val_metrics = self.validate()

            elapsed = time.time() - t0
            val_cd = val_metrics.get("cd_l1", float("inf"))

            logger.info(
                "[Epoch %3d/%d] train_loss=%.4f  val_cd=%.4f  lr=%.2e  time=%.1fs",
                epoch + 1, n,
                train_metrics.get("loss", 0),
                val_cd,
                self.optimizer.param_groups[0]["lr"],
                elapsed,
            )

            # Periodic checkpoint
            if (epoch + 1) % self.save_every == 0:
                self._save_checkpoint(epoch, val_cd, prefix="periodic")

            # Top-K checkpointing by val CD
            self._maybe_save_best(epoch, val_cd)

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(
        self, epoch: int, val_cd: float, prefix: str = "ckpt"
    ) -> Path:
        path = self.ckpt_dir / f"{prefix}_epoch{epoch+1:03d}_cd{val_cd:.4f}.pth"
        torch.save(
            {
                "epoch": epoch,
                "global_step": self.global_step,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "val_cd": val_cd,
                "config": self.config,
            },
            str(path),
        )
        logger.info("Saved checkpoint to %s", path)
        return path

    def _maybe_save_best(self, epoch: int, val_cd: float) -> None:
        """Save to top-K checkpoints by val_cd (lower is better)."""
        path = self._save_checkpoint(epoch, val_cd, prefix="best")
        # Min-heap with negated val_cd (so we pop the worst = highest CD)
        entry = (-val_cd, epoch, str(path))
        if len(self._best_ckpts) < self.top_k:
            heapq.heappush(self._best_ckpts, entry)
        else:
            worst = self._best_ckpts[0]
            if -val_cd > worst[0]:   # current is better (lower CD) than worst saved
                heapq.heapreplace(self._best_ckpts, entry)
                # Delete worst checkpoint file
                worst_path = Path(worst[2])
                if worst_path.exists():
                    worst_path.unlink()

    def load_checkpoint(self, path: str | Path) -> int:
        """Load a checkpoint. Returns the epoch number."""
        state = torch.load(str(path), map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["model_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        if "scheduler_state_dict" in state:
            self.scheduler.load_state_dict(state["scheduler_state_dict"])
        self.global_step = state.get("global_step", 0)
        epoch = state.get("epoch", 0)
        logger.info("Loaded checkpoint from %s (epoch %d)", path, epoch)
        return epoch

    # ------------------------------------------------------------------
    # Conditioning helper
    # ------------------------------------------------------------------

    def _build_conditioning(self, meta: dict) -> Optional[dict]:
        """Build conditioning dict from batch metadata."""
        if not meta:
            return None
        cond: dict = {}
        if "taxon_id" in meta:
            ids = meta["taxon_id"]
            if isinstance(ids, list):
                ids = torch.tensor(ids, dtype=torch.long, device=self.device)
            cond["taxon_ids"] = ids
        return cond if cond else None


# ---------------------------------------------------------------------------
# BonePretrainer (Stage 1 VAE only)
# ---------------------------------------------------------------------------

class BonePretrainer(Trainer):
    """
    Pre-trains the VAE (Stage 1) on proxy bone datasets (VerSe, BoneDat, ICL).

    Only the VAE encoder and decoder are trained; the diffusion model and
    conditioning encoders are frozen.
    """

    def __init__(
        self,
        model: nn.Module,
        config: dict,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
    ) -> None:
        super().__init__(model, config, train_loader, val_loader, device)

        # Freeze diffusion and conditioning modules
        for name, param in model.named_parameters():
            if "diffusion" in name or "taxonomy_encoder" in name:
                param.requires_grad = False

        # Re-create optimizer with only unfrozen params
        train_cfg = config.get("training", config)
        lr = float(train_cfg.get("lr", 1e-4))
        self.optimizer = AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr,
            weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
        )
        logger.info("BonePretrainer: frozen diffusion + conditioning modules.")

    def train_epoch(self) -> dict:
        """VAE-only training epoch (no diffusion loss)."""
        self.model.train()
        total_metrics: Dict[str, float] = {}
        n_batches = 0
        self.optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(self.train_loader):
            partial = batch["partial"].to(self.device, non_blocking=True)
            full = batch["full"].to(self.device, non_blocking=True)

            with autocast(enabled=self.use_amp):
                vae_out = self.model.vae(partial, full, None)
                pred = vae_out["pred"]
                mu, logvar = vae_out["mu"], vae_out["logvar"]
                sym_info = vae_out["sym_info"]

                from src.training.losses import chamfer_distance_l1, kl_divergence, fidelity_loss
                cd = chamfer_distance_l1(pred, full)
                kl = kl_divergence(mu, logvar)
                fidel = fidelity_loss(partial, pred)
                loss = (cd + self.loss_fn.kl_beta * kl + 0.5 * fidel) / self.grad_accum_steps

            self.scaler.scale(loss).backward()

            if (batch_idx + 1) % self.grad_accum_steps == 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self._warmup_lr()
                if self.global_step >= self.warmup_steps:
                    self.scheduler.step()
                self.loss_fn.step()
                self.global_step += 1

            metrics = {
                "loss": float((cd + kl + fidel).item()),
                "cd_l1": float(cd.item()),
                "kl": float(kl.item()),
                "fidelity": float(fidel.item()),
            }
            for k, v in metrics.items():
                total_metrics[k] = total_metrics.get(k, 0.0) + v
            n_batches += 1

        return {k: v / max(n_batches, 1) for k, v in total_metrics.items()}


# ---------------------------------------------------------------------------
# DiffusionTrainer (Stage 2 only)
# ---------------------------------------------------------------------------

class DiffusionTrainer(Trainer):
    """
    Trains the Stage 2 DDPM in z_global latent space.

    Encoder, symmetry module, and decoder are frozen.
    Only the DDPMDenoiser and TaxonomyEncoder are trained.
    """

    def __init__(
        self,
        model: nn.Module,
        config: dict,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
    ) -> None:
        super().__init__(model, config, train_loader, val_loader, device)

        # Freeze VAE encoder/decoder
        for name, param in model.named_parameters():
            if "vae.encoder" in name or "vae.decoder" in name:
                param.requires_grad = False

        train_cfg = config.get("training", config)
        lr = float(train_cfg.get("lr", 1e-4))
        self.optimizer = AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr,
            weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
        )
        logger.info("DiffusionTrainer: frozen VAE encoder/decoder.")

    def train_epoch(self) -> dict:
        """Diffusion-only training epoch."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        self.optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(self.train_loader):
            partial = batch["partial"].to(self.device, non_blocking=True)
            full = batch["full"].to(self.device, non_blocking=True)
            meta = batch.get("metadata", {})
            conditioning = self._build_conditioning(meta)

            with autocast(enabled=self.use_amp):
                # Get z_global from frozen encoder
                with torch.no_grad():
                    z_global, _, _, _, _ = self.model.vae.encode(partial, None)
                # Get conditioning
                cond_tensor = None
                if conditioning is not None:
                    cond_tensor = self.model.get_conditioning(**conditioning)
                # Diffusion loss
                diff_loss = self.model.diffusion.compute_loss(
                    z_global.detach(), cond_tensor
                ) / self.grad_accum_steps

            self.scaler.scale(diff_loss).backward()

            if (batch_idx + 1) % self.grad_accum_steps == 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self._warmup_lr()
                if self.global_step >= self.warmup_steps:
                    self.scheduler.step()
                self.global_step += 1

            total_loss += float(diff_loss.item())
            n_batches += 1

        return {"loss": total_loss / max(n_batches, 1), "diffusion": total_loss / max(n_batches, 1)}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Train RELIC")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--mode", type=str, default="full",
                        choices=["full", "pretrain", "diffusion"])
    args = parser.parse_args()

    with open(args.config) as fh:
        config = yaml.safe_load(fh)

    seed = config.get("seed", 42)
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # Build dataset and loaders
    from src.datasets.fossil_dataset import FossilDataset, PaleoCompleteDataset, collate_fn
    from src.datasets.augmentations import BoneAugmentation

    dataset_name = config.get("dataset", "paleocomplete")
    data_cfg = config.get("data", {})

    if dataset_name == "paleocomplete":
        train_ds = PaleoCompleteDataset(
            paleocomplete_dir=data_cfg.get("paleocomplete_dir", "data/paleocomplete"),
            split="train",
            completeness_level=config.get("completeness_level", None),
            n_partial=config.get("n_partial", 2048),
            n_full=config.get("n_points", 4096),
            transform=BoneAugmentation(),
        )
        val_ds = PaleoCompleteDataset(
            paleocomplete_dir=data_cfg.get("paleocomplete_dir", "data/paleocomplete"),
            split="val",
            n_partial=config.get("n_partial", 2048),
            n_full=config.get("n_points", 4096),
        )
    else:
        train_ds = FossilDataset(
            processed_dir=data_cfg.get("processed_dir", "data/processed"),
            n_partial=config.get("n_partial", 2048),
            n_full=config.get("n_points", 4096),
            transform=BoneAugmentation(),
        )
        val_ds = FossilDataset(
            processed_dir=data_cfg.get("val_processed_dir",
                                       data_cfg.get("processed_dir", "data/processed")),
            n_partial=config.get("n_partial", 2048),
            n_full=config.get("n_points", 4096),
        )

    batch_size = config.get("batch_size", 32)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=data_cfg.get("num_workers", 4), collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=data_cfg.get("num_workers", 4), collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    # Build model
    from src.models.relic import RELIC
    model = RELIC(config)

    # Choose trainer
    if args.mode == "pretrain":
        trainer = BonePretrainer(model, config, train_loader, val_loader, device)
    elif args.mode == "diffusion":
        trainer = DiffusionTrainer(model, config, train_loader, val_loader, device)
    else:
        trainer = Trainer(model, config, train_loader, val_loader, device)

    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)

    trainer.train()


if __name__ == "__main__":
    main()
