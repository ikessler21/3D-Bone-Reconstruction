"""
PyTorch Dataset classes for fossil bone point cloud completion.

FossilDataset       — general dataset of processed .pt files
PaleoCompleteDataset — PaleoComplete benchmark (L1/L2/L3 splits)
collate_fn          — handles variable-size point clouds via fixed-N sampling
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Helper: fixed-N resampling of a point cloud
# ---------------------------------------------------------------------------

def _resample(points: Tensor, n: int) -> Tensor:
    """Resample `points` (shape [P, 3]) to exactly `n` points.

    Uses random sub-sampling when P > n, and random repetition when P < n.
    """
    p = points.shape[0]
    if p == n:
        return points
    if p > n:
        idx = torch.randperm(p)[:n]
        return points[idx]
    # p < n: pad with random repeats
    repeat_idx = torch.randint(0, p, (n - p,))
    return torch.cat([points, points[repeat_idx]], dim=0)


# ---------------------------------------------------------------------------
# FossilDataset
# ---------------------------------------------------------------------------

class FossilDataset(Dataset):
    """
    Loads processed .pt files containing partial and full point clouds.

    Each .pt file is expected to be a dict with at minimum:
        partial  : Tensor[N, 3]  — fragmentary input
        full     : Tensor[M, 3]  — complete ground-truth
        metadata : dict          — taxon, bone_type, etc.

    Parameters
    ----------
    processed_dir : str | Path
        Directory containing .pt files.
    n_partial : int
        Fixed number of points to resample partial clouds to.
    n_full : int
        Fixed number of points to resample full clouds to.
    bone_types : list[str] | None
        If set, only include specimens whose metadata["bone_type"] is in this list.
    taxon_orders : list[str] | None
        If set, filter by metadata["taxon_order"].
    min_confidence : float
        Minimum completeness_confidence score to include a specimen.
    ontogenetic_stage : str | None
        Filter by ontogenetic stage (e.g., "adult").
    transform : callable | None
        Optional augmentation callable applied to the partial cloud.
    """

    COMPLETENESS_LEVELS = {"L1", "L2", "L3"}

    def __init__(
        self,
        processed_dir: str | Path,
        n_partial: int = 2048,
        n_full: int = 4096,
        bone_types: Optional[List[str]] = None,
        taxon_orders: Optional[List[str]] = None,
        min_confidence: float = 0.5,
        ontogenetic_stage: Optional[str] = None,
        completeness_levels: Optional[List[str]] = None,
        transform=None,
    ) -> None:
        self.processed_dir = Path(processed_dir)
        self.n_partial = n_partial
        self.n_full = n_full
        self.bone_types = set(bone_types) if bone_types else None
        self.taxon_orders = set(taxon_orders) if taxon_orders else None
        self.min_confidence = min_confidence
        self.ontogenetic_stage = ontogenetic_stage
        self.completeness_levels = set(completeness_levels) if completeness_levels else None
        self.transform = transform

        self.samples: List[Path] = self._collect_samples()

    def _collect_samples(self) -> List[Path]:
        """Scan processed_dir for .pt files and apply metadata filters."""
        all_pt = sorted(self.processed_dir.glob("**/*.pt"))
        filtered: List[Path] = []
        for pt_path in all_pt:
            try:
                data = torch.load(pt_path, map_location="cpu", weights_only=False)
            except Exception:
                continue
            meta = data.get("metadata", {})

            # Confidence filter
            conf = float(meta.get("completeness_confidence", 0.0))
            if conf < self.min_confidence:
                continue

            # Bone type filter
            if self.bone_types is not None:
                if meta.get("bone_type", "") not in self.bone_types:
                    continue

            # Taxon order filter
            if self.taxon_orders is not None:
                if meta.get("taxon_order", "") not in self.taxon_orders:
                    continue

            # Ontogenetic stage filter
            if self.ontogenetic_stage is not None:
                if meta.get("ontogenetic_stage", "").lower() != self.ontogenetic_stage.lower():
                    continue

            # Completeness level filter
            if self.completeness_levels is not None:
                if meta.get("completeness_level", "") not in self.completeness_levels:
                    continue

            filtered.append(pt_path)
        return filtered

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        pt_path = self.samples[idx]
        data = torch.load(pt_path, map_location="cpu", weights_only=False)

        partial: Tensor = data["partial"].float()  # [N, 3]
        full: Tensor = data["full"].float()        # [M, 3]
        meta: dict = data.get("metadata", {})

        # Fixed-size resampling
        partial = _resample(partial, self.n_partial)
        full = _resample(full, self.n_full)

        # Optional augmentation on partial cloud only
        if self.transform is not None:
            partial = self.transform(partial)
            partial = _resample(partial, self.n_partial)  # re-normalize after aug (some transforms add/remove points)

        return {"partial": partial, "full": full, "metadata": meta}


# ---------------------------------------------------------------------------
# PaleoCompleteDataset
# ---------------------------------------------------------------------------

class PaleoCompleteDataset(FossilDataset):
    """
    Subclass of FossilDataset for the PaleoComplete benchmark.

    Loads specimens from the paleocomplete/ directory using official split
    JSON files (train_split.json / val_split.json / test_split.json).

    Parameters
    ----------
    paleocomplete_dir : str | Path
        Root of the paleocomplete benchmark directory.
    split : str
        One of "train", "val", "test".
    completeness_level : str | list[str] | None
        "L1", "L2", "L3", or a list; None means all levels.
    """

    def __init__(
        self,
        paleocomplete_dir: str | Path,
        split: str = "train",
        completeness_level: Optional[str | List[str]] = None,
        n_partial: int = 2048,
        n_full: int = 4096,
        bone_types: Optional[List[str]] = None,
        taxon_orders: Optional[List[str]] = None,
        min_confidence: float = 0.0,
        transform=None,
    ) -> None:
        self.paleocomplete_dir = Path(paleocomplete_dir)
        self.split = split

        # Normalise completeness_level
        if completeness_level is None:
            levels = None
        elif isinstance(completeness_level, str):
            levels = [completeness_level]
        else:
            levels = list(completeness_level)

        super().__init__(
            processed_dir=self.paleocomplete_dir / split,
            n_partial=n_partial,
            n_full=n_full,
            bone_types=bone_types,
            taxon_orders=taxon_orders,
            min_confidence=min_confidence,
            completeness_levels=levels,
            transform=transform,
        )

        # Prefer loading from official split JSON if present (overrides glob)
        split_json = self.paleocomplete_dir / f"{split}_split.json"
        if split_json.exists():
            self.samples = self._load_from_split_json(split_json, levels)

    def _load_from_split_json(
        self,
        split_json: Path,
        completeness_levels: Optional[List[str]],
    ) -> List[Path]:
        """Load sample paths from the official split manifest."""
        with open(split_json) as fh:
            records: List[dict] = json.load(fh)
        paths: List[Path] = []
        for rec in records:
            if completeness_levels is not None:
                if rec.get("completeness_level", "") not in completeness_levels:
                    continue
            pt_path = self.paleocomplete_dir / rec["pt_path"]
            if pt_path.exists():
                paths.append(pt_path)
        return paths


# ---------------------------------------------------------------------------
# collate_fn
# ---------------------------------------------------------------------------

def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate a list of sample dicts into batched tensors.

    Point clouds are already fixed-size (handled by __getitem__),
    so we simply stack them. Metadata dicts are kept as a list of dicts.
    """
    partials = torch.stack([item["partial"] for item in batch], dim=0)   # [B, N, 3]
    fulls = torch.stack([item["full"] for item in batch], dim=0)         # [B, M, 3]
    metadatas = [item["metadata"] for item in batch]

    # Build a batched metadata dict with lists of values
    batched_meta: Dict[str, List[Any]] = {}
    for meta in metadatas:
        for k, v in meta.items():
            batched_meta.setdefault(k, []).append(v)

    return {
        "partial": partials,
        "full": fulls,
        "metadata": batched_meta,
    }
