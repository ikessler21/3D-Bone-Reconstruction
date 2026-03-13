"""
Build the PaleoComplete benchmark from preprocessed Phenome10K .pt files.

Each specimen is augmented into L1/L2/L3 partial versions using proxy
PCA bounding-box landmarks (±x, ±y, ±z extrema).  Tier A specimens
(completeness_confidence ≥ 0.8) go to test only; Tier B (0.5–0.8) are
stratified by bone_type into 70/15/15 train/val/test splits.

Usage:
    python scripts/construct_paleocomplete.py
    python scripts/construct_paleocomplete.py --source-dir data/processed/phenome10k --out-dir data/paleocomplete
    python scripts/construct_paleocomplete.py --resume
    python scripts/construct_paleocomplete.py --radius 0.20 --workers 8 --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor
from tqdm import tqdm

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.datasets.augmentations import LandmarkShardGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("construct_paleocomplete")

LEVELS = ["L1", "L2", "L3"]
MIN_PARTIAL_PTS = 64


# ---------------------------------------------------------------------------
# Proxy landmark computation
# ---------------------------------------------------------------------------

def compute_proxy_landmarks(full_pts: Tensor) -> Dict[str, Tensor]:
    """
    Compute 6 PCA bounding-box extrema as proxy anatomical landmarks.

    Uses the already-normalised PCA frame (coordinates are in unit-sphere).
    Returns dict: {pos_x, neg_x, pos_y, neg_y, pos_z, neg_z} → Tensor[3].
    """
    axes = [(0, "x"), (1, "y"), (2, "z")]
    landmarks: Dict[str, Tensor] = {}
    for dim, name in axes:
        col = full_pts[:, dim]
        landmarks[f"pos_{name}"] = full_pts[col.argmax()]
        landmarks[f"neg_{name}"] = full_pts[col.argmin()]
    return landmarks


# ---------------------------------------------------------------------------
# Specimen discovery
# ---------------------------------------------------------------------------

def discover_specimens(
    source_dir: Path,
    min_confidence: float = 0.5,
) -> List[dict]:
    """
    Load metadata from every .pt in source_dir and assign tiers.

    Returns list of records:
        id, pt_path (absolute), bone_type, taxon, completeness_confidence,
        geological_age, taxon_order, tier ("A" or "B")
    """
    records: List[dict] = []
    pt_files = sorted(source_dir.glob("*.pt"))
    if not pt_files:
        log.error("No .pt files found in %s", source_dir)
        return records

    for pt_path in tqdm(pt_files, desc="Scanning specimens", unit="file"):
        try:
            data = torch.load(str(pt_path), map_location="cpu", weights_only=False)
        except Exception as exc:
            log.debug("Could not load %s: %s", pt_path.name, exc)
            continue

        meta = data.get("metadata", {})
        conf = float(meta.get("completeness_confidence", 0.6))
        if conf < min_confidence:
            continue

        tier = "A" if conf >= 0.8 else "B"
        records.append({
            "id": pt_path.stem,
            "pt_path": str(pt_path),
            "bone_type": meta.get("bone_type", "unknown"),
            "taxon": meta.get("taxon", "unknown"),
            "taxon_order": meta.get("taxon_order", "unknown"),
            "geological_age": meta.get("geological_age", "unknown"),
            "completeness_confidence": conf,
            "tier": tier,
        })

    log.info("Discovered %d specimens (Tier A: %d, Tier B: %d)",
             len(records),
             sum(1 for r in records if r["tier"] == "A"),
             sum(1 for r in records if r["tier"] == "B"))
    return records


# ---------------------------------------------------------------------------
# Split assignment
# ---------------------------------------------------------------------------

def split_specimens(
    records: List[dict],
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 42,
) -> Tuple[List[dict], List[dict], List[dict]]:
    """
    Tier A → test only.
    Tier B → stratified by bone_type into train/val/test.
    """
    rng = random.Random(seed)

    tier_a = [r for r in records if r["tier"] == "A"]
    tier_b = [r for r in records if r["tier"] == "B"]

    # Group Tier B by bone_type for stratification
    by_bone: Dict[str, List[dict]] = defaultdict(list)
    for r in tier_b:
        by_bone[r["bone_type"]].append(r)

    train, val, test = [], [], []
    for bone_type, group in by_bone.items():
        rng.shuffle(group)
        n = len(group)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        train.extend(group[:n_train])
        val.extend(group[n_train:n_train + n_val])
        test.extend(group[n_train + n_val:])

    test.extend(tier_a)

    log.info("Split: train=%d  val=%d  test=%d (incl. %d Tier A)",
             len(train), len(val), len(test), len(tier_a))
    return train, val, test


# ---------------------------------------------------------------------------
# Per-specimen shard generation
# ---------------------------------------------------------------------------

def _generate_shards(
    specimen: dict,
    out_dir: Path,
    split: str,
    radius: float,
    resume: bool,
    seed: Optional[int],
) -> List[dict]:
    """
    Generate L1/L2/L3 shards for one specimen.  Returns a list of JSON records.
    Called in subprocess — no shared state.
    """
    try:
        data = torch.load(specimen["pt_path"], map_location="cpu", weights_only=False)
    except Exception as exc:
        log.debug("Load failed %s: %s", specimen["id"], exc)
        return []

    full_pts: Tensor = data["full"]
    meta: dict = data.get("metadata", {})

    landmarks = compute_proxy_landmarks(full_pts)
    shard_gen = LandmarkShardGenerator(
        landmark_centers=landmarks,
        landmark_radius=radius,
        seed=seed,
    )

    records = []
    for level in LEVELS:
        out_path = out_dir / split / f"{specimen['id']}_{level}.pt"

        if resume and out_path.exists():
            # Reconstruct the JSON record from existing file
            fraction_removed = 0.0
            try:
                existing = torch.load(str(out_path), map_location="cpu", weights_only=False)
                fraction_removed = float(
                    existing.get("metadata", {}).get("fraction_removed", 0.0)
                )
            except Exception:
                pass
        else:
            partial, _ = shard_gen.generate(full_pts, level)

            # Fallback for degenerate specimens
            if partial.shape[0] < MIN_PARTIAL_PTS:
                import numpy as np
                from scripts.download_and_preprocess import make_partial
                pts_np = full_pts.numpy()
                n_partial = max(MIN_PARTIAL_PTS, full_pts.shape[0] // 2)
                partial_np = make_partial(pts_np, n_partial)
                partial = torch.from_numpy(partial_np)

            fraction_removed = 1.0 - partial.shape[0] / max(full_pts.shape[0], 1)

            shard_meta = {
                **meta,
                "completeness_level": level,
                "fraction_removed": fraction_removed,
            }
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"full": full_pts, "partial": partial, "metadata": shard_meta},
                       str(out_path))

        records.append({
            "id": f"{specimen['id']}_{level}",
            "pt_path": f"{split}/{specimen['id']}_{level}.pt",
            "completeness_level": level,
            "bone_type": specimen.get("bone_type", "unknown"),
            "taxon": specimen.get("taxon", "unknown"),
            "completeness_confidence": specimen.get("completeness_confidence", 0.6),
            "fraction_removed": fraction_removed,
            "specimen_id": specimen["id"],
            "geological_age": specimen.get("geological_age", "unknown"),
            "taxon_order": specimen.get("taxon_order", "unknown"),
        })

    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Construct PaleoComplete benchmark from preprocessed Phenome10K .pt files."
    )
    parser.add_argument(
        "--source-dir", type=Path,
        default=REPO_ROOT / "data" / "processed" / "phenome10k",
        help="Directory containing preprocessed Phenome10K .pt files.",
    )
    parser.add_argument(
        "--out-dir", type=Path,
        default=REPO_ROOT / "data" / "paleocomplete",
        help="Output directory for PaleoComplete benchmark.",
    )
    parser.add_argument(
        "--radius", type=float, default=0.20,
        help="Landmark removal radius in normalised units (default: 0.20).",
    )
    parser.add_argument(
        "--min-confidence", type=float, default=0.5,
        help="Minimum completeness_confidence to include a specimen.",
    )
    parser.add_argument(
        "--train-frac", type=float, default=0.70,
    )
    parser.add_argument(
        "--val-frac", type=float, default=0.15,
    )
    parser.add_argument(
        "--seed", type=int, default=42,
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Number of worker processes for shard generation.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Reload existing split JSONs and skip already-written .pt files.",
    )
    args = parser.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------
    # 1. Discover specimens
    # ----------------------------------------------------------------
    records = discover_specimens(args.source_dir, min_confidence=args.min_confidence)
    if not records:
        log.error("No specimens found — aborting.")
        return

    # ----------------------------------------------------------------
    # 2. Split assignment (or reload)
    # ----------------------------------------------------------------
    split_json_paths = {
        split: out_dir / f"{split}_split.json"
        for split in ("train", "val", "test")
    }

    if args.resume and all(p.exists() for p in split_json_paths.values()):
        log.info("--resume: loading existing split assignments.")
        id_to_split: Dict[str, str] = {}
        for split, path in split_json_paths.items():
            with open(path) as fh:
                existing = json.load(fh)
            for rec in existing:
                id_to_split[rec["specimen_id"]] = split

        train_recs = [r for r in records if id_to_split.get(r["id"]) == "train"]
        val_recs   = [r for r in records if id_to_split.get(r["id"]) == "val"]
        test_recs  = [r for r in records if id_to_split.get(r["id"]) == "test"]
        # Any specimens not in existing splits → assign fresh
        unassigned = [r for r in records if r["id"] not in id_to_split]
        if unassigned:
            log.info("  %d new specimens not in existing splits — assigning now.", len(unassigned))
            new_train, new_val, new_test = split_specimens(
                unassigned, args.train_frac, args.val_frac, args.seed
            )
            train_recs.extend(new_train)
            val_recs.extend(new_val)
            test_recs.extend(new_test)
    else:
        train_recs, val_recs, test_recs = split_specimens(
            records, args.train_frac, args.val_frac, args.seed
        )

    splits: Dict[str, List[dict]] = {
        "train": train_recs,
        "val": val_recs,
        "test": test_recs,
    }

    # ----------------------------------------------------------------
    # 3. Generate shards for each split
    # ----------------------------------------------------------------
    split_json_records: Dict[str, List[dict]] = {s: [] for s in splits}

    for split_name, specimens in splits.items():
        log.info("Generating shards for split=%s (%d specimens × 3 levels = %d files)",
                 split_name, len(specimens), len(specimens) * 3)

        if args.workers > 1:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                futures = {
                    pool.submit(
                        _generate_shards,
                        spec, out_dir, split_name, args.radius, args.resume, args.seed
                    ): spec
                    for spec in specimens
                }
                for future in tqdm(as_completed(futures), total=len(futures),
                                   desc=split_name, unit="specimen"):
                    try:
                        split_json_records[split_name].extend(future.result())
                    except Exception as exc:
                        log.warning("Shard generation failed for %s: %s",
                                    futures[future]["id"], exc)
        else:
            for spec in tqdm(specimens, desc=split_name, unit="specimen"):
                try:
                    split_json_records[split_name].extend(
                        _generate_shards(spec, out_dir, split_name, args.radius,
                                         args.resume, args.seed)
                    )
                except Exception as exc:
                    log.warning("Shard generation failed for %s: %s", spec["id"], exc)

    # ----------------------------------------------------------------
    # 4. Write split JSONs
    # ----------------------------------------------------------------
    for split_name, json_records in split_json_records.items():
        # Normalise pt_path to forward slashes for cross-platform compatibility
        for rec in json_records:
            rec["pt_path"] = rec["pt_path"].replace("\\", "/")

        out_path = split_json_paths[split_name]
        with open(out_path, "w") as fh:
            json.dump(json_records, fh, indent=2)
        log.info("Wrote %s (%d records)", out_path, len(json_records))

    log.info("PaleoComplete benchmark complete → %s", out_dir)


if __name__ == "__main__":
    main()
