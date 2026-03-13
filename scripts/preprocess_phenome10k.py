"""
Preprocess downloaded Phenome10K CTM/STL files into .pt format for RELIC training.

Reads mesh files from data/raw/phenome10k/ (produced by download_phenome10k.py),
applies the same mesh→point-cloud→PCA-normalise→partial pipeline as
download_and_preprocess.py, and writes data/processed/phenome10k/<slug>.pt.

Usage:
    python scripts/preprocess_phenome10k.py
    python scripts/preprocess_phenome10k.py --raw data/raw/phenome10k --out data/processed
    python scripts/preprocess_phenome10k.py --workers 8 --resume
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh
from tqdm import tqdm

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.utils.geometry import area_weighted_sample, pca_normalize, pca_aspect_ratio

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("preprocess_phenome10k")

KEYWORD_EXCLUDE = {
    "reconstruction", "cast", "restored", "plaster", "composite",
    "missing", "juvenile", "subadult", "hatchling", "fetal",
}

SUPPORTED_SUFFIXES = {".ctm", ".stl", ".ply", ".obj"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _should_exclude(scientific_name: str) -> bool:
    name_lower = scientific_name.lower()
    return any(kw in name_lower for kw in KEYWORD_EXCLUDE)


def _load_mesh(path: Path) -> trimesh.Trimesh | None:
    try:
        mesh = trimesh.load(str(path), force="mesh")
        if not isinstance(mesh, trimesh.Trimesh):
            if hasattr(mesh, "dump"):
                mesh = trimesh.util.concatenate(mesh.dump())
            else:
                return None
        if len(mesh.vertices) < 100:
            return None
        return mesh
    except Exception as exc:
        log.debug("Could not load %s: %s", path.name, exc)
        return None


def _make_partial(pts: np.ndarray, n_partial: int) -> np.ndarray:
    N = len(pts)
    if N <= n_partial:
        return pts.astype(np.float32)
    centre = pts[np.random.randint(N)]
    dists = np.linalg.norm(pts - centre, axis=1)
    target_remove = N - n_partial
    radius = np.sort(dists)[min(target_remove, N - 1)]
    keep = pts[dists > radius]
    if len(keep) < n_partial // 4:
        keep = pts[np.random.choice(N, n_partial, replace=False)]
    return keep.astype(np.float32)


def process_one(
    mesh_path: Path,
    meta_path: Path | None,
    out_path: Path,
    n_points: int,
    n_partial: int,
    max_aspect_ratio: float,
) -> str | None:
    """
    Full pipeline for one mesh file. Returns None on success, error string on failure.
    Run inside a subprocess-safe function (no CUDA).
    """
    # Load metadata if available
    scientific_name = mesh_path.stem
    if meta_path and meta_path.exists():
        try:
            with open(meta_path) as fh:
                meta_json = json.load(fh)
            scientific_name = meta_json.get("scientific_name", scientific_name)
        except Exception:
            pass

    if _should_exclude(scientific_name):
        return f"excluded: {scientific_name}"

    mesh = _load_mesh(mesh_path)
    if mesh is None:
        return f"load failed: {mesh_path.name}"

    # Sample point cloud
    try:
        pts = area_weighted_sample(mesh, n_points).numpy()
    except Exception:
        try:
            pts, _ = trimesh.sample.sample_surface(mesh, n_points)
            pts = pts.astype(np.float32)
        except Exception as exc:
            return f"sampling failed: {exc}"

    pts_t = torch.from_numpy(pts)

    # Taphonomic filter
    ar = float(pca_aspect_ratio(pts_t))
    if ar > max_aspect_ratio:
        return f"aspect ratio {ar:.1f} > {max_aspect_ratio}: {mesh_path.name}"

    # PCA normalise
    pts_norm, transform = pca_normalize(pts_t)
    partial = torch.from_numpy(_make_partial(pts_norm.numpy(), n_partial))

    result = {
        "full": pts_norm,
        "partial": partial,
        "metadata": {
            "id": mesh_path.stem,
            "dataset": "phenome10k",
            "scientific_name": scientific_name,
            "bone_type": "unknown",       # Phenome10K doesn't tag bone type
            "taxon": scientific_name,
            "taxon_order": "unknown",
            "geological_age": "unknown",
            "museum_catalog_number": mesh_path.stem,
            "source_file": mesh_path.name,
            "n_points": n_points,
            "n_partial": len(partial),
            "pca_transform": {k: v.tolist() if hasattr(v, "tolist") else v
                              for k, v in transform.items()},
            "aspect_ratio": ar,
            "tier": "B",
            "ontogenetic_stage": "adult",
            "completeness_confidence": 0.6,
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, str(out_path))
    return None   # success


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess Phenome10K downloads into .pt files."
    )
    parser.add_argument(
        "--raw", type=Path,
        default=REPO_ROOT / "data" / "raw" / "phenome10k",
    )
    parser.add_argument(
        "--out", type=Path,
        default=REPO_ROOT / "data" / "processed",
    )
    parser.add_argument("--n-points",  type=int, default=4096)
    parser.add_argument("--n-partial", type=int, default=2048)
    parser.add_argument("--max-aspect-ratio", type=float, default=20.0,
                        help="Phenome10K contains whole skulls and fish; 20.0 filters only"
                             " genuinely pathological/deformed scans")
    parser.add_argument("--resume", action="store_true",
                        help="Skip slugs whose .pt already exists")
    args = parser.parse_args()

    if not args.raw.exists():
        log.error("Raw directory not found: %s", args.raw)
        sys.exit(1)

    # Collect mesh files (skip *_metadata.json and _slugs.json).
    # Deduplicate by stem: prefer .ply > .stl > .obj > .ctm so that
    # converted PLY files take priority over the original CTM downloads.
    _SUFFIX_PRIORITY = {".ply": 0, ".stl": 1, ".obj": 2, ".ctm": 3}
    _by_stem: dict[str, Path] = {}
    for p in args.raw.iterdir():
        if p.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        stem = p.stem
        if stem not in _by_stem or (
            _SUFFIX_PRIORITY[p.suffix.lower()]
            < _SUFFIX_PRIORITY[_by_stem[stem].suffix.lower()]
        ):
            _by_stem[stem] = p
    mesh_files = list(_by_stem.values())
    if not mesh_files:
        log.error("No mesh files found in %s", args.raw)
        sys.exit(1)

    log.info("Found %d mesh files in %s", len(mesh_files), args.raw)

    out_dir = args.out / "phenome10k"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build work list
    tasks = []
    skipped = 0
    for mesh_path in mesh_files:
        slug = mesh_path.stem
        out_path = out_dir / f"{slug}.pt"
        if args.resume and out_path.exists():
            skipped += 1
            continue
        meta_path = args.raw / f"{slug}_metadata.json"
        tasks.append((mesh_path, meta_path if meta_path.exists() else None,
                      out_path, args.n_points, args.n_partial, args.max_aspect_ratio))

    log.info("Tasks: %d to process, %d already done", len(tasks), skipped)

    success, fail, excluded = 0, 0, 0

    bar = tqdm(tasks, total=len(tasks), desc="Preprocessing", unit="mesh")
    for t in bar:
        err = process_one(*t)
        if err is None:
            success += 1
        elif err.startswith("excluded"):
            excluded += 1
        else:
            fail += 1
            log.warning("  failed: %s", err)
        bar.set_postfix(ok=success, skip=excluded, fail=fail)

    log.info("=" * 60)
    log.info("Done: %d saved, %d excluded (keyword filter), %d failed",
             success, excluded, fail)
    log.info("Output: %s", out_dir)
    log.info(
        "Next: run download_and_preprocess.py to generate splits.json "
        "(or it will pick these up automatically if already run)."
    )


if __name__ == "__main__":
    main()
