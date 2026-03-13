"""
Download and preprocess proxy bone datasets for RELIC pre-training.

Datasets handled (no account required, all CC-licensed):
  - ICL Femur/Tibia   STL meshes   CC-BY-4.0   ~70 specimens
  - VerSe Vertebrae   NIfTI CT     CC-BY-SA-4.0 ~374 CT scans
  - BoneDat Pelvis    NIfTI/VTK    CC-BY-4.0   ~278 CT scans

Output: data/processed/<dataset>/<id>.pt
Each .pt file: {partial, full, metadata}
  - full    : Tensor[4096, 3]  PCA-normalised point cloud
  - partial : Tensor[~2048, 3] simulated partial (random region removal)
  - metadata: dict with bone_type, dataset, tier, ontogenetic_stage, etc.

Usage:
    python scripts/download_and_preprocess.py
    python scripts/download_and_preprocess.py --datasets icl verse
    python scripts/download_and_preprocess.py --skip-download   # preprocess only
    python scripts/download_and_preprocess.py --n-points 4096 --n-partial 2048
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import shutil
import sys
import traceback
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import trimesh
from tqdm import tqdm

# Add repo root to path
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.utils.geometry import poisson_disk_sample, area_weighted_sample, pca_normalize, pca_aspect_ratio

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("download_preprocess")


# ---------------------------------------------------------------------------
# Dataset URLs
# ---------------------------------------------------------------------------

DATASETS: Dict[str, Dict] = {
    "icl": {
        "name": "ICL Femur/Tibia",
        "license": "CC-BY-4.0",
        "files": {
            "femur": {
                "url": "https://zenodo.org/record/167808/files/femur_3D_surface_meshes.zip",
                "filename": "icl_femur.zip",
                "sha256": None,  # no verification — check file size instead
                "bone_type": "femur",
            },
            "tibia": {
                "url": "https://zenodo.org/record/167808/files/tibia_3D_surface_meshes.zip",
                "filename": "icl_tibia.zip",
                "sha256": None,
                "bone_type": "tibia",
            },
        },
        "format": "stl",
    },
    "verse": {
        "name": "VerSe Vertebrae",
        "license": "CC-BY-SA-4.0",
        "files": {
            "verse19": {
                "url": "https://s3.bonescreen.de/public/VerSe-complete/dataset-verse19training.zip",
                "filename": "verse19training.zip",
                "sha256": None,
                "bone_type": "vertebra",
            },
            "verse20": {
                "url": "https://s3.bonescreen.de/public/VerSe-complete/dataset-verse20training.zip",
                "filename": "verse20training.zip",
                "sha256": None,
                "bone_type": "vertebra",
            },
        },
        "format": "nifti",
    },
    "bonedat": {
        "name": "BoneDat Pelvis",
        "license": "CC-BY-4.0",
        "files": {
            "bonedat": {
                "url": "https://zenodo.org/api/records/15189761/files-archive",
                "filename": "bonedat.zip",
                "sha256": None,
                "bone_type": "pelvis",
            },
        },
        "format": "nifti",
    },
}

# VerSe label → vertebra name mapping (1=C1 ... 25=L6)
VERSE_LABELS = {
    1: "C1", 2: "C2", 3: "C3", 4: "C4", 5: "C5", 6: "C6", 7: "C7",
    8: "T1", 9: "T2", 10: "T3", 11: "T4", 12: "T5", 13: "T6",
    14: "T7", 15: "T8", 16: "T9", 17: "T10", 18: "T11", 19: "T12",
    20: "L1", 21: "L2", 22: "L3", 23: "L4", 24: "L5", 25: "L6",
    26: "Sacrum", 28: "Cocc",
}


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def download_file(url: str, dest: Path, desc: str = "") -> bool:
    """Stream-download url → dest with a tqdm progress bar. Returns True on success."""
    import requests
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1_000:
        log.info("  already downloaded: %s", dest.name)
        return True

    log.info("  downloading %s → %s", url, dest.name)
    try:
        resp = requests.get(url, stream=True, timeout=120,
                            headers={"User-Agent": "RELIC-fossil/0.1"},
                            verify=False)  # some academic servers have expired certs
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        chunk = 1 << 20  # 1 MB
        with open(dest, "wb") as fh, tqdm(
            total=total, unit="B", unit_scale=True, desc=desc or dest.name, leave=False
        ) as bar:
            for data in resp.iter_content(chunk_size=chunk):
                fh.write(data)
                bar.update(len(data))
        log.info("  saved %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)
        return True
    except Exception as exc:
        log.error("  download failed: %s", exc)
        if dest.exists():
            dest.unlink()
        return False


def extract_zip(zip_path: Path, out_dir: Path) -> bool:
    """Extract a zip archive if not already extracted."""
    if out_dir.exists() and any(out_dir.iterdir()):
        log.info("  already extracted: %s", out_dir.name)
        return True
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("  extracting %s → %s", zip_path.name, out_dir)
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(out_dir)
        return True
    except Exception as exc:
        log.error("  extraction failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Mesh / NIfTI processing helpers
# ---------------------------------------------------------------------------

def load_stl_mesh(path: Path) -> Optional[trimesh.Trimesh]:
    try:
        mesh = trimesh.load(str(path), force="mesh")
        if not isinstance(mesh, trimesh.Trimesh):
            mesh = trimesh.util.concatenate(mesh.dump()) if hasattr(mesh, "dump") else None
        if mesh is None or len(mesh.vertices) < 100:
            return None
        return mesh
    except Exception as exc:
        log.debug("  could not load %s: %s", path.name, exc)
        return None


def nifti_label_to_mesh(
    vol: np.ndarray,
    label: int,
    voxel_size: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Optional[trimesh.Trimesh]:
    """Extract a binary label from a segmentation volume and run marching cubes."""
    from skimage.measure import marching_cubes
    from skimage.morphology import binary_closing, ball

    mask = (vol == label).astype(np.uint8)
    if mask.sum() < 500:
        return None

    # Gentle closing to fill small holes from segmentation
    mask = binary_closing(mask, ball(2)).astype(np.uint8)

    try:
        verts, faces, normals, _ = marching_cubes(mask, level=0.5, spacing=voxel_size)
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
        mesh = mesh.simplify_quadric_decimation(face_count=min(len(mesh.faces), 20_000))
        if len(mesh.vertices) < 50:
            return None
        return mesh
    except Exception as exc:
        log.debug("  marching cubes failed for label %d: %s", label, exc)
        return None


def mesh_to_pointcloud(mesh: trimesh.Trimesh, n_points: int) -> Optional[np.ndarray]:
    """Sample n_points from a mesh using area-weighted sampling."""
    try:
        pts = area_weighted_sample(mesh, n_points)
        return pts.numpy()
    except Exception:
        # Fallback: uniform random surface sample via trimesh
        try:
            pts, _ = trimesh.sample.sample_surface(mesh, n_points)
            return pts.astype(np.float32)
        except Exception:
            return None


def make_partial(
    full_pts: np.ndarray,
    n_partial: int,
    method: str = "region",
) -> np.ndarray:
    """
    Simulate a partial point cloud by removing a contiguous region.

    method='region' : remove all points within a random sphere (simulates fragment)
    method='random' : random subsample (simpler fallback)
    """
    N = len(full_pts)
    if method == "region" and N > n_partial:
        # Pick a random centre point, remove everything within radius r
        centre = full_pts[np.random.randint(N)]
        dists = np.linalg.norm(full_pts - centre, axis=1)
        # Choose r so that ~(N - n_partial) points fall inside
        target_remove = N - n_partial
        sorted_dists = np.sort(dists)
        radius = sorted_dists[min(target_remove, N - 1)]
        keep_mask = dists > radius
        partial = full_pts[keep_mask]
        if len(partial) < n_partial // 4:
            # Fallback if removal was too aggressive
            partial = full_pts[np.random.choice(N, n_partial, replace=False)]
    else:
        idx = np.random.choice(N, min(n_partial, N), replace=False)
        partial = full_pts[idx]

    return partial.astype(np.float32)


def process_mesh(
    mesh: trimesh.Trimesh,
    metadata: dict,
    n_points: int,
    n_partial: int,
    max_aspect_ratio: float = 3.0,
) -> Optional[dict]:
    """
    Full processing pipeline for one mesh:
      load → sample → PCA check → normalise → make partial → return dict
    """
    pts = mesh_to_pointcloud(mesh, n_points)
    if pts is None:
        return None

    pts_tensor = torch.from_numpy(pts)

    # Taphonomic deformation filter (PCA aspect ratio)
    ar = pca_aspect_ratio(pts_tensor)
    if ar > max_aspect_ratio:
        log.debug("  skipping (aspect ratio %.2f > %.2f): %s", ar, max_aspect_ratio, metadata.get("id", ""))
        return None

    # PCA normalise
    pts_norm, transform = pca_normalize(pts_tensor)
    pts_norm_np = pts_norm.numpy()

    # Make partial
    partial_np = make_partial(pts_norm_np, n_partial)
    partial_tensor = torch.from_numpy(partial_np)

    return {
        "full": pts_norm,
        "partial": partial_tensor,
        "metadata": {
            **metadata,
            "n_points": n_points,
            "n_partial": len(partial_np),
            "pca_transform": {k: v.tolist() if hasattr(v, "tolist") else v
                              for k, v in transform.items()},
            "aspect_ratio": float(ar),
            "tier": "B",
            "ontogenetic_stage": "adult",
            "completeness_confidence": 0.7,
        },
    }


# ---------------------------------------------------------------------------
# Per-dataset processors
# ---------------------------------------------------------------------------

def process_icl(
    raw_dir: Path,
    processed_dir: Path,
    n_points: int,
    n_partial: int,
    bone_type: str,
) -> int:
    """Process ICL STL meshes from raw_dir → processed_dir. Returns count processed."""
    stl_files = list(raw_dir.rglob("*.stl")) + list(raw_dir.rglob("*.STL"))
    if not stl_files:
        log.warning("  no STL files found in %s", raw_dir)
        return 0

    out_dir = processed_dir / "icl" / bone_type
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    for stl_path in tqdm(stl_files, desc=f"ICL {bone_type}", unit="mesh"):
        specimen_id = stl_path.stem
        out_path = out_dir / f"{specimen_id}.pt"
        if out_path.exists():
            count += 1
            continue

        mesh = load_stl_mesh(stl_path)
        if mesh is None:
            continue

        metadata = {
            "id": specimen_id,
            "dataset": "icl",
            "bone_type": bone_type,
            "taxon": "Homo sapiens",
            "taxon_order": "Primates",
            "geological_age": "Recent",
            "museum_catalog_number": specimen_id,
            "source_file": str(stl_path.name),
        }

        # Proxy modern bones can be naturally very elongated (femur AR ~94),
        # so skip the taphonomic deformation filter for these clean datasets.
        result = process_mesh(mesh, metadata, n_points, n_partial, max_aspect_ratio=500.0)
        if result is None:
            continue

        torch.save(result, str(out_path))
        count += 1

    return count


def process_verse(
    raw_dir: Path,
    processed_dir: Path,
    n_points: int,
    n_partial: int,
) -> int:
    """Process VerSe NIfTI segmentation volumes → one .pt per vertebra per scan."""
    import nibabel as nib

    # VerSe structure: raw_dir/derivatives/labels/<subject>/<session>/<subject>_<session>_seg-vert_msk.nii.gz
    seg_files = (
        list(raw_dir.rglob("*seg-vert_msk.nii.gz"))
        + list(raw_dir.rglob("*seg-subreg_ctd.nii.gz"))
        + list(raw_dir.rglob("*_seg*.nii.gz"))
        + list(raw_dir.rglob("*.nii.gz"))
    )
    # Filter to segmentation files only
    seg_files = [f for f in seg_files if "seg" in f.name.lower() or "msk" in f.name.lower()]

    if not seg_files:
        # Fallback: grab all nii.gz
        seg_files = list(raw_dir.rglob("*.nii.gz"))

    if not seg_files:
        log.warning("  no NIfTI files found in %s", raw_dir)
        return 0

    out_dir = processed_dir / "verse"
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    for nii_path in tqdm(seg_files, desc="VerSe", unit="scan"):
        try:
            img = nib.load(str(nii_path))
            vol = np.asarray(img.dataobj, dtype=np.int16)
            zooms = img.header.get_zooms()[:3]
            voxel_size = tuple(float(z) for z in zooms)
        except Exception as exc:
            log.debug("  could not load %s: %s", nii_path.name, exc)
            continue

        unique_labels = [int(l) for l in np.unique(vol) if l > 0]

        for label in unique_labels:
            vert_name = VERSE_LABELS.get(label, f"vert_{label}")
            specimen_id = f"{nii_path.stem.split('.')[0]}_label{label}_{vert_name}"
            out_path = out_dir / f"{specimen_id}.pt"
            if out_path.exists():
                count += 1
                continue

            mesh = nifti_label_to_mesh(vol, label, voxel_size)
            if mesh is None:
                continue

            metadata = {
                "id": specimen_id,
                "dataset": "verse",
                "bone_type": "vertebra",
                "bone_subtype": vert_name,
                "taxon": "Homo sapiens",
                "taxon_order": "Primates",
                "geological_age": "Recent",
                "museum_catalog_number": specimen_id,
                "source_file": nii_path.name,
                "verse_label": label,
            }

            result = process_mesh(mesh, metadata, n_points, n_partial, max_aspect_ratio=500.0)
            if result is None:
                continue

            torch.save(result, str(out_path))
            count += 1

    return count


def process_bonedat(
    raw_dir: Path,
    processed_dir: Path,
    n_points: int,
    n_partial: int,
) -> int:
    """Process BoneDat NIfTI pelvis volumes."""
    import nibabel as nib

    # BoneDat: NIfTI segmentation masks (.nii.gz) + images
    # Segmentation files typically contain 'seg' or 'mask' in name
    seg_files = (
        list(raw_dir.rglob("*seg*.nii.gz"))
        + list(raw_dir.rglob("*mask*.nii.gz"))
        + list(raw_dir.rglob("*label*.nii.gz"))
    )
    if not seg_files:
        seg_files = list(raw_dir.rglob("*.nii.gz"))

    # Also try VTK
    vtk_files = list(raw_dir.rglob("*.vtk")) + list(raw_dir.rglob("*.vtp"))

    out_dir = processed_dir / "bonedat"
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    # Process NIfTI
    for nii_path in tqdm(seg_files, desc="BoneDat NIfTI", unit="scan"):
        try:
            img = nib.load(str(nii_path))
            vol = np.asarray(img.dataobj, dtype=np.int16)
            zooms = img.header.get_zooms()[:3]
            voxel_size = tuple(float(z) for z in zooms)
        except Exception as exc:
            log.debug("  could not load %s: %s", nii_path.name, exc)
            continue

        unique_labels = [int(l) for l in np.unique(vol) if l > 0]
        if not unique_labels:
            continue

        # If binary mask, treat as single bone
        for label in (unique_labels if len(unique_labels) <= 5 else [1]):
            specimen_id = f"{nii_path.stem.split('.')[0]}_l{label}"
            out_path = out_dir / f"{specimen_id}.pt"
            if out_path.exists():
                count += 1
                continue

            mesh = nifti_label_to_mesh(vol, label, voxel_size)
            if mesh is None:
                # Try treating whole non-zero region as one bone
                if label == 1:
                    mesh = nifti_label_to_mesh((vol > 0).astype(np.int16) * 1,
                                               1, voxel_size)
            if mesh is None:
                continue

            metadata = {
                "id": specimen_id,
                "dataset": "bonedat",
                "bone_type": "pelvis",
                "taxon": "Homo sapiens",
                "taxon_order": "Primates",
                "geological_age": "Recent",
                "museum_catalog_number": specimen_id,
                "source_file": nii_path.name,
            }

            result = process_mesh(mesh, metadata, n_points, n_partial, max_aspect_ratio=500.0)
            if result is None:
                continue

            torch.save(result, str(out_path))
            count += 1

    # Process VTK (trimesh can load these)
    for vtk_path in tqdm(vtk_files, desc="BoneDat VTK", unit="mesh"):
        specimen_id = vtk_path.stem
        out_path = out_dir / f"{specimen_id}.pt"
        if out_path.exists():
            count += 1
            continue

        try:
            mesh = trimesh.load(str(vtk_path), force="mesh")
            if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) < 100:
                continue
        except Exception:
            continue

        metadata = {
            "id": specimen_id,
            "dataset": "bonedat",
            "bone_type": "pelvis",
            "taxon": "Homo sapiens",
            "taxon_order": "Primates",
            "geological_age": "Recent",
            "museum_catalog_number": specimen_id,
            "source_file": vtk_path.name,
        }

        result = process_mesh(mesh, metadata, n_points, n_partial, max_aspect_ratio=500.0)
        if result is None:
            continue

        torch.save(result, str(out_path))
        count += 1

    return count


# ---------------------------------------------------------------------------
# Split generation
# ---------------------------------------------------------------------------

def generate_splits(
    processed_dir: Path,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> Dict[str, List[str]]:
    """
    Walk processed_dir, collect all .pt files, split into train/val by dataset+bone_type
    to ensure stratification. Writes splits.json to processed_dir.
    """
    random.seed(seed)
    all_files = list(processed_dir.rglob("*.pt"))
    log.info("Found %d processed .pt files total.", len(all_files))

    # Group by (dataset, bone_type)
    groups: Dict[str, List[Path]] = {}
    for pt in all_files:
        try:
            meta = torch.load(str(pt), map_location="cpu", weights_only=False)["metadata"]
            key = f"{meta.get('dataset','unk')}_{meta.get('bone_type','unk')}"
        except Exception:
            key = "unknown"
        groups.setdefault(key, []).append(pt)

    train, val = [], []
    for key, paths in groups.items():
        random.shuffle(paths)
        n_val = max(1, int(len(paths) * val_fraction))
        val.extend(str(p) for p in paths[:n_val])
        train.extend(str(p) for p in paths[n_val:])

    splits = {"train": sorted(train), "val": sorted(val)}
    splits_path = processed_dir / "splits.json"
    with open(splits_path, "w") as fh:
        json.dump(splits, fh, indent=2)

    log.info("Splits: %d train / %d val → %s", len(train), len(val), splits_path)
    return splits


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and preprocess proxy bone datasets for RELIC pre-training."
    )
    parser.add_argument(
        "--datasets", nargs="+",
        choices=list(DATASETS.keys()),
        default=list(DATASETS.keys()),
        help="Which datasets to process (default: all)",
    )
    parser.add_argument(
        "--raw-dir", type=Path,
        default=REPO_ROOT / "data" / "raw",
        help="Directory to store downloaded archives and extracted files",
    )
    parser.add_argument(
        "--processed-dir", type=Path,
        default=REPO_ROOT / "data" / "processed",
        help="Directory to write .pt files",
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Skip downloading; process already-extracted files only",
    )
    parser.add_argument("--n-points",  type=int, default=4096, help="Points in full cloud")
    parser.add_argument("--n-partial", type=int, default=2048, help="Points in partial cloud")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.raw_dir.mkdir(parents=True, exist_ok=True)
    args.processed_dir.mkdir(parents=True, exist_ok=True)

    total_processed = 0

    # ----------------------------------------------------------------
    # ICL Femur / Tibia
    # ----------------------------------------------------------------
    if "icl" in args.datasets:
        log.info("=" * 60)
        log.info("ICL Femur/Tibia (CC-BY-4.0)")
        log.info("=" * 60)
        ds = DATASETS["icl"]
        for key, info in ds["files"].items():
            zip_path = args.raw_dir / info["filename"]
            extract_dir = args.raw_dir / f"icl_{key}"

            if not args.skip_download:
                ok = download_file(info["url"], zip_path, desc=info["filename"])
                if not ok:
                    log.error("Skipping ICL %s — download failed.", key)
                    continue

            if zip_path.exists():
                extract_zip(zip_path, extract_dir)

            if extract_dir.exists():
                n = process_icl(extract_dir, args.processed_dir,
                                args.n_points, args.n_partial, info["bone_type"])
                log.info("  ICL %s: %d specimens processed.", key, n)
                total_processed += n
            else:
                log.warning("  ICL %s: extract dir not found, skipping.", key)

    # ----------------------------------------------------------------
    # VerSe Vertebrae
    # ----------------------------------------------------------------
    if "verse" in args.datasets:
        log.info("=" * 60)
        log.info("VerSe Vertebrae (CC-BY-SA-4.0)")
        log.info("=" * 60)
        ds = DATASETS["verse"]
        for key, info in ds["files"].items():
            zip_path = args.raw_dir / info["filename"]
            extract_dir = args.raw_dir / key

            if not args.skip_download:
                ok = download_file(info["url"], zip_path, desc=info["filename"])
                if not ok:
                    log.error("Skipping VerSe %s — download failed.", key)
                    continue

            if zip_path.exists():
                extract_zip(zip_path, extract_dir)

            if extract_dir.exists():
                n = process_verse(extract_dir, args.processed_dir,
                                  args.n_points, args.n_partial)
                log.info("  VerSe %s: %d vertebra point clouds processed.", key, n)
                total_processed += n
            else:
                log.warning("  VerSe %s: extract dir not found, skipping.", key)

    # ----------------------------------------------------------------
    # BoneDat Pelvis
    # ----------------------------------------------------------------
    if "bonedat" in args.datasets:
        log.info("=" * 60)
        log.info("BoneDat Pelvis (CC-BY-4.0)")
        log.info("=" * 60)
        ds = DATASETS["bonedat"]
        for key, info in ds["files"].items():
            zip_path = args.raw_dir / info["filename"]
            extract_dir = args.raw_dir / key

            if not args.skip_download:
                ok = download_file(info["url"], zip_path, desc=info["filename"])
                if not ok:
                    log.error("Skipping BoneDat — download failed.")
                    continue

            if zip_path.exists():
                extract_zip(zip_path, extract_dir)

            if extract_dir.exists():
                n = process_bonedat(extract_dir, args.processed_dir,
                                    args.n_points, args.n_partial)
                log.info("  BoneDat: %d pelvis point clouds processed.", n)
                total_processed += n
            else:
                log.warning("  BoneDat: extract dir not found, skipping.")

    # ----------------------------------------------------------------
    # Generate train/val splits
    # ----------------------------------------------------------------
    log.info("=" * 60)
    log.info("Total processed: %d specimens", total_processed)
    if total_processed > 0:
        splits = generate_splits(args.processed_dir, args.val_fraction, args.seed)
        log.info("Done. Ready to train:")
        log.info("  python -m src.training.trainer --config experiments/relic_full.yaml --stage pretrain")
    else:
        log.warning("No specimens were processed. Check download logs above.")


if __name__ == "__main__":
    main()
