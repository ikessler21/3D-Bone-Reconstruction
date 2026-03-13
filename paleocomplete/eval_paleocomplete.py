"""
Standalone PaleoComplete benchmark evaluation script.

Self-contained: no imports from src/. Can be distributed with the benchmark.

Usage:
    python paleocomplete/eval_paleocomplete.py \
        --predictions predictions.json \
        --gt_file paleocomplete/test_split.json \
        --split test \
        --output results.json

predictions.json format:
    [
        {
            "id": "specimen_001",
            "predicted_points": [[x,y,z], ...],
            "samples": [[[x,y,z], ...], ...]   # optional, for diversity/uncertainty
        },
        ...
    ]

test_split.json format:
    [
        {
            "id": "specimen_001",
            "pt_path": "test/specimen_001.pt",
            "completeness_level": "L2",
            "bone_type": "femur",
            "taxon": "Tyrannosaurus rex",
            "geological_age": "Maastrichtian",
            "museum_catalog_number": "AMNH 5027",
            "completeness_confidence": 0.92
        },
        ...
    ]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure-NumPy metric implementations (no torch dependency)
# ---------------------------------------------------------------------------

def _cdist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise Euclidean distance [N, M]."""
    diff = a[:, None, :] - b[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=-1) + 1e-16)


def cd_l1(pred: np.ndarray, gt: np.ndarray) -> float:
    d = _cdist(pred, gt)
    return float(d.min(axis=1).mean() + d.min(axis=0).mean())


def cd_l2(pred: np.ndarray, gt: np.ndarray) -> float:
    d = _cdist(pred, gt) ** 2
    return float(d.min(axis=1).mean() + d.min(axis=0).mean())


def fscore(pred: np.ndarray, gt: np.ndarray, threshold: float = 0.01) -> float:
    d = _cdist(pred, gt)
    precision = (d.min(axis=1) < threshold).mean()
    recall = (d.min(axis=0) < threshold).mean()
    return float(2 * precision * recall / (precision + recall + 1e-8))


def normal_consistency(
    pred: np.ndarray, gt: np.ndarray,
    pred_normals: np.ndarray, gt_normals: np.ndarray,
) -> float:
    d = _cdist(pred, gt)
    nn_idx = d.argmin(axis=1)
    pn = pred_normals / (np.linalg.norm(pred_normals, axis=-1, keepdims=True) + 1e-12)
    gn = gt_normals / (np.linalg.norm(gt_normals, axis=-1, keepdims=True) + 1e-12)
    cos_sim = np.abs((pn * gn[nn_idx]).sum(axis=-1))
    return float(cos_sim.mean())


def bse(
    pred: np.ndarray,
    plane_normal: np.ndarray,
    plane_offset: float,
    confidence: float,
    threshold: float = 0.25,
) -> float:
    if confidence < threshold:
        return 0.0
    n = plane_normal / (np.linalg.norm(plane_normal) + 1e-12)
    dist_to_plane = pred @ n - plane_offset
    reflected = pred - 2 * dist_to_plane[:, None] * n[None, :]
    d = _cdist(pred, reflected)
    return float(confidence * (d.min(axis=1).mean() + d.min(axis=0).mean()) * 0.5)


def diversity(samples: List[np.ndarray]) -> Dict[str, float]:
    k = len(samples)
    if k < 2:
        return {"mmd": 0.0, "cov": 1.0}
    cd_mat = np.zeros((k, k), dtype=np.float32)
    for i in range(k):
        for j in range(i + 1, k):
            val = cd_l1(samples[i], samples[j])
            cd_mat[i, j] = val
            cd_mat[j, i] = val
    np.fill_diagonal(cd_mat, np.inf)
    mmd = float(cd_mat.min(axis=1).mean())
    nn_of = cd_mat.argmin(axis=1)
    cov = float(len(set(nn_of.tolist())) / k)
    return {"mmd": mmd, "cov": cov}


def uncertainty_calibration(
    samples: List[np.ndarray],
    gt: np.ndarray,
) -> Dict[str, float]:
    if len(samples) < 2:
        return {"spearman_rho": 0.0, "p_value": 1.0}
    from scipy.stats import spearmanr
    pts_stack = np.stack(samples, axis=0)  # [k, N, 3]
    per_point_std = pts_stack.std(axis=0)
    uncertainty = np.linalg.norm(per_point_std, axis=-1)
    ref = pts_stack[0]
    dist_to_gt = _cdist(ref, gt).min(axis=1)
    rho, pval = spearmanr(uncertainty, dist_to_gt)
    return {"spearman_rho": float(rho), "p_value": float(pval)}


def estimate_normals(points: np.ndarray, k: int = 10) -> np.ndarray:
    """Simple PCA-based normal estimation without torch."""
    N = points.shape[0]
    k = min(k, N - 1)
    normals = np.zeros_like(points)

    d2 = ((points[:, None, :] - points[None, :, :]) ** 2).sum(axis=-1)
    for i in range(N):
        idx = np.argpartition(d2[i], k + 1)[:k + 1]
        nbrs = points[idx]
        centered = nbrs - nbrs.mean(axis=0, keepdims=True)
        cov = centered.T @ centered / k
        try:
            _, _, vt = np.linalg.svd(cov)
            normals[i] = vt[-1]
        except np.linalg.LinAlgError:
            normals[i] = np.array([0, 0, 1], dtype=np.float32)

    norms = np.linalg.norm(normals, axis=-1, keepdims=True)
    return normals / (norms + 1e-12)


# ---------------------------------------------------------------------------
# Ground truth loading
# ---------------------------------------------------------------------------

def load_ground_truth(gt_file: str, data_dir: str = ".") -> Dict[str, np.ndarray]:
    """
    Load ground-truth complete point clouds from the split JSON.

    Returns a dict: specimen_id → np.ndarray[M, 3]
    """
    with open(gt_file) as fh:
        records = json.load(fh)

    gt_dict: Dict[str, np.ndarray] = {}
    data_dir = Path(data_dir)

    for rec in records:
        spec_id = rec["id"]
        pt_path = data_dir / rec.get("pt_path", f"{spec_id}.pt")

        if pt_path.exists():
            try:
                # Try to load as torch .pt file
                import torch
                data = torch.load(str(pt_path), map_location="cpu", weights_only=False)
                full = np.array(data["full"].numpy(), dtype=np.float32)
                gt_dict[spec_id] = full
            except ImportError:
                # torch not available: try numpy .npy
                npy_path = pt_path.with_suffix(".npy")
                if npy_path.exists():
                    gt_dict[spec_id] = np.load(str(npy_path)).astype(np.float32)
            except Exception as exc:
                logger.warning("Could not load GT for %s: %s", spec_id, exc)
        else:
            # Accept JSON inline ground truth
            if "complete_points" in rec:
                gt_dict[spec_id] = np.array(rec["complete_points"], dtype=np.float32)

    return gt_dict


def load_metadata(gt_file: str) -> Dict[str, dict]:
    """Load metadata (completeness level, bone type, etc.) from split JSON."""
    with open(gt_file) as fh:
        records = json.load(fh)
    return {rec["id"]: rec for rec in records}


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def evaluate(
    predictions: List[dict],
    gt_dict: Dict[str, np.ndarray],
    metadata: Dict[str, dict],
    threshold_fscore: float = 0.01,
    sym_threshold: float = 0.25,
) -> Dict:
    """
    Evaluate predictions against ground truth.

    Parameters
    ----------
    predictions : list of prediction dicts
        Each must have "id" and "predicted_points" ([N, 3]).
        Optional: "samples" for diversity/uncertainty metrics.
        Optional: "plane_normal", "plane_offset", "sym_confidence" for BSE.
    gt_dict     : specimen_id → complete points [M, 3]
    metadata    : specimen_id → metadata dict
    """
    all_metrics: Dict[str, List[float]] = {
        "cd_l1": [], "cd_l2": [], "fscore": [], "normal_con": [], "bse": [],
        "mmd": [], "cov": [], "spearman_rho": [],
    }

    # Per-completeness-level metrics
    level_metrics: Dict[str, Dict[str, List[float]]] = {
        "L1": {"cd_l1": [], "fscore": []},
        "L2": {"cd_l1": [], "fscore": []},
        "L3": {"cd_l1": [], "fscore": []},
    }

    n_missing = 0
    n_evaluated = 0

    for pred_rec in predictions:
        spec_id = pred_rec["id"]
        if spec_id not in gt_dict:
            logger.warning("No GT for specimen %s — skipping", spec_id)
            n_missing += 1
            continue

        pred_pts = np.array(pred_rec["predicted_points"], dtype=np.float32)
        gt_pts = gt_dict[spec_id]

        # Basic metrics
        all_metrics["cd_l1"].append(cd_l1(pred_pts, gt_pts))
        all_metrics["cd_l2"].append(cd_l2(pred_pts, gt_pts))
        all_metrics["fscore"].append(fscore(pred_pts, gt_pts, threshold_fscore))

        # Normal consistency
        try:
            pred_normals = estimate_normals(pred_pts, k=10)
            gt_normals = estimate_normals(gt_pts, k=10)
            all_metrics["normal_con"].append(
                normal_consistency(pred_pts, gt_pts, pred_normals, gt_normals)
            )
        except Exception:
            pass

        # BSE
        if "plane_normal" in pred_rec and "plane_offset" in pred_rec:
            pn = np.array(pred_rec["plane_normal"], dtype=np.float32)
            po = float(pred_rec["plane_offset"])
            sc = float(pred_rec.get("sym_confidence", 0.0))
            all_metrics["bse"].append(bse(pred_pts, pn, po, sc, sym_threshold))

        # Diversity + calibration from samples
        samples = pred_rec.get("samples", None)
        if samples and len(samples) >= 2:
            sample_arrays = [np.array(s, dtype=np.float32) for s in samples]
            div = diversity(sample_arrays)
            all_metrics["mmd"].append(div["mmd"])
            all_metrics["cov"].append(div["cov"])

            calib = uncertainty_calibration(sample_arrays, gt_pts)
            all_metrics["spearman_rho"].append(calib["spearman_rho"])

        # Per-level metrics
        meta = metadata.get(spec_id, {})
        level = meta.get("completeness_level", "")
        if level in level_metrics:
            level_metrics[level]["cd_l1"].append(cd_l1(pred_pts, gt_pts))
            level_metrics[level]["fscore"].append(fscore(pred_pts, gt_pts, threshold_fscore))

        n_evaluated += 1

    # Aggregate
    results: dict = {
        "n_evaluated": n_evaluated,
        "n_missing_gt": n_missing,
        "metrics": {},
        "per_level": {},
    }

    for k, vals in all_metrics.items():
        if vals:
            results["metrics"][k] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "n": len(vals),
            }

    for level, level_data in level_metrics.items():
        results["per_level"][level] = {}
        for k, vals in level_data.items():
            if vals:
                results["per_level"][level][k] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "n": len(vals),
                }

    return results


# ---------------------------------------------------------------------------
# Pretty-print table
# ---------------------------------------------------------------------------

def print_table(results: dict) -> None:
    """Print a nicely formatted results table."""
    metrics = results.get("metrics", {})

    print("\n" + "=" * 70)
    print("PaleoComplete Benchmark Results")
    print(f"Specimens evaluated: {results['n_evaluated']}")
    if results['n_missing_gt'] > 0:
        print(f"WARNING: {results['n_missing_gt']} specimens had no ground truth")
    print("=" * 70)
    print(f"{'Metric':<25} {'Mean':>10} {'Std':>10} {'N':>6}")
    print("-" * 55)

    display_order = ["cd_l1", "cd_l2", "fscore", "normal_con", "bse",
                     "mmd", "cov", "spearman_rho"]
    labels = {
        "cd_l1": "CD-L1 ↓",
        "cd_l2": "CD-L2 ↓",
        "fscore": "F-Score@1% ↑",
        "normal_con": "Normal Consistency ↑",
        "bse": "BSE ↓",
        "mmd": "Diversity MMD ↑",
        "cov": "Diversity COV ↑",
        "spearman_rho": "Uncertainty Calibration ρ ↑",
    }

    for k in display_order:
        if k in metrics:
            m = metrics[k]
            label = labels.get(k, k)
            print(f"  {label:<23} {m['mean']:>10.4f} {m['std']:>10.4f} {m['n']:>6}")

    if results.get("per_level"):
        print("\nPer-Completeness-Level Results:")
        print("-" * 55)
        for level in ["L1", "L2", "L3"]:
            level_data = results["per_level"].get(level, {})
            if level_data:
                cd = level_data.get("cd_l1", {})
                fs = level_data.get("fscore", {})
                n = cd.get("n", 0)
                print(
                    f"  {level}  CD-L1: {cd.get('mean', float('nan')):.4f} ± {cd.get('std', 0):.4f}  "
                    f"F-Score: {fs.get('mean', float('nan')):.4f} ± {fs.get('std', 0):.4f}  (n={n})"
                )
    print("=" * 70)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="PaleoComplete standalone benchmark evaluation"
    )
    parser.add_argument("--predictions", type=str, required=True,
                        help="JSON file with predictions")
    parser.add_argument("--gt_file", type=str,
                        default="paleocomplete/test_split.json",
                        help="Ground truth split JSON file")
    parser.add_argument("--data_dir", type=str,
                        default=".",
                        help="Base directory for .pt GT files")
    parser.add_argument("--split", type=str, default="test",
                        help="Split name (for display)")
    parser.add_argument("--output", type=str, default="paleocomplete_results.json",
                        help="Output JSON for results")
    parser.add_argument("--threshold_fscore", type=float, default=0.01)
    parser.add_argument("--sym_threshold", type=float, default=0.25)
    args = parser.parse_args()

    # Load predictions
    with open(args.predictions) as fh:
        predictions = json.load(fh)
    logger.info("Loaded %d predictions from %s", len(predictions), args.predictions)

    # Load ground truth
    if not Path(args.gt_file).exists():
        logger.error("GT file not found: %s", args.gt_file)
        sys.exit(1)

    gt_dict = load_ground_truth(args.gt_file, args.data_dir)
    metadata = load_metadata(args.gt_file)
    logger.info("Loaded GT for %d specimens", len(gt_dict))

    # Evaluate
    results = evaluate(
        predictions, gt_dict, metadata,
        threshold_fscore=args.threshold_fscore,
        sym_threshold=args.sym_threshold,
    )
    results["split"] = args.split
    results["predictions_file"] = args.predictions
    results["gt_file"] = args.gt_file

    # Display
    print_table(results)

    # Save
    with open(args.output, "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    logger.info("Results saved to %s", args.output)


if __name__ == "__main__":
    main()
