"""
Benchmark evaluation for RELIC.

evaluate_on_paleocomplete  — full PaleoComplete benchmark evaluation
evaluate_on_pcn            — PCN ShapeNet benchmark
run_ablation_table         — generate ablation results as DataFrame
CLI                        — python -m src.eval.evaluate --benchmark paleocomplete --checkpoint ckpt.pth
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from src.eval.metrics import (
    MetricsTracker,
    compute_bse,
    compute_cd_l1,
    compute_cd_l2,
    compute_diversity,
    compute_fscore,
    compute_normal_consistency,
    compute_uncertainty_calibration,
)
from src.utils.geometry import compute_normals

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PaleoComplete evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_on_paleocomplete(
    model: torch.nn.Module,
    dataset,
    config: dict,
    n_samples: int = 10,
    batch_size: int = 8,
    device: Optional[torch.device] = None,
) -> dict:
    """
    Evaluate RELIC on the PaleoComplete benchmark.

    Computes: CD-L1, CD-L2, F-Score@1%, Normal Consistency, BSE (symmetric
    vs asymmetric subsets), Diversity (MMD, COV), Uncertainty Calibration.

    Parameters
    ----------
    model    : RELIC (or any model with .sample() method)
    dataset  : PaleoCompleteDataset
    config   : experiment config dict
    n_samples: number of completion samples per input for diversity/uncertainty
    batch_size: evaluation batch size
    device   : torch device

    Returns dict of metric summaries
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    from src.datasets.fossil_dataset import collate_fn
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn,
        num_workers=config.get("data", {}).get("num_workers", 2),
    )

    tracker = MetricsTracker()
    sym_tracker = MetricsTracker()     # high-confidence symmetric specimens
    asym_tracker = MetricsTracker()    # low-confidence / asymmetric specimens
    diversity_results = []
    calibration_results = []

    for batch_idx, batch in enumerate(loader):
        partial = batch["partial"].to(device)   # [B, N, 3]
        full = batch["full"].to(device)          # [B, M, 3]
        meta = batch.get("metadata", {})

        B = partial.shape[0]

        # Single best prediction (deterministic: take first of n_samples)
        try:
            samples = model.sample(partial, n_samples=n_samples)  # [B, S, M, 3]
        except AttributeError:
            # Fallback for non-RELIC models
            out = model(partial, full, mode="inference")
            pred = out["pred"]
            samples = pred.unsqueeze(1).expand(-1, n_samples, -1, -1)

        pred = samples[:, 0, :, :]   # [B, M, 3]  — use first sample as point estimate

        for b in range(B):
            p_np = pred[b].cpu().numpy()
            g_np = full[b].cpu().numpy()

            # Basic metrics
            cd1 = compute_cd_l1(p_np, g_np)
            cd2 = compute_cd_l2(p_np, g_np)
            fs = compute_fscore(p_np, g_np, threshold=0.01)

            # Normal consistency (estimate normals on-the-fly)
            try:
                pn = compute_normals(torch.from_numpy(p_np), k=10).numpy()
                gn = compute_normals(torch.from_numpy(g_np), k=10).numpy()
                nc = compute_normal_consistency(p_np, g_np, pn, gn)
            except Exception:
                nc = 0.0

            metrics = {"cd_l1": cd1, "cd_l2": cd2, "fscore": fs, "normal_con": nc}
            tracker.update(metrics)

            # BSE — get symmetry info from model if available
            sym_conf = 0.0
            if hasattr(model, "vae") and hasattr(model.vae, "symmetry_module") and model.vae.symmetry_module is not None:
                normals_t, offsets_t, confs_t = model.vae.symmetry_module.detector.detect_batch(
                    partial[b:b+1]
                )
                sym_conf = float(confs_t[0].item())
                bse = compute_bse(
                    p_np,
                    normals_t[0].cpu().numpy(),
                    float(offsets_t[0].item()),
                    sym_conf,
                )
                metrics["bse"] = bse
                if sym_conf >= 0.5:
                    sym_tracker.update({"bse": bse, "cd_l1": cd1, "fscore": fs})
                else:
                    asym_tracker.update({"bse": bse, "cd_l1": cd1, "fscore": fs})

            # Diversity
            sample_list = [samples[b, s, :, :].cpu().numpy() for s in range(n_samples)]
            div = compute_diversity(sample_list)
            diversity_results.append(div)

            # Uncertainty calibration
            calib = compute_uncertainty_calibration(sample_list, g_np)
            calibration_results.append(calib)

        if (batch_idx + 1) % 10 == 0:
            logger.info("Evaluated %d batches...", batch_idx + 1)

    # Aggregate
    results = {"all": tracker.summary()}
    if sym_tracker._data:
        results["symmetric"] = sym_tracker.summary()
    if asym_tracker._data:
        results["asymmetric"] = asym_tracker.summary()

    # Diversity aggregates
    if diversity_results:
        results["diversity"] = {
            "mmd": {
                "mean": float(np.mean([d["mmd"] for d in diversity_results])),
                "std": float(np.std([d["mmd"] for d in diversity_results])),
            },
            "cov": {
                "mean": float(np.mean([d["cov"] for d in diversity_results])),
                "std": float(np.std([d["cov"] for d in diversity_results])),
            },
        }

    # Calibration aggregates
    if calibration_results:
        rhos = [c["spearman_rho"] for c in calibration_results]
        results["uncertainty_calibration"] = {
            "spearman_rho": {"mean": float(np.mean(rhos)), "std": float(np.std(rhos))},
        }

    return results


# ---------------------------------------------------------------------------
# PCN evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_on_pcn(
    model: torch.nn.Module,
    pcn_data_dir: str | Path,
    config: dict,
    device: Optional[torch.device] = None,
) -> dict:
    """
    Evaluate on the PCN ShapeNet benchmark (8 categories).

    Expected directory structure:
        pcn_data_dir/
            {category}/
                test/
                    {id}/
                        partial.ply
                        complete.ply

    Returns dict of per-category and overall metrics.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    pcn_dir = Path(pcn_data_dir)
    if not pcn_dir.exists():
        logger.warning("PCN data directory not found: %s", pcn_dir)
        return {}

    import trimesh
    from src.utils.geometry import poisson_disk_sample

    categories = sorted([d.name for d in pcn_dir.iterdir() if d.is_dir()])
    n_points_out = config.get("n_points", 4096)
    n_partial = config.get("n_partial", 2048)

    per_category: dict = {}
    overall_tracker = MetricsTracker()

    for cat in categories:
        cat_tracker = MetricsTracker()
        test_dir = pcn_dir / cat / "test"
        if not test_dir.exists():
            test_dir = pcn_dir / cat
        if not test_dir.exists():
            continue

        sample_dirs = [d for d in test_dir.iterdir() if d.is_dir()]
        for sample_dir in sample_dirs:
            partial_path = sample_dir / "partial.ply"
            complete_path = sample_dir / "complete.ply"

            if not partial_path.exists() or not complete_path.exists():
                # Try .npy format
                partial_path = sample_dir / "partial.npy"
                complete_path = sample_dir / "complete.npy"
                if not partial_path.exists():
                    continue
                partial_pts = torch.from_numpy(
                    np.load(str(partial_path)).astype(np.float32)
                )
                complete_pts = torch.from_numpy(
                    np.load(str(complete_path)).astype(np.float32)
                )
            else:
                from src.utils.io import load_mesh
                try:
                    p_mesh = load_mesh(partial_path)
                    c_mesh = load_mesh(complete_path)
                    partial_pts = poisson_disk_sample(p_mesh, n_partial)
                    complete_pts = poisson_disk_sample(c_mesh, n_points_out)
                except Exception as exc:
                    logger.warning("Could not load %s: %s", sample_dir, exc)
                    continue

            # Resample to fixed size
            from src.datasets.fossil_dataset import _resample
            partial_pts = _resample(partial_pts, n_partial)
            complete_pts = _resample(complete_pts, n_points_out)

            partial_t = partial_pts.unsqueeze(0).to(device)
            full_t = complete_pts.unsqueeze(0).to(device)

            try:
                out = model(partial_t, full_t, mode="inference")
                pred = out["pred"][0]
            except Exception as exc:
                logger.warning("Model inference failed: %s", exc)
                continue

            p_np = pred.cpu().numpy()
            g_np = complete_pts.numpy()

            cat_tracker.update({
                "cd_l1": compute_cd_l1(p_np, g_np),
                "cd_l2": compute_cd_l2(p_np, g_np),
                "fscore": compute_fscore(p_np, g_np),
            })

        per_category[cat] = cat_tracker.summary()
        for k, v in cat_tracker.summary().items():
            overall_tracker.update({k: v["mean"]})

    return {
        "per_category": per_category,
        "overall": overall_tracker.summary(),
    }


# ---------------------------------------------------------------------------
# Ablation table
# ---------------------------------------------------------------------------

def run_ablation_table(
    model_configs: Dict[str, dict],
    dataset,
    device: Optional[torch.device] = None,
) -> pd.DataFrame:
    """
    Generate the ablation table by evaluating each model configuration.

    Parameters
    ----------
    model_configs : dict[name → config_dict]
    dataset       : PaleoCompleteDataset (test split)
    device        : torch device

    Returns pd.DataFrame with rows = model variants, cols = metrics.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from src.models.relic import RELIC

    rows = []
    for model_name, config in model_configs.items():
        logger.info("Evaluating model: %s", model_name)
        model = RELIC(config)

        # Load checkpoint if specified
        ckpt_path = config.get("checkpoint_path")
        if ckpt_path and Path(ckpt_path).exists():
            state = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(state["model_state_dict"])

        results = evaluate_on_paleocomplete(model, dataset, config, device=device)
        summary = results.get("all", {})

        row = {"model": model_name}
        for metric_name, stats in summary.items():
            row[f"{metric_name}_mean"] = stats.get("mean", float("nan"))
            row[f"{metric_name}_std"] = stats.get("std", float("nan"))

        # Diversity
        if "diversity" in results:
            row["mmd_mean"] = results["diversity"]["mmd"]["mean"]
            row["cov_mean"] = results["diversity"]["cov"]["mean"]

        # Calibration
        if "uncertainty_calibration" in results:
            row["spearman_rho"] = results["uncertainty_calibration"]["spearman_rho"]["mean"]

        rows.append(row)

    df = pd.DataFrame(rows).set_index("model")
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Evaluate RELIC on benchmarks")
    parser.add_argument("--benchmark", choices=["paleocomplete", "pcn", "ablation"],
                        default="paleocomplete")
    parser.add_argument("--checkpoint", type=str, required=False)
    parser.add_argument("--config", type=str, default="experiments/relic_full.yaml")
    parser.add_argument("--output", type=str, default="eval_results.json")
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--completeness_level", type=str, default=None)
    parser.add_argument("--pcn_dir", type=str, default="data/pcn")
    args = parser.parse_args()

    with open(args.config) as fh:
        config = yaml.safe_load(fh)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from src.models.relic import RELIC
    model = RELIC(config)

    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        logger.info("Loaded checkpoint from %s", args.checkpoint)

    if args.benchmark == "paleocomplete":
        from src.datasets.fossil_dataset import PaleoCompleteDataset
        dataset = PaleoCompleteDataset(
            paleocomplete_dir=config.get("data", {}).get("paleocomplete_dir", "data/paleocomplete"),
            split=args.split,
            completeness_level=args.completeness_level,
        )
        results = evaluate_on_paleocomplete(
            model, dataset, config, n_samples=args.n_samples, device=device
        )

    elif args.benchmark == "pcn":
        results = evaluate_on_pcn(model, args.pcn_dir, config, device=device)

    elif args.benchmark == "ablation":
        # Load all ablation configs
        config_dir = Path("src/training/configs")
        ablation_configs = {}
        for cfg_file in config_dir.glob("relic_ablation_*.yaml"):
            with open(cfg_file) as fh:
                ablation_configs[cfg_file.stem] = yaml.safe_load(fh)
        ablation_configs["relic_full"] = config

        from src.datasets.fossil_dataset import PaleoCompleteDataset
        dataset = PaleoCompleteDataset(
            paleocomplete_dir=config.get("data", {}).get("paleocomplete_dir", "data/paleocomplete"),
            split="test",
        )
        df = run_ablation_table(ablation_configs, dataset, device=device)
        print(df.to_string())
        df.to_csv(Path(args.output).with_suffix(".csv"))
        results = df.to_dict()

    # Save JSON results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    logger.info("Saved results to %s", output_path)

    # Pretty print
    if args.benchmark != "ablation":
        import pprint
        pprint.pprint(results)


if __name__ == "__main__":
    main()
