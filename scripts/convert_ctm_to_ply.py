"""
Batch-convert CTM files to PLY using PyMeshLab.

PyMeshLab is a headless Python API — install it with:
    pip install pymeshlab

After conversion, preprocess_phenome10k.py will pick up the PLY files
instead of the CTM files, avoiding C-extension segfaults in openctm.

Usage:
    python scripts/convert_ctm_to_ply.py
    python scripts/convert_ctm_to_ply.py --raw data/raw/phenome10k
    python scripts/convert_ctm_to_ply.py --delete-ctm   # remove CTM after success
    python scripts/convert_ctm_to_ply.py --resume       # skip already-converted
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from tqdm import tqdm

REPO_ROOT = Path(__file__).parent.parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("convert_ctm_to_ply")


def convert_one(ctm_path: Path, ply_path: Path) -> str | None:
    """Convert a single CTM file to PLY. Returns None on success, error string on failure."""
    try:
        import pymeshlab
    except ImportError:
        return "pymeshlab not installed — run: pip install pymeshlab"

    try:
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(str(ctm_path))
        ms.save_current_mesh(str(ply_path))
        return None  # success
    except Exception as exc:
        if ply_path.exists():
            ply_path.unlink()
        return str(exc)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch-convert Phenome10K CTM files to PLY via PyMeshLab."
    )
    parser.add_argument(
        "--raw", type=Path,
        default=REPO_ROOT / "data" / "raw" / "phenome10k",
        help="Directory containing CTM files (default: data/raw/phenome10k)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip CTM files that already have a matching PLY",
    )
    parser.add_argument(
        "--delete-ctm", action="store_true",
        help="Delete the CTM file after a successful conversion",
    )
    args = parser.parse_args()

    if not args.raw.exists():
        log.error("Directory not found: %s", args.raw)
        sys.exit(1)

    ctm_files = sorted(args.raw.glob("*.ctm"))
    if not ctm_files:
        log.error("No CTM files found in %s", args.raw)
        sys.exit(1)

    log.info("Found %d CTM files in %s", len(ctm_files), args.raw)

    tasks = []
    skipped = 0
    for ctm_path in ctm_files:
        ply_path = ctm_path.with_suffix(".ply")
        if args.resume and ply_path.exists():
            skipped += 1
            continue
        tasks.append((ctm_path, ply_path))

    log.info("Tasks: %d to convert, %d already done", len(tasks), skipped)

    success, fail = 0, 0
    bar = tqdm(tasks, total=len(tasks), desc="Converting", unit="file")
    for ctm_path, ply_path in bar:
        err = convert_one(ctm_path, ply_path)
        if err is None:
            success += 1
            if args.delete_ctm:
                ctm_path.unlink()
        else:
            fail += 1
            log.warning("  failed %s: %s", ctm_path.name, err)
        bar.set_postfix(ok=success, fail=fail)

    log.info("=" * 60)
    log.info("Done: %d converted, %d failed", success, fail)
    if fail:
        log.info("Failed files were skipped; preprocess_phenome10k.py will ignore them.")
    log.info(
        "Next: python scripts/preprocess_phenome10k.py --resume"
    )


if __name__ == "__main__":
    main()
