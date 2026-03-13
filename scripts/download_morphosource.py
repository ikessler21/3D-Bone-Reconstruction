"""
Bulk download fossil meshes from MorphoSource using the REST API.

Reads MorphoSource_API_KEY from .env in the repo root.

Usage:
    python scripts/download_morphosource.py
    python scripts/download_morphosource.py --query "fossil bone" --max 500
    python scripts/download_morphosource.py --query "dinosaur" --max 200 --out data/raw/morphosource
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Load .env
try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass  # python-dotenv not installed; rely on shell env

from src.datasets.morphosource import MorphoSourceDownloader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("download_morphosource")

# Queries that together cover the fossil bone catalog.
# Each query is run separately so per-query result limits don't cut off coverage.
DEFAULT_QUERIES = [
    "fossil bone",
    "fossil vertebra",
    "fossil skull",
    "fossil mandible",
    "fossil limb",
    "fossil rib",
    "fossil pelvis",
    "fossil scapula",
    "paleontology skeleton",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bulk-download open-access fossil meshes from MorphoSource."
    )
    parser.add_argument(
        "--query", nargs="+", default=None,
        help="Search query/queries (default: runs a built-in set covering major bone types)",
    )
    parser.add_argument(
        "--max", type=int, default=2000,
        help="Maximum specimens to download per query (default: 2000)",
    )
    parser.add_argument(
        "--out", type=Path,
        default=REPO_ROOT / "data" / "raw" / "morphosource",
        help="Output directory for downloaded meshes",
    )
    parser.add_argument(
        "--rate-limit", type=float, default=0.5,
        help="Seconds to sleep between API calls (default: 0.5)",
    )
    parser.add_argument(
        "--all-access", action="store_true",
        help="Include restricted-access media (not recommended; open-access only by default)",
    )
    args = parser.parse_args()

    api_key = os.environ.get("MorphoSource_API_KEY", "").strip()
    if not api_key:
        log.error(
            "MorphoSource_API_KEY not set. "
            "Add it to .env or export it as an environment variable."
        )
        sys.exit(1)

    queries = args.query or DEFAULT_QUERIES
    downloader = MorphoSourceDownloader(
        api_key=api_key,
        rate_limit_sleep=args.rate_limit,
    )

    total = 0
    for q in queries:
        log.info("=" * 60)
        log.info("Query: %r  (max=%d)", q, args.max)
        log.info("=" * 60)
        results = downloader.download_batch(
            query=q,
            max_results=args.max,
            output_dir=args.out,
            open_access=not args.all_access,
        )
        total += len(results)
        log.info("Query %r: downloaded %d specimens (running total: %d)", q, len(results), total)

    log.info("=" * 60)
    log.info("Done. Total accepted specimens: %d → %s", total, args.out)
    log.info(
        "Next step: run the preprocessor:\n"
        "  python scripts/download_and_preprocess.py --skip-download"
    )


if __name__ == "__main__":
    main()
