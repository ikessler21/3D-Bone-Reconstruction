"""
Bulk download all 3D scans from Phenome10K using Playwright for browser automation.

Reads credentials from .env in the repo root:
    Phenome10K_USERNAME=your@email.com
    Phenome10K_PASSWORD=yourpassword

The login form is JavaScript-rendered, so we use Playwright to authenticate,
then extract CTM file URLs from each scan page and download them via requests.

Files are saved as:  data/raw/phenome10k/<slug>.ctm
Metadata saved as:   data/raw/phenome10k/<slug>_metadata.json

Usage:
    python scripts/download_phenome10k.py
    python scripts/download_phenome10k.py --out data/raw/phenome10k --workers 4
    python scripts/download_phenome10k.py --resume          # skip already-downloaded slugs
    python scripts/download_phenome10k.py --pages 1-5       # download only pages 1–5 (for testing)

Requirements:
    pip install playwright python-dotenv requests tqdm
    playwright install chromium
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass

import requests
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("download_phenome10k")

BASE_URL = "https://www.phenome10k.org"
TOTAL_PAGES = 41


# ---------------------------------------------------------------------------
# Step 1: Login via Playwright and return session cookies for requests
# ---------------------------------------------------------------------------

def login_and_get_cookies() -> dict[str, str]:
    """
    Open a visible Chromium window pointing at the Phenome10K login page.
    Wait for the user to log in manually (handles Cloudflare/CAPTCHA), then
    extract and return the session cookies for use with requests.
    """
    from playwright.sync_api import sync_playwright

    log.info("Opening browser for manual Phenome10K login ...")
    log.info("  → Log in to your account, then come back here and press Enter.")

    # Use a persistent profile so the browser looks like a real returning user
    user_data_dir = Path.home() / ".cache" / "phenome10k_browser_profile"
    user_data_dir.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as pw:
        ctx = pw.chromium.launch_persistent_context(
            str(user_data_dir),
            headless=False,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
            ],
            ignore_default_args=["--enable-automation"],
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
        )
        # Strip the webdriver flag that Cloudflare checks for
        ctx.add_init_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
        )
        page = ctx.new_page()

        page.goto(f"{BASE_URL}/login", wait_until="domcontentloaded", timeout=60_000)

        # Block here until the user confirms they are logged in
        input("\n  [Phenome10K] Press Enter once you have logged in successfully ... ")

        # After user interaction the active page may differ from the original tab
        all_pages = ctx.pages
        current_url = next(
            (p.url for p in all_pages if "/login" not in p.url),
            all_pages[0].url if all_pages else page.url,
        )
        log.info("  Current URL: %s", current_url)

        if "/login" in current_url:
            log.error("Still on the login page — please log in before pressing Enter.")
            ctx.close()
            sys.exit(1)

        cookies = {c["name"]: c["value"] for c in ctx.cookies()}
        log.info("  Captured %d cookies.", len(cookies))
        ctx.close()

    return cookies


# ---------------------------------------------------------------------------
# Step 2: Collect all scan slugs from the 41 listing pages
# ---------------------------------------------------------------------------

def collect_slugs(session: requests.Session, page_range: range) -> list[dict]:
    """
    Fetch each listing page and extract scan slugs + basic metadata from the
    embedded window.p10k_defaultData JavaScript variable.
    Returns a list of dicts with at least {'slug': str, 'scientific_name': str}.
    """
    all_scans: list[dict] = []

    for page_num in tqdm(page_range, desc="Collecting scan list", unit="page"):
        url = f"{BASE_URL}/scans/{page_num}"
        try:
            resp = session.get(url, timeout=20)
            resp.raise_for_status()
        except requests.RequestException as exc:
            log.warning("Could not fetch listing page %d: %s", page_num, exc)
            continue

        # Extract window.p10k_defaultData JSON blob from inline JS
        match = re.search(r"window\.p10k_defaultData\s*=\s*(\{.*?\});", resp.text, re.DOTALL)
        if not match:
            # Fallback: grab hrefs that look like scan slugs
            slugs = re.findall(r'href="/([a-z0-9][a-z0-9\-]+)"', resp.text)
            for slug in set(slugs):
                if slug not in ("login", "scans", "about", "contact", "static"):
                    all_scans.append({"slug": slug, "scientific_name": slug, "page": page_num})
            continue

        try:
            data = json.loads(match.group(1))
        except json.JSONDecodeError:
            log.warning("Could not parse JS data on page %d", page_num)
            continue

        scans = data.get("scans", data.get("results", data.get("data", [])))
        if not scans and isinstance(data, list):
            scans = data

        for scan in scans:
            slug = (
                scan.get("url_slug")
                or scan.get("slug")
                or scan.get("id")
            )
            if slug:
                all_scans.append({
                    "slug": str(slug),
                    "scientific_name": scan.get("scientific_name", slug),
                    "thumbnail": scan.get("thumbnail", ""),
                    "page": page_num,
                })

    log.info("Collected %d scan slugs across %d pages", len(all_scans), len(page_range))
    return all_scans


# ---------------------------------------------------------------------------
# Step 3: For each scan, fetch its page and extract the CTM download URL
# ---------------------------------------------------------------------------

_CTM_RE = re.compile(r'["\'](/files/uploads/[^"\']+\.ctm)["\']', re.IGNORECASE)
_ANY_3D_RE = re.compile(
    r'["\'](/files/[^"\']+\.(ctm|stl|ply|obj|zip))["\']',
    re.IGNORECASE,
)


def get_download_url(session: requests.Session, slug: str) -> Optional[str]:
    """Fetch a scan's detail page and return the CTM (or other 3D) file URL."""
    url = f"{BASE_URL}/{slug}/"
    try:
        resp = session.get(url, timeout=20)
        resp.raise_for_status()
    except requests.RequestException as exc:
        log.debug("Could not fetch scan page %s: %s", slug, exc)
        return None

    # Prefer CTM
    m = _CTM_RE.search(resp.text)
    if m:
        return urljoin(BASE_URL, m.group(1))

    # Fall back to any 3D format
    m = _ANY_3D_RE.search(resp.text)
    if m:
        return urljoin(BASE_URL, m.group(1))

    return None


# ---------------------------------------------------------------------------
# Step 4: Download a single file
# ---------------------------------------------------------------------------

def download_scan(
    session: requests.Session,
    slug: str,
    scientific_name: str,
    out_dir: Path,
) -> Optional[Path]:
    """
    Fetch download URL for `slug`, download the file, and return its local path.
    Returns None if the scan should be skipped or download failed.
    """
    # Determine expected output path (CTM or unknown suffix until we fetch URL)
    meta_path = out_dir / f"{slug}_metadata.json"

    # Skip if already downloaded
    existing = list(out_dir.glob(f"{slug}.*"))
    existing = [f for f in existing if not f.name.endswith("_metadata.json")]
    if existing:
        return existing[0]

    file_url = get_download_url(session, slug)
    if not file_url:
        log.debug("No download URL found for %s", slug)
        return None

    suffix = Path(file_url).suffix or ".ctm"
    dest = out_dir / f"{slug}{suffix}"

    try:
        with session.get(file_url, stream=True, timeout=120) as resp:
            if resp.status_code == 401:
                log.warning("Auth required for %s — session may have expired", slug)
                return None
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            with open(dest, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=65536):
                    fh.write(chunk)
    except requests.RequestException as exc:
        log.debug("Download failed for %s: %s", slug, exc)
        if dest.exists():
            dest.unlink()
        return None

    # Save metadata
    with open(meta_path, "w") as fh:
        json.dump({"slug": slug, "scientific_name": scientific_name,
                   "file_url": file_url, "local_path": str(dest)}, fh, indent=2)

    return dest


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bulk-download all 3D scans from Phenome10K."
    )
    parser.add_argument(
        "--out", type=Path,
        default=REPO_ROOT / "data" / "raw" / "phenome10k",
        help="Output directory (default: data/raw/phenome10k)",
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Parallel download threads (default: 4; be polite to the server)",
    )
    parser.add_argument(
        "--pages", type=str, default=None,
        help="Page range to crawl, e.g. '1-5' or '10' (default: all 41 pages)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip slugs that already have a downloaded file in --out",
    )
    parser.add_argument(
        "--delay", type=float, default=0.3,
        help="Seconds to wait between detail-page fetches (default: 0.3)",
    )
    args = parser.parse_args()

    # Parse page range
    if args.pages:
        if "-" in args.pages:
            start, end = args.pages.split("-", 1)
            page_range = range(int(start), int(end) + 1)
        else:
            page_range = range(int(args.pages), int(args.pages) + 1)
    else:
        page_range = range(1, TOTAL_PAGES + 1)

    args.out.mkdir(parents=True, exist_ok=True)

    # --- Login ---
    cookies = login_and_get_cookies()
    session = requests.Session()
    session.cookies.update(cookies)
    session.headers["User-Agent"] = "RELIC-fossil-research/0.1 (academic)"

    # --- Collect slugs ---
    slugs_path = args.out / "_slugs.json"
    if args.resume and slugs_path.exists():
        with open(slugs_path) as fh:
            all_scans = json.load(fh)
        log.info("Resuming: loaded %d slugs from %s", len(all_scans), slugs_path)
    else:
        all_scans = collect_slugs(session, page_range)
        with open(slugs_path, "w") as fh:
            json.dump(all_scans, fh, indent=2)

    if not all_scans:
        log.error("No scans found. Check login credentials or page range.")
        sys.exit(1)

    # Filter already-downloaded if resuming
    if args.resume:
        pending = [
            s for s in all_scans
            if not list(args.out.glob(f"{s['slug']}.*"))
            or all(f.name.endswith("_metadata.json") for f in args.out.glob(f"{s['slug']}.*"))
        ]
        log.info(
            "Resume mode: %d already done, %d remaining",
            len(all_scans) - len(pending), len(pending),
        )
        all_scans = pending

    # --- Download ---
    log.info("Downloading %d scans with %d worker(s) ...", len(all_scans), args.workers)
    success, fail = 0, 0

    def _download(scan: dict) -> tuple[str, Optional[Path]]:
        time.sleep(args.delay)  # be polite
        path = download_scan(session, scan["slug"], scan["scientific_name"], args.out)
        return scan["slug"], path

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_download, s): s for s in all_scans}
        bar = tqdm(as_completed(futures), total=len(futures), desc="Downloading", unit="scan")
        for future in bar:
            slug, path = future.result()
            if path:
                success += 1
            else:
                fail += 1
            bar.set_postfix(ok=success, fail=fail)

    log.info("=" * 60)
    log.info("Done. %d downloaded, %d skipped/failed → %s", success, fail, args.out)
    log.info(
        "Next step: add 'phenome10k' to download_and_preprocess.py or run:\n"
        "  python scripts/preprocess_phenome10k.py"
    )


if __name__ == "__main__":
    main()
