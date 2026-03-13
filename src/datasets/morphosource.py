"""
MorphoSource REST API client with quality-based curation filter.

Applies keyword filtering, Gaussian curvature anomaly detection, and assigns
completeness_confidence scores for the two-tier quality system.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import requests
import trimesh

logger = logging.getLogger(__name__)

# Keywords that indicate a specimen is not a pristine complete fossil
EXCLUSION_KEYWORDS = [
    "reconstruction",
    "cast",
    "restored",
    "plaster",
    "composite",
    "missing",
    "juvenile",
    "subadult",
    "hatchling",
    "fetal",
]

MORPHOSOURCE_API = "https://www.morphosource.org"


class MorphoSourceDownloader:
    """
    Downloads and curates open-access fossil meshes from MorphoSource.

    Uses the MorphoSource REST API v1 to search for fossil specimens,
    applies keyword and curvature quality filters, and assigns
    completeness_confidence scores (0–1).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        rate_limit_sleep: float = 1.0,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.api_key = api_key
        self.rate_limit_sleep = rate_limit_sleep
        self.session = session or requests.Session()
        if api_key:
            self.session.headers["Authorization"] = f"Bearer {api_key}"
        self.session.headers["Accept"] = "application/json"

    # ------------------------------------------------------------------
    # Quality filtering
    # ------------------------------------------------------------------

    def filter_specimen(self, metadata: dict) -> bool:
        """
        Return False if the specimen should be excluded.

        Exclusion triggers if any EXCLUSION_KEYWORD appears (case-insensitive)
        in the project title, specimen title, or description fields.
        """
        text_fields = [
            metadata.get("title", ""),
            metadata.get("description", ""),
            metadata.get("project_title", ""),
            metadata.get("project_description", ""),
            metadata.get("media_label", ""),
        ]
        combined = " ".join(str(f) for f in text_fields).lower()
        for kw in EXCLUSION_KEYWORDS:
            if kw in combined:
                logger.debug("Excluding specimen %s — keyword '%s'", metadata.get("id"), kw)
                return False
        return True

    def curvature_anomaly_score(self, mesh_path: str | Path) -> float:
        """
        Compute the fraction of mesh surface area with near-zero Gaussian curvature
        in patches larger than 1 cm².

        Plaster fills and 3D-printed repairs tend to be anomalously smooth relative
        to true fossil bone texture; this heuristic flags them.

        Returns a float in [0, 1]: higher → more anomalous.
        """
        mesh = trimesh.load(str(mesh_path), force="mesh", process=False)
        if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
            return 0.0

        # Approximate per-face Gaussian curvature via the angle deficit method.
        # For each vertex, angle deficit = 2π - sum of incident face angles.
        # We spread deficit to each incident face proportionally.
        vertex_angles = np.zeros(len(mesh.vertices), dtype=np.float64)
        for i, face in enumerate(mesh.faces):
            v0, v1, v2 = (mesh.vertices[face[j]] for j in range(3))
            e01 = v1 - v0
            e02 = v2 - v0
            e12 = v2 - v1
            def _angle(a: np.ndarray, b: np.ndarray) -> float:
                cos_a = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
                return float(np.arccos(np.clip(cos_a, -1.0, 1.0)))
            vertex_angles[face[0]] += _angle(e01, e02)
            vertex_angles[face[1]] += _angle(-e01, e12)
            vertex_angles[face[2]] += _angle(-e02, -e12)

        gaussian_curvature_vertex = 2 * np.pi - vertex_angles  # angle deficit per vertex

        # Map per-vertex curvature to per-face (average of three vertices)
        face_curvatures = gaussian_curvature_vertex[mesh.faces].mean(axis=1)

        # Near-zero threshold: |κ| < 0.05 rad²/unit² ≈ nearly flat
        near_zero = np.abs(face_curvatures) < 0.05

        face_areas = mesh.area_faces  # shape (F,)
        total_area = face_areas.sum()
        if total_area < 1e-12:
            return 0.0

        # Find connected components of near-zero faces with contiguous patches > 1 cm²
        # (1 cm² = 1e-4 m²; units assumed to be mm → 1 cm² = 100 mm²)
        PATCH_AREA_THRESHOLD = 100.0  # mm²; adjust if mesh units differ

        near_zero_indices = np.where(near_zero)[0]
        if len(near_zero_indices) == 0:
            return 0.0

        # Build adjacency for near-zero faces
        face_adj: dict[int, list[int]] = {int(i): [] for i in near_zero_indices}
        near_zero_set = set(near_zero_indices.tolist())

        # Edges between faces sharing two vertices
        edge_to_faces: dict[tuple[int, int], list[int]] = {}
        for fi, face in enumerate(mesh.faces):
            for a, b in [(0, 1), (1, 2), (0, 2)]:
                edge = (min(int(face[a]), int(face[b])), max(int(face[a]), int(face[b])))
                edge_to_faces.setdefault(edge, []).append(fi)
        for faces_sharing in edge_to_faces.values():
            if len(faces_sharing) == 2:
                f1, f2 = faces_sharing
                if f1 in near_zero_set and f2 in near_zero_set:
                    face_adj[f1].append(f2)
                    face_adj[f2].append(f1)

        # BFS to find connected patches
        visited: set[int] = set()
        large_patch_area = 0.0
        for start in near_zero_indices:
            start = int(start)
            if start in visited:
                continue
            # BFS
            queue = [start]
            patch = []
            while queue:
                node = queue.pop()
                if node in visited:
                    continue
                visited.add(node)
                patch.append(node)
                queue.extend(face_adj.get(node, []))
            patch_area = face_areas[np.array(patch)].sum()
            if patch_area >= PATCH_AREA_THRESHOLD:
                large_patch_area += patch_area

        return float(large_patch_area / total_area)

    def compute_completeness_confidence(
        self,
        mesh_path: str | Path,
        keywords_clean: bool,
    ) -> float:
        """
        Assign a completeness_confidence score in [0, 1].

        Rules (following the two-tier system from the project spec):
        - keywords_clean=True, curvature_anomaly < 0.05 → 0.9
        - keywords_clean=True, curvature_anomaly in [0.05, 0.15) → 0.7
        - keywords_clean=True, curvature_anomaly >= 0.15 → 0.55
        - keywords_clean=False → 0.3 (keyword-flagged specimens are not used in test split)
        """
        if not keywords_clean:
            return 0.3
        anomaly = self.curvature_anomaly_score(mesh_path)
        if anomaly < 0.05:
            return 0.9
        elif anomaly < 0.15:
            return 0.7
        else:
            return 0.55

    # ------------------------------------------------------------------
    # API queries
    # ------------------------------------------------------------------

    def _get(self, endpoint: str, params: dict | None = None) -> dict:
        url = f"{MORPHOSOURCE_API.rstrip('/')}/{endpoint.lstrip('/')}"
        resp = self.session.get(url, params=params, timeout=30)
        resp.raise_for_status()
        time.sleep(self.rate_limit_sleep)
        return resp.json()

    def search_media(
        self,
        query: str,
        open_access: bool = True,
        media_type: str = "Mesh",
        page: int = 1,
        per_page: int = 100,
    ) -> dict:
        """Query the MorphoSource media catalog."""
        # Pass params as list of tuples so publication_status[] is encoded correctly
        params: list = [
            ("q", query),
            ("media_type", media_type),
            ("page", str(page)),
            ("per_page", str(per_page)),
        ]
        if open_access:
            params.append(("f[publication_status][]", "Open Download"))
        url = f"{MORPHOSOURCE_API.rstrip('/')}/api/media"
        resp = self.session.get(url, params=params, timeout=30)
        resp.raise_for_status()
        time.sleep(self.rate_limit_sleep)
        return resp.json()

    def get_media_metadata(self, media_id: str) -> dict:
        """Fetch full metadata for a single media record."""
        return self._get(f"api/media/{media_id}")

    def download_file(self, url: str, dest_path: Path) -> None:
        """Stream-download a file to dest_path."""
        resp = self.session.get(url, stream=True, timeout=120)
        resp.raise_for_status()
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dest_path, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=65536):
                fh.write(chunk)

    def download_batch(
        self,
        query: str,
        max_results: int,
        output_dir: str | Path,
        open_access: bool = True,
    ) -> list[dict]:
        """
        Search MorphoSource for `query`, apply curation filter, and download
        passing specimens to `output_dir`.

        Returns a list of metadata dicts for successfully downloaded specimens.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        collected: list[dict] = []
        page = 1
        per_page = 100

        while len(collected) < max_results:
            try:
                response = self.search_media(
                    query=query,
                    open_access=open_access,
                    page=page,
                    per_page=per_page,
                )
            except requests.RequestException as exc:
                logger.error("API error on page %d: %s", page, exc)
                break

            # Response shape: {"response": {"media": [...], "pages": {...}}}
            resp_body = response.get("response", response)
            items = resp_body.get("media", [])
            pages_info = resp_body.get("pages", {})

            if not items:
                break

            for item in items:
                if len(collected) >= max_results:
                    break

                # IDs are returned as single-element arrays
                raw_id = item.get("id", [])
                media_id = raw_id[0] if isinstance(raw_id, list) else str(raw_id)
                if not media_id:
                    continue

                # Skip restricted items (server-side filter is unreliable; double-check)
                vis = item.get("visibility", [])
                if isinstance(vis, list):
                    vis = vis[0] if vis else ""
                if vis == "restricted_download":
                    continue

                # Apply keyword filter using search-result fields directly
                # (avoids a separate metadata fetch for every record)
                flat = {
                    "title": (item.get("title") or [""])[0],
                    "description": (item.get("short_description") or [""])[0],
                    "project_title": "",
                    "project_description": "",
                    "media_label": (item.get("part") or [""])[0],
                }
                if not self.filter_specimen(flat):
                    continue

                # Build download URL: /api/download/{media-id}
                download_url = f"{MORPHOSOURCE_API}/api/download/{media_id}"

                # Derive filename from title or use media_id
                title_str = (item.get("title") or [f"specimen_{media_id}"])[0]
                mesh_path = output_dir / f"{media_id}.ply"

                # Skip if already downloaded
                if mesh_path.exists() and mesh_path.stat().st_size > 1000:
                    logger.info("Already downloaded %s, skipping.", media_id)
                    collected.append({"media_id": media_id, "local_path": str(mesh_path)})
                    continue

                # Download
                try:
                    logger.info("Downloading %s (%s) → %s", media_id, title_str[:50], mesh_path.name)
                    self.download_file(download_url, mesh_path)
                except requests.RequestException as exc:
                    logger.warning("Download failed for %s: %s", media_id, exc)
                    continue

                # Detect actual file format from magic bytes and rename if needed
                if mesh_path.exists() and mesh_path.stat().st_size > 100:
                    with open(mesh_path, "rb") as fh:
                        magic = fh.read(4)
                    ext = ".ply"
                    if magic[:3] == b"ply" or magic[:3] == b"PLY":
                        ext = ".ply"
                    elif magic[:5] == b"solid" or magic[:4] in (b"\x00\x00\x00\x00",):
                        ext = ".stl"
                    elif magic[:2] == b"PK":
                        ext = ".zip"
                    if ext != ".ply":
                        new_path = mesh_path.with_suffix(ext)
                        mesh_path.rename(new_path)
                        mesh_path = new_path

                # Compute completeness confidence
                try:
                    confidence = self.compute_completeness_confidence(
                        mesh_path, keywords_clean=True
                    )
                except Exception:
                    confidence = 0.7
                tier = "A" if confidence >= 0.8 else ("B" if confidence >= 0.5 else "reject")

                if tier == "reject":
                    mesh_path.unlink(missing_ok=True)
                    continue

                record = {
                    **{k: (v[0] if isinstance(v, list) and v else v) for k, v in item.items()},
                    "completeness_confidence": confidence,
                    "tier": tier,
                    "local_path": str(mesh_path),
                    "media_id": media_id,
                }

                meta_path = output_dir / f"{media_id}_metadata.json"
                with open(meta_path, "w") as fh:
                    json.dump(record, fh, indent=2, default=str)

                collected.append(record)
                logger.info(
                    "Accepted %s (confidence=%.2f, tier=%s) [%d/%d]",
                    media_id, confidence, tier, len(collected), max_results,
                )

            # Paginate using the pages object
            if not pages_info.get("next_page"):
                break
            page += 1

        logger.info("Downloaded %d specimens to %s", len(collected), output_dir)
        return collected
