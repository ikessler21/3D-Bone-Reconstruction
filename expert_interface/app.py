"""
RELIC Expert-in-the-Loop Gradio Interface.

Provides:
- File upload for partial fossil scan (.ply/.obj/.stl)
- Text input (taxon description → BioBERT + OTL phylo lookup)
- Reference image input → CLIP encoder
- Top-5 diverse completions in 3D viewer
- Per-point uncertainty heatmap
- Symmetry plane overlay + confidence score
- Download button for selected completion
- 2AFC rating widget → SQLite feedback database

Usage:
    python expert_interface/app.py --checkpoint path/to/relic.pth
    python expert_interface/app.py --checkpoint relic.pth --share   # public demo
"""

from __future__ import annotations

import argparse
import logging
import os
import sqlite3
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def init_db(db_path: str) -> None:
    """Create feedback DB tables if they don't exist."""
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ratings (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT,
            specimen_id TEXT,
            taxon_text  TEXT,
            rating      TEXT,
            relic_better INTEGER,
            notes       TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       TEXT,
            specimen_file   TEXT,
            taxon_text      TEXT,
            n_completions   INTEGER,
            selected_idx    INTEGER
        )
    """)
    conn.commit()
    conn.close()


def save_rating(db_path: str, specimen_id: str, taxon_text: str, relic_better: bool, notes: str = "") -> None:
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO ratings (timestamp, specimen_id, taxon_text, rating, relic_better, notes) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (datetime.utcnow().isoformat(), specimen_id, taxon_text,
         "relic_better" if relic_better else "baseline_better",
         1 if relic_better else 0,
         notes),
    )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

_model_cache: dict = {}


def load_model(checkpoint_path: str, config_path: Optional[str] = None, device: str = "cpu"):
    """Load RELIC from checkpoint, cached after first call."""
    if checkpoint_path in _model_cache:
        return _model_cache[checkpoint_path]

    # Find config
    if config_path is None:
        config_path = str(Path(checkpoint_path).parent / "config.yaml")
        if not Path(config_path).exists():
            config_path = "experiments/relic_full.yaml"

    with open(config_path) as fh:
        config = yaml.safe_load(fh)

    # Add src to path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.models.relic import RELIC

    model = RELIC(config)
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model = model.to(device).eval()
    _model_cache[checkpoint_path] = (model, config)
    logger.info("Loaded RELIC from %s", checkpoint_path)
    return model, config


# ---------------------------------------------------------------------------
# Processing helpers
# ---------------------------------------------------------------------------

def process_upload(file_path: str, n_points: int = 2048) -> torch.Tensor:
    """Load a mesh file and sample n_points from it."""
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.utils.io import load_mesh
    from src.utils.geometry import poisson_disk_sample, pca_normalize
    from src.datasets.fossil_dataset import _resample

    mesh = load_mesh(file_path)
    pts = poisson_disk_sample(mesh, n_points)
    pts, _ = pca_normalize(pts)
    pts = _resample(pts, n_points)
    return pts.unsqueeze(0)  # [1, N, 3]


def encode_text_conditioning(taxon_text: str, model, device: str) -> Optional[dict]:
    """Encode taxon description text to conditioning dict."""
    if not taxon_text.strip():
        return None
    try:
        tokenizer = model.taxonomy_encoder.morpho_enc.tokenizer
        if tokenizer is None:
            return None
        enc = tokenizer(
            [taxon_text], padding=True, truncation=True, max_length=128, return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].to(device),
            "attention_mask": enc["attention_mask"].to(device),
        }
    except Exception as exc:
        logger.warning("Text conditioning failed: %s", exc)
        return None


def export_completion_ply(points: np.ndarray, save_path: str) -> None:
    """Export a point cloud as a .ply file."""
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.utils.io import save_point_cloud
    save_point_cloud(points, save_path)


def make_colored_ply(points: np.ndarray, uncertainty: np.ndarray, save_path: str) -> str:
    """Create a colored .ply file with uncertainty heatmap using trimesh."""
    import matplotlib.pyplot as plt
    import trimesh

    unc_norm = (uncertainty - uncertainty.min()) / (uncertainty.max() - uncertainty.min() + 1e-12)
    cmap = plt.get_cmap("plasma")
    colors_rgba = (cmap(unc_norm) * 255).astype(np.uint8)  # [N, 4]

    cloud = trimesh.points.PointCloud(points.astype(np.float64), colors=colors_rgba)
    cloud.export(save_path)
    return save_path


# ---------------------------------------------------------------------------
# Main Gradio app
# ---------------------------------------------------------------------------

_RELIC_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,500;0,600;1,300;1,400&family=Azeret+Mono:wght@300;400;500&family=Source+Serif+4:ital,opsz,wght@0,8..60,300;0,8..60,400;1,8..60,300&display=swap');

/* ── Variables ──────────────────────────────────────── */
:root {
  --bg-void:     #070a0d;
  --bg-deep:     #0c1118;
  --bg-surface:  #111921;
  --bg-raised:   #182030;
  --bg-panel:    #1d2840;

  --amber:       #c8923a;
  --amber-hi:    #e8aa48;
  --amber-lo:    #7a5520;
  --amber-glow:  rgba(200, 146, 58, 0.14);
  --amber-edge:  rgba(200, 146, 58, 0.35);

  --bone:        #e0cfb4;
  --bone-dim:    #9a8a70;
  --bone-ghost:  rgba(224, 207, 180, 0.06);

  --terra:       #8b4030;
  --terra-hi:    #b05040;

  --text-hi:     #ddd0b5;
  --text-mid:    #8090a8;
  --text-lo:     #3d4d60;

  --border:      rgba(200, 146, 58, 0.22);
  --border-sub:  rgba(224, 207, 180, 0.07);
  --border-line: rgba(200, 146, 58, 0.45);

  --font-disp:  'Cormorant Garamond', Georgia, serif;
  --font-mono:  'Azeret Mono', 'Courier New', monospace;
  --font-body:  'Source Serif 4', Georgia, serif;

  --ease: cubic-bezier(0.4, 0, 0.2, 1);
}

/* ── Page base ─────────────────────────────────────── */
body, .gradio-container {
  background: var(--bg-void) !important;
  color: var(--text-hi) !important;
  font-family: var(--font-body) !important;
}

.gradio-container {
  max-width: 1600px !important;
  margin: 0 auto !important;
  padding: 0 !important;
}

/* Subtle grain texture overlay */
.gradio-container::before {
  content: '';
  position: fixed;
  inset: 0;
  background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)' opacity='0.035'/%3E%3C/svg%3E");
  pointer-events: none;
  z-index: 9999;
  opacity: 0.6;
}

/* ── Strata separator ──────────────────────────────── */
.relic-strata {
  width: 100%;
  height: 3px;
  background: linear-gradient(90deg,
    transparent 0%,
    var(--amber-lo) 15%,
    var(--amber) 40%,
    var(--amber-hi) 50%,
    var(--amber) 60%,
    var(--amber-lo) 85%,
    transparent 100%);
  margin: 0;
  opacity: 0.7;
}

.relic-strata-thin {
  width: 100%;
  height: 1px;
  background: linear-gradient(90deg,
    transparent 0%,
    var(--border) 20%,
    var(--amber-edge) 50%,
    var(--border) 80%,
    transparent 100%);
  margin: 8px 0 24px;
}

/* ── Hero header ────────────────────────────────────── */
.relic-hero {
  padding: 56px 64px 40px;
  background: linear-gradient(180deg, rgba(12,17,24,0) 0%, rgba(12,17,24,0.6) 100%),
              linear-gradient(135deg, rgba(200,146,58,0.04) 0%, transparent 60%);
  border-bottom: 1px solid var(--border-sub);
  position: relative;
  overflow: hidden;
}

.relic-hero::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 3px;
  background: linear-gradient(90deg,
    transparent, var(--amber-lo), var(--amber), var(--amber-hi),
    var(--amber), var(--amber-lo), transparent);
}

.relic-hero-eyebrow {
  font-family: var(--font-mono);
  font-size: 10px;
  letter-spacing: 0.28em;
  color: var(--amber);
  text-transform: uppercase;
  margin-bottom: 16px;
  opacity: 0.85;
}

.relic-hero h1 {
  font-family: var(--font-disp) !important;
  font-size: clamp(32px, 4vw, 56px) !important;
  font-weight: 400 !important;
  line-height: 1.05 !important;
  color: var(--bone) !important;
  letter-spacing: -0.01em !important;
  margin: 0 0 6px !important;
}

.relic-hero h1 em {
  color: var(--amber);
  font-style: italic;
}

.relic-hero-sub {
  font-family: var(--font-disp);
  font-size: 18px;
  font-weight: 300;
  font-style: italic;
  color: var(--bone-dim);
  letter-spacing: 0.01em;
  margin-bottom: 20px;
}

.relic-hero-desc {
  font-family: var(--font-body);
  font-size: 14px;
  color: var(--text-mid);
  max-width: 640px;
  line-height: 1.7;
}

.relic-badge {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-family: var(--font-mono);
  font-size: 10px;
  letter-spacing: 0.15em;
  color: var(--amber);
  border: 1px solid var(--amber-edge);
  padding: 3px 10px 3px 8px;
  background: var(--amber-glow);
  margin-right: 8px;
  margin-top: 16px;
}

.relic-badge::before {
  content: '';
  width: 5px; height: 5px;
  background: var(--amber);
  border-radius: 50%;
  animation: pulse-badge 2.4s ease-in-out infinite;
}

@keyframes pulse-badge {
  0%, 100% { opacity: 1; transform: scale(1); }
  50% { opacity: 0.4; transform: scale(0.7); }
}

/* ── Section header component ──────────────────────── */
.relic-section {
  padding: 0 64px;
}

.relic-section-header {
  display: flex;
  align-items: baseline;
  gap: 16px;
  padding: 28px 0 16px;
}

.relic-section-num {
  font-family: var(--font-mono);
  font-size: 11px;
  color: var(--amber);
  letter-spacing: 0.12em;
  opacity: 0.7;
  min-width: 28px;
}

.relic-section-title {
  font-family: var(--font-disp);
  font-size: 22px;
  font-weight: 400;
  color: var(--bone);
  letter-spacing: 0.01em;
}

.relic-section-desc {
  font-family: var(--font-mono);
  font-size: 10px;
  color: var(--text-mid);
  letter-spacing: 0.1em;
  margin-left: auto;
  text-transform: uppercase;
}

/* ── Panels / cards ─────────────────────────────────── */
.relic-panel {
  background: var(--bg-surface);
  border: 1px solid var(--border-sub);
  padding: 28px;
  position: relative;
}

.relic-panel::before {
  content: '';
  position: absolute;
  top: 0; left: 0;
  width: 3px; height: 100%;
  background: linear-gradient(180deg, var(--amber) 0%, var(--amber-lo) 100%);
  opacity: 0.6;
}

.relic-data-card {
  background: var(--bg-raised);
  border: 1px solid var(--border-sub);
  padding: 20px 24px;
  position: relative;
}

/* ── Gradio component overrides ─────────────────────── */

/* Labels */
.gradio-container label,
.gradio-container .label-wrap,
.gradio-container .svelte-1ipelgc {
  font-family: var(--font-mono) !important;
  font-size: 10px !important;
  letter-spacing: 0.18em !important;
  text-transform: uppercase !important;
  color: var(--bone-dim) !important;
  font-weight: 400 !important;
}

/* Inputs / textareas */
.gradio-container input[type="text"],
.gradio-container input[type="number"],
.gradio-container textarea {
  background: var(--bg-raised) !important;
  border: 1px solid var(--border-sub) !important;
  border-radius: 0 !important;
  color: var(--text-hi) !important;
  font-family: var(--font-mono) !important;
  font-size: 12px !important;
  padding: 10px 14px !important;
  transition: border-color 200ms var(--ease), background 200ms var(--ease) !important;
}

.gradio-container input[type="text"]:focus,
.gradio-container input[type="number"]:focus,
.gradio-container textarea:focus {
  border-color: var(--amber-edge) !important;
  background: var(--bg-panel) !important;
  outline: none !important;
  box-shadow: 0 0 0 1px var(--amber-glow) inset !important;
}

/* File upload */
.gradio-container .upload-container,
.gradio-container [data-testid="file"] {
  background: var(--bg-raised) !important;
  border: 1px dashed var(--border) !important;
  border-radius: 0 !important;
  transition: border-color 200ms var(--ease), background 200ms var(--ease) !important;
}

.gradio-container .upload-container:hover,
.gradio-container [data-testid="file"]:hover {
  border-color: var(--amber) !important;
  background: var(--amber-glow) !important;
}

/* Sliders */
.gradio-container input[type="range"] {
  accent-color: var(--amber) !important;
  background: transparent !important;
}

.gradio-container .range-slider {
  background: var(--bg-raised) !important;
  border: 1px solid var(--border-sub) !important;
  border-radius: 0 !important;
  padding: 12px 14px !important;
}

/* Select / dropdown */
.gradio-container select,
.gradio-container .wrap-inner {
  background: var(--bg-raised) !important;
  border: 1px solid var(--border-sub) !important;
  border-radius: 0 !important;
  color: var(--text-hi) !important;
  font-family: var(--font-mono) !important;
  font-size: 12px !important;
}

/* Blocks / containers */
.gradio-container .block,
.gradio-container .form,
.gradio-container .gap {
  background: transparent !important;
  border: none !important;
  border-radius: 0 !important;
  padding: 0 !important;
  gap: 14px !important;
}

.gradio-container .block.padded {
  padding: 0 !important;
}

/* Image component */
.gradio-container .image-frame img {
  border: 1px solid var(--border-sub) !important;
}

/* ── Buttons ─────────────────────────────────────────── */
.gradio-container button {
  border-radius: 0 !important;
  font-family: var(--font-mono) !important;
  letter-spacing: 0.14em !important;
  text-transform: uppercase !important;
  font-size: 11px !important;
  transition: all 200ms var(--ease) !important;
  cursor: pointer !important;
}

/* Primary — Run RELIC */
.gradio-container button.primary,
#run-btn {
  background: var(--amber) !important;
  color: var(--bg-void) !important;
  border: 1px solid var(--amber) !important;
  padding: 14px 28px !important;
  font-weight: 500 !important;
  position: relative !important;
  overflow: hidden !important;
}

.gradio-container button.primary::after {
  content: '';
  position: absolute;
  inset: 0;
  background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, transparent 60%);
  pointer-events: none;
}

.gradio-container button.primary:hover {
  background: var(--amber-hi) !important;
  border-color: var(--amber-hi) !important;
  box-shadow: 0 0 20px rgba(200,146,58,0.4) !important;
}

/* Secondary */
.gradio-container button.secondary {
  background: var(--bg-raised) !important;
  color: var(--bone-dim) !important;
  border: 1px solid var(--border) !important;
  padding: 12px 20px !important;
}

.gradio-container button.secondary:hover {
  background: var(--amber-glow) !important;
  color: var(--amber-hi) !important;
  border-color: var(--amber-edge) !important;
}

/* Rating buttons */
#btn-relic-better {
  background: rgba(74, 138, 90, 0.12) !important;
  border-color: rgba(74, 138, 90, 0.4) !important;
  color: #7abf8a !important;
}
#btn-relic-better:hover {
  background: rgba(74, 138, 90, 0.25) !important;
  box-shadow: 0 0 16px rgba(74, 138, 90, 0.25) !important;
}

#btn-baseline-better {
  background: rgba(139, 64, 48, 0.12) !important;
  border-color: rgba(139, 64, 48, 0.4) !important;
  color: #c87060 !important;
}
#btn-baseline-better:hover {
  background: rgba(139, 64, 48, 0.25) !important;
  box-shadow: 0 0 16px rgba(139, 64, 48, 0.25) !important;
}

/* Download button */
#dl-btn {
  background: var(--bg-raised) !important;
  border: 1px solid var(--border) !important;
  color: var(--amber) !important;
}
#dl-btn:hover {
  background: var(--amber-glow) !important;
  border-color: var(--amber) !important;
}

/* ── 3-D viewer panels ──────────────────────────────── */
.relic-viewers-grid {
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  gap: 10px;
  padding: 0 64px 32px;
}

.relic-viewer-wrap {
  position: relative;
  background: var(--bg-surface);
  border: 1px solid var(--border-sub);
  transition: border-color 200ms var(--ease), box-shadow 200ms var(--ease);
}

.relic-viewer-wrap:hover {
  border-color: var(--amber-edge);
  box-shadow: 0 0 28px rgba(200,146,58,0.08);
}

.relic-viewer-num {
  position: absolute;
  top: 10px; left: 14px;
  z-index: 10;
  font-family: var(--font-mono);
  font-size: 9px;
  letter-spacing: 0.2em;
  color: var(--amber);
  opacity: 0.75;
  pointer-events: none;
}

.relic-viewer-tag {
  position: absolute;
  bottom: 10px; right: 10px;
  z-index: 10;
  font-family: var(--font-mono);
  font-size: 8px;
  letter-spacing: 0.15em;
  color: var(--text-lo);
  pointer-events: none;
  text-transform: uppercase;
}

.relic-viewer-wrap .gradio-container,
.relic-viewer-wrap canvas {
  border-radius: 0 !important;
}

/* Make Model3D fill the wrap */
.relic-viewers-grid .model3D-viewer,
.relic-viewers-grid [data-testid="model3D"] {
  background: #090d12 !important;
  border: none !important;
  min-height: 240px !important;
  border-radius: 0 !important;
}

/* ── Analysis row ─────────────────────────────────────── */
.relic-analysis-row {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 10px;
  padding: 0 64px 32px;
}

.relic-analysis-panel {
  background: var(--bg-surface);
  border: 1px solid var(--border-sub);
  padding: 24px;
  position: relative;
}

.relic-analysis-panel::after {
  content: '';
  position: absolute;
  bottom: 0; left: 0; right: 0;
  height: 1px;
  background: linear-gradient(90deg, transparent, var(--amber-lo), transparent);
  opacity: 0.4;
}

.relic-analysis-label {
  font-family: var(--font-mono);
  font-size: 9px;
  letter-spacing: 0.25em;
  color: var(--amber);
  text-transform: uppercase;
  margin-bottom: 14px;
  display: flex;
  align-items: center;
  gap: 8px;
}

.relic-analysis-label::before {
  content: '';
  display: block;
  width: 16px; height: 1px;
  background: var(--amber);
  opacity: 0.6;
}

/* Override image/textbox inside analysis panels */
.relic-analysis-panel .gradio-container {
  padding: 0 !important;
}

/* Symmetry textbox styling */
#sym-text textarea {
  background: var(--bg-void) !important;
  border: 1px solid var(--border-sub) !important;
  font-family: var(--font-mono) !important;
  font-size: 12px !important;
  color: var(--text-hi) !important;
  line-height: 1.9 !important;
  padding: 16px !important;
}

/* Status / rating status boxes */
#status-box textarea,
#rating-status textarea {
  background: var(--bg-void) !important;
  border-top: 1px solid var(--border-sub) !important;
  border-left: none !important;
  border-right: none !important;
  border-bottom: none !important;
  font-family: var(--font-mono) !important;
  font-size: 11px !important;
  color: var(--amber) !important;
  letter-spacing: 0.06em !important;
  padding: 10px 16px !important;
}

/* ── Download row ──────────────────────────────────── */
.relic-download-row {
  padding: 0 64px 32px;
  display: grid;
  grid-template-columns: 1fr auto 1fr;
  align-items: end;
  gap: 12px;
}

/* ── Rating section ────────────────────────────────── */
.relic-rating-section {
  margin: 0 64px 64px;
  background: var(--bg-surface);
  border: 1px solid var(--border-sub);
  padding: 32px;
  position: relative;
  overflow: hidden;
}

.relic-rating-section::before {
  content: '2AFC';
  position: absolute;
  top: -16px; right: 32px;
  font-family: var(--font-disp);
  font-size: 96px;
  font-weight: 600;
  color: var(--bone-ghost);
  letter-spacing: -0.04em;
  pointer-events: none;
  line-height: 1;
}

.relic-rating-prompt {
  font-family: var(--font-disp);
  font-size: 18px;
  font-style: italic;
  color: var(--bone-dim);
  margin-bottom: 6px;
}

.relic-rating-meta {
  font-family: var(--font-mono);
  font-size: 10px;
  letter-spacing: 0.12em;
  color: var(--text-lo);
  text-transform: uppercase;
  margin-bottom: 24px;
}

/* ── Footer ────────────────────────────────────────── */
.relic-footer {
  padding: 24px 64px;
  border-top: 1px solid var(--border-sub);
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.relic-footer-left {
  font-family: var(--font-mono);
  font-size: 10px;
  letter-spacing: 0.12em;
  color: var(--text-lo);
  text-transform: uppercase;
}

.relic-footer-right {
  font-family: var(--font-mono);
  font-size: 10px;
  color: var(--text-lo);
}

/* ── Animations ───────────────────────────────────── */
@keyframes fade-up {
  from { opacity: 0; transform: translateY(12px); }
  to   { opacity: 1; transform: translateY(0); }
}

.relic-hero    { animation: fade-up 0.6s var(--ease) both; }
.relic-section { animation: fade-up 0.6s 0.1s var(--ease) both; }

/* scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-void); }
::-webkit-scrollbar-thumb { background: var(--border); }
::-webkit-scrollbar-thumb:hover { background: var(--amber-lo); }
"""

# Section header HTML template
def _sec(num: str, title: str, desc: str = "") -> str:
    return f"""
<div class="relic-section-header">
  <span class="relic-section-num">{num}</span>
  <span class="relic-section-title">{title}</span>
  {"<span class='relic-section-desc'>" + desc + "</span>" if desc else ""}
</div>
<div class="relic-strata-thin"></div>
"""


def build_app(checkpoint_path: str, config_path: Optional[str] = None, db_path: str = "feedback.db"):
    """Build and return the Gradio app."""
    try:
        import gradio as gr
    except ImportError:
        raise ImportError("gradio is required: pip install gradio>=4.20")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    init_db(db_path)

    # ----------------------------------------------------------------
    # Processing function
    # ----------------------------------------------------------------

    def run_inference(
        mesh_file,
        taxon_text: str,
        ref_image,
        n_completions: int,
        guidance_scale: float,
    ):
        """
        Main inference callback.

        Returns: (completion_1, ..., completion_5, uncertainty_img, sym_text, status_text)
        """
        if mesh_file is None:
            return [None] * 5 + [None, "No symmetry info", "Please upload a mesh file."]

        try:
            model, config = load_model(checkpoint_path, config_path, device)
        except Exception as exc:
            return [None] * 5 + [None, "", f"Error loading model: {exc}"]

        try:
            # Load mesh
            partial = process_upload(mesh_file.name, n_points=config.get("n_partial", 2048))
            partial = partial.to(device)

            # Build conditioning
            cond = {}
            text_cond = encode_text_conditioning(taxon_text, model, device)
            if text_cond:
                cond.update(text_cond)

            # Image conditioning
            if ref_image is not None:
                try:
                    from PIL import Image
                    from transformers import CLIPProcessor
                    proc = model.taxonomy_encoder.image_enc.processor
                    if proc is not None:
                        img = Image.fromarray(ref_image).convert("RGB")
                        enc = proc(images=img, return_tensors="pt")
                        cond["pixel_values"] = enc["pixel_values"].to(device)
                except Exception as exc:
                    logger.warning("Image conditioning failed: %s", exc)

            model.diffusion.sampler.num_steps = 20
            model.guidance_scale = guidance_scale

            # Sample completions
            n_samples = min(int(n_completions), 5)
            with torch.no_grad():
                samples = model.sample(
                    partial,
                    conditioning=cond if cond else None,
                    n_samples=n_samples,
                )  # [1, S, M, 3]

            samples_np = samples[0].cpu().numpy()  # [S, M, 3]

            # Get uncertainty map
            uncertainty = model.get_uncertainty_map(samples[0]).cpu().numpy()

            # Get symmetry info
            sym_conf = 0.0
            sym_text = "Symmetry: not computed"
            if hasattr(model.vae, "symmetry_module") and model.vae.symmetry_module is not None:
                n_t, o_t, c_t = model.vae.symmetry_module.detector.detect_batch(partial)
                sym_conf = float(c_t[0].item())
                n_arr = n_t[0].cpu().numpy()
                sym_text = (
                    f"Symmetry confidence: {sym_conf:.3f}\n"
                    f"Plane normal: [{n_arr[0]:.3f}, {n_arr[1]:.3f}, {n_arr[2]:.3f}]\n"
                    f"Plane offset: {float(o_t[0].item()):.4f}"
                )

            # Export completions as temp .ply files for Gradio Model3D
            completion_paths = []
            tmpdir = tempfile.mkdtemp()
            for s in range(5):
                if s < n_samples:
                    ply_path = os.path.join(tmpdir, f"completion_{s}.ply")
                    export_completion_ply(samples_np[s], ply_path)
                else:
                    ply_path = None
                completion_paths.append(ply_path)

            # Uncertainty heatmap image
            unc_ply = os.path.join(tmpdir, "uncertainty.ply")
            make_colored_ply(samples_np[0], uncertainty, unc_ply)
            # Also create a matplotlib PNG for display
            unc_img_path = os.path.join(tmpdir, "uncertainty_heatmap.png")
            try:
                import matplotlib.pyplot as plt
                fig = plt.figure(figsize=(8, 6))
                ax = fig.add_subplot(111, projection="3d")
                unc_norm = (uncertainty - uncertainty.min()) / (uncertainty.max() - uncertainty.min() + 1e-12)
                sc = ax.scatter(
                    samples_np[0, :, 0], samples_np[0, :, 1], samples_np[0, :, 2],
                    c=unc_norm, cmap="plasma", s=1, alpha=0.7,
                )
                plt.colorbar(sc, ax=ax, label="Uncertainty")
                ax.set_title("Per-point uncertainty heatmap")
                ax.set_axis_off()
                plt.tight_layout()
                plt.savefig(unc_img_path, dpi=100, bbox_inches="tight")
                plt.close()
            except Exception:
                unc_img_path = None

            status = f"Generated {n_samples} completions. Model loaded from {checkpoint_path}"
            return completion_paths + [unc_img_path, sym_text, status]

        except Exception as exc:
            logger.exception("Inference failed")
            return [None] * 5 + [None, "", f"Error during inference: {exc}"]

    def rate_better(taxon_text: str, notes: str):
        """Save 2AFC rating: RELIC is better."""
        save_rating(db_path, "session", taxon_text, relic_better=True, notes=notes)
        return "Saved: RELIC better ✓"

    def rate_worse(taxon_text: str, notes: str):
        """Save 2AFC rating: Baseline (AdaPoinTr) is better."""
        save_rating(db_path, "session", taxon_text, relic_better=False, notes=notes)
        return "Saved: Baseline better ✓"

    def download_completion(ply_path: str, save_name: str):
        """Return the selected completion for download."""
        if ply_path is None:
            return None
        return ply_path

    # ----------------------------------------------------------------
    # Build Gradio UI
    # ----------------------------------------------------------------

    import gradio as gr

    _theme = gr.themes.Base(
        primary_hue=gr.themes.colors.orange,
        secondary_hue=gr.themes.colors.gray,
        neutral_hue=gr.themes.colors.slate,
        font=[gr.themes.GoogleFont("Azeret Mono"), "monospace"],
        font_mono=[gr.themes.GoogleFont("Azeret Mono"), "monospace"],
    ).set(
        body_background_fill="#070a0d",
        body_background_fill_dark="#070a0d",
        background_fill_primary="#111921",
        background_fill_primary_dark="#111921",
        background_fill_secondary="#182030",
        background_fill_secondary_dark="#182030",
        border_color_primary="rgba(200,146,58,0.22)",
        border_color_primary_dark="rgba(200,146,58,0.22)",
        color_accent="#c8923a",
        color_accent_soft="rgba(200,146,58,0.14)",
        block_background_fill="#111921",
        block_background_fill_dark="#111921",
        block_border_color="rgba(224,207,180,0.07)",
        block_border_color_dark="rgba(224,207,180,0.07)",
        block_label_text_color="#9a8a70",
        block_label_text_color_dark="#9a8a70",
        block_title_text_color="#e0cfb4",
        block_title_text_color_dark="#e0cfb4",
        input_background_fill="#182030",
        input_background_fill_dark="#182030",
        input_border_color="rgba(224,207,180,0.07)",
        input_border_color_dark="rgba(224,207,180,0.07)",
        input_border_color_focus="#c8923a",
        input_border_color_focus_dark="#c8923a",
        button_primary_background_fill="#c8923a",
        button_primary_background_fill_dark="#c8923a",
        button_primary_background_fill_hover="#e8aa48",
        button_primary_background_fill_hover_dark="#e8aa48",
        button_primary_text_color="#070a0d",
        button_primary_text_color_dark="#070a0d",
        button_secondary_background_fill="#182030",
        button_secondary_background_fill_dark="#182030",
        button_secondary_background_fill_hover="rgba(200,146,58,0.14)",
        button_secondary_background_fill_hover_dark="rgba(200,146,58,0.14)",
        button_secondary_border_color="rgba(200,146,58,0.22)",
        button_secondary_border_color_dark="rgba(200,146,58,0.22)",
        button_secondary_text_color="#9a8a70",
        button_secondary_text_color_dark="#9a8a70",
        slider_color="#c8923a",
        table_odd_background_fill="#111921",
        table_even_background_fill="#0c1118",
    )

    with gr.Blocks(title="RELIC — Fossil Bone Reconstruction") as demo:

        # ── Hero ──────────────────────────────────────────────────
        gr.HTML("""
        <div class="relic-strata"></div>
        <div class="relic-hero">
          <div class="relic-hero-eyebrow">Palaeontological Reconstruction System · v2.0</div>
          <h1>RELIC — <em>Reconstruction of</em><br>Extinct Life via Informed Completion</h1>
          <div class="relic-hero-sub">
            Phylogenetically-conditioned latent diffusion · Confidence-gated bilateral symmetry
          </div>
          <div class="relic-hero-desc">
            Upload a fragmentary fossil bone point cloud to generate diverse, anatomically plausible
            completions conditioned on taxonomic identity, morphological description, and optional
            reference imagery. Per-point uncertainty heatmaps validate reconstruction confidence.
          </div>
          <div>
            <span class="relic-badge">DDIM 20-step inference</span>
            <span class="relic-badge">PaleoComplete benchmark</span>
            <span class="relic-badge">OSF pre-registered</span>
          </div>
        </div>
        <div class="relic-strata"></div>
        """)

        # ── Section 01 · Specimen Input ────────────────────────────
        with gr.Row(elem_classes=["relic-section"]):
            gr.HTML(_sec("01", "Specimen Input", "conditioning parameters"))

        with gr.Row(equal_height=True, elem_classes=["relic-section"]):
            with gr.Column(scale=1, min_width=280):
                gr.HTML("""<div class="relic-analysis-label">3D scan</div>""")
                mesh_input = gr.File(
                    label="Partial fossil scan (.ply / .obj / .stl)",
                    file_types=[".ply", ".obj", ".stl"],
                    elem_id="mesh-input",
                )
            with gr.Column(scale=1, min_width=280):
                gr.HTML("""<div class="relic-analysis-label">Taxonomic conditioning</div>""")
                taxon_input = gr.Textbox(
                    label="Taxon description",
                    placeholder="e.g. right femur, Tyrannosauridae, Maastrichtian",
                    lines=3,
                    elem_id="taxon-input",
                )
            with gr.Column(scale=1, min_width=280):
                gr.HTML("""<div class="relic-analysis-label">Reference image (optional)</div>""")
                image_input = gr.Image(
                    label="Reference bone image",
                    type="numpy",
                    sources=["upload"],
                    elem_id="image-input",
                )
            with gr.Column(scale=1, min_width=260):
                gr.HTML("""<div class="relic-analysis-label">Inference parameters</div>""")
                n_completions = gr.Slider(
                    minimum=1, maximum=5, step=1, value=5,
                    label="Number of completions",
                    elem_id="n-completions",
                )
                guidance_scale = gr.Slider(
                    minimum=1.0, maximum=10.0, step=0.5, value=3.0,
                    label="Guidance scale (CFG)",
                    elem_id="guidance-scale",
                )
                run_btn = gr.Button(
                    "▶  Run RELIC",
                    variant="primary",
                    elem_id="run-btn",
                    size="lg",
                )

        # ── Section 02 · Completions ───────────────────────────────
        with gr.Row(elem_classes=["relic-section"]):
            gr.HTML(_sec("02", "Reconstructed Completions", "top-5 diverse samples · DDIM"))

        with gr.Row(equal_height=True, elem_classes=["relic-section"]):
            comp_1 = gr.Model3D(
                label="Specimen I",
                elem_id="comp-1",
                height=280,
            )
            comp_2 = gr.Model3D(
                label="Specimen II",
                elem_id="comp-2",
                height=280,
            )
            comp_3 = gr.Model3D(
                label="Specimen III",
                elem_id="comp-3",
                height=280,
            )
            comp_4 = gr.Model3D(
                label="Specimen IV",
                elem_id="comp-4",
                height=280,
            )
            comp_5 = gr.Model3D(
                label="Specimen V · Most diverse",
                elem_id="comp-5",
                height=280,
            )

        # ── Section 03 · Analysis ──────────────────────────────────
        with gr.Row(elem_classes=["relic-section"]):
            gr.HTML(_sec("03", "Uncertainty & Symmetry Analysis", "per-point std · RANSAC plane"))

        with gr.Row(equal_height=False, elem_classes=["relic-section"]):
            with gr.Column(scale=3):
                gr.HTML("""<div class="relic-analysis-label">Uncertainty heatmap · plasma colormap</div>""")
                unc_display = gr.Image(
                    label="Per-point uncertainty (std across samples)",
                    type="filepath",
                    elem_id="unc-display",
                    height=340,
                )
            with gr.Column(scale=2):
                gr.HTML("""<div class="relic-analysis-label">Symmetry plane · RANSAC detection</div>""")
                sym_text = gr.Textbox(
                    label="Symmetry plane data",
                    lines=8,
                    interactive=False,
                    elem_id="sym-text",
                    placeholder="Confidence · Normal vector · Plane offset\n\nRun inference to populate...",
                )

        # ── Section 04 · Export ────────────────────────────────────
        with gr.Row(elem_classes=["relic-section"]):
            gr.HTML(_sec("04", "Export", "download selected completion"))

        with gr.Row(elem_classes=["relic-section"]):
            with gr.Column(scale=2):
                dl_choice = gr.Dropdown(
                    choices=["Completion 1", "Completion 2", "Completion 3", "Completion 4", "Completion 5"],
                    value="Completion 1",
                    label="Select specimen",
                    elem_id="dl-choice",
                )
            with gr.Column(scale=1):
                dl_btn = gr.Button(
                    "⬇  Prepare .ply",
                    variant="secondary",
                    elem_id="dl-btn",
                )
            with gr.Column(scale=2):
                dl_file = gr.File(label="Download", interactive=False, elem_id="dl-file")

        # ── Section 05 · Expert Rating ─────────────────────────────
        with gr.Row(elem_classes=["relic-section"]):
            gr.HTML(_sec("05", "Expert Rating", "2AFC forced-choice · pre-registered OSF"))

        gr.HTML("""
        <div class="relic-rating-section">
          <div class="relic-rating-prompt">
            Is this RELIC reconstruction more anatomically plausible than the AdaPoinTr baseline?
          </div>
          <div class="relic-rating-meta">
            Forced-choice · ≥15 domain-matched raters · Krippendorff's α inter-rater reliability
          </div>
        """)

        with gr.Row(elem_classes=["relic-section"]):
            rating_notes = gr.Textbox(
                label="Qualitative notes (optional)",
                placeholder="e.g. proximal epiphysis morphology looks anatomically correct; distal condyle over-smoothed",
                lines=2,
                elem_id="rating-notes",
            )

        with gr.Row(elem_classes=["relic-section"]):
            btn_relic_better = gr.Button(
                "✓  RELIC reconstruction is more plausible",
                variant="secondary",
                elem_id="btn-relic-better",
            )
            btn_baseline_better = gr.Button(
                "✗  AdaPoinTr baseline is more plausible",
                variant="secondary",
                elem_id="btn-baseline-better",
            )

        with gr.Row(elem_classes=["relic-section"]):
            rating_status = gr.Textbox(
                label="Rating status",
                interactive=False,
                elem_id="rating-status",
            )

        gr.HTML("""</div>""")

        # ── Status bar ─────────────────────────────────────────────
        with gr.Row(elem_classes=["relic-section"]):
            status_box = gr.Textbox(
                label="System status",
                interactive=False,
                elem_id="status-box",
                placeholder="Awaiting input...",
            )

        # ── Footer ──────────────────────────────────────────────────
        gr.HTML(f"""
        <div class="relic-footer">
          <div class="relic-footer-left">
            RELIC · Reconstruction of Extinct Life via Informed Completion ·
            Checkpoint: {os.path.basename(checkpoint_path) or "no checkpoint · preview mode"}
          </div>
          <div class="relic-footer-right">
            AdaPoinTr · SeedFormer · DiffComplete · PoinTr — SOTA baselines
          </div>
        </div>
        <div class="relic-strata"></div>
        """)

        # Internal state: store completion paths
        _paths = gr.State([None] * 5)

        # ----------------------------------------------------------------
        # Callbacks
        # ----------------------------------------------------------------

        def _run(mesh_file, taxon_text, ref_image, n_compl, guidance):
            result = run_inference(mesh_file, taxon_text, ref_image, n_compl, guidance)
            paths = result[:5]
            unc_img = result[5]
            sym = result[6]
            status = result[7]
            return (*paths, unc_img, sym, status, paths)

        run_btn.click(
            fn=_run,
            inputs=[mesh_input, taxon_input, image_input, n_completions, guidance_scale],
            outputs=[comp_1, comp_2, comp_3, comp_4, comp_5, unc_display, sym_text, status_box, _paths],
        )

        def _download(choice, paths):
            idx_map = {
                "Completion 1": 0, "Completion 2": 1, "Completion 3": 2,
                "Completion 4": 3, "Completion 5": 4,
            }
            idx = idx_map.get(choice, 0)
            if paths and idx < len(paths) and paths[idx] is not None:
                return paths[idx]
            return None

        dl_btn.click(fn=_download, inputs=[dl_choice, _paths], outputs=dl_file)

        btn_relic_better.click(
            fn=lambda t, n: rate_better(t, n),
            inputs=[taxon_input, rating_notes],
            outputs=rating_status,
        )
        btn_baseline_better.click(
            fn=lambda t, n: rate_worse(t, n),
            inputs=[taxon_input, rating_notes],
            outputs=rating_status,
        )

    return demo, _theme, _RELIC_CSS


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="RELIC Expert Gradio Interface")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to RELIC checkpoint (omit to preview UI without a model)")
    parser.add_argument("--config", type=str, default=None, help="YAML config path")
    parser.add_argument("--db", type=str, default="feedback.db", help="SQLite DB for user ratings")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")
    args = parser.parse_args()

    demo, _theme, _css = build_app(args.checkpoint, args.config, args.db)
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        theme=_theme,
        css=_css,
    )


if __name__ == "__main__":
    main()
