"""
Visualization utilities for RELIC.

render_completion          — plotly render of partial / predicted / GT
uncertainty_heatmap        — color point cloud by per-point std
tsne_latent_space          — t-SNE plot of z_global embeddings
symmetry_plane_overlay     — visualize symmetry plane on point cloud
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch import Tensor

logger = logging.getLogger(__name__)


def _to_np(x) -> np.ndarray:
    if isinstance(x, Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float32)


# ---------------------------------------------------------------------------
# Render completion
# ---------------------------------------------------------------------------

def render_completion(
    partial: np.ndarray | Tensor,
    predicted: np.ndarray | Tensor,
    gt: Optional[np.ndarray | Tensor] = None,
    save_path: Optional[str | Path] = None,
    window_name: str = "RELIC Completion",
) -> None:
    """
    Render partial (blue), predicted (green), and optional GT (red) side-by-side
    using Plotly.

    Parameters
    ----------
    partial   : [N, 3]
    predicted : [M, 3]
    gt        : [M, 3] or None
    save_path : path to save as .html or .png (requires kaleido for .png)
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        logger.warning("plotly not installed. Skipping render_completion.")
        return

    p_np = _to_np(partial)
    pred_np = _to_np(predicted)

    x_offset = max(p_np[:, 0].max() - p_np[:, 0].min(), 1.0) * 1.5

    traces = [
        go.Scatter3d(
            x=p_np[:, 0], y=p_np[:, 1], z=p_np[:, 2],
            mode="markers",
            marker=dict(size=2, color="royalblue", opacity=0.8),
            name="Partial input",
        ),
        go.Scatter3d(
            x=pred_np[:, 0] + x_offset, y=pred_np[:, 1], z=pred_np[:, 2],
            mode="markers",
            marker=dict(size=2, color="limegreen", opacity=0.8),
            name="Prediction",
        ),
    ]

    if gt is not None:
        gt_np = _to_np(gt)
        traces.append(go.Scatter3d(
            x=gt_np[:, 0] + x_offset * 2, y=gt_np[:, 1], z=gt_np[:, 2],
            mode="markers",
            marker=dict(size=2, color="tomato", opacity=0.8),
            name="Ground truth",
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=window_name,
        scene=dict(aspectmode="data"),
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(x=0, y=1),
    )

    if save_path is not None:
        save_path = Path(save_path)
        if save_path.suffix == ".html":
            fig.write_html(str(save_path))
        else:
            fig.write_image(str(save_path))  # requires kaleido
        logger.info("Saved render to %s", save_path)
    else:
        fig.show()


# ---------------------------------------------------------------------------
# Uncertainty heatmap
# ---------------------------------------------------------------------------

def uncertainty_heatmap(
    points: np.ndarray | Tensor,
    uncertainty: np.ndarray | Tensor,
    save_path: Optional[str | Path] = None,
    colormap: str = "plasma",
) -> None:
    """
    Render a point cloud colored by per-point uncertainty (std across samples).

    High uncertainty → warm colors; low uncertainty → cool colors.

    Parameters
    ----------
    points      : [N, 3]
    uncertainty : [N]   per-point std values
    save_path   : .html (interactive) or .png (static, requires kaleido)
    colormap    : plotly colorscale name (e.g. "plasma", "viridis")
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        logger.warning("plotly not installed. Skipping uncertainty_heatmap.")
        return

    pts = _to_np(points)
    unc = _to_np(uncertainty).astype(np.float32)

    u_min, u_max = float(unc.min()), float(unc.max())
    unc_norm = (unc - u_min) / (u_max - u_min + 1e-12)

    fig = go.Figure(data=go.Scatter3d(
        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
        mode="markers",
        marker=dict(
            size=2,
            color=unc_norm,
            colorscale=colormap,
            showscale=True,
            colorbar=dict(title="Uncertainty (std)"),
            opacity=0.85,
        ),
    ))
    fig.update_layout(
        title="Per-point uncertainty heatmap",
        scene=dict(aspectmode="data"),
        margin=dict(l=0, r=0, b=0, t=40),
    )

    if save_path is not None:
        save_path = Path(save_path)
        if save_path.suffix == ".html":
            fig.write_html(str(save_path))
        else:
            # Fall back to matplotlib static image when kaleido not available
            try:
                fig.write_image(str(save_path))
            except Exception:
                import matplotlib.pyplot as plt
                fig2, ax = plt.subplots(figsize=(8, 6))
                ax2 = fig2.add_subplot(111, projection="3d") if hasattr(ax, "get_zlim") else plt.figure().add_subplot(111, projection="3d")
                sc = ax2.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=unc_norm, cmap=colormap, s=2, alpha=0.7)
                plt.colorbar(sc, ax=ax2, label="Uncertainty (std)")
                ax2.set_title("Per-point uncertainty heatmap")
                ax2.set_axis_off()
                plt.tight_layout()
                plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
                plt.close()
        logger.info("Saved uncertainty heatmap to %s", save_path)
    else:
        fig.show()


# ---------------------------------------------------------------------------
# t-SNE latent space visualization
# ---------------------------------------------------------------------------

def tsne_latent_space(
    z_globals: np.ndarray | Tensor,
    labels: List[str] | np.ndarray,
    label_type: str = "taxon",
    save_path: Optional[str | Path] = None,
    perplexity: float = 30.0,
    n_iter: int = 1000,
) -> None:
    """
    Create a t-SNE plot of z_global embeddings colored by label.

    Parameters
    ----------
    z_globals  : [N, D]   latent vectors
    labels     : [N]      string labels (taxon names, bone types, etc.)
    label_type : str      label description for plot title
    save_path  : .html or image path
    """
    try:
        from sklearn.manifold import TSNE
        from sklearn.preprocessing import LabelEncoder
        import plotly.express as px
        import pandas as pd
    except ImportError:
        logger.warning("sklearn or plotly not installed. Skipping t-SNE.")
        return

    z = _to_np(z_globals)
    labels_arr = np.asarray(labels)
    n = z.shape[0]

    if n < 5:
        logger.warning("Too few samples (%d) for t-SNE.", n)
        return

    perp = min(perplexity, (n - 1) / 3)
    tsne = TSNE(n_components=2, perplexity=perp, n_iter=n_iter, random_state=42)
    z2d = tsne.fit_transform(z)

    df = pd.DataFrame({"x": z2d[:, 0], "y": z2d[:, 1], label_type: labels_arr})
    fig = px.scatter(
        df, x="x", y="y", color=label_type,
        title=f"t-SNE of z_global (colored by {label_type})",
        labels={"x": "t-SNE 1", "y": "t-SNE 2"},
        opacity=0.8,
    )
    fig.update_traces(marker=dict(size=6))
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=40))

    if save_path is not None:
        save_path = Path(save_path)
        if save_path.suffix == ".html":
            fig.write_html(str(save_path))
        else:
            try:
                fig.write_image(str(save_path))
            except Exception:
                import matplotlib.pyplot as plt
                le = LabelEncoder()
                label_ids = le.fit_transform(labels_arr)
                fig2, ax = plt.subplots(figsize=(10, 8))
                scatter = ax.scatter(z2d[:, 0], z2d[:, 1], c=label_ids, cmap="tab20", s=20, alpha=0.8)
                ax.set_title(f"t-SNE of z_global (colored by {label_type})")
                ax.set_xlabel("t-SNE 1")
                ax.set_ylabel("t-SNE 2")
                plt.tight_layout()
                plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
                plt.close()
        logger.info("Saved t-SNE plot to %s", save_path)
    else:
        fig.show()


# ---------------------------------------------------------------------------
# Symmetry plane overlay
# ---------------------------------------------------------------------------

def symmetry_plane_overlay(
    points: np.ndarray | Tensor,
    plane_normal: np.ndarray | Tensor,
    plane_offset: float,
    confidence: float,
    save_path: Optional[str | Path] = None,
) -> None:
    """
    Visualize the detected symmetry plane overlaid on the point cloud using Plotly.

    Parameters
    ----------
    points       : [N, 3]
    plane_normal : [3]
    plane_offset : float
    confidence   : float in [0, 1]
    save_path    : .html or image path (optional)
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        logger.warning("plotly not installed. Skipping symmetry_plane_overlay.")
        return

    pts = _to_np(points)
    n = _to_np(plane_normal).astype(np.float64)
    n = n / (np.linalg.norm(n) + 1e-12)

    # Color points by signed distance to plane
    signed_dists = pts.astype(np.float64) @ n - plane_offset
    d_min, d_max = signed_dists.min(), signed_dists.max()
    dist_norm = (signed_dists - d_min) / (d_max - d_min + 1e-12)

    pcd_trace = go.Scatter3d(
        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
        mode="markers",
        marker=dict(
            size=2,
            color=dist_norm,
            colorscale="RdBu",
            showscale=True,
            colorbar=dict(title="Signed dist to plane"),
            opacity=0.8,
        ),
        name="Point cloud",
    )

    # Build a square plane mesh centred at the cloud centroid
    centroid = pts.mean(axis=0)
    proj = centroid - (np.dot(centroid, n) - plane_offset) * n
    plane_scale = float(np.linalg.norm(pts - centroid, axis=1).max()) * 1.2
    arb = np.array([1, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1, 0])
    u = np.cross(n, arb); u /= np.linalg.norm(u) + 1e-12
    v = np.cross(n, u);   v /= np.linalg.norm(v) + 1e-12
    corners = np.array([
        proj + plane_scale * ( u + v),
        proj + plane_scale * ( u - v),
        proj + plane_scale * (-u - v),
        proj + plane_scale * (-u + v),
    ])
    # Two triangles forming the square
    i_idx = [0, 0]
    j_idx = [1, 2]
    k_idx = [2, 3]
    plane_trace = go.Mesh3d(
        x=corners[:, 0], y=corners[:, 1], z=corners[:, 2],
        i=i_idx, j=j_idx, k=k_idx,
        color="gold",
        opacity=max(0.15, confidence * 0.5),
        name=f"Symmetry plane (conf={confidence:.2f})",
    )

    fig = go.Figure(data=[pcd_trace, plane_trace])
    fig.update_layout(
        title=f"Symmetry plane — confidence {confidence:.3f}",
        scene=dict(aspectmode="data"),
        margin=dict(l=0, r=0, b=0, t=40),
    )

    if save_path is not None:
        save_path = Path(save_path)
        if save_path.suffix == ".html":
            fig.write_html(str(save_path))
        else:
            try:
                fig.write_image(str(save_path))
            except Exception:
                logger.warning("Could not save image (install kaleido for PNG export). Saving HTML instead.")
                fig.write_html(str(save_path.with_suffix(".html")))
        logger.info("Saved symmetry overlay to %s", save_path)
    else:
        fig.show()
