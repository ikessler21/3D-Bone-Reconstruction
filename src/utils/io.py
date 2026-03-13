"""
I/O utilities for loading and saving meshes, NIfTI volumes, and processed point clouds.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import trimesh


# ---------------------------------------------------------------------------
# Mesh loading
# ---------------------------------------------------------------------------

def load_mesh(path: str | Path) -> trimesh.Trimesh:
    """
    Load a mesh from .ply, .obj, .stl, or .vtk format.

    Returns a trimesh.Trimesh. Raises ValueError for unsupported formats.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in {".ply", ".obj", ".stl", ".vtk", ".off", ".gltf", ".glb"}:
        raise ValueError(f"Unsupported mesh format: {suffix}")

    if suffix == ".vtk":
        # trimesh does not natively support VTK; try pyvista as fallback
        try:
            import pyvista as pv
            pv_mesh = pv.read(str(path))
            # Extract surface
            surf = pv_mesh.extract_surface()
            vertices = np.array(surf.points, dtype=np.float64)
            faces_raw = surf.faces.reshape(-1, 4)[:, 1:]  # vtk format: [n, v0, v1, v2]
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces_raw, process=False)
        except ImportError:
            raise ImportError("pyvista is required for .vtk files: pip install pyvista")
    else:
        loaded = trimesh.load(str(path), force="mesh", process=False)
        if isinstance(loaded, trimesh.Scene):
            # Merge all geometries in the scene
            meshes = list(loaded.geometry.values())
            if not meshes:
                raise ValueError(f"No mesh geometry found in {path}")
            mesh = trimesh.util.concatenate(meshes)
        else:
            mesh = loaded

    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Could not load {path} as a Trimesh object")
    return mesh


# ---------------------------------------------------------------------------
# NIfTI loading and conversion
# ---------------------------------------------------------------------------

def load_nifti(path: str | Path) -> np.ndarray:
    """
    Load a NIfTI file (.nii or .nii.gz) as a float32 NumPy array.
    """
    import nibabel as nib
    img = nib.load(str(path))
    data = np.array(img.get_fdata(), dtype=np.float32)
    return data


def nifti_to_mesh(
    volume: np.ndarray,
    threshold: float = 0.5,
    step_size: int = 2,
) -> trimesh.Trimesh:
    """
    Extract an isosurface mesh from a 3D binary/scalar volume using marching cubes.

    Parameters
    ----------
    volume    : np.ndarray  [D, H, W]
    threshold : float       isosurface threshold
    step_size : int         marching cubes step size (higher = coarser / faster)

    Returns
    -------
    trimesh.Trimesh
    """
    from skimage.measure import marching_cubes
    verts, faces, normals, _ = marching_cubes(volume, level=threshold, step_size=step_size)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals, process=False)
    return mesh


# ---------------------------------------------------------------------------
# Point cloud saving
# ---------------------------------------------------------------------------

def save_point_cloud(
    points: "np.ndarray | torch.Tensor",
    path: str | Path,
    normals: "np.ndarray | torch.Tensor | None" = None,
) -> None:
    """
    Save a point cloud to a .ply file.

    Parameters
    ----------
    points  : array-like [N, 3]
    path    : output .ply path
    normals : optional [N, 3] normals
    """
    if hasattr(points, "numpy"):
        pts_np = points.detach().cpu().numpy().astype(np.float64)
    else:
        pts_np = np.asarray(points, dtype=np.float64)

    Path(path).parent.mkdir(parents=True, exist_ok=True)

    try:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_np)
        if normals is not None:
            n_np = normals.detach().cpu().numpy() if hasattr(normals, "numpy") else np.asarray(normals, dtype=np.float64)
            pcd.normals = o3d.utility.Vector3dVector(n_np.astype(np.float64))
        o3d.io.write_point_cloud(str(path), pcd, write_ascii=False)
    except ImportError:
        # Fallback: save via trimesh
        cloud = trimesh.points.PointCloud(pts_np)
        cloud.export(str(path))


# ---------------------------------------------------------------------------
# Processed .pt file I/O
# ---------------------------------------------------------------------------

def save_processed(data_dict: Dict[str, Any], path: str | Path) -> None:
    """
    Save a processed data dict (partial, full, metadata) as a .pt file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data_dict, str(path))


def load_processed(path: str | Path) -> Dict[str, Any]:
    """
    Load a processed .pt file and return the dict.
    """
    return torch.load(str(path), map_location="cpu", weights_only=False)
