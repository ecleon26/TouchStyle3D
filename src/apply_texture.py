"""Apply a generated heightfield to a 3D model surface.

Implements the texture-application step from Section 5.5 of the paper:
vertices are displaced along their normals according to the heightmap value
sampled at their UV coordinates.

Example::

    python -m src.apply_texture \\
        --mesh    examples/airpods/airpods.obj \\
        --height  examples/airpods/airpods_heightfield.png \\
        --output  examples/airpods/airpods_styled.obj \\
        --scale   0.05 \\
        --remesh
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image


def _generate_uv_coordinates(mesh: trimesh.Trimesh) -> np.ndarray:
    """Planar XY UV unwrap normalized to ``[0, 1]^2``."""
    min_b, max_b = mesh.bounds
    uv = (mesh.vertices - min_b) / (max_b - min_b)
    return uv[:, :2]


def apply_heightmap(
    mesh: trimesh.Trimesh, heightmap: np.ndarray, scale: float = 0.05
) -> trimesh.Trimesh:
    """Displace vertices along their normals using the heightmap.

    Parameters
    ----------
    mesh:
        Input ``trimesh.Trimesh``. Vertex normals are used; if the mesh does
        not provide UVs, planar XY coordinates are used as a fallback.
    heightmap:
        2-D numpy array of float values in ``[0, 1]``.
    scale:
        Multiplier for the displacement magnitude. ``0.05`` was used for the
        application examples in the paper.
    """
    uv = (
        mesh.visual.uv
        if hasattr(mesh.visual, "uv") and mesh.visual.uv is not None
        else _generate_uv_coordinates(mesh)
    )

    h, w = heightmap.shape
    normals = mesh.vertex_normals
    vertices = mesh.vertices.copy()

    px = np.clip((uv[:, 0] * (w - 1)).astype(int), 0, w - 1)
    py = np.clip(((1.0 - uv[:, 1]) * (h - 1)).astype(int), 0, h - 1)
    heights = heightmap[py, px]

    vertices += normals * heights[:, None] * scale
    return trimesh.Trimesh(vertices=vertices, faces=mesh.faces)


def remesh(mesh_path: str | Path, target_edge_pct: float = 0.5, iterations: int = 20) -> None:
    """Optional pre-processing step: isotropic remesh in-place via PyMeshLab.

    The paper uses a remeshed surface (~25k faces) to give the heightfield
    enough vertices to displace.
    """
    import pymeshlab

    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(mesh_path))
    ms.meshing_isotropic_explicit_remeshing(
        iterations=iterations,
        targetlen=pymeshlab.PercentageValue(target_edge_pct),
    )
    ms.save_current_mesh(str(mesh_path))


def stylize(
    mesh_path: str | Path,
    heightmap_path: str | Path,
    output_path: str | Path,
    scale: float = 0.05,
    do_remesh: bool = False,
) -> None:
    if do_remesh:
        remesh(mesh_path)

    mesh = trimesh.load(str(mesh_path), force="mesh")
    height_image = Image.open(str(heightmap_path)).convert("L")
    heightmap = np.asarray(height_image, dtype=np.float32) / 255.0
    styled = apply_heightmap(mesh, heightmap, scale=scale)
    styled.export(str(output_path))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stylize a 3D model with a generated heightfield.")
    p.add_argument("--mesh", required=True)
    p.add_argument("--height", required=True, help="Grayscale heightfield image.")
    p.add_argument("--output", required=True)
    p.add_argument("--scale", type=float, default=0.05, help="Texture magnification factor (paper default: 0.05).")
    p.add_argument("--remesh", action="store_true", help="Run isotropic remeshing before displacement.")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    stylize(args.mesh, args.height, args.output, scale=args.scale, do_remesh=args.remesh)
    print(f"Wrote stylized mesh to {args.output}")
