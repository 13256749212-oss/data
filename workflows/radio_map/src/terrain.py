from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import trimesh
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import cKDTree


@dataclass
class TerrainModel:
    source_path: Path
    vertices: np.ndarray
    faces: np.ndarray
    bounds: np.ndarray
    _linear: LinearNDInterpolator
    _tree: cKDTree

    @classmethod
    def load(cls, path: Path) -> "TerrainModel":
        path = Path(path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"找不到ground.ply: {path}")
        mesh = trimesh.load_mesh(path, process=False)
        if not isinstance(mesh, trimesh.Trimesh):
            raise TypeError(f"ground.ply不是单一三角网格: {type(mesh)}")
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        if len(vertices) < 3 or len(faces) < 1:
            raise ValueError("ground.ply没有有效顶点或三角面")
        points = vertices[:, :2]
        linear = LinearNDInterpolator(points, vertices[:, 2], fill_value=np.nan)
        tree = cKDTree(points)
        return cls(
            source_path=path,
            vertices=vertices,
            faces=faces,
            bounds=np.asarray(mesh.bounds, dtype=float),
            _linear=linear,
            _tree=tree,
        )

    def query(self, x: np.ndarray | float, y: np.ndarray | float) -> np.ndarray:
        x_arr, y_arr = np.broadcast_arrays(
            np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
        )
        shape = x_arr.shape
        points = np.column_stack([x_arr.ravel(), y_arr.ravel()])
        z = np.asarray(self._linear(points), dtype=np.float64).reshape(-1)
        missing = ~np.isfinite(z)
        if np.any(missing):
            _, idx = self._tree.query(points[missing], k=1)
            z[missing] = self.vertices[np.asarray(idx, dtype=int), 2]
        return z.reshape(shape)

    def assert_map_inside(self, center_x: float, center_y: float, size_x: float, size_y: float) -> None:
        x0, y0 = center_x - size_x / 2.0, center_y - size_y / 2.0
        x1, y1 = x0 + size_x, y0 + size_y
        bx0, by0 = self.bounds[0, :2]
        bx1, by1 = self.bounds[1, :2]
        if x0 < bx0 or y0 < by0 or x1 > bx1 or y1 > by1:
            raise ValueError(
                f"512m地图超出ground.ply范围: map=({x0:.2f},{x1:.2f},{y0:.2f},{y1:.2f}), "
                f"ground=({bx0:.2f},{bx1:.2f},{by0:.2f},{by1:.2f})"
            )


@dataclass(frozen=True)
class SurfaceInfo:
    path: Path
    n_cells: int
    n_faces: int
    cell_ix: np.ndarray
    cell_iy: np.ndarray
    cell_center_x: np.ndarray
    cell_center_y: np.ndarray
    cell_ground_z: np.ndarray
    cell_rx_z: np.ndarray
    nx: int | None = None
    ny: int | None = None


def _export_mesh(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False, validate=False)
    # Explicitly ensure upward-facing normals for the measurement surface.
    if len(mesh.face_normals) and np.nanmedian(mesh.face_normals[:, 2]) < 0:
        mesh.faces = mesh.faces[:, ::-1]
    mesh.export(path, file_type="ply", encoding="binary")


def build_sparse_measurement_surface(
    terrain: TerrainModel,
    cells: Iterable[tuple[int, int]],
    x_min: float,
    y_min: float,
    cell_size_m: float,
    rx_height_agl_m: float,
    output_path: Path,
) -> SurfaceInfo:
    ordered = sorted(set((int(ix), int(iy)) for ix, iy in cells), key=lambda p: (p[1], p[0]))
    if not ordered:
        raise ValueError("稀疏测量面没有cell")
    n = len(ordered)
    vertices = np.empty((n * 4, 3), dtype=np.float32)
    faces = np.empty((n * 2, 3), dtype=np.int32)
    cell_ix = np.asarray([p[0] for p in ordered], dtype=np.int32)
    cell_iy = np.asarray([p[1] for p in ordered], dtype=np.int32)

    x0 = x_min + cell_ix.astype(float) * cell_size_m
    y0 = y_min + cell_iy.astype(float) * cell_size_m
    x1 = x0 + cell_size_m
    y1 = y0 + cell_size_m
    corners_x = np.column_stack([x0, x1, x1, x0])
    corners_y = np.column_stack([y0, y0, y1, y1])
    corners_z = terrain.query(corners_x, corners_y) + float(rx_height_agl_m)

    vertices[:, 0] = corners_x.reshape(-1)
    vertices[:, 1] = corners_y.reshape(-1)
    vertices[:, 2] = corners_z.reshape(-1)
    base = np.arange(n, dtype=np.int32) * 4
    faces[0::2] = np.column_stack([base, base + 1, base + 2])
    faces[1::2] = np.column_stack([base, base + 2, base + 3])
    _export_mesh(output_path, vertices, faces)

    center_x = x0 + 0.5 * cell_size_m
    center_y = y0 + 0.5 * cell_size_m
    ground = terrain.query(center_x, center_y)
    return SurfaceInfo(
        path=Path(output_path).resolve(),
        n_cells=n,
        n_faces=2 * n,
        cell_ix=cell_ix,
        cell_iy=cell_iy,
        cell_center_x=center_x,
        cell_center_y=center_y,
        cell_ground_z=ground,
        cell_rx_z=ground + float(rx_height_agl_m),
    )


def build_dense_measurement_surface(
    terrain: TerrainModel,
    center_x: float,
    center_y: float,
    size_x_m: int,
    size_y_m: int,
    cell_size_m: float,
    rx_height_agl_m: float,
    output_path: Path,
) -> SurfaceInfo:
    nx = int(round(size_x_m / cell_size_m))
    ny = int(round(size_y_m / cell_size_m))
    if abs(nx * cell_size_m - size_x_m) > 1e-6 or abs(ny * cell_size_m - size_y_m) > 1e-6:
        raise ValueError("地图尺寸必须能被cell_size整除")
    x_min = center_x - size_x_m / 2.0
    y_min = center_y - size_y_m / 2.0
    x_edges = x_min + np.arange(nx + 1, dtype=np.float64) * cell_size_m
    y_edges = y_min + np.arange(ny + 1, dtype=np.float64) * cell_size_m
    xx, yy = np.meshgrid(x_edges, y_edges)
    zz = terrain.query(xx, yy) + float(rx_height_agl_m)
    vertices = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()]).astype(np.float32)

    row = np.arange(ny, dtype=np.int64)[:, None]
    col = np.arange(nx, dtype=np.int64)[None, :]
    v00 = row * (nx + 1) + col
    v10 = v00 + 1
    v01 = v00 + (nx + 1)
    v11 = v01 + 1
    n_cells = nx * ny
    faces = np.empty((n_cells * 2, 3), dtype=np.int32)
    faces[0::2] = np.column_stack([v00.ravel(), v10.ravel(), v11.ravel()])
    faces[1::2] = np.column_stack([v00.ravel(), v11.ravel(), v01.ravel()])
    _export_mesh(output_path, vertices, faces)

    cx = x_min + (np.arange(nx, dtype=float) + 0.5) * cell_size_m
    cy = y_min + (np.arange(ny, dtype=float) + 0.5) * cell_size_m
    cxx, cyy = np.meshgrid(cx, cy)
    ground = terrain.query(cxx, cyy)
    iy, ix = np.indices((ny, nx))
    return SurfaceInfo(
        path=Path(output_path).resolve(),
        n_cells=n_cells,
        n_faces=2 * n_cells,
        cell_ix=ix.ravel().astype(np.int32),
        cell_iy=iy.ravel().astype(np.int32),
        cell_center_x=cxx.ravel(),
        cell_center_y=cyy.ravel(),
        cell_ground_z=ground.ravel(),
        cell_rx_z=(ground + float(rx_height_agl_m)).ravel(),
        nx=nx,
        ny=ny,
    )


def inspect_mesh(path: Path) -> dict:
    path = Path(path).expanduser().resolve()
    info: dict = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        return info
    info["size_bytes"] = path.stat().st_size
    try:
        mesh = trimesh.load_mesh(path, process=False)
        if isinstance(mesh, trimesh.Trimesh):
            info.update(
                {
                    "vertices": int(len(mesh.vertices)),
                    "faces": int(len(mesh.faces)),
                    "empty": bool(len(mesh.vertices) == 0 or len(mesh.faces) == 0),
                    "bounds": None if mesh.bounds is None else np.asarray(mesh.bounds).tolist(),
                }
            )
        else:
            info.update({"type": type(mesh).__name__, "empty": True})
    except Exception as exc:  # pragma: no cover - diagnostic path
        info.update({"error": f"{type(exc).__name__}: {exc}", "empty": True})
    return info
