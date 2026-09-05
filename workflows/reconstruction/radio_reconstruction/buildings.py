from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import trimesh


def collect_building_paths(assets_dir: Path, explicit: Sequence[str] | None = None) -> list[Path]:
    assets_dir = Path(assets_dir).expanduser().resolve()
    if explicit:
        raw = [Path(value).expanduser().resolve() for value in explicit]
    else:
        raw = sorted(assets_dir.glob("ynu_chenggong_campus*.ply"))

    result: list[Path] = []
    seen: set[str] = set()
    for path in raw:
        key = str(path).casefold()
        if key in seen:
            continue
        seen.add(key)
        if path.exists() and path.is_file() and path.stat().st_size > 0:
            result.append(path.resolve())
    return result


def _iter_meshes(path: Path):
    loaded = trimesh.load(str(path), process=False)
    if isinstance(loaded, trimesh.Trimesh):
        yield loaded
    elif isinstance(loaded, trimesh.Scene):
        for geometry in loaded.geometry.values():
            if isinstance(geometry, trimesh.Trimesh):
                yield geometry


def load_local_xy_triangles(
    building_paths: Sequence[Path],
    extent: Sequence[float],
    coordinate_margin_m: float = 100.0,
) -> tuple[np.ndarray, list[dict]]:
    x_min, x_max, y_min, y_max = map(float, extent)
    triangles: list[np.ndarray] = []
    diagnostics: list[dict] = []

    for path in building_paths:
        file_triangles: list[np.ndarray] = []
        raw_count = 0
        try:
            for mesh in _iter_meshes(Path(path)):
                vertices = np.asarray(mesh.vertices, dtype=float)
                faces = np.asarray(mesh.faces, dtype=np.int64)
                if len(vertices) == 0 or len(faces) == 0:
                    continue
                tri = vertices[faces][:, :, :2]
                raw_count += len(tri)
                # Keep every triangle whose XY bounding box overlaps the current
                # 512 m map window.  Using triangle centers alone can incorrectly
                # drop large roof/facade triangles that cross the map boundary.
                tri_min = np.min(tri, axis=1)
                tri_max = np.max(tri, axis=1)
                local = (
                    (tri_max[:, 0] >= x_min - coordinate_margin_m)
                    & (tri_min[:, 0] <= x_max + coordinate_margin_m)
                    & (tri_max[:, 1] >= y_min - coordinate_margin_m)
                    & (tri_min[:, 1] <= y_max + coordinate_margin_m)
                )
                tri = tri[local]
                if len(tri):
                    area2 = np.abs(
                        np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
                    )
                    tri = tri[area2 > 1e-6]
                if len(tri):
                    file_triangles.append(tri)
            kept = int(sum(len(v) for v in file_triangles))
            if file_triangles:
                triangles.extend(file_triangles)
            diagnostics.append({
                "path": str(Path(path).resolve()),
                "raw_xy_triangle_count": int(raw_count),
                "local_nondegenerate_triangle_count": kept,
                "status": "used" if kept else "no_local_nondegenerate_xy_triangle",
            })
        except Exception as exc:
            diagnostics.append({
                "path": str(Path(path).resolve()),
                "raw_xy_triangle_count": int(raw_count),
                "local_nondegenerate_triangle_count": 0,
                "status": f"load_failed:{type(exc).__name__}:{exc}",
            })

    if not triangles:
        return np.empty((0, 3, 2), dtype=np.float64), diagnostics
    return np.concatenate(triangles, axis=0).astype(np.float64), diagnostics


def rasterize_building_mask(
    triangles_xy: np.ndarray,
    x_centers: np.ndarray,
    y_centers: np.ndarray,
    buffer_cells: int = 1,
) -> np.ndarray:
    ny, nx = len(y_centers), len(x_centers)
    mask = np.zeros((ny, nx), dtype=bool)
    if len(triangles_xy) == 0:
        return mask

    from matplotlib.path import Path as MplPath

    dx = float(np.median(np.diff(x_centers))) if nx > 1 else 1.0
    dy = float(np.median(np.diff(y_centers))) if ny > 1 else 1.0
    x0 = float(x_centers[0])
    y0 = float(y_centers[0])

    for triangle in triangles_xy:
        min_x, min_y = np.min(triangle, axis=0)
        max_x, max_y = np.max(triangle, axis=0)
        ix0 = max(0, int(np.floor((min_x - x0) / dx)) - 1)
        ix1 = min(nx - 1, int(np.ceil((max_x - x0) / dx)) + 1)
        iy0 = max(0, int(np.floor((min_y - y0) / dy)) - 1)
        iy1 = min(ny - 1, int(np.ceil((max_y - y0) / dy)) + 1)
        if ix1 < ix0 or iy1 < iy0:
            continue
        xx, yy = np.meshgrid(x_centers[ix0:ix1 + 1], y_centers[iy0:iy1 + 1])
        points = np.column_stack([xx.ravel(), yy.ravel()])
        inside = MplPath(triangle, closed=True).contains_points(points, radius=1e-9)
        if np.any(inside):
            local = inside.reshape(yy.shape)
            mask[iy0:iy1 + 1, ix0:ix1 + 1] |= local

    if int(buffer_cells) > 0 and np.any(mask):
        from scipy.ndimage import binary_dilation
        mask = binary_dilation(mask, iterations=int(buffer_cells))
    return mask


def building_outline_segments(triangles_xy: np.ndarray) -> list[np.ndarray]:
    if len(triangles_xy) == 0:
        return []
    try:
        from shapely.geometry import Polygon
        from shapely.ops import unary_union

        polygons = []
        for triangle in triangles_xy:
            polygon = Polygon(triangle)
            if polygon.is_valid and polygon.area > 1e-6:
                polygons.append(polygon)
        if polygons:
            geometry = unary_union(polygons)
            segments: list[np.ndarray] = []
            geometries = [geometry] if geometry.geom_type == "Polygon" else list(geometry.geoms)
            for polygon in geometries:
                segments.append(np.asarray(polygon.exterior.coords, dtype=float)[:, :2])
                for interior in polygon.interiors:
                    segments.append(np.asarray(interior.coords, dtype=float)[:, :2])
            return segments
    except Exception:
        pass

    # fallback：三角形边，图会更密但不会丢建筑。
    segments = []
    for triangle in triangles_xy:
        segments.extend([
            np.asarray([triangle[0], triangle[1]], dtype=float),
            np.asarray([triangle[1], triangle[2]], dtype=float),
            np.asarray([triangle[2], triangle[0]], dtype=float),
        ])
    return segments
