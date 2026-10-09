#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Smooth campus outer-boundary extraction from aligned drive-test measurements.

The publication joint-map figures use a smooth outer envelope inferred only from the
coordinate-aligned drive-test trajectories.  Radio measurements are not used.

Unlike a convex hull, which connects a few extreme samples with long straight chords,
this implementation rasterizes the aligned trajectories, constructs the smallest
connected buffered envelope that contains the main measurement component, fills its
interior, smooths the envelope, and extracts a dense closed contour.  The resulting
red overlay follows the curved campus footprint much more naturally while remaining
fully reproducible from the released aligned X/Y coordinates.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.ndimage import binary_fill_holes, distance_transform_edt, gaussian_filter, label


@dataclass(frozen=True)
class CampusOutline:
    xy: np.ndarray
    source_files: tuple[str, ...]
    source_point_count: int
    hull_vertex_count: int
    method: str = "smoothed_trajectory_envelope"
    grid_m: float = 5.0
    buffer_m: float = 40.0
    smooth_m: float = 18.0
    covered_fraction: float = 1.0
    straight_segment_applied: bool = False
    straight_segment_half_span_m: float = 0.0
    straight_segment_target_x_m: float = 1447.05
    straight_segment_target_y_m: float = 384.09


def _read_csv_robust(path: Path) -> pd.DataFrame:
    last: Exception | None = None
    for enc in ("utf-8-sig", "utf-8", "gb18030", "latin1"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except Exception as exc:  # pragma: no cover - encoding fallback
            last = exc
    if last is not None:
        raise last
    raise RuntimeError(f"Unable to read CSV: {path}")


def _coordinate_columns(columns: Iterable[str]) -> tuple[str, str]:
    norm = {str(c).replace("\ufeff", "").strip().lower(): str(c) for c in columns}
    pairs = (
        ("blender_x", "blender_y"),
        ("blender_x_m", "blender_y_m"),
        ("x_m", "y_m"),
        ("x", "y"),
    )
    for x_name, y_name in pairs:
        if x_name in norm and y_name in norm:
            return norm[x_name], norm[y_name]
    raise KeyError("Aligned measurement CSV does not contain Blender X/Y coordinate columns")


def _resolve_measurement_files(source: Path) -> list[Path]:
    source = Path(source).expanduser().resolve()
    if source.is_file():
        return [source]
    if not source.exists() or not source.is_dir():
        return []

    # Prefer the concatenated alignment product to avoid reading duplicated samples.
    combined = source / "all_measurements_blender_xyz.csv"
    if combined.exists() and combined.is_file():
        return [combined]

    return sorted(p for p in source.glob("*_with_blender_xyz.csv") if p.is_file())


def _load_aligned_xy(
    source: Path,
    extent: tuple[float, float, float, float] | None,
) -> tuple[np.ndarray, tuple[str, ...]]:
    files = _resolve_measurement_files(Path(source))
    if not files:
        raise FileNotFoundError(
            f"No aligned measurement CSVs found under: {Path(source).expanduser()}"
        )

    chunks: list[np.ndarray] = []
    used_files: list[str] = []
    for path in files:
        df = _read_csv_robust(path)
        x_col, y_col = _coordinate_columns(df.columns)
        x = pd.to_numeric(df[x_col], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(df[y_col], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(x) & np.isfinite(y)
        if "dem_hit" in df.columns:
            dem_hit = pd.to_numeric(df["dem_hit"], errors="coerce").to_numpy(dtype=float)
            valid &= np.isfinite(dem_hit) & (dem_hit > 0)
        if extent is not None:
            xmin, xmax, ymin, ymax = map(float, extent)
            valid &= (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)
        if np.any(valid):
            chunks.append(np.column_stack([x[valid], y[valid]]))
            used_files.append(str(path))

    if not chunks:
        raise ValueError("Aligned measurement files contain no valid Blender X/Y samples")

    xy = np.vstack(chunks)
    xy = np.unique(np.round(xy, decimals=2), axis=0)
    if xy.shape[0] < 3:
        raise ValueError(f"At least three distinct aligned points are required; got {xy.shape[0]}")
    return xy, tuple(used_files)


def _largest_component(mask: np.ndarray) -> tuple[np.ndarray, int]:
    labels, n = label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    if n <= 0:
        return np.zeros_like(mask, dtype=bool), 0
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    idx = int(np.argmax(counts))
    return labels == idx, idx


def _shoelace_area(xy: np.ndarray) -> float:
    arr = np.asarray(xy, dtype=float)
    if len(arr) < 3:
        return 0.0
    if np.allclose(arr[0], arr[-1]):
        arr = arr[:-1]
    x = arr[:, 0]
    y = arr[:, 1]
    return 0.5 * float(np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def _extract_contour(
    field: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    *,
    level_value: float = 0.5,
) -> np.ndarray:
    # contourpy is installed with Matplotlib and avoids creating a temporary figure.
    try:
        from contourpy import contour_generator
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("contourpy is required for smooth campus-outline extraction") from exc

    cg = contour_generator(x=x_coords, y=y_coords, z=np.asarray(field, dtype=float))
    lines = cg.lines(float(level_value))
    candidates: list[np.ndarray] = []
    for line_xy in lines:
        arr = np.asarray(line_xy, dtype=float)
        if arr.ndim != 2 or arr.shape[0] < 4 or arr.shape[1] != 2:
            continue
        # Close the line if contourpy leaves the last point numerically distinct.
        if np.linalg.norm(arr[0] - arr[-1]) > max(1e-6, 0.25 * np.mean(np.diff(x_coords))):
            arr = np.vstack([arr, arr[0]])
        candidates.append(arr)
    if not candidates:
        raise ValueError("Unable to extract a closed campus outline from aligned trajectories")
    return max(candidates, key=_shoelace_area)


def _resample_closed_polyline(xy: np.ndarray, spacing_m: float = 4.0) -> np.ndarray:
    arr = np.asarray(xy, dtype=float)
    if np.linalg.norm(arr[0] - arr[-1]) > 1e-8:
        arr = np.vstack([arr, arr[0]])
    seg = np.linalg.norm(np.diff(arr, axis=0), axis=1)
    keep = np.r_[True, seg > 1e-9]
    arr = arr[keep]
    if np.linalg.norm(arr[0] - arr[-1]) > 1e-8:
        arr = np.vstack([arr, arr[0]])
    seg = np.linalg.norm(np.diff(arr, axis=0), axis=1)
    cumulative = np.r_[0.0, np.cumsum(seg)]
    total = float(cumulative[-1])
    if total <= 0:
        return arr
    n = max(64, int(np.ceil(total / max(float(spacing_m), 0.5))))
    target = np.linspace(0.0, total, n, endpoint=False)
    x = np.interp(target, cumulative, arr[:, 0])
    y = np.interp(target, cumulative, arr[:, 1])
    out = np.column_stack([x, y])
    return np.vstack([out, out[0]])




def _walk_closed_polyline_distance(
    points: np.ndarray,
    start_idx: int,
    direction: int,
    distance_m: float,
) -> int:
    """Walk around a closed polyline until the requested arc distance is reached."""
    pts = np.asarray(points, dtype=float)
    n = len(pts)
    if n < 2:
        return int(start_idx)
    idx = int(start_idx) % n
    travelled = 0.0
    for _ in range(n - 1):
        nxt = (idx + int(direction)) % n
        travelled += float(np.linalg.norm(pts[nxt] - pts[idx]))
        idx = nxt
        if travelled >= float(distance_m):
            break
    return int(idx)


def _cyclic_indices_inclusive(start: int, end: int, n: int) -> np.ndarray:
    """Indices from start to end (inclusive) while moving forward on a closed ring."""
    out = [int(start) % int(n)]
    cur = out[0]
    for _ in range(int(n)):
        if cur == int(end) % int(n):
            break
        cur = (cur + 1) % int(n)
        out.append(cur)
    return np.asarray(out, dtype=int)


def _straighten_user_marked_right_mid_segment(
    xy: np.ndarray,
    *,
    half_span_m: float,
    target_xy: tuple[float, float] = (1447.05, 384.09),
) -> tuple[np.ndarray, bool]:
    """Remove the small rounded kink at the user-marked right-middle boundary.

    The marked location corresponds to the east-side boundary around physical station
    38 (Blender XY approximately 1447.05, 384.09 m).  This is *not* the S39 corner
    farther to the right/lower side.  The requested result is one mathematically
    straight chord through the marked bend, rather than two straight lines meeting at
    another corner.

    The operation is intentionally performed after the global Gaussian smoothing and
    dense resampling.  Points on the local curved arc are deleted and the two retained
    anchors are connected directly by one line segment, so no later smoothing can
    re-introduce curvature.
    """
    arr = np.asarray(xy, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] < 8 or float(half_span_m) <= 0:
        return arr, False

    closed = float(np.linalg.norm(arr[0] - arr[-1])) <= 1e-8
    pts = arr[:-1].copy() if closed else arr.copy()
    n = len(pts)
    if n < 7:
        return arr, False

    target = np.asarray(target_xy, dtype=float)
    dist = np.linalg.norm(pts - target[None, :], axis=1)
    target_idx = int(np.argmin(dist))

    # Do nothing on unrelated/synthetic coordinate domains.  On the real campus
    # outline the marked boundary is very close to station 38.
    max_target_distance_m = max(220.0, 1.25 * float(half_span_m))
    if not np.isfinite(dist[target_idx]) or float(dist[target_idx]) > max_target_distance_m:
        return arr, False

    prev_anchor = _walk_closed_polyline_distance(pts, target_idx, -1, float(half_span_m))
    next_anchor = _walk_closed_polyline_distance(pts, target_idx, +1, float(half_span_m))
    if prev_anchor == target_idx or next_anchor == target_idx or prev_anchor == next_anchor:
        return arr, False

    # Keep the long, unaffected side of the ring.  Closing the rebuilt ring connects
    # prev_anchor directly back to next_anchor, which is the required single straight
    # segment across the marked kink.
    keep_idx = _cyclic_indices_inclusive(next_anchor, prev_anchor, n)
    kept = pts[keep_idx]
    if len(kept) < 3:
        return arr, False
    rebuilt = np.vstack([kept, kept[0]])
    return np.asarray(rebuilt, dtype=float), True


def extract_campus_outer_outline(
    source: Path,
    *,
    extent: tuple[float, float, float, float] | None = None,
    grid_m: float = 5.0,
    buffer_m: float = 40.0,
    max_buffer_m: float = 120.0,
    smooth_m: float = 18.0,
    min_coverage: float = 0.97,
    straight_segment_m: float = 220.0,
) -> CampusOutline:
    """Return a smooth closed campus envelope from aligned measurement trajectories.

    ``buffer_m`` is the initial geometric buffer around the aligned trajectories.  If
    the main connected component contains less than ``min_coverage`` of occupied
    trajectory cells, the buffer is increased automatically, up to ``max_buffer_m``.
    The final binary envelope is hole-filled and Gaussian-smoothed before a dense
    contour is extracted.  Only X/Y geometry is used; RSRP and other radio fields are
    never read by this routine.
    """
    xy, used_files = _load_aligned_xy(Path(source), extent)

    grid_m = float(grid_m)
    buffer_m = float(buffer_m)
    max_buffer_m = float(max_buffer_m)
    smooth_m = float(smooth_m)
    min_coverage = float(min_coverage)
    if grid_m <= 0:
        raise ValueError("grid_m must be positive")
    if buffer_m <= 0 or max_buffer_m < buffer_m:
        raise ValueError("Require 0 < buffer_m <= max_buffer_m")
    if smooth_m < 0:
        raise ValueError("smooth_m must be non-negative")
    if not (0.5 <= min_coverage <= 1.0):
        raise ValueError("min_coverage must lie in [0.5, 1.0]")

    if extent is None:
        pad = max(max_buffer_m + 3.0 * smooth_m, 20.0)
        xmin = float(np.min(xy[:, 0]) - pad)
        xmax = float(np.max(xy[:, 0]) + pad)
        ymin = float(np.min(xy[:, 1]) - pad)
        ymax = float(np.max(xy[:, 1]) + pad)
    else:
        xmin, xmax, ymin, ymax = map(float, extent)

    nx = max(3, int(np.floor((xmax - xmin) / grid_m)) + 1)
    ny = max(3, int(np.floor((ymax - ymin) / grid_m)) + 1)
    xs = xmin + np.arange(nx, dtype=float) * grid_m
    ys = ymin + np.arange(ny, dtype=float) * grid_m

    ix = np.rint((xy[:, 0] - xmin) / grid_m).astype(int)
    iy = np.rint((xy[:, 1] - ymin) / grid_m).astype(int)
    valid = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    if not np.any(valid):
        raise ValueError("No aligned measurement points fall inside the outline raster")
    ix = ix[valid]
    iy = iy[valid]

    occupied = np.zeros((ny, nx), dtype=bool)
    occupied[iy, ix] = True
    occupied_count = max(int(np.count_nonzero(occupied)), 1)

    # Euclidean distance to the nearest trajectory cell in metres.
    distance_m = distance_transform_edt(~occupied, sampling=(grid_m, grid_m))

    chosen_component: np.ndarray | None = None
    chosen_buffer = buffer_m
    chosen_coverage = 0.0
    trial_buffers = np.unique(
        np.clip(
            np.r_[buffer_m, buffer_m * 1.25, buffer_m * 1.5, buffer_m * 2.0, max_buffer_m],
            buffer_m,
            max_buffer_m,
        )
    )
    for trial in trial_buffers:
        buffered = distance_m <= float(trial)
        component, _ = _largest_component(buffered)
        coverage = float(np.count_nonzero(component & occupied)) / float(occupied_count)
        if coverage > chosen_coverage or chosen_component is None:
            chosen_component = component
            chosen_buffer = float(trial)
            chosen_coverage = coverage
        if coverage >= min_coverage:
            break

    if chosen_component is None or not np.any(chosen_component):
        raise ValueError("Unable to build a connected campus envelope")

    # Fill internal road-network holes: the required line is the campus *outer* edge.
    filled = binary_fill_holes(chosen_component)

    # Smooth the occupancy field so the extracted line is curved rather than a polygon
    # with abrupt QHull corners.  A small padding prevents a boundary touching the
    # raster edge from being clipped by the contour extractor.
    sigma = smooth_m / grid_m if smooth_m > 0 else 0.0
    field = gaussian_filter(filled.astype(np.float32), sigma=sigma, mode="nearest") if sigma > 0 else filled.astype(np.float32)

    contour = _extract_contour(field, xs, ys, level_value=0.5)
    contour = _resample_closed_polyline(contour, spacing_m=max(2.0, grid_m * 0.75))

    # Final publication-only geometry correction requested for the marked bend near
    # station 38.  This is deliberately after all smoothing/resampling.
    contour, straight_applied = _straighten_user_marked_right_mid_segment(
        contour, half_span_m=float(straight_segment_m)
    )

    return CampusOutline(
        xy=np.asarray(contour, dtype=float),
        source_files=used_files,
        source_point_count=int(xy.shape[0]),
        hull_vertex_count=int(contour.shape[0] - 1),
        method=(
            "smoothed_buffered_envelope_with_user_marked_right_mid_straight_segment"
            if straight_applied
            else "smoothed_buffered_envelope_of_aligned_measurement_trajectories"
        ),
        grid_m=grid_m,
        buffer_m=float(chosen_buffer),
        smooth_m=smooth_m,
        covered_fraction=float(chosen_coverage),
        straight_segment_applied=bool(straight_applied),
        straight_segment_half_span_m=float(straight_segment_m if straight_applied else 0.0),
    )
