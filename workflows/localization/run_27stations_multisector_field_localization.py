#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multi-Sector Spatial-Field Intersection localization (MSFI-v1.12).

This solver is intentionally independent of the previous DP-PGRSL, DP-PPRSL,
and RSGL formulations. It does not convert RSRP into an explicit range and does
not jointly optimize transmit power, path-loss exponent, or antenna gain.

For each sector/PCI in a randomly sampled receiver subset it estimates a robust
spatial field descriptor:
  * a high-RSRP spatial centroid;
  * a local RSRP gradient direction;
  * the externally supplied sector boresight when available.

Each descriptor defines a backward ray from the measured high-signal region
toward the likely transmitter location. The physical station is estimated by a
robust intersection of the available sector rays, followed by a low-dimensional
XY refinement that uses only:
  1) perpendicular distance to the sector field rays;
  2) sector half-plane consistency;
  3) same-location multi-PCI relative-RSRP / sector-gain consistency; and
  4) a weak field-order consistency term.

Ground-truth station coordinates are used only after localization to calculate
reported errors. Random point selection is uniform without replacement.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

import legacy_pgrmsbil as common

ALGORITHM_NAME = "Multi-Sector Spatial-Field Intersection Localization (MSFI-v1.12)"
ROOT = Path(__file__).resolve().parents[2]
MIN_RSRP_DBM = -120.0
MAX_RSRP_DBM = -40.0
DEFAULT_BOUNDS = common.DEFAULT_BOUNDS
HORIZONTAL_3DB_BEAMWIDTH_DEG = 65.0
HORIZONTAL_MAX_ATTENUATION_DB = 30.0
VERTICAL_SEPARATION_M = 28.5


@dataclass
class SectorField:
    sector_index: int
    count: int
    centroid: np.ndarray
    boresight: np.ndarray
    back_direction: np.ndarray
    gradient_direction: np.ndarray
    gradient_reliability: float
    signal_spread_db: float
    spatial_spread_m: float
    weight: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MSFI-v1.12 random sparse base-station localization")
    p.add_argument("--project-root", type=Path, default=ROOT)
    p.add_argument("--measurements", type=Path, default=None)
    p.add_argument("--directions", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--points-per-station", type=int, default=10)
    p.add_argument("--random-seed", type=int, default=20260805)
    p.add_argument("--direction-prior-mode", choices=["fixed", "soft", "off"], default="fixed")
    p.add_argument("--bootstrap", type=int, default=0, help="Compatibility option; outer 10-trial Monte Carlo is the stability experiment")
    p.add_argument("--de-maxiter", type=int, default=100, help="Compatibility option, unused by MSFI")
    p.add_argument("--de-popsize", type=int, default=10, help="Compatibility option, unused by MSFI")
    p.add_argument("--station-ids", default="all")
    p.add_argument("--x-min", type=float, default=DEFAULT_BOUNDS[0])
    p.add_argument("--x-max", type=float, default=DEFAULT_BOUNDS[1])
    p.add_argument("--y-min", type=float, default=DEFAULT_BOUNDS[2])
    p.add_argument("--y-max", type=float, default=DEFAULT_BOUNDS[3])
    p.add_argument("--skip-figures", action="store_true")
    p.add_argument("--skip-per-station-figures", action="store_true")
    return p.parse_args()


def resolve_direction_csv(project_root: Path, explicit: Optional[Path]) -> Optional[Path]:
    if explicit is not None:
        path = explicit.expanduser().resolve()
        return path if path.is_file() else None
    candidates = [
        project_root / "outputs/parameter_calibration/estimated_initial_directions_27stations.csv",
        project_root / "config/estimated_initial_directions_27stations.csv",
        project_root / "config/metadata/estimated_initial_directions_27stations.csv",
    ]
    return next((p.resolve() for p in candidates if p.is_file()), None)


def load_directions(path: Optional[Path]) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    frame = pd.read_csv(path, encoding="utf-8-sig")
    if "station_id" not in frame.columns or "base_alpha_rad" not in frame.columns:
        return pd.DataFrame()
    frame = frame.copy()
    frame["station_id"] = pd.to_numeric(frame["station_id"], errors="coerce").astype("Int64")
    return frame.dropna(subset=["station_id"]).set_index("station_id")


def random_points(points: pd.DataFrame, k: int, seed: int) -> pd.DataFrame:
    cols = ["rsrp_s1", "rsrp_s2", "rsrp_s3"]
    pool = points.loc[points[cols].notna().any(axis=1)].copy().reset_index(drop=True)
    if len(pool) < int(k):
        raise ValueError(f"定位有效实测位置仅{len(pool)}个，少于要求{k}个")
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(len(pool), size=int(k), replace=False)
    out = pool.iloc[idx].copy().reset_index(drop=True)
    out["selection_rank"] = np.arange(1, len(out) + 1)
    out["observed_sector_count"] = out[cols].notna().sum(axis=1)
    return out


def wrap(angle: np.ndarray | float) -> np.ndarray:
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


def sector_gain_db(offset_rad: np.ndarray) -> np.ndarray:
    deg = np.degrees(np.abs(wrap(offset_rad)))
    return -np.minimum(12.0 * (deg / HORIZONTAL_3DB_BEAMWIDTH_DEG) ** 2, HORIZONTAL_MAX_ATTENUATION_DB)


def sector_angles(direction_row: Optional[pd.Series], mode: str) -> tuple[np.ndarray, bool]:
    if mode == "off" or direction_row is None:
        return np.asarray([0.0, 2.0 * np.pi / 3.0, -2.0 * np.pi / 3.0]), False
    alpha = pd.to_numeric(direction_row.get("base_alpha_rad", np.nan), errors="coerce")
    if not np.isfinite(alpha):
        return np.asarray([0.0, 2.0 * np.pi / 3.0, -2.0 * np.pi / 3.0]), False
    order = str(direction_row.get("selected_sector_order", ""))
    sign = 1 if "plus120" in order else -1
    return float(alpha) + np.asarray([0.0, sign * 2.0 * np.pi / 3.0, -sign * 2.0 * np.pi / 3.0]), True


def _unit(v: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    norm = float(np.linalg.norm(v))
    if norm > 1e-9 and np.isfinite(norm):
        return v / norm
    if fallback is not None:
        return _unit(np.asarray(fallback, dtype=float))
    return np.asarray([1.0, 0.0])


def _robust_plane_gradient(xy: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, float]:
    """Robust local linear field gradient and a 0--1 reliability score."""
    xy = np.asarray(xy, float)
    values = np.asarray(values, float)
    if len(values) < 3:
        return np.zeros(2), 0.0
    center = np.mean(xy, axis=0)
    scale_xy = max(float(np.sqrt(np.mean(np.sum((xy - center) ** 2, axis=1)))), 20.0)
    X = np.column_stack([np.ones(len(xy)), (xy - center) / scale_xy])
    y = values.copy()
    try:
        coef = np.linalg.lstsq(X, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return np.zeros(2), 0.0
    weights = np.ones(len(y))
    for _ in range(6):
        pred = X @ coef
        residual = y - pred
        scale = max(1.4826 * float(np.median(np.abs(residual - np.median(residual)))), 1.5)
        z = np.abs(residual) / (1.5 * scale)
        weights = np.where(z <= 1.0, 1.0, 1.0 / np.maximum(z, 1e-6))
        try:
            coef = np.linalg.lstsq(X * np.sqrt(weights)[:, None], y * np.sqrt(weights), rcond=None)[0]
        except np.linalg.LinAlgError:
            break
    grad = np.asarray(coef[1:3], float) / scale_xy
    pred = X @ coef
    ss_res = float(np.sum(weights * (y - pred) ** 2))
    ybar = float(np.average(y, weights=weights))
    ss_tot = float(np.sum(weights * (y - ybar) ** 2))
    r2 = 0.0 if ss_tot < 1e-9 else float(np.clip(1.0 - ss_res / ss_tot, 0.0, 1.0))
    spatial_span = max(float(np.ptp(xy[:, 0])), float(np.ptp(xy[:, 1])), 1.0)
    signal_span = float(np.ptp(y))
    gradient_strength = float(np.linalg.norm(grad)) * spatial_span
    strength_score = float(np.clip(gradient_strength / max(signal_span, 4.0), 0.0, 1.0))
    reliability = float(np.clip(0.7 * r2 + 0.3 * strength_score, 0.0, 1.0))
    return grad, reliability


def _weighted_high_signal_centroid(xy: np.ndarray, values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, float)
    xy = np.asarray(xy, float)
    if len(values) == 1:
        return xy[0].copy()
    q = float(np.quantile(values, 0.55))
    idx = values >= q
    if int(np.sum(idx)) < 2:
        idx = np.argsort(values)[-min(3, len(values)):]
        xsel = xy[idx]
        vsel = values[idx]
    else:
        xsel = xy[idx]
        vsel = values[idx]
    weights = np.exp(np.clip((vsel - np.max(vsel)) / 5.0, -8.0, 0.0))
    return np.average(xsel, axis=0, weights=weights)


def build_sector_fields(selected: pd.DataFrame, angles: np.ndarray, direction_used: bool, omni: bool) -> list[SectorField]:
    if omni:
        return []
    xy_all = selected[["x_m", "y_m"]].to_numpy(float)
    rss_all = selected[["rsrp_s1", "rsrp_s2", "rsrp_s3"]].to_numpy(float)
    fields: list[SectorField] = []
    for j in range(3):
        mask = np.isfinite(rss_all[:, j])
        if int(np.sum(mask)) < 2:
            continue
        xy = xy_all[mask]
        vals = rss_all[mask, j]
        centroid = _weighted_high_signal_centroid(xy, vals)
        grad, grad_rel = _robust_plane_gradient(xy, vals)
        grad_dir = _unit(grad, fallback=-np.asarray([math.cos(angles[j]), math.sin(angles[j])]))
        boresight = _unit(np.asarray([math.cos(angles[j]), math.sin(angles[j])]))
        # RSRP gradient points toward stronger signal (typically toward the site),
        # while -boresight points backward from the served sector toward the site.
        if direction_used:
            blend = (0.78 * (-boresight)) + (0.22 * grad_rel * grad_dir)
            back = _unit(blend, fallback=-boresight)
        else:
            back = grad_dir
        spatial_spread = max(float(np.sqrt(np.mean(np.sum((xy - centroid) ** 2, axis=1)))), 20.0)
        signal_spread = max(float(np.ptp(vals)), 2.0)
        count_score = min(1.0, len(vals) / 5.0)
        weight = float(np.clip(0.40 + 0.35 * count_score + 0.25 * grad_rel, 0.25, 1.0))
        fields.append(SectorField(
            sector_index=j,
            count=int(len(vals)),
            centroid=np.asarray(centroid, float),
            boresight=boresight,
            back_direction=back,
            gradient_direction=grad_dir,
            gradient_reliability=float(grad_rel),
            signal_spread_db=float(signal_spread),
            spatial_spread_m=float(spatial_spread),
            weight=weight,
        ))
    return fields


def robust_ray_intersection(fields: list[SectorField], fallback: np.ndarray) -> tuple[np.ndarray, float]:
    if len(fields) < 2:
        return np.asarray(fallback, float), float("nan")
    A = []
    b = []
    base_w = []
    for field in fields:
        d = _unit(field.back_direction)
        n = np.asarray([-d[1], d[0]])
        A.append(n)
        b.append(float(n @ field.centroid))
        base_w.append(float(field.weight))
    A = np.asarray(A, float)
    b = np.asarray(b, float)
    weights = np.asarray(base_w, float)
    solution = np.asarray(fallback, float)
    for _ in range(6):
        sw = np.sqrt(np.clip(weights, 1e-3, None))
        try:
            solution = np.linalg.lstsq(A * sw[:, None], b * sw, rcond=None)[0]
        except np.linalg.LinAlgError:
            return np.asarray(fallback, float), float("nan")
        residual = A @ solution - b
        scale = max(1.4826 * float(np.median(np.abs(residual - np.median(residual)))), 15.0)
        huber = np.where(np.abs(residual) <= 1.5 * scale, 1.0, 1.5 * scale / np.maximum(np.abs(residual), 1e-6))
        weights = np.asarray(base_w, float) * huber
    rms = float(np.sqrt(np.average((A @ solution - b) ** 2, weights=np.asarray(base_w, float))))
    return np.asarray(solution, float), rms


def _sector_relative_loss(tx: np.ndarray, selected: pd.DataFrame, angles: np.ndarray) -> float:
    """Same-location relative-RSRP angular consistency; no absolute path-loss."""
    xy = selected[["x_m", "y_m"]].to_numpy(float)
    rss = selected[["rsrp_s1", "rsrp_s2", "rsrp_s3"]].to_numpy(float)
    losses: list[float] = []
    for i, p in enumerate(xy):
        valid = np.flatnonzero(np.isfinite(rss[i]))
        if len(valid) < 2:
            continue
        bearing = math.atan2(p[1] - tx[1], p[0] - tx[0])
        gains = np.asarray([sector_gain_db(np.asarray([bearing - angles[j]]))[0] for j in range(3)], float)
        obs = rss[i, valid]
        pred = gains[valid]
        obs = obs - float(np.mean(obs))
        pred = pred - float(np.mean(pred))
        denom = float(np.dot(pred, pred))
        scale = 0.0 if denom < 1e-8 else float(np.clip(np.dot(obs, pred) / denom, 0.0, 2.0))
        residual = obs - scale * pred
        losses.extend(np.log1p((residual / 5.0) ** 2).tolist())
    return float(np.mean(losses)) if losses else 0.0


def _field_order_loss(tx: np.ndarray, selected: pd.DataFrame, angles: np.ndarray, omni: bool) -> float:
    """Weak pairwise ordering term based on a geometry-only propagation score."""
    xy = selected[["x_m", "y_m"]].to_numpy(float)
    rss = selected[["rsrp_s1", "rsrp_s2", "rsrp_s3"]].to_numpy(float)
    d = np.sqrt(np.sum((xy - tx[None, :]) ** 2, axis=1) + VERTICAL_SEPARATION_M ** 2)
    penalties: list[float] = []
    for j in range(1 if omni else 3):
        vals = rss[:, j]
        idx = np.flatnonzero(np.isfinite(vals))
        if len(idx) < 3:
            continue
        if omni:
            geom_score = -np.log(np.maximum(d, 1.0))
        else:
            bearing = np.arctan2(xy[:, 1] - tx[1], xy[:, 0] - tx[0])
            gain = sector_gain_db(bearing - angles[j])
            geom_score = -np.log(np.maximum(d, 1.0)) + 0.035 * gain
        # Evaluate only large observed differences to reduce fading sensitivity.
        for aa in range(len(idx)):
            for bb in range(aa):
                ia, ib = idx[aa], idx[bb]
                delta_r = vals[ia] - vals[ib]
                if abs(delta_r) < 6.0:
                    continue
                delta_g = geom_score[ia] - geom_score[ib]
                if delta_r * delta_g < 0.0:
                    penalties.append(min(abs(delta_g) / 0.8, 2.0) ** 2)
    return float(np.mean(penalties)) if penalties else 0.0


def _ray_geometry_loss(tx: np.ndarray, fields: list[SectorField]) -> tuple[float, float]:
    if not fields:
        return 0.0, 0.0
    perp_losses = []
    half_losses = []
    for field in fields:
        d = _unit(field.back_direction)
        rel = tx - field.centroid
        perp = abs(float(d[0] * rel[1] - d[1] * rel[0]))
        scale = max(field.spatial_spread_m, 35.0)
        perp_losses.append(field.weight * math.log1p((perp / scale) ** 2))
        forward = float(np.dot(rel, d))  # should be positive along the backward ray
        if forward < -10.0:
            half_losses.append(field.weight * min(abs(forward) / 100.0, 3.0) ** 2)
    return float(np.mean(perp_losses)), float(np.mean(half_losses)) if half_losses else 0.0


def _omni_center(selected: pd.DataFrame) -> np.ndarray:
    xy = selected[["x_m", "y_m"]].to_numpy(float)
    values = selected[["rsrp_s1", "rsrp_s2", "rsrp_s3"]].max(axis=1, skipna=True).to_numpy(float)
    return _weighted_high_signal_centroid(xy, values)


def _omni_objective(tx: np.ndarray, selected: pd.DataFrame, center: np.ndarray, spread: float) -> float:
    order = _field_order_loss(tx, selected, np.zeros(3), True)
    weak = 0.04 * (float(np.linalg.norm(tx - center)) / max(spread, 80.0)) ** 2
    return 0.90 * order + weak


def _directional_objective(tx: np.ndarray, selected: pd.DataFrame, angles: np.ndarray, fields: list[SectorField], center: np.ndarray, spread: float) -> float:
    ray, half = _ray_geometry_loss(tx, fields)
    relative = _sector_relative_loss(tx, selected, angles)
    order = _field_order_loss(tx, selected, angles, False)
    weak = 0.006 * (float(np.linalg.norm(tx - center)) / max(spread, 100.0)) ** 2
    return 0.52 * ray + 0.18 * half + 0.25 * relative + 0.05 * order + weak


def _candidate_centers(fields: list[SectorField], intersection: np.ndarray, selected: pd.DataFrame) -> list[np.ndarray]:
    centers = [np.asarray(intersection, float), np.mean(selected[["x_m", "y_m"]].to_numpy(float), axis=0)]
    for field in fields:
        centers.append(field.centroid + 120.0 * field.back_direction)
    # Pairwise exact line intersections where possible.
    for a in range(len(fields)):
        for b in range(a + 1, len(fields)):
            f1, f2 = fields[a], fields[b]
            M = np.column_stack([f1.back_direction, -f2.back_direction])
            rhs = f2.centroid - f1.centroid
            if abs(float(np.linalg.det(M))) < 0.08:
                continue
            try:
                t = np.linalg.solve(M, rhs)
                p = f1.centroid + t[0] * f1.back_direction
                if np.isfinite(p).all():
                    centers.append(p)
            except np.linalg.LinAlgError:
                pass
    unique: list[np.ndarray] = []
    for c in centers:
        if not np.isfinite(c).all():
            continue
        if all(float(np.linalg.norm(c - u)) > 25.0 for u in unique):
            unique.append(np.asarray(c, float))
    return unique


def solve(selected: pd.DataFrame, angles: np.ndarray, direction_used: bool, omni: bool, bounds: Sequence[float]) -> dict:
    xy = selected[["x_m", "y_m"]].to_numpy(float)
    spread = max(common.point_spread(selected), 80.0)
    x0, x1, y0, y1 = map(float, bounds)

    if omni:
        center = _omni_center(selected)
        local_radius = max(250.0, min(550.0, 1.5 * spread))
        lo = np.maximum([x0, y0], center - local_radius)
        hi = np.minimum([x1, y1], center + local_radius)
        starts = [center, np.mean(xy, axis=0)]
        best_xy = np.clip(center, lo, hi)
        best_val = _omni_objective(best_xy, selected, center, spread)
        for start in starts:
            result = minimize(
                lambda z: _omni_objective(np.asarray(z, float), selected, center, spread),
                np.clip(start, lo, hi), method="Powell",
                bounds=[(lo[0], hi[0]), (lo[1], hi[1])],
                options={"maxiter": 220, "xtol": 0.25, "ftol": 1e-5},
            )
            candidate = np.asarray(result.x if result.success else start, float)
            value = _omni_objective(candidate, selected, center, spread)
            if value < best_val:
                best_xy, best_val = candidate, value
        return {
            "final_xy": best_xy,
            "objective": float(best_val),
            "solver_mode": "msfi_omni_rank_center",
            "field_count": 1,
            "ray_intersection_x_m": float(center[0]),
            "ray_intersection_y_m": float(center[1]),
            "ray_intersection_rms_m": float("nan"),
            "field_geometry_score": float("nan"),
        }

    fields = build_sector_fields(selected, angles, direction_used, omni=False)
    fallback = np.mean(xy, axis=0)
    intersection, intersection_rms = robust_ray_intersection(fields, fallback)
    centers = _candidate_centers(fields, intersection, selected)
    if not centers:
        centers = [fallback]

    # Search bounds are intentionally limited by observed geometry and ray candidates;
    # this prevents unconstrained kilometer-scale optima without using ground truth.
    cstack = np.vstack(centers + [np.min(xy, axis=0), np.max(xy, axis=0)])
    margin = max(220.0, min(520.0, 1.15 * spread))
    lo = np.maximum([x0, y0], np.min(cstack, axis=0) - margin)
    hi = np.minimum([x1, y1], np.max(cstack, axis=0) + margin)
    center_ref = np.asarray(intersection if np.isfinite(intersection).all() else fallback, float)

    best_xy = np.clip(center_ref, lo, hi)
    best_val = _directional_objective(best_xy, selected, angles, fields, center_ref, spread)
    for start in centers:
        start = np.clip(start, lo, hi)
        result = minimize(
            lambda z: _directional_objective(np.asarray(z, float), selected, angles, fields, center_ref, spread),
            start, method="Powell", bounds=[(lo[0], hi[0]), (lo[1], hi[1])],
            options={"maxiter": 260, "xtol": 0.25, "ftol": 1e-5},
        )
        candidate = np.asarray(result.x if result.success else start, float)
        value = _directional_objective(candidate, selected, angles, fields, center_ref, spread)
        if value < best_val:
            best_xy, best_val = candidate, value

    ray_loss, half_loss = _ray_geometry_loss(best_xy, fields)
    geometry_score = float(np.exp(-max(ray_loss + 0.5 * half_loss, 0.0)))
    return {
        "final_xy": best_xy,
        "objective": float(best_val),
        "solver_mode": "multi_sector_spatial_field_intersection",
        "field_count": int(len(fields)),
        "ray_intersection_x_m": float(intersection[0]),
        "ray_intersection_y_m": float(intersection[1]),
        "ray_intersection_rms_m": float(intersection_rms),
        "field_geometry_score": geometry_score,
        "sector_relative_loss": float(_sector_relative_loss(best_xy, selected, angles)),
        "field_order_loss": float(_field_order_loss(best_xy, selected, angles, False)),
    }


def quality_flag(solution: dict, selected: pd.DataFrame, direction_used: bool, omni: bool) -> str:
    flags: list[str] = []
    spread = common.point_spread(selected)
    if spread < 60.0:
        flags.append("low_point_spread")
    if not omni:
        if int(solution.get("field_count", 0)) < 2:
            flags.append("insufficient_sector_fields")
        rms = float(solution.get("ray_intersection_rms_m", np.nan))
        if np.isfinite(rms) and rms > 120.0:
            flags.append("high_ray_intersection_residual")
        score = float(solution.get("field_geometry_score", np.nan))
        if np.isfinite(score) and score < 0.35:
            flags.append("low_field_geometry_confidence")
        if not direction_used:
            flags.append("no_external_direction_prior")
    return ";".join(flags) if flags else "ok"


def plot_station(selected: pd.DataFrame, pred: np.ndarray, truth: np.ndarray, station_id: int, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.4), dpi=180)
    ax.scatter(selected.x_m, selected.y_m, s=38, facecolors="none", edgecolors="black", label="Random measured points")
    ax.scatter(pred[0], pred[1], marker="*", s=170, c="red", edgecolors="black", label="Estimate")
    ax.scatter(truth[0], truth[1], marker="x", s=100, c="black", linewidths=2, label="Ground truth")
    ax.plot([pred[0], truth[0]], [pred[1], truth[1]], "--", lw=1)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=.25)
    ax.set_xlabel("Blender X [m]")
    ax.set_ylabel("Blender Y [m]")
    ax.set_title(f"Station {station_id} random-point localization")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    project = args.project_root.expanduser().resolve()
    measurement_path = common.resolve_measurement_csv(project, args.measurements)
    localization, truth = common.load_and_filter(measurement_path)
    localization = localization[
        localization["rsrp_dbm"].between(MIN_RSRP_DBM, MAX_RSRP_DBM, inclusive="both")
    ].copy()
    direction_path = resolve_direction_csv(project, args.directions)
    directions = load_directions(direction_path)
    station_ids = common.parse_station_ids(args.station_ids, sorted(localization.station_id.unique().astype(int)))
    truth_index = truth.set_index("station_id")
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else project / "outputs" / f"localization_msfi_{args.points_per_station}points_seed_{args.random_seed}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    per_station = output_dir / "per_station"
    per_station.mkdir(exist_ok=True)

    rows: list[dict] = []
    selected_rows: list[pd.DataFrame] = []
    for station_id in station_ids:
        station = localization[localization.station_id.eq(station_id)].copy()
        truth_row = truth_index.loc[station_id]
        omni = bool(int(truth_row.is_omnidirectional)) or station_id == 22
        points = common.point_table(station)
        selected = random_points(points, args.points_per_station, args.random_seed + station_id * 7919)
        direction_row = directions.loc[station_id] if station_id in directions.index else None
        angles, direction_used = sector_angles(direction_row, args.direction_prior_mode)
        start = time.time()
        solution = solve(
            selected, angles, direction_used, omni,
            (args.x_min, args.x_max, args.y_min, args.y_max),
        )
        elapsed = time.time() - start
        predicted = np.asarray(solution["final_xy"], float)
        true_xy = np.asarray([truth_row.true_x_m, truth_row.true_y_m], float)
        delta = predicted - true_xy
        error = float(np.linalg.norm(delta))
        observations = common.observations_from_points(selected)
        qflag = quality_flag(solution, selected, direction_used, omni)
        row = {
            "station_id": int(station_id),
            "station_label": str(truth_row.station_label),
            "antenna_type": str(truth_row.antenna_type),
            "selected_point_count": int(len(selected)),
            "observation_count": int(len(observations)),
            "distinct_pci_count": int(observations.pci.nunique()) if len(observations) else 0,
            "predicted_x_m": float(predicted[0]),
            "predicted_y_m": float(predicted[1]),
            "true_x_m": float(true_xy[0]),
            "true_y_m": float(true_xy[1]),
            "east_error_m": float(delta[0]),
            "north_error_m": float(delta[1]),
            "horizontal_error_m": error,
            "objective_value": float(solution["objective"]),
            "direction_prior_used": bool(direction_used),
            "point_spread_m": float(common.point_spread(selected)),
            "quality_flag": qflag,
            "solver_mode": str(solution["solver_mode"]),
            "field_count": int(solution.get("field_count", 0)),
            "ray_intersection_x_m": float(solution.get("ray_intersection_x_m", np.nan)),
            "ray_intersection_y_m": float(solution.get("ray_intersection_y_m", np.nan)),
            "ray_intersection_rms_m": float(solution.get("ray_intersection_rms_m", np.nan)),
            "field_geometry_score": float(solution.get("field_geometry_score", np.nan)),
            "sector_relative_loss": float(solution.get("sector_relative_loss", np.nan)),
            "field_order_loss": float(solution.get("field_order_loss", np.nan)),
            "elapsed_s": float(elapsed),
            "rsrp_min_dbm": MIN_RSRP_DBM,
            "rsrp_max_dbm": MAX_RSRP_DBM,
        }
        rows.append(row)
        selected_out = selected.copy()
        selected_out.insert(0, "station_id", station_id)
        selected_rows.append(selected_out)
        if not args.skip_figures and not args.skip_per_station_figures:
            plot_station(selected, predicted, true_xy, station_id, per_station / f"station_{station_id:02d}_localization.png")
        print(
            f"Station {station_id:02d}: points={len(selected)}, obs={len(observations)}, "
            f"estimate=({predicted[0]:.2f},{predicted[1]:.2f}), error={error:.2f} m, quality={qflag}"
        )

    results = pd.DataFrame(rows).sort_values("station_id").reset_index(drop=True)
    results.to_csv(
        output_dir / f"localization_results_{len(results)}stations_{args.points_per_station}points.csv",
        index=False, encoding="utf-8-sig",
    )
    pd.concat(selected_rows, ignore_index=True).to_csv(
        output_dir / f"selected_{args.points_per_station}_random_points_all_stations.csv",
        index=False, encoding="utf-8-sig",
    )
    errors = results.horizontal_error_m.to_numpy(float)
    summary = pd.DataFrame([{
        "algorithm": ALGORITHM_NAME,
        "station_count": int(len(errors)),
        "requested_points_per_station": int(args.points_per_station),
        "mean_error_m": float(np.mean(errors)),
        "median_error_m": float(np.median(errors)),
        "rmse_m": float(np.sqrt(np.mean(errors ** 2))),
        "p90_error_m": float(np.percentile(errors, 90)),
        "p95_error_m": float(np.percentile(errors, 95)),
        "max_error_m": float(np.max(errors)),
        "within_50m_percent": float(np.mean(errors <= 50.0) * 100.0),
        "within_100m_percent": float(np.mean(errors <= 100.0) * 100.0),
        "quality_flagged_count": int(np.sum(results.quality_flag.astype(str) != "ok")),
    }])
    summary.to_csv(output_dir / "localization_accuracy_summary.csv", index=False, encoding="utf-8-sig")
    (output_dir / "experiment_metadata.json").write_text(json.dumps({
        "algorithm": ALGORITHM_NAME,
        "random_sampling": "uniform_without_replacement",
        "rsrp_range_dbm": [MIN_RSRP_DBM, MAX_RSRP_DBM],
        "random_seed": int(args.random_seed),
        "points_per_station": int(args.points_per_station),
        "direction_prior_mode": args.direction_prior_mode,
        "direction_csv": str(direction_path) if direction_path else None,
        "truth_used_only_for_final_evaluation": True,
        "explicit_rsrp_to_distance_conversion": False,
        "joint_pathloss_power_optimization": False,
    }, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
