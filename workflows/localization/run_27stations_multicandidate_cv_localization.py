#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multi-Candidate Cross-Validated Localization (MCVL-v1.13).

The previous MSFI solver used one spatial-field intersection objective as the
position estimate.  MCVL intentionally separates *candidate generation* from
*candidate validation*:

1. Random receiver locations are sampled uniformly without replacement.
2. Multiple geometry-only candidate transmitter positions are generated from
   sector high-RSRP centroids, fixed sector boresights, robust RSRP gradients,
   pairwise ray intersections, and measurement centroids.
3. Every candidate is evaluated by leave-one-location-out (LOLO) prediction of
   the held-out PCI-RSRP observations.  A small ridge validation model is fitted
   only on the remaining receiver locations.  It contains sector offsets, a
   shared distance-decay term and (when a direction prior is available) a fixed
   antenna-gain feature.
4. The best few candidates are refined by a small two-dimensional local grid,
   again using only cross-validated measurement consistency.

The propagation model is therefore a *validator*, not the direct range inverse
used to create the position.  Ground-truth coordinates are read only after the
final estimate has been chosen, solely to calculate evaluation errors.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import legacy_pgrmsbil as common

RECONSTRUCTION_ROOT = Path(__file__).resolve().parents[1] / "reconstruction"
if str(RECONSTRUCTION_ROOT) not in sys.path:
    sys.path.insert(0, str(RECONSTRUCTION_ROOT))

from radio_reconstruction.simulation_prior import (  # noqa: E402
    discover_simulation_prior,
    load_simulation_prior,
)

ALGORITHM_NAME = "Multi-Candidate Leave-One-Location-Out Cross-Validated Localization (MCVL-v1.13)"
ROOT = Path(__file__).resolve().parents[2]
MIN_RSRP_DBM = -120.0
MAX_RSRP_DBM = -40.0
DEFAULT_BOUNDS = common.DEFAULT_BOUNDS
VERTICAL_SEPARATION_M = 28.5
HORIZONTAL_3DB_BEAMWIDTH_DEG = 65.0
HORIZONTAL_MAX_ATTENUATION_DB = 30.0


@dataclass
class SectorDescriptor:
    sector_index: int
    count: int
    high_centroid: np.ndarray
    boresight: np.ndarray
    back_boresight: np.ndarray
    gradient_direction: np.ndarray
    gradient_reliability: float
    spatial_spread_m: float
    signal_span_db: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MCVL-v1.13 random sparse base-station localization")
    p.add_argument("--project-root", type=Path, default=ROOT)
    p.add_argument("--measurements", type=Path, default=None)
    p.add_argument("--directions", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--points-per-station", type=int, default=10)
    p.add_argument("--random-seed", type=int, default=20260805)
    p.add_argument(
        "--previous-results", type=Path, default=None,
        help="上一点数同一trial的定位结果CSV；用于渐进式候选继承，不使用真值进行候选选择",
    )
    p.add_argument(
        "--previous-without-simulation-results", type=Path, default=None,
        help="compare模式下无仿真分支的上一点数结果CSV",
    )
    p.add_argument(
        "--previous-with-simulation-results", type=Path, default=None,
        help="compare模式下有仿真分支的上一点数结果CSV",
    )
    p.add_argument(
        "--progressive-min-improvement-db", type=float, default=0.25,
        help="新候选相对上一点数位置至少需要改善的当前数据候选分数[dB]；不足时保留上一位置",
    )
    p.add_argument("--direction-prior-mode", choices=["fixed", "soft", "off"], default="fixed")
    p.add_argument("--bootstrap", type=int, default=0, help="Compatibility option; the outer 10 random trials are the stability experiment")
    p.add_argument("--de-maxiter", type=int, default=100, help="Compatibility option, unused by MCVL")
    p.add_argument("--de-popsize", type=int, default=10, help="Compatibility option, unused by MCVL")
    p.add_argument("--station-ids", default="all")
    p.add_argument("--x-min", type=float, default=DEFAULT_BOUNDS[0])
    p.add_argument("--x-max", type=float, default=DEFAULT_BOUNDS[1])
    p.add_argument("--y-min", type=float, default=DEFAULT_BOUNDS[2])
    p.add_argument("--y-max", type=float, default=DEFAULT_BOUNDS[3])
    p.add_argument("--skip-figures", action="store_true")
    p.add_argument("--skip-per-station-figures", action="store_true")
    p.add_argument(
        "--simulation-mode", choices=["without", "with", "compare"], default="without",
        help=(
            "without=仅使用实测PCI-RSRP；with=在相同接收位置联合使用实测与仿真PCI-RSRP；"
            "compare=在完全相同实测位置和随机划分上同时运行两者"
        ),
    )
    p.add_argument(
        "--simulation-root", type=Path, default=None,
        help="纯仿真NPZ搜索根目录；默认outputs/bestparam_dem_vs_zplane_512m",
    )
    p.add_argument(
        "--strict-simulation-data", action="store_true",
        help="任一站/PCI缺少纯仿真NPZ时立即失败；默认记录缺失并退化为仅实测",
    )
    p.add_argument(
        "--simulation-weight", type=float, default=0.50,
        help="联合候选评分中仿真RSRP通道的权重，范围0--1；默认0.50",
    )
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
    """Nested uniform random sampling from all localization-eligible locations.

    A deterministic random permutation is generated from ``seed`` and the first
    ``k`` locations are returned.  Therefore, when the same trial seed is reused
    for 10,11,...,15 points, the samples are strictly nested:
    S10 ⊂ S11 ⊂ ... ⊂ S15.  The ranking uses no RSRP magnitude or ground truth.
    """
    cols = ["rsrp_s1", "rsrp_s2", "rsrp_s3"]
    pool = points.loc[points[cols].notna().any(axis=1)].copy().reset_index(drop=True)
    if len(pool) < int(k):
        raise ValueError(f"定位有效实测位置仅{len(pool)}个，少于要求{k}个")
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(len(pool))
    idx = order[: int(k)]
    out = pool.iloc[idx].copy().reset_index(drop=True)
    out["selection_rank"] = np.arange(1, len(out) + 1)
    out["observed_sector_count"] = out[cols].notna().sum(axis=1)
    out["selection_strategy"] = "nested_uniform_random_prefix"
    return out


def load_previous_predictions(path: Optional[Path]) -> dict[int, np.ndarray]:
    """Load previous-count predictions only; true coordinates are deliberately ignored."""
    if path is None:
        return {}
    p = path.expanduser().resolve()
    if not p.is_file():
        return {}
    frame = pd.read_csv(p, encoding="utf-8-sig")
    required = {"station_id", "predicted_x_m", "predicted_y_m"}
    if not required.issubset(frame.columns):
        return {}
    out: dict[int, np.ndarray] = {}
    for row in frame.itertuples(index=False):
        sid = int(getattr(row, "station_id"))
        xy = np.asarray([getattr(row, "predicted_x_m"), getattr(row, "predicted_y_m")], dtype=float)
        if np.isfinite(xy).all():
            out[sid] = xy
    return out


def wrap(angle: np.ndarray | float) -> np.ndarray:
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


def sector_gain_db(offset_rad: np.ndarray) -> np.ndarray:
    deg = np.degrees(np.abs(wrap(offset_rad)))
    return -np.minimum(12.0 * (deg / HORIZONTAL_3DB_BEAMWIDTH_DEG) ** 2, HORIZONTAL_MAX_ATTENUATION_DB)


def sector_angles(direction_row: Optional[pd.Series], mode: str) -> tuple[np.ndarray, bool]:
    default = np.asarray([0.0, 2.0 * np.pi / 3.0, -2.0 * np.pi / 3.0])
    if mode == "off" or direction_row is None:
        return default, False
    alpha = pd.to_numeric(direction_row.get("base_alpha_rad", np.nan), errors="coerce")
    if not np.isfinite(alpha):
        return default, False
    order = str(direction_row.get("selected_sector_order", ""))
    sign = 1 if "plus120" in order else -1
    return float(alpha) + np.asarray([0.0, sign * 2.0 * np.pi / 3.0, -sign * 2.0 * np.pi / 3.0]), True


def _unit(v: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    norm = float(np.linalg.norm(v))
    if np.isfinite(norm) and norm > 1e-9:
        return v / norm
    if fallback is not None:
        return _unit(np.asarray(fallback, dtype=float))
    return np.asarray([1.0, 0.0])


def _weighted_centroid(xy: np.ndarray, values: np.ndarray, quantile: float = 0.55) -> np.ndarray:
    xy = np.asarray(xy, float)
    values = np.asarray(values, float)
    if len(xy) == 1:
        return xy[0].copy()
    threshold = float(np.quantile(values, quantile))
    mask = values >= threshold
    if int(mask.sum()) < 2:
        idx = np.argsort(values)[-min(3, len(values)):]
        xsel, vsel = xy[idx], values[idx]
    else:
        xsel, vsel = xy[mask], values[mask]
    weights = np.exp(np.clip((vsel - np.max(vsel)) / 5.0, -8.0, 0.0))
    return np.average(xsel, axis=0, weights=weights)


def _robust_gradient(xy: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, float]:
    xy = np.asarray(xy, float)
    values = np.asarray(values, float)
    if len(values) < 3:
        return np.zeros(2), 0.0
    center = np.mean(xy, axis=0)
    scale_xy = max(float(np.sqrt(np.mean(np.sum((xy - center) ** 2, axis=1)))), 20.0)
    X = np.column_stack([np.ones(len(xy)), (xy - center) / scale_xy])
    y = values.copy()
    w = np.ones(len(y))
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    for _ in range(5):
        residual = y - X @ coef
        scale = max(1.4826 * float(np.median(np.abs(residual - np.median(residual)))), 1.5)
        z = np.abs(residual) / (1.5 * scale)
        w = np.where(z <= 1.0, 1.0, 1.0 / np.maximum(z, 1e-6))
        sw = np.sqrt(w)
        coef = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)[0]
    grad = np.asarray(coef[1:3], float) / scale_xy
    pred = X @ coef
    ss_res = float(np.sum(w * (y - pred) ** 2))
    ybar = float(np.average(y, weights=w))
    ss_tot = float(np.sum(w * (y - ybar) ** 2))
    r2 = 0.0 if ss_tot < 1e-9 else float(np.clip(1.0 - ss_res / ss_tot, 0.0, 1.0))
    return grad, r2


def build_sector_descriptors(selected: pd.DataFrame, angles: np.ndarray, direction_used: bool, omni: bool) -> list[SectorDescriptor]:
    if omni:
        return []
    xy_all = selected[["x_m", "y_m"]].to_numpy(float)
    rss_all = selected[["rsrp_s1", "rsrp_s2", "rsrp_s3"]].to_numpy(float)
    out: list[SectorDescriptor] = []
    for j in range(3):
        mask = np.isfinite(rss_all[:, j])
        if int(mask.sum()) < 2:
            continue
        xy = xy_all[mask]
        values = rss_all[mask, j]
        centroid = _weighted_centroid(xy, values)
        grad, grad_rel = _robust_gradient(xy, values)
        bore = _unit(np.asarray([math.cos(angles[j]), math.sin(angles[j])]))
        grad_dir = _unit(grad, fallback=-bore)
        out.append(SectorDescriptor(
            sector_index=j,
            count=int(len(values)),
            high_centroid=np.asarray(centroid, float),
            boresight=bore,
            back_boresight=-bore if direction_used else grad_dir,
            gradient_direction=grad_dir,
            gradient_reliability=float(grad_rel),
            spatial_spread_m=max(float(np.sqrt(np.mean(np.sum((xy - centroid) ** 2, axis=1)))), 20.0),
            signal_span_db=max(float(np.ptp(values)), 2.0),
        ))
    return out


def _pairwise_line_intersection(p1: np.ndarray, d1: np.ndarray, p2: np.ndarray, d2: np.ndarray) -> np.ndarray | None:
    M = np.column_stack([d1, -d2])
    if abs(float(np.linalg.det(M))) < 0.06:
        return None
    try:
        t = np.linalg.solve(M, p2 - p1)
    except np.linalg.LinAlgError:
        return None
    p = p1 + float(t[0]) * d1
    return p if np.isfinite(p).all() else None


def _robust_line_intersection(desc: list[SectorDescriptor], fallback: np.ndarray) -> np.ndarray:
    if len(desc) < 2:
        return np.asarray(fallback, float)
    A, b = [], []
    for item in desc:
        d = _unit(item.back_boresight)
        n = np.asarray([-d[1], d[0]])
        A.append(n)
        b.append(float(n @ item.high_centroid))
    A = np.asarray(A, float)
    b = np.asarray(b, float)
    try:
        return np.linalg.lstsq(A, b, rcond=None)[0]
    except np.linalg.LinAlgError:
        return np.asarray(fallback, float)


def _clip_candidate(p: np.ndarray, bounds: Sequence[float]) -> np.ndarray:
    x0, x1, y0, y1 = map(float, bounds)
    return np.asarray([np.clip(p[0], x0, x1), np.clip(p[1], y0, y1)], float)


def build_collocated_simulation_dataset(
    project_root: Path,
    simulation_root: Path,
    station: pd.DataFrame,
    selected: pd.DataFrame,
    station_id: int,
    omni: bool,
    *,
    strict: bool = False,
    simulation_sampling: str = "linear",
    fill_missing_with_floor: bool = False,
    simulation_floor_dbm: float = MIN_RSRP_DBM,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Read simulation RSRP at exactly the selected measured receiver locations.

    The joint experiment receives two collocated observation channels: measured
    PCI-RSRP and pure-Sionna PCI-RSRP. It never uses the simulation-map extent,
    peak, center, or transmitter-coordinate metadata to propose a location.
    """
    simulation_selected = selected.copy()
    selected_with_simulation = selected.copy()
    prior_paths: list[str] = []
    missing: list[str] = []
    matched_observation_count = 0
    diagnostics_direct_matches = 0
    diagnostics_floor_fills = 0
    sector_count = 1 if omni else 3
    xy = selected[["x_m", "y_m"]].to_numpy(float)

    for sector_index in range(1, sector_count + 1):
        sector_rows = station.loc[station["sector_index"].eq(sector_index)]
        if sector_rows.empty:
            missing.append(f"sector={sector_index}: no mapped PCI")
            simulation_selected[f"rsrp_s{sector_index}"] = np.nan
            selected_with_simulation[f"simulated_rsrp_s{sector_index}"] = np.nan
            continue
        pci = int(pd.to_numeric(sector_rows["pci"], errors="coerce").dropna().iloc[0])
        try:
            path = discover_simulation_prior(
                simulation_root if simulation_root.exists() else project_root,
                station_id=int(station_id),
                pci=pci,
            )
            prior = load_simulation_prior(path, station_id=int(station_id), pci=pci)
            if str(simulation_sampling).lower() == "nearest":
                simulated = np.asarray(prior.sample_nearest(xy), dtype=float)
            elif str(simulation_sampling).lower() == "linear":
                simulated = np.asarray(prior.sample(xy), dtype=float)
            else:
                raise ValueError(f"Unsupported simulation_sampling={simulation_sampling!r}")
        except Exception as exc:
            message = f"sector={sector_index}, PCI={pci}: {type(exc).__name__}: {exc}"
            if strict:
                raise RuntimeError(f"站{station_id}仿真数据加载失败：{message}") from exc
            missing.append(message)
            simulation_selected[f"rsrp_s{sector_index}"] = np.nan
            selected_with_simulation[f"simulated_rsrp_s{sector_index}"] = np.nan
            continue

        prior_paths.append(str(path))
        direct_valid = np.isfinite(simulated) & (simulated >= MIN_RSRP_DBM) & (simulated <= MAX_RSRP_DBM)
        direct_match_count = int(direct_valid.sum())
        if fill_missing_with_floor:
            # A missing/invalid Sionna sample is a simulation-side no-path /
            # below-analysis-floor observation, not a missing measurement.  Keep
            # the exact same measured receiver locations in both branches and
            # encode the unavailable simulated power conservatively at the
            # configured lower RSRP floor.  No measured RSRP or ground truth is
            # used to create these values.
            floor = float(np.clip(simulation_floor_dbm, MIN_RSRP_DBM, MAX_RSRP_DBM))
            simulated = np.where(direct_valid, simulated, floor)
        else:
            simulated = np.where(direct_valid, simulated, np.nan)
        simulation_selected[f"rsrp_s{sector_index}"] = simulated
        selected_with_simulation[f"simulated_rsrp_s{sector_index}"] = simulated
        matched_observation_count += int(np.isfinite(simulated).sum())
        diagnostics_direct_matches += direct_match_count
        diagnostics_floor_fills += int((~direct_valid).sum() if fill_missing_with_floor else 0)

    for sector_index in range(sector_count + 1, 4):
        simulation_selected[f"rsrp_s{sector_index}"] = np.nan
        selected_with_simulation[f"simulated_rsrp_s{sector_index}"] = np.nan

    simulation_selected["observed_sector_count"] = simulation_selected[
        ["rsrp_s1", "rsrp_s2", "rsrp_s3"]
    ].notna().sum(axis=1)
    matched_location_count = int((simulation_selected["observed_sector_count"] > 0).sum())
    diagnostics = {
        "simulation_prior_count": int(len(prior_paths)),
        "simulation_matched_location_count": matched_location_count,
        "simulation_matched_observation_count": int(matched_observation_count),
        "simulation_prior_paths": prior_paths,
        "simulation_missing_inputs": missing,
        "simulation_sampling": f"{simulation_sampling} collocation at the selected measured receiver locations",
        "simulation_direct_valid_observation_count": int(diagnostics_direct_matches),
        "simulation_floor_filled_observation_count": int(diagnostics_floor_fills),
        "simulation_missing_value_policy": (
            f"simulation-only lower censoring at {float(np.clip(simulation_floor_dbm, MIN_RSRP_DBM, MAX_RSRP_DBM)):.1f} dBm"
            if fill_missing_with_floor else "keep missing as NaN"
        ),
        "simulation_map_geometry_used_for_candidate_location": False,
    }
    return simulation_selected, selected_with_simulation, diagnostics

def generate_candidates(selected: pd.DataFrame, angles: np.ndarray, direction_used: bool, omni: bool, bounds: Sequence[float]) -> list[tuple[str, np.ndarray]]:
    """Generate diverse candidate positions without using ground truth."""
    xy = selected[["x_m", "y_m"]].to_numpy(float)
    rss = selected[["rsrp_s1", "rsrp_s2", "rsrp_s3"]].to_numpy(float)
    finite_rss = np.where(np.isfinite(rss), rss, -np.inf)
    envelope = np.max(finite_rss, axis=1)
    envelope[~np.isfinite(envelope)] = np.nan
    centroid = np.mean(xy, axis=0)
    strong = _weighted_centroid(xy, envelope, quantile=0.60)
    spread = max(common.point_spread(selected), 80.0)
    raw: list[tuple[str, np.ndarray]] = [("measurement_centroid", centroid), ("strong_rsrp_centroid", strong)]

    if omni:
        # For an omni site, create a symmetric candidate cloud around the high-RSRP center.
        radii = [60.0, 120.0, 220.0, min(360.0, 1.25 * spread)]
        for radius in radii:
            for angle in np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False):
                raw.append((f"omni_ring_{int(radius)}", strong + radius * np.asarray([math.cos(angle), math.sin(angle)])))
    else:
        desc = build_sector_descriptors(selected, angles, direction_used, omni=False)
        intersection = _robust_line_intersection(desc, centroid)
        raw.append(("robust_boresight_intersection", intersection))

        # Candidate distances scale with the measured geometry rather than using one hard-coded range.
        base_distances = sorted(set([70.0, 140.0, 240.0, min(420.0, max(280.0, 1.15 * spread))]))
        for item in desc:
            for dist in base_distances:
                raw.append((f"sector{item.sector_index+1}_back_{int(dist)}", item.high_centroid + dist * item.back_boresight))
            if item.gradient_reliability >= 0.20:
                for dist in [80.0, min(220.0, max(120.0, 0.65 * spread))]:
                    raw.append((f"sector{item.sector_index+1}_gradient_{int(dist)}", item.high_centroid + dist * item.gradient_direction))
        for a in range(len(desc)):
            for b in range(a + 1, len(desc)):
                p = _pairwise_line_intersection(desc[a].high_centroid, desc[a].back_boresight, desc[b].high_centroid, desc[b].back_boresight)
                if p is not None:
                    raw.append((f"pair_{desc[a].sector_index+1}_{desc[b].sector_index+1}_intersection", p))

    # Dedupe close candidates after clipping to the global map bounds.
    unique: list[tuple[str, np.ndarray]] = []
    for label, p in raw:
        p = _clip_candidate(np.asarray(p, float), bounds)
        if not np.isfinite(p).all():
            continue
        if all(float(np.linalg.norm(p - q)) >= 22.0 for _, q in unique):
            unique.append((label, p))
    return unique


def _validation_rows(tx: np.ndarray, selected: pd.DataFrame, angles: np.ndarray, direction_used: bool, omni: bool):
    xy = selected[["x_m", "y_m"]].to_numpy(float)
    rss = selected[["rsrp_s1", "rsrp_s2", "rsrp_s3"]].to_numpy(float)
    rows_X: list[list[float]] = []
    rows_y: list[float] = []
    location_ids: list[int] = []
    sector_ids: list[int] = []
    d = np.sqrt(np.sum((xy - tx[None, :]) ** 2, axis=1) + VERTICAL_SEPARATION_M ** 2)
    bearing = np.arctan2(xy[:, 1] - tx[1], xy[:, 0] - tx[0])
    sector_count = 1 if omni else 3
    for i in range(len(selected)):
        for j in range(sector_count):
            y = rss[i, j]
            if not np.isfinite(y):
                continue
            # Common intercept + sector-2/3 offsets + distance slope + gain coefficient.
            if omni:
                feat = [1.0, math.log10(max(d[i], 1.0))]
            else:
                gain = float(sector_gain_db(np.asarray([bearing[i] - angles[j]]))[0]) if direction_used else 0.0
                feat = [1.0, 1.0 if j == 1 else 0.0, 1.0 if j == 2 else 0.0, math.log10(max(d[i], 1.0))]
                if direction_used:
                    feat.append(gain)
            rows_X.append(feat)
            rows_y.append(float(y))
            location_ids.append(i)
            sector_ids.append(j)
    return np.asarray(rows_X, float), np.asarray(rows_y, float), np.asarray(location_ids, int), np.asarray(sector_ids, int)


def _ridge_fit(X: np.ndarray, y: np.ndarray, omni: bool, direction_used: bool) -> np.ndarray:
    p = X.shape[1]
    prior = np.zeros(p, float)
    penalty = np.zeros(p, float)
    if omni:
        prior[-1] = -27.0
        penalty[-1] = 3.5
    else:
        # sector offsets weakly shrink to zero; distance slope and gain have physical priors.
        penalty[1:3] = 0.35
        prior[-1 if not direction_used else -2] = -27.0
        penalty[-1 if not direction_used else -2] = 3.5
        if direction_used:
            prior[-1] = 1.0
            penalty[-1] = 2.5
    # tiny ridge on global intercept for numerical stability only.
    penalty[0] = 1e-7
    A = X.T @ X + np.diag(penalty)
    rhs = X.T @ y + penalty * prior
    try:
        return np.linalg.solve(A, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(A, rhs, rcond=None)[0]


def lolo_support_diagnostics(selected: pd.DataFrame, direction_used: bool, omni: bool) -> dict:
    """Check that a channel supports at least two valid leave-one-location-out folds."""
    sector_count = 1 if omni else 3
    rss = selected[["rsrp_s1", "rsrp_s2", "rsrp_s3"]].to_numpy(float)[:, :sector_count]
    observations_per_location = np.isfinite(rss).sum(axis=1).astype(int)
    total_observations = int(observations_per_location.sum())
    distinct_locations = int(np.sum(observations_per_location > 0))
    feature_count = 2 if omni else (5 if direction_used else 4)
    minimum_training_observations = int(feature_count + 1)
    minimum_total_observations = int(max(5, minimum_training_observations))
    valid_fold_count = int(np.sum(
        (observations_per_location > 0)
        & ((total_observations - observations_per_location) >= minimum_training_observations)
    ))
    usable = bool(
        total_observations >= minimum_total_observations
        and distinct_locations >= 2
        and valid_fold_count >= 2
    )
    return {
        "simulation_lolo_usable": usable,
        "simulation_lolo_valid_fold_count": valid_fold_count,
        "simulation_lolo_distinct_location_count": distinct_locations,
        "simulation_lolo_minimum_training_observations": minimum_training_observations,
    }


def _lolo_cv_score(tx: np.ndarray, selected: pd.DataFrame, angles: np.ndarray, direction_used: bool, omni: bool) -> dict:
    """Leave-one-location-out prediction score for one candidate position."""
    X, y, loc_ids, _ = _validation_rows(tx, selected, angles, direction_used, omni)
    if len(y) < max(5, X.shape[1] + 1):
        return {"cv_rmse_db": 1e6, "cv_mae_db": 1e6, "cv_bias_db": np.nan, "fit_rmse_db": np.nan,
                "pathloss_exponent": np.nan, "gain_scale": np.nan, "parameter_boundary_penalty": 10.0}
    residuals: list[float] = []
    unique_locs = np.unique(loc_ids)
    for loc in unique_locs:
        train = loc_ids != loc
        test = loc_ids == loc
        if int(train.sum()) < X.shape[1] + 1 or int(test.sum()) == 0:
            continue
        beta = _ridge_fit(X[train], y[train], omni, direction_used)
        pred = X[test] @ beta
        residuals.extend((pred - y[test]).tolist())
    if not residuals:
        return {"cv_rmse_db": 1e6, "cv_mae_db": 1e6, "cv_bias_db": np.nan, "fit_rmse_db": np.nan,
                "pathloss_exponent": np.nan, "gain_scale": np.nan, "parameter_boundary_penalty": 10.0}
    residuals_arr = np.asarray(residuals, float)
    beta = _ridge_fit(X, y, omni, direction_used)
    fit_res = X @ beta - y
    slope = float(beta[-1] if omni else beta[-1 if not direction_used else -2])
    n = -slope / 10.0
    gain_scale = float(beta[-1]) if (not omni and direction_used) else np.nan
    # Soft diagnostic penalty: values outside broad plausible ranges indicate that the candidate
    # can explain the observations only through an implausible validation model.
    penalty = 0.0
    if n < 1.2:
        penalty += (1.2 - n) * 2.0
    elif n > 5.5:
        penalty += (n - 5.5) * 2.0
    if np.isfinite(gain_scale):
        if gain_scale < 0.15:
            penalty += (0.15 - gain_scale) * 3.0
        elif gain_scale > 1.8:
            penalty += (gain_scale - 1.8) * 3.0
    return {
        "cv_rmse_db": float(np.sqrt(np.mean(residuals_arr ** 2))),
        "cv_mae_db": float(np.mean(np.abs(residuals_arr))),
        "cv_bias_db": float(np.mean(residuals_arr)),
        "fit_rmse_db": float(np.sqrt(np.mean(fit_res ** 2))),
        "pathloss_exponent": float(n),
        "gain_scale": gain_scale,
        "parameter_boundary_penalty": float(penalty),
    }


def _direction_alignment_penalty(tx: np.ndarray, selected: pd.DataFrame, angles: np.ndarray, direction_used: bool, omni: bool) -> tuple[float, float]:
    """Direction consistency from *dominant-sector* measurement regions only.

    A sector that is deeply attenuated at every sampled location provides almost no
    information about its boresight direction.  Using its high-value centroid can
    bias one-sided road measurements, so only locations where the sector is within
    4 dB of the strongest observed sector are allowed to contribute.
    """
    if omni or not direction_used:
        return 0.0, float("nan")
    rss = selected[["rsrp_s1", "rsrp_s2", "rsrp_s3"]].to_numpy(float)
    xy = selected[["x_m", "y_m"]].to_numpy(float)
    finite_rss = np.where(np.isfinite(rss), rss, -np.inf)
    envelope = np.max(finite_rss, axis=1)
    envelope[~np.isfinite(envelope)] = np.nan
    errors: list[float] = []
    weights: list[float] = []
    for j in range(3):
        mask = np.isfinite(rss[:, j]) & np.isfinite(envelope) & (rss[:, j] >= envelope - 4.0)
        if int(mask.sum()) < 2:
            continue
        high = _weighted_centroid(xy[mask], rss[mask, j])
        bearing = math.atan2(high[1] - tx[1], high[0] - tx[0])
        err = abs(float(np.degrees(wrap(bearing - angles[j]))))
        errors.append(err)
        weights.append(min(1.0, float(mask.sum()) / 4.0))
    if not errors:
        return 0.0, float("nan")
    mean_err = float(np.average(np.asarray(errors), weights=np.asarray(weights)))
    # Direction is an auxiliary discriminator; LOLO RSRP prediction remains dominant.
    return 0.022 * mean_err, mean_err


def score_candidate(
    tx: np.ndarray,
    selected: pd.DataFrame,
    angles: np.ndarray,
    direction_used: bool,
    omni: bool,
    simulation_selected: pd.DataFrame | None = None,
    simulation_weight: float = 0.0,
) -> dict:
    measured_cv = _lolo_cv_score(tx, selected, angles, direction_used, omni)
    measured_direction_penalty, measured_direction_error = _direction_alignment_penalty(
        tx, selected, angles, direction_used, omni
    )
    xy = selected[["x_m", "y_m"]].to_numpy(float)
    centroid = np.mean(xy, axis=0)
    spread = max(common.point_spread(selected), 80.0)
    dist_from_cloud = float(np.linalg.norm(tx - centroid))
    soft_limit = 2.4 * spread + 120.0
    extrap_penalty = 0.0 if dist_from_cloud <= soft_limit else 0.004 * (dist_from_cloud - soft_limit)
    measured_total = float(
        measured_cv["cv_rmse_db"]
        + measured_cv["parameter_boundary_penalty"]
        + measured_direction_penalty
        + extrap_penalty
    )

    if simulation_selected is None or float(simulation_weight) <= 0.0:
        return {
            **measured_cv,
            "measured_cv_rmse_db": float(measured_cv["cv_rmse_db"]),
            "simulation_cv_rmse_db": float("nan"),
            "joint_cv_rmse_db": float(measured_cv["cv_rmse_db"]),
            "direction_alignment_error_deg": measured_direction_error,
            "direction_penalty": float(measured_direction_penalty),
            "extrapolation_penalty": float(extrap_penalty),
            "candidate_score": measured_total,
        }

    simulation_cv = _lolo_cv_score(tx, simulation_selected, angles, direction_used, omni)
    simulation_direction_penalty, simulation_direction_error = _direction_alignment_penalty(
        tx, simulation_selected, angles, direction_used, omni
    )
    simulation_total = float(
        simulation_cv["cv_rmse_db"]
        + simulation_cv["parameter_boundary_penalty"]
        + simulation_direction_penalty
        + extrap_penalty
    )
    weight = float(np.clip(simulation_weight, 0.0, 1.0))
    joint_cv_rmse = (1.0 - weight) * float(measured_cv["cv_rmse_db"]) + weight * float(simulation_cv["cv_rmse_db"])
    total = (1.0 - weight) * measured_total + weight * simulation_total
    return {
        **measured_cv,
        "cv_rmse_db": float(joint_cv_rmse),
        "measured_cv_rmse_db": float(measured_cv["cv_rmse_db"]),
        "simulation_cv_rmse_db": float(simulation_cv["cv_rmse_db"]),
        "joint_cv_rmse_db": float(joint_cv_rmse),
        "simulation_direction_alignment_error_deg": float(simulation_direction_error),
        "direction_alignment_error_deg": float(measured_direction_error),
        "direction_penalty": float((1.0 - weight) * measured_direction_penalty + weight * simulation_direction_penalty),
        "extrapolation_penalty": float(extrap_penalty),
        "candidate_score": float(total),
    }


def _dedupe_labeled(candidates: list[tuple[str, np.ndarray]], min_sep: float = 12.0) -> list[tuple[str, np.ndarray]]:
    out: list[tuple[str, np.ndarray]] = []
    for label, p in candidates:
        p = np.asarray(p, float)
        if np.isfinite(p).all() and all(float(np.linalg.norm(p - q)) >= min_sep for _, q in out):
            out.append((label, p))
    return out


def solve(
    selected: pd.DataFrame,
    angles: np.ndarray,
    direction_used: bool,
    omni: bool,
    bounds: Sequence[float],
    previous_xy: np.ndarray | None = None,
    progressive_min_improvement_db: float = 0.25,
    simulation_selected: pd.DataFrame | None = None,
    simulation_weight: float = 0.0,
) -> dict:
    candidates = generate_candidates(selected, angles, direction_used, omni, bounds)
    if simulation_selected is not None:
        simulation_candidates = generate_candidates(simulation_selected, angles, direction_used, omni, bounds)
        candidates.extend((f"simulation_channel_{label}", np.asarray(xy, float)) for label, xy in simulation_candidates)
    if previous_xy is not None and np.isfinite(np.asarray(previous_xy, float)).all():
        prev = _clip_candidate(np.asarray(previous_xy, float), bounds)
        candidates = [("previous_count_solution", prev)] + candidates
        # Small local neighborhood around the inherited solution lets the new point
        # refine rather than forcing an all-or-nothing jump.
        for radius in (24.0, 55.0):
            for dx, dy in ((radius, 0.0), (-radius, 0.0), (0.0, radius), (0.0, -radius)):
                candidates.append((f"previous_local_{int(radius)}_{int(dx)}_{int(dy)}", _clip_candidate(prev + np.asarray([dx, dy]), bounds)))
    if not candidates:
        candidates = [("measurement_centroid", np.mean(selected[["x_m", "y_m"]].to_numpy(float), axis=0))]
    candidates = _dedupe_labeled(candidates, min_sep=6.0)

    evaluate = lambda point: score_candidate(
        point, selected, angles, direction_used, omni,
        simulation_selected=simulation_selected, simulation_weight=simulation_weight,
    )
    scored: list[tuple[str, np.ndarray, dict]] = []
    for label, p in candidates:
        scored.append((label, p, evaluate(p)))
    scored.sort(key=lambda item: item[2]["candidate_score"])

    # Local two-dimensional refinement around the best candidates.  This is a grid search,
    # not a propagation inversion: every refined point is still selected by LOLO prediction.
    top = scored[: min(3, len(scored))]
    refined: list[tuple[str, np.ndarray]] = [(label, p) for label, p, _ in scored]
    for rank, (label, center, _) in enumerate(top, start=1):
        current = center.copy()
        for radius in (70.0, 28.0):
            neighborhood = []
            for dx in (-radius, 0.0, radius):
                for dy in (-radius, 0.0, radius):
                    neighborhood.append((f"{label}_ref{rank}_{int(radius)}_{int(dx)}_{int(dy)}", _clip_candidate(current + np.asarray([dx, dy]), bounds)))
            local_scored = [(lab, p, evaluate(p)) for lab, p in _dedupe_labeled(neighborhood, min_sep=3.0)]
            local_scored.sort(key=lambda item: item[2]["candidate_score"])
            current = local_scored[0][1]
            refined.extend([(lab, p) for lab, p, _ in local_scored])

    final_candidates = _dedupe_labeled(refined, min_sep=6.0)
    final_scored = [(label, p, evaluate(p)) for label, p in final_candidates]
    final_scored.sort(key=lambda item: item[2]["candidate_score"])
    unconstrained_label, unconstrained_xy, unconstrained_best = final_scored[0]

    previous_score = float("nan")
    progressive_update_accepted = True
    best_label, best_xy, best = unconstrained_label, unconstrained_xy, unconstrained_best
    if previous_xy is not None and np.isfinite(np.asarray(previous_xy, float)).all():
        prev = _clip_candidate(np.asarray(previous_xy, float), bounds)
        prev_eval = evaluate(prev)
        previous_score = float(prev_eval["candidate_score"])
        improvement = previous_score - float(unconstrained_best["candidate_score"])
        if improvement < float(progressive_min_improvement_db):
            # Hysteresis: an extra point must provide clear evidence under the active input channels
            # before the estimate is allowed to move away from the previous solution.
            best_label, best_xy, best = "previous_count_solution_retained", prev, prev_eval
            progressive_update_accepted = False

    second_scores = [float(item[2]["candidate_score"]) for item in final_scored if float(np.linalg.norm(item[1] - best_xy)) > 1e-6]
    second_score = min(second_scores) if second_scores else float("nan")
    top_positions = np.vstack([item[1] for item in final_scored[: min(5, len(final_scored))]])
    if len(top_positions) >= 2:
        pair = np.linalg.norm(top_positions[:, None, :] - top_positions[None, :, :], axis=2)
        top_spread = float(np.max(pair))
    else:
        top_spread = 0.0

    return {
        "final_xy": np.asarray(best_xy, float),
        "objective": float(best["candidate_score"]),
        "solver_mode": "progressive_nested_mcvl" if previous_xy is not None else "multi_candidate_lolo_cross_validated",
        "candidate_count": int(len(final_scored)),
        "previous_candidate_score": previous_score,
        "unconstrained_candidate_score": float(unconstrained_best["candidate_score"]),
        "progressive_update_accepted": bool(progressive_update_accepted),
        "progressive_score_improvement_db": float(previous_score - unconstrained_best["candidate_score"]) if np.isfinite(previous_score) else float("nan"),
        "progressive_min_improvement_db": float(progressive_min_improvement_db),
        "selected_candidate_label": str(best_label),
        "cv_rmse_db": float(best["cv_rmse_db"]),
        "measured_cv_rmse_db": float(best.get("measured_cv_rmse_db", best["cv_rmse_db"])),
        "simulation_cv_rmse_db": float(best.get("simulation_cv_rmse_db", np.nan)),
        "joint_cv_rmse_db": float(best.get("joint_cv_rmse_db", best["cv_rmse_db"])),
        "simulation_weight": float(np.clip(simulation_weight, 0.0, 1.0)) if simulation_selected is not None else 0.0,
        "cv_mae_db": float(best["cv_mae_db"]),
        "cv_bias_db": float(best["cv_bias_db"]),
        "validation_fit_rmse_db": float(best["fit_rmse_db"]),
        "validation_pathloss_exponent": float(best["pathloss_exponent"]),
        "validation_gain_scale": float(best["gain_scale"]),
        "direction_alignment_error_deg": float(best["direction_alignment_error_deg"]),
        "candidate_score_gap": float(second_score - best["candidate_score"]) if np.isfinite(second_score) else float("nan"),
        "top5_candidate_spread_m": top_spread,
        "candidate_table": final_scored,
    }


def quality_flag(solution: dict, selected: pd.DataFrame, direction_used: bool, omni: bool) -> str:
    flags: list[str] = []
    if common.point_spread(selected) < 60.0:
        flags.append("low_point_spread")
    cv = float(solution.get("cv_rmse_db", np.nan))
    if np.isfinite(cv) and cv > 12.0:
        flags.append("high_cross_validated_rsrp_error")
    gap = float(solution.get("candidate_score_gap", np.nan))
    if np.isfinite(gap) and gap < 0.35:
        flags.append("ambiguous_candidate_score")
    spread = float(solution.get("top5_candidate_spread_m", np.nan))
    if np.isfinite(spread) and spread > 220.0:
        flags.append("multi_candidate_position_ambiguity")
    n = float(solution.get("validation_pathloss_exponent", np.nan))
    if np.isfinite(n) and (n < 1.2 or n > 5.5):
        flags.append("validation_model_boundary")
    if (not omni) and not direction_used:
        flags.append("no_external_direction_prior")
    return ";".join(flags) if flags else "ok"


def build_result_row(
    *,
    station_id: int,
    truth_row: pd.Series,
    selected: pd.DataFrame,
    solution: dict,
    direction_used: bool,
    omni: bool,
    previous_xy: np.ndarray | None,
    elapsed_s: float,
    simulation_data_used: bool,
    simulation_diagnostics: dict,
) -> dict:
    predicted = np.asarray(solution["final_xy"], float)
    true_xy = np.asarray([truth_row.true_x_m, truth_row.true_y_m], float)
    delta = predicted - true_xy
    observations = common.observations_from_points(selected)
    return {
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
        "horizontal_error_m": float(np.linalg.norm(delta)),
        "objective_value": float(solution["objective"]),
        "direction_prior_used": bool(direction_used),
        "point_spread_m": float(common.point_spread(selected)),
        "quality_flag": quality_flag(solution, selected, direction_used, omni),
        "solver_mode": str(solution["solver_mode"]),
        "candidate_count": int(solution.get("candidate_count", 0)),
        "selected_candidate_label": str(solution.get("selected_candidate_label", "")),
        "cv_rmse_db": float(solution.get("cv_rmse_db", np.nan)),
        "measured_cv_rmse_db": float(solution.get("measured_cv_rmse_db", solution.get("cv_rmse_db", np.nan))),
        "simulation_cv_rmse_db": float(solution.get("simulation_cv_rmse_db", np.nan)),
        "joint_cv_rmse_db": float(solution.get("joint_cv_rmse_db", solution.get("cv_rmse_db", np.nan))),
        "simulation_weight": float(solution.get("simulation_weight", 0.0)),
        "cv_mae_db": float(solution.get("cv_mae_db", np.nan)),
        "cv_bias_db": float(solution.get("cv_bias_db", np.nan)),
        "validation_fit_rmse_db": float(solution.get("validation_fit_rmse_db", np.nan)),
        "validation_pathloss_exponent": float(solution.get("validation_pathloss_exponent", np.nan)),
        "validation_gain_scale": float(solution.get("validation_gain_scale", np.nan)),
        "direction_alignment_error_deg": float(solution.get("direction_alignment_error_deg", np.nan)),
        "candidate_score_gap": float(solution.get("candidate_score_gap", np.nan)),
        "top5_candidate_spread_m": float(solution.get("top5_candidate_spread_m", np.nan)),
        "previous_solution_available": bool(previous_xy is not None),
        "previous_predicted_x_m": float(previous_xy[0]) if previous_xy is not None else float("nan"),
        "previous_predicted_y_m": float(previous_xy[1]) if previous_xy is not None else float("nan"),
        "progressive_jump_m": float(np.linalg.norm(predicted - previous_xy)) if previous_xy is not None else float("nan"),
        "previous_candidate_score": float(solution.get("previous_candidate_score", np.nan)),
        "unconstrained_candidate_score": float(solution.get("unconstrained_candidate_score", np.nan)),
        "progressive_update_accepted": bool(solution.get("progressive_update_accepted", True)),
        "progressive_score_improvement_db": float(solution.get("progressive_score_improvement_db", np.nan)),
        "elapsed_s": float(elapsed_s),
        "rsrp_min_dbm": MIN_RSRP_DBM,
        "rsrp_max_dbm": MAX_RSRP_DBM,
        "simulation_data_used": bool(simulation_data_used),
        "simulation_prior_count": int(simulation_diagnostics.get("simulation_prior_count", 0)),
        "simulation_matched_location_count": int(simulation_diagnostics.get("simulation_matched_location_count", 0)),
        "simulation_matched_observation_count": int(simulation_diagnostics.get("simulation_matched_observation_count", 0)),
        "simulation_lolo_usable": bool(simulation_diagnostics.get("simulation_lolo_usable", False)),
        "simulation_lolo_valid_fold_count": int(simulation_diagnostics.get("simulation_lolo_valid_fold_count", 0)),
        "simulation_lolo_distinct_location_count": int(simulation_diagnostics.get("simulation_lolo_distinct_location_count", 0)),
        "simulation_lolo_minimum_training_observations": int(simulation_diagnostics.get("simulation_lolo_minimum_training_observations", 0)),
        "simulation_prior_paths": "|".join(simulation_diagnostics.get("simulation_prior_paths", [])),
        "simulation_missing_inputs": "|".join(simulation_diagnostics.get("simulation_missing_inputs", [])),
    }


def summarize_simulation_ablation(without: pd.DataFrame, with_simulation: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = without[["station_id", "horizontal_error_m", "cv_rmse_db", "predicted_x_m", "predicted_y_m"]].copy()
    assisted = with_simulation[["station_id", "horizontal_error_m", "cv_rmse_db", "predicted_x_m", "predicted_y_m"]].copy()
    paired = base.merge(assisted, on="station_id", suffixes=("_without_simulation", "_with_simulation"))
    paired["error_delta_with_minus_without_m"] = (
        paired["horizontal_error_m_with_simulation"] - paired["horizontal_error_m_without_simulation"]
    )
    paired["simulation_improved_localization"] = paired["error_delta_with_minus_without_m"] < 0.0
    paired["estimate_shift_m"] = np.hypot(
        paired["predicted_x_m_with_simulation"] - paired["predicted_x_m_without_simulation"],
        paired["predicted_y_m_with_simulation"] - paired["predicted_y_m_without_simulation"],
    )

    summary_rows = []
    for label, frame in (("without_simulation", without), ("with_simulation", with_simulation)):
        errors = frame["horizontal_error_m"].to_numpy(float)
        summary_rows.append({
            "variant": label,
            "station_count": int(len(errors)),
            "mean_error_m": float(np.mean(errors)),
            "median_error_m": float(np.median(errors)),
            "rmse_m": float(np.sqrt(np.mean(errors ** 2))),
            "p90_error_m": float(np.percentile(errors, 90)),
            "within_50m_percent": float(np.mean(errors <= 50.0) * 100.0),
            "within_100m_percent": float(np.mean(errors <= 100.0) * 100.0),
        })
    summary = pd.DataFrame(summary_rows)
    summary["improved_station_count"] = int(paired["simulation_improved_localization"].sum())
    summary["mean_error_delta_with_minus_without_m"] = float(paired["error_delta_with_minus_without_m"].mean())
    return paired, summary


def plot_station(selected: pd.DataFrame, pred: np.ndarray, truth: np.ndarray, station_id: int, out: Path, solution: dict | None = None) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.4), dpi=180)
    ax.scatter(selected.x_m, selected.y_m, s=38, facecolors="none", edgecolors="black", label="Random measured points")
    if solution is not None:
        table = solution.get("candidate_table", [])
        if table:
            cxy = np.vstack([np.asarray(item[1], float) for item in table[: min(12, len(table))]])
            ax.scatter(cxy[:, 0], cxy[:, 1], s=20, c="0.65", alpha=0.55, label="Candidate positions")
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
    previous_predictions = load_previous_predictions(args.previous_results)
    previous_without_simulation = (
        load_previous_predictions(args.previous_without_simulation_results)
        if args.previous_without_simulation_results is not None
        else previous_predictions
    )
    previous_with_simulation = (
        load_previous_predictions(args.previous_with_simulation_results)
        if args.previous_with_simulation_results is not None
        else previous_predictions
    )
    simulation_root = (
        args.simulation_root.expanduser().resolve()
        if args.simulation_root is not None
        else project / "outputs" / "bestparam_dem_vs_zplane_512m"
    )
    if args.simulation_root is not None and not simulation_root.is_dir():
        raise FileNotFoundError(f"--simulation-root不存在或不是目录：{simulation_root}")
    station_ids = common.parse_station_ids(args.station_ids, sorted(localization.station_id.unique().astype(int)))
    truth_index = truth.set_index("station_id")
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else project / "outputs" / f"localization_mcvl_{args.points_per_station}points_seed_{args.random_seed}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    per_station = output_dir / "per_station"
    per_station.mkdir(exist_ok=True)

    rows: list[dict] = []
    without_simulation_rows: list[dict] = []
    with_simulation_rows: list[dict] = []
    selected_rows: list[pd.DataFrame] = []
    for station_id in station_ids:
        station = localization[localization.station_id.eq(station_id)].copy()
        truth_row = truth_index.loc[station_id]
        omni = bool(int(truth_row.is_omnidirectional)) or station_id == 22
        points = common.point_table(station)
        selected = random_points(points, args.points_per_station, args.random_seed + station_id * 7919)
        direction_row = directions.loc[station_id] if station_id in directions.index else None
        angles, direction_used = sector_angles(direction_row, args.direction_prior_mode)
        bounds = (args.x_min, args.x_max, args.y_min, args.y_max)
        previous_without_xy = previous_without_simulation.get(int(station_id))
        previous_with_xy = previous_with_simulation.get(int(station_id))
        empty_simulation_diagnostics = {
            "simulation_prior_count": 0,
            "simulation_matched_location_count": 0,
            "simulation_matched_observation_count": 0,
            "simulation_prior_paths": [],
            "simulation_missing_inputs": [],
            "simulation_lolo_usable": False,
            "simulation_lolo_valid_fold_count": 0,
            "simulation_lolo_distinct_location_count": 0,
            "simulation_lolo_minimum_training_observations": 0,
        }
        simulation_selected = None
        selected_with_simulation = selected.copy()
        simulation_diagnostics = dict(empty_simulation_diagnostics)
        if args.simulation_mode in {"with", "compare"}:
            simulation_selected_raw, selected_with_simulation, simulation_diagnostics = build_collocated_simulation_dataset(
                project_root=project,
                simulation_root=simulation_root,
                station=station,
                selected=selected,
                station_id=int(station_id),
                omni=omni,
                strict=bool(args.strict_simulation_data),
            )
            minimum_simulation_observations = 5 if omni else 7
            support_diagnostics = lolo_support_diagnostics(simulation_selected_raw, direction_used, omni)
            simulation_diagnostics.update(support_diagnostics)
            if (
                int(simulation_diagnostics["simulation_matched_observation_count"]) >= minimum_simulation_observations
                and bool(simulation_diagnostics["simulation_lolo_usable"])
            ):
                simulation_selected = simulation_selected_raw
            elif args.strict_simulation_data:
                raise RuntimeError(
                    f"站{station_id}共址仿真观测不足："
                    f"matched={simulation_diagnostics['simulation_matched_observation_count']}, "
                    f"valid_LOLO_folds={simulation_diagnostics['simulation_lolo_valid_fold_count']}"
                )

        baseline_solution = None
        baseline_elapsed = float("nan")
        if args.simulation_mode in {"without", "compare"}:
            start = time.time()
            baseline_solution = solve(
                selected, angles, direction_used, omni, bounds,
                previous_xy=previous_without_xy,
                progressive_min_improvement_db=float(args.progressive_min_improvement_db),
            )
            baseline_elapsed = time.time() - start

        assisted_solution = None
        assisted_elapsed = float("nan")
        if args.simulation_mode in {"with", "compare"}:
            start = time.time()
            assisted_solution = solve(
                selected, angles, direction_used, omni, bounds,
                previous_xy=previous_with_xy,
                progressive_min_improvement_db=float(args.progressive_min_improvement_db),
                simulation_selected=simulation_selected,
                simulation_weight=float(args.simulation_weight),
            )
            assisted_elapsed = time.time() - start

        if baseline_solution is not None:
            baseline_row = build_result_row(
                station_id=int(station_id), truth_row=truth_row, selected=selected,
                solution=baseline_solution, direction_used=direction_used, omni=omni,
                previous_xy=previous_without_xy, elapsed_s=baseline_elapsed,
                simulation_data_used=False,
                simulation_diagnostics=empty_simulation_diagnostics,
            )
            without_simulation_rows.append(baseline_row)
        if assisted_solution is not None:
            assisted_row = build_result_row(
                station_id=int(station_id), truth_row=truth_row, selected=selected,
                solution=assisted_solution, direction_used=direction_used, omni=omni,
                previous_xy=previous_with_xy, elapsed_s=assisted_elapsed,
                simulation_data_used=simulation_selected is not None,
                simulation_diagnostics=simulation_diagnostics,
            )
            with_simulation_rows.append(assisted_row)

        if args.simulation_mode == "without":
            solution = baseline_solution
            row = baseline_row
        else:
            solution = assisted_solution
            row = assisted_row
        assert solution is not None
        predicted = np.asarray(solution["final_xy"], float)
        true_xy = np.asarray([truth_row.true_x_m, truth_row.true_y_m], float)
        observations = common.observations_from_points(selected)
        rows.append(row)
        selected_out = selected_with_simulation.copy()
        selected_out.insert(0, "station_id", station_id)
        selected_out["data_source"] = "measurement_with_collocated_simulation" if simulation_selected is not None else "measurement_only"
        selected_rows.append(selected_out)
        if not args.skip_figures and not args.skip_per_station_figures:
            plot_station(selected, predicted, true_xy, station_id, per_station / f"station_{station_id:02d}_localization.png", solution=solution)
        print(
            f"Station {station_id:02d}: points={len(selected)}, obs={len(observations)}, "
            f"estimate=({predicted[0]:.2f},{predicted[1]:.2f}), error={row['horizontal_error_m']:.2f} m, "
            f"CV={solution.get('cv_rmse_db', np.nan):.2f} dB, quality={row['quality_flag']}, "
            f"simulation_observations={simulation_diagnostics.get('simulation_matched_observation_count', 0)}"
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
        "median_cv_rmse_db": float(pd.to_numeric(results.cv_rmse_db, errors="coerce").median()),
        "median_candidate_score_gap": float(pd.to_numeric(results.candidate_score_gap, errors="coerce").median()),
    }])
    summary.to_csv(output_dir / "localization_accuracy_summary.csv", index=False, encoding="utf-8-sig")

    if args.simulation_mode == "compare":
        without_frame = pd.DataFrame(without_simulation_rows).sort_values("station_id").reset_index(drop=True)
        with_frame = pd.DataFrame(with_simulation_rows).sort_values("station_id").reset_index(drop=True)
        without_frame.to_csv(output_dir / "localization_without_simulation_results.csv", index=False, encoding="utf-8-sig")
        with_frame.to_csv(output_dir / "localization_with_simulation_results.csv", index=False, encoding="utf-8-sig")
        paired, ablation_summary = summarize_simulation_ablation(without_frame, with_frame)
        paired.to_csv(output_dir / "localization_simulation_ablation_by_station.csv", index=False, encoding="utf-8-sig")
        ablation_summary.to_csv(output_dir / "localization_simulation_ablation_summary.csv", index=False, encoding="utf-8-sig")
        if not args.skip_figures:
            fig, ax = plt.subplots(figsize=(8.2, 5.0), dpi=180)
            x = np.arange(len(paired))
            ax.plot(x, paired["horizontal_error_m_without_simulation"], "o-", lw=1.1, ms=3.5, label="Measured only")
            ax.plot(x, paired["horizontal_error_m_with_simulation"], "s-", lw=1.1, ms=3.5, label="Measured + collocated simulation")
            ax.set_xticks(x)
            ax.set_xticklabels(paired["station_id"].astype(int), rotation=90)
            ax.set_xlabel("Station ID")
            ax.set_ylabel("Horizontal localization error [m]")
            ax.set_title("Base-station localization: simulation-data ablation")
            ax.grid(True, alpha=0.25)
            ax.legend(loc="best")
            fig.tight_layout()
            fig.savefig(output_dir / "localization_simulation_ablation.png", dpi=300, bbox_inches="tight", facecolor="white")
            plt.close(fig)
    (output_dir / "experiment_metadata.json").write_text(json.dumps({
        "algorithm": ALGORITHM_NAME,
        "random_sampling": "nested_uniform_random_prefix; same trial seed shared across 10--15 point counts",
        "candidate_generation": "measurement centroids + sector high-RSRP boresight/gradient candidates + ray intersections",
        "candidate_selection": "leave-one-location-out PCI-RSRP validation; measured-only or weighted measured+collocated-simulation channels",
        "validation_model": "ridge sector offsets + shared log-distance decay + fixed antenna-gain feature",
        "rsrp_range_dbm": [MIN_RSRP_DBM, MAX_RSRP_DBM],
        "random_seed": int(args.random_seed),
        "points_per_station": int(args.points_per_station),
        "direction_prior_mode": args.direction_prior_mode,
        "simulation_mode": args.simulation_mode,
        "simulation_root": str(simulation_root),
        "simulation_role": (
            "pure-simulation PCI-RSRP is sampled at exactly the selected measured receiver locations; "
            "measured and simulated RSRP form two collocated validation channels"
        ),
        "simulation_weight": float(args.simulation_weight),
        "simulation_map_extent_peak_or_center_used": False,
        "simulation_tx_metadata_used": False,
        "strict_simulation_data": bool(args.strict_simulation_data),
        "direction_csv": str(direction_path) if direction_path else None,
        "truth_used_only_for_final_evaluation": True,
        "direct_rsrp_to_range_inversion": False,
        "truth_used_for_candidate_generation_or_selection": False,
        "previous_results": str(args.previous_results.expanduser().resolve()) if args.previous_results is not None else None,
        "previous_without_simulation_results": str(args.previous_without_simulation_results.expanduser().resolve()) if args.previous_without_simulation_results is not None else None,
        "previous_with_simulation_results": str(args.previous_with_simulation_results.expanduser().resolve()) if args.previous_with_simulation_results is not None else None,
        "progressive_candidate_inheritance": True,
        "progressive_min_improvement_db": float(args.progressive_min_improvement_db),
    }, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
