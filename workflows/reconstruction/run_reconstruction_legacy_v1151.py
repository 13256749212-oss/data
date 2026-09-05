#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Radio-map reconstruction with a paired simulation-data ablation.

Experiment definition
---------------------
* The no-simulation baseline is nearest-neighbor interpolation.
* The simulation-assisted method uses the calibrated pure-simulation map as a
  fixed spatial trend and applies nearest-neighbor interpolation to the
  measured-minus-simulation residual.  This avoids re-estimating a low-sample
  multiplicative RSRP slope at every percentage.  It falls back cell-by-cell to
  the baseline where the prior is invalid.
* The sampling population is the actual outdoor 1-m measured cells for one
  physical station / PCI inside the 512 m x 512 m radio-map window.
* Eligible measured cells are ordered once using a progressive coverage-aware
  nested ranking.  The 1%, 2%, ..., 10% subsets are prefixes of the same ranking,
  so spatial coverage expands progressively while isolated local RSRP outliers are
  de-prioritized.  The legacy fixed-seed random ranking remains available by CLI.
* The reference/ground-truth map is the already generated
  ``measurement_reconstructed`` (filled) single-PCI radio map.
* RMSE is computed against every finite outdoor cell of that filled map.  An
  additional unsampled-cell RMSE is reported after excluding selected measured
  cells from evaluation.
* Both the filled reference map and every nearest-neighbor reconstruction are
  saved as PNG and NPZ.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent

from radio_reconstruction.data import read_processed_long_table
from radio_reconstruction.simulation_prior import (
    discover_simulation_prior,
    load_simulation_prior,
)

DEFAULT_PERCENTAGES = list(range(1, 11))


def parse_percentages(text: str) -> list[int]:
    values: list[int] = []
    for token in str(text).replace(";", ",").split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if not (1 <= value <= 100):
            raise ValueError(f"采样百分比必须位于1--100，当前得到{value}")
        if value not in values:
            values.append(value)
    if not values:
        raise ValueError("至少需要一个采样百分比")
    return sorted(values)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "严格配对无线电地图重构：比较仅实测最近邻与"
            "实测+纯Sionna RT仿真残差最近邻"
        )
    )
    p.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument("--measurements", type=Path, default=None)
    p.add_argument("--filled-reference-npz", type=Path, default=None, help="补齐后的measurement_reconstructed单PCI NPZ；默认自动搜索")
    p.add_argument("--output-root", type=Path, default=None)
    p.add_argument("--station-id", type=int, default=3)
    p.add_argument("--pci", type=int, default=558)
    p.add_argument("--percentages", default=",".join(map(str, DEFAULT_PERCENTAGES)))
    p.add_argument("--random-seed", type=int, default=20260805)
    p.add_argument(
        "--selection-mode",
        choices=["coverage_aware", "random"],
        default="coverage_aware",
        help=(
            "coverage_aware=渐进嵌套空间覆盖选点（默认，用于降低单个随机序列造成的RMSE反跳）；"
            "random=旧版固定seed随机嵌套序列"
        ),
    )
    p.add_argument(
        "--simulation-correction",
        choices=["additive_residual", "affine_residual"],
        default="additive_residual",
        help=(
            "additive_residual=固定Sionna空间趋势+实测残差NN（默认，避免小样本斜率漂移）；"
            "affine_residual=旧版每个采样比例重新拟合a*simulation+b"
        ),
    )
    p.add_argument("--min-rsrp-dbm", type=float, default=-120.0)
    p.add_argument("--max-rsrp-dbm", type=float, default=-40.0)
    p.add_argument("--display-min-dbm", type=float, default=-120.0)
    p.add_argument("--display-max-dbm", type=float, default=-40.0)
    p.add_argument("--dpi", type=int, default=1000)
    p.add_argument("--skip-figures", action="store_true")
    p.add_argument(
        "--simulation-mode", choices=["without", "with", "compare"], default="compare",
        help="without=仅最近邻；with=仅输出仿真残差重构；compare=在相同选点上输出并对比两者",
    )
    p.add_argument("--simulation-npz", type=Path, default=None, help="纯仿真单PCI NPZ；默认自动搜索")
    return p.parse_args()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _resample_prior_building_mask(prior, x_centers: np.ndarray, y_centers: np.ndarray) -> np.ndarray:
    """Compatibility helper retained for the project test suite.

    The v1.9 reconstruction experiment reads its building mask directly from the
    filled reference map, but older workflows/tests call this nearest-neighbor
    resampler explicitly.
    """
    target_shape = (len(y_centers), len(x_centers))
    if getattr(prior, "building_mask", None) is None:
        return np.zeros(target_shape, dtype=bool)
    source = np.asarray(prior.building_mask, dtype=bool)
    if source.shape == target_shape and np.allclose(prior.x_axis_m, x_centers) and np.allclose(prior.y_axis_m, y_centers):
        return source.copy()
    from scipy.interpolate import RegularGridInterpolator
    interp = RegularGridInterpolator(
        (np.asarray(prior.y_axis_m, dtype=float), np.asarray(prior.x_axis_m, dtype=float)),
        source.astype(float), method="nearest", bounds_error=False, fill_value=0.0,
    )
    xx, yy = np.meshgrid(np.asarray(x_centers, dtype=float), np.asarray(y_centers, dtype=float))
    values = interp(np.column_stack([yy.ravel(), xx.ravel()]))
    return values.reshape(target_shape) >= 0.5


def discover_filled_reference(project_root: Path, station_id: int, pci: int, explicit: Path | None) -> Path:
    if explicit is not None:
        path = explicit.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"--filled-reference-npz不存在：{path}")
        return path

    roots = [project_root / "outputs", project_root]
    patterns = [
        f"**/station_{station_id:02d}/**/02_measurement_reconstructed/npz/*station_{station_id:02d}_pci_{pci}*measurement_reconstructed*.npz",
        f"**/*station_{station_id:02d}_pci_{pci}_measurement_reconstructed*.npz",
        f"**/*station_{station_id:02d}*pci_{pci}*measurement_reconstructed*.npz",
    ]
    candidates: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        for pattern in patterns:
            candidates.update(p.resolve() for p in root.glob(pattern) if p.is_file())

    scored: list[tuple[int, float, Path]] = []
    for path in candidates:
        name = str(path).lower().replace("\\", "/")
        score = 0
        if "measurement_reconstructed" in name:
            score += 50
        if "dem_following" in name or "dem_plus_1p5" in name:
            score += 30
        if "bestparam_dem_vs_zplane_512m" in name:
            score += 20
        if "bestparam" in name:
            score += 10
        if "zplane" in name:
            score -= 15
        if "best_server" in name:
            score -= 40
        scored.append((score, path.stat().st_mtime, path))
    if not scored:
        raise FileNotFoundError(
            "未自动找到补齐后的单PCI无线电地图NPZ。请使用 --filled-reference-npz 指定类似：\n"
            f"outputs/.../station_{station_id:02d}/.../02_measurement_reconstructed/npz/"
            f"station_{station_id:02d}_pci_{pci}_..._measurement_reconstructed.npz"
        )
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return scored[0][2]


def grid_edges(axis: np.ndarray) -> tuple[float, float, float]:
    axis = np.asarray(axis, dtype=float).reshape(-1)
    if len(axis) < 2:
        raise ValueError("无线电地图坐标轴长度不足")
    step = float(np.median(np.diff(axis)))
    if not np.isfinite(step) or step <= 0:
        raise ValueError("无线电地图坐标轴不是递增规则网格")
    return float(axis[0] - step / 2.0), float(axis[-1] + step / 2.0), step


def prepare_measured_cells(
    measurements_csv: Path,
    station_id: int,
    pci: int,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    reference_map: np.ndarray,
    building_mask: np.ndarray,
    min_rsrp_dbm: float,
    max_rsrp_dbm: float,
) -> pd.DataFrame:
    frame = read_processed_long_table(measurements_csv)
    selected = frame.loc[
        frame["station_id"].eq(int(station_id))
        & frame["pci"].eq(int(pci))
        & frame["measured_rsrp_dbm"].between(float(min_rsrp_dbm), float(max_rsrp_dbm), inclusive="both")
    ].copy()
    if selected.empty:
        raise ValueError(f"station={station_id}, PCI={pci}没有可用实测记录")

    x_min, x_max, dx = grid_edges(x_axis)
    y_min, y_max, dy = grid_edges(y_axis)
    selected = selected.loc[
        selected["blender_x"].ge(x_min) & selected["blender_x"].lt(x_max)
        & selected["blender_y"].ge(y_min) & selected["blender_y"].lt(y_max)
    ].copy()
    if selected.empty:
        raise ValueError("补齐地图窗口内没有实测点")

    selected["ix"] = np.floor((selected["blender_x"] - x_min) / dx).astype(int)
    selected["iy"] = np.floor((selected["blender_y"] - y_min) / dy).astype(int)
    nx, ny = len(x_axis), len(y_axis)
    selected = selected.loc[selected["ix"].between(0, nx - 1) & selected["iy"].between(0, ny - 1)].copy()

    grouped = (
        selected.groupby(["station_id", "pci", "ix", "iy"], as_index=False)
        .agg(
            x_m=("blender_x", "median"),
            y_m=("blender_y", "median"),
            measured_rsrp_dbm=("measured_rsrp_dbm", "median"),
            raw_record_count=("measured_rsrp_dbm", "size"),
        )
        .sort_values(["iy", "ix"])
        .reset_index(drop=True)
    )
    iy = grouped["iy"].to_numpy(dtype=int)
    ix = grouped["ix"].to_numpy(dtype=int)
    valid = (~building_mask[iy, ix]) & np.isfinite(reference_map[iy, ix])
    grouped = grouped.loc[valid].copy().reset_index(drop=True)
    if grouped.empty:
        raise ValueError("实测点与补齐地图有效室外区域没有交集")
    return grouped


def basic_random_nested_ranking(measured: pd.DataFrame, max_count: int, seed: int = 20260805) -> pd.DataFrame:
    """Create the simplest reproducible nested random ranking.

    No radio-map values, geometry score, coverage heuristic, or local-RSRP
    representativeness is used.  One fixed-seed random permutation is created,
    and each percentage takes a prefix of that permutation.
    """
    candidates = measured.copy().reset_index(drop=True)
    if len(candidates) == 0:
        raise ValueError("没有可用于随机选点的实测点")
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(len(candidates))
    max_count = int(min(max(1, max_count), len(candidates)))
    ranked = candidates.iloc[order[:max_count]].copy().reset_index(drop=True)
    ranked["random_rank"] = np.arange(1, len(ranked) + 1, dtype=int)
    ranked["selection_method"] = "basic_random_nested_fixed_seed"
    return ranked


def coverage_aware_nested_ranking(
    measured: pd.DataFrame,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    reference_eval_mask: np.ndarray,
    max_count: int,
    seed: int = 20260805,
) -> pd.DataFrame:
    """Build a progressive nested ranking for stable sparse NN reconstruction.

    The filled-reference RSRP values and Sionna prior values are never used for
    point selection.  Geometry is the dominant criterion (candidate coordinates
    plus the finite outdoor-domain mask).  A mild local-consistency penalty uses
    only the measured candidate pool to de-prioritize isolated RSRP outliers that
    would otherwise control an unrealistically large nearest-neighbor Voronoi
    region.  The procedure remains nested: every larger percentage is a strict
    prefix extension of the smaller one.
    """
    candidates = measured.copy().reset_index(drop=True)
    candidate_xy = candidates[["x_m", "y_m"]].to_numpy(dtype=float)
    rng = np.random.default_rng(int(seed))
    tie_jitter = rng.uniform(0.0, 1e-9, size=len(candidates))
    if len(candidate_xy) == 0:
        raise ValueError("没有可用于覆盖选点的实测点")
    max_count = int(min(max(1, max_count), len(candidate_xy)))

    xx, yy = np.meshgrid(np.asarray(x_axis, dtype=float), np.asarray(y_axis, dtype=float))
    valid_flat = np.asarray(reference_eval_mask, dtype=bool).ravel()
    domain_xy = np.column_stack([xx.ravel()[valid_flat], yy.ravel()[valid_flat]])
    if len(domain_xy) == 0:
        raise ValueError("参考地图没有有效室外区域")

    candidate_tree = cKDTree(candidate_xy)

    # Local measured-RSRP representativeness.  A nearest-neighbor map gives each
    # selected sample a potentially large Voronoi region, so isolated measurement
    # outliers are poor representatives even if their coordinates improve coverage.
    measured_values = pd.to_numeric(candidates["measured_rsrp_dbm"], errors="coerce").to_numpy(dtype=float)
    k_local = min(9, len(candidate_xy))
    _, local_idx = candidate_tree.query(candidate_xy, k=k_local)
    local_idx = np.atleast_2d(local_idx)
    local_median = np.asarray([np.nanmedian(measured_values[np.asarray(row, dtype=int)]) for row in local_idx], dtype=float)
    local_deviation = np.abs(measured_values - local_median)
    finite_dev = local_deviation[np.isfinite(local_deviation)]
    dev_scale = max(float(np.percentile(finite_dev, 90)) if len(finite_dev) else 1.0, 1.0)
    candidates["local_rsrp_deviation_db"] = local_deviation

    domain_centroid = np.mean(domain_xy, axis=0)
    # Choose the first point near the domain center but avoid a local RSRP outlier.
    k_start = min(16, len(candidate_xy))
    d_start, start_candidates = candidate_tree.query(domain_centroid, k=k_start)
    start_candidates = np.atleast_1d(start_candidates).astype(int)
    d_start = np.atleast_1d(d_start).astype(float)
    start_cost = d_start / max(float(np.max(d_start)), 1.0) + 0.35 * np.clip(local_deviation[start_candidates] / dev_scale, 0.0, 3.0)
    start_idx = int(start_candidates[int(np.nanargmin(start_cost + tie_jitter[start_candidates]))])
    selected: list[int] = [int(start_idx)]
    selected_set = {int(start_idx)}
    min_domain_distance = np.linalg.norm(domain_xy - candidate_xy[int(start_idx)][None, :], axis=1)

    # Candidate-to-measured coverage is also tracked so road-sampling geometry is
    # represented even when a large reference-map area is unreachable by roads.
    min_measured_distance = np.linalg.norm(candidate_xy - candidate_xy[int(start_idx)][None, :], axis=1)

    while len(selected) < max_count:
        # Blend the worst-covered full-map cell and worst-covered measured route.
        domain_target = domain_xy[int(np.argmax(min_domain_distance))]
        measured_target_idx = int(np.argmax(min_measured_distance))
        measured_target = candidate_xy[measured_target_idx]

        target_candidates: list[int] = []
        for target in (domain_target, measured_target):
            kq = min(24, len(candidate_xy))
            _, near_idx = candidate_tree.query(target, k=kq)
            added = 0
            for cand in np.atleast_1d(near_idx).astype(int):
                if int(cand) not in selected_set:
                    target_candidates.append(int(cand))
                    added += 1
                    if added >= 6:
                        break
        if not target_candidates:
            remaining = np.asarray([i for i in range(len(candidate_xy)) if i not in selected_set], dtype=int)
            if len(remaining) == 0:
                break
            target_candidates = [int(remaining[np.argmax(min_measured_distance[remaining])])]

        # Score the small candidate set by reduction of high-tail spatial gaps.
        best_idx = None
        best_score = -np.inf
        for cand in dict.fromkeys(target_candidates):
            d_domain_new = np.minimum(min_domain_distance, np.linalg.norm(domain_xy - candidate_xy[cand][None, :], axis=1))
            d_meas_new = np.minimum(min_measured_distance, np.linalg.norm(candidate_xy - candidate_xy[cand][None, :], axis=1))
            # Lower P90/max distances are better.  The constant sign makes this a
            # maximization score and uses no radio-map values.
            spatial_cost = (
                0.55 * float(np.percentile(d_domain_new, 90))
                + 0.20 * float(np.max(d_domain_new))
                + 0.20 * float(np.percentile(d_meas_new, 90))
                + 0.05 * float(np.max(d_meas_new))
            )
            # Mild penalty only: geometry remains dominant, but among similarly
            # useful locations prefer a measured point representative of its local
            # radio neighborhood rather than an isolated RSRP outlier.
            outlier_penalty = 0.12 * max(float(np.percentile(min_measured_distance, 90)), 20.0) * float(np.clip(local_deviation[cand] / dev_scale, 0.0, 3.0))
            score = -(spatial_cost + outlier_penalty)
            score += float(tie_jitter[cand])
            if score > best_score:
                best_score = score
                best_idx = int(cand)
        if best_idx is None:
            break
        selected.append(best_idx)
        selected_set.add(best_idx)
        min_domain_distance = np.minimum(min_domain_distance, np.linalg.norm(domain_xy - candidate_xy[best_idx][None, :], axis=1))
        min_measured_distance = np.minimum(min_measured_distance, np.linalg.norm(candidate_xy - candidate_xy[best_idx][None, :], axis=1))

    ranked = candidates.iloc[selected].copy().reset_index(drop=True)
    ranked["coverage_rank"] = np.arange(1, len(ranked) + 1, dtype=int)
    ranked["selection_method"] = "coverage_aware_nested_kcenter"
    return ranked


def measured_pool_coverage_metrics(all_measured: pd.DataFrame, selected: pd.DataFrame) -> tuple[float, float, float]:
    all_xy = all_measured[["x_m", "y_m"]].to_numpy(dtype=float)
    sel_xy = selected[["x_m", "y_m"]].to_numpy(dtype=float)
    if len(sel_xy) == 0:
        return float("nan"), float("nan"), float("nan")
    d, _ = cKDTree(sel_xy).query(all_xy, k=1)
    return float(np.mean(d)), float(np.percentile(d, 90)), float(np.max(d))


def rmse(reference: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> float:
    valid = np.asarray(mask, dtype=bool) & np.isfinite(reference) & np.isfinite(prediction)
    if not np.any(valid):
        return float("nan")
    diff = prediction[valid] - reference[valid]
    return float(np.sqrt(np.mean(diff ** 2)))


def mae(reference: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> float:
    valid = np.asarray(mask, dtype=bool) & np.isfinite(reference) & np.isfinite(prediction)
    if not np.any(valid):
        return float("nan")
    return float(np.mean(np.abs(prediction[valid] - reference[valid])))


def bias(reference: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> float:
    valid = np.asarray(mask, dtype=bool) & np.isfinite(reference) & np.isfinite(prediction)
    if not np.any(valid):
        return float("nan")
    return float(np.mean(prediction[valid] - reference[valid]))


def robust_affine_calibration(prior_values: np.ndarray, measured_values: np.ndarray) -> tuple[float, float]:
    """Fit measured ~= a * simulation + b using Huber IRLS and slope shrinkage."""
    x = np.asarray(prior_values, dtype=float)
    y = np.asarray(measured_values, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    n = len(x)
    if n == 0:
        return 1.0, 0.0
    if n < 4 or float(np.ptp(x)) < 1e-6:
        return 1.0, float(np.median(y - x))

    design = np.column_stack([x, np.ones(n, dtype=float)])
    beta = np.asarray([1.0, float(np.median(y - x))], dtype=float)
    for _ in range(20):
        residual = y - design @ beta
        center = float(np.median(residual))
        mad = float(np.median(np.abs(residual - center)))
        scale = max(1.4826 * mad, 0.75)
        cutoff = 1.345 * scale
        weights = np.ones(n, dtype=float)
        tail = np.abs(residual) > cutoff
        weights[tail] = cutoff / np.maximum(np.abs(residual[tail]), 1e-9)
        slope_lambda = 8.0 / max(float(n), 1.0)
        weighted_design = design * np.sqrt(weights)[:, None]
        weighted_y = y * np.sqrt(weights)
        augmented_design = np.vstack([weighted_design, [np.sqrt(slope_lambda), 0.0]])
        augmented_y = np.concatenate([weighted_y, [np.sqrt(slope_lambda)]])
        updated, *_ = np.linalg.lstsq(augmented_design, augmented_y, rcond=None)
        updated[0] = float(np.clip(updated[0], 0.20, 1.80))
        updated[1] = float(np.median(y - updated[0] * x))
        if float(np.linalg.norm(updated - beta)) < 1e-6:
            beta = updated
            break
        beta = updated
    return float(beta[0]), float(beta[1])


def simulation_residual_nearest_neighbor(
    *,
    prior,
    train_xy: np.ndarray,
    train_y: np.ndarray,
    query_xy: np.ndarray,
    baseline_prediction: np.ndarray,
    correction_mode: str = "additive_residual",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Correct a calibrated Sionna prior with sparse measured residuals.

    ``additive_residual`` is the formal default.  The Sionna map already comes
    from the measurement-calibrated propagation workflow, so its dB-scale spatial
    variation is kept fixed and measurements only estimate local additive residuals.
    For nearest residual anchor j,

        pred(q) = S(q) + [y_j - S(x_j)].

    This removes the percentage-to-percentage slope drift that occurred when a
    small set of 10--50 points was repeatedly used to fit ``a*S+b``.  The old
    affine formulation is retained as ``affine_residual`` for reproducibility.
    """
    train_xy = np.asarray(train_xy, dtype=float)
    train_y = np.asarray(train_y, dtype=float)
    query_xy = np.asarray(query_xy, dtype=float)
    baseline = np.asarray(baseline_prediction, dtype=float).reshape(-1)
    prior_train = np.asarray(prior.sample(train_xy), dtype=float)
    valid_train = np.isfinite(prior_train) & np.isfinite(train_y)
    if int(valid_train.sum()) < 2:
        return baseline.copy(), {
            "simulation_prior_training_count": int(valid_train.sum()),
            "simulation_correction_mode": str(correction_mode),
            "affine_prior_slope": 1.0,
            "affine_prior_intercept_db": 0.0,
            "simulation_fallback_fraction": 1.0,
        }

    mode = str(correction_mode).strip().lower()
    if mode == "additive_residual":
        slope = 1.0
        intercept = float(np.median(train_y[valid_train] - prior_train[valid_train]))
    elif mode == "affine_residual":
        slope, intercept = robust_affine_calibration(
            prior_train[valid_train], train_y[valid_train]
        )
    else:
        raise ValueError(f"未知simulation correction mode: {correction_mode}")

    train_trend = slope * prior_train[valid_train] + intercept
    residual = train_y[valid_train] - train_trend
    residual_tree = cKDTree(train_xy[valid_train])
    _, nearest = residual_tree.query(query_xy, k=1)
    prior_query = np.asarray(prior.sample(query_xy), dtype=float)
    corrected = slope * prior_query + intercept + residual[np.asarray(nearest, dtype=int)]
    valid_query = np.isfinite(corrected)
    prediction = baseline.copy()
    prediction[valid_query] = corrected[valid_query]
    return prediction, {
        "simulation_prior_training_count": int(valid_train.sum()),
        "simulation_correction_mode": mode,
        "affine_prior_slope": float(slope),
        "affine_prior_intercept_db": float(intercept),
        "training_residual_median_db": float(np.median(residual)),
        "training_residual_mad_db": float(np.median(np.abs(residual - np.median(residual)))),
        "simulation_fallback_fraction": float(1.0 - np.mean(valid_query)),
    }


def save_map_npz(
    path: Path,
    *,
    station_id: int,
    pci: int,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    rsrp_map: np.ndarray,
    building_mask: np.ndarray,
    map_role: str,
    metadata: dict[str, Any],
    selected_points: pd.DataFrame | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "station_id": np.asarray([int(station_id)], dtype=np.int32),
        "pci": np.asarray([int(pci)], dtype=np.int32),
        "map_role": np.asarray([str(map_role)]),
        "x_m": np.asarray(x_axis, dtype=np.float32),
        "y_m": np.asarray(y_axis, dtype=np.float32),
        "rsrp_dbm": np.asarray(rsrp_map, dtype=np.float32),
        "building_mask": np.asarray(building_mask, dtype=np.bool_),
        "metadata_json": np.asarray([json.dumps(_json_safe(metadata), ensure_ascii=False)]),
    }
    if selected_points is not None:
        payload.update({
            "selected_x_m": selected_points["x_m"].to_numpy(dtype=np.float32),
            "selected_y_m": selected_points["y_m"].to_numpy(dtype=np.float32),
            "selected_rsrp_dbm": selected_points["measured_rsrp_dbm"].to_numpy(dtype=np.float32),
            "selected_ix": selected_points["ix"].to_numpy(dtype=np.int32),
            "selected_iy": selected_points["iy"].to_numpy(dtype=np.int32),
        })
    np.savez_compressed(path, **payload)


def plot_map(
    path: Path,
    *,
    rsrp_map: np.ndarray,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    building_mask: np.ndarray,
    title: str,
    min_dbm: float,
    max_dbm: float,
    dpi: int,
    selected_points: pd.DataFrame | None = None,
    show_selected_points: bool = True,
) -> None:
    x_min, x_max, _ = grid_edges(x_axis)
    y_min, y_max, _ = grid_edges(y_axis)
    display = np.asarray(rsrp_map, dtype=float).copy()
    display[building_mask] = np.nan
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("white")
    fig, ax = plt.subplots(figsize=(7.0, 5.8), dpi=int(dpi))
    im = ax.imshow(
        display, origin="lower", extent=[x_min, x_max, y_min, y_max],
        cmap=cmap, vmin=float(min_dbm), vmax=float(max_dbm), interpolation="nearest", aspect="equal",
    )
    if show_selected_points and selected_points is not None and len(selected_points):
        ax.scatter(
            selected_points["x_m"], selected_points["y_m"],
            s=14, facecolors="none", edgecolors="black", linewidths=0.55,
            label="Selected measured points",
        )
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.15), fontsize=7.0, framealpha=0.95)
    ax.set_xlabel("Blender X [m]")
    ax.set_ylabel("Blender Y [m]")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("RSRP [dBm]")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=int(dpi), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_paired_reconstruction_composite(
    path: Path,
    *,
    representative_maps: dict[int, tuple[np.ndarray, np.ndarray, float, float]],
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    building_mask: np.ndarray,
    min_dbm: float,
    max_dbm: float,
    dpi: int,
) -> None:
    """Plot measured-only and measured+simulation maps on matched columns."""
    percentages = [pct for pct in (1, 5, 10) if pct in representative_maps]
    if not percentages:
        return
    x_min, x_max, _ = grid_edges(x_axis)
    y_min, y_max, _ = grid_edges(y_axis)
    extent = [x_min, x_max, y_min, y_max]
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("white")
    fig, axes = plt.subplots(
        2,
        len(percentages),
        figsize=(7.48, 5.25),
        dpi=int(dpi),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes = np.asarray(axes, dtype=object).reshape(2, len(percentages))
    image = None
    for col, pct in enumerate(percentages):
        baseline, assisted, baseline_rmse, assisted_rmse = representative_maps[pct]
        for row, (values, label, score) in enumerate((
            (baseline, "Measured only", baseline_rmse),
            (assisted, "Measured + simulation", assisted_rmse),
        )):
            display = np.asarray(values, dtype=float).copy()
            display[np.asarray(building_mask, dtype=bool)] = np.nan
            ax = axes[row, col]
            image = ax.imshow(
                display,
                origin="lower",
                extent=extent,
                cmap=cmap,
                vmin=float(min_dbm),
                vmax=float(max_dbm),
                interpolation="nearest",
                aspect="equal",
            )
            panel_index = row * len(percentages) + col
            panel_label = chr(ord("a") + panel_index)
            ax.set_title(
                f"({panel_label}) {pct}% measured points\nRMSE={score:.2f} dB",
                fontsize=8.2,
                pad=3.0,
            )
            ax.tick_params(axis="both", labelsize=6.8, pad=1.5)
            if col == 0:
                ax.set_ylabel(f"{label}\nBlender Y [m]", fontsize=7.5)
            if row == 1:
                ax.set_xlabel("Blender X [m]", fontsize=7.5)
    if image is not None:
        cbar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.86, pad=0.015)
        cbar.set_label("RSRP [dBm]", fontsize=8.0)
        cbar.ax.tick_params(labelsize=7.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=int(dpi), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_ablation_analysis(path: Path, paired: pd.DataFrame) -> None:
    """Write a compact, reproducible analysis of the paired reconstruction."""
    baseline_mean = float(paired["rmse_db_without_simulation"].mean())
    assisted_mean = float(paired["rmse_db_with_simulation"].mean())
    gain_mean = baseline_mean - assisted_mean
    relative_gain = 100.0 * gain_mean / baseline_mean if baseline_mean else float("nan")
    improved = int((paired["rmse_gain_with_simulation_db"] > 0.0).sum())
    lines = [
        "# 无线电地图重构：仅实测与实测+仿真配对实验",
        "",
        "两组使用相同的候选实测位置、固定随机排列、嵌套采样前缀、建筑物掩膜、评价域和Measurement-filled radio map参考。唯一差别是联合组是否使用纯Sionna RT地图。",
        "",
        f"- 1%--10%的 {improved}/{len(paired)} 个采样比例中，联合组RMSE低于仅实测组。",
        f"- 十种比例平均RMSE：仅实测 {baseline_mean:.2f} dB，实测+仿真 {assisted_mean:.2f} dB，降低 {gain_mean:.2f} dB（{relative_gain:.2f}%）。",
        "- 联合方法先用所选实测点稳健校准仿真趋势，再对实测减仿真的残差执行最近邻插值；仿真无效栅格退化为仅实测最近邻。",
        "",
        "| 实测比例 | 点数 | 仅实测RMSE(dB) | 实测+仿真RMSE(dB) | 改善(dB) | 仅实测MAE(dB) | 实测+仿真MAE(dB) |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in paired.itertuples(index=False):
        lines.append(
            f"| {int(row.sampling_percent)}% | {int(row.selected_measured_point_count)} | "
            f"{float(row.rmse_db_without_simulation):.2f} | "
            f"{float(row.rmse_db_with_simulation):.2f} | "
            f"{float(row.rmse_gain_with_simulation_db):.2f} | "
            f"{float(row.mae_db_without_simulation):.2f} | "
            f"{float(row.mae_db_with_simulation):.2f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    project_root = args.project_root.expanduser().resolve()
    measurements_csv = (
        args.measurements.expanduser().resolve()
        if args.measurements is not None
        else project_root / "data" / "processed" / "cell_pci_rsrp_1m_calibration.csv"
    )
    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root is not None
        else project_root / "outputs" / "radio_map_reconstruction_nn_random_percent_1to10_v111"
    )
    percentages = parse_percentages(args.percentages)
    reference_path = discover_filled_reference(project_root, args.station_id, args.pci, args.filled_reference_npz)

    # The generic loader reads the 2-D rsrp_dbm field and axes from a filled map NPZ.
    reference = load_simulation_prior(
        reference_path,
        station_id=int(args.station_id),
        pci=int(args.pci),
        fallback_extent=None,
    )
    simulation_prior = None
    simulation_path = None
    if args.simulation_mode in {"with", "compare"}:
        simulation_path = discover_simulation_prior(
            project_root=project_root,
            station_id=int(args.station_id),
            pci=int(args.pci),
            explicit_path=args.simulation_npz,
        )
        if simulation_path.resolve() == reference_path.resolve():
            raise ValueError("纯仿真NPZ不能与补齐后的实测参考地图使用同一文件")
        simulation_prior = load_simulation_prior(
            simulation_path,
            station_id=int(args.station_id),
            pci=int(args.pci),
            fallback_extent=None,
        )
    reference_map = np.asarray(reference.rsrp_dbm, dtype=float).copy()
    finite_reference = np.isfinite(reference_map)
    reference_map[finite_reference] = np.clip(reference_map[finite_reference], -120.0, -40.0)
    building_mask = (
        np.asarray(reference.building_mask, dtype=bool)
        if reference.building_mask is not None
        else np.zeros(reference_map.shape, dtype=bool)
    )
    if building_mask.shape != reference_map.shape:
        raise ValueError("补齐地图building_mask与RSRP地图尺寸不一致")
    reference_eval_mask = (~building_mask) & np.isfinite(reference_map)
    if not np.any(reference_eval_mask):
        raise ValueError("补齐后的无线电地图没有有效室外栅格")

    measured = prepare_measured_cells(
        measurements_csv=measurements_csv,
        station_id=int(args.station_id),
        pci=int(args.pci),
        x_axis=reference.x_axis_m,
        y_axis=reference.y_axis_m,
        reference_map=reference_map,
        building_mask=building_mask,
        min_rsrp_dbm=float(args.min_rsrp_dbm),
        max_rsrp_dbm=float(args.max_rsrp_dbm),
    )
    n_measured = int(len(measured))
    max_sample_count = min(
        n_measured,
        max(max(1, int(math.ceil(n_measured * float(pct) / 100.0))) for pct in percentages),
    )
    if args.selection_mode == "coverage_aware":
        ranked = coverage_aware_nested_ranking(
            measured=measured,
            x_axis=reference.x_axis_m,
            y_axis=reference.y_axis_m,
            reference_eval_mask=reference_eval_mask,
            max_count=max_sample_count,
            seed=int(args.random_seed),
        )
    else:
        ranked = basic_random_nested_ranking(
            measured=measured, max_count=max_sample_count, seed=int(args.random_seed)
        )

    station_root = output_root / f"station_{int(args.station_id):02d}_pci_{int(args.pci)}"
    station_root.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(station_root / "all_measured_points_nested_ranking.csv", index=False, encoding="utf-8-sig")

    reference_dir = station_root / "reference_filled_map"
    reference_meta = {
        "source_filled_reference_npz": str(reference_path),
        "station_id": int(args.station_id),
        "pci": int(args.pci),
        "valid_outdoor_reference_cell_count": int(reference_eval_mask.sum()),
        "eligible_measured_point_count": n_measured,
        "evaluation_reference": "measurement_reconstructed / filled radio map",
    }
    save_map_npz(
        reference_dir / f"station_{int(args.station_id):02d}_pci_{int(args.pci)}_filled_reference.npz",
        station_id=args.station_id, pci=args.pci,
        x_axis=reference.x_axis_m, y_axis=reference.y_axis_m,
        rsrp_map=reference_map, building_mask=building_mask,
        map_role="filled_reference", metadata=reference_meta,
    )
    if not args.skip_figures:
        plot_map(
            reference_dir / f"station_{int(args.station_id):02d}_pci_{int(args.pci)}_filled_reference.png",
            rsrp_map=reference_map, x_axis=reference.x_axis_m, y_axis=reference.y_axis_m,
            building_mask=building_mask, title="Filled radio map (reference)",
            min_dbm=args.display_min_dbm, max_dbm=args.display_max_dbm, dpi=args.dpi,
        )

    xx, yy = np.meshgrid(reference.x_axis_m, reference.y_axis_m)
    query_xy = np.column_stack([xx.ravel(), yy.ravel()])
    metrics_rows: list[dict[str, Any]] = []
    ablation_rows: list[dict[str, Any]] = []
    representative_maps: dict[int, tuple[np.ndarray, np.ndarray, float, float]] = {}

    print("=" * 88)
    print("Radio-map reconstruction with paired simulation-data ablation")
    print(f"Station={args.station_id}, PCI={args.pci}")
    print(f"Filled reference: {reference_path}")
    print(f"Simulation mode: {args.simulation_mode}")
    print(f"Pure simulation prior: {simulation_path if simulation_path else 'not used'}")
    print(f"Eligible measured 1-m cells: {n_measured}")
    print(f"Percentages: {percentages}")
    print(f"Selection mode: {args.selection_mode}")
    print(f"Simulation correction: {args.simulation_correction}")
    print("=" * 88)

    for pct in percentages:
        sample_count = max(1, int(math.ceil(n_measured * float(pct) / 100.0)))
        sample_count = min(sample_count, n_measured)
        selected = ranked.iloc[:sample_count].copy().reset_index(drop=True)
        selected["sampling_percent"] = int(pct)
        selected["sample_count"] = int(sample_count)

        train_xy = selected[["x_m", "y_m"]].to_numpy(dtype=float)
        train_y = selected["measured_rsrp_dbm"].to_numpy(dtype=float)
        tree = cKDTree(train_xy)
        _, nearest = tree.query(query_xy, k=1)
        prediction = train_y[np.asarray(nearest, dtype=int)].reshape(reference_map.shape)
        prediction = np.clip(prediction, -120.0, -40.0)
        prediction[building_mask] = np.nan

        sampled_cell_mask = np.zeros(reference_map.shape, dtype=bool)
        sampled_cell_mask[
            selected["iy"].to_numpy(dtype=int), selected["ix"].to_numpy(dtype=int)
        ] = True
        unsampled_mask = reference_eval_mask & (~sampled_cell_mask)

        row = {
            "station_id": int(args.station_id),
            "pci": int(args.pci),
            "sampling_percent": int(pct),
            "total_eligible_measured_points": n_measured,
            "selected_measured_point_count": int(sample_count),
            "reference_valid_outdoor_cell_count": int(reference_eval_mask.sum()),
            "rmse_db": rmse(reference_map, prediction, reference_eval_mask),
            "mae_db": mae(reference_map, prediction, reference_eval_mask),
            "bias_pred_minus_reference_db": bias(reference_map, prediction, reference_eval_mask),
            "unsampled_reference_cell_count": int(unsampled_mask.sum()),
            "rmse_unsampled_db": rmse(reference_map, prediction, unsampled_mask),
            "mae_unsampled_db": mae(reference_map, prediction, unsampled_mask),
            "selection_seed": int(args.random_seed),
            "method": "nearest_neighbor",
            "variant": "without_simulation",
            "simulation_data_used": False,
            "reference": "filled_measurement_reconstructed_radio_map",
        }
        coverage_mean_m, coverage_p90_m, coverage_max_m = measured_pool_coverage_metrics(measured, selected)
        row["measured_pool_nearest_distance_mean_m"] = coverage_mean_m
        row["measured_pool_nearest_distance_p90_m"] = coverage_p90_m
        row["measured_pool_nearest_distance_max_m"] = coverage_max_m
        metrics_rows.append(row)
        ablation_rows.append(dict(row))

        pct_dir = station_root / f"percent_{int(pct):02d}" / "nearest_neighbor"
        if args.simulation_mode in {"without", "compare"}:
            pct_dir.mkdir(parents=True, exist_ok=True)
            selected.to_csv(pct_dir / f"selected_measured_points_{int(pct):02d}pct.csv", index=False, encoding="utf-8-sig")
        metadata = {
            **row,
            "selection": ("nested prefix of progressive coverage-aware ranking" if args.selection_mode == "coverage_aware" else "nested prefix of one fixed-seed uniform random permutation of eligible measured points"),
            "filled_reference_npz": str(reference_path),
        }
        if args.simulation_mode in {"without", "compare"}:
            save_map_npz(
                pct_dir / f"station_{int(args.station_id):02d}_pci_{int(args.pci)}_nn_{int(pct):02d}pct.npz",
                station_id=args.station_id, pci=args.pci,
                x_axis=reference.x_axis_m, y_axis=reference.y_axis_m,
                rsrp_map=prediction, building_mask=building_mask,
                map_role=f"nearest_neighbor_{int(pct)}pct_measured_points",
                metadata=metadata, selected_points=selected,
            )
        if not args.skip_figures and args.simulation_mode in {"without", "compare"}:
            # Save two versions for each reconstruction map: one with the sampled measured
            # points overlaid, and one clean version without the point markers.
            base_png = pct_dir / f"station_{int(args.station_id):02d}_pci_{int(args.pci)}_nn_{int(pct):02d}pct.png"
            plot_map(
                base_png,
                rsrp_map=prediction, x_axis=reference.x_axis_m, y_axis=reference.y_axis_m,
                building_mask=building_mask,
                title=f"Nearest-neighbor reconstruction ({int(pct)}% measured points)",
                min_dbm=args.display_min_dbm, max_dbm=args.display_max_dbm, dpi=args.dpi,
                selected_points=selected, show_selected_points=True,
            )
            plot_map(
                pct_dir / f"station_{int(args.station_id):02d}_pci_{int(args.pci)}_nn_{int(pct):02d}pct_no_points.png",
                rsrp_map=prediction, x_axis=reference.x_axis_m, y_axis=reference.y_axis_m,
                building_mask=building_mask,
                title=f"Nearest-neighbor reconstruction ({int(pct)}% measured points)",
                min_dbm=args.display_min_dbm, max_dbm=args.display_max_dbm, dpi=args.dpi,
                selected_points=selected, show_selected_points=False,
            )

        simulation_row = None
        if simulation_prior is not None:
            assisted_flat, simulation_diagnostics = simulation_residual_nearest_neighbor(
                prior=simulation_prior,
                train_xy=train_xy,
                train_y=train_y,
                query_xy=query_xy,
                baseline_prediction=prediction.ravel(),
                correction_mode=args.simulation_correction,
            )
            assisted_prediction = assisted_flat.reshape(reference_map.shape)
            assisted_prediction = np.clip(assisted_prediction, -120.0, -40.0)
            assisted_prediction[building_mask] = np.nan
            simulation_row = {
                **row,
                "rmse_db": rmse(reference_map, assisted_prediction, reference_eval_mask),
                "mae_db": mae(reference_map, assisted_prediction, reference_eval_mask),
                "bias_pred_minus_reference_db": bias(reference_map, assisted_prediction, reference_eval_mask),
                "rmse_unsampled_db": rmse(reference_map, assisted_prediction, unsampled_mask),
                "mae_unsampled_db": mae(reference_map, assisted_prediction, unsampled_mask),
                "method": "simulation_residual_nearest_neighbor",
                "variant": "with_simulation",
                "simulation_data_used": True,
                "simulation_prior_npz": str(simulation_path),
                **simulation_diagnostics,
            }
            ablation_rows.append(simulation_row)
            if int(pct) in {1, 5, 10}:
                representative_maps[int(pct)] = (
                    prediction.copy(),
                    assisted_prediction.copy(),
                    float(row["rmse_db"]),
                    float(simulation_row["rmse_db"]),
                )
            assisted_dir = station_root / f"percent_{int(pct):02d}" / "simulation_residual_nearest_neighbor"
            assisted_dir.mkdir(parents=True, exist_ok=True)
            selected.to_csv(
                assisted_dir / f"selected_measured_points_{int(pct):02d}pct.csv",
                index=False, encoding="utf-8-sig",
            )
            save_map_npz(
                assisted_dir / f"station_{int(args.station_id):02d}_pci_{int(args.pci)}_sim_residual_nn_{int(pct):02d}pct.npz",
                station_id=args.station_id, pci=args.pci,
                x_axis=reference.x_axis_m, y_axis=reference.y_axis_m,
                rsrp_map=assisted_prediction, building_mask=building_mask,
                map_role=f"simulation_residual_nearest_neighbor_{int(pct)}pct_measured_points",
                metadata={**simulation_row, "filled_reference_npz": str(reference_path)},
                selected_points=selected,
            )
            if not args.skip_figures:
                assisted_title = f"Simulation-residual NN reconstruction ({int(pct)}% measured points)"
                assisted_png = assisted_dir / f"station_{int(args.station_id):02d}_pci_{int(args.pci)}_sim_residual_nn_{int(pct):02d}pct.png"
                plot_map(
                    assisted_png,
                    rsrp_map=assisted_prediction, x_axis=reference.x_axis_m, y_axis=reference.y_axis_m,
                    building_mask=building_mask, title=assisted_title,
                    min_dbm=args.display_min_dbm, max_dbm=args.display_max_dbm, dpi=args.dpi,
                    selected_points=selected, show_selected_points=True,
                )
                plot_map(
                    assisted_dir / f"station_{int(args.station_id):02d}_pci_{int(args.pci)}_sim_residual_nn_{int(pct):02d}pct_no_points.png",
                    rsrp_map=assisted_prediction, x_axis=reference.x_axis_m, y_axis=reference.y_axis_m,
                    building_mask=building_mask, title=assisted_title,
                    min_dbm=args.display_min_dbm, max_dbm=args.display_max_dbm, dpi=args.dpi,
                    selected_points=selected, show_selected_points=False,
                )
        print(
            f"{pct:2d}%: selected={sample_count}/{n_measured}, "
            f"without-sim RMSE={row['rmse_db']:.3f} dB"
            + (
                f", with-sim RMSE={simulation_row['rmse_db']:.3f} dB"
                if simulation_row is not None else ""
            )
        )

    metrics = pd.DataFrame(metrics_rows).sort_values("sampling_percent").reset_index(drop=True)
    metrics.to_csv(station_root / "nearest_neighbor_percentage_metrics.csv", index=False, encoding="utf-8-sig")
    metrics[[
        "sampling_percent", "selected_measured_point_count", "total_eligible_measured_points",
        "rmse_db", "rmse_unsampled_db", "mae_db", "mae_unsampled_db",
        "measured_pool_nearest_distance_mean_m", "measured_pool_nearest_distance_p90_m",
        "measured_pool_nearest_distance_max_m",
    ]].to_csv(station_root / "nearest_neighbor_percentage_comparison.csv", index=False, encoding="utf-8-sig")

    ablation = pd.DataFrame(ablation_rows).sort_values(["sampling_percent", "simulation_data_used"]).reset_index(drop=True)
    ablation.to_csv(station_root / "reconstruction_simulation_ablation_metrics.csv", index=False, encoding="utf-8-sig")
    paired = None
    if simulation_prior is not None:
        paired = metrics[["sampling_percent", "selected_measured_point_count", "rmse_db", "rmse_unsampled_db", "mae_db"]].merge(
            ablation.loc[ablation["simulation_data_used"]].copy()[
                ["sampling_percent", "rmse_db", "rmse_unsampled_db", "mae_db"]
            ],
            on="sampling_percent", suffixes=("_without_simulation", "_with_simulation"),
        )
        paired["rmse_gain_with_simulation_db"] = paired["rmse_db_without_simulation"] - paired["rmse_db_with_simulation"]
        paired["rmse_unsampled_gain_with_simulation_db"] = (
            paired["rmse_unsampled_db_without_simulation"] - paired["rmse_unsampled_db_with_simulation"]
        )
        paired.to_csv(station_root / "reconstruction_simulation_ablation_comparison.csv", index=False, encoding="utf-8-sig")
        write_ablation_analysis(
            station_root / "reconstruction_simulation_ablation_analysis.md",
            paired,
        )

    if not args.skip_figures:
        fig, ax = plt.subplots(figsize=(7.48, 5.2), dpi=int(args.dpi))
        ax.plot(metrics["sampling_percent"], metrics["rmse_db"], marker="o", linewidth=1.4, label="All reference cells")
        ax.plot(metrics["sampling_percent"], metrics["rmse_unsampled_db"], marker="s", linewidth=1.25, label="Unsampled reference cells")
        ax.set_xlabel("Selected measured points [%]")
        ax.set_ylabel("RMSE against filled radio map [dB]")
        ax.set_title("Nearest-neighbor reconstruction versus measured-point percentage")
        ax.set_xticks(metrics["sampling_percent"].to_numpy(dtype=int))
        ax.grid(True, alpha=0.28)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(station_root / "nearest_neighbor_rmse_vs_measured_percentage.png", dpi=int(args.dpi), bbox_inches="tight", facecolor="white")
        plt.close(fig)

        if simulation_prior is not None:
            plot_paired_reconstruction_composite(
                station_root / "reconstruction_simulation_ablation_representative_maps.png",
                representative_maps=representative_maps,
                x_axis=reference.x_axis_m,
                y_axis=reference.y_axis_m,
                building_mask=building_mask,
                min_dbm=args.display_min_dbm,
                max_dbm=args.display_max_dbm,
                dpi=args.dpi,
            )
            fig, ax = plt.subplots(figsize=(7.48, 5.2), dpi=int(args.dpi))
            ax.plot(paired["sampling_percent"], paired["rmse_db_without_simulation"], marker="o", label="Without simulation: NN")
            ax.plot(paired["sampling_percent"], paired["rmse_db_with_simulation"], marker="s", label="With simulation: calibrated residual NN")
            ax.set_xlabel("Selected measured points [%]")
            ax.set_ylabel("RMSE against filled radio map [dB]")
            ax.set_title("Radio-map reconstruction: simulation-data ablation")
            ax.set_xticks(paired["sampling_percent"].to_numpy(dtype=int))
            ax.grid(True, alpha=0.28)
            ax.legend(loc="best")
            fig.tight_layout()
            fig.savefig(station_root / "reconstruction_simulation_ablation_rmse.png", dpi=int(args.dpi), bbox_inches="tight", facecolor="white")
            plt.close(fig)

    experiment_metadata = {
        "experiment": "paired radio-map reconstruction simulation-data ablation",
        "station_id": int(args.station_id),
        "pci": int(args.pci),
        "measurement_csv": str(measurements_csv),
        "filled_reference_npz": str(reference_path),
        "percentages": percentages,
        "selection_seed": int(args.random_seed),
        "eligible_measured_point_count": n_measured,
        "sampling_definition": (
            "percent of actual outdoor 1-m measured cells; progressive coverage-aware nested prefixes shared across 1--10%"
            if args.selection_mode == "coverage_aware"
            else "percent of actual outdoor 1-m measured cells; one fixed-seed uniform random permutation shared across 1--10% so subsets are nested"
        ),
        "simulation_mode": args.simulation_mode,
        "simulation_prior_npz": str(simulation_path) if simulation_path else None,
        "methods": [
            "nearest_neighbor_without_simulation",
            (
                "fixed_sionna_trend_plus_nearest_neighbor_measured_residual"
                if args.simulation_correction == "additive_residual"
                else "robust_affine_simulation_trend_plus_nearest_neighbor_measured_residual"
            ) if simulation_prior is not None else None,
        ],
        "rmse_reference": "all finite outdoor cells of the filled measurement-reconstructed radio map",
        "additional_metric": "RMSE on reference cells excluding selected measurement cells",
        "rsrp_analysis_range_dbm": [-120.0, -40.0],
        "selection_method": ("coverage_aware_nested_kcenter" if args.selection_mode == "coverage_aware" else "basic_random_nested_fixed_seed"),
        "simulation_correction_mode": args.simulation_correction,
        "figure_outputs": [
            "reconstruction PNG with selected measured points",
            "reconstruction PNG without selected measured points",
            "paired 2x3 representative comparison at 1%, 5%, and 10%",
            "paired RMSE curve for all percentages",
        ],
    }
    (station_root / "experiment_metadata.json").write_text(
        json.dumps(_json_safe(experiment_metadata), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n完成。结果目录：", station_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
