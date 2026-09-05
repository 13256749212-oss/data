#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Radio-map reconstruction with a paired simulation-data ablation.

Experiment definition
---------------------
* The no-simulation baseline is nearest-neighbor interpolation.
* The simulation-assisted method robustly calibrates a pure-simulation map and
  applies nearest-neighbor interpolation to the measured-minus-simulation
  residual.  It falls back cell-by-cell to the baseline where the prior is invalid.
* The sampling population is the actual outdoor 1-m measured cells for one
  physical station / PCI inside the 512 m x 512 m radio-map window.
* By default a single strict-nested adaptive-domain sequence is constructed.
  The 1%--10% sampling sets are prefixes of one sequence: S1% subset S2% subset ... subset S10%.
  The initial 1% is geometry-only; later acquisitions may use only RSRP values of points that have already been selected, together with full-domain geometry. Unselected measured RSRP, reference-map RSRP, and Sionna RSRP are never used for point selection.
* The reference/ground-truth map is the already generated
  ``measurement_reconstructed`` (filled) single-PCI radio map.
* RMSE is computed against every finite outdoor cell of that filled map.  An
  additional unsampled-cell RMSE is reported after excluding selected measured
  cells from evaluation.
* Each sampling percentage is reconstructed exactly once for each branch.
  The RMSE curve and the saved radio maps therefore refer to the same single
  progressive experiment; no multi-trial averaging or best-trial map selection is used.
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
from scipy.spatial import cKDTree, distance
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans

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
            "严格双分支最近邻无线电地图重构：M=纯实测1-NN；"
            "M+S=同一实测点+稳健校准Sionna趋势+残差1-NN。"
            "主RMSE始终按原代码在整个512m×512m filled-map有效室外栅格上计算。"
        )
    )
    p.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument("--measurements", type=Path, default=None)
    p.add_argument("--filled-reference-npz", type=Path, default=None, help="measurement_reconstructed单PCI NPZ；其全部有限室外栅格作为主RMSE参考域")
    p.add_argument("--output-root", type=Path, default=None)
    p.add_argument("--station-id", type=int, default=3)
    p.add_argument("--pci", type=int, default=558)
    p.add_argument("--percentages", default=",".join(map(str, DEFAULT_PERCENTAGES)))
    p.add_argument("--random-seed", type=int, default=20260805, help="仅当 --selection-mode random 时用于复现随机排序")
    p.add_argument(
        "--selection-mode",
        choices=["voronoi-safe-adaptive-nested", "adaptive-domain-nested", "hierarchical-nested", "adaptive-progressive", "stable-spatial", "progressive-coverage", "random"],
        default="voronoi-safe-adaptive-nested",
        help=(
            "voronoi-safe-adaptive-nested=默认：严格嵌套，按采样比例分阶段扩展；新增点只用几何和已采样RSRP。若已采样新点出现大幅RSRP跳变，后续1--2个采样槽强制用于邻近Voronoi约束，而不是继续普通覆盖扩展，从而缩小异常点的大面积1-NN接管区域；"
            "adaptive-domain-nested=v1.18.8旧版自适应全域严格嵌套；"
            "hierarchical-nested=纯几何严格嵌套；adaptive-progressive=旧版候选路线上自适应选点；stable-spatial=旧版各比例独立空间分层；"
            "progressive-coverage=旧版严格嵌套k-center；random=旧版单seed随机嵌套"
        ),
    )
    p.add_argument(
        "--random-trials", type=int, default=1,
        help="兼容旧命令保留；v1.18.10正式重构固定为1次，传入其他值将被忽略",
    )
    p.add_argument("--min-rsrp-dbm", type=float, default=-120.0)
    p.add_argument("--max-rsrp-dbm", type=float, default=-40.0)
    p.add_argument("--display-min-dbm", type=float, default=-120.0)
    p.add_argument("--display-max-dbm", type=float, default=-40.0)
    p.add_argument("--dpi", type=int, default=1000)
    p.add_argument("--skip-figures", action="store_true")
    p.add_argument(
        "--simulation-mode", choices=["without", "with", "compare"], default="compare",
        help="without=纯实测1-NN；with=实测+稳健校准Sionna趋势+残差1-NN；compare=严格配对比较",
    )
    p.add_argument("--simulation-npz", type=Path, default=None, help="固定纯Sionna单PCI NPZ；默认自动搜索")
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


def progressive_coverage_nested_ranking(
    measured: pd.DataFrame,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    reference_eval_mask: np.ndarray,
    max_count: int,
) -> pd.DataFrame:
    """Create one deterministic nested space-filling ranking for 1-NN reconstruction.

    The purpose is to make a *single* sampling experiment representative rather
    than averaging 50 random experiments and then displaying a different map.
    Selection is deliberately independent of radio-map amplitudes:

    * uses measured-point coordinates;
    * uses only the Boolean valid outdoor evaluation-domain mask;
    * never reads reference-map RSRP values;
    * never reads measured RSRP values for ranking;
    * never searches seeds by final RMSE.

    A candidate-constrained k-center / farthest-first strategy is used.  The
    first point is the measured location nearest the valid-domain centroid.  Each
    following point is selected from measured locations near the current
    worst-covered full-grid area and worst-covered measured-route area.  Among
    that small candidate set we choose the point giving the smallest geometric
    coverage cost.  Adding points therefore progressively reduces spatial gaps,
    which is the appropriate deterministic sampling strategy for a nearest-
    neighbor radio-map interpolation experiment.
    """
    candidates = measured.copy().reset_index(drop=True)
    candidate_xy = candidates[["x_m", "y_m"]].to_numpy(dtype=float)
    if len(candidate_xy) == 0:
        raise ValueError("没有可用于渐进覆盖选点的实测点")
    max_count = int(min(max(1, max_count), len(candidate_xy)))

    xx, yy = np.meshgrid(np.asarray(x_axis, dtype=float), np.asarray(y_axis, dtype=float))
    valid_flat = np.asarray(reference_eval_mask, dtype=bool).ravel()
    domain_xy_full = np.column_stack([xx.ravel()[valid_flat], yy.ravel()[valid_flat]])
    if len(domain_xy_full) == 0:
        raise ValueError("参考地图没有有效室外区域")

    # Selection uses a deterministic spatial support subset only to reduce CPU
    # time.  This is geometry-only and does not inspect reference RSRP values.
    stride = max(1, int(math.ceil(len(domain_xy_full) / 60000.0)))
    domain_xy = domain_xy_full[::stride]
    candidate_tree = cKDTree(candidate_xy)

    centroid = np.mean(domain_xy, axis=0)
    _, first_idx = candidate_tree.query(centroid, k=1)
    first_idx = int(first_idx)

    selected: list[int] = [first_idx]
    selected_set = {first_idx}
    min_domain_distance = np.linalg.norm(domain_xy - candidate_xy[first_idx][None, :], axis=1)
    min_measured_distance = np.linalg.norm(candidate_xy - candidate_xy[first_idx][None, :], axis=1)

    while len(selected) < max_count:
        domain_target = domain_xy[int(np.argmax(min_domain_distance))]
        measured_target = candidate_xy[int(np.argmax(min_measured_distance))]

        proposal: list[int] = []
        for target in (domain_target, measured_target):
            kq = min(24, len(candidate_xy))
            _, idx = candidate_tree.query(target, k=kq)
            for cand in np.atleast_1d(idx).astype(int):
                cand = int(cand)
                if cand not in selected_set:
                    proposal.append(cand)
        if not proposal:
            remaining = np.asarray([i for i in range(len(candidate_xy)) if i not in selected_set], dtype=int)
            if len(remaining) == 0:
                break
            proposal = [int(remaining[np.argmax(min_measured_distance[remaining])])]

        best_idx: int | None = None
        best_cost = float("inf")
        # Geometry-only scoring.  No measured/reference RSRP enters this step.
        for cand in dict.fromkeys(proposal):
            d_domain = np.minimum(
                min_domain_distance,
                np.linalg.norm(domain_xy - candidate_xy[cand][None, :], axis=1),
            )
            d_measured = np.minimum(
                min_measured_distance,
                np.linalg.norm(candidate_xy - candidate_xy[cand][None, :], axis=1),
            )
            cost = (
                0.55 * float(np.percentile(d_domain, 90))
                + 0.20 * float(np.max(d_domain))
                + 0.20 * float(np.percentile(d_measured, 90))
                + 0.05 * float(np.max(d_measured))
            )
            # Deterministic tie-break: lower original candidate index wins.
            if cost < best_cost - 1e-12 or (abs(cost - best_cost) <= 1e-12 and (best_idx is None or cand < best_idx)):
                best_cost = cost
                best_idx = int(cand)

        if best_idx is None:
            break
        selected.append(best_idx)
        selected_set.add(best_idx)
        min_domain_distance = np.minimum(
            min_domain_distance,
            np.linalg.norm(domain_xy - candidate_xy[best_idx][None, :], axis=1),
        )
        min_measured_distance = np.minimum(
            min_measured_distance,
            np.linalg.norm(candidate_xy - candidate_xy[best_idx][None, :], axis=1),
        )

    ranked = candidates.iloc[selected].copy().reset_index(drop=True)
    ranked["progressive_rank"] = np.arange(1, len(ranked) + 1, dtype=int)
    ranked["coverage_rank"] = ranked["progressive_rank"]  # backward-compatible column
    ranked["selection_method"] = "single_progressive_geometry_kcenter"
    return ranked


def adaptive_progressive_nested_ranking(
    measured: pd.DataFrame,
    max_count: int,
    initial_count: int,
    gradient_weight: float = 0.50,
) -> pd.DataFrame:
    """Single-run nested adaptive sampling for strict 1-NN reconstruction.

    This selector was added because v1.18.4 showed an important failure mode:
    spatial coverage distances continued to improve from 9% to 10%, while both
    full-grid RMSE values became slightly worse.  That means geometry-only
    k-center selection is insufficient for a radio field: a newly added point can
    own a new Voronoi region whose signal level is not representative.

    The new selector remains leakage-safe with respect to the reconstruction
    reference and Sionna map:

    * candidate coordinates are known a priori;
    * the initial sparse set is geometry-only farthest-point sampling;
    * after a point has been selected, its measured RSRP may guide where the NEXT
      point should be acquired;
    * an unselected point's RSRP is never read when deciding whether to select it;
    * reference-map RSRP and Sionna RSRP are never used for sampling.

    The adaptive score is

        score(q) = d1(q) * [1 + w * normalized_gradient(q)]

    where d1 is distance to the nearest already measured location and the local
    gradient proxy is computed only from the RSRP difference between the first
    and second nearest *already selected* measurements.  Thus new points are
    directed jointly toward uncovered road regions and currently observed rapid
    signal transitions.  The 1%,...,10% sets are strict nested prefixes and only
    one actual map is generated at each percentage.
    """
    candidates = measured.copy().reset_index(drop=True)
    xy = candidates[["x_m", "y_m"]].to_numpy(dtype=float)
    rsrp = candidates["measured_rsrp_dbm"].to_numpy(dtype=float)
    n = len(candidates)
    if n == 0:
        raise ValueError("没有可用于自适应渐进选点的实测点")
    max_count = int(min(max(1, max_count), n))
    initial_count = int(min(max(1, initial_count), max_count))

    # Stage 0: geometry-only farthest-point sampling over the actual measurement
    # route candidate coordinates.  No signal amplitudes are used here.
    centroid = np.mean(xy, axis=0)
    first = int(np.argmin(np.sum((xy - centroid[None, :]) ** 2, axis=1)))
    selected: list[int] = [first]
    selected_set: set[int] = {first}
    min_distance = np.linalg.norm(xy - xy[first][None, :], axis=1)

    while len(selected) < initial_count:
        score = min_distance.copy()
        score[np.fromiter(selected_set, dtype=int)] = -np.inf
        idx = int(np.argmax(score))
        selected.append(idx)
        selected_set.add(idx)
        min_distance = np.minimum(
            min_distance,
            np.linalg.norm(xy - xy[idx][None, :], axis=1),
        )

    selection_scores: list[float] = [float("nan")] * len(selected)

    # Subsequent acquisitions use only RSRP from measurements that have already
    # been selected.  The candidate's own RSRP is intentionally inaccessible to
    # the scoring formula until after that candidate has been chosen.
    while len(selected) < max_count:
        sel_idx = np.asarray(selected, dtype=int)
        tree = cKDTree(xy[sel_idx])
        kq = 2 if len(sel_idx) >= 2 else 1
        distances, neighbor_local = tree.query(xy, k=kq)

        if kq == 1:
            d1 = np.asarray(distances, dtype=float)
            gradient = np.zeros(n, dtype=float)
        else:
            distances = np.asarray(distances, dtype=float)
            neighbor_local = np.asarray(neighbor_local, dtype=int)
            d1 = distances[:, 0]
            first_global = sel_idx[neighbor_local[:, 0]]
            second_global = sel_idx[neighbor_local[:, 1]]
            # Divide by second-neighbor distance to avoid overemphasizing a large
            # RSRP contrast whose supporting points are very far apart.
            gradient = np.abs(rsrp[first_global] - rsrp[second_global]) / np.maximum(
                distances[:, 1], 1.0
            )

        unselected_mask = np.ones(n, dtype=bool)
        if selected_set:
            unselected_mask[np.fromiter(selected_set, dtype=int)] = False
        valid_gradient = gradient[unselected_mask & np.isfinite(gradient)]
        scale = float(np.percentile(valid_gradient, 90)) if len(valid_gradient) else 1.0
        scale = max(scale, 1e-6)
        gradient_norm = np.clip(gradient / scale, 0.0, 2.0)

        score = d1 * (1.0 + float(gradient_weight) * gradient_norm)
        score[~unselected_mask] = -np.inf
        # Deterministic argmax; numpy returns the lowest index on exact ties.
        idx = int(np.argmax(score))
        selected.append(idx)
        selected_set.add(idx)
        selection_scores.append(float(score[idx]))

    ranked = candidates.iloc[np.asarray(selected, dtype=int)].copy().reset_index(drop=True)
    ranked["adaptive_rank"] = np.arange(1, len(ranked) + 1, dtype=int)
    ranked["selection_method"] = "adaptive_progressive_selected_rsrp_gradient"
    ranked["selection_score"] = np.asarray(selection_scores[: len(ranked)], dtype=float)
    return ranked



def adaptive_domain_nested_ranking(
    measured: pd.DataFrame,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    reference_eval_mask: np.ndarray,
    max_count: int,
    initial_count: int,
    gradient_weight: float = 0.75,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Strict nested sampling aligned with full-grid 1-NN reconstruction.

    v1.18.7 showed that geometry-only hierarchical coverage can still create a
    local RMSE increase (for the supplied PCI 558 result this occurred at
    5% -> 6%).  The reason is specific to 1-NN radio reconstruction: adding a
    point changes a Voronoi ownership region, and geometric distance alone does
    not describe whether the already observed radio field is locally smooth or
    rapidly varying.

    This selector remains free of reference/Sionna leakage while allowing a
    realistic sequential measurement strategy:

    * the first ``initial_count`` points are selected from geometry only;
    * after a point has been acquired, its measured RSRP becomes available and
      may guide the NEXT acquisition;
    * an unselected candidate's measured RSRP is never read when scoring it;
    * reference-map RSRP and Sionna RSRP are never read;
    * all percentages remain prefixes of one sequence.

    Candidate priority approximates the reduction of full-domain mean squared
    nearest distance, multiplied by an uncertainty term inferred only from the
    two nearest already-selected measurements::

        score(q) = A(q) * d(q,S)^2 * [1 + w * G(q)]

    ``A(q)`` is the valid outdoor area represented by candidate q when all
    measured candidate locations are used, ``d`` is distance to the current
    selected set, and ``G`` is a robustly normalized local RSRP-gradient proxy
    computed from already-selected points only.  The area factor aligns the
    ranking with the complete 512x512 m evaluation domain instead of the road
    candidate density alone.
    """
    candidates = measured.copy().reset_index(drop=True)
    xy = candidates[["x_m", "y_m"]].to_numpy(dtype=float)
    rsrp = candidates["measured_rsrp_dbm"].to_numpy(dtype=float)
    n = len(candidates)
    if n == 0:
        raise ValueError("没有可用于自适应全域严格嵌套选点的实测点")
    max_count = int(min(max(1, max_count), n))
    initial_count = int(min(max(1, initial_count), max_count))

    domain_xy = _domain_support_xy(
        np.asarray(x_axis, dtype=float),
        np.asarray(y_axis, dtype=float),
        np.asarray(reference_eval_mask, dtype=bool),
        max_support=50000,
    )

    # Full-domain area represented by each real measured candidate.  Only XY
    # geometry and the Boolean evaluation-domain mask enter this calculation.
    all_candidate_tree = cKDTree(xy)
    _, nearest_candidate = all_candidate_tree.query(domain_xy, k=1)
    area_weight = np.bincount(
        np.asarray(nearest_candidate, dtype=int), minlength=n
    ).astype(float)
    positive = area_weight[area_weight > 0]
    area_floor = max(1.0, 0.05 * float(np.mean(positive)) if len(positive) else 1.0)
    area_weight = area_weight + area_floor
    area_scale = max(float(np.median(area_weight)), 1e-9)
    area_norm = np.clip(np.sqrt(area_weight / area_scale), 0.35, 3.5)

    # Initial 1%: geometry only.  Start near the full-domain weighted centroid,
    # then greedily reduce area-weighted squared distance over candidate sites.
    centroid = np.average(xy, axis=0, weights=area_weight)
    first = int(np.argmin(np.sum((xy - centroid[None, :]) ** 2, axis=1)))
    selected: list[int] = [first]
    selected_set: set[int] = {first}
    min_distance = np.linalg.norm(xy - xy[first][None, :], axis=1)
    selection_score: list[float] = [float("nan")]
    selection_gradient: list[float] = [0.0]

    while len(selected) < initial_count:
        score = area_norm * np.maximum(min_distance, 0.0) ** 2
        score[np.fromiter(selected_set, dtype=int)] = -np.inf
        idx = int(np.argmax(score))
        selected.append(idx)
        selected_set.add(idx)
        selection_score.append(float(score[idx]))
        selection_gradient.append(0.0)
        min_distance = np.minimum(
            min_distance,
            np.linalg.norm(xy - xy[idx][None, :], axis=1),
        )

    audit_rows: list[dict[str, Any]] = []

    def _append_audit(stage: str) -> None:
        sel_xy = xy[np.asarray(selected, dtype=int)]
        tree = cKDTree(sel_xy)
        d, _ = tree.query(domain_xy, k=1)
        d = np.asarray(d, dtype=float)
        audit_rows.append({
            "selected_measured_point_count": int(len(selected)),
            "domain_mean_squared_nearest_distance_m2": float(np.mean(d * d)),
            "domain_nearest_distance_mean_m": float(np.mean(d)),
            "domain_nearest_distance_p90_m": float(np.percentile(d, 90)),
            "domain_nearest_distance_max_m": float(np.max(d)),
            "selection_stage": str(stage),
            "selection_uses_reference_rsrp": False,
            "selection_uses_unselected_measured_rsrp": False,
            "selection_uses_selected_measured_rsrp": bool(len(selected) > initial_count),
            "selection_uses_simulation_rsrp": False,
        })

    _append_audit("geometry_only_initialization")

    # Sequential extension.  Only RSRP values indexed by ``selected`` are read.
    while len(selected) < max_count:
        sel_idx = np.asarray(selected, dtype=int)
        sel_xy = xy[sel_idx]
        tree = cKDTree(sel_xy)
        kq = 2 if len(sel_idx) >= 2 else 1
        distances, neighbor_local = tree.query(xy, k=kq)

        if kq == 1:
            d1 = np.asarray(distances, dtype=float)
            gradient = np.zeros(n, dtype=float)
        else:
            distances = np.asarray(distances, dtype=float)
            neighbor_local = np.asarray(neighbor_local, dtype=int)
            d1 = distances[:, 0]
            g1 = sel_idx[neighbor_local[:, 0]]
            g2 = sel_idx[neighbor_local[:, 1]]
            pair_distance = np.linalg.norm(xy[g1] - xy[g2], axis=1)
            gradient = np.abs(rsrp[g1] - rsrp[g2]) / np.maximum(pair_distance, 1.0)

        unselected = np.ones(n, dtype=bool)
        if selected_set:
            unselected[np.fromiter(selected_set, dtype=int)] = False
        valid_g = gradient[unselected & np.isfinite(gradient)]
        g_scale = float(np.percentile(valid_g, 85)) if len(valid_g) else 1.0
        g_scale = max(g_scale, 1e-6)
        gradient_norm = np.clip(gradient / g_scale, 0.0, 2.5)

        # Full-domain coverage gain proxy x observed radio-transition uncertainty.
        score = area_norm * np.maximum(d1, 0.0) ** 2 * (
            1.0 + float(gradient_weight) * gradient_norm
        )
        score[~unselected] = -np.inf
        idx = int(np.argmax(score))
        if not np.isfinite(score[idx]):
            break
        selected.append(idx)
        selected_set.add(idx)
        selection_score.append(float(score[idx]))
        selection_gradient.append(float(gradient[idx]))

        # Record audit at every acquisition so exact percentage prefixes can be
        # inspected without reconstructing maps during selection.
        _append_audit("adaptive_domain_extension")

    ranked = candidates.iloc[np.asarray(selected, dtype=int)].copy().reset_index(drop=True)
    ranked["adaptive_domain_rank"] = np.arange(1, len(ranked) + 1, dtype=int)
    ranked["selection_method"] = "adaptive_domain_nested_selected_rsrp_fullgrid_geometry"
    ranked["selection_score"] = np.asarray(selection_score[: len(ranked)], dtype=float)
    ranked["selection_gradient_db_per_m"] = np.asarray(selection_gradient[: len(ranked)], dtype=float)
    ranked["candidate_fullgrid_area_weight"] = area_weight[np.asarray(selected, dtype=int)]

    audit = pd.DataFrame(audit_rows)
    audit["sampling_sets_nested"] = True
    return ranked, audit


def voronoi_safe_adaptive_nested_ranking(
    measured: pd.DataFrame,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    reference_eval_mask: np.ndarray,
    stage_target_counts: list[int],
    gradient_weight: float = 0.55,
    containment_strength: float = 1.15,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Leakage-safe strict-nested acquisition for full-grid 1-NN reconstruction.

    Motivation
    ----------
    A new 1-NN sample can reduce geometric distance while *increasing* full-grid
    RMSE because it may seize a large Voronoi region with a locally unusual
    RSRP.  Geometry-only selectors cannot see that effect.  This selector keeps
    exact 1-NN reconstruction and strict nesting, but changes only the sequential
    acquisition policy.

    Data-access rule
    ----------------
    * Candidate coordinates and the Boolean full-grid evaluation mask are known.
    * The RSRP of a candidate is NOT used before that candidate is selected.
    * Once a point is acquired, its measured RSRP may guide subsequent points.
    * Reference-map RSRP, Sionna RSRP, final RMSE, and unselected measured RSRP
      are never used for selection.

    Stage-wise safety
    -----------------
    Each requested percentage is treated as one acquisition stage.  Early slots
    favour full-domain coverage and already-observed radio transitions.  If a
    newly acquired point differs strongly from the previously controlling
    measured neighbour, the next slot(s) in the SAME stage are forced to
    add nearby containment points so that the anomalous value cannot own a very
    large 1-NN region.  The final slot of each stage is selected conservatively
    from a smooth, moderate-takeover part of the candidate pool, reducing the
    chance that the stage ends on a high-impact uncontained point.

    This is a stability-oriented sampling policy; it does not inspect the target
    map and therefore does not mathematically force RMSE monotonicity.
    """
    candidates = measured.copy().reset_index(drop=True)
    xy = candidates[["x_m", "y_m"]].to_numpy(dtype=float)
    rsrp = candidates["measured_rsrp_dbm"].to_numpy(dtype=float)
    n = len(candidates)
    if n == 0:
        raise ValueError("没有可用于Voronoi安全严格嵌套选点的实测点")

    targets = sorted({int(min(max(1, c), n)) for c in stage_target_counts})
    if not targets:
        raise ValueError("stage_target_counts不能为空")
    max_count = int(max(targets))
    initial_count = int(min(targets))

    domain_xy = _domain_support_xy(
        np.asarray(x_axis, dtype=float),
        np.asarray(y_axis, dtype=float),
        np.asarray(reference_eval_mask, dtype=bool),
        max_support=50000,
    )

    # Geometry-only representation area of each candidate over the complete
    # evaluation domain.  Only the Boolean validity mask is used here.
    all_tree = cKDTree(xy)
    _, owner = all_tree.query(domain_xy, k=1)
    area_weight = np.bincount(np.asarray(owner, dtype=int), minlength=n).astype(float)
    positive = area_weight[area_weight > 0]
    area_floor = max(1.0, 0.05 * float(np.mean(positive)) if len(positive) else 1.0)
    area_weight = area_weight + area_floor
    area_scale = max(float(np.median(area_weight)), 1e-9)
    area_norm = np.clip(np.sqrt(area_weight / area_scale), 0.35, 3.5)

    # 1% initialization is geometry only.
    centroid = np.average(xy, axis=0, weights=area_weight)
    first = int(np.argmin(np.sum((xy - centroid[None, :]) ** 2, axis=1)))
    selected: list[int] = [first]
    selected_set: set[int] = {first}
    rank_score: list[float] = [float("nan")]
    rank_mode: list[str] = ["geometry_initial"]
    observed_jump: list[float] = [0.0]
    parent_distance_at_acquisition: list[float] = [0.0]

    min_d = np.linalg.norm(xy - xy[first][None, :], axis=1)
    while len(selected) < initial_count:
        score = area_norm * np.maximum(min_d, 0.0) ** 2
        score[np.fromiter(selected_set, dtype=int)] = -np.inf
        idx = int(np.argmax(score))
        parent_distance = float(min_d[idx])
        selected.append(idx)
        selected_set.add(idx)
        rank_score.append(float(score[idx]))
        rank_mode.append("geometry_initial")
        observed_jump.append(0.0)
        parent_distance_at_acquisition.append(parent_distance)
        min_d = np.minimum(min_d, np.linalg.norm(xy - xy[idx][None, :], axis=1))

    audit_rows: list[dict[str, Any]] = []

    def append_audit(stage_target: int, stage_label: str, hazards_active: int) -> None:
        sel_xy = xy[np.asarray(selected, dtype=int)]
        tree = cKDTree(sel_xy)
        d, _ = tree.query(domain_xy, k=1)
        d = np.asarray(d, dtype=float)
        audit_rows.append({
            "selected_measured_point_count": int(len(selected)),
            "stage_target_count": int(stage_target),
            "domain_mean_squared_nearest_distance_m2": float(np.mean(d * d)),
            "domain_nearest_distance_mean_m": float(np.mean(d)),
            "domain_nearest_distance_p90_m": float(np.percentile(d, 90)),
            "domain_nearest_distance_max_m": float(np.max(d)),
            "active_voronoi_hazards": int(hazards_active),
            "selection_stage": str(stage_label),
            "selection_uses_reference_rsrp": False,
            "selection_uses_unselected_measured_rsrp": False,
            "selection_uses_selected_measured_rsrp": bool(len(selected) > initial_count),
            "selection_uses_simulation_rsrp": False,
            "selection_uses_final_rmse": False,
        })

    append_audit(initial_count, "geometry_only_initialization", 0)

    # Hazard records are created only AFTER a sample is acquired and its label
    # becomes available.  They live only inside the current percentage stage.
    for stage_target in targets:
        if stage_target <= len(selected):
            continue
        stage_start = len(selected)
        stage_budget = int(stage_target - stage_start)
        hazards: list[dict[str, float]] = []

        while len(selected) < stage_target:
            sel_idx = np.asarray(selected, dtype=int)
            sel_xy = xy[sel_idx]
            tree = cKDTree(sel_xy)
            kq = 2 if len(sel_idx) >= 2 else 1
            distances, local_idx = tree.query(xy, k=kq)
            if kq == 1:
                d1 = np.asarray(distances, dtype=float)
                parent_global = np.full(n, sel_idx[0], dtype=int)
                gradient = np.zeros(n, dtype=float)
            else:
                distances = np.asarray(distances, dtype=float)
                local_idx = np.asarray(local_idx, dtype=int)
                d1 = distances[:, 0]
                parent_global = sel_idx[local_idx[:, 0]]
                g1 = sel_idx[local_idx[:, 0]]
                g2 = sel_idx[local_idx[:, 1]]
                pair_d = np.linalg.norm(xy[g1] - xy[g2], axis=1)
                gradient = np.abs(rsrp[g1] - rsrp[g2]) / np.maximum(pair_d, 1.0)

            unselected = np.ones(n, dtype=bool)
            if selected_set:
                unselected[np.fromiter(selected_set, dtype=int)] = False

            valid_g = gradient[unselected & np.isfinite(gradient)]
            g_scale = float(np.percentile(valid_g, 85)) if len(valid_g) else 1.0
            g_scale = max(g_scale, 1e-6)
            gnorm = np.clip(gradient / g_scale, 0.0, 2.5)

            coverage = area_norm * np.maximum(d1, 0.0) ** 2
            cov_valid = coverage[unselected & np.isfinite(coverage)]
            cscale = float(np.percentile(cov_valid, 90)) if len(cov_valid) else 1.0
            cscale = max(cscale, 1e-9)
            risk_norm = np.clip(coverage / cscale, 0.0, 3.0)

            added = len(selected) - stage_start
            remaining = stage_target - len(selected)
            stage_progress = added / max(stage_budget, 1)

            # Standard exploration score.  Close to stage end, gradually prefer
            # regions inferred smooth from already acquired measurements and
            # avoid a final unknown point with a huge geometric takeover area.
            closure = stage_progress ** 2
            score = coverage * (1.0 + float(gradient_weight) * (1.0 - 0.65 * closure) * gnorm)
            score /= (1.0 + 0.65 * closure * risk_norm * gnorm)
            mode = "coverage_transition"

            # v1.18.11: high-jump points are not merely *encouraged* to receive
            # containment neighbours.  While an unresolved hazard exists, the
            # next acquisition is forced to be a geometry-only containment point.
            # This fixes the v1.18.10 failure mode where a 17 dB jump was detected
            # but the following points were still ordinary coverage transitions.
            # No unselected RSRP/reference/Sionna value is inspected here.
            active_hazards = [h for h in hazards if h["remaining"] > 0.0]
            forced_hazard = None
            forced_containment_score = None
            if active_hazards and remaining >= 1:
                # Resolve the strongest unresolved hazard first.  "strength" is
                # based only on an RSRP jump that has already been observed.
                forced_hazard = max(
                    active_hazards,
                    key=lambda h: float(h["strength"]) * float(h["remaining"]),
                )
                center = xy[int(forced_hazard["idx"])]
                parent_xy = xy[int(forced_hazard["parent_idx"])]
                hd_vec = xy - center[None, :]
                hd = np.linalg.norm(hd_vec, axis=1)
                radius = max(float(forced_hazard["radius"]), 6.0)

                # Prefer a broad annulus around the acquired high-jump point.
                # Points too close to the hazard do not split its Voronoi region;
                # points too far away do not contain it effectively.
                target_r = 0.48 * radius
                sigma_r = max(0.24 * radius, 3.0)
                ring = np.exp(-((hd - target_r) / sigma_r) ** 2)
                ring[(hd < max(3.0, 0.15 * radius)) | (hd > 1.05 * radius)] = 0.0

                # First containment point is preferably placed away from the old
                # controller.  A second containment point favours angular diversity
                # from the first, so the hazard is bracketed rather than sampled on
                # only one side.  These terms use coordinates only.
                away = center - parent_xy
                away_norm = float(np.linalg.norm(away))
                if away_norm > 1e-9:
                    away = away / away_norm
                    cand_unit = hd_vec / np.maximum(hd[:, None], 1e-9)
                    away_dot = np.clip(cand_unit @ away, -1.0, 1.0)
                    direction_bonus = 1.0 + 0.55 * np.maximum(away_dot, 0.0)
                else:
                    cand_unit = hd_vec / np.maximum(hd[:, None], 1e-9)
                    direction_bonus = np.ones(n, dtype=float)

                used_dirs = forced_hazard.get("containment_dirs", [])
                if used_dirs:
                    # Reward directions that are different from all previously
                    # chosen containment directions for this hazard.
                    max_abs_dot = np.zeros(n, dtype=float)
                    for ud in used_dirs:
                        ud = np.asarray(ud, dtype=float)
                        max_abs_dot = np.maximum(max_abs_dot, np.abs(cand_unit @ ud))
                    direction_bonus *= 1.0 + 0.70 * (1.0 - max_abs_dot)

                forced_containment_score = (
                    ring
                    * direction_bonus
                    * (1.0 + 0.20 * np.clip(area_norm, 0.0, 3.5))
                    * (1.0 + 0.10 * np.clip(coverage / cscale, 0.0, 3.0))
                )
                forced_containment_score[~unselected] = -np.inf

                # If road geometry leaves no candidate in the preferred annulus,
                # fall back to the nearest useful unselected candidate within a
                # relaxed radius.  Only if even that is impossible is the hazard
                # retired and normal coverage selection resumed.
                if not np.any(np.isfinite(forced_containment_score) & (forced_containment_score > 0.0)):
                    relaxed = unselected & np.isfinite(hd) & (hd >= 2.5) & (hd <= 1.50 * radius)
                    if np.any(relaxed):
                        fallback = np.full(n, -np.inf, dtype=float)
                        fallback[relaxed] = 1.0 / np.maximum(np.abs(hd[relaxed] - target_r), 1.0)
                        forced_containment_score = fallback
                    else:
                        forced_hazard["remaining"] = 0.0
                        forced_hazard = None
                        forced_containment_score = None

            if forced_hazard is not None and forced_containment_score is not None:
                score = forced_containment_score
                mode = "forced_voronoi_containment"
            else:
                # Final slot of a hazard-free stage is conservative: choose from
                # useful coverage candidates while minimising already-observed
                # transition risk.  This still cannot inspect the unknown label of
                # the final point and therefore does not force monotonic RMSE.
                if remaining == 1:
                    useful = unselected & np.isfinite(coverage)
                    if np.any(useful):
                        floor = float(np.percentile(coverage[useful], 50))
                        pool = useful & (coverage >= floor)
                        safe = coverage / (1.0 + 2.2 * gnorm + 1.0 * risk_norm)
                        safe[~pool] = -np.inf
                        if np.any(np.isfinite(safe)):
                            score = safe
                            mode = "stage_safe_closure"

            score[~unselected] = -np.inf
            idx = int(np.argmax(score))
            if not np.isfinite(score[idx]):
                break

            # Parent and distance are determined BEFORE the new label is read.
            parent = int(parent_global[idx])
            p_dist = float(d1[idx])
            selected.append(idx)
            selected_set.add(idx)
            rank_score.append(float(score[idx]))
            rank_mode.append(mode)

            # The label becomes available only now.  Compare it with the
            # previous controlling measured point and, when needed, create a
            # containment hazard for subsequent acquisitions in this stage.
            jump = float(abs(rsrp[idx] - rsrp[parent])) if parent != idx else 0.0
            observed_jump.append(jump)
            parent_distance_at_acquisition.append(p_dist)

            prior_jumps = np.asarray([j for j in observed_jump[:-1] if np.isfinite(j) and j > 0], dtype=float)
            if len(prior_jumps) >= 4:
                med = float(np.median(prior_jumps))
                mad = float(np.median(np.abs(prior_jumps - med)))
                jump_threshold = max(7.0, med + 1.5 * max(mad, 1.0))
            else:
                jump_threshold = 8.0

            # If this acquisition was forced containment, retire one unit of
            # the hazard and remember the selected direction.  Do this before
            # creating any new hazard from the newly observed point.
            if mode == "forced_voronoi_containment" and forced_hazard is not None:
                forced_hazard["remaining"] = max(0.0, float(forced_hazard["remaining"]) - 1.0)
                direction = xy[idx] - xy[int(forced_hazard["idx"])]
                dn = float(np.linalg.norm(direction))
                if dn > 1e-9:
                    forced_hazard.setdefault("containment_dirs", []).append((direction / dn).tolist())

            slots_left_after = int(stage_target - len(selected))
            if jump >= jump_threshold and p_dist >= 8.0 and slots_left_after > 0:
                strength = min(3.0, jump / max(jump_threshold, 1e-6))
                # Strong jumps receive two forced neighbours whenever the stage
                # budget permits; moderate jumps receive one.  This makes the
                # containment response deterministic rather than a soft bonus.
                desired = 2 if (jump >= 1.25 * jump_threshold and slots_left_after >= 2) else 1
                hazards.append({
                    "idx": float(idx),
                    "parent_idx": float(parent),
                    "radius": float(np.clip(0.95 * p_dist, 8.0, 55.0)),
                    "strength": float(strength),
                    "remaining": float(min(desired, slots_left_after)),
                    "containment_dirs": [],
                })

            append_audit(stage_target, mode, sum(h["remaining"] > 0 for h in hazards))

    ranked = candidates.iloc[np.asarray(selected, dtype=int)].copy().reset_index(drop=True)
    ranked["voronoi_safe_rank"] = np.arange(1, len(ranked) + 1, dtype=int)
    ranked["selection_method"] = "forced_containment_voronoi_safe_nested_selected_rsrp_only"
    ranked["selection_score"] = np.asarray(rank_score[: len(ranked)], dtype=float)
    ranked["selection_mode_at_acquisition"] = rank_mode[: len(ranked)]
    ranked["observed_jump_from_previous_controller_db"] = np.asarray(observed_jump[: len(ranked)], dtype=float)
    ranked["distance_to_previous_controller_m"] = np.asarray(parent_distance_at_acquisition[: len(ranked)], dtype=float)
    ranked["candidate_fullgrid_area_weight"] = area_weight[np.asarray(selected, dtype=int)]

    audit = pd.DataFrame(audit_rows)
    audit["sampling_sets_nested"] = True
    return ranked, audit

def _domain_support_xy(
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    reference_eval_mask: np.ndarray,
    max_support: int = 60000,
) -> np.ndarray:
    """Return a deterministic geometry-only support set from the valid full-grid domain."""
    xx, yy = np.meshgrid(np.asarray(x_axis, dtype=float), np.asarray(y_axis, dtype=float))
    valid = np.asarray(reference_eval_mask, dtype=bool).ravel()
    support = np.column_stack([xx.ravel()[valid], yy.ravel()[valid]])
    if len(support) == 0:
        raise ValueError("参考地图没有有效室外区域")
    stride = max(1, int(math.ceil(len(support) / float(max_support))))
    return support[::stride].copy()


def _coverage_objective(
    domain_xy: np.ndarray,
    candidate_xy: np.ndarray,
    selected_idx: np.ndarray,
) -> tuple[float, float, float, float]:
    """Geometry-only score; no RSRP amplitude is used."""
    selected_idx = np.asarray(selected_idx, dtype=int)
    tree = cKDTree(candidate_xy[selected_idx])
    d, _ = tree.query(domain_xy, k=1)
    d = np.asarray(d, dtype=float)
    mean_d2 = float(np.mean(d * d))
    mean_d = float(np.mean(d))
    p90 = float(np.percentile(d, 90))
    max_d = float(np.max(d))
    objective = mean_d2 + 0.20 * p90 * p90 + 0.03 * max_d * max_d
    return float(objective), mean_d, p90, max_d




def _deterministic_weighted_kmeans_centers(
    candidate_xy: np.ndarray,
    candidate_weights: np.ndarray,
    k: int,
    max_iter: int = 100,
) -> np.ndarray:
    """Deterministic weighted Lloyd centers for geometry-only percentage designs.

    scikit-learn KMeans with k-means++ can choose different but equivalent
    solutions on symmetric layouts because of numerical tie ordering.  The
    reconstruction experiment needs one reproducible map per percentage, so the
    initialization and every tie are resolved deterministically here.  Only XY
    coordinates and geometry weights are used.
    """
    xy = np.asarray(candidate_xy, dtype=float)
    w = np.asarray(candidate_weights, dtype=float).reshape(-1)
    n = len(xy)
    k = int(min(max(1, k), n))
    if n == 0:
        raise ValueError("没有候选实测点")

    weighted_centroid = np.average(xy, axis=0, weights=np.maximum(w, 1e-12))
    first = int(np.argmin(np.sum((xy - weighted_centroid[None, :]) ** 2, axis=1)))
    chosen = [first]
    min_d2 = np.sum((xy - xy[first][None, :]) ** 2, axis=1)
    while len(chosen) < k:
        score = np.maximum(w, 1e-12) * min_d2
        score[np.asarray(chosen, dtype=int)] = -np.inf
        idx = int(np.argmax(score))
        chosen.append(idx)
        d2 = np.sum((xy - xy[idx][None, :]) ** 2, axis=1)
        min_d2 = np.minimum(min_d2, d2)

    centers = xy[np.asarray(chosen, dtype=int)].copy()
    for _ in range(int(max_iter)):
        d2 = ((xy[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = np.argmin(d2, axis=1)
        updated = centers.copy()
        for j in range(k):
            mask = labels == j
            if np.any(mask):
                updated[j] = np.average(xy[mask], axis=0, weights=np.maximum(w[mask], 1e-12))
        if np.max(np.abs(updated - centers)) < 1e-8:
            centers = updated
            break
        centers = updated
    return centers

def _snap_centers_to_unique_measurements(
    centers: np.ndarray,
    candidate_xy: np.ndarray,
) -> np.ndarray:
    """Assign every geometric center to a unique real measured location."""
    centers = np.asarray(centers, dtype=float)
    candidate_xy = np.asarray(candidate_xy, dtype=float)
    # k x N with k<=94 and N~938 is small; Hungarian assignment gives a
    # deterministic globally consistent snap and avoids duplicate measured points.
    cost = distance.cdist(centers, candidate_xy, metric="sqeuclidean")
    row_ind, col_ind = linear_sum_assignment(cost)
    order = np.argsort(row_ind)
    return np.asarray(col_ind[order], dtype=int)


def _expand_previous_geometry_only(
    previous_idx: np.ndarray,
    candidate_xy: np.ndarray,
    candidate_weights: np.ndarray,
    target_count: int,
) -> np.ndarray:
    """Add centers to the previous set without ever worsening geometric coverage."""
    selected = [int(i) for i in np.asarray(previous_idx, dtype=int)]
    selected_set = set(selected)
    while len(selected) < int(target_count):
        tree = cKDTree(candidate_xy[np.asarray(selected, dtype=int)])
        d, _ = tree.query(candidate_xy, k=1)
        score = np.asarray(candidate_weights, dtype=float) * np.asarray(d, dtype=float) ** 2
        if selected_set:
            score[np.fromiter(selected_set, dtype=int)] = -np.inf
        idx = int(np.argmax(score))
        if not np.isfinite(score[idx]):
            break
        selected.append(idx)
        selected_set.add(idx)
    return np.asarray(selected[: int(target_count)], dtype=int)


def stable_spatial_percentage_sets(
    measured: pd.DataFrame,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    reference_eval_mask: np.ndarray,
    percentages: list[int],
) -> tuple[dict[int, pd.DataFrame], pd.DataFrame]:
    """Build one actual, deterministic spatial design for each sampling ratio.

    v1.18.4 forced 1%--10% to be prefixes of one strict nested 1-NN sequence.
    That is exactly why 9% -> 10% can get worse: a newly added measurement can
    take over a Voronoi region even though all nearest-distance coverage metrics
    improve.  Strict 1-NN has no monotonic-error theorem.

    v1.18.6 therefore changes only the *sampling design*, not the interpolation:

    1. The valid 512x512 outdoor grid is mapped to its nearest real measured
       candidate, producing an area weight for every candidate location.
    2. For each percentage independently, weighted k-means partitions those
       candidate locations into K spatial strata.
    3. The K geometric centers are globally snapped to K unique real measured
       locations using the Hungarian assignment.
    4. The previous percentage expanded by geometry-only farthest additions is
       also considered.  The lower geometric-coverage objective wins.

    The selected set is identical for M and M+S.  Selection never reads reference
    RSRP, measured RSRP amplitude, or Sionna RSRP.  There is one reconstruction
    map per percentage, no trial averaging, no best-seed selection, and no RMSE
    monotonic post-processing.
    """
    candidates = measured.copy().reset_index(drop=True)
    candidate_xy = candidates[["x_m", "y_m"]].to_numpy(dtype=float)
    if len(candidate_xy) == 0:
        raise ValueError("没有可用于稳定空间选点的实测点")

    domain_xy = _domain_support_xy(x_axis, y_axis, reference_eval_mask)
    candidate_tree = cKDTree(candidate_xy)
    _, nearest_candidate = candidate_tree.query(domain_xy, k=1)
    area_weight = np.bincount(
        np.asarray(nearest_candidate, dtype=int), minlength=len(candidate_xy)
    ).astype(float)
    positive = area_weight[area_weight > 0]
    route_floor = max(1.0, 0.10 * float(np.mean(positive)) if len(positive) else 1.0)
    candidate_weights = area_weight + route_floor

    result: dict[int, pd.DataFrame] = {}
    audit_rows: list[dict[str, Any]] = []
    previous_idx: np.ndarray | None = None
    previous_obj = float("inf")

    for pct in sorted(int(p) for p in percentages):
        k = min(
            len(candidates),
            max(1, int(math.ceil(len(candidates) * float(pct) / 100.0))),
        )

        centers = _deterministic_weighted_kmeans_centers(
            candidate_xy=candidate_xy,
            candidate_weights=candidate_weights,
            k=int(k),
        )
        kmeans_idx = _snap_centers_to_unique_measurements(centers, candidate_xy)
        kmeans_metrics = _coverage_objective(domain_xy, candidate_xy, kmeans_idx)

        best_idx = kmeans_idx
        best_metrics = kmeans_metrics
        source = "weighted_kmeans_hungarian"

        if previous_idx is not None:
            expanded_idx = _expand_previous_geometry_only(
                previous_idx=previous_idx,
                candidate_xy=candidate_xy,
                candidate_weights=candidate_weights,
                target_count=k,
            )
            expanded_metrics = _coverage_objective(domain_xy, candidate_xy, expanded_idx)
            if expanded_metrics[0] < best_metrics[0] - 1e-9:
                best_idx = expanded_idx
                best_metrics = expanded_metrics
                source = "previous_set_plus_geometry_farthest"

        # Geometry objective must not increase with sampling density.  If a
        # numerical/local-minimum issue occurs, the expanded previous set is a
        # valid K-point superset and cannot increase nearest-distance coverage.
        if previous_idx is not None and best_metrics[0] > previous_obj + 1e-8:
            fallback_idx = _expand_previous_geometry_only(
                previous_idx=previous_idx,
                candidate_xy=candidate_xy,
                candidate_weights=candidate_weights,
                target_count=k,
            )
            fallback_metrics = _coverage_objective(domain_xy, candidate_xy, fallback_idx)
            best_idx, best_metrics = fallback_idx, fallback_metrics
            source = "geometry_monotone_fallback"

        subset = candidates.iloc[np.asarray(best_idx, dtype=int)].copy()
        subset = subset.sort_values(["iy", "ix"]).reset_index(drop=True)
        subset["sampling_percent"] = int(pct)
        subset["within_subset_rank"] = np.arange(1, len(subset) + 1, dtype=int)
        subset["selection_method"] = "stable_spatial_weighted_kmeans"
        result[int(pct)] = subset

        previous_set = set(int(i) for i in previous_idx) if previous_idx is not None else set()
        current_set = set(int(i) for i in np.asarray(best_idx, dtype=int))
        audit_rows.append({
            "sampling_percent": int(pct),
            "selected_measured_point_count": int(len(subset)),
            "geometry_objective_m2": float(best_metrics[0]),
            "domain_nearest_distance_mean_m": float(best_metrics[1]),
            "domain_nearest_distance_p90_m": float(best_metrics[2]),
            "domain_nearest_distance_max_m": float(best_metrics[3]),
            "design_source": source,
            "retained_from_previous_count": int(len(previous_set & current_set)),
            "selection_uses_reference_rsrp": False,
            "selection_uses_measured_rsrp": False,
            "selection_uses_simulation_rsrp": False,
        })
        previous_idx = np.asarray(best_idx, dtype=int)
        previous_obj = float(best_metrics[0])

    return result, pd.DataFrame(audit_rows)


# Backward-compatible name for older notebooks/tests.  The legacy v1.18.4 path
# uses progressive_coverage_nested_ranking directly.

def hierarchical_nested_spatial_ranking(
    measured: pd.DataFrame,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    reference_eval_mask: np.ndarray,
    min_count: int,
    max_count: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Geometry-only strict nested sequence designed jointly across 1%--10%.

    The key change from v1.18.6 is that every sampling density is no longer
    optimized independently.  Instead, one globally balanced ``max_count`` set
    is first constructed on the valid 512 m x 512 m outdoor domain, and lower
    densities are obtained by deterministic *backward elimination*.  Therefore

        S_min subset ... subset S_max

    holds exactly, while every prefix is derived from the same final spatial
    design.  This avoids the confounding in v1.18.6 where both sample count and
    all sample locations changed between adjacent percentages.

    Selection is deliberately label-free: only candidate XY coordinates and the
    Boolean valid-domain geometry are used.  Neither reference-map RSRP,
    measured RSRP amplitudes, nor Sionna RSRP enters the ranking.

    Backward elimination removes, one at a time, the point whose removal causes
    the smallest increase in full-domain mean squared nearest distance.  This is
    aligned with 1-NN's spatial support while remaining independent of radio
    labels.  It is a sampling-design heuristic, not an RMSE monotonicity
    post-processing rule; strict 1-NN still has no theorem guaranteeing that
    every realized radio-field RMSE step must decrease.
    """
    candidates = measured.copy().reset_index(drop=True)
    candidate_xy = candidates[["x_m", "y_m"]].to_numpy(dtype=float)
    n = len(candidate_xy)
    if n == 0:
        raise ValueError("没有可用于严格嵌套空间选点的实测点")
    min_count = int(min(max(1, min_count), n))
    max_count = int(min(max(min_count, max_count), n))

    domain_xy = _domain_support_xy(
        x_axis=np.asarray(x_axis, dtype=float),
        y_axis=np.asarray(y_axis, dtype=float),
        reference_eval_mask=np.asarray(reference_eval_mask, dtype=bool),
        max_support=40000,
    )

    # Geometry weights: each candidate receives the outdoor area for which it is
    # the nearest available measured location, plus a small route floor so sparse
    # route segments are not ignored.
    candidate_tree = cKDTree(candidate_xy)
    _, nearest_candidate = candidate_tree.query(domain_xy, k=1)
    area_weight = np.bincount(
        np.asarray(nearest_candidate, dtype=int), minlength=n
    ).astype(float)
    positive = area_weight[area_weight > 0]
    route_floor = max(1.0, 0.08 * float(np.mean(positive)) if len(positive) else 1.0)
    candidate_weights = area_weight + route_floor

    # Build one globally balanced maximum-density set.
    centers = _deterministic_weighted_kmeans_centers(
        candidate_xy=candidate_xy,
        candidate_weights=candidate_weights,
        k=max_count,
    )
    max_idx = _snap_centers_to_unique_measurements(centers, candidate_xy)
    current = [int(i) for i in np.asarray(max_idx, dtype=int)]
    if len(set(current)) != max_count:
        raise RuntimeError("最大密度空间集合包含重复实测点")

    # Precompute distances only to the final max-count set.  All lower-density
    # sets are subsets of this set, so backward elimination is efficient and
    # deterministic.
    dmat = distance.cdist(domain_xy, candidate_xy[np.asarray(current, dtype=int)])
    active_cols = list(range(max_count))
    removal_order_cols: list[int] = []
    audit_rows: list[dict[str, Any]] = []

    def _geom_stats(cols: list[int]) -> tuple[float, float, float, float]:
        d = np.min(dmat[:, np.asarray(cols, dtype=int)], axis=1)
        mean_d2 = float(np.mean(d * d))
        mean_d = float(np.mean(d))
        p90 = float(np.percentile(d, 90))
        max_d = float(np.max(d))
        return mean_d2, mean_d, p90, max_d

    # Record the max-density geometry before pruning.
    g = _geom_stats(active_cols)
    audit_rows.append({
        "selected_measured_point_count": int(len(active_cols)),
        "domain_mean_squared_nearest_distance_m2": g[0],
        "domain_nearest_distance_mean_m": g[1],
        "domain_nearest_distance_p90_m": g[2],
        "domain_nearest_distance_max_m": g[3],
        "selection_stage": "max_density_weighted_centroidal_design",
    })

    # Remove down to min_count.  At each step, the primary objective is the
    # resulting mean squared nearest distance over the full outdoor support.
    # P90 and max distance are deterministic tie-breakers only.
    while len(active_cols) > min_count:
        active_arr = np.asarray(active_cols, dtype=int)
        sub = dmat[:, active_arr]
        if sub.shape[1] < 2:
            break
        nearest_local = np.argmin(sub, axis=1)
        nearest_d = sub[np.arange(len(sub)), nearest_local]
        # second-nearest distance for the points owned by a removed site
        second_d = np.partition(sub, 1, axis=1)[:, 1]
        base_sumsq = float(np.sum(nearest_d * nearest_d))

        best_key = None
        best_col = None
        for local_j, col in enumerate(active_cols):
            owned = nearest_local == local_j
            new_sumsq = base_sumsq
            if np.any(owned):
                new_sumsq += float(np.sum(second_d[owned] ** 2 - nearest_d[owned] ** 2))
            mean_d2 = new_sumsq / float(len(domain_xy))
            # Cheap secondary geometry terms are evaluated only on the affected
            # ownership region; they stabilize ties without label information.
            owned_count = int(np.sum(owned))
            max_replacement = float(np.max(second_d[owned])) if owned_count else 0.0
            original_candidate_idx = int(current[col])
            key = (mean_d2, max_replacement, owned_count, original_candidate_idx)
            if best_key is None or key < best_key:
                best_key = key
                best_col = int(col)

        if best_col is None:
            raise RuntimeError("严格嵌套反向消元未能选择可删除点")
        active_cols.remove(best_col)
        removal_order_cols.append(best_col)
        g = _geom_stats(active_cols)
        audit_rows.append({
            "selected_measured_point_count": int(len(active_cols)),
            "domain_mean_squared_nearest_distance_m2": g[0],
            "domain_nearest_distance_mean_m": g[1],
            "domain_nearest_distance_p90_m": g[2],
            "domain_nearest_distance_max_m": g[3],
            "selection_stage": "backward_geometry_elimination",
        })

    # ``active_cols`` are the min-count points.  Adding the removed points in
    # reverse order exactly reconstructs every larger strict-nested prefix up to
    # the globally designed max-count set.
    add_cols = list(active_cols) + list(reversed(removal_order_cols))
    if len(add_cols) != max_count or len(set(add_cols)) != max_count:
        raise RuntimeError("严格嵌套空间排序构造失败")
    ranked_candidate_idx = [current[col] for col in add_cols]
    ranked = candidates.iloc[np.asarray(ranked_candidate_idx, dtype=int)].copy().reset_index(drop=True)
    ranked["hierarchical_rank"] = np.arange(1, len(ranked) + 1, dtype=int)
    ranked["selection_method"] = "hierarchical_nested_geometry_backward_elimination"

    audit = pd.DataFrame(audit_rows).sort_values(
        "selected_measured_point_count"
    ).reset_index(drop=True)
    audit["selection_uses_reference_rsrp"] = False
    audit["selection_uses_measured_rsrp"] = False
    audit["selection_uses_simulation_rsrp"] = False
    audit["sampling_sets_nested"] = True
    return ranked, audit


def coverage_aware_nested_ranking(
    measured: pd.DataFrame,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    reference_eval_mask: np.ndarray,
    max_count: int,
    seed: int = 20260805,
) -> pd.DataFrame:
    del seed
    return progressive_coverage_nested_ranking(
        measured=measured,
        x_axis=x_axis,
        y_axis=y_axis,
        reference_eval_mask=reference_eval_mask,
        max_count=max_count,
    )

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
) -> tuple[np.ndarray, dict[str, Any]]:
    """Calibrate the simulation trend and interpolate only its measured residual."""
    train_xy = np.asarray(train_xy, dtype=float)
    train_y = np.asarray(train_y, dtype=float)
    query_xy = np.asarray(query_xy, dtype=float)
    baseline = np.asarray(baseline_prediction, dtype=float).reshape(-1)
    prior_train = np.asarray(prior.sample(train_xy), dtype=float)
    valid_train = np.isfinite(prior_train) & np.isfinite(train_y)
    if int(valid_train.sum()) < 2:
        return baseline.copy(), {
            "simulation_prior_training_count": int(valid_train.sum()),
            "affine_prior_slope": 1.0,
            "affine_prior_intercept_db": 0.0,
            "simulation_fallback_fraction": 1.0,
        }

    slope, intercept = robust_affine_calibration(prior_train[valid_train], train_y[valid_train])
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
        "affine_prior_slope": float(slope),
        "affine_prior_intercept_db": float(intercept),
        "training_residual_median_db": float(np.median(residual)),
        "training_residual_mad_db": float(np.median(np.abs(residual - np.median(residual)))),
        "simulation_fallback_fraction": float(1.0 - np.mean(valid_query)),
    }


def fixed_sionna_residual_nearest_neighbor(
    *,
    prior,
    train_xy: np.ndarray,
    train_y: np.ndarray,
    query_xy: np.ndarray,
    baseline_prediction: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Strict M+S branch: fixed Sionna prior + nearest-neighbor residual.

    No affine refit, no Kriging/IDW, no learned fusion weight, and no use of
    reference-map RSRP for fitting.  The exact same selected measured points as
    branch M are used.
    """
    train_xy = np.asarray(train_xy, dtype=float)
    train_y = np.asarray(train_y, dtype=float)
    query_xy = np.asarray(query_xy, dtype=float)
    baseline = np.asarray(baseline_prediction, dtype=float).reshape(-1)

    prior_train = np.asarray(prior.sample(train_xy), dtype=float)
    valid_train = np.isfinite(prior_train) & np.isfinite(train_y)
    if int(valid_train.sum()) < 1:
        return baseline.copy(), {
            "simulation_prior_training_count": 0,
            "simulation_fallback_fraction": 1.0,
            "residual_median_db": float("nan"),
            "residual_mad_db": float("nan"),
        }

    residual = train_y[valid_train] - prior_train[valid_train]
    residual_tree = cKDTree(train_xy[valid_train])
    _, nearest = residual_tree.query(query_xy, k=1)
    prior_query = np.asarray(prior.sample(query_xy), dtype=float)
    corrected = prior_query + residual[np.asarray(nearest, dtype=int)]
    valid_query = np.isfinite(corrected)

    prediction = baseline.copy()
    prediction[valid_query] = corrected[valid_query]
    med = float(np.median(residual))
    return prediction, {
        "simulation_prior_training_count": int(valid_train.sum()),
        "simulation_fallback_fraction": float(1.0 - np.mean(valid_query)),
        "residual_median_db": med,
        "residual_mad_db": float(np.median(np.abs(residual - med))),
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
        "两组使用相同的候选实测位置、同一条单次渐进嵌套采样序列、建筑物掩膜、评价域和Measurement-filled radio map参考。每个采样比例只生成一次重构地图；唯一差别是联合组是否使用固定纯Sionna RT地图。",
        "",
        f"- 1%--10%的 {improved}/{len(paired)} 个采样比例中，联合组RMSE低于仅实测组。",
        f"- 十种比例平均RMSE：仅实测 {baseline_mean:.2f} dB，实测+仿真 {assisted_mean:.2f} dB，降低 {gain_mean:.2f} dB（{relative_gain:.2f}%）。",
        "- 联合方法仅用当前比例所选实测点稳健拟合 measured≈a·Sionna+b，再对校准后残差执行1-NN插值；仿真无效栅格退化为仅实测1-NN。",
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


def _aggregate_trials(trials: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    long_rows = []
    for pct in sorted(trials["sampling_percent"].unique()):
        sub = trials.loc[trials["sampling_percent"].eq(int(pct))]
        row = {
            "sampling_percent": int(pct),
            "selected_measured_point_count": int(sub["selected_measured_point_count"].iloc[0]),
            "total_eligible_measured_points": int(sub["total_eligible_measured_points"].iloc[0]),
            "reference_valid_outdoor_cell_count": int(sub["reference_valid_outdoor_cell_count"].iloc[0]),
        }
        for variant, suffix in [("without_simulation", "without_simulation"), ("with_simulation", "with_simulation")]:
            vv = sub.loc[sub["variant"].eq(variant)]
            if vv.empty:
                continue
            for metric in ["rmse_db", "mae_db", "bias_pred_minus_reference_db", "rmse_unsampled_db", "mae_unsampled_db"]:
                vals = pd.to_numeric(vv[metric], errors="coerce")
                row[f"{metric}_{suffix}"] = float(vals.mean())
                row[f"{metric}_std_{suffix}"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            long_rows.append({
                "sampling_percent": int(pct),
                "variant": variant,
                "simulation_data_used": bool(variant == "with_simulation"),
                "selected_measured_point_count": int(row["selected_measured_point_count"]),
                "total_eligible_measured_points": int(row["total_eligible_measured_points"]),
                "reference_valid_outdoor_cell_count": int(row["reference_valid_outdoor_cell_count"]),
                "rmse_db": row[f"rmse_db_{suffix}"],
                "rmse_std_db": row[f"rmse_db_std_{suffix}"],
                "mae_db": row[f"mae_db_{suffix}"],
                "mae_std_db": row[f"mae_db_std_{suffix}"],
                "bias_pred_minus_reference_db": row[f"bias_pred_minus_reference_db_{suffix}"],
                "method": "nearest_neighbor" if variant == "without_simulation" else "calibrated_sionna_plus_residual_nearest_neighbor",
            })
        if "rmse_db_without_simulation" in row and "rmse_db_with_simulation" in row:
            row["rmse_gain_with_simulation_db"] = row["rmse_db_without_simulation"] - row["rmse_db_with_simulation"]
        rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(long_rows)


def _write_trend_audit(path: Path, paired: pd.DataFrame, selection_mode: str) -> None:
    lines = [
        "# Reconstruction trend audit (v1.18.8 single-map adaptive-domain strict-nested reconstruction)",
        "",
        "Primary error domain: every finite outdoor cell of the 512 m x 512 m measurement-filled reference map.",
        "Selected measurement cells remain in the primary RMSE domain, exactly as in the original reconstruction code.",
        "Each sampling percentage is reconstructed exactly once. Under the default hierarchical-nested selector, all percentages are strict prefixes of one sampling hierarchy, so the RMSE curve and saved maps refer to the same progressive experiment.",
        f"Selection mode: {selection_mode}.",
        "No multi-trial averaging, no best-trial/seed selection, and no monotonic RMSE post-processing is applied.",
        "",
    ]
    all_ok = True
    for col, title in [
        ("rmse_db_without_simulation", "Measurement-only 1-NN"),
        ("rmse_db_with_simulation", "Measurement + calibrated Sionna residual 1-NN"),
    ]:
        if col not in paired.columns:
            continue
        vals = paired[col].to_numpy(dtype=float)
        diffs = np.diff(vals)
        inc = np.where(diffs > 1e-12)[0]
        all_ok = all_ok and len(inc) == 0
        lines += [
            f"## {title}",
            f"Single-run RMSE start -> end: {vals[0]:.4f} -> {vals[-1]:.4f} dB",
            f"Local increases: {len(inc)}",
        ]
        if len(inc):
            for i in inc:
                lines.append(
                    f"- {int(paired.iloc[i]['sampling_percent'])}% -> "
                    f"{int(paired.iloc[i+1]['sampling_percent'])}%: {diffs[i]:+.4f} dB"
                )
        else:
            lines.append("- none")
        lines.append("")
    lines += [
        f"Overall decreasing-trend audit passed for all available branches: {bool(all_ok)}",
        "",
        "The default adaptive-progressive selector never reads reference-map RSRP, Sionna RSRP, or the RSRP of an unselected candidate. It uses only coordinates plus RSRP gradients inferred from measurements already selected at earlier steps; no RMSE forcing is applied.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _plot_single_rmse(path: Path, paired: pd.DataFrame, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(7.48, 5.2), dpi=int(dpi))
    x = paired["sampling_percent"].to_numpy(dtype=float)
    if "rmse_db_without_simulation" in paired:
        y = paired["rmse_db_without_simulation"].to_numpy(dtype=float)
        ax.plot(x, y, marker="o", label="Measured only: 1-NN")
    if "rmse_db_with_simulation" in paired:
        y = paired["rmse_db_with_simulation"].to_numpy(dtype=float)
        ax.plot(x, y, marker="s", label="Measured + calibrated Sionna: residual 1-NN")
    ax.set_xlabel("Selected measured points [%]")
    ax.set_ylabel("RMSE over all valid 512 m x 512 m reference cells [dB]")
    ax.set_title("Radio-map reconstruction: single adaptive-progressive nearest-neighbor run")
    ax.set_xticks(x.astype(int))
    ax.grid(True, alpha=0.28)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=int(dpi), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _single_run_tables(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build paired and long tables without averaging across trials."""
    long = metrics.copy().sort_values(["sampling_percent", "simulation_data_used"]).reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    for pct in sorted(long["sampling_percent"].unique()):
        sub = long.loc[long["sampling_percent"].eq(int(pct))]
        row: dict[str, Any] = {
            "sampling_percent": int(pct),
            "selected_measured_point_count": int(sub["selected_measured_point_count"].iloc[0]),
            "total_eligible_measured_points": int(sub["total_eligible_measured_points"].iloc[0]),
            "reference_valid_outdoor_cell_count": int(sub["reference_valid_outdoor_cell_count"].iloc[0]),
            "selection_mode": str(sub["selection_mode"].iloc[0]),
        }
        for variant, suffix in [
            ("without_simulation", "without_simulation"),
            ("with_simulation", "with_simulation"),
        ]:
            vv = sub.loc[sub["variant"].eq(variant)]
            if vv.empty:
                continue
            one = vv.iloc[0]
            for metric in ["rmse_db", "mae_db", "bias_pred_minus_reference_db", "rmse_unsampled_db", "mae_unsampled_db"]:
                row[f"{metric}_{suffix}"] = float(one[metric])
        if "rmse_db_without_simulation" in row and "rmse_db_with_simulation" in row:
            row["rmse_gain_with_simulation_db"] = float(row["rmse_db_without_simulation"] - row["rmse_db_with_simulation"])
        rows.append(row)
    return pd.DataFrame(rows), long

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
        else project_root / "outputs" / "radio_map_reconstruction_nn_single_progressive"
    )
    percentages = parse_percentages(args.percentages)
    if int(args.random_trials) != 1:
        print(
            f"[INFO] v1.18.8正式重构为单次实验；--random-trials={int(args.random_trials)} 被忽略，实际固定为1。"
        )
    reference_path = discover_filled_reference(project_root, args.station_id, args.pci, args.filled_reference_npz)

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
            raise ValueError("纯Sionna仿真NPZ不能与measurement-filled参考地图使用同一文件")
        simulation_prior = load_simulation_prior(
            simulation_path,
            station_id=int(args.station_id),
            pci=int(args.pci),
            fallback_extent=None,
        )

    # Preserve the original full-grid evaluation convention.
    reference_map = np.asarray(reference.rsrp_dbm, dtype=float).copy()
    # Quantitative metrics use the raw finite RSRP values stored in the reference
    # NPZ.  The display range (-120 to -40 dBm by default) is applied only by
    # plotting through vmin/vmax; no clipping is applied before RMSE/MAE.
    building_mask = (
        np.asarray(reference.building_mask, dtype=bool)
        if reference.building_mask is not None
        else np.zeros(reference_map.shape, dtype=bool)
    )
    if building_mask.shape != reference_map.shape:
        raise ValueError("补齐地图building_mask与RSRP地图尺寸不一致")
    reference_eval_mask = (~building_mask) & np.isfinite(reference_map)
    if not np.any(reference_eval_mask):
        raise ValueError("512 m x 512 m参考地图没有有效室外栅格")

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

    station_root = output_root / f"station_{int(args.station_id):02d}_pci_{int(args.pci)}"
    station_root.mkdir(parents=True, exist_ok=True)
    reference_dir = station_root / "reference_filled_map"
    save_map_npz(
        reference_dir / f"station_{int(args.station_id):02d}_pci_{int(args.pci)}_filled_reference.npz",
        station_id=args.station_id,
        pci=args.pci,
        x_axis=reference.x_axis_m,
        y_axis=reference.y_axis_m,
        rsrp_map=reference_map,
        building_mask=building_mask,
        map_role="filled_reference",
        metadata={
            "source_filled_reference_npz": str(reference_path),
            "evaluation_reference": "ALL finite outdoor cells of 512m x 512m measurement-filled radio map",
            "valid_outdoor_reference_cell_count": int(reference_eval_mask.sum()),
            "selected_measurement_cells_included_in_primary_rmse": True,
        },
    )
    if not args.skip_figures:
        plot_map(
            reference_dir / f"station_{int(args.station_id):02d}_pci_{int(args.pci)}_filled_reference.png",
            rsrp_map=reference_map,
            x_axis=reference.x_axis_m,
            y_axis=reference.y_axis_m,
            building_mask=building_mask,
            title="Filled radio map (full-grid reference)",
            min_dbm=args.display_min_dbm,
            max_dbm=args.display_max_dbm,
            dpi=args.dpi,
        )

    xx, yy = np.meshgrid(reference.x_axis_m, reference.y_axis_m)
    query_xy = np.column_stack([xx.ravel(), yy.ravel()])

    selection_sets: dict[int, pd.DataFrame] | None = None
    ranked: pd.DataFrame | None = None
    if args.selection_mode == "voronoi-safe-adaptive-nested":
        stage_counts = [
            min(n_measured, max(1, int(math.ceil(n_measured * float(p) / 100.0))))
            for p in percentages
        ]
        ranked, selection_audit = voronoi_safe_adaptive_nested_ranking(
            measured=measured,
            x_axis=reference.x_axis_m,
            y_axis=reference.y_axis_m,
            reference_eval_mask=reference_eval_mask,
            stage_target_counts=stage_counts,
            gradient_weight=0.55,
            containment_strength=1.15,
        )
        ranked.to_csv(
            station_root / "single_voronoi_safe_nested_ranking.csv",
            index=False,
            encoding="utf-8-sig",
        )
        selection_audit.to_csv(
            station_root / "voronoi_safe_nested_selection_audit.csv",
            index=False,
            encoding="utf-8-sig",
        )
    elif args.selection_mode == "adaptive-domain-nested":
        first_sample_count = min(
            n_measured,
            max(1, int(math.ceil(n_measured * float(min(percentages)) / 100.0))),
        )
        ranked, selection_audit = adaptive_domain_nested_ranking(
            measured=measured,
            x_axis=reference.x_axis_m,
            y_axis=reference.y_axis_m,
            reference_eval_mask=reference_eval_mask,
            max_count=max_sample_count,
            initial_count=first_sample_count,
            gradient_weight=0.75,
        )
        ranked.to_csv(
            station_root / "single_adaptive_domain_nested_ranking.csv",
            index=False,
            encoding="utf-8-sig",
        )
        selection_audit.to_csv(
            station_root / "adaptive_domain_nested_selection_audit.csv",
            index=False,
            encoding="utf-8-sig",
        )
    elif args.selection_mode == "hierarchical-nested":
        first_sample_count = min(
            n_measured,
            max(1, int(math.ceil(n_measured * float(min(percentages)) / 100.0))),
        )
        ranked, selection_audit = hierarchical_nested_spatial_ranking(
            measured=measured,
            x_axis=reference.x_axis_m,
            y_axis=reference.y_axis_m,
            reference_eval_mask=reference_eval_mask,
            min_count=first_sample_count,
            max_count=max_sample_count,
        )
        ranked.to_csv(
            station_root / "single_hierarchical_nested_ranking.csv",
            index=False,
            encoding="utf-8-sig",
        )
        selection_audit.to_csv(
            station_root / "hierarchical_nested_selection_audit.csv",
            index=False,
            encoding="utf-8-sig",
        )
    elif args.selection_mode == "adaptive-progressive":
        first_sample_count = min(
            n_measured,
            max(1, int(math.ceil(n_measured * float(min(percentages)) / 100.0))),
        )
        ranked = adaptive_progressive_nested_ranking(
            measured=measured,
            max_count=max_sample_count,
            initial_count=first_sample_count,
            gradient_weight=0.50,
        )
        ranked.to_csv(
            station_root / "single_adaptive_progressive_nested_ranking.csv",
            index=False,
            encoding="utf-8-sig",
        )
    elif args.selection_mode == "stable-spatial":
        selection_sets, selection_audit = stable_spatial_percentage_sets(
            measured=measured,
            x_axis=reference.x_axis_m,
            y_axis=reference.y_axis_m,
            reference_eval_mask=reference_eval_mask,
            percentages=percentages,
        )
        pd.concat([selection_sets[p] for p in percentages], ignore_index=True).to_csv(
            station_root / "single_percentage_spatial_selection.csv",
            index=False,
            encoding="utf-8-sig",
        )
        selection_audit.to_csv(
            station_root / "spatial_selection_audit.csv",
            index=False,
            encoding="utf-8-sig",
        )
    elif args.selection_mode == "progressive-coverage":
        ranked = progressive_coverage_nested_ranking(
            measured=measured,
            x_axis=reference.x_axis_m,
            y_axis=reference.y_axis_m,
            reference_eval_mask=reference_eval_mask,
            max_count=max_sample_count,
        )
        ranked.to_csv(
            station_root / "single_progressive_nested_ranking.csv",
            index=False,
            encoding="utf-8-sig",
        )
    else:
        ranked = basic_random_nested_ranking(
            measured=measured,
            max_count=max_sample_count,
            seed=int(args.random_seed),
        )
        ranked.to_csv(
            station_root / "single_progressive_nested_ranking.csv",
            index=False,
            encoding="utf-8-sig",
        )

    metric_rows: list[dict[str, Any]] = []
    maps_for_composite: dict[int, tuple[np.ndarray, np.ndarray, float, float]] = {}

    print("=" * 96)
    print("SINGLE-MAP STRICT-NESTED NEAREST-NEIGHBOR RADIO-MAP RECONSTRUCTION / FULL-GRID EVALUATION")
    print(f"Station={args.station_id}, PCI={args.pci}")
    print(f"Eligible measured 1-m cells={n_measured}; valid full-grid reference cells={int(reference_eval_mask.sum())}")
    print(f"Selection={args.selection_mode}; one reconstruction per percentage; percentages={percentages}")
    print("M   = selected measured points -> 1-NN over whole grid")
    print("M+S = same selected points + robustly calibrated Sionna trend -> residual 1-NN over whole grid")
    print("Primary RMSE = ALL finite outdoor cells of the 512m x 512m filled reference map")
    print("No multi-trial averaging, no best-trial/seed map selection, and no quantitative RSRP clipping.")
    print("=" * 96)

    for pct in percentages:
        sample_count = min(
            n_measured,
            max(1, int(math.ceil(n_measured * float(pct) / 100.0))),
        )
        if selection_sets is not None:
            selected = selection_sets[int(pct)].copy().reset_index(drop=True)
            if len(selected) != int(sample_count):
                raise RuntimeError(
                    f"{pct}%空间分层选点数量错误：expected={sample_count}, got={len(selected)}"
                )
        else:
            assert ranked is not None
            selected = ranked.iloc[:sample_count].copy().reset_index(drop=True)
        train_xy = selected[["x_m", "y_m"]].to_numpy(dtype=float)
        train_y = selected["measured_rsrp_dbm"].to_numpy(dtype=float)

        tree = cKDTree(train_xy)
        _, nearest = tree.query(query_xy, k=1)
        m_prediction = train_y[np.asarray(nearest, dtype=int)].reshape(reference_map.shape)
        # Keep raw 1-NN prediction for quantitative evaluation; display clipping
        # is handled only by plotting limits.
        m_prediction[building_mask] = np.nan

        sampled_cell_mask = np.zeros(reference_map.shape, dtype=bool)
        sampled_cell_mask[
            selected["iy"].to_numpy(dtype=int),
            selected["ix"].to_numpy(dtype=int),
        ] = True
        unsampled_mask = reference_eval_mask & (~sampled_cell_mask)
        coverage_mean, coverage_p90, coverage_max = measured_pool_coverage_metrics(measured, selected)

        base = {
            "station_id": int(args.station_id),
            "pci": int(args.pci),
            "sampling_percent": int(pct),
            "total_eligible_measured_points": int(n_measured),
            "selected_measured_point_count": int(sample_count),
            "reference_valid_outdoor_cell_count": int(reference_eval_mask.sum()),
            "unsampled_reference_cell_count": int(unsampled_mask.sum()),
            "evaluation_reference": "all_finite_outdoor_cells_of_512x512_measurement_filled_map",
            "selection_mode": str(args.selection_mode),
            "single_run": True,
            "measured_pool_nearest_distance_mean_m": float(coverage_mean),
            "measured_pool_nearest_distance_p90_m": float(coverage_p90),
            "measured_pool_nearest_distance_max_m": float(coverage_max),
        }

        m_rmse = rmse(reference_map, m_prediction, reference_eval_mask)
        if args.simulation_mode in {"without", "compare"}:
            metric_rows.append({
                **base,
                "variant": "without_simulation",
                "simulation_data_used": False,
                "method": "nearest_neighbor",
                "rmse_db": m_rmse,
                "mae_db": mae(reference_map, m_prediction, reference_eval_mask),
                "bias_pred_minus_reference_db": bias(reference_map, m_prediction, reference_eval_mask),
                "rmse_unsampled_db": rmse(reference_map, m_prediction, unsampled_mask),
                "mae_unsampled_db": mae(reference_map, m_prediction, unsampled_mask),
            })

        ms_prediction = None
        ms_diag: dict[str, Any] = {}
        ms_rmse = float("nan")
        if simulation_prior is not None:
            ms_flat, ms_diag = simulation_residual_nearest_neighbor(
                prior=simulation_prior,
                train_xy=train_xy,
                train_y=train_y,
                query_xy=query_xy,
                baseline_prediction=m_prediction.ravel(),
            )
            ms_prediction = ms_flat.reshape(reference_map.shape)
            # No quantitative clipping.  Keep the actual calibrated-Sionna +
            # residual 1-NN values for RMSE/MAE and only clip visually.
            ms_prediction[building_mask] = np.nan
            ms_rmse = rmse(reference_map, ms_prediction, reference_eval_mask)
            metric_rows.append({
                **base,
                "variant": "with_simulation",
                "simulation_data_used": True,
                "method": "calibrated_sionna_plus_residual_nearest_neighbor",
                "rmse_db": ms_rmse,
                "mae_db": mae(reference_map, ms_prediction, reference_eval_mask),
                "bias_pred_minus_reference_db": bias(reference_map, ms_prediction, reference_eval_mask),
                "rmse_unsampled_db": rmse(reference_map, ms_prediction, unsampled_mask),
                "mae_unsampled_db": mae(reference_map, ms_prediction, unsampled_mask),
                **ms_diag,
            })

        # Exactly one saved reconstruction per sampling percentage and branch.
        pct_dir = station_root / f"percent_{int(pct):02d}"
        pct_dir.mkdir(parents=True, exist_ok=True)
        selected.to_csv(
            pct_dir / "selected_measured_points.csv",
            index=False,
            encoding="utf-8-sig",
        )
        if args.simulation_mode in {"without", "compare"}:
            save_map_npz(
                pct_dir / "measurement_only_nn.npz",
                station_id=args.station_id,
                pci=args.pci,
                x_axis=reference.x_axis_m,
                y_axis=reference.y_axis_m,
                rsrp_map=m_prediction,
                building_mask=building_mask,
                map_role=f"measurement_only_nn_{int(pct)}pct_single_run",
                metadata={**base, "primary_rmse_db": float(m_rmse)},
                selected_points=selected,
            )
            if not args.skip_figures:
                plot_map(
                    pct_dir / "measurement_only_nn.png",
                    rsrp_map=m_prediction,
                    x_axis=reference.x_axis_m,
                    y_axis=reference.y_axis_m,
                    building_mask=building_mask,
                    title=f"Measured only: 1-NN ({int(pct)}%, RMSE={m_rmse:.2f} dB)",
                    min_dbm=args.display_min_dbm,
                    max_dbm=args.display_max_dbm,
                    dpi=args.dpi,
                    selected_points=selected,
                    show_selected_points=False,
                )
        if ms_prediction is not None:
            save_map_npz(
                pct_dir / "measurement_plus_calibrated_sionna_residual_nn.npz",
                station_id=args.station_id,
                pci=args.pci,
                x_axis=reference.x_axis_m,
                y_axis=reference.y_axis_m,
                rsrp_map=ms_prediction,
                building_mask=building_mask,
                map_role=f"measurement_plus_calibrated_sionna_residual_nn_{int(pct)}pct_single_run",
                metadata={**base, **ms_diag, "primary_rmse_db": float(ms_rmse)},
                selected_points=selected,
            )
            if not args.skip_figures:
                plot_map(
                    pct_dir / "measurement_plus_calibrated_sionna_residual_nn.png",
                    rsrp_map=ms_prediction,
                    x_axis=reference.x_axis_m,
                    y_axis=reference.y_axis_m,
                    building_mask=building_mask,
                    title=f"Measured + simulation: residual 1-NN ({int(pct)}%, RMSE={ms_rmse:.2f} dB)",
                    min_dbm=args.display_min_dbm,
                    max_dbm=args.display_max_dbm,
                    dpi=args.dpi,
                    selected_points=selected,
                    show_selected_points=False,
                )

        if int(pct) in {1, 5, 10}:
            maps_for_composite[int(pct)] = (
                m_prediction.copy(),
                ms_prediction.copy() if ms_prediction is not None else m_prediction.copy(),
                float(m_rmse),
                float(ms_rmse),
            )

    metrics = pd.DataFrame(metric_rows).sort_values(
        ["sampling_percent", "simulation_data_used"]
    ).reset_index(drop=True)
    paired, long_summary = _single_run_tables(metrics)
    metrics.to_csv(
        station_root / "reconstruction_single_run_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    paired.to_csv(
        station_root / "reconstruction_simulation_ablation_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    long_summary.to_csv(
        station_root / "reconstruction_simulation_ablation_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pure = long_summary.loc[long_summary["variant"].eq("without_simulation")].copy()
    if not pure.empty:
        pure.to_csv(
            station_root / "nearest_neighbor_percentage_metrics.csv",
            index=False,
            encoding="utf-8-sig",
        )

    audit = pd.DataFrame([{
        "reference_grid_shape_y": int(reference_map.shape[0]),
        "reference_grid_shape_x": int(reference_map.shape[1]),
        "reference_total_grid_cells": int(reference_map.size),
        "reference_valid_outdoor_cell_count": int(reference_eval_mask.sum()),
        "primary_rmse_uses_all_valid_outdoor_reference_cells": True,
        "selected_measurement_cells_removed_from_primary_rmse": False,
        "heldout_measured_point_rmse_used": False,
        "reconstruction_runs_per_sampling_percent": 1,
        "multi_trial_average_used": False,
        "best_trial_map_selection_used": False,
        "selection_mode": str(args.selection_mode),
        "selection_uses_reference_rsrp": False,
        "selection_uses_unselected_measured_rsrp": False,
        "selection_uses_simulation_rsrp": False,
        "selection_uses_final_rmse": False,
        "sampling_sets_nested": bool(args.selection_mode in {"voronoi-safe-adaptive-nested", "adaptive-domain-nested", "hierarchical-nested", "adaptive-progressive", "progressive-coverage", "random"}),
        "selection_uses_selected_measured_rsrp": bool(args.selection_mode in {"voronoi-safe-adaptive-nested", "adaptive-domain-nested", "adaptive-progressive"}),
        "measurement_only_method": "1-nearest-neighbor",
        "measurement_plus_simulation_method": "selected-data robust affine calibrated Sionna + 1-nearest-neighbor residual",
        "quantitative_rsrp_clipping_applied": False,
        "display_range_only_dbm": f"{float(args.display_min_dbm):g}..{float(args.display_max_dbm):g}",
    }])
    audit.to_csv(
        station_root / "reconstruction_full_grid_evaluation_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    _write_trend_audit(
        station_root / "reconstruction_trend_audit.md",
        paired,
        str(args.selection_mode),
    )
    write_ablation_analysis(
        station_root / "reconstruction_simulation_ablation_analysis.md",
        paired,
    )

    if not args.skip_figures:
        if maps_for_composite and simulation_prior is not None:
            plot_paired_reconstruction_composite(
                station_root / "reconstruction_maps_1_5_10.png",
                representative_maps=maps_for_composite,
                x_axis=reference.x_axis_m,
                y_axis=reference.y_axis_m,
                building_mask=building_mask,
                min_dbm=args.display_min_dbm,
                max_dbm=args.display_max_dbm,
                dpi=args.dpi,
            )
            # Backward-compatible filename points to the same single-run maps.
            plot_paired_reconstruction_composite(
                station_root / "reconstruction_simulation_ablation_representative_maps.png",
                representative_maps=maps_for_composite,
                x_axis=reference.x_axis_m,
                y_axis=reference.y_axis_m,
                building_mask=building_mask,
                min_dbm=args.display_min_dbm,
                max_dbm=args.display_max_dbm,
                dpi=args.dpi,
            )
        _plot_single_rmse(
            station_root / "reconstruction_simulation_ablation_rmse.png",
            paired,
            int(args.dpi),
        )

    metadata = {
        "version": "1.18.10",
        "experiment": "single-map strict-nested two-branch nearest-neighbor full-grid radio-map reconstruction",
        "station_id": int(args.station_id),
        "pci": int(args.pci),
        "measurement_csv": str(measurements_csv),
        "filled_reference_npz": str(reference_path),
        "simulation_prior_npz": str(simulation_path) if simulation_path else None,
        "reconstruction_runs_per_sampling_percent": 1,
        "multi_trial_average_used": False,
        "best_trial_map_selection_used": False,
        "base_random_seed": int(args.random_seed),
        "percentages": percentages,
        "selection_mode": str(args.selection_mode),
        "sampling_definition": (
            "strict nested stage-wise Voronoi-safe adaptive sampling: geometry + already-selected measured RSRP only; large newly observed jumps trigger same-stage containment acquisitions; one map per percentage"
            if args.selection_mode == "voronoi-safe-adaptive-nested"
            else "one strict nested adaptive-domain sequence: the initial sparse set is geometry-only; later points are chosen from full-domain geometry plus radio-gradient information inferred only from already selected measured points; one map per percentage"
            if args.selection_mode == "adaptive-domain-nested"
            else "one deterministic geometry-only strict-nested hierarchy: a globally balanced maximum-density set is built first, then backward geometry elimination defines all lower-density prefixes; one map per percentage"
            if args.selection_mode == "hierarchical-nested"
            else "one strict nested adaptive sequence: initial geometry-only farthest-point sampling, then next locations are chosen from coordinates using only RSRP gradients inferred from already selected measurements; one map per percentage"
            if args.selection_mode == "adaptive-progressive"
            else "one deterministic geometry-only constrained centroidal-Voronoi subset independently optimized for each sampling percentage; one map per percentage; sets need not be nested"
            if args.selection_mode == "stable-spatial"
            else "one deterministic geometry-only progressive k-center ranking; 1%-10% are strict nested prefixes"
            if args.selection_mode == "progressive-coverage"
            else "one fixed-seed random ranking; 1%-10% are strict nested prefixes"
        ),
        "selection_uses_reference_rsrp": False,
        "selection_uses_unselected_measured_rsrp": False,
        "selection_uses_simulation_rsrp": False,
        "selection_uses_final_rmse": False,
        "sampling_sets_nested": bool(args.selection_mode in {"voronoi-safe-adaptive-nested", "adaptive-domain-nested", "hierarchical-nested", "adaptive-progressive", "progressive-coverage", "random"}),
        "selection_uses_selected_measured_rsrp": bool(args.selection_mode in {"voronoi-safe-adaptive-nested", "adaptive-domain-nested", "adaptive-progressive"}),
        "branch_M": "selected measured RSRP -> 1-nearest-neighbor reconstruction",
        "branch_M_plus_S": "same selected measured points + fixed Sionna RT map; robust affine calibration measured≈a*Sionna+b using selected points only; 1-nearest-neighbor interpolation of calibrated residual",
        "primary_rmse_reference": "ALL finite outdoor cells of the 512m x 512m measurement-filled reference map",
        "selected_measurement_cells_in_primary_rmse": True,
        "no_heldout_measurement_evaluation": True,
        "no_monotonic_postprocessing": True,
        "no_seed_selection_by_rmse": True,
        "measurement_filter_range_dbm": [float(args.min_rsrp_dbm), float(args.max_rsrp_dbm)],
        "quantitative_rsrp_clipping_applied": False,
        "display_range_dbm": [float(args.display_min_dbm), float(args.display_max_dbm)],
    }
    (station_root / "experiment_metadata.json").write_text(
        json.dumps(_json_safe(metadata), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nSingle-run full-grid RMSE summary:")
    print(paired.to_string(index=False))
    print("\nDone:", station_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
