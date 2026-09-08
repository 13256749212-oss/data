#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
27站稀疏实测定位优化版（支持每站多个原始接收采样点）
================================================

算法：Direction-Prior Constrained Physics-Guided Robust Sparse Localization
(DP-PGRSL，方向先验约束物理引导鲁棒稀疏定位)

核心改进：
1. 输出目录随点数和随机种子动态变化，避免不同点数运行互相覆盖；
2. 先在完整坐标转换数据中筛选PCI完整、信号较强且扇区互补的候选点，
   再选出指定数量的原始接收采样位置，避免旧版过度追求最大空间跨度而选到远端弱点；
3. 三扇区站优先使用已估计的120°扇区方向作为外部方向先验；
4. 用同一点三个扇区的相对RSRP消除未知发射功率/路径损耗共同项，先约束角度；
5. 用绝对RSRP包络和物理参考截距/路径损耗先验约束距离；
6. 不再为每个扇区自由拟合大幅功率偏差，减少5点极稀疏条件下的不可辨识参数；
7. 最终结果以物理逆解为主，强信号空间锚点仅作弱先验，并由bootstrap不确定度控制轻度收缩；
8. 新增不依赖真实坐标的几何可靠性诊断：角度覆盖、凸包外推、PCI空间均衡和model-anchor分歧；
9. 对低几何可靠性/大分歧站自动触发多搜索中心鲁棒求解，再按物理目标与几何可信度选择候选解；
10. 22号全向站不使用方向模型，采用绝对功率逆解，并使用同一可靠性诊断框架；
11. 真实基站坐标仍只在最终评价阶段读取，不进入点选择、候选解选择或稳定化。

重要说明：默认 fixed 方向先验来自 estimated_initial_directions_27stations.csv。
这会显著提高3点极稀疏定位稳定性，但它不是“仅靠5条原始测量、无任何先验”的盲定位。
可用 --direction-prior-mode off 运行严格无方向先验对照。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull, QhullError, cKDTree

import legacy_pgrmsbil as common

ALGORITHM_NAME = (
    "Direction-Prior Constrained Profiled Robust Sparse Localization (DP-PPRSL-v1.10)"
)
DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POINTS = 5
DEFAULT_SEED = 20260805
DEFAULT_BOUNDS = common.DEFAULT_BOUNDS

# 传播与优化参数
TX_RX_VERTICAL_SEPARATION_M = 28.5
HORIZONTAL_3DB_BEAMWIDTH_DEG = 65.0
HORIZONTAL_MAX_ATTENUATION_DB = 30.0
PATHLOSS_EXPONENT_PRIOR = 2.70
PATHLOSS_EXPONENT_STD = 0.65
REFERENCE_RSRP_AT_1M_PRIOR_DBM = -22.0
REFERENCE_RSRP_AT_1M_STD_DB = 9.0
ANCHOR_PRIOR_SIGMA_M = 180.0
LOCAL_SEARCH_RADIUS_M = 350.0
ROBUST_SECTOR_DIFF_SCALE_DB = 5.0
ROBUST_ENVELOPE_SCALE_DB = 7.0


@dataclass
class ResultRow:
    station_id: int
    station_label: str
    antenna_type: str
    selected_point_count: int
    observation_count: int
    distinct_pci_count: int
    predicted_x_m: float
    predicted_y_m: float
    model_x_m: float
    model_y_m: float
    anchor_x_m: float
    anchor_y_m: float
    model_weight: float
    true_x_m: float
    true_y_m: float
    east_error_m: float
    north_error_m: float
    horizontal_error_m: float
    pathloss_exponent: float
    alpha_deg: float
    sector_order_sign: int
    antenna_gain_scale: float
    reference_rsrp_1m_dbm: float
    objective_value: float
    direction_prior_used: bool
    direction_fit_rms_deg: float
    uncertainty_radius_p90_m: float
    bootstrap_success_count: int
    point_spread_m: float
    quality_flag: str
    model_anchor_disagreement_m: float
    optimizer_boundary_margin_m: float
    uncertainty_shrinkage_factor: float
    objective_per_observation: float
    angular_coverage_deg: float
    max_angular_gap_deg: float
    inside_measurement_convex_hull: bool
    extrapolation_ratio: float
    pci_spatial_balance: float
    geometry_reliability_score: float
    multistart_triggered: bool
    multistart_candidate_count: int
    multistart_selected_label: str
    elapsed_s: float
    selection_geometry_score: float
    solver_mode: str
    direction_ray_x_m: float
    direction_ray_y_m: float
    direction_ray_perpendicular_rms_m: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="27站稀疏实测定位优化版")
    p.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    p.add_argument("--measurements", type=Path, default=None)
    p.add_argument("--directions", type=Path, default=None)
    p.add_argument("--calibration-root", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--points-per-station", type=int, default=DEFAULT_POINTS)
    p.add_argument(
        "--selection-max-points", type=int, default=None,
        help="用于多点数消融的统一最大选点数；设置15后，10--15点均使用同一1--15排序的前缀",
    )
    p.add_argument(
        "--selection-mode", choices=["balanced", "random"], default="balanced",
        help="balanced=原强信号/扇区均衡选点；random=从定位可用空间点中等概率无放回随机抽样",
    )
    p.add_argument("--trial-index", type=int, default=1, help="随机重复实验编号，仅写入元数据")
    p.add_argument("--random-seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--bootstrap", type=int, default=20)
    p.add_argument("--de-maxiter", type=int, default=100)
    p.add_argument("--de-popsize", type=int, default=10)
    p.add_argument(
        "--direction-prior-mode",
        choices=["fixed", "soft", "off"],
        default="fixed",
        help="fixed=固定外部方向；soft=方向软约束；off=不使用方向文件",
    )
    p.add_argument("--station-ids", default="all")
    p.add_argument("--x-min", type=float, default=DEFAULT_BOUNDS[0])
    p.add_argument("--x-max", type=float, default=DEFAULT_BOUNDS[1])
    p.add_argument("--y-min", type=float, default=DEFAULT_BOUNDS[2])
    p.add_argument("--y-max", type=float, default=DEFAULT_BOUNDS[3])
    p.add_argument("--skip-figures", action="store_true", help="跳过全部定位图，仅输出CSV/JSON")
    p.add_argument(
        "--skip-per-station-figures", action="store_true",
        help="仅跳过逐站高分辨率图，仍生成每个点数的总览图和CDF；多点数批量实验推荐",
    )
    return p.parse_args()


def first_existing(paths: Iterable[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists() and path.is_file():
            return path.resolve()
    return None


def resolve_direction_csv(project_root: Path, explicit: Optional[Path]) -> Optional[Path]:
    if explicit is not None:
        path = explicit.expanduser()
        if not path.exists():
            raise FileNotFoundError(f"找不到方向CSV：{path}")
        return path.resolve()
    here = Path(__file__).resolve().parent
    candidates = [
        project_root / "config" / "estimated_initial_directions_27stations.csv",
        project_root / "config" / "estimated_initial_directions_27stations(1).csv",
        project_root / "config" / "metadata" / "estimated_initial_directions_27stations.csv",
        project_root / "outputs" / "estimated_initial_directions_27stations.csv",
        here / "estimated_initial_directions_27stations.csv",
    ]
    return first_existing(candidates)


def load_direction_priors(path: Optional[Path]) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    frame = pd.read_csv(path, encoding="utf-8-sig")
    required = {
        "station_id", "selected_sector_order", "base_alpha_rad",
        "direction_fit_rms_deg",
    }
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"方向CSV缺少字段：{sorted(missing)}")
    frame = frame.copy()
    frame["station_id"] = pd.to_numeric(frame["station_id"], errors="raise").astype(int)
    for c in ["base_alpha_rad", "direction_fit_rms_deg"]:
        frame[c] = pd.to_numeric(frame[c], errors="coerce")
    return frame.set_index("station_id")


def build_signal_strength_points(points: pd.DataFrame) -> pd.DataFrame:
    """Build the measured signal-strength background used by the station figure.

    Each row corresponds to one original receiver location.  The plotted value is
    the strongest measured RSRP among the mapped sectors available at that
    receiver location.  These points are used only for visualization and do not
    change the selected-point subset or localization result.
    """
    if points is None or points.empty:
        return pd.DataFrame(columns=["x_m", "y_m", "strongest_rsrp_dbm"])
    out = points.copy()
    rsrp_cols = [c for c in ("rsrp_s1", "rsrp_s2", "rsrp_s3") if c in out.columns]
    if not rsrp_cols:
        return pd.DataFrame(columns=["x_m", "y_m", "strongest_rsrp_dbm"])
    out["strongest_rsrp_dbm"] = out[rsrp_cols].max(axis=1, skipna=True)
    out = out[
        out[["x_m", "y_m", "strongest_rsrp_dbm"]].notna().all(axis=1)
    ].copy()
    return out[["x_m", "y_m", "strongest_rsrp_dbm"]].reset_index(drop=True)


def _draw_signal_strength_background(
    ax,
    signal_points: pd.DataFrame | None,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    *,
    vmin: float,
    vmax: float,
):
    """Draw locally visible measured RSRP samples as a signal-strength map."""
    if signal_points is None or signal_points.empty:
        return None
    x0, x1 = map(float, xlim)
    y0, y1 = map(float, ylim)
    pad_x = 0.04 * max(x1 - x0, 1.0)
    pad_y = 0.04 * max(y1 - y0, 1.0)
    local = signal_points[
        signal_points["x_m"].between(x0 - pad_x, x1 + pad_x)
        & signal_points["y_m"].between(y0 - pad_y, y1 + pad_y)
    ].copy()
    if local.empty:
        return None
    return ax.scatter(
        local["x_m"], local["y_m"],
        c=local["strongest_rsrp_dbm"],
        cmap="viridis", vmin=float(vmin), vmax=float(vmax),
        s=18, alpha=0.72, linewidths=0, rasterized=True, zorder=1,
        label="_nolegend_",
    )

def normalize(values: np.ndarray) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    out = np.zeros_like(v)
    finite = np.isfinite(v)
    if not finite.any():
        return out
    lo = float(np.nanmin(v[finite]))
    hi = float(np.nanmax(v[finite]))
    if hi - lo < 1e-12:
        out[finite] = 1.0
    else:
        out[finite] = (v[finite] - lo) / (hi - lo)
    return out


def strong_anchor(points: pd.DataFrame, top_n: int = 20, temperature_db: float = 8.0) -> np.ndarray:
    rss = points[["rsrp_s1", "rsrp_s2", "rsrp_s3"]].to_numpy(float)
    envelope = np.nanmax(rss, axis=1)
    order = np.argsort(envelope)[::-1]
    top = order[: min(int(top_n), len(order))]
    weights = np.exp((envelope[top] - float(np.nanmax(envelope[top]))) / temperature_db)
    xy = points[["x_m", "y_m"]].to_numpy(float)
    return np.average(xy[top], axis=0, weights=weights)


def select_balanced_strong_points(
    points: pd.DataFrame,
    k: int,
    seed: int,
    omni: bool,
    ranking_k: int | None = None,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """离线数据集子集选择：强覆盖优先、扇区互补、适度空间分散。"""
    reference_k = max(int(k), int(ranking_k) if ranking_k is not None else int(k))
    if len(points) < reference_k:
        raise ValueError(f"唯一空间点仅{len(points)}个，少于统一排序要求{reference_k}个")
    pts = points.copy().reset_index(drop=True)
    rss_cols = ["rsrp_s1", "rsrp_s2", "rsrp_s3"]
    rss = pts[rss_cols].to_numpy(float)
    coverage = np.isfinite(rss).sum(axis=1)
    envelope = np.nanmax(rss, axis=1)
    mean_rss = np.nanmean(rss, axis=1)
    xy = pts[["x_m", "y_m"]].to_numpy(float)

    required_coverage = 1 if omni else 3
    complete = np.flatnonzero(coverage >= required_coverage)
    # Multi-count fairness: candidate-pool policy depends on the common maximum
    # point count, not on the current prefix length. This prevents 11->12 point
    # experiments from silently switching to a different candidate population.
    pool = complete if len(complete) >= reference_k else np.arange(len(pts))
    order = pool[np.argsort(envelope[pool])[::-1]]

    # 只在较强候选池中选点，避免旧版把一个点推到很远的弱覆盖边缘。
    strong_count = min(len(order), max(40, int(math.ceil(0.25 * len(order)))))
    strong_pool = order[:strong_count]
    anchor = strong_anchor(pts)
    distance_to_anchor = np.linalg.norm(xy - anchor[None, :], axis=1)
    candidate = strong_pool[distance_to_anchor[strong_pool] <= LOCAL_SEARCH_RADIUS_M]
    if len(candidate) < reference_k:
        candidate = strong_pool

    selected: List[int] = [int(order[0])]

    if not omni:
        # 每个扇区至少选一个具有相对优势的点。
        for sector in range(3):
            values = rss[:, sector]
            other = np.delete(rss, sector, axis=1)
            other_count = np.sum(np.isfinite(other), axis=1)
            other_sum = np.nansum(other, axis=1)
            other_mean = np.divide(
                other_sum, other_count,
                out=np.full(len(other_sum), np.nan, dtype=float),
                where=other_count > 0,
            )
            dominance = values - other_mean
            available = np.asarray(
                [i for i in candidate if int(i) not in selected and np.isfinite(values[i])],
                dtype=int,
            )
            if len(available) == 0:
                continue
            sep = np.min(
                np.linalg.norm(
                    xy[available, None, :] - xy[np.asarray(selected), :][None, :, :],
                    axis=2,
                ),
                axis=1,
            )
            score = (
                0.55 * normalize(dominance[available])
                + 0.25 * normalize(envelope[available])
                + 0.20 * normalize(np.minimum(sep, 180.0))
            )
            selected.append(int(available[int(np.nanargmax(score))]))
            if len(selected) >= reference_k:
                break

    while len(selected) < reference_k:
        available = np.asarray([i for i in candidate if int(i) not in selected], dtype=int)
        if len(available) == 0:
            available = np.asarray([i for i in pool if int(i) not in selected], dtype=int)
        sep = np.min(
            np.linalg.norm(
                xy[available, None, :] - xy[np.asarray(selected), :][None, :, :],
                axis=2,
            ),
            axis=1,
        )
        # 目标间距约120 m；不奖励无限增大的跨度。
        moderate_sep = np.exp(-((sep - 120.0) / 100.0) ** 2)
        score = (
            0.45 * normalize(envelope[available])
            + 0.25 * normalize(mean_rss[available])
            + 0.20 * moderate_sep
            + 0.10 * normalize(-distance_to_anchor[available])
        )
        selected.append(int(available[int(np.nanargmax(score))]))

    # Build the full deterministic ranking once, then return only the requested prefix.
    selected = selected[:reference_k]
    out = pts.iloc[selected[:int(k)]].copy().reset_index(drop=True)
    out["selection_rank"] = np.arange(1, len(out) + 1)
    out["observed_sector_count"] = out[rss_cols].notna().sum(axis=1)
    out["anchor_x_m"] = float(anchor[0])
    out["anchor_y_m"] = float(anchor[1])
    return out, anchor


def _hull_area(xy: np.ndarray) -> float:
    xy = np.asarray(xy, dtype=float)
    if len(xy) < 3:
        return 0.0
    try:
        return float(ConvexHull(xy).volume)
    except QhullError:
        return 0.0


def _random_subset_geometry_score(
    pts: pd.DataFrame,
    subset_idx: np.ndarray,
    eligible_idx: np.ndarray,
    omni: bool,
) -> float:
    """Geometry/PCI quality of a random subset, without using RSRP magnitude or truth."""
    rss_cols = ["rsrp_s1", "rsrp_s2", "rsrp_s3"]
    selected = pts.iloc[np.asarray(subset_idx, dtype=int)]
    all_xy = pts.iloc[np.asarray(eligible_idx, dtype=int)][["x_m", "y_m"]].to_numpy(float)
    sel_xy = selected[["x_m", "y_m"]].to_numpy(float)
    if len(sel_xy) == 0:
        return -np.inf

    all_diag = float(np.linalg.norm(np.ptp(all_xy, axis=0))) if len(all_xy) else 1.0
    sel_diag = float(np.linalg.norm(np.ptp(sel_xy, axis=0))) if len(sel_xy) else 0.0
    spread_score = float(np.clip(sel_diag / max(all_diag, 1.0), 0.0, 1.0))

    all_area = _hull_area(all_xy)
    sel_area = _hull_area(sel_xy)
    hull_score = float(np.clip(sel_area / max(all_area, 1.0), 0.0, 1.0)) if all_area > 0 else spread_score

    # Directly score how well the selected points cover the full candidate pool.
    tree = cKDTree(sel_xy)
    distances, _ = tree.query(all_xy, k=1)
    scale = max(0.20 * all_diag, 40.0)
    coverage_score = float(np.exp(-float(np.mean(distances)) / scale))
    tail_score = float(np.exp(-float(np.percentile(distances, 90)) / max(0.30 * all_diag, 60.0)))

    if omni:
        pci_score = 1.0
    else:
        finite = selected[rss_cols].notna().to_numpy(dtype=bool)
        sector_counts = finite.sum(axis=0).astype(float)
        pci_balance = float(np.min(sector_counts) / max(np.max(sector_counts), 1.0))
        pair_count = float(
            np.sum(finite[:, 0] & finite[:, 1])
            + np.sum(finite[:, 0] & finite[:, 2])
            + np.sum(finite[:, 1] & finite[:, 2])
        )
        pair_score = float(np.clip(pair_count / max(0.5 * len(selected), 2.0), 0.0, 1.0))
        pci_score = 0.65 * pci_balance + 0.35 * pair_score

    return float(
        0.38 * coverage_score
        + 0.22 * tail_score
        + 0.18 * hull_score
        + 0.10 * spread_score
        + 0.12 * pci_score
    )


def select_random_localization_points(
    points: pd.DataFrame,
    k: int,
    seed: int,
    omni: bool,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Random 10--15 point sampling with geometry-quality rejection.

    The requested experiment remains stochastic: each trial draws many random
    subsets and then randomly chooses one from the best geometry-quality quartile.
    This avoids pathological clustered random subsets while preserving Monte-Carlo
    randomness.  No RSRP magnitude, reference station coordinate, or localization
    error is used in this filtering stage.
    """
    pts = points.copy().reset_index(drop=True)
    rss_cols = ["rsrp_s1", "rsrp_s2", "rsrp_s3"]
    coverage = pts[rss_cols].notna().sum(axis=1).to_numpy(dtype=int)
    eligible = np.flatnonzero(coverage >= 1)
    if len(eligible) < int(k):
        raise ValueError(
            f"定位可用空间点仅{len(eligible)}个，少于要求{k}个；"
            "候选定义为至少包含1个目标PCI的有效实测位置"
        )

    rng = np.random.default_rng(int(seed))
    complete = np.flatnonzero(coverage >= 3) if not omni else np.empty(0, dtype=int)
    draws: list[tuple[float, np.ndarray]] = []
    n_draws = int(np.clip(12 * int(k), 96, 240))

    for _ in range(n_draws):
        if omni:
            trial = rng.choice(eligible, size=int(k), replace=False)
        elif len(complete) >= 2 and int(k) >= 2:
            mandatory = rng.choice(complete, size=2, replace=False)
            mandatory_set = set(int(v) for v in mandatory)
            remaining_pool = np.asarray([int(v) for v in eligible if int(v) not in mandatory_set], dtype=int)
            remainder = rng.choice(remaining_pool, size=int(k) - 2, replace=False)
            trial = np.concatenate([mandatory, remainder])
            rng.shuffle(trial)
        else:
            trial = rng.choice(eligible, size=int(k), replace=False)

        trial_frame = pts.iloc[trial]
        finite = trial_frame[rss_cols].notna().to_numpy(dtype=bool)
        if not omni:
            sector_counts = finite.sum(axis=0)
            pair_count = int(
                np.sum(finite[:, 0] & finite[:, 1])
                + np.sum(finite[:, 0] & finite[:, 2])
                + np.sum(finite[:, 1] & finite[:, 2])
            )
            if not np.all(sector_counts >= 1) or pair_count < 2:
                continue
        score = _random_subset_geometry_score(pts, trial, eligible, omni)
        if np.isfinite(score):
            draws.append((float(score), np.asarray(trial, dtype=int)))

    if not draws:
        raise ValueError("无法构造满足PCI可辨识约束的随机定位样本")

    draws.sort(key=lambda item: item[0], reverse=True)
    top_n = max(1, int(math.ceil(0.25 * len(draws))))
    # Random choice inside the top geometry-quality quartile: still Monte-Carlo,
    # but avoids letting one highly clustered subset dominate the experiment.
    pick = int(rng.integers(0, top_n))
    chosen_score, chosen = draws[pick]

    out = pts.iloc[np.asarray(chosen, dtype=int)].copy().reset_index(drop=True)
    out["selection_rank"] = np.arange(1, len(out) + 1, dtype=int)
    out["observed_sector_count"] = out[rss_cols].notna().sum(axis=1)
    anchor = strong_anchor(out, top_n=min(20, len(out)))
    out["anchor_x_m"] = float(anchor[0])
    out["anchor_y_m"] = float(anchor[1])
    out["selection_mode"] = "random"
    out["selection_strategy"] = "geometry_filtered_top_quartile"
    out["selection_constraint"] = "global_pci_identifiability_and_geometry_quality"
    out["random_candidate_count"] = int(len(eligible))
    out["complete_three_pci_candidate_count"] = int(np.sum(coverage >= 3)) if not omni else 0
    out["random_subset_geometry_score"] = float(chosen_score)
    out["random_subset_candidate_draws"] = int(len(draws))
    out["random_subset_top_quartile_size"] = int(top_n)
    return out, np.asarray(anchor, dtype=float)

def wrap_angle(value: np.ndarray | float) -> np.ndarray | float:
    return (value + np.pi) % (2.0 * np.pi) - np.pi


def horizontal_gain_db(phi: np.ndarray, scale: float) -> np.ndarray:
    bw = math.radians(HORIZONTAL_3DB_BEAMWIDTH_DEG)
    attenuation = np.minimum(
        12.0 * (wrap_angle(phi) / bw) ** 2,
        HORIZONTAL_MAX_ATTENUATION_DB,
    )
    return -float(scale) * attenuation


def prepare_arrays(selected: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xy = selected[["x_m", "y_m"]].to_numpy(float)
    rss = selected[["rsrp_s1", "rsrp_s2", "rsrp_s3"]].to_numpy(float)
    envelope = np.nanmax(rss, axis=1)
    return xy, rss, envelope



def geometry_diagnostics(tx_xy: np.ndarray, selected: pd.DataFrame, omni: bool) -> Dict[str, float | bool]:
    """Geometry-only confidence diagnostics; never uses the true station coordinate."""
    xy = selected[["x_m", "y_m"]].to_numpy(float)
    tx = np.asarray(tx_xy, dtype=float)
    if len(xy) == 0 or not np.isfinite(tx).all():
        return {
            "angular_coverage_deg": 0.0,
            "max_angular_gap_deg": 360.0,
            "inside_measurement_convex_hull": False,
            "extrapolation_ratio": float("inf"),
            "pci_spatial_balance": 0.0,
            "geometry_reliability_score": 0.0,
        }

    bearings = np.mod(np.degrees(np.arctan2(xy[:, 1] - tx[1], xy[:, 0] - tx[0])), 360.0)
    bearings = np.sort(bearings)
    if len(bearings) >= 2:
        gaps = np.diff(np.r_[bearings, bearings[0] + 360.0])
        max_gap = float(np.max(gaps))
        coverage = float(360.0 - max_gap)
    else:
        max_gap, coverage = 360.0, 0.0

    centroid = np.mean(xy, axis=0)
    radius = np.linalg.norm(xy - centroid[None, :], axis=1)
    support_radius = float(np.quantile(radius, 0.75)) if len(radius) else 1.0
    extrapolation = float(np.linalg.norm(tx - centroid) / max(support_radius, 10.0))

    inside = False
    if len(xy) >= 3:
        try:
            hull = ConvexHull(xy)
            inside = bool(np.all(hull.equations[:, :-1] @ tx + hull.equations[:, -1] <= 1e-7))
        except QhullError:
            inside = False

    if omni:
        pci_balance = 1.0
    else:
        counts = selected[["rsrp_s1", "rsrp_s2", "rsrp_s3"]].notna().sum(axis=0).to_numpy(float)
        pci_balance = float(np.min(counts) / max(np.max(counts), 1.0))

    coverage_score = float(np.clip(coverage / 180.0, 0.0, 1.0))
    extrap_score = float(np.exp(-max(0.0, extrapolation - 1.0)))
    hull_score = 1.0 if inside else 0.72
    reliability = float(np.clip(0.42 * coverage_score + 0.23 * pci_balance + 0.20 * extrap_score + 0.15 * hull_score, 0.0, 1.0))
    return {
        "angular_coverage_deg": coverage,
        "max_angular_gap_deg": max_gap,
        "inside_measurement_convex_hull": inside,
        "extrapolation_ratio": extrapolation,
        "pci_spatial_balance": pci_balance,
        "geometry_reliability_score": reliability,
    }


def _weighted_xy(xy: np.ndarray, values: np.ndarray, top_n: int = 5) -> np.ndarray:
    mask = np.isfinite(values)
    if not np.any(mask):
        return np.mean(xy, axis=0)
    idx = np.flatnonzero(mask)
    idx = idx[np.argsort(values[idx])[::-1]][: min(top_n, len(idx))]
    w = np.exp((values[idx] - np.max(values[idx])) / 8.0)
    return np.average(xy[idx], axis=0, weights=w)


def multistart_search_centers(selected: pd.DataFrame, anchor: np.ndarray, omni: bool) -> List[Tuple[str, np.ndarray]]:
    """Build measurement-only alternative search centers for difficult sites."""
    xy, rss, envelope = prepare_arrays(selected)
    centers: List[Tuple[str, np.ndarray]] = [("strong_anchor", np.asarray(anchor, dtype=float))]
    centers.append(("measurement_centroid", np.mean(xy, axis=0)))
    centers.append(("envelope_centroid", _weighted_xy(xy, envelope, top_n=max(3, min(6, len(xy))))))
    if not omni:
        sector_centers = [_weighted_xy(xy, rss[:, j], top_n=max(2, min(4, len(xy)))) for j in range(3)]
        centers.append(("sector_balanced_centroid", np.mean(np.vstack(sector_centers), axis=0)))
    unique: List[Tuple[str, np.ndarray]] = []
    for label, center in centers:
        center = np.asarray(center, dtype=float)
        if not np.isfinite(center).all():
            continue
        if all(np.linalg.norm(center - prev) >= 20.0 for _, prev in unique):
            unique.append((label, center))
    return unique


def _candidate_selection_score(solution: Dict[str, Any], selected: pd.DataFrame, omni: bool) -> float:
    """Physics objective with a mild geometry penalty; no truth coordinate is used."""
    diag = geometry_diagnostics(np.asarray(solution["model_xy"], dtype=float), selected, omni)
    score = float(solution["objective"])
    score += 0.35 * max(0.0, (125.0 - float(diag["angular_coverage_deg"])) / 45.0) ** 2
    score += 0.25 * max(0.0, float(diag["extrapolation_ratio"]) - 1.8) ** 2
    margin = float(solution.get("optimizer_boundary_margin_m", np.nan))
    if np.isfinite(margin) and margin < 8.0:
        score += 0.4
    return float(score)

def robust_log_loss(residual_scaled: np.ndarray, dof: float = 4.0) -> float:
    return float(np.sum(np.log1p((residual_scaled ** 2) / dof)))


def directional_objective(
    params: np.ndarray,
    xy: np.ndarray,
    rss: np.ndarray,
    envelope: np.ndarray,
    anchor: np.ndarray,
    sign: int,
    alpha_mode: str,
    alpha_prior: Optional[float],
    alpha_sigma_deg: float,
) -> float:
    if alpha_mode == "fixed":
        tx_x, tx_y, gain_scale, exponent, intercept = [float(v) for v in params]
        if alpha_prior is None:
            return 1e12
        alpha = float(alpha_prior)
    else:
        tx_x, tx_y, alpha, gain_scale, exponent, intercept = [float(v) for v in params]

    tx = np.asarray([tx_x, tx_y])
    distance = np.sqrt(np.sum((xy - tx[None, :]) ** 2, axis=1) + TX_RX_VERTICAL_SEPARATION_M ** 2)
    bearing = np.arctan2(xy[:, 1] - tx_y, xy[:, 0] - tx_x)
    offsets = np.asarray([0.0, sign * 2.0 * np.pi / 3.0, -sign * 2.0 * np.pi / 3.0])
    gains = np.stack(
        [horizontal_gain_db(bearing - (alpha + offset), gain_scale) for offset in offsets],
        axis=1,
    )

    complete = np.isfinite(rss).all(axis=1)

    # 同点扇区差分：优先使用完整三PCI点；若随机样本中的完整点不足2个，
    # 则使用所有可用的同点PCI成对差分。这样允许随机选中的位置只有1/2个PCI，
    # 同时仍保留消除共同路径损耗和未知功率截距的方向约束。
    if int(np.sum(complete)) >= 2:
        observed_centered = rss[complete] - np.mean(rss[complete], axis=1, keepdims=True)
        predicted_centered = gains[complete] - np.mean(gains[complete], axis=1, keepdims=True)
        sector_residual = (observed_centered - predicted_centered) / ROBUST_SECTOR_DIFF_SCALE_DB
        loss = 1.6 * robust_log_loss(sector_residual)
    else:
        pair_residuals: list[np.ndarray] = []
        for i, j in ((0, 1), (0, 2), (1, 2)):
            mask = np.isfinite(rss[:, i]) & np.isfinite(rss[:, j])
            if np.any(mask):
                observed_diff = rss[mask, i] - rss[mask, j]
                predicted_diff = gains[mask, i] - gains[mask, j]
                pair_residuals.append((observed_diff - predicted_diff) / ROBUST_SECTOR_DIFF_SCALE_DB)
        pair_count = int(sum(len(v) for v in pair_residuals))
        if pair_count < 2:
            return 1e12
        sector_residual = np.concatenate(pair_residuals)
        loss = 1.6 * robust_log_loss(sector_residual)

    # 绝对包络：提供距离约束。
    predicted_envelope = (
        intercept
        - 10.0 * exponent * np.log10(np.maximum(distance, 1.0))
        + np.max(gains, axis=1)
    )
    envelope_residual = (envelope - predicted_envelope) / ROBUST_ENVELOPE_SCALE_DB
    loss += robust_log_loss(envelope_residual)

    # 物理与空间先验。
    loss += 0.8 * float(np.sum(((tx - anchor) / ANCHOR_PRIOR_SIGMA_M) ** 2))
    loss += 0.45 * ((exponent - PATHLOSS_EXPONENT_PRIOR) / PATHLOSS_EXPONENT_STD) ** 2
    loss += 0.45 * ((intercept - REFERENCE_RSRP_AT_1M_PRIOR_DBM) / REFERENCE_RSRP_AT_1M_STD_DB) ** 2
    loss += 0.15 * ((gain_scale - 0.8) / 0.35) ** 2

    if alpha_mode == "soft" and alpha_prior is not None:
        sigma = math.radians(float(np.clip(alpha_sigma_deg, 15.0, 60.0)))
        loss += 1.5 * (float(wrap_angle(alpha - alpha_prior)) / sigma) ** 2
    return float(loss)


def omni_objective(
    params: np.ndarray,
    xy: np.ndarray,
    envelope: np.ndarray,
    anchor: np.ndarray,
) -> float:
    tx_x, tx_y, exponent, intercept = [float(v) for v in params]
    tx = np.asarray([tx_x, tx_y])
    distance = np.sqrt(np.sum((xy - tx[None, :]) ** 2, axis=1) + TX_RX_VERTICAL_SEPARATION_M ** 2)
    predicted = intercept - 10.0 * exponent * np.log10(np.maximum(distance, 1.0))
    residual = (envelope - predicted) / 6.0
    loss = robust_log_loss(residual)
    loss += 1.0 * float(np.sum(((tx - anchor) / ANCHOR_PRIOR_SIGMA_M) ** 2))
    loss += 0.5 * ((exponent - PATHLOSS_EXPONENT_PRIOR) / 0.60) ** 2
    loss += 0.6 * ((intercept - REFERENCE_RSRP_AT_1M_PRIOR_DBM) / 8.0) ** 2
    return float(loss)


def _profile_pathloss_parameters(
    distance_m: np.ndarray,
    adjusted_envelope_dbm: np.ndarray,
    *,
    residual_scale_db: float,
    exponent_prior: float = PATHLOSS_EXPONENT_PRIOR,
    exponent_std: float = PATHLOSS_EXPONENT_STD,
    intercept_prior: float = REFERENCE_RSRP_AT_1M_PRIOR_DBM,
    intercept_std: float = REFERENCE_RSRP_AT_1M_STD_DB,
) -> tuple[float, float, np.ndarray]:
    """Robustly profile out path-loss exponent/intercept for a candidate XY.

    This removes two poorly identifiable variables from the global optimizer.
    For every candidate transmitter location the nuisance parameters are fitted
    by a small IRLS ridge regression with the same physical priors.
    """
    distance = np.asarray(distance_m, dtype=float)
    y = np.asarray(adjusted_envelope_dbm, dtype=float)
    finite = np.isfinite(distance) & np.isfinite(y) & (distance > 0)
    if int(np.sum(finite)) < 3:
        return float(exponent_prior), float(intercept_prior), np.full_like(y, np.nan, dtype=float)
    d = distance[finite]
    yf = y[finite]
    x = -10.0 * np.log10(np.maximum(d, 1.0))
    X = np.column_stack([np.ones_like(x), x])
    weights = np.ones_like(yf)
    beta = np.asarray([intercept_prior, exponent_prior], dtype=float)
    scale2 = max(float(residual_scale_db) ** 2, 1.0)
    prior_precision = np.diag([
        0.40 / max(float(intercept_std) ** 2, 1e-6),
        0.40 / max(float(exponent_std) ** 2, 1e-6),
    ])
    prior_rhs = prior_precision @ np.asarray([intercept_prior, exponent_prior], dtype=float)
    for _ in range(5):
        W = weights / scale2
        lhs = X.T @ (W[:, None] * X) + prior_precision
        rhs = X.T @ (W * yf) + prior_rhs
        try:
            beta = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            beta = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
        beta[0] = float(np.clip(beta[0], -42.0, -3.0))
        beta[1] = float(np.clip(beta[1], 1.5, 4.2))
        resid = yf - X @ beta
        z = np.abs(resid) / max(float(residual_scale_db), 1e-6)
        weights = np.where(z <= 1.5, 1.0, 1.5 / np.maximum(z, 1e-6))
    full_resid = np.full_like(y, np.nan, dtype=float)
    full_resid[finite] = yf - X @ beta
    return float(beta[1]), float(beta[0]), full_resid


def _profiled_directional_objective(
    params: np.ndarray,
    xy: np.ndarray,
    rss: np.ndarray,
    envelope: np.ndarray,
    anchor: np.ndarray,
    sign: int,
    alpha_mode: str,
    alpha_prior: Optional[float],
    alpha_sigma_deg: float,
) -> float:
    if alpha_mode == "fixed":
        tx_x, tx_y, gain_scale = [float(v) for v in params]
        if alpha_prior is None:
            return 1e12
        alpha = float(alpha_prior)
    else:
        tx_x, tx_y, alpha, gain_scale = [float(v) for v in params]

    tx = np.asarray([tx_x, tx_y], dtype=float)
    distance = np.sqrt(np.sum((xy - tx[None, :]) ** 2, axis=1) + TX_RX_VERTICAL_SEPARATION_M ** 2)
    bearing = np.arctan2(xy[:, 1] - tx_y, xy[:, 0] - tx_x)
    offsets = np.asarray([0.0, sign * 2.0 * np.pi / 3.0, -sign * 2.0 * np.pi / 3.0])
    gains = np.stack([horizontal_gain_db(bearing - (alpha + offset), gain_scale) for offset in offsets], axis=1)

    complete = np.isfinite(rss).all(axis=1)
    if int(np.sum(complete)) >= 2:
        observed_centered = rss[complete] - np.mean(rss[complete], axis=1, keepdims=True)
        predicted_centered = gains[complete] - np.mean(gains[complete], axis=1, keepdims=True)
        sector_residual = (observed_centered - predicted_centered) / ROBUST_SECTOR_DIFF_SCALE_DB
    else:
        pair_residuals: list[np.ndarray] = []
        for i, j in ((0, 1), (0, 2), (1, 2)):
            mask = np.isfinite(rss[:, i]) & np.isfinite(rss[:, j])
            if np.any(mask):
                pair_residuals.append(
                    ((rss[mask, i] - rss[mask, j]) - (gains[mask, i] - gains[mask, j]))
                    / ROBUST_SECTOR_DIFF_SCALE_DB
                )
        if int(sum(len(v) for v in pair_residuals)) < 2:
            return 1e12
        sector_residual = np.concatenate(pair_residuals)
    loss = 1.8 * robust_log_loss(sector_residual)

    max_gain = np.nanmax(gains, axis=1)
    adjusted = envelope - max_gain
    exponent, intercept, residual = _profile_pathloss_parameters(
        distance, adjusted, residual_scale_db=ROBUST_ENVELOPE_SCALE_DB,
    )
    env_scaled = residual[np.isfinite(residual)] / ROBUST_ENVELOPE_SCALE_DB
    if len(env_scaled) < 3:
        return 1e12
    loss += robust_log_loss(env_scaled)

    # One weak position prior only.  The v1.9 random experiment showed that a
    # strong anchor plus post-hoc fusion can systematically bias random subsets.
    loss += 0.25 * float(np.sum(((tx - anchor) / ANCHOR_PRIOR_SIGMA_M) ** 2))
    loss += 0.12 * ((gain_scale - 0.8) / 0.35) ** 2
    loss += 0.20 * ((exponent - PATHLOSS_EXPONENT_PRIOR) / PATHLOSS_EXPONENT_STD) ** 2
    loss += 0.20 * ((intercept - REFERENCE_RSRP_AT_1M_PRIOR_DBM) / REFERENCE_RSRP_AT_1M_STD_DB) ** 2
    if alpha_mode == "soft" and alpha_prior is not None:
        sigma = math.radians(float(np.clip(alpha_sigma_deg, 15.0, 60.0)))
        loss += 1.5 * (float(wrap_angle(alpha - alpha_prior)) / sigma) ** 2
    return float(loss)


def _profiled_omni_objective(
    params: np.ndarray,
    xy: np.ndarray,
    envelope: np.ndarray,
    anchor: np.ndarray,
) -> float:
    tx = np.asarray(params[:2], dtype=float)
    distance = np.sqrt(np.sum((xy - tx[None, :]) ** 2, axis=1) + TX_RX_VERTICAL_SEPARATION_M ** 2)
    exponent, intercept, residual = _profile_pathloss_parameters(
        distance, envelope, residual_scale_db=6.0,
        exponent_std=0.60, intercept_std=8.0,
    )
    scaled = residual[np.isfinite(residual)] / 6.0
    if len(scaled) < 3:
        return 1e12
    loss = robust_log_loss(scaled)
    loss += 0.30 * float(np.sum(((tx - anchor) / ANCHOR_PRIOR_SIGMA_M) ** 2))
    loss += 0.20 * ((exponent - PATHLOSS_EXPONENT_PRIOR) / 0.60) ** 2
    loss += 0.20 * ((intercept - REFERENCE_RSRP_AT_1M_PRIOR_DBM) / 8.0) ** 2
    return float(loss)


def direction_ray_intersection_estimate(
    selected: pd.DataFrame,
    direction_row: Optional[pd.Series],
) -> tuple[np.ndarray | None, float]:
    """Robust line-intersection estimate from fixed sector-direction priors.

    This estimate uses only receiver coordinates, PCI availability and the
    external sector directions.  It does not use the true station coordinate or
    filled radio-map labels.  It is used as an additional optimization center.
    """
    if direction_row is None or not np.isfinite(direction_row.get("base_alpha_rad", np.nan)):
        return None, float("nan")
    alpha = float(direction_row["base_alpha_rad"])
    order_text = str(direction_row.get("selected_sector_order", ""))
    sign = 1 if "plus120" in order_text else -1
    offsets = np.asarray([0.0, sign * 2.0 * np.pi / 3.0, -sign * 2.0 * np.pi / 3.0])
    xy = selected[["x_m", "y_m"]].to_numpy(float)
    rss = selected[["rsrp_s1", "rsrp_s2", "rsrp_s3"]].to_numpy(float)
    A_rows: list[np.ndarray] = []
    b_rows: list[float] = []
    dirs: list[np.ndarray] = []
    points: list[np.ndarray] = []
    for i in range(len(xy)):
        for j in range(3):
            if not np.isfinite(rss[i, j]):
                continue
            theta = alpha + float(offsets[j])
            u = np.asarray([math.cos(theta), math.sin(theta)], dtype=float)
            n = np.asarray([-u[1], u[0]], dtype=float)
            A_rows.append(n)
            b_rows.append(float(n @ xy[i]))
            dirs.append(u)
            points.append(xy[i])
    if len(A_rows) < 4:
        return None, float("nan")
    A = np.vstack(A_rows)
    b = np.asarray(b_rows, dtype=float)
    weights = np.ones(len(b), dtype=float)
    solution = None
    for _ in range(6):
        Aw = A * np.sqrt(weights)[:, None]
        bw = b * np.sqrt(weights)
        try:
            solution = np.linalg.lstsq(Aw, bw, rcond=None)[0]
        except np.linalg.LinAlgError:
            return None, float("nan")
        residual = A @ solution - b
        scale = max(float(np.median(np.abs(residual))) * 1.4826, 15.0)
        z = np.abs(residual) / scale
        weights = np.where(z <= 1.5, 1.0, 1.5 / np.maximum(z, 1e-6))
        # A receiver should normally be in front of its serving sector.  Lines
        # violating this half-line condition are softly down-weighted.
        for q, (u, point) in enumerate(zip(dirs, points)):
            forward_distance = float(u @ (point - solution))
            if forward_distance < -20.0:
                weights[q] *= 0.20
    if solution is None or not np.isfinite(solution).all():
        return None, float("nan")
    residual = A @ solution - b
    return np.asarray(solution, dtype=float), float(np.sqrt(np.mean(residual ** 2)))


def clipped_local_bounds(anchor: np.ndarray, global_bounds: Sequence[float]) -> Tuple[float, float, float, float]:
    gx0, gx1, gy0, gy1 = [float(v) for v in global_bounds]
    return (
        max(gx0, float(anchor[0]) - LOCAL_SEARCH_RADIUS_M),
        min(gx1, float(anchor[0]) + LOCAL_SEARCH_RADIUS_M),
        max(gy0, float(anchor[1]) - LOCAL_SEARCH_RADIUS_M),
        min(gy1, float(anchor[1]) + LOCAL_SEARCH_RADIUS_M),
    )


def optimize_station(
    selected: pd.DataFrame,
    station_id: int,
    omni: bool,
    anchor: np.ndarray,
    direction_row: Optional[pd.Series],
    direction_mode: str,
    global_bounds: Sequence[float],
    seed: int,
    de_maxiter: int,
    de_popsize: int,
    search_center: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Profiled robust localization solver (v1.10).

    Compared with v1.9, the global optimizer no longer jointly searches the
    path-loss exponent and reference-power intercept.  Those nuisance parameters
    are robustly profiled at each candidate location, reducing the random-sample
    inverse problem from 5/6 dimensions to 2--4 dimensions.
    """
    xy, rss, envelope = prepare_arrays(selected)
    bound_center = np.asarray(anchor if search_center is None else search_center, dtype=float)
    x0, x1, y0, y1 = clipped_local_bounds(bound_center, global_bounds)

    if omni:
        result = differential_evolution(
            lambda p: _profiled_omni_objective(p, xy, envelope, anchor),
            bounds=[(x0, x1), (y0, y1)],
            seed=int(seed % (2**32 - 1)), popsize=int(de_popsize), maxiter=int(de_maxiter),
            tol=1e-6, mutation=(0.5, 1.0), recombination=0.75, polish=True, workers=1,
        )
        model_xy = np.asarray(result.x[:2], dtype=float)
        distance = np.sqrt(np.sum((xy - model_xy[None, :]) ** 2, axis=1) + TX_RX_VERTICAL_SEPARATION_M ** 2)
        exponent, intercept, _ = _profile_pathloss_parameters(
            distance, envelope, residual_scale_db=6.0, exponent_std=0.60, intercept_std=8.0,
        )
        params = np.asarray([model_xy[0], model_xy[1], exponent, intercept], dtype=float)
        disagreement = float(np.linalg.norm(model_xy - anchor))
        boundary_margin = float(min(model_xy[0]-x0, x1-model_xy[0], model_xy[1]-y0, y1-model_xy[1]))
        omni_weight = 0.90
        final_xy = omni_weight * model_xy + (1.0 - omni_weight) * np.asarray(anchor, dtype=float)
        return {
            "final_xy": final_xy, "model_xy": model_xy, "anchor": anchor,
            "model_weight": omni_weight, "model_anchor_disagreement_m": disagreement,
            "optimizer_boundary_margin_m": boundary_margin, "params": params,
            "sign": 1, "alpha": 0.0, "objective": float(result.fun),
            "direction_prior_used": False, "direction_fit_rms_deg": float("nan"),
            "solver_mode": "profiled_pathloss",
        }

    prior_available = direction_row is not None and np.isfinite(direction_row.get("base_alpha_rad", np.nan))
    if direction_mode in {"fixed", "soft"} and not prior_available:
        print(f"警告：站{station_id}缺少方向先验，自动退回off模式。")
        direction_mode = "off"
    if prior_available:
        alpha_prior = float(direction_row["base_alpha_rad"])
        order_text = str(direction_row["selected_sector_order"])
        prior_sign = 1 if "plus120" in order_text else -1
        direction_rms = float(direction_row.get("direction_fit_rms_deg", np.nan))
    else:
        alpha_prior = None
        prior_sign = 1
        direction_rms = float("nan")

    candidates: List[Tuple[float, np.ndarray, int, float]] = []
    signs = [prior_sign] if direction_mode in {"fixed", "soft"} else [1, -1]
    for sign_idx, sign in enumerate(signs):
        if direction_mode == "fixed":
            bounds = [(x0, x1), (y0, y1), (0.30, 1.20)]
        else:
            bounds = [(x0, x1), (y0, y1), (-math.pi, math.pi), (0.30, 1.20)]
        local_seed = int((seed + sign_idx * 100003) % (2**32 - 1))
        result = differential_evolution(
            lambda p: _profiled_directional_objective(
                p, xy, rss, envelope, anchor, int(sign), direction_mode,
                alpha_prior, direction_rms,
            ),
            bounds=bounds, seed=local_seed, popsize=int(de_popsize), maxiter=int(de_maxiter),
            tol=1e-6, mutation=(0.5, 1.0), recombination=0.75, polish=True, workers=1,
        )
        alpha = float(alpha_prior) if direction_mode == "fixed" else float(result.x[2])
        candidates.append((float(result.fun), np.asarray(result.x, dtype=float), int(sign), alpha))

    objective, compact_params, sign, alpha = min(candidates, key=lambda item: item[0])
    model_xy = np.asarray(compact_params[:2], dtype=float)
    gain_scale = float(compact_params[2] if direction_mode == "fixed" else compact_params[3])
    distance = np.sqrt(np.sum((xy - model_xy[None, :]) ** 2, axis=1) + TX_RX_VERTICAL_SEPARATION_M ** 2)
    bearing = np.arctan2(xy[:, 1] - model_xy[1], xy[:, 0] - model_xy[0])
    offsets = np.asarray([0.0, sign * 2.0 * np.pi / 3.0, -sign * 2.0 * np.pi / 3.0])
    gains = np.stack([horizontal_gain_db(bearing - (alpha + offset), gain_scale) for offset in offsets], axis=1)
    exponent, intercept, _ = _profile_pathloss_parameters(
        distance, envelope - np.nanmax(gains, axis=1), residual_scale_db=ROBUST_ENVELOPE_SCALE_DB,
    )
    if direction_mode == "fixed":
        params = np.asarray([model_xy[0], model_xy[1], gain_scale, exponent, intercept], dtype=float)
    else:
        params = np.asarray([model_xy[0], model_xy[1], alpha, gain_scale, exponent, intercept], dtype=float)
    disagreement = float(np.linalg.norm(model_xy - anchor))
    boundary_margin = float(min(model_xy[0]-x0, x1-model_xy[0], model_xy[1]-y0, y1-model_xy[1]))
    return {
        "final_xy": model_xy.copy(), "model_xy": model_xy, "anchor": anchor,
        "model_weight": 1.0, "model_anchor_disagreement_m": disagreement,
        "optimizer_boundary_margin_m": boundary_margin, "params": params,
        "sign": sign, "alpha": alpha, "objective": objective,
        "direction_prior_used": direction_mode in {"fixed", "soft"} and prior_available,
        "direction_fit_rms_deg": direction_rms, "solver_mode": "profiled_pathloss",
    }


def optimize_station_robust(
    selected: pd.DataFrame,
    station_id: int,
    omni: bool,
    anchor: np.ndarray,
    direction_row: Optional[pd.Series],
    direction_mode: str,
    global_bounds: Sequence[float],
    seed: int,
    de_maxiter: int,
    de_popsize: int,
) -> Dict[str, Any]:
    """Robust profiled solver with direction-ray and geometry multistart centers."""
    candidates: List[Tuple[str, Dict[str, Any]]] = []

    # Always evaluate the standard random-subset anchor center.
    base = optimize_station(
        selected, station_id, omni, anchor, direction_row, direction_mode,
        global_bounds, seed, de_maxiter, de_popsize, search_center=anchor,
    )
    candidates.append(("strong_anchor", base))

    # With a fixed/soft external sector prior, a robust line-intersection estimate
    # provides a complementary position cue independent of path-loss magnitude.
    ray_xy: np.ndarray | None = None
    ray_rms = float("nan")
    if (not omni) and direction_mode in {"fixed", "soft"}:
        ray_xy, ray_rms = direction_ray_intersection_estimate(selected, direction_row)
        if ray_xy is not None and np.isfinite(ray_xy).all():
            gx0, gx1, gy0, gy1 = [float(v) for v in global_bounds]
            if gx0 <= ray_xy[0] <= gx1 and gy0 <= ray_xy[1] <= gy1:
                try:
                    ray_trial = optimize_station(
                        selected, station_id, omni, anchor, direction_row, direction_mode,
                        global_bounds, seed + 104729,
                        max(45, int(round(0.80 * de_maxiter))),
                        max(7, int(round(0.85 * de_popsize))),
                        search_center=ray_xy,
                    )
                    candidates.append(("direction_ray", ray_trial))
                except Exception:
                    pass

    # Additional centers are used only when the base geometry is visibly weak.
    diag = geometry_diagnostics(np.asarray(base["model_xy"], dtype=float), selected, omni)
    disagreement = float(base.get("model_anchor_disagreement_m", 0.0))
    boundary = float(base.get("optimizer_boundary_margin_m", np.nan))
    geometry_bad = (
        (float(diag["angular_coverage_deg"]) < 110.0 and float(diag["extrapolation_ratio"]) > 1.25)
        or float(diag["extrapolation_ratio"]) > 2.10
        or float(diag["geometry_reliability_score"]) < 0.50
    )
    trigger_extra = bool(disagreement > 180.0 or (np.isfinite(boundary) and boundary < 8.0) or geometry_bad)
    if trigger_extra:
        centers = multistart_search_centers(selected, anchor, omni)
        alt_iter = max(40, int(round(0.65 * de_maxiter)))
        alt_pop = max(7, int(round(0.75 * de_popsize)))
        existing_centers = [anchor]
        if ray_xy is not None:
            existing_centers.append(ray_xy)
        for j, (label, center) in enumerate(centers):
            if any(np.linalg.norm(np.asarray(center, float) - np.asarray(prev, float)) < 25.0 for prev in existing_centers):
                continue
            try:
                trial = optimize_station(
                    selected, station_id, omni, anchor, direction_row, direction_mode,
                    global_bounds, seed + (j + 2) * 104729, alt_iter, alt_pop,
                    search_center=center,
                )
                candidates.append((label, trial))
                existing_centers.append(np.asarray(center, float))
            except Exception:
                continue

    label, best = min(candidates, key=lambda item: _candidate_selection_score(item[1], selected, omni))
    best = dict(best)
    best["multistart_triggered"] = bool(len(candidates) > 1)
    best["multistart_candidate_count"] = int(len(candidates))
    best["multistart_selected_label"] = label
    best["geometry_diagnostics"] = geometry_diagnostics(np.asarray(best["model_xy"], dtype=float), selected, omni)
    best["anchor_xy"] = np.asarray(anchor, dtype=float)
    best["omni"] = bool(omni)
    best["direction_ray_x_m"] = float(ray_xy[0]) if ray_xy is not None else float("nan")
    best["direction_ray_y_m"] = float(ray_xy[1]) if ray_xy is not None else float("nan")
    best["direction_ray_perpendicular_rms_m"] = float(ray_rms)
    return best


def bootstrap_uncertainty(
    optimum: Dict[str, Any],
    selected: pd.DataFrame,
    station_id: int,
    omni: bool,
    direction_row: Optional[pd.Series],
    direction_mode: str,
    global_bounds: Sequence[float],
    count: int,
    seed: int,
) -> Tuple[float, np.ndarray]:
    if count <= 0:
        return float("nan"), np.empty((0, 2), dtype=float)
    rng = np.random.default_rng(seed)
    locations: List[np.ndarray] = []
    for index in range(count):
        noisy = selected.copy()
        for col in ["rsrp_s1", "rsrp_s2", "rsrp_s3"]:
            values = noisy[col].to_numpy(float)
            mask = np.isfinite(values)
            values[mask] += rng.standard_t(df=4, size=int(np.sum(mask))) * 2.0
            noisy[col] = values
        try:
            trial = optimize_station(
                selected=noisy,
                station_id=station_id,
                omni=omni,
                anchor=np.asarray(optimum["anchor"], dtype=float),
                direction_row=direction_row,
                direction_mode=direction_mode,
                global_bounds=global_bounds,
                seed=seed + index * 7919,
                de_maxiter=18,
                de_popsize=5,
            )
            xy = np.asarray(trial["final_xy"], dtype=float)
            if np.isfinite(xy).all():
                locations.append(xy)
        except Exception:
            continue
    if not locations:
        return float("nan"), np.empty((0, 2), dtype=float)
    array = np.vstack(locations)
    center = np.median(array, axis=0)
    radius = np.linalg.norm(array - center[None, :], axis=1)
    return float(np.quantile(radius, 0.90)), array


def apply_uncertainty_shrinkage(solution: Dict[str, Any], uncertainty_m: float, omni: bool) -> float:
    """Shrink an unstable inverse solution toward its measurement-only anchor.

    No ground-truth coordinate is used.  The gate depends only on bootstrap
    dispersion produced from the selected measurements.
    """
    if not np.isfinite(uncertainty_m):
        solution["uncertainty_shrinkage_factor"] = 1.0
        return 1.0
    # Anchor is deliberately weak in v1.6.  Bootstrap uncertainty can reduce
    # model influence, but it may no longer collapse a well-supported inverse
    # solution almost completely back to a potentially biased anchor.
    scale = 120.0 if omni else 140.0
    raw_gate = 1.0 / (1.0 + (float(uncertainty_m) / scale) ** 2)
    # v1.10 profiled solver already uses the anchor as a weak objective prior.
    # Keep only a very light uncertainty-controlled post-hoc shrinkage so random
    # subsets are not pulled back toward a noisy strong-signal anchor twice.
    ceiling = 0.97 if omni else 0.995
    floor = 0.92 if omni else 0.96
    effective_weight = float(floor + (ceiling - floor) * raw_gate)
    factor = float(effective_weight / max(float(solution["model_weight"]), 1e-9))
    solution["model_weight"] = effective_weight
    solution["final_xy"] = (
        effective_weight * np.asarray(solution["model_xy"], dtype=float)
        + (1.0 - effective_weight) * np.asarray(solution["anchor"], dtype=float)
    )
    solution["uncertainty_shrinkage_factor"] = factor
    return factor

def extract_model_parameters(solution: Dict[str, Any], omni: bool, direction_mode: str) -> Tuple[float, float, float]:
    params = np.asarray(solution["params"], dtype=float)
    if omni:
        exponent = float(params[2])
        gain_scale = 0.0
        intercept = float(params[3])
    elif direction_mode == "fixed" and bool(solution["direction_prior_used"]):
        gain_scale = float(params[2])
        exponent = float(params[3])
        intercept = float(params[4])
    else:
        gain_scale = float(params[3])
        exponent = float(params[4])
        intercept = float(params[5])
    return exponent, gain_scale, intercept


def summarize(results: pd.DataFrame, requested_points: int) -> pd.DataFrame:
    error = results["horizontal_error_m"].to_numpy(float)
    row: Dict[str, Any] = {
        "algorithm": ALGORITHM_NAME,
        "station_count": int(len(results)),
        "requested_points_per_station": int(requested_points),
        "selected_points_min": int(results["selected_point_count"].min()),
        "selected_points_max": int(results["selected_point_count"].max()),
        "mean_error_m": float(np.mean(error)),
        "median_error_m": float(np.median(error)),
        "rmse_m": float(np.sqrt(np.mean(error ** 2))),
        "p75_error_m": float(np.quantile(error, 0.75)),
        "p90_error_m": float(np.quantile(error, 0.90)),
        "p95_error_m": float(np.quantile(error, 0.95)),
        "max_error_m": float(np.max(error)),
        "quality_flagged_count": int(np.sum(results["quality_flag"].astype(str) != "ok")) if "quality_flag" in results else 0,
        "quality_ok_count": int(np.sum(results["quality_flag"].astype(str) == "ok")) if "quality_flag" in results else int(len(results)),
        "multistart_triggered_count": int(results.get("multistart_triggered", pd.Series(False, index=results.index)).astype(bool).sum()),
        "low_geometry_confidence_count": int((pd.to_numeric(results.get("geometry_reliability_score", pd.Series(np.nan, index=results.index)), errors="coerce") < 0.62).sum()),
        "high_model_anchor_disagreement_count": int((pd.to_numeric(results.get("model_anchor_disagreement_m", pd.Series(np.nan, index=results.index)), errors="coerce") > 150.0).sum()),
        "median_geometry_reliability_score": float(pd.to_numeric(results.get("geometry_reliability_score", pd.Series(np.nan, index=results.index)), errors="coerce").median()),
    }
    for threshold in (20, 50, 100, 200):
        count = int(np.sum(error <= threshold))
        row[f"within_{threshold}m_count"] = count
        row[f"within_{threshold}m_percent"] = float(100.0 * count / len(error))
    return pd.DataFrame([row])


FULL_PAGE_WIDTH_IN = 7.48  # 7480 px at 1000 dpi
FIGURE_HEIGHT_IN = 5.61
MAP_DPI = 1000
COMPARISON_DPI = 1000


def _publication_figsize_inches() -> tuple[float, float]:
    return (FULL_PAGE_WIDTH_IN, FIGURE_HEIGHT_IN)


def _style_publication_text(fig) -> None:
    for _ax in fig.axes:
        try:
            _ax.title.set_fontsize(10.0)
            _ax.xaxis.label.set_fontsize(9.0)
            _ax.yaxis.label.set_fontsize(9.0)
            _ax.tick_params(axis='both', which='major', labelsize=8.2, pad=2.0)
            _legend = _ax.get_legend()
            if _legend is not None:
                for _txt in _legend.get_texts():
                    _txt.set_fontsize(8.0)
        except Exception:
            pass


def _save_png(fig, output_path: Path, dpi: int = MAP_DPI, facecolor: str | None = None) -> Path:
    """Save one publication PNG.

    Full-page canvas width is fixed to 7.48 in. At 1000 dpi this gives 7480 px. No PNG duplicate is produced.
    """
    base = Path(output_path).with_suffix('')
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.set_size_inches(*_publication_figsize_inches(), forward=True)
    _style_publication_text(fig)
    fc = facecolor if facecolor is not None else fig.get_facecolor()
    out = base.with_suffix('.png')
    fig.savefig(out, format='png', dpi=int(dpi), bbox_inches=None, pad_inches=0.0, facecolor=fc)
    return out


def _load_final_calibrated_alphas(
    project_root: Path, calibration_root: Path | None, station_id: int, omni: bool
) -> list[float]:
    if omni:
        return []
    root = (
        calibration_root.expanduser().resolve()
        if calibration_root is not None
        else project_root / "outputs" / "parameter_calibration"
    )
    for name in (f"station_{station_id}", f"station_{station_id:02d}"):
        path = root / name / "best_parameters.json"
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        alphas = payload.get("alphas_rad")
        if isinstance(alphas, list) and len(alphas) == 3:
            values = [float(v) for v in alphas]
            if all(np.isfinite(values)):
                return values
    raise FileNotFoundError(
        f"station {station_id}缺少调参后的alphas_rad。请先完成参数调节并保留 "
        f"{root}/station_{station_id}/best_parameters.json。"
    )


def _draw_calibrated_sector_rays(ax, origin_xy: np.ndarray, alphas_rad: list[float], xlim, ylim) -> None:
    """Draw only the three calibrated sector directions as dashed rays.

    The rays start at the final estimated site position and use the final
    calibrated ``alphas_rad`` written by parameter calibration.  They are
    visualization aids only and do not alter the localization
    solution.
    """
    if len(alphas_rad) != 3:
        return
    ray_length = 0.22 * math.hypot(float(xlim[1] - xlim[0]), float(ylim[1] - ylim[0]))
    cmap = plt.get_cmap("tab10")
    for idx, alpha in enumerate(alphas_rad):
        direction = np.asarray([math.cos(alpha), math.sin(alpha)], dtype=float)
        endpoint = origin_xy + ray_length * direction
        ax.plot(
            [origin_xy[0], endpoint[0]], [origin_xy[1], endpoint[1]],
            linestyle="--", linewidth=1.35, color=cmap(idx), alpha=0.98, zorder=5,
            label="Calibrated sector directions" if idx == 0 else None,
        )


def _safe_to_csv(frame: pd.DataFrame, path: Path, *, index: bool = False, encoding: str = "utf-8-sig") -> Path:
    """Write a CSV without aborting the full localization run if the target is locked.

    On Windows, opening a result CSV in Excel can lock the file and make a normal
    pandas ``to_csv`` raise ``PermissionError``.  This helper first writes to a
    temporary file in the same directory and then atomically replaces the target.
    If Windows still refuses the replacement because the old file is open, the new
    result is preserved under a timestamped ``*_rerun_YYYYMMDD_HHMMSS.csv`` name
    and the pipeline continues.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    tmp = path.with_name(f".{path.stem}.{os.getpid()}.{stamp}.tmp{path.suffix}")
    try:
        frame.to_csv(tmp, index=index, encoding=encoding)
        try:
            os.replace(tmp, path)
            return path
        except PermissionError:
            fallback = path.with_name(f"{path.stem}_rerun_{stamp}{path.suffix}")
            os.replace(tmp, fallback)
            print(
                f"[WARN] 输出文件正被其他程序占用，无法覆盖：{path}\n"
                f"       本次结果已保存为：{fallback}\n"
                "       通常是Excel/记事本正在打开旧CSV；关闭后下次运行会恢复标准文件名。"
            )
            return fallback
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def _expand_limits(values: np.ndarray, pad_ratio: float = 0.08, min_pad: float = 40.0) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    lo = float(np.min(values))
    hi = float(np.max(values))
    span = max(hi - lo, 1.0)
    pad = max(min_pad, span * pad_ratio)
    return lo - pad, hi + pad


def _create_fixed_equal_axes(
    fig,
    *,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    left: float = 0.100,
    bottom: float = 0.245,
    top: float = 0.835,
    right_margin: float = 0.110,
    cbar_pad: float = 0.012,
    cbar_width: float = 0.022,
):
    x0, x1 = map(float, xlim)
    y0, y1 = map(float, ylim)
    data_ratio = abs((y1 - y0) / max(x1 - x0, 1e-9))
    fig_w, fig_h = fig.get_size_inches()
    avail_w = 1.0 - left - right_margin - cbar_pad - cbar_width
    avail_h = top - bottom
    normalized_h_if_full_w = avail_w * (fig_w / fig_h) * data_ratio
    if normalized_h_if_full_w <= avail_h:
        ax_w = avail_w
        ax_h = normalized_h_if_full_w
        ax_left = left
        ax_bottom = bottom + 0.5 * (avail_h - ax_h)
    else:
        ax_h = avail_h
        ax_w = avail_h / ((fig_w / fig_h) * data_ratio)
        ax_left = left + 0.5 * (avail_w - ax_w)
        ax_bottom = bottom
    cax_left = ax_left + ax_w + cbar_pad
    ax = fig.add_axes([ax_left, ax_bottom, ax_w, ax_h])
    cax = fig.add_axes([cax_left, ax_bottom, cbar_width, ax_h])
    return ax, cax


def _add_fixed_colorbar(fig, cax, mappable, label: str):
    cbar = fig.colorbar(mappable, cax=cax)
    cbar.set_label(label)
    return cbar


def plot_station_result(
    station_id: int,
    label: str,
    selected: pd.DataFrame,
    solution: Dict[str, Any],
    true_xy: np.ndarray,
    bootstrap_xy: np.ndarray,
    error_m: float,
    output: Path,
    calibrated_alphas_rad: list[float] | None = None,
    signal_strength_points: pd.DataFrame | None = None,
) -> None:
    """Plot one station-localization result for the selected point count.

    The figure intentionally shows only information needed for interpretation:
    measured signal strength, the selected receiver points, calibrated
    sector directions, final estimate, and ground truth.  Intermediate solver
    states (bootstrap samples, strong-signal anchor, and physics inverse
    solution) remain available numerically but are not drawn.
    """
    final = np.asarray(solution["final_xy"], float)
    x_parts = [
        selected["x_m"].to_numpy(float),
        np.asarray([true_xy[0], final[0]], float),
    ]
    y_parts = [
        selected["y_m"].to_numpy(float),
        np.asarray([true_xy[1], final[1]], float),
    ]
    xlim = _expand_limits(np.concatenate(x_parts))
    ylim = _expand_limits(np.concatenate(y_parts))

    selected_strength = selected[["rsrp_s1", "rsrp_s2", "rsrp_s3"]].max(axis=1)
    signal_values = (
        signal_strength_points["strongest_rsrp_dbm"].to_numpy(float)
        if signal_strength_points is not None and not signal_strength_points.empty
        else selected_strength.to_numpy(float)
    )
    finite_signal = signal_values[np.isfinite(signal_values)]
    if finite_signal.size:
        vmin = max(-120.0, float(np.nanmin(finite_signal)))
        vmax = min(-40.0, float(np.nanmax(finite_signal)))
        if vmax - vmin < 1.0:
            vmin, vmax = vmax - 1.0, vmax
    else:
        vmin, vmax = -120.0, -40.0

    fig = plt.figure(figsize=_publication_figsize_inches(), dpi=MAP_DPI)
    ax, cax = _create_fixed_equal_axes(fig, xlim=xlim, ylim=ylim)

    signal_sc = _draw_signal_strength_background(
        ax, signal_strength_points, xlim, ylim, vmin=vmin, vmax=vmax
    )
    if signal_sc is None:
        signal_sc = ax.scatter(
            selected["x_m"], selected["y_m"], c=selected_strength,
            cmap="viridis", vmin=vmin, vmax=vmax,
            s=18, alpha=0.72, linewidths=0, zorder=1,
            label="_nolegend_",
        )

    # Selected points are emphasized as larger hollow circles so the
    # signal-strength colors beneath them remain visible.
    ax.scatter(
        selected["x_m"], selected["y_m"],
        s=92, facecolors="none", edgecolors="black", linewidths=1.25,
        label=f"{len(selected)} selected points", zorder=7,
    )
    _draw_calibrated_sector_rays(ax, final, calibrated_alphas_rad or [], xlim, ylim)
    ax.scatter(
        final[0], final[1], marker="*", s=210,
        c="red", edgecolors="black", linewidths=0.8,
        label="Final estimate", zorder=9,
    )
    ax.scatter(
        true_xy[0], true_xy[1], marker="x", s=140,
        c="black", linewidths=2.0, label="Ground truth", zorder=9,
    )

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("Blender X [m]")
    ax.set_ylabel("Blender Y [m]")
    ax.set_title(
        f"Station {station_id} localization result (error = {error_m:.2f} m)",
        fontsize=9.8, pad=8.0,
    )
    handles, labels = ax.get_legend_handles_labels()
    label_map = {
        "Measured signal strength": "Measurements",
        f"{len(selected)} selected points": "Selected points",
        "Calibrated sector directions": "Sector directions",
        "Final estimate": "Estimate",
        "Ground truth": "Ground truth",
    }
    labels = [label_map.get(lbl, lbl) for lbl in labels]
    ax.legend(
        handles, labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=3,
        fontsize=7.0,
        framealpha=0.96,
        markerscale=0.78,
        handlelength=1.35,
        handletextpad=0.42,
        columnspacing=0.80,
        borderpad=0.30,
        labelspacing=0.30,
    )
    _add_fixed_colorbar(fig, cax, signal_sc, "Measured strongest RSRP [dBm]")
    output.parent.mkdir(parents=True, exist_ok=True)
    _save_png(fig, output, dpi=COMPARISON_DPI)
    plt.close(fig)

def plot_station_algorithm_display(
    station_id: int,
    label: str,
    selected: pd.DataFrame,
    solution: Dict[str, Any],
    true_xy: np.ndarray,
    bootstrap_xy: np.ndarray,
    error_m: float,
    output: Path,
    calibrated_alphas_rad: list[float] | None = None,
    signal_strength_points: pd.DataFrame | None = None,
) -> None:
    """Clean single-station algorithm visualization without text annotations.

    The figure keeps only graphical elements needed to understand the localization
    workflow. Numeric diagnostics remain in CSV/JSON outputs instead of being
    written inside the image.
    """
    final = np.asarray(solution["final_xy"], float)
    model = np.asarray(solution.get("model_xy", final), float)
    anchor = np.asarray(solution.get("anchor_xy", solution.get("anchor", final)), float)
    if not np.isfinite(anchor).all():
        anchor = final.copy()

    x_parts = [
        selected["x_m"].to_numpy(float),
        np.asarray([true_xy[0], final[0], model[0], anchor[0]], float),
    ]
    y_parts = [
        selected["y_m"].to_numpy(float),
        np.asarray([true_xy[1], final[1], model[1], anchor[1]], float),
    ]
    centers = multistart_search_centers(selected, anchor, bool(solution.get("omni", False)))
    if centers:
        x_parts.append(np.asarray([center[1][0] for center in centers], float))
        y_parts.append(np.asarray([center[1][1] for center in centers], float))
    if bootstrap_xy is not None and len(bootstrap_xy):
        x_parts.append(np.asarray(bootstrap_xy[:, 0], float))
        y_parts.append(np.asarray(bootstrap_xy[:, 1], float))

    xlim = _expand_limits(np.concatenate(x_parts), pad_ratio=0.09, min_pad=45.0)
    ylim = _expand_limits(np.concatenate(y_parts), pad_ratio=0.09, min_pad=45.0)

    selected_strength = selected[["rsrp_s1", "rsrp_s2", "rsrp_s3"]].max(axis=1)
    signal_values = (
        signal_strength_points["strongest_rsrp_dbm"].to_numpy(float)
        if signal_strength_points is not None and not signal_strength_points.empty
        else selected_strength.to_numpy(float)
    )
    finite_signal = signal_values[np.isfinite(signal_values)]
    if finite_signal.size:
        vmin = max(-120.0, float(np.nanmin(finite_signal)))
        vmax = min(-40.0, float(np.nanmax(finite_signal)))
        if vmax - vmin < 1.0:
            vmin, vmax = vmax - 1.0, vmax
    else:
        vmin, vmax = -120.0, -40.0

    # More bottom space for a compact two-row legend; no right-side annotation box.
    fig = plt.figure(figsize=(7.9, 6.4), dpi=MAP_DPI)
    ax, cax = _create_fixed_equal_axes(
        fig, xlim=xlim, ylim=ylim,
        left=0.10, bottom=0.20, top=0.86,
        right_margin=0.10, cbar_pad=0.014, cbar_width=0.024,
    )

    signal_sc = _draw_signal_strength_background(
        ax, signal_strength_points, xlim, ylim, vmin=vmin, vmax=vmax
    )
    if signal_sc is None:
        signal_sc = ax.scatter(
            selected["x_m"], selected["y_m"], c=selected_strength,
            cmap="viridis", vmin=vmin, vmax=vmax,
            s=17, alpha=0.72, linewidths=0, zorder=1,
        )

    if bootstrap_xy is not None and len(bootstrap_xy):
        ax.scatter(
            bootstrap_xy[:, 0], bootstrap_xy[:, 1],
            s=16, marker="o", facecolors="none", edgecolors="tab:purple",
            alpha=0.38, linewidths=0.75, label="Bootstrap", zorder=2,
        )

    ax.scatter(
        selected["x_m"], selected["y_m"],
        s=78, facecolors="none", edgecolors="black", linewidths=1.15,
        label="Selected points", zorder=5,
    )

    _draw_calibrated_sector_rays(ax, final, calibrated_alphas_rad or [], xlim, ylim)

    if bool(solution.get("multistart_triggered", False)) and centers:
        for i, (center_label, center) in enumerate(centers):
            center = np.asarray(center, float)
            selected_center = center_label == str(solution.get("multistart_selected_label", ""))
            ax.scatter(
                center[0], center[1], marker="D" if selected_center else "d",
                s=42 if selected_center else 30,
                facecolors="white", edgecolors="tab:orange", linewidths=1.0,
                alpha=0.85, zorder=4,
                label="Search centers" if i == 0 else None,
            )

    ax.scatter(
        anchor[0], anchor[1], marker="P", s=100,
        c="gold", edgecolors="black", linewidths=0.7,
        label="Weak anchor", zorder=7,
    )
    ax.scatter(
        model[0], model[1], marker="^", s=100,
        c="tab:blue", edgecolors="black", linewidths=0.7,
        label="Model solution", zorder=8,
    )
    ax.scatter(
        final[0], final[1], marker="*", s=185,
        c="red", edgecolors="black", linewidths=0.8,
        label="Estimate", zorder=9,
    )
    ax.scatter(
        true_xy[0], true_xy[1], marker="x", s=125,
        c="black", linewidths=1.9,
        label="Ground truth", zorder=10,
    )

    # Keep only light graphical connections; no numerical/text callouts.
    ax.plot(
        [anchor[0], final[0]], [anchor[1], final[1]],
        linestyle="--", linewidth=0.9, color="goldenrod", alpha=0.70, zorder=3,
    )
    ax.plot(
        [model[0], final[0]], [model[1], final[1]],
        linestyle="--", linewidth=0.9, color="tab:blue", alpha=0.70, zorder=3,
    )
    ax.plot(
        [true_xy[0], final[0]], [true_xy[1], final[1]],
        linestyle="-", linewidth=0.9, color="tab:red", alpha=0.55, zorder=3,
    )

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.22)
    ax.set_xlabel("Blender X [m]")
    ax.set_ylabel("Blender Y [m]")
    ax.set_title(
        f"Station {station_id} localization algorithm",
        fontsize=9.8, pad=8.0,
    )

    handles, labels = ax.get_legend_handles_labels()
    # Keep first occurrence of each label and use short labels only.
    unique = []
    seen = set()
    for h, lbl in zip(handles, labels):
        short = "Sector directions" if lbl == "Calibrated sector directions" else lbl
        if short and short not in seen:
            seen.add(short)
            unique.append((h, short))
    if unique:
        ax.legend(
            [u[0] for u in unique], [u[1] for u in unique],
            loc="upper center", bbox_to_anchor=(0.5, -0.145),
            ncol=4, fontsize=6.7, framealpha=0.96,
            markerscale=0.78, handlelength=1.25,
            handletextpad=0.38, columnspacing=0.72,
            borderpad=0.30, labelspacing=0.28,
        )

    _add_fixed_colorbar(fig, cax, signal_sc, "Measured strongest RSRP [dBm]")
    output.parent.mkdir(parents=True, exist_ok=True)
    _save_png(fig, output, dpi=COMPARISON_DPI)
    plt.close(fig)


def plot_localization_dashboard(results: pd.DataFrame, requested_points: int, output: Path) -> None:
    """Clean two-panel summary: spatial result map plus per-station error bars."""
    results = results.sort_values("station_id").reset_index(drop=True)
    x_all = np.r_[results["predicted_x_m"].to_numpy(float), results["true_x_m"].to_numpy(float)]
    y_all = np.r_[results["predicted_y_m"].to_numpy(float), results["true_y_m"].to_numpy(float)]
    xlim = _expand_limits(x_all, pad_ratio=0.06, min_pad=80.0)
    ylim = _expand_limits(y_all, pad_ratio=0.06, min_pad=80.0)
    errors = results["horizontal_error_m"].to_numpy(float)

    fig = plt.figure(figsize=(10.2, 5.7), dpi=MAP_DPI)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1.0], wspace=0.25)

    ax_map = fig.add_subplot(gs[0, 0])
    for _, row in results.iterrows():
        ax_map.plot(
            [row["true_x_m"], row["predicted_x_m"]],
            [row["true_y_m"], row["predicted_y_m"]],
            linewidth=0.75, alpha=0.45, color="0.55", zorder=1,
        )
    sc = ax_map.scatter(
        results["predicted_x_m"], results["predicted_y_m"],
        c=errors, s=55, cmap="viridis", edgecolors="black", linewidths=0.35,
        zorder=3, label="Estimated",
    )
    ax_map.scatter(
        results["true_x_m"], results["true_y_m"],
        marker="x", s=48, linewidths=1.3, color="black",
        zorder=4, label="Actual",
    )
    ax_map.set_title(f"All-station localization ({requested_points} points per station)", fontsize=9.8)
    ax_map.set_xlim(xlim)
    ax_map.set_ylim(ylim)
    ax_map.set_aspect("equal", adjustable="box")
    ax_map.grid(True, alpha=0.22)
    ax_map.set_xlabel("Blender X [m]")
    ax_map.set_ylabel("Blender Y [m]")
    ax_map.legend(loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=2, fontsize=7.0, framealpha=0.96)
    cbar = fig.colorbar(sc, ax=ax_map, fraction=0.046, pad=0.02)
    cbar.set_label("Horizontal error [m]")

    ax_bar = fig.add_subplot(gs[0, 1])
    by_station = results.sort_values("station_id")
    xpos = np.arange(len(by_station))
    ax_bar.bar(xpos, by_station["horizontal_error_m"].to_numpy(float), width=0.78)
    ax_bar.set_xticks(xpos)
    ax_bar.set_xticklabels([str(int(v)) for v in by_station["station_id"]], rotation=90, fontsize=6.4)
    ax_bar.set_xlabel("Station ID")
    ax_bar.set_ylabel("Horizontal error [m]")
    ax_bar.set_title("Per-station localization error", fontsize=9.8)
    ax_bar.grid(True, axis="y", alpha=0.24)

    fig.suptitle("DP-PGRSL localization results", fontsize=10.6, y=0.98)
    output.parent.mkdir(parents=True, exist_ok=True)
    _save_png(fig, output, dpi=COMPARISON_DPI)
    plt.close(fig)


def plot_overview(results: pd.DataFrame, requested_points: int, output: Path) -> None:
    x_all = np.r_[results["predicted_x_m"].to_numpy(float), results["true_x_m"].to_numpy(float)]
    y_all = np.r_[results["predicted_y_m"].to_numpy(float), results["true_y_m"].to_numpy(float)]
    xlim = _expand_limits(x_all, pad_ratio=0.06, min_pad=80.0)
    ylim = _expand_limits(y_all, pad_ratio=0.06, min_pad=80.0)

    fig = plt.figure(figsize=_publication_figsize_inches(), dpi=MAP_DPI)
    ax, cax = _create_fixed_equal_axes(fig, xlim=xlim, ylim=ylim)
    for _, row in results.iterrows():
        ax.plot([row["true_x_m"], row["predicted_x_m"]], [row["true_y_m"], row["predicted_y_m"]], linewidth=0.8, alpha=0.55)
    sc = ax.scatter(results["predicted_x_m"], results["predicted_y_m"], c=results["horizontal_error_m"], s=65, edgecolors="black", linewidths=0.4, label="Estimated")
    ax.scatter(results["true_x_m"], results["true_y_m"], marker="x", s=55, linewidths=1.4, label="Actual")
    ax.set_title(f"Localization results ({requested_points} points per station)", fontsize=9.8, pad=8.0)
    ax.set_xlabel("Blender X [m]")
    ax.set_ylabel("Blender Y [m]")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2,
        fontsize=7.0, framealpha=0.96, markerscale=0.8,
        handlelength=1.4, columnspacing=1.0, borderpad=0.35,
    )
    _add_fixed_colorbar(fig, cax, sc, "Horizontal localization error [m]")
    _save_png(fig, output, dpi=COMPARISON_DPI)
    plt.close(fig)


def plot_cdf(results: pd.DataFrame, requested_points: int, output: Path) -> None:
    errors = np.sort(results["horizontal_error_m"].to_numpy(float))
    cdf = np.arange(1, len(errors) + 1) / len(errors)
    fig, ax = plt.subplots(figsize=_publication_figsize_inches(), dpi=MAP_DPI)
    ax.plot(errors, cdf, marker="o", markersize=3)
    ax.set_xlabel("Horizontal localization error [m]")
    ax.set_ylabel("Empirical CDF")
    ax.set_title(f"Localization error CDF: {len(results)} stations, {requested_points} points each", fontsize=9.5, pad=7.0)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_png(fig, output, dpi=COMPARISON_DPI)
    plt.close(fig)

def main() -> int:
    args = parse_args()
    if args.points_per_station < 3:
        raise ValueError("三扇区极稀疏定位建议至少3个空间点。")

    project_root = args.project_root.expanduser().resolve()
    measurement_csv = common.resolve_measurement_csv(project_root, args.measurements)
    direction_csv = resolve_direction_csv(project_root, args.directions)
    direction_table = load_direction_priors(direction_csv)

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else project_root
        / "outputs"
        / f"localization_27stations_{args.points_per_station}points_dppgrsl"
        / f"seed_{args.random_seed}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    per_station_dir = output_dir / "per_station"
    per_station_dir.mkdir(exist_ok=True)

    localization, truth = common.load_and_filter(measurement_csv)
    available = sorted(int(v) for v in localization["station_id"].unique())
    station_ids = common.parse_station_ids(args.station_ids, available)
    truth_index = truth.set_index("station_id")
    global_bounds = (args.x_min, args.x_max, args.y_min, args.y_max)

    print("=" * 94)
    print(ALGORITHM_NAME)
    print(f"输入实测长表：{measurement_csv}")
    print("定位输入：直接使用长表中的原始接收采样位置；不做2.77 m或其他空间聚合。")
    print(f"方向先验：{direction_csv if direction_csv else '未使用'}")
    print(f"方向模式：{args.direction_prior_mode}")
    print(f"站点数：{len(station_ids)}；每站空间点：{args.points_per_station}")
    print(f"选点模式：{args.selection_mode}；trial={args.trial_index}；seed={args.random_seed}")
    print("单站背景图：使用该站全部有效实测位置的最强RSRP绘制信号强度图，仅用于展示。")
    print("真实坐标仅在定位结束后计算误差。")
    print("=" * 94)

    result_rows: List[Dict[str, Any]] = []
    selected_rows: List[pd.DataFrame] = []
    signal_rows: List[pd.DataFrame] = []

    for station_id in station_ids:
        start = time.time()
        station = localization[localization["station_id"] == station_id].copy()
        truth_row = truth_index.loc[station_id]
        label = str(truth_row["station_label"])
        omni = bool(int(truth_row["is_omnidirectional"])) or station_id == 22
        antenna_type = str(truth_row["antenna_type"])
        calibrated_alphas_rad = (
            _load_final_calibrated_alphas(
                project_root, args.calibration_root, station_id, omni
            )
            if not args.skip_figures and not args.skip_per_station_figures
            else []
        )

        points = common.point_table(station)
        station_signal_points = build_signal_strength_points(points)
        signal_out = station_signal_points.copy()
        signal_out.insert(0, "station_id", station_id)
        signal_out.insert(1, "station_label", label)
        signal_rows.append(signal_out)
        if args.selection_mode == "random":
            selected, anchor = select_random_localization_points(
                points,
                args.points_per_station,
                args.random_seed + station_id * 1009,
                omni,
            )
        else:
            selected, anchor = select_balanced_strong_points(
                points,
                args.points_per_station,
                args.random_seed + station_id,
                omni,
                ranking_k=args.selection_max_points,
            )
        observations = common.observations_from_points(selected)
        if omni:
            observations = observations[observations["sector_index"] == 1].copy()

        direction_row = (
            direction_table.loc[station_id]
            if station_id in direction_table.index and not omni
            else None
        )
        solution = optimize_station_robust(
            selected=selected,
            station_id=station_id,
            omni=omni,
            anchor=anchor,
            direction_row=direction_row,
            direction_mode=args.direction_prior_mode,
            global_bounds=global_bounds,
            seed=args.random_seed + station_id * 1009,
            de_maxiter=args.de_maxiter,
            de_popsize=args.de_popsize,
        )
        uncertainty, bootstrap_xy = bootstrap_uncertainty(
            optimum=solution,
            selected=selected,
            station_id=station_id,
            omni=omni,
            direction_row=direction_row,
            direction_mode=args.direction_prior_mode,
            global_bounds=global_bounds,
            count=args.bootstrap,
            seed=args.random_seed + station_id * 1709,
        )
        solution["uncertainty_radius_p90_m"] = float(uncertainty)
        uncertainty_shrink = apply_uncertainty_shrinkage(solution, uncertainty, omni)

        final_xy = np.asarray(solution["final_xy"], dtype=float)
        true_xy = np.asarray([truth_row["true_x_m"], truth_row["true_y_m"]], dtype=float)
        delta = final_xy - true_xy
        error = float(np.linalg.norm(delta))
        exponent, gain_scale, intercept = extract_model_parameters(
            solution, omni, args.direction_prior_mode
        )
        spread_m = common.point_spread(selected)
        disagreement_m = float(solution.get("model_anchor_disagreement_m", np.linalg.norm(np.asarray(solution["model_xy"]) - anchor)))
        boundary_margin_m = float(solution.get("optimizer_boundary_margin_m", np.nan))
        geom = solution.get("geometry_diagnostics") or geometry_diagnostics(np.asarray(solution["model_xy"], dtype=float), selected, omni)
        angular_coverage = float(geom["angular_coverage_deg"])
        max_angular_gap = float(geom["max_angular_gap_deg"])
        inside_hull = bool(geom["inside_measurement_convex_hull"])
        extrapolation_ratio = float(geom["extrapolation_ratio"])
        pci_balance = float(geom["pci_spatial_balance"])
        geometry_score = float(geom["geometry_reliability_score"])
        flags: list[str] = []
        if np.isfinite(uncertainty) and uncertainty > 90.0:
            flags.append("high_uncertainty")
        if spread_m < 60.0:
            flags.append("low_point_spread")
        # v1.8: disagreement is a confidence flag only; never force the solution back to anchor.
        if disagreement_m > 150.0:
            flags.append("high_model_anchor_disagreement")
        if angular_coverage < 120.0:
            flags.append("low_angular_coverage")
        if extrapolation_ratio > 1.8:
            flags.append("high_geometry_extrapolation")
        if (not omni) and pci_balance < 0.70:
            flags.append("low_pci_spatial_balance")
        if geometry_score < 0.62:
            flags.append("low_geometry_confidence")
        if np.isfinite(uncertainty) and uncertainty < 45.0 and geometry_score < 0.65:
            flags.append("stable_but_low_geometry_confidence")
        if np.isfinite(boundary_margin_m) and boundary_margin_m < 10.0:
            flags.append("boundary_solution")
        if np.isfinite(uncertainty) and uncertainty > 80.0:
            flags.append("unstable_solution")
        quality = ";".join(dict.fromkeys(flags)) if flags else "ok"

        row = ResultRow(
            station_id=station_id,
            station_label=label,
            antenna_type=antenna_type,
            selected_point_count=int(len(selected)),
            observation_count=int(len(observations)),
            distinct_pci_count=int(observations["pci"].nunique()),
            predicted_x_m=float(final_xy[0]),
            predicted_y_m=float(final_xy[1]),
            model_x_m=float(solution["model_xy"][0]),
            model_y_m=float(solution["model_xy"][1]),
            anchor_x_m=float(anchor[0]),
            anchor_y_m=float(anchor[1]),
            model_weight=float(solution["model_weight"]),
            true_x_m=float(true_xy[0]),
            true_y_m=float(true_xy[1]),
            east_error_m=float(delta[0]),
            north_error_m=float(delta[1]),
            horizontal_error_m=error,
            pathloss_exponent=exponent,
            alpha_deg=0.0 if omni else float(math.degrees(solution["alpha"])),
            sector_order_sign=int(solution["sign"]),
            antenna_gain_scale=gain_scale,
            reference_rsrp_1m_dbm=intercept,
            objective_value=float(solution["objective"]),
            direction_prior_used=bool(solution["direction_prior_used"]),
            direction_fit_rms_deg=float(solution["direction_fit_rms_deg"]),
            uncertainty_radius_p90_m=float(uncertainty),
            bootstrap_success_count=int(len(bootstrap_xy)),
            point_spread_m=spread_m,
            quality_flag=quality,
            model_anchor_disagreement_m=disagreement_m,
            optimizer_boundary_margin_m=boundary_margin_m,
            uncertainty_shrinkage_factor=float(uncertainty_shrink),
            objective_per_observation=float(solution["objective"]) / max(int(len(observations)), 1),
            angular_coverage_deg=angular_coverage,
            max_angular_gap_deg=max_angular_gap,
            inside_measurement_convex_hull=inside_hull,
            extrapolation_ratio=extrapolation_ratio,
            pci_spatial_balance=pci_balance,
            geometry_reliability_score=geometry_score,
            multistart_triggered=bool(solution.get("multistart_triggered", False)),
            multistart_candidate_count=int(solution.get("multistart_candidate_count", 1)),
            multistart_selected_label=str(solution.get("multistart_selected_label", "strong_anchor")),
            elapsed_s=float(time.time() - start),
            selection_geometry_score=float(selected.get("random_subset_geometry_score", pd.Series([np.nan])).iloc[0]) if "random_subset_geometry_score" in selected.columns else float("nan"),
            solver_mode=str(solution.get("solver_mode", "profiled_pathloss")),
            direction_ray_x_m=float(solution.get("direction_ray_x_m", np.nan)),
            direction_ray_y_m=float(solution.get("direction_ray_y_m", np.nan)),
            direction_ray_perpendicular_rms_m=float(solution.get("direction_ray_perpendicular_rms_m", np.nan)),
        )
        result_rows.append(row.__dict__)

        selected_out = selected.copy()
        selected_out.insert(0, "station_id", station_id)
        selected_out.insert(1, "station_label", label)
        selected_rows.append(selected_out)
        _safe_to_csv(
            selected_out,
            per_station_dir / f"station_{station_id:02d}_selected_{args.points_per_station}_points.csv",
            index=False,
            encoding="utf-8-sig",
        )
        if not args.skip_figures and not args.skip_per_station_figures:
            plot_station_result(
                station_id, label, selected, solution, true_xy, bootstrap_xy, error,
                per_station_dir / f"station_{station_id:02d}_localization.png",
                calibrated_alphas_rad=calibrated_alphas_rad,
                signal_strength_points=station_signal_points,
            )
            plot_station_algorithm_display(
                station_id, label, selected, solution, true_xy, bootstrap_xy, error,
                per_station_dir / f"station_{station_id:02d}_localization_algorithm_display.png",
                calibrated_alphas_rad=calibrated_alphas_rad,
                signal_strength_points=station_signal_points,
            )
        print(
            f"站{station_id:02d} {label}: points={len(selected)}, obs={len(observations)}, "
            f"estimate=({final_xy[0]:.2f},{final_xy[1]:.2f}), error={error:.2f} m"
        )

    results = pd.DataFrame(result_rows).sort_values("station_id").reset_index(drop=True)
    selected_all = pd.concat(selected_rows, ignore_index=True)
    summary = summarize(results, args.points_per_station)

    result_name = f"localization_results_{len(station_ids)}stations_{args.points_per_station}points.csv"
    selected_name = f"selected_{args.points_per_station}_spatial_points_all_stations.csv"
    result_csv_path = _safe_to_csv(results, output_dir / result_name, index=False, encoding="utf-8-sig")
    selected_csv_path = _safe_to_csv(selected_all, output_dir / selected_name, index=False, encoding="utf-8-sig")
    signal_all = pd.concat(signal_rows, ignore_index=True) if signal_rows else pd.DataFrame(
        columns=["station_id", "station_label", "x_m", "y_m", "strongest_rsrp_dbm"]
    )
    signal_csv_path = _safe_to_csv(
        signal_all,
        output_dir / "measurement_signal_strength_points.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary_csv_path = _safe_to_csv(summary, output_dir / "localization_accuracy_summary.csv", index=False, encoding="utf-8-sig")
    truth_csv_path = _safe_to_csv(truth, output_dir / "ground_truth_used_only_for_evaluation.csv", index=False, encoding="utf-8-sig")

    if not args.skip_figures:
        plot_overview(
            results, args.points_per_station,
            output_dir / "all_27stations_actual_vs_estimated.png",
        )
        plot_localization_dashboard(
            results, args.points_per_station,
            output_dir / "all_27stations_localization_dashboard.png",
        )
        plot_cdf(
            results, args.points_per_station,
            output_dir / "localization_error_cdf.png",
        )

    metadata = {
        "algorithm": ALGORITHM_NAME,
        "measurement_csv": str(measurement_csv),
        "direction_csv": str(direction_csv) if direction_csv else None,
        "direction_prior_mode": args.direction_prior_mode,
        "points_per_station": int(args.points_per_station),
        "selection_mode": str(args.selection_mode),
        "trial_index": int(args.trial_index),
        "selection_max_points": int(args.selection_max_points) if args.selection_max_points is not None else int(args.points_per_station),
        "nested_selection_prefix": bool(args.selection_max_points is not None and args.selection_mode != "random"),
        "points_definition": (
            f"{args.points_per_station} unique spatial receiver locations per physical station; "
            "all available same-station PCI observations at each selected location are used"
        ),
        "selection_scope": (
            "uniform random sampling without replacement from localization-eligible measurement locations"
            if args.selection_mode == "random"
            else "offline subset selection uses the full transformed measurement pool to choose the requested spatial locations"
        ),
        "spatial_aggregation": "none; raw receiver samples from the long table are used directly",
        "random_seed": int(args.random_seed),
        "dynamic_output_directory": str(output_dir),
        "written_csv_files": {
            "localization_results": str(result_csv_path),
            "selected_points": str(selected_csv_path),
            "measurement_signal_strength_points": str(signal_csv_path),
            "accuracy_summary": str(summary_csv_path),
            "ground_truth_evaluation_copy": str(truth_csv_path),
        },
        "anti_leakage": {
            "true_coordinates_removed_before_selection_and_optimization": True,
            "true_coordinates_used_only_for_final_error": True,
            "direction_prior_is_external_to_the_selected_sparse_points": bool(
                args.direction_prior_mode != "off" and direction_csv is not None
            ),
        },
        "station_22": "omnidirectional single-PCI special case; v1.8 uses inverse-model-dominant estimation with reliability diagnostics and uncertainty-controlled weak-anchor shrinkage",
        "limitations": [
            f"{args.points_per_station} locations remain a sparse inverse problem; point geometry still affects identifiability.",
            "The fixed/soft direction prior was estimated outside the selected sparse subset.",
            "Offline point selection sees the complete candidate measurement pool.",
            "Results from direction-prior mode and strict blind mode must be reported separately.",
        ],
    }
    (output_dir / "experiment_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n总体结果：")
    print(summary.to_string(index=False))
    print(f"\n完成。结果目录：{output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\n运行失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        raise
