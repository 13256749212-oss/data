#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
27个物理基站：每站5个坐标转换后实测点的鲁棒多扇区定位
=============================================================

默认输入
--------
D:/AppData/work/cell_pci_rsrp_long_27stations.csv

“5个点”的严格定义
------------------
每个物理基站选择5个唯一空间接收位置，而不是每个PCI选择5个点。
对于三扇区站，每个空间点会使用该位置所有可用的同站PCI-RSRP，
因此最多得到 5×3=15 条扇区观测；22号全向站得到5条观测。

算法
----
Physics-Guided Robust Multi-Sector Bayesian Inverse Localization
(PG-RMSBIL，物理引导鲁棒多扇区贝叶斯逆定位)：

1. 直接使用原始接收采样位置，不做2.77 m或其他空间聚合；
2. 不使用真实基站坐标，按多PCI完整度、空间最大最小距离和RSS多样性选择5个点；
3. 三扇区模型采用120°共站约束和3GPP风格水平方向图；
4. 联合估计未知基站(x,y)、路径损耗指数、主方向、扇区排列和方向图缩放；
5. 发射功率截距及扇区偏差在每次候选位置下用带岭约束线性解消元；
6. 使用Student-t型鲁棒似然抑制NLoS、反射热点和异常RSRP；
7. 差分进化全局搜索后局部精修，并通过参数扰动bootstrap给出不确定性；
8. 最后才读取真实基站坐标计算定位误差，真实坐标不进入点选择与优化。

说明
----
这是针对当前数据和极少样本条件设计的可落地先进混合方法，不宣称是所有
数据集上的普适最优算法。2025年的RadioDiff-Loc使用条件扩散模型和环境布局，
但需要专门训练数据/预训练模型；本脚本不伪造该训练条件，而采用可直接复现的
物理约束鲁棒逆问题方法。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize


ALGORITHM_NAME = (
    "Physics-Guided Robust Multi-Sector Bayesian Inverse Localization (PG-RMSBIL)"
)
DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POINTS_PER_STATION = 5
DEFAULT_SEED = 20260805

# 与当前4000×3000 m联合地图一致；仅作为无真值的搜索域约束。
DEFAULT_BOUNDS = (-1732.5, 2267.5, -1430.47, 1569.53)

TARGET_BAND = 41
TARGET_CENTER_ARFCN = 513000
TARGET_BANDWIDTH_MHZ = 100.0
MIN_RSRP_DBM = -140.0
MAX_RSRP_DBM = -40.0

# 物理模型参数
TX_RX_VERTICAL_SEPARATION_M = 25.0
PATHLOSS_EXPONENT_PRIOR = 2.8
PATHLOSS_EXPONENT_PRIOR_STD = 0.8
SECTOR_BIAS_PRIOR_STD_DB = 6.0
HORIZONTAL_3DB_BEAMWIDTH_DEG = 65.0
HORIZONTAL_MAX_ATTENUATION_DB = 25.0
ROBUST_SCALE_DB = 6.0
ROBUST_STUDENT_DOF = 3.0


@dataclass
class StationResult:
    station_id: int
    station_label: str
    antenna_type: str
    selected_point_count: int
    observation_count: int
    distinct_pci_count: int
    predicted_x_m: float
    predicted_y_m: float
    true_x_m: float
    true_y_m: float
    east_error_m: float
    north_error_m: float
    horizontal_error_m: float
    pathloss_exponent: float
    alpha_deg: float
    sector_order_sign: int
    antenna_gain_scale: float
    fitted_intercept_dbm: float
    selected_fit_rmse_db: float
    selected_fit_mae_db: float
    objective_value: float
    uncertainty_radius_p90_m: float
    bootstrap_success_count: int
    point_spread_m: float
    quality_flag: str
    elapsed_s: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="27个基站、每站5个坐标转换实测点的PG-RMSBIL定位"
    )
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--measurements", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--points-per-station", type=int, default=DEFAULT_POINTS_PER_STATION)
    parser.add_argument("--random-seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--bootstrap", type=int, default=20)
    parser.add_argument("--de-maxiter", type=int, default=120)
    parser.add_argument("--de-popsize", type=int, default=12)
    parser.add_argument("--x-min", type=float, default=DEFAULT_BOUNDS[0])
    parser.add_argument("--x-max", type=float, default=DEFAULT_BOUNDS[1])
    parser.add_argument("--y-min", type=float, default=DEFAULT_BOUNDS[2])
    parser.add_argument("--y-max", type=float, default=DEFAULT_BOUNDS[3])
    parser.add_argument(
        "--station-ids",
        default="all",
        help="all或逗号分隔站号，例如2,3,22",
    )
    parser.add_argument("--skip-figures", action="store_true")
    return parser.parse_args()


def first_existing(paths: Iterable[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists() and path.is_file():
            return path.resolve()
    return None


def resolve_measurement_csv(project_root: Path, explicit: Optional[Path]) -> Path:
    if explicit is not None:
        p = explicit.expanduser()
        if not p.exists():
            raise FileNotFoundError(f"找不到实测CSV：{p}")
        return p.resolve()
    names = [
        "cell_pci_rsrp_long_27stations.csv",
        "cell_pci_rsrp_long_27stations(1).csv",
    ]
    candidates: List[Path] = []
    for name in names:
        candidates.extend([
            project_root / "data" / "processed" / name,
            project_root / "data" / name,
            project_root / "outputs" / "01_extracted" / name,
            project_root / "outputs" / "02_processed" / name,
            Path(__file__).resolve().parent / "data" / name,
        ])
    found = first_existing(candidates)
    if found is None:
        raise FileNotFoundError(
            "未找到cell_pci_rsrp_long_27stations.csv，请用--measurements指定。"
        )
    return found


def canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "x_m": ["blender_x", "x", "receiver_x_m"],
        "y_m": ["blender_y", "y", "receiver_y_m"],
        "z_m": ["receiver_z_m", "z", "receiver_z"],
        "rsrp_dbm": ["measured_rsrp_dbm", "rsrp_dbm", "NR5G SS RSRP"],
    }
    out = df.copy()
    for target, choices in aliases.items():
        if target in out.columns:
            continue
        source = next((c for c in choices if c in out.columns), None)
        if source is None:
            raise KeyError(f"实测CSV缺少{target}对应字段，候选={choices}")
        out[target] = out[source]
    return out


def load_and_filter(path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    required_hint = [
        "rx_point_id", "measurement_id", "source_file", "source_row", "time",
        "station_id", "pci", "sector_index", "antenna_type",
        "is_omnidirectional", "blender_x", "blender_y", "receiver_z_m",
        "measured_rsrp_dbm", "nr5g_band", "nr5g_center_arfcn_dl",
        "nr5g_bandwidth_dl_mhz", "dem_hit", "tx_x_initial_m",
        "tx_y_initial_m", "station_label",
    ]
    header = pd.read_csv(path, nrows=0).columns.tolist()
    usecols = [c for c in required_hint if c in header]
    frame = pd.read_csv(path, usecols=usecols, low_memory=False)
    frame = canonicalize_columns(frame)

    mandatory = {
        "station_id", "pci", "sector_index", "antenna_type",
        "is_omnidirectional", "x_m", "y_m", "z_m", "rsrp_dbm",
        "tx_x_initial_m", "tx_y_initial_m",
    }
    missing = mandatory - set(frame.columns)
    if missing:
        raise KeyError(f"实测CSV缺少必要字段：{sorted(missing)}")

    for c in [
        "station_id", "pci", "sector_index", "is_omnidirectional",
        "x_m", "y_m", "z_m", "rsrp_dbm", "tx_x_initial_m",
        "tx_y_initial_m",
    ]:
        frame[c] = pd.to_numeric(frame[c], errors="coerce")

    valid = frame[
        frame[["station_id", "pci", "sector_index", "x_m", "y_m", "rsrp_dbm"]]
        .notna().all(axis=1)
    ].copy()
    if "nr5g_band" in valid.columns:
        valid = valid[pd.to_numeric(valid["nr5g_band"], errors="coerce") == TARGET_BAND]
    if "nr5g_center_arfcn_dl" in valid.columns:
        valid = valid[
            pd.to_numeric(valid["nr5g_center_arfcn_dl"], errors="coerce")
            == TARGET_CENTER_ARFCN
        ]
    if "nr5g_bandwidth_dl_mhz" in valid.columns:
        bw = pd.to_numeric(valid["nr5g_bandwidth_dl_mhz"], errors="coerce")
        valid = valid[np.isclose(bw, TARGET_BANDWIDTH_MHZ, atol=1e-6)]
    if "dem_hit" in valid.columns:
        valid = valid[pd.to_numeric(valid["dem_hit"], errors="coerce") == 1]
    valid = valid[valid["rsrp_dbm"].between(MIN_RSRP_DBM, MAX_RSRP_DBM)]

    valid["station_id"] = valid["station_id"].astype(int)
    valid["pci"] = valid["pci"].astype(int)
    valid["sector_index"] = valid["sector_index"].astype(int)
    valid["is_omnidirectional"] = valid["is_omnidirectional"].astype(int)
    if "station_label" not in valid.columns:
        valid["station_label"] = valid["station_id"].map(lambda v: f"station-{v}")

    # 真值表单独保存。定位输入随后删除所有tx真值字段。
    truth = (
        valid.groupby("station_id", as_index=False)
        .agg(
            station_label=("station_label", "first"),
            true_x_m=("tx_x_initial_m", "first"),
            true_y_m=("tx_y_initial_m", "first"),
            antenna_type=("antenna_type", "first"),
            is_omnidirectional=("is_omnidirectional", "max"),
        )
        .sort_values("station_id")
    )
    localization = valid.drop(
        columns=[c for c in valid.columns if c.startswith("tx_")], errors="ignore"
    )
    assert not any(c.startswith("tx_") for c in localization.columns)
    return localization, truth


def point_table(station: pd.DataFrame) -> pd.DataFrame:
    """Build receiver-location rows without spatial aggregation.

    ``rx_point_id`` is the preferred identity because all PCI observations recorded
    at the same receiver sample share that identifier.  For older long tables that
    do not contain it, exact x/y/z coordinates are used as the identity.  No
    2.77-m grid, rounding, or neighborhood aggregation is performed here.
    """
    work = station.copy()
    if "rx_point_id" in work.columns and work["rx_point_id"].notna().any():
        work["receiver_point_key"] = work["rx_point_id"].astype(str)
    else:
        work["receiver_point_key"] = (
            work["x_m"].map(lambda v: f"{float(v):.9f}") + "|"
            + work["y_m"].map(lambda v: f"{float(v):.9f}") + "|"
            + work["z_m"].map(lambda v: f"{float(v):.6f}")
        )

    index_cols = ["receiver_point_key"]
    xyz = work.groupby(index_cols)[["x_m", "y_m", "z_m"]].first()
    pwr = work.pivot_table(
        index=index_cols, columns="sector_index", values="rsrp_dbm", aggfunc="median"
    )
    pci = work.pivot_table(
        index=index_cols, columns="sector_index", values="pci", aggfunc="first"
    )
    for sector in (1, 2, 3):
        if sector not in pwr.columns:
            pwr[sector] = np.nan
        if sector not in pci.columns:
            pci[sector] = np.nan
    pwr = pwr[[1, 2, 3]].rename(columns={s: f"rsrp_s{s}" for s in (1, 2, 3)})
    pci = pci[[1, 2, 3]].rename(columns={s: f"pci_s{s}" for s in (1, 2, 3)})
    return xyz.join(pwr).join(pci).reset_index()


def select_information_optimal_points(
    points: pd.DataFrame,
    k: int,
    seed: int,
) -> pd.DataFrame:
    """不使用基站真值的确定性信息最优点选择。"""
    if len(points) < k:
        raise ValueError(f"唯一空间点仅{len(points)}个，少于要求{k}个")
    rss_cols = ["rsrp_s1", "rsrp_s2", "rsrp_s3"]
    rss = points[rss_cols].to_numpy(float)
    mask = np.isfinite(rss)
    coverage = mask.sum(axis=1)
    max_coverage = int(coverage.max())
    pool = np.flatnonzero(coverage == max_coverage)
    if len(pool) < k:
        pool = np.argsort(-coverage)[: max(k, len(pool))]

    safe = np.where(mask, rss, np.nan)
    strongest = np.nanmax(safe, axis=1)
    mean_rss = np.nanmean(safe, axis=1)
    rss_span = np.nanmax(safe, axis=1) - np.nanmin(safe, axis=1)
    xy = points[["x_m", "y_m"]].to_numpy(float)

    # 第一位置优先：PCI完整、信号较强但不过度只取单一峰值。
    first_score = (
        0.55 * normalize(strongest[pool])
        + 0.25 * normalize(mean_rss[pool])
        + 0.20 * normalize(rss_span[pool])
    )
    selected = [int(pool[int(np.nanargmax(first_score))])]

    while len(selected) < k:
        remaining = np.asarray([i for i in pool if int(i) not in selected], dtype=int)
        if len(remaining) == 0:
            remaining = np.asarray(
                [i for i in range(len(points)) if i not in selected], dtype=int
            )
        dist = np.sqrt(
            ((xy[remaining, None, :] - xy[np.asarray(selected)][None, :, :]) ** 2)
            .sum(axis=2)
        ).min(axis=1)
        score = (
            0.55 * normalize(dist)
            + 0.25 * (coverage[remaining] / max(max_coverage, 1))
            + 0.12 * normalize(strongest[remaining])
            + 0.08 * normalize(rss_span[remaining])
        )
        selected.append(int(remaining[int(np.nanargmax(score))]))

    out = points.iloc[selected].copy().reset_index(drop=True)
    out["selection_rank"] = np.arange(1, len(out) + 1)
    out["observed_sector_count"] = out[rss_cols].notna().sum(axis=1)
    return out


def normalize(values: np.ndarray) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    finite = np.isfinite(v)
    out = np.zeros_like(v, dtype=float)
    if not np.any(finite):
        return out
    lo = float(np.nanmin(v[finite]))
    hi = float(np.nanmax(v[finite]))
    if hi - lo < 1e-12:
        out[finite] = 1.0
    else:
        out[finite] = (v[finite] - lo) / (hi - lo)
    return out


def observations_from_points(selected: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for _, row in selected.iterrows():
        for sector in (1, 2, 3):
            value = row.get(f"rsrp_s{sector}")
            if pd.notna(value):
                rows.append(
                    {
                        "selection_rank": int(row["selection_rank"]),
                        "x_m": float(row["x_m"]),
                        "y_m": float(row["y_m"]),
                        "z_m": float(row["z_m"]),
                        "sector_index": int(sector),
                        "pci": int(row[f"pci_s{sector}"]),
                        "rsrp_dbm": float(value),
                    }
                )
    return pd.DataFrame(rows)


def wrap_angle(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def horizontal_gain_db(phi: np.ndarray, gain_scale: float) -> np.ndarray:
    bw = math.radians(HORIZONTAL_3DB_BEAMWIDTH_DEG)
    attenuation = np.minimum(
        12.0 * (wrap_angle(phi) / bw) ** 2,
        HORIZONTAL_MAX_ATTENUATION_DB,
    )
    return -float(gain_scale) * attenuation


def solve_linear_offsets(
    observed: np.ndarray,
    base_prediction: np.ndarray,
    sector_index: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    sectors = sorted(int(v) for v in np.unique(sector_index))
    design = np.ones((len(observed), 1 + max(0, len(sectors) - 1)), dtype=float)
    for column, sector in enumerate(sectors[1:], start=1):
        design[:, column] = sector_index == sector
    ridge = np.diag([1e-8] + [0.18] * (design.shape[1] - 1))
    beta = np.linalg.solve(
        design.T @ design + ridge,
        design.T @ (observed - base_prediction),
    )
    return beta, design


def model_prediction(
    params: np.ndarray,
    order_sign: int,
    obs: pd.DataFrame,
    omni: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = obs["x_m"].to_numpy(float)
    y = obs["y_m"].to_numpy(float)
    sector = obs["sector_index"].to_numpy(int)
    measured = obs["rsrp_dbm"].to_numpy(float)

    tx_x, tx_y, exponent = [float(v) for v in params[:3]]
    distance = np.sqrt(
        (x - tx_x) ** 2 + (y - tx_y) ** 2 + TX_RX_VERTICAL_SEPARATION_M ** 2
    )
    base = -10.0 * exponent * np.log10(np.maximum(distance, 1.0))

    if omni:
        alpha = 0.0
        gain_scale = 0.0
    else:
        alpha = float(params[3])
        gain_scale = float(params[4])
        offsets = np.where(
            sector == 1,
            0.0,
            np.where(
                sector == 2,
                order_sign * 2.0 * np.pi / 3.0,
                -order_sign * 2.0 * np.pi / 3.0,
            ),
        )
        bearing = np.arctan2(y - tx_y, x - tx_x)
        base = base + horizontal_gain_db(bearing - (alpha + offsets), gain_scale)

    beta, design = solve_linear_offsets(measured, base, sector)
    predicted = base + design @ beta
    return predicted, beta, distance


def robust_objective(
    params: np.ndarray,
    order_sign: int,
    obs: pd.DataFrame,
    omni: bool,
    bounds_xy: Sequence[float],
) -> float:
    predicted, beta, distance = model_prediction(params, order_sign, obs, omni)
    measured = obs["rsrp_dbm"].to_numpy(float)
    residual = measured - predicted
    nu = ROBUST_STUDENT_DOF
    scale = ROBUST_SCALE_DB
    loss = float(np.sum(np.log1p((residual / scale) ** 2 / nu)))

    exponent = float(params[2])
    loss += 0.35 * ((exponent - PATHLOSS_EXPONENT_PRIOR) / PATHLOSS_EXPONENT_PRIOR_STD) ** 2
    if len(beta) > 1:
        loss += 0.05 * float(np.sum((beta[1:] / SECTOR_BIAS_PRIOR_STD_DB) ** 2))
    if not omni:
        gain_scale = float(params[4])
        loss += 0.12 * ((gain_scale - 0.75) / 0.35) ** 2

    # 极远解的弱正则，只依赖接收点而不依赖真值。
    nearest = float(np.min(distance))
    if nearest > 1800.0:
        loss += ((nearest - 1800.0) / 500.0) ** 2
    x_min, x_max, y_min, y_max = [float(v) for v in bounds_xy]
    tx_x, tx_y = float(params[0]), float(params[1])
    if not (x_min <= tx_x <= x_max and y_min <= tx_y <= y_max):
        return 1e12
    return loss


def parameter_bounds(bounds_xy: Sequence[float], omni: bool) -> List[Tuple[float, float]]:
    x_min, x_max, y_min, y_max = [float(v) for v in bounds_xy]
    result: List[Tuple[float, float]] = [
        (x_min, x_max),
        (y_min, y_max),
        (1.3, 4.8),
    ]
    if not omni:
        result.extend([(-math.pi, math.pi), (0.25, 1.20)])
    return result


def localize_station(
    obs: pd.DataFrame,
    omni: bool,
    bounds_xy: Sequence[float],
    seed: int,
    de_maxiter: int,
    de_popsize: int,
) -> Tuple[np.ndarray, int, float]:
    bounds = parameter_bounds(bounds_xy, omni)
    signs = [1] if omni else [1, -1]
    best_params: Optional[np.ndarray] = None
    best_sign = 1
    best_value = float("inf")
    for offset, sign in enumerate(signs):
        result = differential_evolution(
            lambda p: robust_objective(p, sign, obs, omni, bounds_xy),
            bounds=bounds,
            seed=int(seed + offset * 100003),
            popsize=int(de_popsize),
            maxiter=int(de_maxiter),
            tol=1e-7,
            mutation=(0.5, 1.0),
            recombination=0.75,
            polish=True,
            updating="immediate",
            workers=1,
        )
        if float(result.fun) < best_value:
            best_value = float(result.fun)
            best_params = np.asarray(result.x, dtype=float)
            best_sign = int(sign)
    if best_params is None:
        raise RuntimeError("全局优化未返回结果")
    return best_params, best_sign, best_value


def bootstrap_uncertainty(
    optimum: np.ndarray,
    sign: int,
    obs: pd.DataFrame,
    omni: bool,
    bounds_xy: Sequence[float],
    count: int,
    seed: int,
) -> Tuple[float, np.ndarray]:
    if count <= 0:
        return float("nan"), np.empty((0, 2), dtype=float)
    rng = np.random.default_rng(seed)
    bounds = parameter_bounds(bounds_xy, omni)
    locations: List[np.ndarray] = []
    base = obs.copy()
    for _ in range(count):
        noisy = base.copy()
        noisy["rsrp_dbm"] = (
            noisy["rsrp_dbm"].to_numpy(float)
            + rng.standard_t(df=4, size=len(noisy)) * 2.0
        )
        start = optimum.copy()
        start[0] += rng.normal(0.0, 25.0)
        start[1] += rng.normal(0.0, 25.0)
        for i, (lo, hi) in enumerate(bounds):
            start[i] = np.clip(start[i], lo, hi)
        result = minimize(
            lambda p: robust_objective(p, sign, noisy, omni, bounds_xy),
            x0=start,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 250, "ftol": 1e-10},
        )
        if result.success and np.all(np.isfinite(result.x[:2])):
            locations.append(np.asarray(result.x[:2], dtype=float))
    if not locations:
        return float("nan"), np.empty((0, 2), dtype=float)
    array = np.vstack(locations)
    center = np.median(array, axis=0)
    radius = np.sqrt(np.sum((array - center) ** 2, axis=1))
    return float(np.quantile(radius, 0.90)), array


def point_spread(selected: pd.DataFrame) -> float:
    xy = selected[["x_m", "y_m"]].to_numpy(float)
    if len(xy) < 2:
        return 0.0
    distances = []
    for i in range(len(xy)):
        for j in range(i):
            distances.append(float(np.linalg.norm(xy[i] - xy[j])))
    return float(max(distances)) if distances else 0.0


def quality_flag(
    selected: pd.DataFrame,
    uncertainty_p90: float,
    params: np.ndarray,
    bounds_xy: Sequence[float],
) -> str:
    flags: List[str] = []
    spread = point_spread(selected)
    if spread < 80.0:
        flags.append("low_spatial_spread")
    min_sector_count = int(selected["observed_sector_count"].min())
    if min_sector_count < 2 and int(selected["observed_sector_count"].max()) > 1:
        flags.append("incomplete_multi_pci_points")
    if np.isfinite(uncertainty_p90) and uncertainty_p90 > 150.0:
        flags.append("high_bootstrap_uncertainty")
    x_min, x_max, y_min, y_max = [float(v) for v in bounds_xy]
    x, y = float(params[0]), float(params[1])
    boundary_distance = min(x - x_min, x_max - x, y - y_min, y_max - y)
    if boundary_distance < 30.0:
        flags.append("solution_near_search_boundary")
    return "ok" if not flags else ";".join(flags)


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


def _expand_limits(values: np.ndarray, pad_ratio: float = 0.08, min_pad: float = 40.0) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    lo = float(np.min(values))
    hi = float(np.max(values))
    span = max(hi - lo, 1.0)
    pad = max(min_pad, span * pad_ratio)
    return lo - pad, hi + pad


def _create_fixed_equal_axes(fig, *, xlim: tuple[float, float], ylim: tuple[float, float], left: float = 0.100, bottom: float = 0.180, top: float = 0.820, right_margin: float = 0.110, cbar_pad: float = 0.012, cbar_width: float = 0.022):
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


def plot_station(station_id: int, label: str, selected: pd.DataFrame, predicted_xy: np.ndarray, true_xy: np.ndarray, bootstrap_xy: np.ndarray, error_m: float, output: Path) -> None:
    x_parts = [selected["x_m"].to_numpy(float), np.asarray([predicted_xy[0], true_xy[0]], float)]
    y_parts = [selected["y_m"].to_numpy(float), np.asarray([predicted_xy[1], true_xy[1]], float)]
    if len(bootstrap_xy):
        x_parts.append(np.asarray(bootstrap_xy[:, 0], float))
        y_parts.append(np.asarray(bootstrap_xy[:, 1], float))
    xlim = _expand_limits(np.concatenate(x_parts))
    ylim = _expand_limits(np.concatenate(y_parts))

    fig = plt.figure(figsize=_publication_figsize_inches(), dpi=MAP_DPI)
    ax, cax = _create_fixed_equal_axes(fig, xlim=xlim, ylim=ylim)
    if len(bootstrap_xy):
        ax.scatter(bootstrap_xy[:, 0], bootstrap_xy[:, 1], s=18, alpha=0.25, label="Bootstrap estimates")
    scatter = ax.scatter(selected["x_m"], selected["y_m"], c=selected[["rsrp_s1", "rsrp_s2", "rsrp_s3"]].max(axis=1), s=90, edgecolors="black", linewidths=0.7, label="Selected 5 RX points")
    for _, row in selected.iterrows():
        ax.text(float(row["x_m"]) + 5.0, float(row["y_m"]) + 5.0, str(int(row["selection_rank"])), fontsize=8)
    ax.scatter(predicted_xy[0], predicted_xy[1], marker="*", s=240, edgecolors="black", linewidths=0.8, label="Estimated base station")
    ax.scatter(true_xy[0], true_xy[1], marker="x", s=140, linewidths=2.0, label="Actual base station (evaluation only)")
    ax.plot([predicted_xy[0], true_xy[0]], [predicted_xy[1], true_xy[1]], linestyle="--", linewidth=1.2)
    ax.set_title(f"Station {station_id} localization error = {error_m:.2f} m", fontsize=9.5, pad=7.0)
    ax.set_xlabel("Blender X [m]")
    ax.set_ylabel("Blender Y [m]")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=8, framealpha=1.0)
    _add_fixed_colorbar(fig, cax, scatter, "Maximum same-station RSRP [dBm]")
    output.parent.mkdir(parents=True, exist_ok=True)
    _save_png(fig, output, dpi=COMPARISON_DPI)
    plt.close(fig)


def plot_all_stations(results: pd.DataFrame, output: Path) -> None:
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
    for _, row in results.iterrows():
        ax.text(row["predicted_x_m"] + 8, row["predicted_y_m"] + 8, str(int(row["station_id"])), fontsize=7)
    ax.set_title("Localization results for 27 stations (5 points per station)", fontsize=9.5, pad=7.0)
    ax.set_xlabel("Blender X [m]")
    ax.set_ylabel("Blender Y [m]")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", framealpha=1.0)
    _add_fixed_colorbar(fig, cax, sc, "Horizontal localization error [m]")
    _save_png(fig, output, dpi=COMPARISON_DPI)
    plt.close(fig)


def plot_error_distribution(results: pd.DataFrame, output: Path) -> None:
    errors = np.sort(results["horizontal_error_m"].to_numpy(float))
    cdf = np.arange(1, len(errors) + 1) / len(errors)
    fig, ax = plt.subplots(figsize=_publication_figsize_inches(), dpi=MAP_DPI)
    ax.plot(errors, cdf, marker="o", markersize=3)
    ax.set_xlabel("Horizontal localization error [m]")
    ax.set_ylabel("Empirical CDF")
    ax.set_title("Localization error CDF: 27 stations, 5 points each", fontsize=9.5, pad=7.0)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_png(fig, output, dpi=COMPARISON_DPI)
    plt.close(fig)

def summarize(results: pd.DataFrame) -> pd.DataFrame:
    e = results["horizontal_error_m"].to_numpy(float)
    row: Dict[str, Any] = {
        "algorithm": ALGORITHM_NAME,
        "station_count": int(len(results)),
        "points_per_station": int(results["selected_point_count"].iloc[0]),
        "mean_error_m": float(np.mean(e)),
        "median_error_m": float(np.median(e)),
        "rmse_m": float(np.sqrt(np.mean(e ** 2))),
        "p75_error_m": float(np.quantile(e, 0.75)),
        "p90_error_m": float(np.quantile(e, 0.90)),
        "p95_error_m": float(np.quantile(e, 0.95)),
        "max_error_m": float(np.max(e)),
    }
    for threshold in (20, 50, 100, 200):
        count = int(np.sum(e <= threshold))
        row[f"within_{threshold}m_count"] = count
        row[f"within_{threshold}m_percent"] = float(100.0 * count / len(e))
    return pd.DataFrame([row])


def parse_station_ids(text: str, available: Sequence[int]) -> List[int]:
    if text.strip().lower() == "all":
        return list(sorted(int(v) for v in available))
    requested = [int(v.strip()) for v in text.split(",") if v.strip()]
    missing = sorted(set(requested) - set(available))
    if missing:
        raise ValueError(f"实测数据中缺少站号：{missing}")
    return requested


def main() -> int:
    args = parse_args()
    project_root = args.project_root.expanduser().resolve()
    measurement_csv = resolve_measurement_csv(project_root, args.measurements)
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else (project_root / "outputs" / "localization_27stations_5points_pgrmsbil")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    per_station_dir = output_dir / "per_station"
    per_station_dir.mkdir(exist_ok=True)

    bounds_xy = (args.x_min, args.x_max, args.y_min, args.y_max)
    localization, truth = load_and_filter(measurement_csv)
    available = sorted(localization["station_id"].unique().tolist())
    station_ids = parse_station_ids(args.station_ids, available)
    truth_index = truth.set_index("station_id")

    print("=" * 88)
    print(ALGORITHM_NAME)
    print(f"输入：{measurement_csv}")
    print(f"站点：{len(station_ids)}；每站空间点：{args.points_per_station}")
    print("真实坐标仅在定位完成后计算误差，不进入点选择与优化。")
    print("=" * 88)

    selected_rows: List[pd.DataFrame] = []
    result_rows: List[Dict[str, Any]] = []

    for station_id in station_ids:
        start_time = time.time()
        station = localization[localization["station_id"] == station_id].copy()
        truth_row = truth_index.loc[station_id]
        label = str(truth_row["station_label"])
        omni = bool(int(truth_row["is_omnidirectional"]))
        antenna_type = str(truth_row["antenna_type"])
        points = point_table(station)
        selected = select_information_optimal_points(
            points, args.points_per_station, args.random_seed + station_id
        )
        observations = observations_from_points(selected)
        if omni:
            observations = observations[observations["sector_index"] == 1].copy()
        if len(observations) < (args.points_per_station if omni else 7):
            raise RuntimeError(
                f"站{station_id}有效观测仅{len(observations)}条，无法稳定定位。"
            )

        params, sign, objective = localize_station(
            observations,
            omni,
            bounds_xy,
            args.random_seed + station_id * 1009,
            args.de_maxiter,
            args.de_popsize,
        )
        predicted, beta, _ = model_prediction(params, sign, observations, omni)
        measured = observations["rsrp_dbm"].to_numpy(float)
        residual = predicted - measured
        fit_rmse = float(np.sqrt(np.mean(residual ** 2)))
        fit_mae = float(np.mean(np.abs(residual)))

        uncertainty, bootstrap_xy = bootstrap_uncertainty(
            params,
            sign,
            observations,
            omni,
            bounds_xy,
            args.bootstrap,
            args.random_seed + station_id * 1709,
        )

        pred_xy = np.asarray(params[:2], dtype=float)
        true_xy = np.asarray(
            [float(truth_row["true_x_m"]), float(truth_row["true_y_m"])], dtype=float
        )
        delta = pred_xy - true_xy
        error = float(np.linalg.norm(delta))
        alpha_deg = 0.0 if omni else float(math.degrees(params[3]))
        gain_scale = 0.0 if omni else float(params[4])
        qflag = quality_flag(selected, uncertainty, params, bounds_xy)
        elapsed = time.time() - start_time

        result = StationResult(
            station_id=station_id,
            station_label=label,
            antenna_type=antenna_type,
            selected_point_count=int(len(selected)),
            observation_count=int(len(observations)),
            distinct_pci_count=int(observations["pci"].nunique()),
            predicted_x_m=float(pred_xy[0]),
            predicted_y_m=float(pred_xy[1]),
            true_x_m=float(true_xy[0]),
            true_y_m=float(true_xy[1]),
            east_error_m=float(delta[0]),
            north_error_m=float(delta[1]),
            horizontal_error_m=error,
            pathloss_exponent=float(params[2]),
            alpha_deg=alpha_deg,
            sector_order_sign=int(sign),
            antenna_gain_scale=gain_scale,
            fitted_intercept_dbm=float(beta[0]),
            selected_fit_rmse_db=fit_rmse,
            selected_fit_mae_db=fit_mae,
            objective_value=float(objective),
            uncertainty_radius_p90_m=float(uncertainty),
            bootstrap_success_count=int(len(bootstrap_xy)),
            point_spread_m=point_spread(selected),
            quality_flag=qflag,
            elapsed_s=float(elapsed),
        )
        result_rows.append(result.__dict__)

        selected_out = selected.copy()
        selected_out.insert(0, "station_id", station_id)
        selected_out.insert(1, "station_label", label)
        selected_rows.append(selected_out)
        selected_out.to_csv(
            per_station_dir / f"station_{station_id:02d}_selected_{args.points_per_station}_points.csv",
            index=False,
            encoding="utf-8-sig",
        )
        observations.assign(predicted_rsrp_dbm=predicted, residual_db=residual).to_csv(
            per_station_dir / f"station_{station_id:02d}_selected_observation_fit.csv",
            index=False,
            encoding="utf-8-sig",
        )
        if not args.skip_figures:
            plot_station(
                station_id,
                label,
                selected,
                pred_xy,
                true_xy,
                bootstrap_xy,
                error,
                per_station_dir / f"station_{station_id:02d}_localization.png",
            )
        print(
            f"站{station_id:02d} {label}: obs={len(observations):2d}, "
            f"estimate=({pred_xy[0]:.2f},{pred_xy[1]:.2f}), "
            f"error={error:.2f} m, uncertainty_p90={uncertainty:.2f} m, flag={qflag}"
        )

    results = pd.DataFrame(result_rows).sort_values("station_id").reset_index(drop=True)
    selected_all = pd.concat(selected_rows, ignore_index=True)
    summary = summarize(results)

    results.to_csv(
        output_dir / f"localization_results_{len(station_ids)}stations_{args.points_per_station}points.csv",
        index=False,
        encoding="utf-8-sig",
    )
    selected_all.to_csv(
        output_dir / f"selected_{args.points_per_station}_spatial_points_all_stations.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary.to_csv(
        output_dir / "localization_accuracy_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    truth.to_csv(
        output_dir / "ground_truth_used_only_for_evaluation.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if not args.skip_figures:
        plot_all_stations(results, output_dir / "all_27stations_actual_vs_estimated.png")
        plot_error_distribution(results, output_dir / "localization_error_cdf.png")

    metadata = {
        "algorithm": ALGORITHM_NAME,
        "measurement_csv": str(measurement_csv),
        "points_definition": "exactly 5 unique spatial receiver locations per physical station; all same-station PCI observations at each selected location are used",
        "points_per_station": int(args.points_per_station),
        "spatial_aggregation": "none; raw receiver samples are used directly",
        "station_ids": station_ids,
        "search_bounds_xy": list(map(float, bounds_xy)),
        "random_seed": int(args.random_seed),
        "bootstrap_requested": int(args.bootstrap),
        "radio_filter": {
            "nr_band": TARGET_BAND,
            "center_arfcn_dl": TARGET_CENTER_ARFCN,
            "bandwidth_mhz": TARGET_BANDWIDTH_MHZ,
            "rsrp_dbm_range": [MIN_RSRP_DBM, MAX_RSRP_DBM],
            "require_dem_hit": True,
        },
        "anti_leakage": {
            "true_coordinates_removed_before_selection_and_optimization": True,
            "true_coordinates_used_only_for_final_error": True,
            "estimated_direction_csv_used": False,
        },
        "limitations": [
            "Five spatial points per station is an extremely sparse inverse problem.",
            "The analytical propagation model cannot fully reproduce campus multipath and building penetration.",
            "A low selected-point fit error does not guarantee a low coordinate error.",
            "The method is directly reproducible but is not claimed to be universally state of the art on every dataset.",
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
