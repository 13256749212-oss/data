#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""All-PCI cluster-center localization for one physical 3-sector base station.

This workflow is intentionally separate from the five-point DP-PGRSL workflow.
It uses *all valid measured observations* of the mapped PCIs belonging to one
physical base station.  Every valid receiver observation is retained directly;
there is no 2.77 m grid aggregation, coordinate rounding, or spatial
de-duplication before localization.  Then:

1. each PCI point cloud is clustered with DBSCAN using all raw observations;
2. a robust RSRP-weighted center is computed for every spatial cluster;
3. a weighted principal axis is fitted to all raw observations of each PCI;
4. the three sector axes are intersected in least-squares sense;
5. the true station coordinate is used only for final evaluation, never for
   clustering, line fitting, or localization.

The main figure is designed to resemble a "measured localization steps" plot:
raw observations, per-PCI points, per-PCI cluster centers, three dashed sector
directions from the estimated site, the final estimate, and the ground truth.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import legacy_pgrmsbil as common

DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DBSCAN_EPS_M = 12.0
DEFAULT_DBSCAN_MIN_SAMPLES = 3
RESULT_DPI = 1000
FULL_PAGE_WIDTH_IN = 7.48
FIGURE_HEIGHT_IN = 6.2


@dataclass
class SectorFit:
    pci: int
    sector_index: int
    point_count: int
    cluster_count: int
    center_xy: np.ndarray
    direction_xy: np.ndarray
    anisotropy: float
    strongest_cluster_xy: np.ndarray
    strongest_cluster_rsrp_dbm: float


def _wrap_deg(value: float) -> float:
    return ((float(value) + 180.0) % 360.0) - 180.0


def _load_calibrated_sector_angles(
    project_root: Path,
    calibration_root: Path | None,
    station_id: int,
    sector_map: pd.DataFrame,
    is_omni: bool,
) -> tuple[dict[int, float], str]:
    """Load final calibrated sector azimuths used by Sionna RT.

    Priority 1: station_<id>/best_parameters.json -> alphas_rad.
    Fallback: estimated_initial_directions_27stations.csv + all_27stations_summary.csv
    (initial alpha + calibrated common azimuth offset).
    """
    if is_omni:
        return {}, "omnidirectional-no-sector-rays"

    root = (
        Path(calibration_root).expanduser().resolve()
        if calibration_root is not None
        else project_root / "outputs" / "parameter_calibration"
    )
    ordered = sector_map.sort_values("sector_index").reset_index(drop=True)
    pcis = [int(v) for v in ordered["pci"].tolist()]

    for station_dir_name in (f"station_{station_id}", f"station_{station_id:02d}"):
        best_path = root / station_dir_name / "best_parameters.json"
        if not best_path.is_file():
            continue
        payload = json.loads(best_path.read_text(encoding="utf-8"))
        values = payload.get("alphas_rad")
        if isinstance(values, list) and len(values) == len(pcis):
            alphas = [float(v) for v in values]
            if all(np.isfinite(alphas)):
                return {pci: alpha for pci, alpha in zip(pcis, alphas)}, str(best_path)

    directions_path = root / "estimated_initial_directions_27stations.csv"
    summary_path = root / "all_27stations_summary.csv"
    if directions_path.is_file() and summary_path.is_file():
        directions = pd.read_csv(directions_path)
        summary = pd.read_csv(summary_path)
        drow = directions[directions["station_id"].astype(int).eq(int(station_id))]
        srow = summary[summary["station_id"].astype(int).eq(int(station_id))]
        if len(drow) == 1 and len(srow) == 1:
            d = drow.iloc[0]
            s = srow.iloc[0]
            offset = math.radians(float(s["azimuth_offset_deg"]))
            initial = [float(d[f"alpha_{idx}_rad"]) for idx in (1, 2, 3)]
            final = [float((a + offset + math.pi) % (2.0 * math.pi) - math.pi) for a in initial]
            return {pci: alpha for pci, alpha in zip(pcis, final)}, f"{directions_path} + {summary_path}"

    raise FileNotFoundError(
        f"station {station_id}没有找到调参后的扇区方向。需要 {root}/station_{station_id}/best_parameters.json "
        "中的 alphas_rad，或 estimated_initial_directions_27stations.csv + all_27stations_summary.csv。"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "使用物理基站的全部映射PCI原始实测点定位（不做空间聚合）：三扇区站采用PCI聚类中心+扇区轴交汇，"
            "全向单PCI站采用PCI 800的最强聚类中心定位"
        )
    )
    p.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    p.add_argument("--measurements", type=Path, default=None)
    p.add_argument("--station-id", default="2", help="物理基站编号，或 all 表示自动运行全部基站")
    p.add_argument("--dbscan-eps-m", type=float, default=DEFAULT_DBSCAN_EPS_M)
    p.add_argument("--dbscan-min-samples", type=int, default=DEFAULT_DBSCAN_MIN_SAMPLES)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument(
        "--calibration-root", type=Path, default=None,
        help="默认 outputs/parameter_calibration；绘图扇区方向严格读取调参后的alphas_rad",
    )
    p.add_argument("--skip-figure", action="store_true")
    p.add_argument("--stop-on-error", action="store_true", help="all模式下某站失败时立即停止")
    return p.parse_args()


def _rsrp_weights(rsrp: np.ndarray, scale_db: float = 12.0) -> np.ndarray:
    """Signal-strength weights for raw observations, without aggregation weights."""
    r = np.asarray(rsrp, dtype=float)
    rmax = float(np.nanmax(r))
    w = np.exp(np.clip((r - rmax) / float(scale_db), -20.0, 0.0))
    w[~np.isfinite(w)] = 0.0
    if float(np.sum(w)) <= 0:
        w = np.ones_like(r, dtype=float)
    return w


def _weighted_center(points: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.average(np.asarray(points, dtype=float), axis=0, weights=np.asarray(weights, dtype=float))


def _weighted_pca_axis(points: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    pts = np.asarray(points, dtype=float)
    w = np.asarray(weights, dtype=float)
    center = _weighted_center(pts, w)
    centered = pts - center
    cov = (centered * w[:, None]).T @ centered / max(float(np.sum(w)), 1e-12)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)
    direction = vecs[:, order[-1]].astype(float)
    direction /= max(float(np.linalg.norm(direction)), 1e-12)
    major = max(float(vals[order[-1]]), 1e-12)
    minor = max(float(vals[order[-2]]), 0.0)
    anisotropy = float(np.clip(1.0 - minor / major, 0.0, 1.0))
    return center, direction, anisotropy


def _orient_axis_by_rsrp(
    points: np.ndarray,
    rsrp: np.ndarray,
    center: np.ndarray,
    direction: np.ndarray,
) -> np.ndarray:
    """Orient the axis from stronger/nearer side toward weaker/farther side.

    This affects only arrow direction in the figure.  The geometric line and
    estimated intersection are unchanged by the sign of the direction vector.
    """
    t = (np.asarray(points, float) - center) @ direction
    r = np.asarray(rsrp, float)
    finite = np.isfinite(t) & np.isfinite(r)
    if finite.sum() >= 3:
        corr = np.corrcoef(t[finite], r[finite])[0, 1]
        if np.isfinite(corr) and corr > 0:
            direction = -direction
    return direction


def _cluster_pci_points(
    part: pd.DataFrame,
    eps_m: float,
    min_samples: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    xy = part[["x_m", "y_m"]].to_numpy(float)
    labels = DBSCAN(eps=float(eps_m), min_samples=int(min_samples)).fit_predict(xy)
    assigned = part.copy()
    assigned["cluster_id"] = labels.astype(int)

    rows: list[dict[str, Any]] = []
    for cluster_id in sorted(int(v) for v in np.unique(labels) if int(v) >= 0):
        q = assigned[assigned["cluster_id"] == cluster_id]
        qxy = q[["x_m", "y_m"]].to_numpy(float)
        qr = q["rsrp_dbm"].to_numpy(float)
        qw = _rsrp_weights(qr, scale_db=8.0)
        center = _weighted_center(qxy, qw)
        rows.append(
            {
                "pci": int(q["pci"].iloc[0]),
                "sector_index": int(q["sector_index"].iloc[0]),
                "cluster_id": int(cluster_id),
                "cluster_point_count": int(len(q)),
                "cluster_observation_count": int(len(q)),
                "cluster_center_x_m": float(center[0]),
                "cluster_center_y_m": float(center[1]),
                "cluster_rsrp_max_dbm": float(qr.max()),
                "cluster_rsrp_median_dbm": float(np.median(qr)),
            }
        )
    centers = pd.DataFrame(rows)
    return centers, labels


def _fit_sector(part: pd.DataFrame, centers: pd.DataFrame) -> SectorFit:
    xy = part[["x_m", "y_m"]].to_numpy(float)
    rsrp = part["rsrp_dbm"].to_numpy(float)
    weights = _rsrp_weights(rsrp, scale_db=12.0)
    center, direction, anisotropy = _weighted_pca_axis(xy, weights)
    direction = _orient_axis_by_rsrp(xy, rsrp, center, direction)

    if centers.empty:
        strongest_xy = center.copy()
        strongest_rsrp = float(np.nanmax(rsrp))
        cluster_count = 0
    else:
        best_idx = centers["cluster_rsrp_max_dbm"].astype(float).idxmax()
        best = centers.loc[best_idx]
        strongest_xy = np.asarray(
            [best["cluster_center_x_m"], best["cluster_center_y_m"]], dtype=float
        )
        strongest_rsrp = float(best["cluster_rsrp_max_dbm"])
        cluster_count = int(len(centers))

    return SectorFit(
        pci=int(part["pci"].iloc[0]),
        sector_index=int(part["sector_index"].iloc[0]),
        point_count=int(len(part)),
        cluster_count=cluster_count,
        center_xy=center,
        direction_xy=direction,
        anisotropy=anisotropy,
        strongest_cluster_xy=strongest_xy,
        strongest_cluster_rsrp_dbm=strongest_rsrp,
    )


def _intersect_lines(fits: list[SectorFit]) -> tuple[np.ndarray, float]:
    if len(fits) < 2:
        raise ValueError("至少需要两个扇区轴才能求交点")
    A: list[np.ndarray] = []
    b: list[float] = []
    w: list[float] = []
    for fit in fits:
        vx, vy = fit.direction_xy
        normal = np.asarray([-vy, vx], dtype=float)
        A.append(normal)
        b.append(float(normal @ fit.center_xy))
        # Avoid a near-zero weight for broad/noisy sectors.
        w.append(max(0.15, fit.anisotropy) * math.sqrt(max(fit.point_count, 1)))
    A_arr = np.asarray(A, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    w_arr = np.asarray(w, dtype=float)
    Aw = A_arr * np.sqrt(w_arr)[:, None]
    bw = b_arr * np.sqrt(w_arr)
    estimate, *_ = np.linalg.lstsq(Aw, bw, rcond=None)
    residual = A_arr @ estimate - b_arr
    rms = float(np.sqrt(np.average(residual**2, weights=w_arr)))
    return estimate.astype(float), rms




def _expand_limits(values: np.ndarray, pad_ratio: float = 0.06, min_pad: float = 80.0) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return -1.0, 1.0
    lo = float(np.min(values))
    hi = float(np.max(values))
    span = max(hi - lo, 1.0)
    pad = max(float(min_pad), span * float(pad_ratio))
    return lo - pad, hi + pad


def _create_overview_axes(
    fig,
    *,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    left: float = 0.095,
    bottom: float = 0.135,
    top: float = 0.875,
    right_margin: float = 0.125,
    cbar_pad: float = 0.018,
    cbar_width: float = 0.028,
):
    """Create an equal-aspect overview axes and an exactly aligned colorbar."""
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


def _plot_all_stations_actual_vs_estimated(summary: pd.DataFrame, output_path: Path) -> None:
    """Plot actual and All-PCI estimated positions for all successfully localized stations.

    The overview uses the final All-PCI estimates already written to the summary
    table. Ground-truth coordinates are shown only for evaluation. Estimated
    positions are colored by horizontal localization error.
    """
    required = {
        "station_id", "predicted_x_m", "predicted_y_m",
        "true_x_m", "true_y_m", "horizontal_error_m",
    }
    missing = sorted(required.difference(summary.columns))
    if missing:
        raise ValueError(f"All-PCI overview缺少字段: {missing}")

    data = summary.copy()
    for col in required.difference({"station_id"}):
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=list(required)).copy()
    if data.empty:
        raise ValueError("没有可用于all_27stations_actual_vs_estimated.png的有效定位结果")

    x_all = np.r_[data["predicted_x_m"].to_numpy(float), data["true_x_m"].to_numpy(float)]
    y_all = np.r_[data["predicted_y_m"].to_numpy(float), data["true_y_m"].to_numpy(float)]
    xlim = _expand_limits(x_all)
    ylim = _expand_limits(y_all)

    fig = plt.figure(figsize=(FULL_PAGE_WIDTH_IN, FIGURE_HEIGHT_IN), dpi=180, facecolor="white")
    ax, cax = _create_overview_axes(fig, xlim=xlim, ylim=ylim)

    # Thin connectors make each actual-estimated pair readable without dominating the map.
    for row in data.itertuples(index=False):
        ax.plot(
            [float(row.true_x_m), float(row.predicted_x_m)],
            [float(row.true_y_m), float(row.predicted_y_m)],
            color="0.70", linewidth=0.70, alpha=0.75, zorder=2,
        )

    sc = ax.scatter(
        data["predicted_x_m"], data["predicted_y_m"],
        c=data["horizontal_error_m"], cmap="viridis",
        s=62, edgecolors="black", linewidths=0.45,
        label="Estimated", zorder=5,
    )
    ax.scatter(
        data["true_x_m"], data["true_y_m"],
        marker="x", s=62, color="black", linewidths=1.45,
        label="Actual", zorder=6,
    )

    dx = 0.010 * max(xlim[1] - xlim[0], 1.0)
    dy = 0.010 * max(ylim[1] - ylim[0], 1.0)
    for row in data.itertuples(index=False):
        ax.text(
            float(row.predicted_x_m) + dx,
            float(row.predicted_y_m) + dy,
            str(int(row.station_id)),
            fontsize=6.8, color="black", zorder=7,
        )

    ax.set_title(
        f"Actual vs. Estimated Base-Station Positions — All-PCI Localization ({len(data)} stations)",
        fontsize=9.5, pad=7.0,
    )
    ax.set_xlabel("Blender X [m]")
    ax.set_ylabel("Blender Y [m]")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.35, alpha=0.28)
    ax.tick_params(labelsize=8.0)
    ax.legend(loc="upper right", fontsize=7.6, framealpha=0.96)

    cbar = fig.colorbar(sc, cax=cax)
    cbar.set_label("Horizontal localization error [m]", fontsize=8.5)
    cbar.ax.tick_params(labelsize=7.8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path, format="png", dpi=RESULT_DPI,
        bbox_inches=None, pad_inches=0.0, facecolor="white",
    )
    plt.close(fig)

def _plot_cluster_centers_only(
    raw_station: pd.DataFrame,
    assignments: pd.DataFrame,
    centers: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot all three PCI point clouds and their DBSCAN weighted cluster centers."""
    fig, ax = plt.subplots(figsize=(FULL_PAGE_WIDTH_IN, FIGURE_HEIGHT_IN), dpi=180)
    ax.scatter(
        raw_station["x_m"], raw_station["y_m"], s=1.7, c="0.84", alpha=0.22,
        linewidths=0, label="All valid measured observations", rasterized=True,
    )
    cmap = plt.get_cmap("tab10")
    sector_order = (
        assignments[["pci", "sector_index"]].drop_duplicates().sort_values("sector_index")
    )
    for idx, row in enumerate(sector_order.itertuples(index=False)):
        pci = int(row.pci)
        color = cmap(idx)
        part = assignments[assignments["pci"] == pci]
        ax.scatter(
            part["x_m"], part["y_m"], s=10, color=color, alpha=0.35,
            linewidths=0, label=f"PCI {pci} measured points", rasterized=True,
        )
        c = centers[centers["pci"] == pci]
        if not c.empty:
            ax.scatter(
                c["cluster_center_x_m"], c["cluster_center_y_m"],
                s=62, facecolors="white", edgecolors=[color], linewidths=1.4,
                label=f"PCI {pci} cluster centers", zorder=8,
            )
            for center_row in c.itertuples(index=False):
                ax.text(
                    float(center_row.cluster_center_x_m) + 4.0,
                    float(center_row.cluster_center_y_m) + 4.0,
                    f"{pci}-{int(center_row.cluster_id)}",
                    fontsize=5.2, color=color, zorder=9,
                )
    station_id = int(raw_station["station_id"].iloc[0])
    ax.set_title(f"Station {station_id}: per-PCI DBSCAN cluster centers", fontsize=9.5, pad=7.0)
    ax.set_xlabel("Blender X [m]")
    ax.set_ylabel("Blender Y [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.35, alpha=0.28)
    ax.tick_params(labelsize=8.0)
    ax.legend(loc="upper right", fontsize=7.0, framealpha=0.94)
    fig.tight_layout(pad=0.45)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="png", dpi=RESULT_DPI, bbox_inches=None, pad_inches=0.0)
    plt.close(fig)

def _plot(
    raw_station: pd.DataFrame,
    measurement_points: pd.DataFrame,
    assignments: pd.DataFrame,
    centers: pd.DataFrame,
    fits: list[SectorFit],
    calibrated_angles_rad: dict[int, float],
    estimate_xy: np.ndarray,
    true_xy: np.ndarray,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(FULL_PAGE_WIDTH_IN, FIGURE_HEIGHT_IN), dpi=180)

    # All valid observations are shown faintly in the background.
    ax.scatter(
        raw_station["x_m"], raw_station["y_m"], s=2.0, c="0.82", alpha=0.28,
        linewidths=0, label="All valid measured observations", rasterized=True,
    )

    cmap = plt.get_cmap("tab10")
    colors: dict[int, Any] = {}
    for idx, fit in enumerate(sorted(fits, key=lambda x: x.sector_index)):
        color = cmap(idx)
        colors[fit.pci] = color
        part = assignments[assignments["pci"] == fit.pci]
        ax.scatter(
            part["x_m"], part["y_m"], s=9.0, color=color, alpha=0.35,
            linewidths=0, label=f"PCI {fit.pci} measured points", rasterized=True,
        )
        c = centers[centers["pci"] == fit.pci]
        if not c.empty:
            ax.scatter(
                c["cluster_center_x_m"], c["cluster_center_y_m"],
                s=48, facecolors="none", edgecolors=[color], linewidths=1.15,
                label=f"PCI {fit.pci} cluster centers", zorder=7,
            )
        ax.scatter(
            [fit.strongest_cluster_xy[0]], [fit.strongest_cluster_xy[1]],
            marker="s", s=54, color=color, edgecolors="black", linewidths=0.6,
            zorder=8,
        )

    # Draw the three sector center directions using the FINAL CALIBRATED Sionna
    # azimuths, not PCA/fitted directions from the localization measurements.
    x_all = np.r_[measurement_points["x_m"].to_numpy(float), estimate_xy[0], true_xy[0]]
    y_all = np.r_[measurement_points["y_m"].to_numpy(float), estimate_xy[1], true_xy[1]]
    x_span = max(float(np.ptp(x_all)), 1.0)
    y_span = max(float(np.ptp(y_all)), 1.0)
    pad = 0.08 * max(x_span, y_span)
    x_min, x_max = float(np.min(x_all) - pad), float(np.max(x_all) + pad)
    y_min, y_max = float(np.min(y_all) - pad), float(np.max(y_all) + pad)
    ray_length = 0.24 * math.hypot(x_max - x_min, y_max - y_min)

    for fit in sorted(fits, key=lambda item: item.sector_index):
        color = colors[fit.pci]
        alpha = float(calibrated_angles_rad[fit.pci])
        direction = np.asarray([math.cos(alpha), math.sin(alpha)], dtype=float)
        p0 = estimate_xy
        p1 = estimate_xy + ray_length * direction
        ax.plot(
            [p0[0], p1[0]], [p0[1], p1[1]],
            color=color, linewidth=1.45, linestyle="--", alpha=0.98, zorder=6,
        )
        ax.text(
            p1[0], p1[1], f"PCI {fit.pci}  {_wrap_deg(math.degrees(alpha)):.1f}°",
            fontsize=6.2, color=color, ha="center", va="bottom", zorder=7,
        )

    ax.scatter(
        [estimate_xy[0]], [estimate_xy[1]], marker="P", s=105,
        color="black", edgecolors="white", linewidths=0.7,
        label="Raw robust line intersection", zorder=10,
    )
    ax.scatter(
        [estimate_xy[0]], [estimate_xy[1]], marker="^", s=145,
        color="red", edgecolors="black", linewidths=0.8,
        label="Final estimate", zorder=11,
    )
    ax.scatter(
        [true_xy[0]], [true_xy[1]], marker="x", s=110,
        color="black", linewidths=1.8, label="Ground truth (evaluation only)", zorder=12,
    )
    ax.plot(
        [estimate_xy[0], true_xy[0]], [estimate_xy[1], true_xy[1]],
        linestyle="--", color="0.25", linewidth=0.9, zorder=9,
    )

    station_id = int(raw_station["station_id"].iloc[0])
    error_m = float(np.linalg.norm(estimate_xy - true_xy))
    ax.set_title(
        f"Station {station_id}: all-PCI localization (error = {error_m:.2f} m)",
        fontsize=9.5, pad=7.0,
    )
    ax.set_xlabel("Blender X [m]")
    ax.set_ylabel("Blender Y [m]")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.35, alpha=0.28)
    ax.tick_params(labelsize=8.0)
    ax.legend(loc="upper right", fontsize=6.9, framealpha=0.94, ncol=1)
    fig.tight_layout(pad=0.45)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="png", dpi=RESULT_DPI, bbox_inches=None, pad_inches=0.0)
    plt.close(fig)



def _plot_single_pci(
    raw_station: pd.DataFrame,
    measurement_points: pd.DataFrame,
    assignments: pd.DataFrame,
    centers: pd.DataFrame,
    fit: SectorFit,
    estimate_xy: np.ndarray,
    true_xy: np.ndarray,
    output_path: Path,
) -> None:
    """Plot the single-PCI (omnidirectional site) localization workflow."""
    fig, ax = plt.subplots(figsize=(FULL_PAGE_WIDTH_IN, FIGURE_HEIGHT_IN), dpi=180)
    ax.scatter(
        raw_station["x_m"], raw_station["y_m"], s=2.0, c="0.82", alpha=0.25,
        linewidths=0, label="All valid PCI observations", rasterized=True,
    )
    color = plt.get_cmap("tab10")(0)
    ax.scatter(
        assignments["x_m"], assignments["y_m"], s=9.0, color=color, alpha=0.38,
        linewidths=0, label=f"PCI {fit.pci} measured points", rasterized=True,
    )
    if not centers.empty:
        ax.scatter(
            centers["cluster_center_x_m"], centers["cluster_center_y_m"],
            s=54, facecolors="white", edgecolors=[color], linewidths=1.2,
            label=f"PCI {fit.pci} DBSCAN cluster centers", zorder=7,
        )
        for row in centers.itertuples(index=False):
            ax.text(
                float(row.cluster_center_x_m) + 4.0,
                float(row.cluster_center_y_m) + 4.0,
                f"C{int(row.cluster_id)}",
                fontsize=5.4, color=color, zorder=8,
            )
    ax.scatter(
        [estimate_xy[0]], [estimate_xy[1]], marker="^", s=150,
        color="red", edgecolors="black", linewidths=0.8,
        label="Final estimate (strongest PCI cluster center)", zorder=10,
    )
    ax.scatter(
        [true_xy[0]], [true_xy[1]], marker="x", s=110,
        color="black", linewidths=1.8,
        label="Ground truth (evaluation only)", zorder=11,
    )
    ax.plot(
        [estimate_xy[0], true_xy[0]], [estimate_xy[1], true_xy[1]],
        linestyle="--", color="0.25", linewidth=0.9, zorder=9,
    )

    x_all = np.r_[measurement_points["x_m"].to_numpy(float), estimate_xy[0], true_xy[0]]
    y_all = np.r_[measurement_points["y_m"].to_numpy(float), estimate_xy[1], true_xy[1]]
    x_span = max(float(np.ptp(x_all)), 1.0)
    y_span = max(float(np.ptp(y_all)), 1.0)
    pad = 0.08 * max(x_span, y_span)
    ax.set_xlim(float(np.min(x_all) - pad), float(np.max(x_all) + pad))
    ax.set_ylim(float(np.min(y_all) - pad), float(np.max(y_all) + pad))
    station_id = int(raw_station["station_id"].iloc[0])
    error_m = float(np.linalg.norm(estimate_xy - true_xy))
    ax.set_title(
        f"Station {station_id} / PCI {fit.pci}: localization (error = {error_m:.2f} m)",
        fontsize=9.5, pad=7.0,
    )
    ax.set_xlabel("Blender X [m]")
    ax.set_ylabel("Blender Y [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.35, alpha=0.28)
    ax.tick_params(labelsize=8.0)
    ax.legend(loc="upper right", fontsize=6.9, framealpha=0.94)
    fig.tight_layout(pad=0.45)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="png", dpi=RESULT_DPI, bbox_inches=None, pad_inches=0.0)
    plt.close(fig)


def _run_one_station(
    *,
    station_id: int,
    localization: pd.DataFrame,
    truth: pd.DataFrame,
    project_root: Path,
    args: argparse.Namespace,
    batch_root: Path | None = None,
) -> dict[str, Any]:
    raw_station = localization[localization["station_id"] == int(station_id)].copy()
    if raw_station.empty:
        raise ValueError(f"实测长表中没有station_id={station_id}的有效目标PCI数据")

    truth_rows = truth[truth["station_id"] == int(station_id)]
    if truth_rows.empty:
        raise ValueError(f"找不到station_id={station_id}的真实站点坐标用于最终评价")
    truth_row = truth_rows.iloc[0]
    is_omni = int(truth_row["is_omnidirectional"]) == 1

    sector_map = (
        raw_station[["pci", "sector_index"]]
        .drop_duplicates()
        .sort_values(["sector_index", "pci"])
        .reset_index(drop=True)
    )
    if is_omni:
        if len(sector_map) != 1:
            raise ValueError(
                f"station {station_id}标记为全向站，但当前有效PCI映射不是1个："
                f"{sector_map.to_dict(orient='records')}"
            )
    else:
        if len(sector_map) != 3 or set(sector_map["sector_index"].astype(int)) != {1, 2, 3}:
            raise ValueError(
                f"station {station_id}需要恰好3个扇区PCI，当前映射为："
                f"{sector_map.to_dict(orient='records')}"
            )

    calibrated_angles_rad, calibrated_angle_source = _load_calibrated_sector_angles(
        project_root=project_root,
        calibration_root=args.calibration_root,
        station_id=station_id,
        sector_map=sector_map,
        is_omni=is_omni,
    )

    measurement_points = raw_station.copy().reset_index(drop=True)
    all_centers: list[pd.DataFrame] = []
    all_assignments: list[pd.DataFrame] = []
    fits: list[SectorFit] = []

    for _, row in sector_map.iterrows():
        pci = int(row["pci"])
        part = measurement_points[measurement_points["pci"] == pci].copy()
        if len(part) < max(6, int(args.dbscan_min_samples)):
            raise ValueError(f"PCI {pci}有效空间点仅{len(part)}个，不足以进行全部点聚类定位")
        centers, labels = _cluster_pci_points(
            part, eps_m=float(args.dbscan_eps_m), min_samples=int(args.dbscan_min_samples)
        )
        assigned = part.copy()
        assigned["cluster_id"] = labels
        all_centers.append(centers)
        all_assignments.append(assigned)
        fits.append(_fit_sector(part, centers))

    centers_df = pd.concat(all_centers, ignore_index=True) if all_centers else pd.DataFrame()
    assignments_df = pd.concat(all_assignments, ignore_index=True)

    if is_omni:
        fit = fits[0]
        estimate_xy = fit.strongest_cluster_xy.astype(float).copy()
        line_residual_rms_m = float("nan")
        method = "Single-PCI raw-observation DBSCAN strongest-cluster-center localization"
    else:
        estimate_xy, line_residual_rms_m = _intersect_lines(fits)
        method = "All-PCI raw-observation DBSCAN cluster-center + weighted sector-axis intersection"

    true_xy = np.asarray([truth_row["true_x_m"], truth_row["true_y_m"]], dtype=float)
    delta = estimate_xy - true_xy
    error_m = float(np.linalg.norm(delta))

    if batch_root is not None:
        output_dir = batch_root / f"station_{station_id:02d}"
    elif args.output_dir is None:
        output_dir = project_root / "outputs" / "localization_all_pci_clusters" / f"station_{station_id:02d}"
    else:
        output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    assignments_df.to_csv(output_dir / "all_pci_measurement_points_with_clusters.csv", index=False, encoding="utf-8-sig")
    centers_df.to_csv(output_dir / "pci_cluster_centers.csv", index=False, encoding="utf-8-sig")

    sector_rows: list[dict[str, Any]] = []
    for fit in sorted(fits, key=lambda x: x.sector_index):
        sector_rows.append(
            {
                "station_id": station_id,
                "pci": fit.pci,
                "sector_index": fit.sector_index,
                "spatial_point_count": fit.point_count,
                "cluster_count": fit.cluster_count,
                "axis_center_x_m": float(fit.center_xy[0]),
                "axis_center_y_m": float(fit.center_xy[1]),
                "axis_direction_x": float(fit.direction_xy[0]),
                "axis_direction_y": float(fit.direction_xy[1]),
                "axis_anisotropy": float(fit.anisotropy),
                "calibrated_azimuth_rad": float(calibrated_angles_rad.get(fit.pci, np.nan)),
                "calibrated_azimuth_deg": float(_wrap_deg(math.degrees(calibrated_angles_rad[fit.pci]))) if fit.pci in calibrated_angles_rad else np.nan,
                "strongest_cluster_x_m": float(fit.strongest_cluster_xy[0]),
                "strongest_cluster_y_m": float(fit.strongest_cluster_xy[1]),
                "strongest_cluster_rsrp_dbm": float(fit.strongest_cluster_rsrp_dbm),
            }
        )
    pd.DataFrame(sector_rows).to_csv(output_dir / "pci_sector_axis_fits.csv", index=False, encoding="utf-8-sig")

    result = {
        "method": method,
        "station_id": station_id,
        "station_type": "omnidirectional_single_pci" if is_omni else "three_sector",
        "pcis": [int(v) for v in sector_map["pci"].tolist()],
        "uses_all_valid_pci_points": True,
        "spatial_aggregation": "none; all valid raw receiver observations are used directly",
        "dbscan_eps_m": float(args.dbscan_eps_m),
        "dbscan_min_samples": int(args.dbscan_min_samples),
        "raw_valid_observation_count": int(len(raw_station)),
        "measurement_point_count": int(len(measurement_points)),
        "cluster_center_count": int(len(centers_df)),
        "predicted_x_m": float(estimate_xy[0]),
        "predicted_y_m": float(estimate_xy[1]),
        "true_x_m": float(true_xy[0]),
        "true_y_m": float(true_xy[1]),
        "east_error_m": float(delta[0]),
        "north_error_m": float(delta[1]),
        "horizontal_error_m": error_m,
        "line_intersection_residual_rms_m": float(line_residual_rms_m),
        "ground_truth_used_only_for_evaluation": True,
        "sector_display_angle_source": calibrated_angle_source,
        "calibrated_sector_azimuth_deg": {
            str(pci): _wrap_deg(math.degrees(alpha))
            for pci, alpha in calibrated_angles_rad.items()
        },
    }
    pd.DataFrame([result]).to_csv(output_dir / "all_pci_localization_result.csv", index=False, encoding="utf-8-sig")
    (output_dir / "all_pci_localization_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )

    if not args.skip_figure:
        _plot_cluster_centers_only(
            raw_station=raw_station,
            assignments=assignments_df,
            centers=centers_df,
            output_path=output_dir / f"01_station_{station_id:02d}_pci_cluster_centers.png",
        )
        if is_omni:
            _plot_single_pci(
                raw_station=raw_station,
                measurement_points=measurement_points,
                assignments=assignments_df,
                centers=centers_df,
                fit=fits[0],
                estimate_xy=estimate_xy,
                true_xy=true_xy,
                output_path=output_dir / f"02_station_{station_id:02d}_single_pci_localization_steps.png",
            )
        else:
            _plot(
                raw_station=raw_station,
                measurement_points=measurement_points,
                assignments=assignments_df,
                centers=centers_df,
                fits=fits,
                calibrated_angles_rad=calibrated_angles_rad,
                estimate_xy=estimate_xy,
                true_xy=true_xy,
                output_path=output_dir / f"02_station_{station_id:02d}_all_pci_localization_steps.png",
            )

    print(f"[OK] station {station_id}: PCI={result['pcis']} type={result['station_type']}")
    print(f"[OK] valid observations={len(raw_station)}, measurement points used directly={len(measurement_points)}, cluster centers={len(centers_df)}")
    print(f"[OK] predicted=({estimate_xy[0]:.3f}, {estimate_xy[1]:.3f}) m")
    print(f"[OK] ground truth=({true_xy[0]:.3f}, {true_xy[1]:.3f}) m")
    print(f"[OK] horizontal error={error_m:.3f} m")
    print(f"[OK] output={output_dir}")
    return result

def run(args: argparse.Namespace) -> int:
    project_root = args.project_root.expanduser().resolve()
    measurement_csv = common.resolve_measurement_csv(project_root, args.measurements)
    localization, truth = common.load_and_filter(measurement_csv)

    station_token = str(args.station_id).strip().lower()
    if args.output_dir is None:
        batch_root = project_root / "outputs" / "localization_all_pci_clusters"
    else:
        batch_root = args.output_dir.expanduser().resolve()

    if station_token != "all":
        station_id = int(station_token)
        _run_one_station(
            station_id=station_id,
            localization=localization,
            truth=truth,
            project_root=project_root,
            args=args,
            batch_root=None,
        )
        return 0

    station_ids = sorted(int(v) for v in truth["station_id"].dropna().unique())
    batch_root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    print(f"[INFO] all-station mode: {len(station_ids)} physical stations -> {station_ids}")
    for idx, station_id in enumerate(station_ids, start=1):
        print("=" * 72)
        print(f"[INFO] [{idx}/{len(station_ids)}] localize station {station_id}")
        try:
            result = _run_one_station(
                station_id=station_id,
                localization=localization,
                truth=truth,
                project_root=project_root,
                args=args,
                batch_root=batch_root,
            )
            results.append(result)
        except Exception as exc:
            failures.append({"station_id": station_id, "error": str(exc)})
            print(f"[ERROR] station {station_id}: {exc}")
            if getattr(args, "stop_on_error", False):
                break

    if results:
        summary = pd.DataFrame(results).sort_values("station_id").reset_index(drop=True)
        summary.to_csv(batch_root / "all_stations_all_pci_localization_summary.csv", index=False, encoding="utf-8-sig")
        if not args.skip_figure:
            overview_path = batch_root / "all_27stations_actual_vs_estimated.png"
            _plot_all_stations_actual_vs_estimated(summary, overview_path)
            print(f"[OK] all-station actual-vs-estimated figure: {overview_path}")
        errors = summary["horizontal_error_m"].to_numpy(float)
        metrics = {
            "station_count": int(len(summary)),
            "mean_error_m": float(np.mean(errors)),
            "median_error_m": float(np.median(errors)),
            "rmse_error_m": float(np.sqrt(np.mean(errors**2))),
            "p90_error_m": float(np.quantile(errors, 0.90)),
            "p95_error_m": float(np.quantile(errors, 0.95)),
            "max_error_m": float(np.max(errors)),
            "three_sector_count": int((summary["station_type"] == "three_sector").sum()),
            "single_pci_omnidirectional_count": int((summary["station_type"] == "omnidirectional_single_pci").sum()),
        }
        pd.DataFrame([metrics]).to_csv(batch_root / "all_stations_all_pci_accuracy_summary.csv", index=False, encoding="utf-8-sig")
        (batch_root / "all_stations_all_pci_accuracy_summary.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    if failures:
        pd.DataFrame(failures).to_csv(batch_root / "all_stations_all_pci_failures.csv", index=False, encoding="utf-8-sig")

    print("=" * 72)
    print(f"[DONE] succeeded={len(results)} failed={len(failures)}")
    print(f"[DONE] output root={batch_root}")
    return 0 if (results and not (failures and getattr(args, "stop_on_error", False))) else 1

def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
