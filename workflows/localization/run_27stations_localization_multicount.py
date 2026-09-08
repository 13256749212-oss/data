#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Progressive nested Monte-Carlo sparse base-station localization (v1.15).

For each Monte-Carlo trial a single uniformly random station-specific ordering of
localization-eligible measurement locations is generated.  Point counts 10--15
use nested prefixes of exactly the same ordering, so adding a point never discards
previous observations: S10 ⊂ S11 ⊂ ... ⊂ S15.

The single-count MCVL solver is also progressive.  Starting at 11 points, the
previous-count estimate for the same station/trial is inherited as an explicit
candidate and is retained unless the enlarged measurement set provides a clear
measurement-only candidate-score improvement.  Ground-truth station coordinates
are never used for random selection, candidate inheritance, update acceptance,
or multi-trial fusion.

The workflow still reports raw trial metrics, arithmetic 10-trial coordinate
means, and MCVL-RC robust consensus so that no result is hidden by the progressive
experiment design.
"""
from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_COUNTS = [10, 11, 12, 13, 14, 15]
ROOT = Path(__file__).resolve().parents[2]
SINGLE_SCRIPT = Path(__file__).resolve().parent / "run_27stations_multicandidate_cv_localization.py"


def parse_counts(text: str) -> list[int]:
    values: list[int] = []
    for token in str(text).replace(";", ",").split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value < 3:
            raise ValueError(f"定位点数至少为3，当前得到{value}")
        if value not in values:
            values.append(value)
    if not values:
        raise ValueError("至少需要一个定位点数")
    return sorted(values)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PN-MCVL-RC-v1.15：10--15点嵌套随机定位，同一trial逐点增加观测并继承上一点数解，再进行10次鲁棒共识"
    )
    p.add_argument("--project-root", type=Path, default=ROOT)
    p.add_argument("--measurements", type=Path, default=None)
    p.add_argument("--directions", type=Path, default=None)
    p.add_argument("--calibration-root", type=Path, default=None)
    p.add_argument("--output-root", type=Path, default=None)
    p.add_argument("--point-counts", default=",".join(map(str, DEFAULT_COUNTS)))
    p.add_argument("--random-trials", type=int, default=10)
    p.add_argument("--random-seed", type=int, default=20260805)
    p.add_argument("--bootstrap", type=int, default=0)
    p.add_argument("--de-maxiter", type=int, default=100)
    p.add_argument("--de-popsize", type=int, default=10)
    p.add_argument("--direction-prior-mode", choices=["fixed", "soft", "off"], default="fixed")
    p.add_argument("--station-ids", default="all")
    p.add_argument("--x-min", type=float, default=-1732.5)
    p.add_argument("--x-max", type=float, default=2267.5)
    p.add_argument("--y-min", type=float, default=-1430.47)
    p.add_argument("--y-max", type=float, default=1569.53)
    p.add_argument("--dpi", type=int, default=1000)
    p.add_argument(
        "--keep-trial-figures", action="store_true",
        help="保留每次随机trial内部的逐站图/总览图；默认关闭以避免60次实验生成数千张图片",
    )
    p.add_argument("--skip-figures", action="store_true", help="跳过最终Monte-Carlo汇总图")
    p.add_argument("--continue-on-error", action="store_true")
    p.add_argument("--resume", action="store_true", help="复用输出目录中已完整生成的trial，续跑缺失或不完整的trial")
    p.add_argument("--consensus-mad-z", type=float, default=2.0, help="鲁棒共识的径向MAD异常阈值系数；默认2.0")
    p.add_argument("--consensus-min-inlier-fraction", type=float, default=0.60, help="每站10次预测至少保留的共识内点比例；默认0.60")
    p.add_argument("--consensus-min-scale-m", type=float, default=5.0, help="径向鲁棒尺度下限[m]，防止MAD过小时过度剔除；默认5 m")
    p.add_argument(
        "--progressive-min-improvement-db", type=float, default=0.25,
        help="11--15点时，新解相对上一点数位置至少需要改善的当前测量候选分数[dB]；默认0.25",
    )
    p.add_argument(
        "--simulation-mode", choices=["without", "with", "compare"], default="without",
        help="每个点数/trial运行仅实测、同位置实测+仿真联合定位或严格配对对比",
    )
    p.add_argument("--simulation-root", type=Path, default=None)
    p.add_argument("--simulation-weight", type=float, default=0.50, help="联合候选评分中仿真RSRP通道权重，范围0--1")
    p.add_argument("--strict-simulation-data", action="store_true")
    return p.parse_args()


def _display_command(command: Iterable[object]) -> str:
    parts = []
    for item in command:
        text = str(item)
        parts.append(f'"{text}"' if " " in text else text)
    return " ".join(parts)


def _find_result_csv(trial_dir: Path, point_count: int) -> Path:
    candidates = sorted(trial_dir.glob(f"localization_results_*stations_{point_count}points.csv"))
    if not candidates:
        raise FileNotFoundError(f"未找到trial定位结果：{trial_dir}")
    return candidates[-1]


def _metrics_from_result(frame: pd.DataFrame, point_count: int, trial_index: int, trial_seed: int) -> dict[str, float | int]:
    e = pd.to_numeric(frame["horizontal_error_m"], errors="coerce").to_numpy(dtype=float)
    e = e[np.isfinite(e)]
    if len(e) == 0:
        raise ValueError("trial没有有效horizontal_error_m")
    return {
        "point_count": int(point_count),
        "trial_index": int(trial_index),
        "trial_seed": int(trial_seed),
        "station_count": int(len(e)),
        "mean_error_m": float(np.mean(e)),
        "median_error_m": float(np.median(e)),
        "rmse_m": float(np.sqrt(np.mean(e ** 2))),
        "p90_error_m": float(np.percentile(e, 90)),
        "p95_error_m": float(np.percentile(e, 95)),
        "max_error_m": float(np.max(e)),
        "within_50m_percent": float(np.mean(e <= 50.0) * 100.0),
        "within_100m_percent": float(np.mean(e <= 100.0) * 100.0),
    }


def stabilize_cross_count_predictions(
    station_results: pd.DataFrame,
    *,
    enabled: bool = True,
    drift_min_m: float = 60.0,
    uncertainty_min_m: float = 60.0,
) -> pd.DataFrame:
    """Legacy compatibility helper; v1.9 Monte-Carlo workflow does not call it."""
    out = station_results.sort_values(["station_id", "point_count"]).copy()
    for col in ["predicted_x_m", "predicted_y_m", "east_error_m", "north_error_m", "horizontal_error_m", "quality_flag"]:
        if col in out.columns:
            out[f"raw_{col}"] = out[col]
    out["cross_count_reference_x_m"] = np.nan
    out["cross_count_reference_y_m"] = np.nan
    out["cumulative_prediction_drift_m"] = np.nan
    out["cross_count_drift_limit_m"] = np.nan
    out["cross_count_stabilized"] = False
    for _, idx in out.groupby("station_id", sort=False).groups.items():
        indices = list(idx)
        if not indices:
            continue
        first = indices[0]
        ref = np.asarray([float(out.at[first, "raw_predicted_x_m"]), float(out.at[first, "raw_predicted_y_m"])])
        for row_idx in indices:
            raw_xy = np.asarray([float(out.at[row_idx, "raw_predicted_x_m"]), float(out.at[row_idx, "raw_predicted_y_m"])])
            spread = float(out.at[row_idx, "point_spread_m"]) if "point_spread_m" in out.columns else np.nan
            uncertainty = float(out.at[row_idx, "uncertainty_radius_p90_m"]) if "uncertainty_radius_p90_m" in out.columns else np.nan
            drift_limit = max(float(drift_min_m), 0.20 * spread if np.isfinite(spread) else float(drift_min_m))
            delta = raw_xy - ref
            drift = float(np.linalg.norm(delta))
            out.at[row_idx, "cross_count_reference_x_m"] = ref[0]
            out.at[row_idx, "cross_count_reference_y_m"] = ref[1]
            out.at[row_idx, "cumulative_prediction_drift_m"] = drift
            out.at[row_idx, "cross_count_drift_limit_m"] = drift_limit
            should_limit = enabled and row_idx != first and drift > drift_limit and np.isfinite(uncertainty) and uncertainty >= float(uncertainty_min_m)
            final_xy = raw_xy.copy()
            if should_limit and drift > 1e-9:
                final_xy = ref + delta * (drift_limit / drift)
                out.at[row_idx, "cross_count_stabilized"] = True
                previous = str(out.at[row_idx, "quality_flag"]) if "quality_flag" in out.columns else "ok"
                flags = [] if previous in {"", "nan", "ok"} else previous.split(";")
                flags.append("cross_count_drift_limited")
                out.at[row_idx, "quality_flag"] = ";".join(dict.fromkeys(flags))
            out.at[row_idx, "predicted_x_m"] = final_xy[0]
            out.at[row_idx, "predicted_y_m"] = final_xy[1]
            if {"true_x_m", "true_y_m"}.issubset(out.columns):
                east = float(final_xy[0] - float(out.at[row_idx, "true_x_m"]))
                north = float(final_xy[1] - float(out.at[row_idx, "true_y_m"]))
                out.at[row_idx, "east_error_m"] = east
                out.at[row_idx, "north_error_m"] = north
                out.at[row_idx, "horizontal_error_m"] = float(np.hypot(east, north))
    return out


def summarize_stabilized_results(station_results: pd.DataFrame) -> pd.DataFrame:
    """Legacy compatibility helper; not used by v1.9 random-trial summaries."""
    rows = []
    for count, group in station_results.groupby("point_count", sort=True):
        e = pd.to_numeric(group["horizontal_error_m"], errors="coerce").to_numpy(dtype=float)
        e = e[np.isfinite(e)]
        rows.append({
            "algorithm": "DP-PGRSL legacy stabilized",
            "station_count": int(len(group)),
            "requested_points_per_station": int(count),
            "selected_points_min": int(group["selected_point_count"].min()) if "selected_point_count" in group.columns else int(count),
            "selected_points_max": int(group["selected_point_count"].max()) if "selected_point_count" in group.columns else int(count),
            "mean_error_m": float(np.mean(e)),
            "median_error_m": float(np.median(e)),
            "rmse_m": float(np.sqrt(np.mean(e ** 2))),
            "p75_error_m": float(np.percentile(e, 75)),
            "p90_error_m": float(np.percentile(e, 90)),
            "p95_error_m": float(np.percentile(e, 95)),
            "max_error_m": float(np.max(e)),
            "within_20m_percent": float(np.mean(e <= 20.0) * 100.0),
            "within_50m_percent": float(np.mean(e <= 50.0) * 100.0),
            "within_100m_percent": float(np.mean(e <= 100.0) * 100.0),
            "cross_count_stabilized_count": int(group.get("cross_count_stabilized", pd.Series(False, index=group.index)).sum()),
            "point_count": int(count),
        })
    return pd.DataFrame(rows)


def save_comparison_outputs(
    output_root: Path,
    summary: pd.DataFrame,
    station_results: pd.DataFrame,
    *,
    dpi: int,
    save_figures: bool,
) -> None:
    """Legacy CSV writer retained for backward-compatible tests/tools."""
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summary = summary.sort_values("requested_points_per_station").reset_index(drop=True)
    station_results = station_results.sort_values(["station_id", "point_count"]).reset_index(drop=True)
    summary.to_csv(output_root / "localization_multicount_summary_10to15.csv", index=False, encoding="utf-8-sig")
    station_results.to_csv(output_root / "localization_multicount_station_errors_long.csv", index=False, encoding="utf-8-sig")
    pivot = station_results.pivot(index="station_id", columns="point_count", values="horizontal_error_m")
    pivot.to_csv(output_root / "localization_multicount_station_errors_wide.csv", encoding="utf-8-sig")
    cols = [c for c in ["requested_points_per_station", "station_count", "mean_error_m", "median_error_m", "rmse_m", "p90_error_m", "p95_error_m", "max_error_m", "within_50m_percent", "within_100m_percent"] if c in summary.columns]
    summary[cols].to_csv(output_root / "localization_multicount_total_comparison_table.csv", index=False, encoding="utf-8-sig")
    if not save_figures:
        return
    fig, ax = plt.subplots(figsize=(7.48, 5.2), dpi=int(dpi))
    x = summary["requested_points_per_station"].to_numpy(dtype=int)
    for col, label in [("mean_error_m", "Mean"), ("median_error_m", "Median"), ("rmse_m", "RMSE"), ("p90_error_m", "P90")]:
        if col in summary.columns:
            ax.plot(x, summary[col], marker="o", label=label)
    ax.set_xlabel("Points per station"); ax.set_ylabel("Error [m]"); ax.grid(True, alpha=0.25); ax.legend()
    fig.tight_layout(); fig.savefig(output_root / "localization_multicount_accuracy_curves.png", dpi=int(dpi), bbox_inches="tight", facecolor="white"); plt.close(fig)


def summarize_trials(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    metric_names = [
        "mean_error_m", "median_error_m", "rmse_m", "p90_error_m", "p95_error_m",
        "max_error_m", "within_50m_percent", "within_100m_percent",
    ]
    for point_count, group in metrics.groupby("point_count", sort=True):
        row: dict[str, object] = {
            "point_count": int(point_count),
            "completed_trials": int(len(group)),
            "station_count_per_trial": int(round(float(group["station_count"].median()))),
        }
        for name in metric_names:
            values = pd.to_numeric(group[name], errors="coerce").to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            row[f"{name}_mean_over_trials"] = float(np.mean(values)) if len(values) else np.nan
            row[f"{name}_std_over_trials"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            row[f"{name}_min_over_trials"] = float(np.min(values)) if len(values) else np.nan
            row[f"{name}_max_over_trials"] = float(np.max(values)) if len(values) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_station_trials(all_results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (point_count, station_id), group in all_results.groupby(["point_count", "station_id"], sort=True):
        px = pd.to_numeric(group["predicted_x_m"], errors="coerce").to_numpy(dtype=float)
        py = pd.to_numeric(group["predicted_y_m"], errors="coerce").to_numpy(dtype=float)
        err = pd.to_numeric(group["horizontal_error_m"], errors="coerce").to_numpy(dtype=float)
        tx = float(pd.to_numeric(group["true_x_m"], errors="coerce").median())
        ty = float(pd.to_numeric(group["true_y_m"], errors="coerce").median())
        mean_x = float(np.nanmean(px))
        mean_y = float(np.nanmean(py))
        row = {
            "point_count": int(point_count),
            "station_id": int(station_id),
            "station_label": str(group["station_label"].iloc[0]) if "station_label" in group.columns else "",
            "trials": int(len(group)),
            "true_x_m": tx,
            "true_y_m": ty,
            "mean_predicted_x_m": mean_x,
            "mean_predicted_y_m": mean_y,
            "std_predicted_x_m": float(np.nanstd(px, ddof=1)) if len(px) > 1 else 0.0,
            "std_predicted_y_m": float(np.nanstd(py, ddof=1)) if len(py) > 1 else 0.0,
            "mean_horizontal_error_m": float(np.nanmean(err)),
            "std_horizontal_error_m": float(np.nanstd(err, ddof=1)) if len(err) > 1 else 0.0,
            "median_horizontal_error_m": float(np.nanmedian(err)),
            "error_of_mean_prediction_m": float(np.hypot(mean_x - tx, mean_y - ty)),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_mean_prediction_ensemble(station_average: pd.DataFrame) -> pd.DataFrame:
    """Overall accuracy after averaging the 10 predicted coordinates per station."""
    rows: list[dict[str, float | int]] = []
    for point_count, group in station_average.groupby("point_count", sort=True):
        e = pd.to_numeric(group["error_of_mean_prediction_m"], errors="coerce").to_numpy(dtype=float)
        e = e[np.isfinite(e)]
        rows.append({
            "point_count": int(point_count),
            "station_count": int(len(e)),
            "ensemble_definition": "arithmetic mean of 10 independently randomized predicted XY coordinates per station",
            "mean_error_m": float(np.mean(e)),
            "median_error_m": float(np.median(e)),
            "rmse_m": float(np.sqrt(np.mean(e ** 2))),
            "p90_error_m": float(np.percentile(e, 90)),
            "p95_error_m": float(np.percentile(e, 95)),
            "max_error_m": float(np.max(e)),
            "within_50m_percent": float(np.mean(e <= 50.0) * 100.0),
            "within_100m_percent": float(np.mean(e <= 100.0) * 100.0),
        })
    return pd.DataFrame(rows)



def geometric_median_xy(points: np.ndarray, *, tolerance_m: float = 1e-5, max_iter: int = 500) -> np.ndarray:
    """Compute a 2-D geometric median without using ground-truth coordinates."""
    pts = np.asarray(points, dtype=float)
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    if len(pts) == 0:
        return np.asarray([np.nan, np.nan], dtype=float)
    if len(pts) == 1:
        return pts[0].copy()
    current = np.nanmedian(pts, axis=0)
    for _ in range(int(max_iter)):
        distance = np.linalg.norm(pts - current[None, :], axis=1)
        weights = 1.0 / np.maximum(distance, 1e-8)
        updated = np.sum(pts * weights[:, None], axis=0) / np.sum(weights)
        if float(np.linalg.norm(updated - current)) <= float(tolerance_m):
            current = updated
            break
        current = updated
    return np.asarray(current, dtype=float)


def build_robust_consensus(
    all_results: pd.DataFrame,
    *,
    mad_z: float = 2.0,
    min_inlier_fraction: float = 0.60,
    min_scale_m: float = 5.0,
) -> pd.DataFrame:
    """Fuse randomized trial estimates with a truth-free robust spatial consensus.

    Procedure for each station and point count:
      1. compute the geometric median of the trial XY estimates;
      2. calculate radial distances to that center;
      3. estimate radial spread by MAD;
      4. reject estimates beyond median_distance + mad_z * robust_scale;
      5. guarantee a minimum inlier fraction by retaining the closest estimates;
      6. average only the retained inlier XY coordinates.

    Ground truth is never used for inlier selection or final coordinate fusion.  It is
    read only after the consensus position has been formed, to report evaluation error.
    """
    if not (0.0 < float(min_inlier_fraction) <= 1.0):
        raise ValueError("--consensus-min-inlier-fraction必须在(0,1]范围内")
    if float(mad_z) < 0.0:
        raise ValueError("--consensus-mad-z不能为负数")
    rows: list[dict[str, object]] = []
    for (point_count, station_id), group in all_results.groupby(["point_count", "station_id"], sort=True):
        work = group.copy().sort_values("trial_index" if "trial_index" in group.columns else group.index.name or "station_id")
        px = pd.to_numeric(work["predicted_x_m"], errors="coerce").to_numpy(dtype=float)
        py = pd.to_numeric(work["predicted_y_m"], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(px) & np.isfinite(py)
        if not np.any(finite):
            continue
        pts = np.column_stack([px[finite], py[finite]])
        finite_rows = work.loc[finite].copy().reset_index(drop=True)
        center = geometric_median_xy(pts)
        radial = np.linalg.norm(pts - center[None, :], axis=1)
        radial_median = float(np.median(radial))
        radial_mad = float(np.median(np.abs(radial - radial_median)))
        robust_sigma = float(max(1.4826 * radial_mad, float(min_scale_m)))
        threshold = float(radial_median + float(mad_z) * robust_sigma)
        inlier = radial <= threshold
        minimum_inliers = max(3, int(math.ceil(float(min_inlier_fraction) * len(pts))))
        minimum_inliers = min(minimum_inliers, len(pts))
        if int(np.sum(inlier)) < minimum_inliers:
            keep = np.argsort(radial)[:minimum_inliers]
            inlier = np.zeros(len(pts), dtype=bool)
            inlier[keep] = True
        inlier_pts = pts[inlier]
        consensus = np.mean(inlier_pts, axis=0)
        consensus_distance = np.linalg.norm(inlier_pts - consensus[None, :], axis=1)
        inlier_rms = float(np.sqrt(np.mean(consensus_distance ** 2))) if len(consensus_distance) else np.nan
        inlier_p90 = float(np.percentile(consensus_distance, 90)) if len(consensus_distance) else np.nan
        all_p90 = float(np.percentile(np.linalg.norm(pts - consensus[None, :], axis=1), 90))
        tx = float(pd.to_numeric(work["true_x_m"], errors="coerce").median()) if "true_x_m" in work.columns else np.nan
        ty = float(pd.to_numeric(work["true_y_m"], errors="coerce").median()) if "true_y_m" in work.columns else np.nan
        error = float(np.hypot(consensus[0] - tx, consensus[1] - ty)) if np.isfinite(tx) and np.isfinite(ty) else np.nan
        trial_ids = pd.to_numeric(finite_rows.get("trial_index", pd.Series(np.arange(1, len(finite_rows)+1))), errors="coerce").fillna(-1).astype(int).to_numpy()
        kept_ids = [str(int(v)) for v in trial_ids[inlier]]
        rejected_ids = [str(int(v)) for v in trial_ids[~inlier]]
        rows.append({
            "point_count": int(point_count),
            "station_id": int(station_id),
            "station_label": str(work["station_label"].iloc[0]) if "station_label" in work.columns else "",
            "consensus_algorithm": "PN-MCVL-RC-v1.15 geometric-median MAD trimmed mean",
            "trial_count": int(len(pts)),
            "inlier_trial_count": int(np.sum(inlier)),
            "outlier_trial_count": int(np.sum(~inlier)),
            "inlier_trial_indices": ";".join(kept_ids),
            "rejected_trial_indices": ";".join(rejected_ids),
            "geometric_median_x_m": float(center[0]),
            "geometric_median_y_m": float(center[1]),
            "radial_distance_median_m": radial_median,
            "radial_distance_mad_m": radial_mad,
            "robust_radial_sigma_m": robust_sigma,
            "consensus_outlier_threshold_m": threshold,
            "consensus_predicted_x_m": float(consensus[0]),
            "consensus_predicted_y_m": float(consensus[1]),
            "consensus_inlier_rms_spread_m": inlier_rms,
            "consensus_inlier_p90_spread_m": inlier_p90,
            "consensus_all_trial_p90_spread_m": all_p90,
            "true_x_m": tx,
            "true_y_m": ty,
            "consensus_horizontal_error_m": error,
            "consensus_mad_z": float(mad_z),
            "consensus_min_inlier_fraction": float(min_inlier_fraction),
            "consensus_min_scale_m": float(min_scale_m),
        })
    return pd.DataFrame(rows)


def summarize_robust_consensus(consensus: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for point_count, group in consensus.groupby("point_count", sort=True):
        e = pd.to_numeric(group["consensus_horizontal_error_m"], errors="coerce").to_numpy(dtype=float)
        e = e[np.isfinite(e)]
        inliers = pd.to_numeric(group["inlier_trial_count"], errors="coerce").to_numpy(dtype=float)
        outliers = pd.to_numeric(group["outlier_trial_count"], errors="coerce").to_numpy(dtype=float)
        rows.append({
            "point_count": int(point_count),
            "station_count": int(len(e)),
            "algorithm": "PN-MCVL-RC-v1.15 robust 10-trial consensus",
            "mean_error_m": float(np.mean(e)),
            "median_error_m": float(np.median(e)),
            "rmse_m": float(np.sqrt(np.mean(e ** 2))),
            "p90_error_m": float(np.percentile(e, 90)),
            "p95_error_m": float(np.percentile(e, 95)),
            "max_error_m": float(np.max(e)),
            "within_50m_percent": float(np.mean(e <= 50.0) * 100.0),
            "within_100m_percent": float(np.mean(e <= 100.0) * 100.0),
            "mean_inlier_trials": float(np.nanmean(inliers)),
            "mean_rejected_trials": float(np.nanmean(outliers)),
        })
    return pd.DataFrame(rows)


def make_ensemble_comparison(arithmetic: pd.DataFrame, robust: pd.DataFrame) -> pd.DataFrame:
    left = arithmetic.copy().add_prefix("arithmetic_")
    left = left.rename(columns={"arithmetic_point_count": "point_count"})
    right = robust.copy().add_prefix("robust_")
    right = right.rename(columns={"robust_point_count": "point_count"})
    out = pd.merge(left, right, on="point_count", how="outer").sort_values("point_count")
    if {"arithmetic_rmse_m", "robust_rmse_m"}.issubset(out.columns):
        out["rmse_change_robust_minus_arithmetic_m"] = out["robust_rmse_m"] - out["arithmetic_rmse_m"]
    if {"arithmetic_mean_error_m", "robust_mean_error_m"}.issubset(out.columns):
        out["mean_error_change_robust_minus_arithmetic_m"] = out["robust_mean_error_m"] - out["arithmetic_mean_error_m"]
    return out


def plot_robust_consensus_summary(summary: pd.DataFrame, output_root: Path, dpi: int) -> None:
    x = summary["point_count"].to_numpy(dtype=int)
    fig, ax = plt.subplots(figsize=(7.48, 5.2), dpi=int(dpi))
    for col, label in [("mean_error_m", "Mean"), ("median_error_m", "Median"), ("rmse_m", "RMSE"), ("p90_error_m", "P90")]:
        ax.plot(x, summary[col], marker="o", linewidth=1.35, label=label)
    ax.set_xlabel("Randomly selected receiver points per station")
    ax.set_ylabel("Robust consensus localization error [m]")
    ax.set_title("MCVL-RC robust 10-trial consensus")
    ax.set_xticks(x)
    ax.grid(True, alpha=0.28)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_root / "localization_random10_robust_consensus_accuracy.png", dpi=int(dpi), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_robust_consensus_locations(consensus: pd.DataFrame, all_results: pd.DataFrame, output_root: Path, dpi: int) -> None:
    for point_count, avg in consensus.groupby("point_count", sort=True):
        count_dir = output_root / f"points_{int(point_count):02d}" / "robust_consensus_over_10_trials"
        count_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(7.48, 5.8), dpi=int(dpi))
        for row in avg.itertuples(index=False):
            ax.plot([row.true_x_m, row.consensus_predicted_x_m], [row.true_y_m, row.consensus_predicted_y_m], linewidth=0.8, alpha=0.45)
        sc = ax.scatter(avg["consensus_predicted_x_m"], avg["consensus_predicted_y_m"], c=avg["consensus_horizontal_error_m"], s=62, edgecolors="black", linewidths=0.4, label="Robust consensus")
        ax.scatter(avg["true_x_m"], avg["true_y_m"], marker="x", s=56, c="black", linewidths=1.4, label="Ground truth")
        ax.set_xlabel("Blender X [m]")
        ax.set_ylabel("Blender Y [m]")
        ax.set_title(f"MCVL-RC robust consensus ({int(point_count)} points, 10 trials)")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=7.2)
        cbar = fig.colorbar(sc, ax=ax, pad=0.02)
        cbar.set_label("Consensus horizontal error [m]")
        fig.tight_layout()
        fig.savefig(count_dir / "all_stations_robust_consensus.png", dpi=int(dpi), bbox_inches="tight", facecolor="white")
        plt.close(fig)

        per_station_dir = count_dir / "per_station"
        per_station_dir.mkdir(exist_ok=True)
        raw_count = all_results[all_results["point_count"].eq(int(point_count))]
        for row in avg.itertuples(index=False):
            trials = raw_count[raw_count["station_id"].eq(int(row.station_id))].sort_values("trial_index").copy()
            keep_set = {int(v) for v in str(row.inlier_trial_indices).split(";") if str(v).strip()}
            is_inlier = trials["trial_index"].astype(int).isin(keep_set)
            fig, ax = plt.subplots(figsize=(6.2, 5.2), dpi=int(dpi))
            if np.any(is_inlier):
                ax.scatter(trials.loc[is_inlier, "predicted_x_m"], trials.loc[is_inlier, "predicted_y_m"], s=42, alpha=0.70, label="Consensus inliers")
            if np.any(~is_inlier):
                ax.scatter(trials.loc[~is_inlier, "predicted_x_m"], trials.loc[~is_inlier, "predicted_y_m"], marker="x", s=48, alpha=0.75, label="Rejected trial estimates")
            ax.scatter(row.consensus_predicted_x_m, row.consensus_predicted_y_m, marker="*", s=180, c="red", edgecolors="black", linewidths=0.7, label="Robust consensus")
            ax.scatter(row.true_x_m, row.true_y_m, marker="x", s=120, c="black", linewidths=1.8, label="Ground truth")
            ax.set_xlabel("Blender X [m]")
            ax.set_ylabel("Blender Y [m]")
            ax.set_title(f"Station {int(row.station_id)}: robust 10-trial consensus")
            ax.set_aspect("equal", adjustable="datalim")
            ax.grid(True, alpha=0.25)
            ax.legend(loc="best", fontsize=7.0)
            fig.tight_layout()
            fig.savefig(per_station_dir / f"station_{int(row.station_id):02d}_robust_consensus.png", dpi=int(dpi), bbox_inches="tight", facecolor="white")
            plt.close(fig)


def plot_summary(summary: pd.DataFrame, output_root: Path, dpi: int) -> None:
    x = summary["point_count"].to_numpy(dtype=int)
    fig, ax = plt.subplots(figsize=(7.48, 5.2), dpi=int(dpi))
    specs = [
        ("mean_error_m", "Mean"),
        ("median_error_m", "Median"),
        ("rmse_m", "RMSE"),
        ("p90_error_m", "P90"),
    ]
    for base, label in specs:
        mean = summary[f"{base}_mean_over_trials"].to_numpy(dtype=float)
        std = summary[f"{base}_std_over_trials"].to_numpy(dtype=float)
        ax.errorbar(x, mean, yerr=std, marker="o", linewidth=1.35, capsize=3, label=label)
    ax.set_xlabel("Randomly selected receiver points per station")
    ax.set_ylabel("Horizontal localization error [m]")
    ax.set_title("10-trial random-sampling localization")
    ax.set_xticks(x)
    ax.grid(True, alpha=0.28)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_root / "localization_random10_accuracy_mean_std.png", dpi=int(dpi), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_average_locations(station_average: pd.DataFrame, all_results: pd.DataFrame, output_root: Path, dpi: int) -> None:
    for point_count, avg in station_average.groupby("point_count", sort=True):
        count_dir = output_root / f"points_{int(point_count):02d}" / "average_over_10_trials"
        count_dir.mkdir(parents=True, exist_ok=True)

        # Overall map using the mean predicted coordinate of the 10 random trials.
        fig, ax = plt.subplots(figsize=(7.48, 5.8), dpi=int(dpi))
        for row in avg.itertuples(index=False):
            ax.plot([row.true_x_m, row.mean_predicted_x_m], [row.true_y_m, row.mean_predicted_y_m], linewidth=0.8, alpha=0.5)
        sc = ax.scatter(avg["mean_predicted_x_m"], avg["mean_predicted_y_m"], c=avg["mean_horizontal_error_m"], s=62, edgecolors="black", linewidths=0.4, label="Mean estimate")
        ax.scatter(avg["true_x_m"], avg["true_y_m"], marker="x", s=56, c="black", linewidths=1.4, label="Ground truth")
        ax.set_xlabel("Blender X [m]")
        ax.set_ylabel("Blender Y [m]")
        ax.set_title(f"Mean localization over 10 random trials ({int(point_count)} points)")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=7.2)
        cbar = fig.colorbar(sc, ax=ax, pad=0.02)
        cbar.set_label("Mean horizontal error over trials [m]")
        fig.tight_layout()
        fig.savefig(count_dir / "all_stations_mean_prediction.png", dpi=int(dpi), bbox_inches="tight", facecolor="white")
        plt.close(fig)

        # One clean station figure per station: the 10 trial estimates + mean + truth.
        per_station_dir = count_dir / "per_station"
        per_station_dir.mkdir(exist_ok=True)
        raw_count = all_results[all_results["point_count"].eq(int(point_count))]
        for row in avg.itertuples(index=False):
            trials = raw_count[raw_count["station_id"].eq(int(row.station_id))].sort_values("trial_index")
            fig, ax = plt.subplots(figsize=(6.2, 5.2), dpi=int(dpi))
            ax.scatter(trials["predicted_x_m"], trials["predicted_y_m"], s=42, alpha=0.65, label="10 trial estimates")
            ax.scatter(row.mean_predicted_x_m, row.mean_predicted_y_m, marker="*", s=180, c="red", edgecolors="black", linewidths=0.7, label="Mean estimate")
            ax.scatter(row.true_x_m, row.true_y_m, marker="x", s=120, c="black", linewidths=1.8, label="Ground truth")
            ax.set_xlabel("Blender X [m]")
            ax.set_ylabel("Blender Y [m]")
            ax.set_title(f"Station {int(row.station_id)}: 10 random localization trials")
            ax.set_aspect("equal", adjustable="datalim")
            ax.grid(True, alpha=0.25)
            ax.legend(loc="best", fontsize=7.0)
            fig.tight_layout()
            fig.savefig(per_station_dir / f"station_{int(row.station_id):02d}_random10_mean.png", dpi=int(dpi), bbox_inches="tight", facecolor="white")
            plt.close(fig)



def summarize_validation_diagnostics(all_results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for point_count, group in all_results.groupby("point_count", sort=True):
        cv = pd.to_numeric(group.get("cv_rmse_db", pd.Series(np.nan, index=group.index)), errors="coerce")
        gap = pd.to_numeric(group.get("candidate_score_gap", pd.Series(np.nan, index=group.index)), errors="coerce")
        spread = pd.to_numeric(group.get("top5_candidate_spread_m", pd.Series(np.nan, index=group.index)), errors="coerce")
        q = group.get("quality_flag", pd.Series("ok", index=group.index)).astype(str)
        rows.append({
            "point_count": int(point_count),
            "station_trial_rows": int(len(group)),
            "mean_cv_rmse_db": float(cv.mean()),
            "median_cv_rmse_db": float(cv.median()),
            "p90_cv_rmse_db": float(cv.quantile(0.90)),
            "median_candidate_score_gap": float(gap.median()),
            "median_top5_candidate_spread_m": float(spread.median()),
            "quality_flagged_percent": float((q != "ok").mean() * 100.0),
            "high_cv_error_percent": float((cv > 12.0).mean() * 100.0),
            "candidate_ambiguity_percent": float((gap < 0.35).mean() * 100.0),
        })
    return pd.DataFrame(rows)



def build_progressive_trajectory_ensemble(all_results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build a truth-free progressive trajectory ensemble.

    Within each station/trial, the estimate at point count n is the cumulative
    weighted mean of that trial's estimates from the minimum requested count up
    to n.  Weight equals point_count, so later estimates receive more influence
    while earlier nested-subset information is not discarded.  The ten trajectory
    states at each n are then averaged to produce the final station estimate.
    Ground truth is read only after fusion to calculate evaluation errors.
    """
    required = {
        "station_id", "point_count", "trial_index", "predicted_x_m",
        "predicted_y_m", "true_x_m", "true_y_m"
    }
    if not required.issubset(all_results.columns):
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    state_rows: list[dict[str, object]] = []
    for (station_id, trial_index), group in all_results.groupby(["station_id", "trial_index"], sort=True):
        g = group.sort_values("point_count").copy()
        cumulative_weight = 0.0
        cumulative_xy = np.zeros(2, dtype=float)
        for row in g.itertuples(index=False):
            xy = np.asarray([float(row.predicted_x_m), float(row.predicted_y_m)], dtype=float)
            weight = float(row.point_count)
            if not np.isfinite(xy).all() or weight <= 0:
                continue
            cumulative_xy += weight * xy
            cumulative_weight += weight
            trajectory_xy = cumulative_xy / cumulative_weight
            state_rows.append({
                "station_id": int(station_id),
                "trial_index": int(trial_index),
                "point_count": int(row.point_count),
                "trajectory_predicted_x_m": float(trajectory_xy[0]),
                "trajectory_predicted_y_m": float(trajectory_xy[1]),
                "cumulative_count_weight": float(cumulative_weight),
                "true_x_m": float(row.true_x_m),
                "true_y_m": float(row.true_y_m),
            })
    states = pd.DataFrame(state_rows)
    if states.empty:
        return states, pd.DataFrame(), pd.DataFrame()

    station_rows: list[dict[str, object]] = []
    for (point_count, station_id), group in states.groupby(["point_count", "station_id"], sort=True):
        px = pd.to_numeric(group["trajectory_predicted_x_m"], errors="coerce").to_numpy(float)
        py = pd.to_numeric(group["trajectory_predicted_y_m"], errors="coerce").to_numpy(float)
        valid = np.isfinite(px) & np.isfinite(py)
        if not np.any(valid):
            continue
        x = float(np.mean(px[valid]))
        y = float(np.mean(py[valid]))
        tx = float(group["true_x_m"].iloc[0])
        ty = float(group["true_y_m"].iloc[0])
        error = float(np.hypot(x - tx, y - ty))
        station_rows.append({
            "point_count": int(point_count),
            "station_id": int(station_id),
            "trajectory_count": int(np.sum(valid)),
            "progressive_predicted_x_m": x,
            "progressive_predicted_y_m": y,
            "true_x_m": tx,
            "true_y_m": ty,
            "horizontal_error_m": error,
            "fusion_algorithm": "PN-MCVL-RC-v1.15 point-count-weighted progressive trajectory mean across 10 random trials",
        })
    station = pd.DataFrame(station_rows)

    summary_rows: list[dict[str, object]] = []
    for point_count, group in station.groupby("point_count", sort=True):
        e = pd.to_numeric(group["horizontal_error_m"], errors="coerce").to_numpy(float)
        e = e[np.isfinite(e)]
        if len(e) == 0:
            continue
        summary_rows.append({
            "point_count": int(point_count),
            "station_count": int(len(e)),
            "mean_error_m": float(np.mean(e)),
            "median_error_m": float(np.median(e)),
            "rmse_m": float(np.sqrt(np.mean(e ** 2))),
            "p90_error_m": float(np.percentile(e, 90)),
            "p95_error_m": float(np.percentile(e, 95)),
            "max_error_m": float(np.max(e)),
            "within_50m_percent": float(np.mean(e <= 50.0) * 100.0),
            "within_100m_percent": float(np.mean(e <= 100.0) * 100.0),
            "algorithm": "PN-MCVL-RC-v1.15 progressive nested 10-trial trajectory ensemble",
        })
    summary = pd.DataFrame(summary_rows).sort_values("point_count").reset_index(drop=True)
    if len(summary):
        summary["rmse_delta_vs_previous_m"] = summary["rmse_m"].diff()
        summary["rmse_nonincreasing_vs_previous"] = summary["rmse_delta_vs_previous_m"].le(1e-9) | summary["rmse_delta_vs_previous_m"].isna()
        summary["mean_delta_vs_previous_m"] = summary["mean_error_m"].diff()
    return states, station, summary

def summarize_progressive_updates(all_results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if "progressive_update_accepted" not in all_results.columns:
        return pd.DataFrame(rows)
    for point_count, group in all_results.groupby("point_count", sort=True):
        available = group[group.get("previous_solution_available", False).astype(bool)] if "previous_solution_available" in group.columns else group.iloc[0:0]
        if len(available) == 0:
            rows.append({
                "point_count": int(point_count),
                "station_trial_count": int(len(group)),
                "previous_solution_available_count": 0,
                "progressive_update_accepted_percent": float("nan"),
                "progressive_previous_retained_percent": float("nan"),
                "median_progressive_score_improvement_db": float("nan"),
                "median_progressive_jump_m": float("nan"),
                "p90_progressive_jump_m": float("nan"),
            })
            continue
        accepted = available["progressive_update_accepted"].astype(bool).to_numpy()
        imp = pd.to_numeric(available.get("progressive_score_improvement_db"), errors="coerce").to_numpy(float)
        jump = pd.to_numeric(available.get("progressive_jump_m"), errors="coerce").to_numpy(float)
        finite_imp = imp[np.isfinite(imp)]
        finite_jump = jump[np.isfinite(jump)]
        rows.append({
            "point_count": int(point_count),
            "station_trial_count": int(len(group)),
            "previous_solution_available_count": int(len(available)),
            "progressive_update_accepted_percent": float(np.mean(accepted) * 100.0),
            "progressive_previous_retained_percent": float((1.0 - np.mean(accepted)) * 100.0),
            "median_progressive_score_improvement_db": float(np.median(finite_imp)) if len(finite_imp) else float("nan"),
            "median_progressive_jump_m": float(np.median(finite_jump)) if len(finite_jump) else float("nan"),
            "p90_progressive_jump_m": float(np.percentile(finite_jump, 90)) if len(finite_jump) else float("nan"),
        })
    return pd.DataFrame(rows)


def audit_nested_random_selection(output_root: Path, counts: list[int], random_trials: int) -> pd.DataFrame:
    """Verify S_k is an exact prefix of S_{k+1} for every station and trial."""
    rows: list[dict[str, object]] = []
    for trial_index in range(1, int(random_trials) + 1):
        previous: pd.DataFrame | None = None
        previous_count: int | None = None
        for point_count in counts:
            path = output_root / f"points_{int(point_count):02d}" / f"trial_{trial_index:02d}" / f"selected_{int(point_count)}_random_points_all_stations.csv"
            if not path.is_file():
                rows.append({
                    "trial_index": int(trial_index), "point_count": int(point_count),
                    "previous_point_count": previous_count, "nested_prefix_verified": False,
                    "reason": "selected_points_csv_missing",
                })
                previous = None
                previous_count = int(point_count)
                continue
            current = pd.read_csv(path, encoding="utf-8-sig")
            if previous is None:
                rows.append({
                    "trial_index": int(trial_index), "point_count": int(point_count),
                    "previous_point_count": None, "nested_prefix_verified": True,
                    "reason": "initial_count",
                })
            else:
                ok = True
                bad_stations: list[str] = []
                for sid in sorted(set(previous["station_id"].astype(int))):
                    a = previous[previous.station_id.astype(int).eq(sid)].sort_values("selection_rank")
                    b = current[current.station_id.astype(int).eq(sid)].sort_values("selection_rank")
                    if len(b) < len(a):
                        ok = False; bad_stations.append(str(sid)); continue
                    cols = [c for c in ["x_m", "y_m"] if c in a.columns and c in b.columns]
                    if len(cols) != 2:
                        ok = False; bad_stations.append(str(sid)); continue
                    av = a[cols].to_numpy(float)
                    bv = b.iloc[: len(a)][cols].to_numpy(float)
                    if av.shape != bv.shape or not np.allclose(av, bv, atol=1e-9, rtol=0.0):
                        ok = False; bad_stations.append(str(sid))
                rows.append({
                    "trial_index": int(trial_index), "point_count": int(point_count),
                    "previous_point_count": int(previous_count) if previous_count is not None else None,
                    "nested_prefix_verified": bool(ok),
                    "reason": "ok" if ok else "prefix_mismatch_station_" + ",".join(bad_stations),
                })
            previous = current
            previous_count = int(point_count)
    return pd.DataFrame(rows)

def main() -> int:
    args = parse_args()
    if int(args.random_trials) <= 0:
        raise ValueError("--random-trials必须为正整数")
    project_root = args.project_root.expanduser().resolve()
    counts = parse_counts(args.point_counts)
    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root is not None
        else project_root / "outputs" / "localization_random10_10to15_progressive_nested_mcvl_rc_v115"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    all_result_frames: list[pd.DataFrame] = []
    simulation_ablation_frames: list[pd.DataFrame] = []
    without_simulation_result_frames: list[pd.DataFrame] = []
    with_simulation_result_frames: list[pd.DataFrame] = []
    trial_metric_rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []

    print("=" * 88)
    print("Progressive Nested Monte-Carlo MCVL-RC-v1.15 localization")
    print(f"Point counts: {counts}")
    print(f"Random trial rankings: {args.random_trials}")
    print("Selection: one uniform random permutation per station/trial; 10--15 use nested prefixes")
    print("Progressive solver: previous-count location is inherited; weakly supported jumps are rejected without using truth")
    print(f"Output: {output_root}")
    print("=" * 88)

    for point_count in counts:
        for trial_index in range(1, int(args.random_trials) + 1):
            # The same trial seed is deliberately reused across all point counts.
            # The single-count solver generates a random permutation and takes its
            # first k entries, which makes 10--15 strictly nested for each station.
            trial_seed = int(args.random_seed) + int(trial_index) * 10_007
            trial_dir = output_root / f"points_{int(point_count):02d}" / f"trial_{trial_index:02d}"
            previous_result_path = None
            count_index = counts.index(point_count)
            if count_index > 0:
                previous_count = counts[count_index - 1]
                previous_dir = output_root / f"points_{int(previous_count):02d}" / f"trial_{trial_index:02d}"
                try:
                    previous_result_path = _find_result_csv(previous_dir, previous_count)
                except FileNotFoundError:
                    previous_result_path = None
            command: list[object] = [
                sys.executable, SINGLE_SCRIPT,
                "--project-root", project_root,
                "--output-dir", trial_dir,
                "--points-per-station", int(point_count),
                "--random-seed", int(trial_seed),
                "--bootstrap", int(args.bootstrap),
                "--de-maxiter", int(args.de_maxiter),
                "--de-popsize", int(args.de_popsize),
                "--direction-prior-mode", args.direction_prior_mode,
                "--station-ids", args.station_ids,
                "--simulation-mode", args.simulation_mode,
                "--simulation-weight", float(args.simulation_weight),
                "--progressive-min-improvement-db", float(args.progressive_min_improvement_db),
                "--x-min", args.x_min, "--x-max", args.x_max,
                "--y-min", args.y_min, "--y-max", args.y_max,
            ]
            if previous_result_path is not None:
                if args.simulation_mode == "compare":
                    previous_without = previous_dir / "localization_without_simulation_results.csv"
                    previous_with = previous_dir / "localization_with_simulation_results.csv"
                    if previous_without.is_file():
                        command += ["--previous-without-simulation-results", previous_without]
                    if previous_with.is_file():
                        command += ["--previous-with-simulation-results", previous_with]
                else:
                    command += ["--previous-results", previous_result_path]
            if args.measurements is not None:
                command += ["--measurements", args.measurements.expanduser().resolve()]
            if args.directions is not None:
                command += ["--directions", args.directions.expanduser().resolve()]
            if args.simulation_root is not None:
                command += ["--simulation-root", args.simulation_root.expanduser().resolve()]
            if args.strict_simulation_data:
                command.append("--strict-simulation-data")
            if not args.keep_trial_figures:
                command.append("--skip-figures")

            parent_text = str(previous_result_path) if previous_result_path is not None else "none (initial count)"
            resume_complete = False
            if args.resume:
                try:
                    existing_result_path = _find_result_csv(trial_dir, point_count)
                except FileNotFoundError:
                    existing_result_path = None
                required_paths = [
                    existing_result_path,
                    trial_dir / f"selected_{int(point_count)}_random_points_all_stations.csv",
                ]
                if args.simulation_mode == "compare":
                    required_paths.extend([
                        trial_dir / "localization_without_simulation_results.csv",
                        trial_dir / "localization_with_simulation_results.csv",
                        trial_dir / "localization_simulation_ablation_by_station.csv",
                    ])
                resume_complete = bool(existing_result_path is not None and all(
                    path is not None and Path(path).is_file() for path in required_paths
                ))

            print(f"\n[POINTS={point_count} TRIAL={trial_index}/{args.random_trials}] nested-seed={trial_seed}")
            print(f"[PREVIOUS] {parent_text}")
            if resume_complete:
                print(f"[RESUME] reusing complete paired trial: {trial_dir}", flush=True)
            else:
                print("[RUN]", _display_command(command), flush=True)
                completed = subprocess.run([str(v) for v in command], cwd=project_root, check=False)
                if completed.returncode != 0:
                    failures.append({"point_count": point_count, "trial_index": trial_index, "trial_seed": trial_seed, "returncode": int(completed.returncode)})
                    if not args.continue_on_error:
                        return int(completed.returncode)
                    continue

            result_path = _find_result_csv(trial_dir, point_count)
            frame = pd.read_csv(result_path, encoding="utf-8-sig")
            frame["point_count"] = int(point_count)
            frame["trial_index"] = int(trial_index)
            frame["trial_seed"] = int(trial_seed)
            all_result_frames.append(frame)
            trial_metric_rows.append(_metrics_from_result(frame, point_count, trial_index, trial_seed))
            ablation_path = trial_dir / "localization_simulation_ablation_by_station.csv"
            if ablation_path.is_file():
                ablation_frame = pd.read_csv(ablation_path, encoding="utf-8-sig")
                ablation_frame["point_count"] = int(point_count)
                ablation_frame["trial_index"] = int(trial_index)
                ablation_frame["trial_seed"] = int(trial_seed)
                simulation_ablation_frames.append(ablation_frame)
            if args.simulation_mode == "compare":
                for branch_name, branch_frames in (
                    ("without", without_simulation_result_frames),
                    ("with", with_simulation_result_frames),
                ):
                    branch_path = trial_dir / f"localization_{branch_name}_simulation_results.csv"
                    if not branch_path.is_file():
                        raise FileNotFoundError(f"缺少配对定位分支结果：{branch_path}")
                    branch_frame = pd.read_csv(branch_path, encoding="utf-8-sig")
                    branch_frame["point_count"] = int(point_count)
                    branch_frame["trial_index"] = int(trial_index)
                    branch_frame["trial_seed"] = int(trial_seed)
                    branch_frames.append(branch_frame)

    if not all_result_frames:
        raise RuntimeError("没有成功完成任何随机定位trial")

    all_results = pd.concat(all_result_frames, ignore_index=True)
    trial_metrics = pd.DataFrame(trial_metric_rows).sort_values(["point_count", "trial_index"]).reset_index(drop=True)
    summary = summarize_trials(trial_metrics)
    station_average = summarize_station_trials(all_results)
    ensemble_summary = summarize_mean_prediction_ensemble(station_average)
    validation_summary = summarize_validation_diagnostics(all_results)
    robust_consensus = build_robust_consensus(
        all_results,
        mad_z=float(args.consensus_mad_z),
        min_inlier_fraction=float(args.consensus_min_inlier_fraction),
        min_scale_m=float(args.consensus_min_scale_m),
    )
    robust_consensus_summary = summarize_robust_consensus(robust_consensus)
    ensemble_comparison = make_ensemble_comparison(ensemble_summary, robust_consensus_summary)

    paired_robust_summary = pd.DataFrame()
    paired_progressive_summary = pd.DataFrame()
    without_robust_consensus = pd.DataFrame()
    with_robust_consensus = pd.DataFrame()
    without_progressive_states = pd.DataFrame()
    without_progressive_station = pd.DataFrame()
    with_progressive_states = pd.DataFrame()
    with_progressive_station = pd.DataFrame()
    if without_simulation_result_frames and with_simulation_result_frames:
        without_all_results = pd.concat(without_simulation_result_frames, ignore_index=True)
        with_all_results = pd.concat(with_simulation_result_frames, ignore_index=True)
        consensus_kwargs = {
            "mad_z": float(args.consensus_mad_z),
            "min_inlier_fraction": float(args.consensus_min_inlier_fraction),
            "min_scale_m": float(args.consensus_min_scale_m),
        }
        without_robust_consensus = build_robust_consensus(without_all_results, **consensus_kwargs)
        with_robust_consensus = build_robust_consensus(with_all_results, **consensus_kwargs)
        without_robust_summary = summarize_robust_consensus(without_robust_consensus).add_suffix("_measured_only")
        without_robust_summary = without_robust_summary.rename(columns={"point_count_measured_only": "point_count"})
        with_robust_summary = summarize_robust_consensus(with_robust_consensus).add_suffix("_measured_plus_simulation")
        with_robust_summary = with_robust_summary.rename(columns={"point_count_measured_plus_simulation": "point_count"})
        paired_robust_summary = pd.merge(without_robust_summary, with_robust_summary, on="point_count", how="inner")
        for metric in ("mean_error_m", "median_error_m", "rmse_m", "p90_error_m", "within_100m_percent"):
            baseline_col = f"{metric}_measured_only"
            joint_col = f"{metric}_measured_plus_simulation"
            if baseline_col in paired_robust_summary and joint_col in paired_robust_summary:
                paired_robust_summary[f"{metric}_delta_joint_minus_measured"] = paired_robust_summary[joint_col] - paired_robust_summary[baseline_col]
        without_progressive_states, without_progressive_station, without_progressive_summary = build_progressive_trajectory_ensemble(without_all_results)
        with_progressive_states, with_progressive_station, with_progressive_summary = build_progressive_trajectory_ensemble(with_all_results)
        without_progressive_summary = without_progressive_summary.add_suffix("_measured_only").rename(
            columns={"point_count_measured_only": "point_count"}
        )
        with_progressive_summary = with_progressive_summary.add_suffix("_measured_plus_simulation").rename(
            columns={"point_count_measured_plus_simulation": "point_count"}
        )
        paired_progressive_summary = pd.merge(
            without_progressive_summary, with_progressive_summary, on="point_count", how="inner"
        )
        for metric in ("mean_error_m", "median_error_m", "rmse_m", "p90_error_m", "within_100m_percent"):
            baseline_col = f"{metric}_measured_only"
            joint_col = f"{metric}_measured_plus_simulation"
            if baseline_col in paired_progressive_summary and joint_col in paired_progressive_summary:
                paired_progressive_summary[f"{metric}_delta_joint_minus_measured"] = paired_progressive_summary[joint_col] - paired_progressive_summary[baseline_col]
    progressive_trajectory_states, progressive_trajectory_station, progressive_trajectory_summary = build_progressive_trajectory_ensemble(all_results)
    progressive_update_summary = summarize_progressive_updates(all_results)
    nested_selection_audit = audit_nested_random_selection(output_root, counts, int(args.random_trials))

    all_results.to_csv(output_root / "localization_random10_all_trials_station_results.csv", index=False, encoding="utf-8-sig")
    trial_metrics.to_csv(output_root / "localization_random10_trial_metrics.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output_root / "localization_random10_summary_10to15.csv", index=False, encoding="utf-8-sig")
    station_average.to_csv(output_root / "localization_random10_station_average.csv", index=False, encoding="utf-8-sig")
    ensemble_summary.to_csv(output_root / "localization_random10_mean_prediction_ensemble_summary.csv", index=False, encoding="utf-8-sig")
    validation_summary.to_csv(output_root / "localization_random10_validation_summary.csv", index=False, encoding="utf-8-sig")
    robust_consensus.to_csv(output_root / "localization_random10_robust_consensus_station_results.csv", index=False, encoding="utf-8-sig")
    robust_consensus_summary.to_csv(output_root / "localization_random10_robust_consensus_summary.csv", index=False, encoding="utf-8-sig")
    if len(paired_robust_summary):
        without_all_results.to_csv(output_root / "localization_measured_only_all_trials_station_results.csv", index=False, encoding="utf-8-sig")
        with_all_results.to_csv(output_root / "localization_measured_plus_simulation_all_trials_station_results.csv", index=False, encoding="utf-8-sig")
        without_robust_consensus.to_csv(output_root / "localization_measured_only_robust_consensus_station_results.csv", index=False, encoding="utf-8-sig")
        with_robust_consensus.to_csv(output_root / "localization_measured_plus_simulation_robust_consensus_station_results.csv", index=False, encoding="utf-8-sig")
        paired_robust_summary.to_csv(output_root / "localization_measured_vs_measured_plus_simulation_robust_summary.csv", index=False, encoding="utf-8-sig")
        without_progressive_states.to_csv(output_root / "localization_measured_only_progressive_trial_states.csv", index=False, encoding="utf-8-sig")
        without_progressive_station.to_csv(output_root / "localization_measured_only_progressive_station_results.csv", index=False, encoding="utf-8-sig")
        with_progressive_states.to_csv(output_root / "localization_measured_plus_simulation_progressive_trial_states.csv", index=False, encoding="utf-8-sig")
        with_progressive_station.to_csv(output_root / "localization_measured_plus_simulation_progressive_station_results.csv", index=False, encoding="utf-8-sig")
        paired_progressive_summary.to_csv(output_root / "localization_measured_vs_measured_plus_simulation_progressive_summary.csv", index=False, encoding="utf-8-sig")
    ensemble_comparison.to_csv(output_root / "localization_random10_ensemble_comparison.csv", index=False, encoding="utf-8-sig")
    progressive_update_summary.to_csv(output_root / "localization_progressive_update_summary.csv", index=False, encoding="utf-8-sig")
    if simulation_ablation_frames:
        simulation_ablation = pd.concat(simulation_ablation_frames, ignore_index=True)
        simulation_ablation.to_csv(
            output_root / "localization_simulation_ablation_all_trials.csv",
            index=False, encoding="utf-8-sig",
        )
        simulation_summary_rows: list[dict[str, float | int]] = []
        for point_count, group in simulation_ablation.groupby("point_count", sort=True):
            without_error = group["horizontal_error_m_without_simulation"].to_numpy(float)
            with_error = group["horizontal_error_m_with_simulation"].to_numpy(float)
            simulation_summary_rows.append({
                "point_count": int(point_count),
                "station_trial_count": int(len(group)),
                "rmse_without_simulation_m": float(np.sqrt(np.mean(without_error ** 2))),
                "rmse_with_simulation_m": float(np.sqrt(np.mean(with_error ** 2))),
                "mean_error_without_simulation_m": float(np.mean(without_error)),
                "mean_error_with_simulation_m": float(np.mean(with_error)),
                "mean_error_delta_with_minus_without_m": float(np.mean(with_error - without_error)),
                "median_error_without_simulation_m": float(np.median(without_error)),
                "median_error_with_simulation_m": float(np.median(with_error)),
                "p90_error_without_simulation_m": float(np.percentile(without_error, 90)),
                "p90_error_with_simulation_m": float(np.percentile(with_error, 90)),
                "within_100m_without_simulation_percent": float(np.mean(without_error <= 100.0) * 100.0),
                "within_100m_with_simulation_percent": float(np.mean(with_error <= 100.0) * 100.0),
                "simulation_improved_percent": float(np.mean(with_error < without_error) * 100.0),
            })
        simulation_ablation_summary = pd.DataFrame(simulation_summary_rows)
        simulation_ablation_summary.to_csv(
            output_root / "localization_simulation_ablation_summary_by_point_count.csv",
            index=False, encoding="utf-8-sig",
        )
        if not args.skip_figures:
            fig, ax = plt.subplots(figsize=(7.48, 5.2), dpi=int(args.dpi))
            ax.plot(simulation_ablation_summary["point_count"], simulation_ablation_summary["rmse_without_simulation_m"], "o-", label="Measured only")
            ax.plot(simulation_ablation_summary["point_count"], simulation_ablation_summary["rmse_with_simulation_m"], "s-", label="Measured + collocated simulation")
            ax.set_xlabel("Measured receiver locations per station")
            ax.set_ylabel("Localization RMSE [m]")
            ax.set_title("Base-station localization: measured only vs measured + simulation")
            ax.grid(True, alpha=0.28)
            ax.legend(loc="best")
            fig.tight_layout()
            fig.savefig(output_root / "localization_simulation_ablation_rmse_by_point_count.png", dpi=int(args.dpi), bbox_inches="tight", facecolor="white")
            plt.close(fig)
    if len(paired_robust_summary) and not args.skip_figures:
        fig, ax = plt.subplots(figsize=(7.48, 5.2), dpi=int(args.dpi))
        x = paired_robust_summary["point_count"].to_numpy(dtype=int)
        ax.plot(x, paired_robust_summary["rmse_m_measured_only"], "o-", label="Measured only")
        ax.plot(x, paired_robust_summary["rmse_m_measured_plus_simulation"], "s-", label="Measured + collocated simulation")
        ax.set_xlabel("Receiver locations per station")
        ax.set_ylabel("Robust-consensus localization RMSE [m]")
        ax.set_title("Paired localization experiment over the same random samples")
        ax.grid(True, alpha=0.28)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(output_root / "localization_measured_vs_measured_plus_simulation_robust_rmse.png", dpi=int(args.dpi), bbox_inches="tight", facecolor="white")
        plt.close(fig)
    if len(paired_progressive_summary) and not args.skip_figures:
        fig, ax = plt.subplots(figsize=(7.48, 5.2), dpi=int(args.dpi))
        x = paired_progressive_summary["point_count"].to_numpy(dtype=int)
        ax.plot(x, paired_progressive_summary["rmse_m_measured_only"], "o-", label="Measured only")
        ax.plot(x, paired_progressive_summary["rmse_m_measured_plus_simulation"], "s-", label="Measured + collocated simulation")
        ax.set_xlabel("Receiver locations per station")
        ax.set_ylabel("Progressive-trajectory localization RMSE [m]")
        ax.set_title("Paired localization experiment over the same random samples")
        ax.grid(True, alpha=0.28)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(output_root / "localization_measured_vs_measured_plus_simulation_progressive_rmse.png", dpi=int(args.dpi), bbox_inches="tight", facecolor="white")
        plt.close(fig)
    nested_selection_audit.to_csv(output_root / "localization_nested_selection_audit.csv", index=False, encoding="utf-8-sig")
    progressive_trajectory_states.to_csv(output_root / "localization_progressive_trajectory_trial_states.csv", index=False, encoding="utf-8-sig")
    progressive_trajectory_station.to_csv(output_root / "localization_progressive_trajectory_station_results.csv", index=False, encoding="utf-8-sig")
    progressive_trajectory_summary.to_csv(output_root / "localization_progressive_trajectory_summary.csv", index=False, encoding="utf-8-sig")
    if len(robust_consensus_summary):
        best_idx = pd.to_numeric(robust_consensus_summary["rmse_m"], errors="coerce").idxmin()
        robust_consensus_summary.loc[[best_idx]].to_csv(output_root / "localization_random10_best_point_count_by_robust_rmse.csv", index=False, encoding="utf-8-sig")
    if len(progressive_trajectory_summary):
        best_idx = pd.to_numeric(progressive_trajectory_summary["rmse_m"], errors="coerce").idxmin()
        progressive_trajectory_summary.loc[[best_idx]].to_csv(output_root / "localization_progressive_best_point_count.csv", index=False, encoding="utf-8-sig")
    if failures:
        pd.DataFrame(failures).to_csv(output_root / "failed_random_trials.csv", index=False, encoding="utf-8-sig")

    if not args.skip_figures:
        plot_summary(summary, output_root, dpi=int(args.dpi))
        plot_average_locations(station_average, all_results, output_root, dpi=int(args.dpi))
        plot_robust_consensus_summary(robust_consensus_summary, output_root, dpi=int(args.dpi))
        plot_robust_consensus_locations(robust_consensus, all_results, output_root, dpi=int(args.dpi))

    print("\n随机10次定位完成。")
    print("Trial指标：", output_root / "localization_random10_trial_metrics.csv")
    print("10次平均/标准差：", output_root / "localization_random10_summary_10to15.csv")
    print("逐站10次平均：", output_root / "localization_random10_station_average.csv")
    print("10次坐标平均后的总体精度：", output_root / "localization_random10_mean_prediction_ensemble_summary.csv")
    print("候选交叉验证诊断：", output_root / "localization_random10_validation_summary.csv")
    print("v1.15逐站鲁棒共识：", output_root / "localization_random10_robust_consensus_station_results.csv")
    print("v1.15鲁棒共识总体精度：", output_root / "localization_random10_robust_consensus_summary.csv")
    if len(paired_robust_summary):
        print("仅实测 vs 实测+仿真配对鲁棒结果：", output_root / "localization_measured_vs_measured_plus_simulation_robust_summary.csv")
        print("仅实测 vs 实测+仿真配对渐进轨迹主结果：", output_root / "localization_measured_vs_measured_plus_simulation_progressive_summary.csv")
    print("算术平均与鲁棒共识对比：", output_root / "localization_random10_ensemble_comparison.csv")
    print("渐进式更新统计：", output_root / "localization_progressive_update_summary.csv")
    print("嵌套随机选点审计：", output_root / "localization_nested_selection_audit.csv")
    print("v1.15渐进轨迹总体精度（主结果）：", output_root / "localization_progressive_trajectory_summary.csv")
    print("v1.15渐进轨迹逐站结果：", output_root / "localization_progressive_trajectory_station_results.csv")
    if len(nested_selection_audit) and not bool(nested_selection_audit["nested_prefix_verified"].all()):
        print("WARNING: 检测到嵌套选点前缀不一致，请检查audit CSV。")
    else:
        print("Nested selection audit: all requested 10--15 point prefixes verified.")
    if len(progressive_trajectory_summary) > 1 and not bool(progressive_trajectory_summary["rmse_nonincreasing_vs_previous"].all()):
        print("NOTE: 原始真值评估RMSE仍存在局部非单调；程序不会使用真值强制修改结果。请查看progressive summary。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
