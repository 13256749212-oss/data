# -*- coding: utf-8 -*-
"""
Compare the saved 4000 m x 3000 m joint best-server radio map with
coordinate-aligned vehicle measurements.

This script does NOT run Sionna RT. It only reads:
  1) outputs/joint_best_server_4000x3000/
       joint_best_server_27stations_4000x3000.npz
  2) data/processed/cell_pci_rsrp_long_27stations.csv

Fair comparison rule
--------------------
The saved joint map is a best-server envelope over all 27 stations / 79 PCIs.
Therefore, at every receiver measurement instance, this script first selects
the strongest measured target PCI. It then optionally aggregates duplicate
measurement instances falling in the same 1 m map cell by the median. The
primary RMSE is computed from untruncated dBm values at the same XY cells.
Display clipping is used only for figures.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_MAP_REL = Path(
    "outputs/joint_best_server_4000x3000/"
    "joint_best_server_27stations_4000x3000.npz"
)
DEFAULT_MEASUREMENTS_REL = Path(
    "data/processed/cell_pci_rsrp_long_27stations.csv"
)
DEFAULT_OUTPUT_REL = Path(
    "outputs/joint_map_measurement_comparison"
)


@dataclass(frozen=True)
class MapData:
    best_rsrp_dbm: np.ndarray
    x_centers_m: np.ndarray
    y_centers_m: np.ndarray
    best_station_id: np.ndarray | None
    best_pci: np.ndarray | None
    outdoor_valid_mask: np.ndarray | None
    building_mask: np.ndarray | None
    metadata: dict[str, Any]


class ComparisonError(RuntimeError):
    """Readable error for project-data problems."""


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).replace("\ufeff", "").strip() for c in out.columns]
    return out


def _first_existing(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    available = set(columns)
    for name in candidates:
        if name in available:
            return name
    return None


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _resolve_path(project_root: Path, value: str | Path | None, default_rel: Path) -> Path:
    if value is None:
        return (project_root / default_rel).resolve()
    path = Path(value)
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def _load_metadata(npz: Any) -> dict[str, Any]:
    if "metadata_json" not in npz.files:
        return {}
    try:
        raw = npz["metadata_json"]
        if np.size(raw) == 0:
            return {}
        return json.loads(str(np.ravel(raw)[0]))
    except Exception:
        return {}


def load_joint_map(path: Path) -> MapData:
    if not path.is_file():
        raise ComparisonError(f"找不到联合无线电地图NPZ: {path}")

    try:
        with np.load(path, allow_pickle=False) as z:
            required = {"best_rsrp_dbm", "x_centers_m", "y_centers_m"}
            missing = sorted(required.difference(z.files))
            if missing:
                raise ComparisonError(
                    f"联合地图NPZ缺少字段: {missing}; 实际字段: {z.files}"
                )

            rsrp = np.asarray(z["best_rsrp_dbm"], dtype=np.float32)
            x = np.asarray(z["x_centers_m"], dtype=np.float64).reshape(-1)
            y = np.asarray(z["y_centers_m"], dtype=np.float64).reshape(-1)
            station = (
                np.asarray(z["best_station_id"], dtype=np.int32)
                if "best_station_id" in z.files
                else None
            )
            pci = (
                np.asarray(z["best_pci"], dtype=np.int32)
                if "best_pci" in z.files
                else None
            )
            outdoor = (
                np.asarray(z["outdoor_valid_mask"], dtype=bool)
                if "outdoor_valid_mask" in z.files
                else None
            )
            building = (
                np.asarray(z["building_mask"], dtype=bool)
                if "building_mask" in z.files
                else None
            )
            metadata = _load_metadata(z)
    except ComparisonError:
        raise
    except Exception as exc:
        raise ComparisonError(f"读取联合地图NPZ失败: {exc}") from exc

    if rsrp.ndim != 2:
        raise ComparisonError(f"best_rsrp_dbm必须为二维数组，实际shape={rsrp.shape}")
    if rsrp.shape != (len(y), len(x)):
        raise ComparisonError(
            "地图数组与坐标轴不匹配: "
            f"map={rsrp.shape}, len(y)={len(y)}, len(x)={len(x)}"
        )
    for name, arr in (
        ("best_station_id", station),
        ("best_pci", pci),
        ("outdoor_valid_mask", outdoor),
        ("building_mask", building),
    ):
        if arr is not None and arr.shape != rsrp.shape:
            raise ComparisonError(f"{name} shape={arr.shape} 与地图shape={rsrp.shape}不一致")

    # Convert descending axes into ascending axes so that searchsorted is reliable.
    if len(x) > 1 and x[1] < x[0]:
        x = x[::-1].copy()
        rsrp = rsrp[:, ::-1].copy()
        station = station[:, ::-1].copy() if station is not None else None
        pci = pci[:, ::-1].copy() if pci is not None else None
        outdoor = outdoor[:, ::-1].copy() if outdoor is not None else None
        building = building[:, ::-1].copy() if building is not None else None
    if len(y) > 1 and y[1] < y[0]:
        y = y[::-1].copy()
        rsrp = rsrp[::-1, :].copy()
        station = station[::-1, :].copy() if station is not None else None
        pci = pci[::-1, :].copy() if pci is not None else None
        outdoor = outdoor[::-1, :].copy() if outdoor is not None else None
        building = building[::-1, :].copy() if building is not None else None

    if len(x) < 2 or len(y) < 2:
        raise ComparisonError("地图坐标轴长度不足，无法进行空间匹配")
    if not (np.all(np.diff(x) > 0) and np.all(np.diff(y) > 0)):
        raise ComparisonError("x_centers_m或y_centers_m不是严格单调坐标轴")

    return MapData(
        best_rsrp_dbm=rsrp,
        x_centers_m=x,
        y_centers_m=y,
        best_station_id=station,
        best_pci=pci,
        outdoor_valid_mask=outdoor,
        building_mask=building,
        metadata=metadata,
    )


def _nearest_indices(axis: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return nearest axis index and whether values are inside cell-center bounds."""
    idx_right = np.searchsorted(axis, values, side="left")
    idx_right = np.clip(idx_right, 0, len(axis) - 1)
    idx_left = np.clip(idx_right - 1, 0, len(axis) - 1)
    choose_left = np.abs(values - axis[idx_left]) <= np.abs(values - axis[idx_right])
    idx = np.where(choose_left, idx_left, idx_right).astype(np.int64)

    dx = float(np.median(np.diff(axis)))
    low = float(axis[0] - dx / 2.0)
    high = float(axis[-1] + dx / 2.0)
    inside = np.isfinite(values) & (values >= low) & (values <= high)
    return idx, inside


def read_measurement_best_server(path: Path, args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not path.is_file():
        raise ComparisonError(f"找不到坐标转换后的27站实测长表: {path}")

    try:
        raw = _clean_columns(pd.read_csv(path, low_memory=False))
    except Exception as exc:
        raise ComparisonError(f"读取实测CSV失败: {exc}") from exc

    x_col = _first_existing(raw.columns, ["blender_x", "x_m", "x", "X"])
    y_col = _first_existing(raw.columns, ["blender_y", "y_m", "y", "Y"])
    rsrp_col = _first_existing(
        raw.columns, ["measured_rsrp_dbm", "rsrp_dbm", "NR5G SS RSRP"]
    )
    pci_col = _first_existing(raw.columns, ["pci", "PCI", "NR5G PCI"])
    station_col = _first_existing(raw.columns, ["station_id", "StationID"])

    missing = [
        label
        for label, col in (
            ("Blender X", x_col),
            ("Blender Y", y_col),
            ("RSRP", rsrp_col),
            ("PCI", pci_col),
            ("station_id", station_col),
        )
        if col is None
    ]
    if missing:
        raise ComparisonError(
            f"实测长表缺少关键字段: {missing}; 实际字段: {list(raw.columns)}"
        )

    data = raw.copy()
    data["_x"] = _to_numeric(data[x_col])
    data["_y"] = _to_numeric(data[y_col])
    data["_rsrp"] = _to_numeric(data[rsrp_col])
    data["_pci"] = _to_numeric(data[pci_col])
    data["_station"] = _to_numeric(data[station_col])

    valid = (
        np.isfinite(data["_x"])
        & np.isfinite(data["_y"])
        & np.isfinite(data["_rsrp"])
        & np.isfinite(data["_pci"])
        & np.isfinite(data["_station"])
    )

    # Use quality flags only when they exist. This keeps the code compatible with
    # both the complete long table and older reduced long tables.
    applied_filters: list[str] = []
    for col, expected, label in (
        ("is_target_27station_pci", 1, "目标27站PCI"),
        ("rsrp_plausible_flag", 1, "合理RSRP"),
        ("dem_hit", 1, "DEM命中"),
    ):
        if col in data.columns:
            values = _to_numeric(data[col])
            valid &= values.eq(expected)
            applied_filters.append(label)

    if args.filter_n41 and "nr5g_band" in data.columns:
        valid &= _to_numeric(data["nr5g_band"]).eq(41)
        applied_filters.append("Band 41")
    if args.filter_center_arfcn and "nr5g_center_arfcn_dl" in data.columns:
        valid &= _to_numeric(data["nr5g_center_arfcn_dl"]).eq(args.center_arfcn)
        applied_filters.append(f"Center ARFCN={args.center_arfcn}")
    if args.filter_bandwidth and "nr5g_bandwidth_dl_mhz" in data.columns:
        bw = _to_numeric(data["nr5g_bandwidth_dl_mhz"])
        valid &= np.isclose(bw, args.bandwidth_mhz, atol=0.01, rtol=0.0)
        applied_filters.append(f"Bandwidth={args.bandwidth_mhz:g} MHz")

    valid &= data["_rsrp"].between(args.rsrp_min_dbm, args.rsrp_max_dbm, inclusive="both")
    data = data.loc[valid].copy()
    if data.empty:
        raise ComparisonError("质量筛选后没有可用的27站实测记录")

    # A receiver instance is one acquisition instant/location before its multiple
    # PCI-RSRP entries are expanded. Prefer rx_point_id; otherwise reconstruct it
    # from source file and source row; finally fall back to coordinates/time.
    if "rx_point_id" in data.columns and data["rx_point_id"].notna().any():
        data["_rx_id"] = data["rx_point_id"].astype(str)
    elif {"source_file", "source_row"}.issubset(data.columns):
        data["_rx_id"] = (
            data["source_file"].astype(str)
            + "::"
            + data["source_row"].astype(str)
        )
    else:
        time_text = data["time"].astype(str) if "time" in data.columns else ""
        data["_rx_id"] = (
            data["_x"].round(3).astype(str)
            + "::"
            + data["_y"].round(3).astype(str)
            + "::"
            + time_text
        )

    # Select the strongest observed target PCI at each receiver instance.
    data = data.sort_values(
        ["_rx_id", "_rsrp", "_pci"], ascending=[True, False, True], kind="mergesort"
    )
    best = data.groupby("_rx_id", sort=False, as_index=False).first()

    rename = {
        "_rx_id": "receiver_instance_id",
        "_x": "x_m",
        "_y": "y_m",
        "_rsrp": "measured_best_rsrp_dbm",
        "_pci": "measured_best_pci",
        "_station": "measured_best_station_id",
    }
    best = best.rename(columns=rename)
    best["measured_best_pci"] = best["measured_best_pci"].astype(np.int32)
    best["measured_best_station_id"] = best["measured_best_station_id"].astype(np.int32)

    keep = [
        "receiver_instance_id",
        "x_m",
        "y_m",
        "measured_best_rsrp_dbm",
        "measured_best_station_id",
        "measured_best_pci",
    ]
    for optional in ("source_file", "source_row", "time", "latitude", "longitude", "speed_mps"):
        if optional in best.columns:
            keep.append(optional)
    best = best[keep].copy()

    report = {
        "raw_long_row_count": int(len(raw)),
        "quality_filtered_long_row_count": int(len(data)),
        "unique_receiver_instance_count": int(len(best)),
        "applied_filters": applied_filters,
        "rsrp_allowed_range_dbm": [args.rsrp_min_dbm, args.rsrp_max_dbm],
    }
    return best, report


def _mode_or_first(series: pd.Series) -> Any:
    values = series.dropna()
    if values.empty:
        return np.nan
    mode = values.mode(dropna=True)
    if not mode.empty:
        return mode.iloc[0]
    return values.iloc[0]


def match_measurements_to_map(
    receiver_best: pd.DataFrame,
    radio_map: MapData,
    aggregate_map_cells: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    points = receiver_best.copy()
    x = points["x_m"].to_numpy(dtype=np.float64)
    y = points["y_m"].to_numpy(dtype=np.float64)
    ix, inside_x = _nearest_indices(radio_map.x_centers_m, x)
    iy, inside_y = _nearest_indices(radio_map.y_centers_m, y)
    inside = inside_x & inside_y

    points["map_ix"] = ix
    points["map_iy"] = iy
    points["inside_map"] = inside
    points = points.loc[inside].copy()
    if points.empty:
        raise ComparisonError("所有有效实测点均位于4000×3000 m联合地图范围之外")

    ix = points["map_ix"].to_numpy(dtype=np.int64)
    iy = points["map_iy"].to_numpy(dtype=np.int64)
    points["map_x_m"] = radio_map.x_centers_m[ix]
    points["map_y_m"] = radio_map.y_centers_m[iy]
    points["simulated_best_rsrp_dbm"] = radio_map.best_rsrp_dbm[iy, ix]

    if radio_map.best_station_id is not None:
        points["simulated_best_station_id"] = radio_map.best_station_id[iy, ix]
    if radio_map.best_pci is not None:
        points["simulated_best_pci"] = radio_map.best_pci[iy, ix]
    if radio_map.outdoor_valid_mask is not None:
        points["outdoor_valid_mask"] = radio_map.outdoor_valid_mask[iy, ix]
    if radio_map.building_mask is not None:
        points["building_mask"] = radio_map.building_mask[iy, ix]

    before_cell_aggregation = len(points)
    if aggregate_map_cells:
        agg: dict[str, Any] = {
            "map_x_m": "first",
            "map_y_m": "first",
            "x_m": "median",
            "y_m": "median",
            "measured_best_rsrp_dbm": "median",
            "simulated_best_rsrp_dbm": "first",
            "measured_best_station_id": _mode_or_first,
            "measured_best_pci": _mode_or_first,
            "receiver_instance_id": "count",
        }
        for col in (
            "simulated_best_station_id",
            "simulated_best_pci",
            "outdoor_valid_mask",
            "building_mask",
        ):
            if col in points.columns:
                agg[col] = "first"
        for col in ("source_file", "source_row", "time", "latitude", "longitude", "speed_mps"):
            if col in points.columns:
                agg[col] = "first"

        points = (
            points.groupby(["map_iy", "map_ix"], as_index=False, sort=True)
            .agg(agg)
            .rename(columns={"receiver_instance_id": "receiver_instance_count_in_cell"})
        )
    else:
        points["receiver_instance_count_in_cell"] = 1

    finite = np.isfinite(points["measured_best_rsrp_dbm"]) & np.isfinite(
        points["simulated_best_rsrp_dbm"]
    )
    if "outdoor_valid_mask" in points.columns:
        finite &= points["outdoor_valid_mask"].astype(bool)
    if "building_mask" in points.columns:
        finite &= ~points["building_mask"].astype(bool)

    matched = points.loc[finite].copy()
    if matched.empty:
        raise ComparisonError(
            "地图范围内的实测点没有可比较的有限仿真RSRP；请检查tile是否全部完成、地图是否存在NaN或坐标是否一致"
        )

    matched["residual_sim_minus_measured_db"] = (
        matched["simulated_best_rsrp_dbm"] - matched["measured_best_rsrp_dbm"]
    )
    matched["absolute_error_db"] = matched["residual_sim_minus_measured_db"].abs()
    matched["squared_error_db2"] = matched["residual_sim_minus_measured_db"] ** 2

    if {"simulated_best_station_id", "measured_best_station_id"}.issubset(matched.columns):
        matched["best_station_match"] = (
            matched["simulated_best_station_id"].astype(int)
            == matched["measured_best_station_id"].astype(int)
        )
    if {"simulated_best_pci", "measured_best_pci"}.issubset(matched.columns):
        matched["best_pci_match"] = (
            matched["simulated_best_pci"].astype(int)
            == matched["measured_best_pci"].astype(int)
        )

    report = {
        "receiver_instances_before_map_extent_filter": int(len(receiver_best)),
        "receiver_instances_inside_map": int(before_cell_aggregation),
        "comparison_rows_after_cell_aggregation": int(len(points)),
        "finite_outdoor_matched_rows": int(len(matched)),
        "aggregate_duplicate_receiver_instances_by_map_cell": bool(aggregate_map_cells),
        "excluded_nonfinite_or_nonoutdoor_rows": int(len(points) - len(matched)),
    }
    return matched, report


def _safe_corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or np.nanstd(a) == 0 or np.nanstd(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def calculate_metrics(matched: pd.DataFrame) -> dict[str, Any]:
    measured = matched["measured_best_rsrp_dbm"].to_numpy(dtype=np.float64)
    simulated = matched["simulated_best_rsrp_dbm"].to_numpy(dtype=np.float64)
    residual = simulated - measured
    abs_error = np.abs(residual)
    mse = float(np.mean(residual**2))
    rmse = float(np.sqrt(mse))
    bias = float(np.mean(residual))
    bias_corrected_residual = residual - bias

    metrics: dict[str, Any] = {
        "comparison_definition": (
            "Joint best-server simulation vs strongest measured target PCI at the same XY; "
            "duplicate receiver instances in one map cell are median-aggregated"
        ),
        "primary_rmse_db": rmse,
        "mse_db2": mse,
        "mae_db": float(np.mean(abs_error)),
        "median_absolute_error_db": float(np.median(abs_error)),
        "bias_sim_minus_measured_db": bias,
        "bias_corrected_rmse_db": float(np.sqrt(np.mean(bias_corrected_residual**2))),
        "p75_absolute_error_db": float(np.percentile(abs_error, 75)),
        "p90_absolute_error_db": float(np.percentile(abs_error, 90)),
        "p95_absolute_error_db": float(np.percentile(abs_error, 95)),
        "max_absolute_error_db": float(np.max(abs_error)),
        "pearson_correlation": _safe_corrcoef(measured, simulated),
        "matched_map_cell_count": int(len(matched)),
        "measured_mean_dbm": float(np.mean(measured)),
        "simulated_mean_dbm": float(np.mean(simulated)),
        "measured_median_dbm": float(np.median(measured)),
        "simulated_median_dbm": float(np.median(simulated)),
    }

    for threshold in (5, 10, 15, 20):
        count = int((abs_error <= threshold).sum())
        metrics[f"within_{threshold}db_count"] = count
        metrics[f"within_{threshold}db_percent"] = float(count / len(abs_error) * 100.0)

    if "best_station_match" in matched.columns:
        metrics["best_station_match_count"] = int(matched["best_station_match"].sum())
        metrics["best_station_match_percent"] = float(matched["best_station_match"].mean() * 100.0)
    if "best_pci_match" in matched.columns:
        metrics["best_pci_match_count"] = int(matched["best_pci_match"].sum())
        metrics["best_pci_match_percent"] = float(matched["best_pci_match"].mean() * 100.0)
    return metrics


def _group_metrics(group: pd.DataFrame, group_name: str, group_value: Any) -> dict[str, Any]:
    residual = group["residual_sim_minus_measured_db"].to_numpy(dtype=float)
    abs_error = np.abs(residual)
    return {
        group_name: group_value,
        "matched_count": int(len(group)),
        "rmse_db": float(np.sqrt(np.mean(residual**2))),
        "mae_db": float(np.mean(abs_error)),
        "bias_sim_minus_measured_db": float(np.mean(residual)),
        "p90_absolute_error_db": float(np.percentile(abs_error, 90)),
    }


def save_group_summaries(matched: pd.DataFrame, output_dir: Path) -> None:
    if "measured_best_station_id" in matched.columns:
        rows = [
            _group_metrics(group, "measured_best_station_id", int(station_id))
            for station_id, group in matched.groupby("measured_best_station_id", sort=True)
        ]
        pd.DataFrame(rows).to_csv(
            output_dir / "rmse_by_measured_best_station.csv", index=False, encoding="utf-8-sig"
        )
    if "source_file" in matched.columns:
        rows = [
            _group_metrics(group, "source_file", source_file)
            for source_file, group in matched.groupby("source_file", sort=True)
        ]
        pd.DataFrame(rows).to_csv(
            output_dir / "rmse_by_source_file.csv", index=False, encoding="utf-8-sig"
        )


def _configure_fonts() -> None:
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def _map_extent(radio_map: MapData) -> tuple[float, float, float, float]:
    dx = float(np.median(np.diff(radio_map.x_centers_m)))
    dy = float(np.median(np.diff(radio_map.y_centers_m)))
    return (
        float(radio_map.x_centers_m[0] - dx / 2),
        float(radio_map.x_centers_m[-1] + dx / 2),
        float(radio_map.y_centers_m[0] - dy / 2),
        float(radio_map.y_centers_m[-1] + dy / 2),
    )


def _draw_route_lines(ax: Any, matched: pd.DataFrame) -> None:
    if "source_file" not in matched.columns:
        return
    for _, group in matched.groupby("source_file", sort=False):
        order_col = "source_row" if "source_row" in group.columns else None
        if order_col is not None:
            group = group.sort_values(order_col)
        ax.plot(group["x_m"], group["y_m"], linewidth=0.45, alpha=1.0, color="0.78", zorder=1)



def _create_fixed_map_axes(
    fig: Any,
    *,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    left: float = 0.095,
    bottom: float = 0.180,
    top: float = 0.820,
    right_margin: float = 0.105,
    cbar_pad: float = 0.012,
    cbar_width: float = 0.022,
) -> tuple[Any, Any]:
    """Create a map axes and a colorbar axes with exactly equal heights.

    The axes rectangles are computed explicitly from the figure size and the map
    aspect ratio, instead of relying on subplot layout engines.  This avoids the
    recurrent colorbar-height mismatch seen with ``tight_layout`` / automatic
    subplot geometry when the map uses ``aspect='equal'``.
    """
    x0, x1 = map(float, xlim)
    y0, y1 = map(float, ylim)
    data_ratio = abs((y1 - y0) / (x1 - x0))
    fig_w, fig_h = fig.get_size_inches()
    avail_w = 1.0 - float(left) - float(right_margin) - float(cbar_pad) - float(cbar_width)
    avail_h = float(top) - float(bottom)
    if avail_w <= 0 or avail_h <= 0:
        raise ValueError('绘图边距设置无效，无法放置主图和色条。')

    normalized_h_if_full_w = avail_w * (fig_w / fig_h) * data_ratio
    if normalized_h_if_full_w <= avail_h:
        ax_w = avail_w
        ax_h = normalized_h_if_full_w
        ax_left = float(left)
        ax_bottom = float(bottom) + 0.5 * (avail_h - ax_h)
    else:
        ax_h = avail_h
        ax_w = avail_h / ((fig_w / fig_h) * data_ratio)
        ax_left = float(left) + 0.5 * (avail_w - ax_w)
        ax_bottom = float(bottom)

    cax_left = ax_left + ax_w + float(cbar_pad)
    ax = fig.add_axes([ax_left, ax_bottom, ax_w, ax_h])
    cax = fig.add_axes([cax_left, ax_bottom, float(cbar_width), ax_h])
    return ax, cax


def _add_colorbar_in_fixed_axes(fig: Any, cax: Any, mappable: Any, label: str):
    cbar = fig.colorbar(mappable, cax=cax)
    cbar.set_label(label)
    return cbar


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


def save_figures(
    matched: pd.DataFrame,
    radio_map: MapData,
    metrics: dict[str, Any],
    output_dir: Path,
    display_min_dbm: float,
    display_max_dbm: float,
) -> None:
    _configure_fonts()
    extent = _map_extent(radio_map)
    common = dict(s=8, linewidths=0, rasterized=True)

    fig = plt.figure(figsize=_publication_figsize_inches(), dpi=MAP_DPI)
    ax, cax = _create_fixed_map_axes(fig, xlim=(extent[0], extent[1]), ylim=(extent[2], extent[3]))
    _draw_route_lines(ax, matched)
    sc = ax.scatter(
        matched["x_m"], matched["y_m"],
        c=np.clip(matched["measured_best_rsrp_dbm"], display_min_dbm, display_max_dbm),
        vmin=display_min_dbm, vmax=display_max_dbm, cmap="viridis", zorder=2, **common
    )
    ax.set_title("实测车载轨迹：27站目标PCI的实测best-server RSRP")
    ax.set_xlabel("Blender X (m)")
    ax.set_ylabel("Blender Y (m)")
    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.3, color="0.88", alpha=1.0)
    _add_colorbar_in_fixed_axes(fig, cax, sc, "Measured best-server RSRP (dBm)")
    _save_png(fig, output_dir / "01_measured_best_server_trajectory.png", dpi=COMPARISON_DPI)
    plt.close(fig)

    fig = plt.figure(figsize=_publication_figsize_inches(), dpi=MAP_DPI)
    ax, cax = _create_fixed_map_axes(fig, xlim=(extent[0], extent[1]), ylim=(extent[2], extent[3]))
    _draw_route_lines(ax, matched)
    sc = ax.scatter(
        matched["x_m"], matched["y_m"],
        c=np.clip(matched["simulated_best_rsrp_dbm"], display_min_dbm, display_max_dbm),
        vmin=display_min_dbm, vmax=display_max_dbm, cmap="viridis", zorder=2, **common
    )
    ax.set_title("同一实测轨迹位置采样的联合仿真best-server RSRP")
    ax.set_xlabel("Blender X (m)")
    ax.set_ylabel("Blender Y (m)")
    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.3, color="0.88", alpha=1.0)
    _add_colorbar_in_fixed_axes(fig, cax, sc, "Simulated best-server RSRP (dBm)")
    _save_png(fig, output_dir / "02_simulated_rsrp_on_measurement_trajectory.png", dpi=COMPARISON_DPI)
    plt.close(fig)

    residual = matched["residual_sim_minus_measured_db"].to_numpy(dtype=float)
    residual_limit = max(10.0, float(np.percentile(np.abs(residual), 95)))
    fig = plt.figure(figsize=_publication_figsize_inches(), dpi=MAP_DPI)
    ax, cax = _create_fixed_map_axes(fig, xlim=(extent[0], extent[1]), ylim=(extent[2], extent[3]))
    _draw_route_lines(ax, matched)
    sc = ax.scatter(
        matched["x_m"], matched["y_m"], c=np.clip(residual, -residual_limit, residual_limit),
        vmin=-residual_limit, vmax=residual_limit, cmap="coolwarm", zorder=2, **common
    )
    ax.set_title(
        "联合无线电地图与实测轨迹残差\n"
        f"RMSE={metrics['primary_rmse_db']:.2f} dB, "
        f"MAE={metrics['mae_db']:.2f} dB, "
        f"Bias={metrics['bias_sim_minus_measured_db']:+.2f} dB"
    )
    ax.set_xlabel("Blender X (m)")
    ax.set_ylabel("Blender Y (m)")
    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.3, color="0.88", alpha=1.0)
    _add_colorbar_in_fixed_axes(fig, cax, sc, "Simulation − measurement (dB)")
    _save_png(fig, output_dir / "03_residual_map_sim_minus_measured.png", dpi=COMPARISON_DPI)
    plt.close(fig)

    measured = matched["measured_best_rsrp_dbm"].to_numpy(dtype=float)
    simulated = matched["simulated_best_rsrp_dbm"].to_numpy(dtype=float)
    low = float(min(np.min(measured), np.min(simulated)))
    high = float(max(np.max(measured), np.max(simulated)))
    pad = max(2.0, 0.03 * (high - low))
    fig, ax = plt.subplots(figsize=_publication_figsize_inches(), dpi=MAP_DPI)
    ax.scatter(measured, simulated, s=9, alpha=1.0, linewidths=0, color="0.55", rasterized=True)
    ax.plot([low-pad, high+pad], [low-pad, high+pad], linestyle="--", linewidth=1.1, color="0.2")
    ax.set_xlim(low-pad, high+pad); ax.set_ylim(low-pad, high+pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Measured best-server RSRP (dBm)")
    ax.set_ylabel("Simulated best-server RSRP (dBm)")
    ax.set_title(
        "Measured vs simulated best-server RSRP\n"
        f"N={len(matched):,}, RMSE={metrics['primary_rmse_db']:.2f} dB, "
        f"r={metrics['pearson_correlation']:.3f}"
    )
    ax.grid(True, linewidth=0.4, color="0.86", alpha=1.0)
    fig.tight_layout()
    _save_png(fig, output_dir / "04_measured_vs_simulated_scatter.png", dpi=COMPARISON_DPI)
    plt.close(fig)

    abs_error = np.sort(matched["absolute_error_db"].to_numpy(dtype=float))
    probability = np.arange(1, len(abs_error)+1, dtype=float) / len(abs_error)
    fig, ax = plt.subplots(figsize=_publication_figsize_inches(), dpi=MAP_DPI)
    ax.plot(abs_error, probability, linewidth=1.6)
    ax.axvline(metrics["mae_db"], linestyle="--", linewidth=1.0, label=f"MAE={metrics['mae_db']:.2f} dB")
    ax.axvline(metrics["p90_absolute_error_db"], linestyle=":", linewidth=1.2, label=f"P90={metrics['p90_absolute_error_db']:.2f} dB")
    ax.set_xlabel("Absolute error (dB)")
    ax.set_ylabel("Cumulative probability")
    ax.set_ylim(0, 1.0)
    ax.set_title("Absolute-error CDF")
    ax.grid(True, linewidth=0.4, color="0.86", alpha=1.0)
    ax.legend(framealpha=1.0)
    fig.tight_layout()
    _save_png(fig, output_dir / "05_absolute_error_cdf.png", dpi=COMPARISON_DPI)
    plt.close(fig)


def _write_text_summary(path: Path, metrics: dict[str, Any], diagnostics: dict[str, Any]) -> None:
    lines = [
        "4000×3000 m 27站联合无线电地图与实测数据比较",
        "=" * 62,
        "",
        "比较口径：",
        "  联合仿真地图为27站/79 PCI的best-server包络。",
        "  实测侧在每个接收时刻先选择目标27站PCI中的最强RSRP。",
        "  默认将落在同一1 m地图网格的重复接收点取中位数，避免停车或重复采样过度加权。",
        "  RMSE使用未截断dBm值；-120~-40 dBm仅用于图片显示。",
        "",
        f"有效匹配地图网格数: {metrics['matched_map_cell_count']:,}",
        f"RMSE: {metrics['primary_rmse_db']:.6f} dB",
        f"MAE: {metrics['mae_db']:.6f} dB",
        f"中位绝对误差: {metrics['median_absolute_error_db']:.6f} dB",
        f"Bias (simulation - measurement): {metrics['bias_sim_minus_measured_db']:+.6f} dB",
        f"去除全局Bias后的RMSE: {metrics['bias_corrected_rmse_db']:.6f} dB",
        f"P90绝对误差: {metrics['p90_absolute_error_db']:.6f} dB",
        f"P95绝对误差: {metrics['p95_absolute_error_db']:.6f} dB",
        f"Pearson相关系数: {metrics['pearson_correlation']:.6f}",
    ]
    if "best_station_match_percent" in metrics:
        lines.append(f"best-station一致率: {metrics['best_station_match_percent']:.3f}%")
    if "best_pci_match_percent" in metrics:
        lines.append(f"best-PCI一致率: {metrics['best_pci_match_percent']:.3f}%")
    lines.extend(["", "数据筛选与空间匹配诊断：", json.dumps(diagnostics, ensure_ascii=False, indent=2)])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="仅使用已保存的4000×3000 m联合无线电地图，与坐标转换后的27站实测长表比较。"
    )
    parser.add_argument(
        "--project-root",
        default=None,
        help="项目根目录。默认自动取本脚本的上两级目录。",
    )
    parser.add_argument("--map-npz", default=None, help="联合地图NPZ；相对路径按项目根目录解析。")
    parser.add_argument("--measurements", default=None, help="坐标转换后的27站实测长表CSV。")
    parser.add_argument("--output-dir", default=None, help="输出目录。")
    parser.add_argument(
        "--keep-duplicate-map-cells",
        action="store_true",
        help="不按1 m地图网格聚合重复接收点。默认聚合，推荐保持默认。",
    )
    parser.add_argument("--rsrp-min-dbm", type=float, default=-140.0)
    parser.add_argument("--rsrp-max-dbm", type=float, default=-40.0)
    parser.add_argument("--display-min-dbm", type=float, default=-120.0)
    parser.add_argument("--display-max-dbm", type=float, default=-40.0)
    parser.add_argument("--center-arfcn", type=int, default=513000)
    parser.add_argument("--bandwidth-mhz", type=float, default=100.0)
    parser.add_argument("--no-filter-n41", dest="filter_n41", action="store_false")
    parser.add_argument("--no-filter-center-arfcn", dest="filter_center_arfcn", action="store_false")
    parser.add_argument("--no-filter-bandwidth", dest="filter_bandwidth", action="store_false")
    parser.set_defaults(filter_n41=True, filter_center_arfcn=True, filter_bandwidth=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    script_path = Path(__file__).resolve()
    inferred_root = script_path.parents[2]
    project_root = Path(args.project_root).resolve() if args.project_root else inferred_root

    map_path = _resolve_path(project_root, args.map_npz, DEFAULT_MAP_REL)
    measurement_path = _resolve_path(project_root, args.measurements, DEFAULT_MEASUREMENTS_REL)
    output_dir = _resolve_path(project_root, args.output_dir, DEFAULT_OUTPUT_REL)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[1/5] 读取已保存的27站联合无线电地图，不执行Sionna仿真")
    print(f"      map: {map_path}")
    radio_map = load_joint_map(map_path)
    print(
        f"      map shape={radio_map.best_rsrp_dbm.shape}, "
        f"x=[{radio_map.x_centers_m[0]:.3f}, {radio_map.x_centers_m[-1]:.3f}], "
        f"y=[{radio_map.y_centers_m[0]:.3f}, {radio_map.y_centers_m[-1]:.3f}]"
    )

    print("[2/5] 读取坐标转换后的27站实测长表并构造实测best-server")
    print(f"      measurements: {measurement_path}")
    receiver_best, measurement_report = read_measurement_best_server(measurement_path, args)

    print("[3/5] 将实测点匹配到同一1 m联合地图网格")
    matched, match_report = match_measurements_to_map(
        receiver_best,
        radio_map,
        aggregate_map_cells=not args.keep_duplicate_map_cells,
    )

    print("[4/5] 计算未截断dBm下的RMSE等指标")
    metrics = calculate_metrics(matched)
    diagnostics = {
        "project_root": str(project_root),
        "map_npz": str(map_path),
        "measurements_csv": str(measurement_path),
        "output_dir": str(output_dir),
        "map_shape": list(radio_map.best_rsrp_dbm.shape),
        "map_x_center_range_m": [float(radio_map.x_centers_m[0]), float(radio_map.x_centers_m[-1])],
        "map_y_center_range_m": [float(radio_map.y_centers_m[0]), float(radio_map.y_centers_m[-1])],
        "measurement_processing": measurement_report,
        "spatial_matching": match_report,
        "rmse_uses_unclipped_dbm": True,
        "figure_display_range_dbm": [args.display_min_dbm, args.display_max_dbm],
    }

    matched.sort_values(["map_iy", "map_ix"]).to_csv(
        output_dir / "matched_measurement_vs_joint_map.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame([metrics]).to_csv(
        output_dir / "comparison_metrics.csv", index=False, encoding="utf-8-sig"
    )
    (output_dir / "comparison_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "comparison_diagnostics.json").write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    save_group_summaries(matched, output_dir)
    _write_text_summary(output_dir / "comparison_summary.txt", metrics, diagnostics)

    print("[5/5] 生成实测轨迹、仿真轨迹、残差、散点和CDF图片")
    save_figures(
        matched,
        radio_map,
        metrics,
        output_dir,
        display_min_dbm=args.display_min_dbm,
        display_max_dbm=args.display_max_dbm,
    )

    print("\n比较完成")
    print(f"  匹配网格数: {metrics['matched_map_cell_count']:,}")
    print(f"  RMSE: {metrics['primary_rmse_db']:.6f} dB")
    print(f"  MAE: {metrics['mae_db']:.6f} dB")
    print(f"  Bias(sim-meas): {metrics['bias_sim_minus_measured_db']:+.6f} dB")
    print(f"  Bias-corrected RMSE: {metrics['bias_corrected_rmse_db']:.6f} dB")
    print(f"  P90 absolute error: {metrics['p90_absolute_error_db']:.6f} dB")
    print(f"  输出目录: {output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ComparisonError as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(2)
