#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""构建下游统一使用的实测数据表。

输入
----
``data/processed/extracted/cell_pci_rsrp_long_27stations.csv``

输出
----
1. ``cell_pci_rsrp_long_27stations.csv``：未经空间聚合的正式长表；
2. ``cell_pci_rsrp_1m_calibration.csv``：1 m、同站同PCI中位数聚合表；
3. ``cell_pci_rsrp_2p77m_localization.csv``：2.77 m定位聚合表。

关键约束
--------
- 聚合后仍保留/重建下游所需的完整元数据；
- 包括 ``rx_point_id``、DEM命中标志、站点坐标、扇区信息和几何方位；
- 不截断原始RSRP，只标记物理合理性；
- 同一空间格内同一PCI采用中位数，停车产生的重复记录不会全部删除。
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "data" / "processed" / "extracted" / "cell_pci_rsrp_long_27stations.csv"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "processed"
DEFAULT_MAPPING = ROOT / "config" / "base_station_pci_mapping.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成1 m校准表和2.77 m定位表")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--mapping", default=str(DEFAULT_MAPPING))
    return parser.parse_args()


def normalize_headers(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out.columns = [str(c).replace("\ufeff", "").strip() for c in out.columns]
    return out


def first_existing(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    lookup = {str(c).strip().lower(): c for c in frame.columns}
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
        hit = lookup.get(candidate.strip().lower())
        if hit is not None:
            return hit
    return None


def ensure_alias(frame: pd.DataFrame, target: str, candidates: list[str], required: bool = True) -> pd.DataFrame:
    if target in frame.columns:
        return frame
    source = first_existing(frame, candidates)
    if source is None:
        if required:
            raise KeyError(f"缺少字段 {target}，候选列={candidates}，实际列={list(frame.columns)}")
        return frame
    frame[target] = frame[source]
    return frame


def attach_mapping_metadata(frame: pd.DataFrame, mapping_path: Path) -> pd.DataFrame:
    """按PCI补齐站号、扇区、站点坐标和天线类型。"""
    if not mapping_path.exists():
        return frame
    mapping = normalize_headers(pd.read_csv(mapping_path, encoding="utf-8-sig", low_memory=False))
    if "pci" not in mapping.columns:
        return frame
    mapping["pci"] = pd.to_numeric(mapping["pci"], errors="coerce")
    mapping = mapping.dropna(subset=["pci"]).copy()
    mapping["pci"] = mapping["pci"].astype(int)
    wanted = [
        "pci", "station_id", "station_name", "station_label", "sector_index",
        "antenna_type", "is_omnidirectional", "tx_x_initial_m",
        "tx_y_initial_m", "tx_z_initial_m",
    ]
    mapping = mapping[[c for c in wanted if c in mapping.columns]].drop_duplicates("pci")
    merged = frame.merge(mapping, on="pci", how="left", suffixes=("", "__map"))
    for column in wanted:
        if column == "pci":
            continue
        mapped = f"{column}__map"
        if mapped not in merged.columns:
            continue
        if column not in merged.columns:
            merged[column] = merged[mapped]
        else:
            current = merged[column]
            missing = current.isna() | current.astype(str).str.strip().isin(["", "nan", "None"])
            merged.loc[missing, column] = merged.loc[missing, mapped]
        merged.drop(columns=[mapped], inplace=True)
    return merged


def _median_columns(frame: pd.DataFrame) -> list[str]:
    candidates = [
        "blender_x", "blender_y", "measured_rsrp_dbm", "ground_z_m", "receiver_z_m",
        "latitude", "longitude", "speed_mps", "accuracy_m", "original_altitude_m",
        "tx_x_initial_m", "tx_y_initial_m", "tx_z_initial_m",
        "nr5g_band", "nr5g_bandwidth_dl_mhz", "nr5g_ssb_arfcn_dl",
        "nr5g_center_arfcn_dl", "serving_ss_rsrp_dbm",
    ]
    return [c for c in candidates if c in frame.columns]


def aggregate(frame: pd.DataFrame, grid_m: float, mapping_path: Path) -> pd.DataFrame:
    if grid_m <= 0:
        raise ValueError("grid_m必须大于0")

    work = normalize_headers(frame)
    work = ensure_alias(work, "blender_x", ["x_m", "x", "receiver_x_m"])
    work = ensure_alias(work, "blender_y", ["y_m", "y", "receiver_y_m"])
    work = ensure_alias(work, "measured_rsrp_dbm", ["rsrp_dbm", "cell_rsrp_dbm", "NR5G SS RSRP"])
    work = ensure_alias(work, "pci", ["NR5G PCI", "nr5g_pci"])
    work = ensure_alias(work, "station_id", ["StationID", "base_station_id"], required=False)

    for column in ["blender_x", "blender_y", "measured_rsrp_dbm", "pci", "station_id"]:
        if column in work.columns:
            work[column] = pd.to_numeric(work[column], errors="coerce")
    work = work.loc[
        np.isfinite(work["blender_x"])
        & np.isfinite(work["blender_y"])
        & np.isfinite(work["measured_rsrp_dbm"])
        & np.isfinite(work["pci"])
    ].copy()
    work["pci"] = work["pci"].astype(int)

    work = attach_mapping_metadata(work, mapping_path)
    work["station_id"] = pd.to_numeric(work.get("station_id"), errors="coerce")
    work = work.loc[np.isfinite(work["station_id"])].copy()
    work["station_id"] = work["station_id"].astype(int)

    # 仅生成合理性标志，不按该标志提前删除原始观测。
    work["rsrp_plausible_flag"] = work["measured_rsrp_dbm"].between(-200.0, 0.0).astype(int)
    if "is_target_27station_pci" not in work.columns:
        work["is_target_27station_pci"] = 1
    work["is_target_27station_pci"] = pd.to_numeric(
        work["is_target_27station_pci"], errors="coerce"
    ).fillna(1).astype(int)

    # 与历史处理保持一致：世界坐标按最近网格中心归并。
    work["grid_ix"] = np.floor(work["blender_x"] / grid_m + 0.5).astype(np.int64)
    work["grid_iy"] = np.floor(work["blender_y"] / grid_m + 0.5).astype(np.int64)
    group_cols = ["station_id", "pci", "grid_ix", "grid_iy"]

    aggregations: dict[str, str] = {c: "median" for c in _median_columns(work)}
    first_cols = [
        "sector_index", "antenna_type", "is_omnidirectional", "station_name",
        "station_label", "mapping_source", "rsrp_unit", "network_type", "operator",
        "nr5g_nci", "nr5g_gnodeb_id", "nr5g_cell_id", "measurement_source",
    ]
    for column in first_cols:
        if column in work.columns:
            aggregations[column] = "first"
    for column in ["dem_hit", "is_target_27station_pci", "rsrp_plausible_flag"]:
        if column in work.columns:
            aggregations[column] = "max"

    out = work.groupby(group_cols, as_index=False, sort=True).agg(aggregations)
    counts = (
        work.groupby(group_cols, sort=True)
        .size()
        .rename("raw_observation_count")
        .reset_index()
    )
    out = out.merge(counts, on=group_cols, how="left")

    # 统一下游字段。
    out["station_id"] = out["station_id"].astype(int)
    out["pci"] = out["pci"].astype(int)
    out["x_m"] = out["blender_x"]
    out["y_m"] = out["blender_y"]
    out["rsrp_dbm"] = out["measured_rsrp_dbm"]
    out["aggregation_grid_m"] = float(grid_m)
    out["rsrp_unit"] = "dBm"
    out["is_target_27station_pci"] = pd.to_numeric(
        out.get("is_target_27station_pci", 1), errors="coerce"
    ).fillna(1).astype(int)
    out["rsrp_plausible_flag"] = out["measured_rsrp_dbm"].between(-200.0, 0.0).astype(int)

    if "ground_z_m" in out.columns:
        derived_dem_hit = np.isfinite(pd.to_numeric(out["ground_z_m"], errors="coerce")).astype(int)
    else:
        derived_dem_hit = np.zeros(len(out), dtype=int)
        out["ground_z_m"] = np.nan
    if "dem_hit" in out.columns:
        source_hit = pd.to_numeric(out["dem_hit"], errors="coerce").fillna(0).astype(int)
        out["dem_hit"] = ((source_hit == 1) & (derived_dem_hit == 1)).astype(int)
    else:
        out["dem_hit"] = derived_dem_hit

    if "receiver_z_m" not in out.columns:
        out["receiver_z_m"] = out["ground_z_m"] + 1.5
    else:
        rz = pd.to_numeric(out["receiver_z_m"], errors="coerce")
        out["receiver_z_m"] = rz.where(np.isfinite(rz), out["ground_z_m"] + 1.5)
    out["receiver_height_agl_m"] = out["receiver_z_m"] - out["ground_z_m"]

    # 一个空间格在不同PCI之间共享同一RX ID，避免把同位置误算成多个接收点。
    grid_tag = str(grid_m).replace(".", "p")
    out["rx_point_id"] = (
        "g" + grid_tag + "_x" + out["grid_ix"].astype(str)
        + "_y" + out["grid_iy"].astype(str)
    )
    out["measurement_id"] = (
        out["rx_point_id"] + "_s" + out["station_id"].astype(str)
        + "_p" + out["pci"].astype(str)
    )

    # 聚合坐标改变后重新计算几何量，不能沿用任意一条原始记录。
    for column in ["tx_x_initial_m", "tx_y_initial_m", "tx_z_initial_m"]:
        if column not in out.columns:
            out[column] = np.nan
        out[column] = pd.to_numeric(out[column], errors="coerce")
    dx = out["blender_x"] - out["tx_x_initial_m"]
    dy = out["blender_y"] - out["tx_y_initial_m"]
    dz = out["receiver_z_m"] - out["tx_z_initial_m"]
    horizontal = np.hypot(dx, dy)
    out["horizontal_distance_initial_m"] = horizontal
    out["distance_3d_initial_m"] = np.sqrt(dx * dx + dy * dy + dz * dz)
    out["bearing_from_tx_math_deg"] = (np.degrees(np.arctan2(dy, dx)) + 360.0) % 360.0
    out["bearing_from_tx_compass_deg"] = (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0
    out["geometric_downtilt_to_rx_deg"] = np.degrees(
        np.arctan2(out["tx_z_initial_m"] - out["receiver_z_m"], np.maximum(horizontal, 1.0e-9))
    )

    # 保证下游布尔/编号字段为数值。
    for column, default in [
        ("sector_index", np.nan), ("is_omnidirectional", 0),
    ]:
        if column not in out.columns:
            out[column] = default
        out[column] = pd.to_numeric(out[column], errors="coerce")

    # 将校准器明确要求的字段放到前部，便于人工检查。
    priority = [
        "measurement_id", "rx_point_id", "station_id", "station_name", "station_label",
        "pci", "sector_index", "antenna_type", "is_omnidirectional",
        "grid_ix", "grid_iy", "aggregation_grid_m", "blender_x", "blender_y",
        "ground_z_m", "receiver_z_m", "measured_rsrp_dbm", "rsrp_unit",
        "raw_observation_count", "is_target_27station_pci", "rsrp_plausible_flag",
        "dem_hit", "tx_x_initial_m", "tx_y_initial_m", "tx_z_initial_m",
        "bearing_from_tx_math_deg", "bearing_from_tx_compass_deg",
        "horizontal_distance_initial_m", "distance_3d_initial_m",
        "geometric_downtilt_to_rx_deg", "x_m", "y_m", "rsrp_dbm",
    ]
    ordered = [c for c in priority if c in out.columns] + [c for c in out.columns if c not in priority]
    return out[ordered].sort_values(group_cols).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    mapping_path = Path(args.mapping).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"找不到提取后的27站长表：{input_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = normalize_headers(pd.read_csv(input_path, encoding="utf-8-sig", low_memory=False))
    raw_path = output_dir / "cell_pci_rsrp_long_27stations.csv"
    frame.to_csv(raw_path, index=False, encoding="utf-8-sig")

    calibration = aggregate(frame, 1.0, mapping_path)
    localization = aggregate(frame, 2.77, mapping_path)
    calibration_path = output_dir / "cell_pci_rsrp_1m_calibration.csv"
    localization_path = output_dir / "cell_pci_rsrp_2p77m_localization.csv"
    calibration.to_csv(calibration_path, index=False, encoding="utf-8-sig")
    localization.to_csv(localization_path, index=False, encoding="utf-8-sig")

    required_calibration = {
        "rx_point_id", "blender_x", "blender_y", "ground_z_m", "receiver_z_m",
        "pci", "measured_rsrp_dbm", "rsrp_unit", "station_id", "station_label",
        "sector_index", "antenna_type", "is_omnidirectional",
        "is_target_27station_pci", "rsrp_plausible_flag", "dem_hit",
        "tx_x_initial_m", "tx_y_initial_m", "bearing_from_tx_math_deg",
    }
    missing = sorted(required_calibration - set(calibration.columns))
    if missing:
        raise RuntimeError(f"内部错误：1m校准表仍缺少字段 {missing}")

    report = {
        "input": str(input_path),
        "raw_long_rows": int(len(frame)),
        "calibration_1m_rows": int(len(calibration)),
        "localization_2p77m_rows": int(len(localization)),
        "station_count": int(calibration["station_id"].nunique()),
        "pci_count": int(calibration["pci"].nunique()),
        "calibration_required_columns_ok": True,
        "calibration_output": str(calibration_path),
        "localization_output": str(localization_path),
    }
    (output_dir / "processing_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
