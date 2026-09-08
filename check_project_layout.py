#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Inspect the self-contained project layout without requiring generated outputs."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent

CORE_FILES = [
    ROOT / "assets/ground.ply",
    ROOT / "assets/ynu_chenggong_campus-001.ply",
    ROOT / "config/base_station_pci_mapping.csv",
    ROOT / "config/station_catalog_27stations.csv",
    ROOT / "config/coordinate_alignment.json",
]

CALIBRATION_REQUIRED_COLUMNS = {
    "rx_point_id",
    "blender_x",
    "blender_y",
    "ground_z_m",
    "receiver_z_m",
    "pci",
    "measured_rsrp_dbm",
    "rsrp_unit",
    "station_id",
    "station_label",
    "sector_index",
    "antenna_type",
    "is_omnidirectional",
    "is_target_27station_pci",
    "rsrp_plausible_flag",
    "dem_hit",
    "tx_x_initial_m",
    "tx_y_initial_m",
    "bearing_from_tx_math_deg",
}


def clean_columns(frame: pd.DataFrame) -> set[str]:
    return {str(c).replace("\ufeff", "").strip() for c in frame.columns}


def status(path: Path, label: str | None = None) -> bool:
    exists = path.exists()
    shown = label if label is not None else str(path.relative_to(ROOT))
    print("[OK]" if exists else "[MISSING]", shown)
    return exists


def main() -> int:
    print("项目根目录:", ROOT)
    print("\n[1] 场景与配置")
    missing_core: list[str] = []
    for path in CORE_FILES:
        if not status(path):
            missing_core.append(str(path.relative_to(ROOT)))

    mapping = ROOT / "config/base_station_pci_mapping.csv"
    if mapping.is_file():
        table = pd.read_csv(mapping, encoding="utf-8-sig")
        table.columns = [str(c).replace("\ufeff", "").strip() for c in table.columns]
        print(
            f"      映射统计: physical_stations={table['station_id'].nunique()}, "
            f"PCI={table['pci'].nunique()}, rows={len(table)}"
        )
        if table["station_id"].nunique() != 27 or table["pci"].nunique() != 79:
            print("[ERROR] PCI映射不是预期的27站/79 PCI。")
            missing_core.append("config/base_station_pci_mapping.csv: invalid station/PCI count")

    print("\n[2] 实测数据阶段")
    raw_files = sorted((ROOT / "data/raw_measurements").glob("*.csv"))
    aligned_files = sorted((ROOT / "data/aligned_measurements").glob("*_with_blender_xyz.csv"))
    processed_long = ROOT / "data/processed/cell_pci_rsrp_long_27stations.csv"
    calibration_csv = ROOT / "data/processed/cell_pci_rsrp_1m_calibration.csv"
    localization_csv = ROOT / "data/processed/cell_pci_rsrp_2p77m_localization.csv"
    print(f"[{'OK' if raw_files else 'MISSING'}] data/raw_measurements/*.csv count={len(raw_files)}")
    print(f"[{'OK' if aligned_files else 'NOT-YET'}] data/aligned_measurements/*_with_blender_xyz.csv count={len(aligned_files)}")
    status(processed_long)
    status(calibration_csv)
    status(localization_csv)

    if calibration_csv.is_file():
        header = pd.read_csv(calibration_csv, encoding="utf-8-sig", nrows=0)
        schema_missing = sorted(CALIBRATION_REQUIRED_COLUMNS - clean_columns(header))
        if schema_missing:
            print("[OLD-SCHEMA] data/processed/cell_pci_rsrp_1m_calibration.csv")
            print("      缺少字段:", schema_missing)
            print("      修复命令: python run_pipeline.py preprocess")
        else:
            print("[OK-SCHEMA] 1 m校准表字段完整")

    legacy_output_dirs = [
        ROOT / "outputs/04_parameter_calibration",
        ROOT / "outputs/11_joint_best_server_4000x3000",
        ROOT / "outputs/12_joint_map_measurement_comparison",
    ]
    existing_legacy = [p for p in legacy_output_dirs if p.exists()]
    if existing_legacy:
        print("\n[OLD-NAMES] 检测到带数字前缀的旧结果目录:")
        for path in existing_legacy:
            print(" -", path.relative_to(ROOT))
        print("  迁移命令: python run_pipeline.py migrate-output-names")

    print("\n[3] 已生成的后续结果（缺失不代表安装失败）")
    generated = [
        ROOT / "outputs/parameter_calibration/all_27stations_summary.csv",
        ROOT / "outputs/parameter_calibration/estimated_initial_directions_27stations.csv",
        ROOT / "outputs/joint_best_server_4000x3000/joint_best_server_27stations_4000x3000.npz",
        ROOT / "outputs/joint_map_measurement_comparison/comparison_metrics.csv",
    ]
    for path in generated:
        print("[READY]" if path.exists() else "[NOT-YET]", path.relative_to(ROOT))

    if missing_core:
        print("\n核心文件不完整:")
        for item in missing_core:
            print(" -", item)
        return 1

    if not raw_files and not processed_long.is_file():
        print(
            "\n[WARNING] 当前既没有原始CSV，也没有已处理27站长表。"
            "请放入data/raw_measurements/*.csv后运行prepare-data，"
            "或放入现成的data/processed/cell_pci_rsrp_long_27stations.csv。"
        )
        return 2

    print("\n项目结构检查完成。")
    print("all_27stations_summary.csv和estimated_initial_directions_27stations.csv由正式调参自动生成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
