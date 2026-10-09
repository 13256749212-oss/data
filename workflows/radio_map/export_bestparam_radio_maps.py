#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
export_bestparam_radio_maps.py

功能
====
使用27个基站调参阶段已经得到的最佳参数，重新运行 512 m × 512 m、
1 m网格、真实 DEM+1.5 m 室外接收面的 Sionna RT 无线电地图，并输出两套数据产品：

A. pure_simulation（纯净仿真版）
   - 使用最佳高度、最佳公共方位角偏移、最佳绝对下倾角和最佳共享功率；
   - 不叠加实测采集点；
   - 建筑物以白色实体块显示，并叠加TX位置；
   - 图中不显示室外地图命中率；
   - 图中只显示“实测位置命中率”。

B. measurement_filled（measurement-filled版）
   - 纯Sionna RT有限值始终保留，不被实测覆盖；
   - 仅当同PCI、同1 m网格的Sionna RT值缺失且存在实测值时，用该实测值补齐；
   - 不对实测值进行跨网格插值；
   - NPZ保存value_source_mask与measurement_available_mask，明确每个网格来源。

默认数据
========
工程根目录：代码包根目录（自动识别）
处理后实测长表：data\processed\cell_pci_rsrp_long_27stations.csv
最佳参数汇总：outputs\parameter_calibration\all_27stations_summary.csv
地形：assets\ground.ply
建筑：assets\ynu_chenggong_campus*.ply

输出
====
outputs\bestparam_radio_maps_512m\station_XX\
    01_pure_simulation\png\
    01_pure_simulation\npz\
    02_measurement_filled\figures\
    02_measurement_filled\npz\

每个扇区分别保存PNG和NPZ，并额外保存best-server PNG和NPZ。

显示参数
========
地图物理范围：512 m × 512 m
数值网格：512 × 512（1 m/格）
PNG输出：严格512 × 512像素，300 dpi（与正式参考图一致）
接收面：DEM + 1.5 m
配色：viridis
RSRP显示范围：-120～-40 dBm\n绘图NaN按-120 dBm截断；建筑内部由building_mask覆盖为白色实体块。
室外未命中仍按-120 dBm显示，不与白色建筑块混淆。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core_dem15m import (
    RuntimeStationConfig,
    build_all_27_station_configs,
    build_dense_outdoor_measurement_surface,
    build_scene_xml_multi,
    configure_tx_array_for_station,
    install_general_sector_support,
    create_dense_grid_with_building_mask,
    load_building_projection_triangles,
    read_27station_long_measurements,
    remove_measurements_inside_buildings,
    run_candidate_multibatch_linear_average,
    sector_values_to_full_maps,
)
from src.measurement_io import prepare_station_measurements
from src.optimizer import evaluate_prediction
from src.simulator import Candidate, configure_scene
from src.terrain import TerrainModel

# 安装1扇区全向站与3扇区站的通用支持。
# 22号站必须在任何仿真调用之前完成该补丁。
install_general_sector_support()

MAP_SIZE_M = 512
CELL_SIZE_M = 1.0
DEFAULT_FREQUENCY_HZ = 2_565_000_000.0
DEFAULT_BANDWIDTH_HZ = 100_000_000.0
DEFAULT_N_RB = 273
DEFAULT_SUBCARRIERS_PER_RB = 12
DEFAULT_SEED = 20260805


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "按27站最佳参数重新运行512m×512m无线电地图，输出纯仿真版和"
            "仅在同PCI同网格仿真缺失位置使用实测值补齐的measurement-filled地图。"
        )
    )
    parser.add_argument(
        "--project-root",
        default=str(Path(__file__).resolve().parents[2]),
        help="工程根目录",
    )
    parser.add_argument(
        "--measurements",
        default=None,
        help="处理后的27站长表CSV",
    )
    parser.add_argument(
        "--summary-csv",
        default=None,
        help="调参最佳参数汇总CSV",
    )
    parser.add_argument("--ground", default=None, help="ground.ply路径")
    parser.add_argument(
        "--buildings",
        nargs="*",
        default=None,
        help="建筑PLY列表；未指定时扫描assets/ynu_chenggong_campus*.ply",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="输出根目录",
    )
    parser.add_argument(
        "--stations",
        default="all",
        help="默认全部27站；也可指定2,3,7",
    )
    parser.add_argument("--force", action="store_true", help="忽略缓存重新仿真")
    parser.add_argument(
        "--final-batches",
        type=int,
        default=None,
        help="最终地图批次数；默认读取summary，缺失时为5",
    )
    parser.add_argument(
        "--final-samples-per-batch",
        type=int,
        default=None,
        help="每批每TX采样数；默认读取summary，缺失时为10000000",
    )
    parser.add_argument(
        "--final-seed-step",
        type=int,
        default=1009,
        help="多批次seed步长",
    )
    parser.add_argument("--final-max-depth", type=int, default=5, help=argparse.SUPPRESS)
    parser.add_argument(
        "--display-min-dbm",
        type=float,
        default=-120.0,
        help="图像色标下限",
    )
    parser.add_argument(
        "--display-max-dbm",
        type=float,
        default=-40.0,
        help="图像色标上限",
    )
    parser.add_argument(
        "--rx-height-agl-m",
        type=float,
        default=1.5,
        help="接收面相对DEM高度",
    )
    parser.add_argument(
        "--building-mask-buffer-cells",
        type=int,
        default=1,
        help="建筑掩膜向外扩展网格数",
    )
    parser.add_argument(
        "--direction-top-fraction",
        type=float,
        default=0.25,
        help="重建初始扇区方向时使用每PCI最强数据比例",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        choices=[300],
        default=300,
        help="正式单站无线电地图固定为512×512像素、300 dpi；该参数仅保留命令兼容性",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="单站失败时继续处理其他站；默认立即停止",
    )
    return parser.parse_args()


# =============================================================================
# 路径解析
# =============================================================================

def _first_existing(paths: Sequence[Path], description: str) -> Path:
    for path in paths:
        resolved = Path(path).expanduser().resolve()
        if resolved.exists():
            return resolved
    raise FileNotFoundError(
        f"未找到{description}。尝试路径：\n" + "\n".join(str(p) for p in paths)
    )


def resolve_measurements(project_root: Path, explicit: str | None) -> Path:
    if explicit:
        return _first_existing([Path(explicit)], "处理后27站实测长表")
    return _first_existing(
        [
            project_root / "data" / "cell_pci_rsrp_long_27stations.csv",
            project_root / "data" / "cell_pci_rsrp_long_27stations(1).csv",
        ],
        "处理后27站实测长表",
    )


def resolve_summary(project_root: Path, explicit: str | None) -> Path:
    if explicit:
        return _first_existing([Path(explicit)], "最佳参数汇总CSV")
    return _first_existing(
        [
            project_root / "outputs" / "parameter_calibration" / "all_27stations_summary.csv",
            
            
        ],
        "最佳参数汇总CSV",
    )


def resolve_ground(project_root: Path, explicit: str | None) -> Path:
    if explicit:
        return _first_existing([Path(explicit)], "ground.ply")
    return _first_existing(
        [
            project_root / "assets" / "ground.ply",
            project_root / "assets" / "ground(1).ply",
        ],
        "ground.ply",
    )


def resolve_buildings(project_root: Path, explicit: Sequence[str] | None) -> list[Path]:
    if explicit:
        candidates = [Path(v).expanduser().resolve() for v in explicit]
    else:
        candidates = sorted((project_root / "assets").glob("ynu_chenggong_campus*.ply"))

    result: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        resolved = path.resolve()
        key = str(resolved).casefold()
        if key in seen:
            continue
        seen.add(key)
        if not resolved.exists() or not resolved.is_file() or resolved.stat().st_size <= 0:
            print(f"[建筑] 跳过不存在/非文件/空文件：{resolved}")
            continue
        result.append(resolved)

    if not result:
        raise FileNotFoundError("没有找到可加载的建筑PLY")
    return result


# =============================================================================
# Summary读取与数值转换
# =============================================================================

def _float_value(row: pd.Series, name: str, default: float) -> float:
    if name not in row.index:
        return float(default)
    value = pd.to_numeric(pd.Series([row[name]]), errors="coerce").iat[0]
    return float(value) if np.isfinite(value) else float(default)


def _int_value(row: pd.Series, name: str, default: int) -> int:
    return int(round(_float_value(row, name, float(default))))


def _bool_value(row: pd.Series, name: str, default: bool) -> bool:
    if name not in row.index or pd.isna(row[name]):
        return bool(default)
    value = row[name]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    return bool(default)


def load_best_parameter_summary(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    required = {
        "station_id",
        "height_agl_m",
        "azimuth_offset_deg",
        "absolute_downtilt_deg",
        "shared_power_dbm",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"最佳参数汇总缺少字段：{missing}")

    frame["station_id"] = pd.to_numeric(frame["station_id"], errors="coerce")
    frame = frame.loc[np.isfinite(frame["station_id"])].copy()
    frame["station_id"] = frame["station_id"].astype(int)
    frame = frame.drop_duplicates("station_id", keep="first").sort_values("station_id")
    return frame.reset_index(drop=True)


def select_station_ids(text: str, available: Sequence[int]) -> list[int]:
    available_ids = sorted(set(int(v) for v in available))
    if str(text).strip().lower() in {"all", "*", ""}:
        return available_ids
    selected = sorted(set(int(v.strip()) for v in str(text).split(",") if v.strip()))
    unknown = sorted(set(selected) - set(available_ids))
    if unknown:
        raise ValueError(f"summary中不存在这些站：{unknown}")
    return selected


# =============================================================================
# Sionna运行配置
# =============================================================================

def make_station_cfg(
    scene_xml: Path,
    station: RuntimeStationConfig,
    rx_height_agl_m: float,
    max_depth: int,
    edge_diffraction: bool,
) -> Dict[str, Any]:
    if station.is_omnidirectional:
        antenna = {
            "num_rows": 1,
            "num_cols": 1,
            "vertical_spacing": 0.5,
            "horizontal_spacing": 0.5,
            "pattern": "iso",
            "polarization": "V",
        }
    else:
        antenna = {
            "num_rows": 8,
            "num_cols": 4,
            "vertical_spacing": 0.5,
            "horizontal_spacing": 0.5,
            "pattern": "tr38901",
            "polarization": "VH",
        }

    return {
        "_resolved_scene_xml": str(scene_xml.resolve()),
        "radio": {
            "frequency_hz": DEFAULT_FREQUENCY_HZ,
            "bandwidth_hz": DEFAULT_BANDWIDTH_HZ,
            "nr_band": 41,
            "center_arfcn_dl": 513000,
            "ssb_arfcn_dl": 504990,
            "n_rb": DEFAULT_N_RB,
            "subcarriers_per_rb": DEFAULT_SUBCARRIERS_PER_RB,
            "ss_rsrp_mode": "3gpp_sss_re_uniform_epre_narrowband_global_calibrated",
            "sss_re_count": 127,
            "sss_re_epre_offset_db": 0.0,
            "rx_height_agl_m": float(rx_height_agl_m),
            "max_depth": int(max_depth),
            "seed": DEFAULT_SEED,
            "los": True,
            "specular_reflection": True,
            "diffuse_reflection": True,
            "refraction": True,
            "diffraction": True,
            "edge_diffraction": bool(edge_diffraction),
        },
        "antenna": antenna,
    }


# =============================================================================
# 实测1m聚合、命中率、实测替换
# =============================================================================

def prepare_station_measurement_grid(
    observations: pd.DataFrame,
    station: RuntimeStationConfig,
    building_mask: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    aggregated = prepare_station_measurements(
        observations=observations,
        station=station,
        map_size_x_m=float(MAP_SIZE_M),
        map_size_y_m=float(MAP_SIZE_M),
        cell_size_m=CELL_SIZE_M,
        min_points_per_pci=1,
        max_cells_per_pci=1_000_000_000,
        strong_signal_sampling_fraction=1.0,
    )
    outdoor, inside = remove_measurements_inside_buildings(
        measurements=aggregated,
        building_mask=building_mask,
    )
    return outdoor.reset_index(drop=True), inside.reset_index(drop=True)


def compute_measurement_hit_statistics(
    station: RuntimeStationConfig,
    measurements: pd.DataFrame,
    evaluation: Dict[str, Any],
) -> Dict[str, Any]:
    per_pci_eval = evaluation.get("per_pci", {})
    per_pci: dict[int, dict[str, Any]] = {}
    for pci in station.pcis:
        total = int(measurements["pci"].eq(int(pci)).sum())
        hit = int(per_pci_eval.get(int(pci), {}).get("count", 0))
        per_pci[int(pci)] = {
            "measurement_cell_count": total,
            "simulated_hit_count": hit,
            "measurement_hit_rate": float(hit / total) if total else float("nan"),
        }

    total = int(len(measurements))
    hit = int(evaluation.get("paired_point_count", 0))
    return {
        "measurement_cell_count": total,
        "simulated_hit_count": hit,
        "measurement_hit_rate": float(hit / total) if total else float("nan"),
        "per_pci": per_pci,
    }


def fill_missing_simulation_cells_from_measurements(
    station: RuntimeStationConfig,
    grid: Any,
    pure_sector_maps: np.ndarray,
    measurements: pd.DataFrame,
) -> Dict[str, np.ndarray]:
    """Create the formal measurement-filled map without overwriting valid RT cells.

    Provenance coding per grid cell:
      0 = missing/invalid or building
      1 = original finite Sionna RT value retained
      2 = originally missing Sionna RT cell filled by a measured value from the same PCI/grid cell

    No interpolation is used in this product.  Measurement availability is saved separately so
    users can distinguish a measured location whose simulation value was retained from a cell that
    was actually filled from measurement.
    """
    sector_maps = np.asarray(pure_sector_maps, dtype=np.float32)
    filled = sector_maps.copy()
    source_mask = np.zeros_like(sector_maps, dtype=np.uint8)
    measurement_available_mask = np.zeros_like(sector_maps, dtype=bool)
    measured_grid = np.full_like(sector_maps, np.nan, dtype=np.float32)

    outdoor = ~np.asarray(grid.building_mask, dtype=bool)
    source_mask[np.isfinite(sector_maps) & outdoor[None, ...]] = 1
    filled[:, ~outdoor] = np.nan

    for sector_index, pci in enumerate(station.pcis):
        part = measurements.loc[
            measurements["pci"].eq(int(pci)),
            ["ix", "iy", "measured_rsrp_dbm"],
        ].copy()
        if part.empty:
            continue
        part = part.groupby(["ix", "iy"], as_index=False).agg(
            measured_rsrp_dbm=("measured_rsrp_dbm", "median")
        )
        ix = part["ix"].to_numpy(dtype=np.int64)
        iy = part["iy"].to_numpy(dtype=np.int64)
        values = part["measured_rsrp_dbm"].to_numpy(dtype=np.float32)
        valid = (
            (ix >= 0) & (ix < grid.nx) & (iy >= 0) & (iy < grid.ny)
            & outdoor[iy, ix] & np.isfinite(values)
        )
        ix, iy, values = ix[valid], iy[valid], values[valid]
        measurement_available_mask[sector_index, iy, ix] = True
        measured_grid[sector_index, iy, ix] = values
        missing_rt = ~np.isfinite(sector_maps[sector_index, iy, ix])
        if np.any(missing_rt):
            fx = ix[missing_rt]; fy = iy[missing_rt]; fv = values[missing_rt]
            filled[sector_index, fy, fx] = fv
            source_mask[sector_index, fy, fx] = 2

    return {
        "filled_sector_maps": filled.astype(np.float32),
        "value_source_mask": source_mask,
        "measurement_available_mask": measurement_available_mask,
        "measured_rsrp_grid_dbm": measured_grid,
    }


def compute_best_server(
    station: RuntimeStationConfig,
    sector_maps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    finite_any = np.any(np.isfinite(sector_maps), axis=0)
    best_index = np.argmax(
        np.where(np.isfinite(sector_maps), sector_maps, -np.inf),
        axis=0,
    )
    best_rsrp = np.take_along_axis(
        sector_maps,
        best_index[None, ...],
        axis=0,
    )[0]
    best_rsrp = np.where(finite_any, best_rsrp, np.nan).astype(np.float32)
    pci_values = np.asarray(station.pcis, dtype=np.int32)
    best_pci = np.where(finite_any, pci_values[best_index], -1).astype(np.int32)
    return best_rsrp, best_pci


# =============================================================================
# NPZ保存
# =============================================================================

def save_sector_npz(
    path: Path,
    station: RuntimeStationConfig,
    pci: int,
    map_version: str,
    rsrp_map: np.ndarray,
    grid: Any,
    metadata: Dict[str, Any],
    extra_arrays: Dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "station_id": np.asarray([station.station_id], dtype=np.int32),
        "station_label": np.asarray([station.label]),
        "pci": np.asarray([pci], dtype=np.int32),
        "map_version": np.asarray([map_version]),
        "x_m": grid.x_m.astype(np.float32),
        "y_m": grid.y_m.astype(np.float32),
        "ground_z_m": grid.ground_z_m.astype(np.float32),
        "receiver_z_m": grid.receiver_z_m.astype(np.float32),
        "building_mask": grid.building_mask.astype(np.bool_),
        "rsrp_dbm": np.asarray(rsrp_map, dtype=np.float32),
        "display_rsrp_dbm": _display_ready_map(
            rsrp_map,
            grid.building_mask,
            float(metadata["display_min_dbm"]),
            float(metadata["display_max_dbm"]),
        ).astype(np.float32),
        "tx_x_m": np.asarray([station.x_m], dtype=np.float32),
        "tx_y_m": np.asarray([station.y_m], dtype=np.float32),
        "height_agl_m": np.asarray([metadata["height_agl_m"]], dtype=np.float32),
        "tx_absolute_z_m": np.asarray([metadata["tx_absolute_z_m"]], dtype=np.float32),
        "shared_power_dbm": np.asarray([metadata["shared_power_dbm"]], dtype=np.float32),
        "azimuth_offset_deg": np.asarray([metadata["azimuth_offset_deg"]], dtype=np.float32),
        "absolute_downtilt_deg": np.asarray([metadata["absolute_downtilt_deg"]], dtype=np.float32),
        "alphas_rad": np.asarray(metadata["alphas_rad"], dtype=np.float64),
        "beta_rad": np.asarray([metadata["beta_rad"]], dtype=np.float64),
        "final_dense_map_rmse_db": np.asarray([metadata["final_dense_map_rmse_db"]], dtype=np.float32),
        "measurement_hit_rate": np.asarray([metadata["measurement_hit_rate"]], dtype=np.float32),
        "display_min_dbm": np.asarray([metadata["display_min_dbm"]], dtype=np.float32),
        "display_max_dbm": np.asarray([metadata["display_max_dbm"]], dtype=np.float32),
    }
    for key, value in (extra_arrays or {}).items():
        payload[str(key)] = np.asarray(value)
    np.savez_compressed(path, **payload)


def _best_server_source_mask(station: RuntimeStationConfig, best_server_pci: np.ndarray, sector_source_mask: np.ndarray) -> np.ndarray:
    out = np.zeros(np.asarray(best_server_pci).shape, dtype=np.uint8)
    for idx, pci in enumerate(station.pcis):
        hit = np.asarray(best_server_pci) == int(pci)
        out[hit] = np.asarray(sector_source_mask[idx], dtype=np.uint8)[hit]
    return out


def save_station_combined_npz(
    path: Path,
    station: RuntimeStationConfig,
    map_version: str,
    sector_maps: np.ndarray,
    best_server_map: np.ndarray,
    best_server_pci: np.ndarray,
    grid: Any,
    metadata: Dict[str, Any],
    extra_arrays: Dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    display_sector_maps = np.stack(
        [
            _display_ready_map(
                sector_maps[index],
                grid.building_mask,
                float(metadata["display_min_dbm"]),
                float(metadata["display_max_dbm"]),
            )
            for index in range(len(sector_maps))
        ],
        axis=0,
    ).astype(np.float32)
    payload: dict[str, Any] = {
        "station_id": np.asarray([station.station_id], dtype=np.int32),
        "station_label": np.asarray([station.label]),
        "pcis": np.asarray(station.pcis, dtype=np.int32),
        "map_version": np.asarray([map_version]),
        "x_m": grid.x_m.astype(np.float32),
        "y_m": grid.y_m.astype(np.float32),
        "ground_z_m": grid.ground_z_m.astype(np.float32),
        "receiver_z_m": grid.receiver_z_m.astype(np.float32),
        "building_mask": grid.building_mask.astype(np.bool_),
        "sector_rsrp_dbm": np.asarray(sector_maps, dtype=np.float32),
        "display_sector_rsrp_dbm": display_sector_maps,
        "best_server_rsrp_dbm": np.asarray(best_server_map, dtype=np.float32),
        "display_best_server_rsrp_dbm": _display_ready_map(
            best_server_map,
            grid.building_mask,
            float(metadata["display_min_dbm"]),
            float(metadata["display_max_dbm"]),
        ).astype(np.float32),
        "best_server_pci": np.asarray(best_server_pci, dtype=np.int32),
        "height_agl_m": np.asarray([metadata["height_agl_m"]], dtype=np.float32),
        "tx_absolute_z_m": np.asarray([metadata["tx_absolute_z_m"]], dtype=np.float32),
        "shared_power_dbm": np.asarray([metadata["shared_power_dbm"]], dtype=np.float32),
        "azimuth_offset_deg": np.asarray([metadata["azimuth_offset_deg"]], dtype=np.float32),
        "absolute_downtilt_deg": np.asarray([metadata["absolute_downtilt_deg"]], dtype=np.float32),
        "alphas_rad": np.asarray(metadata["alphas_rad"], dtype=np.float64),
        "beta_rad": np.asarray([metadata["beta_rad"]], dtype=np.float64),
        "measurement_hit_rate": np.asarray([metadata["measurement_hit_rate"]], dtype=np.float32),
    }
    for key, value in (extra_arrays or {}).items():
        payload[str(key)] = np.asarray(value)
    np.savez_compressed(path, **payload)


# =============================================================================
# Data in Brief风格绘图
# =============================================================================

def _white_building_rgba(building_mask: np.ndarray) -> np.ndarray:
    """Return an RGBA overlay that renders building cells as opaque white blocks."""
    mask = np.asarray(building_mask, dtype=bool)
    rgba = np.zeros(mask.shape + (4,), dtype=np.uint8)
    rgba[mask] = np.asarray([255, 255, 255, 255], dtype=np.uint8)
    return rgba


def draw_building_blocks(ax: Any, building_mask: np.ndarray, extent: Sequence[float]) -> None:
    """Overlay building footprints as solid white blocks on a radio-map axes."""
    mask = np.asarray(building_mask, dtype=bool)
    if mask.size == 0 or not np.any(mask):
        return
    ax.imshow(
        _white_building_rgba(mask),
        origin="lower",
        extent=extent,
        interpolation="nearest",
        zorder=7,
    )


def _display_ready_map(
    rsrp_map: np.ndarray,
    building_mask: np.ndarray,
    display_min_dbm: float,
    display_max_dbm: float,
) -> np.ndarray:
    """Prepare the numeric background used only for PNG display.

    Outdoor NaN / no-hit cells are clipped to the display floor. Building cells
    are intentionally not inpainted: they are covered by an opaque white
    building-footprint overlay during plotting. Raw NPZ arrays remain unchanged.
    """
    arr = np.asarray(rsrp_map, dtype=np.float32).copy()
    arr[~np.isfinite(arr)] = float(display_min_dbm)
    np.clip(arr, float(display_min_dbm), float(display_max_dbm), out=arr)
    return arr


FULL_PAGE_WIDTH_IN = 7.48  # 7480 px at 1000 dpi
FIGURE_HEIGHT_IN = 5.61
MAP_DPI = 300
COMPARISON_DPI = 1000
SQUARE_MAP_SIZE_PX = 512


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
    base = Path(output_path).with_suffix('')
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.set_size_inches(*_publication_figsize_inches(), forward=True)
    _style_publication_text(fig)
    fc = facecolor if facecolor is not None else fig.get_facecolor()
    out = base.with_suffix('.png')
    fig.savefig(out, format='png', dpi=int(dpi), bbox_inches=None, pad_inches=0.0, facecolor=fc)
    return out


def _save_square_png_exact(
    fig,
    output_path: Path,
    size_px: int = SQUARE_MAP_SIZE_PX,
    dpi: int = MAP_DPI,
    facecolor: str | None = None,
) -> Path:
    """Save the formal single-station radio map as exactly 512×512 pixels.

    The reference figure supplied with the project is 512×512 pixels at 300 dpi.
    ``bbox_inches='tight'`` is deliberately not used because it changes the output
    pixel dimensions and can crop/expand the 512 m × 512 m map canvas.
    """
    from PIL import Image

    base = Path(output_path).with_suffix('')
    base.parent.mkdir(parents=True, exist_ok=True)
    dpi = int(dpi)
    if dpi <= 0:
        raise ValueError(f"dpi must be positive, got {dpi}")
    size_in = float(size_px) / float(dpi)
    fig.set_size_inches(size_in, size_in, forward=True)
    fc = facecolor if facecolor is not None else fig.get_facecolor()
    out = base.with_suffix('.png')
    fig.savefig(
        out,
        format='png',
        dpi=dpi,
        bbox_inches=None,
        pad_inches=0.0,
        facecolor=fc,
    )

    # Hard verification: publication-facing single-station maps must remain 512×512 px.
    with Image.open(out) as im:
        if im.size != (int(size_px), int(size_px)):
            raise RuntimeError(
                f"radio-map PNG size mismatch: expected {size_px}×{size_px}px, got {im.size}"
            )
    return out


def plot_rsrp_map(
    path: Path,
    station: RuntimeStationConfig,
    pci: int | None,
    map_version_label: str,
    rsrp_map: np.ndarray,
    grid: Any,
    metadata: Dict[str, Any],
    dpi: int,
    filled_count: int | None = None,
) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    display_min_dbm = float(metadata["display_min_dbm"])
    display_max_dbm = float(metadata["display_max_dbm"])
    display_map = _display_ready_map(
        rsrp_map=rsrp_map,
        building_mask=grid.building_mask,
        display_min_dbm=display_min_dbm,
        display_max_dbm=display_max_dbm,
    )

    # 正式单站 512m×512m 无线电地图：输出严格固定为 512×512 像素，避免额外黑边框。
    dpi = int(dpi)
    fig = plt.figure(
        figsize=(SQUARE_MAP_SIZE_PX / dpi, SQUARE_MAP_SIZE_PX / dpi),
        dpi=dpi,
        facecolor="#24133b",
    )
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_facecolor("#24133b")
    ax.imshow(
        display_map,
        origin="lower",
        extent=grid.extent,
        vmin=display_min_dbm,
        vmax=display_max_dbm,
        cmap="viridis",
        interpolation="nearest",
        zorder=1,
    )

    draw_building_blocks(ax, grid.building_mask, grid.extent)

    ax.scatter(
        [station.x_m],
        [station.y_m],
        marker="^",
        s=34,
        c="#e53935",
        edgecolors="#ffd54f",
        linewidths=0.55,
        zorder=9,
    )
    ax.text(
        station.x_m + 5.0,
        station.y_m + 2.5,
        f"tx-{station.station_id}",
        color="white",
        fontsize=5.0,
        ha="left",
        va="center",
        bbox=dict(
            boxstyle="round,pad=0.08",
            facecolor=(0, 0, 0, 0.42),
            edgecolor=(1, 1, 1, 0.22),
            linewidth=0.25,
        ),
        zorder=10,
    )

    # 只保留用户要求的实测位置命中率，避免标题过重。
    if pci is None:
        label = "Omni" if station.is_omnidirectional else "Best-server"
    else:
        label = f"PCI {int(pci)}"
    info = (
        f"Station {station.station_id:02d} | {label} | {map_version_label}\n"
        f"Measurement-location hit={float(metadata['measurement_hit_rate']):.1%}"
    )
    if filled_count is not None:
        info += f" | filled={int(filled_count):,}"
    ax.text(
        grid.extent[0] + 8.0,
        grid.extent[3] - 8.0,
        info,
        color="white",
        fontsize=5.2,
        ha="left",
        va="top",
        bbox=dict(
            boxstyle="round,pad=0.12",
            facecolor=(0, 0, 0, 0.24),
            edgecolor=(1, 1, 1, 0.15),
            linewidth=0.2,
        ),
        zorder=10,
    )

    ax.set_xlim(grid.extent[0], grid.extent[1])
    ax.set_ylim(grid.extent[2], grid.extent[3])
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    _save_square_png_exact(
        fig,
        path,
        size_px=SQUARE_MAP_SIZE_PX,
        dpi=dpi,
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)


# =============================================================================
# 单站处理
# =============================================================================

def process_station(
    station: RuntimeStationConfig,
    summary_row: pd.Series,
    observations: pd.DataFrame,
    terrain: TerrainModel,
    scene: Any,
    scene_xml: Path,
    building_triangles_xy: np.ndarray,
    output_root: Path,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    sid = int(station.station_id)
    station_dir = output_root / f"station_{sid:02d}"
    work_dir = station_dir / "work"
    work_dir.mkdir(parents=True, exist_ok=True)

    terrain.assert_map_inside(
        station.x_m,
        station.y_m,
        MAP_SIZE_M,
        MAP_SIZE_M,
    )

    grid, grid_diagnostics = create_dense_grid_with_building_mask(
        terrain=terrain,
        building_triangles_xy=building_triangles_xy,
        center_x=station.x_m,
        center_y=station.y_m,
        size_x_m=MAP_SIZE_M,
        size_y_m=MAP_SIZE_M,
        cell_size_m=CELL_SIZE_M,
        rx_height_agl_m=float(args.rx_height_agl_m),
        buffer_cells=int(args.building_mask_buffer_cells),
    )
    # Formal per-station product: exactly 512 m × 512 m at 1 m resolution.
    expected_shape = (int(MAP_SIZE_M / CELL_SIZE_M), int(MAP_SIZE_M / CELL_SIZE_M))
    if (int(grid.ny), int(grid.nx)) != expected_shape:
        raise RuntimeError(
            f"station {sid:02d} grid shape mismatch: expected {expected_shape}, "
            f"got {(int(grid.ny), int(grid.nx))}"
        )
    extent_width = float(grid.extent[1] - grid.extent[0])
    extent_height = float(grid.extent[3] - grid.extent[2])
    if not (abs(extent_width - MAP_SIZE_M) < 1e-9 and abs(extent_height - MAP_SIZE_M) < 1e-9):
        raise RuntimeError(
            f"station {sid:02d} map extent mismatch: expected 512 m × 512 m, "
            f"got {extent_width:g} m × {extent_height:g} m"
        )
    surface = build_dense_outdoor_measurement_surface(
        terrain=terrain,
        grid=grid,
        rx_height_agl_m=float(args.rx_height_agl_m),
        output_path=work_dir / f"station_{sid:02d}_dem_plus_1p5m_outdoor_surface.ply",
    )

    measurements, measurements_inside = prepare_station_measurement_grid(
        observations=observations,
        station=station,
        building_mask=grid.building_mask,
    )
    measurements.to_csv(
        station_dir / "measurement_cells_outdoor_1m.csv",
        index=False,
        encoding="utf-8-sig",
    )
    measurements_inside.to_csv(
        station_dir / "measurement_cells_inside_buildings_excluded.csv",
        index=False,
        encoding="utf-8-sig",
    )

    height_agl_m = _float_value(summary_row, "height_agl_m", 30.0)
    azimuth_offset_deg = _float_value(summary_row, "azimuth_offset_deg", 0.0)
    absolute_downtilt_deg = _float_value(summary_row, "absolute_downtilt_deg", 0.0)
    shared_power_dbm = _float_value(summary_row, "shared_power_dbm", 53.5)
    if "sss_re_epre_offset_db" not in summary_row.index or not np.isfinite(pd.to_numeric(summary_row["sss_re_epre_offset_db"], errors="coerce")):
        raise ValueError("单站制图要求summary包含有限的全网sss_re_epre_offset_db")
    sss_re_epre_offset_db = float(summary_row["sss_re_epre_offset_db"])
    if station.is_omnidirectional:
        # 22号站为单PCI全向站：方向角与下倾角不参与物理建模。
        azimuth_offset_deg = 0.0
        absolute_downtilt_deg = 0.0
    final_rmse_db = _float_value(summary_row, "final_dense_map_rmse_db", float("nan"))

    final_batches = (
        int(args.final_batches)
        if args.final_batches is not None
        else _int_value(summary_row, "final_batch_count", 5)
    )
    final_samples_per_batch = (
        int(args.final_samples_per_batch)
        if args.final_samples_per_batch is not None
        else _int_value(summary_row, "final_samples_per_batch", 10_000_000)
    )
    final_max_depth = 5
    final_edge_diffraction = True

    station_cfg = make_station_cfg(
        scene_xml=scene_xml,
        station=station,
        rx_height_agl_m=float(args.rx_height_agl_m),
        max_depth=5,
        edge_diffraction=True,
    )
    station_cfg["radio"]["diffraction"] = True
    station_cfg["radio"]["sss_re_epre_offset_db"] = float(sss_re_epre_offset_db)
    configure_tx_array_for_station(scene, station_cfg, station)

    candidate = Candidate(
        height_agl_m=height_agl_m,
        azimuth_offset_deg=azimuth_offset_deg,
        downtilt_delta_deg=absolute_downtilt_deg,
        reference_power_dbm=shared_power_dbm,
    )

    simulation = run_candidate_multibatch_linear_average(
        scene=scene,
        terrain=terrain,
        station=station,
        candidate=candidate,
        surface=surface,
        cfg=station_cfg,
        samples_per_batch=final_samples_per_batch,
        batch_count=final_batches,
        seed_step=int(args.final_seed_step),
        cache_dir=station_dir / "cache_final_maps",
        force=bool(args.force),
    )

    evaluation = evaluate_prediction(
        station=station,
        measurements=measurements,
        surface=surface,
        sector_rsrp_at_reference_dbm=np.asarray(simulation["sector_rsrp_dbm"], dtype=np.float32),
        reference_power_dbm=shared_power_dbm,
        power_candidates_dbm=np.asarray([shared_power_dbm], dtype=np.float64),
    )
    hit_stats = compute_measurement_hit_statistics(station, measurements, evaluation)

    pure_sector_maps = sector_values_to_full_maps(
        surface=surface,
        sector_values=np.asarray(simulation["sector_rsrp_dbm"], dtype=np.float32),
        grid=grid,
        fill_value=np.nan,
    ).astype(np.float32)
    pure_best_map, pure_best_pci = compute_best_server(station, pure_sector_maps)

    reconstruction = fill_missing_simulation_cells_from_measurements(
        station=station, grid=grid, pure_sector_maps=pure_sector_maps, measurements=measurements
    )
    reconstructed_sector_maps = reconstruction["filled_sector_maps"]
    reconstructed_best_map, reconstructed_best_pci = compute_best_server(
        station,
        reconstructed_sector_maps,
    )

    metadata = {
        "station_id": sid,
        "station_label": station.label,
        "pcis": [int(v) for v in station.pcis],
        "is_omnidirectional": bool(station.is_omnidirectional),
        "height_agl_m": height_agl_m,
        "tx_absolute_z_m": float(simulation["tx_z_m"]),
        "shared_power_dbm": shared_power_dbm,
        "ss_rsrp_method": "3gpp_ts_38_215_sss_re_uniform_epre_narrowband_global_calibrated",
        "sss_re_count": 127,
        "active_subcarrier_count": DEFAULT_N_RB * DEFAULT_SUBCARRIERS_PER_RB,
        "sss_re_epre_offset_db": sss_re_epre_offset_db,
        "sss_re_tx_epre_dbm": shared_power_dbm - 10.0 * math.log10(float(DEFAULT_N_RB * DEFAULT_SUBCARRIERS_PER_RB)) + sss_re_epre_offset_db,
        "station_specific_sss_re_offset_fitted": False,
        "global_epre_offset_source_subset": "calibration_trajectories_only",
        "tx_power_allocation_mode": "uniform_over_configured_active_subcarriers",
        "azimuth_offset_deg": azimuth_offset_deg,
        "absolute_downtilt_deg": absolute_downtilt_deg,
        "alphas_rad": [float(v) for v in simulation["alphas_rad"]],
        "beta_rad": float(simulation["beta_rad"]),
        "final_dense_map_rmse_db": final_rmse_db,
        "measurement_hit_rate": float(hit_stats["measurement_hit_rate"]),
        "measurement_cell_count": int(hit_stats["measurement_cell_count"]),
        "simulated_hit_count_at_measurement_cells": int(hit_stats["simulated_hit_count"]),
        "final_batches": final_batches,
        "final_samples_per_batch": final_samples_per_batch,
        "final_total_samples_per_tx": int(simulation["total_samples_per_tx"]),
        "final_max_depth": final_max_depth,
        "final_edge_diffraction": final_edge_diffraction,
        "display_min_dbm": float(args.display_min_dbm),
        "display_max_dbm": float(args.display_max_dbm),
        "map_size_x_m": float(MAP_SIZE_M),
        "map_size_y_m": float(MAP_SIZE_M),
        "cell_size_m": float(CELL_SIZE_M),
        "grid_nx": int(grid.nx),
        "grid_ny": int(grid.ny),
        "png_size_px": int(SQUARE_MAP_SIZE_PX),
        "png_dpi": int(args.dpi),
        "grid_diagnostics": grid_diagnostics,
        "white_region_display_policy": (
            "Outdoor NaN/no-hit cells are displayed as -120 dBm. Building-mask cells "
            "copy the nearest outdoor display color only for PNG rendering; raw NPZ values "
            "remain masked/NaN."
        ),
        "measurement_filled_definition": (
            "The measurement-filled product starts from the original Sionna RT map. A measured "
            "RSRP value is inserted only when the corresponding same-PCI, same-grid Sionna RT "
            "cell is missing. Existing finite Sionna RT values are never overwritten and no "
            "spatial interpolation is applied in this formal product."
        ),
        "provenance_definition": {
            "0": "missing_or_invalid",
            "1": "original_sionna_rt",
            "2": "measurement_filled_same_grid",
        },
    }

    pure_png_dir = station_dir / "01_pure_simulation" / "figures"
    pure_npz_dir = station_dir / "01_pure_simulation" / "npz"
    reconstructed_png_dir = station_dir / "02_measurement_filled" / "figures"
    reconstructed_npz_dir = station_dir / "02_measurement_filled" / "npz"
    for directory in [pure_png_dir, pure_npz_dir, reconstructed_png_dir, reconstructed_npz_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    exact_counts: dict[int, int] = {}
    mean_abs_corrections: dict[int, float] = {}

    for index, pci in enumerate(station.pcis):
        pci = int(pci)
        pci_hit_rate = float(hit_stats["per_pci"][pci]["measurement_hit_rate"])
        sector_metadata = dict(metadata)
        sector_metadata["measurement_hit_rate"] = pci_hit_rate
        source_mask_sector = reconstruction["value_source_mask"][index]
        exact_count = int(reconstruction["measurement_available_mask"][index].sum())
        exact_counts[pci] = exact_count
        mean_abs_corrections[pci] = float("nan")

        pure_stem = f"station_{sid:02d}_pci_{pci}_pure_simulation"
        plot_rsrp_map(
            path=pure_png_dir / f"{pure_stem}.png",
            station=station,
            pci=pci,
            map_version_label="Pure simulation",
            rsrp_map=pure_sector_maps[index],
            grid=grid,
            metadata=sector_metadata,
            dpi=args.dpi,
        )
        save_sector_npz(
            path=pure_npz_dir / f"{pure_stem}.npz",
            station=station,
            pci=pci,
            map_version="pure_simulation",
            rsrp_map=pure_sector_maps[index],
            grid=grid,
            metadata=sector_metadata,
            extra_arrays={
                "raw_no_hit_mask": (~np.isfinite(pure_sector_maps[index])) & (~grid.building_mask),
                "value_source_mask": np.where(np.isfinite(pure_sector_maps[index]) & (~grid.building_mask), 1, 0).astype(np.uint8),
                "measurement_available_mask": reconstruction["measurement_available_mask"][index],
                "provenance_codebook": np.asarray(["0=missing_or_invalid;1=original_sionna_rt;2=measurement_filled"]),
            },
        )

        reconstructed_stem = f"station_{sid:02d}_pci_{pci}_measurement_filled"
        plot_rsrp_map(
            path=reconstructed_png_dir / f"{reconstructed_stem}.png",
            station=station,
            pci=pci,
            map_version_label="Measurement-filled",
            rsrp_map=reconstructed_sector_maps[index],
            grid=grid,
            metadata=sector_metadata,
            dpi=args.dpi,
            filled_count=int((source_mask_sector == 2).sum()),
        )
        save_sector_npz(
            path=reconstructed_npz_dir / f"{reconstructed_stem}.npz",
            station=station,
            pci=pci,
            map_version="measurement_filled",
            rsrp_map=reconstructed_sector_maps[index],
            grid=grid,
            metadata=sector_metadata,
            extra_arrays={
                "pure_sim_rsrp_dbm": pure_sector_maps[index],
                "measured_rsrp_grid_dbm": reconstruction["measured_rsrp_grid_dbm"][index],
                "measurement_available_mask": reconstruction["measurement_available_mask"][index],
                "value_source_mask": reconstruction["value_source_mask"][index],
                "provenance_codebook": np.asarray(["0=missing_or_invalid;1=original_sionna_rt;2=measurement_filled"]),
                "fill_method": np.asarray(["same_grid_measurement_only_when_sionna_missing"]),
            },
        )

    pure_best_stem = f"station_{sid:02d}_best_server_pure_simulation"
    plot_rsrp_map(
        path=pure_png_dir / f"{pure_best_stem}.png",
        station=station,
        pci=None,
        map_version_label="Pure simulation",
        rsrp_map=pure_best_map,
        grid=grid,
        metadata=metadata,
        dpi=args.dpi,
    )
    save_station_combined_npz(
        path=pure_npz_dir / f"{pure_best_stem}.npz",
        station=station,
        map_version="pure_simulation",
        sector_maps=pure_sector_maps,
        best_server_map=pure_best_map,
        best_server_pci=pure_best_pci,
        grid=grid,
        metadata=metadata,
        extra_arrays={
            "raw_no_hit_mask_per_sector": (~np.isfinite(pure_sector_maps)) & (~grid.building_mask[None, ...]),
            "value_source_mask_per_sector": np.where(np.isfinite(pure_sector_maps) & (~grid.building_mask[None, ...]), 1, 0).astype(np.uint8),
            "best_server_value_source_mask": _best_server_source_mask(
                station, pure_best_pci, np.where(np.isfinite(pure_sector_maps) & (~grid.building_mask[None, ...]), 1, 0).astype(np.uint8)
            ),
            "measurement_available_mask_per_sector": reconstruction["measurement_available_mask"],
            "provenance_codebook": np.asarray(["0=missing_or_invalid;1=original_sionna_rt;2=measurement_filled"]),
        },
    )

    reconstructed_best_stem = f"station_{sid:02d}_best_server_measurement_filled"
    reconstructed_best_source_mask = _best_server_source_mask(
        station, reconstructed_best_pci, reconstruction["value_source_mask"]
    )
    plot_rsrp_map(
        path=reconstructed_png_dir / f"{reconstructed_best_stem}.png",
        station=station,
        pci=None,
        map_version_label="Measurement-filled",
        rsrp_map=reconstructed_best_map,
        grid=grid,
        metadata=metadata,
        dpi=args.dpi,
        filled_count=int((reconstructed_best_source_mask == 2).sum()),
    )
    save_station_combined_npz(
        path=reconstructed_npz_dir / f"{reconstructed_best_stem}.npz",
        station=station,
        map_version="measurement_filled",
        sector_maps=reconstructed_sector_maps,
        best_server_map=reconstructed_best_map,
        best_server_pci=reconstructed_best_pci,
        grid=grid,
        metadata=metadata,
        extra_arrays={
            "pure_sector_rsrp_dbm": pure_sector_maps,
            "measured_rsrp_grid_dbm": reconstruction["measured_rsrp_grid_dbm"],
            "measurement_available_mask_per_sector": reconstruction["measurement_available_mask"],
            "value_source_mask_per_sector": reconstruction["value_source_mask"],
            "best_server_value_source_mask": reconstructed_best_source_mask,
            "provenance_codebook": np.asarray(["0=missing_or_invalid;1=original_sionna_rt;2=measurement_filled"]),
            "fill_method": np.asarray(["same_grid_measurement_only_when_sionna_missing"]),
        },
    )

    metadata["exact_measurement_cells_per_pci"] = exact_counts
    metadata["measurement_available_cells_per_pci"] = exact_counts
    metadata["measurement_filled_cell_count_per_pci"] = {
        int(pci): int((reconstruction["value_source_mask"][idx] == 2).sum())
        for idx, pci in enumerate(station.pcis)
    }
    metadata["measurement_hit_per_pci"] = hit_stats["per_pci"]
    metadata["output_directories"] = {
        "pure_png": str(pure_png_dir),
        "pure_npz": str(pure_npz_dir),
        "measurement_filled_png": str(reconstructed_png_dir),
        "measurement_filled_npz": str(reconstructed_npz_dir),
    }
    (station_dir / "station_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metadata


# =============================================================================
# 主函数
# =============================================================================

def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).expanduser().resolve()

    measurements_csv = resolve_measurements(project_root, args.measurements)
    summary_csv = resolve_summary(project_root, args.summary_csv)
    ground_ply = resolve_ground(project_root, args.ground)
    building_paths = resolve_buildings(project_root, args.buildings)
    output_root = (
        Path(args.output_root).expanduser().resolve()
        if args.output_root
        else (project_root / "outputs" / "bestparam_radio_maps_512m").resolve()
    )
    output_root.mkdir(parents=True, exist_ok=True)
    work_root = output_root / "work"
    work_root.mkdir(parents=True, exist_ok=True)

    install_general_sector_support()

    print("=" * 100)
    print(f"处理后实测长表：{measurements_csv}")
    print(f"最佳参数汇总：  {summary_csv}")
    print(f"地形PLY：        {ground_ply}")
    print("建筑PLY：")
    for path in building_paths:
        print(f"  - {path}")
    print(f"输出目录：       {output_root}")
    print("=" * 100)

    summary = load_best_parameter_summary(summary_csv)
    selected_ids = select_station_ids(args.stations, summary["station_id"].tolist())
    summary = summary.loc[summary["station_id"].isin(selected_ids)].copy()

    raw_long, observations = read_27station_long_measurements(measurements_csv)
    stations, direction_diagnostics = build_all_27_station_configs(
        raw_long=raw_long,
        top_fraction=float(args.direction_top_fraction),
        initial_power_dbm=53.5,
    )
    pd.DataFrame(direction_diagnostics).to_csv(
        output_root / "estimated_initial_directions_27stations.csv",
        index=False,
        encoding="utf-8-sig",
    )

    missing_station_configs = sorted(set(selected_ids) - set(stations))
    if missing_station_configs:
        raise KeyError(f"实测长表中无法构建这些站的配置：{missing_station_configs}")

    terrain = TerrainModel.load(ground_ply)
    scene_xml = work_root / "generated_scene.xml"
    scene_report = build_scene_xml_multi(
        ground_ply=ground_ply,
        building_candidates=building_paths,
        output_xml=scene_xml,
        cleaned_dir=work_root / "scene_mesh_cache",
        ground_material="itu_wet_ground",
        building_material="itu_concrete",
        allow_no_buildings=False,
    )
    (output_root / "scene_report.json").write_text(
        json.dumps(scene_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    building_triangles_xy, building_diagnostics = load_building_projection_triangles(
        scene_report["building_included_paths"]
    )
    (output_root / "building_projection_diagnostics.json").write_text(
        json.dumps(
            {
                "total_nondegenerate_xy_triangles": int(len(building_triangles_xy)),
                "files": building_diagnostics,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # 先用三扇区配置初始化场景；每个站开始前会重新设置TX阵列。
    bootstrap_station = next(
        (stations[sid] for sid in selected_ids if not stations[sid].is_omnidirectional),
        stations[selected_ids[0]],
    )
    bootstrap_cfg = make_station_cfg(
        scene_xml=scene_xml,
        station=bootstrap_station,
        rx_height_agl_m=float(args.rx_height_agl_m),
        max_depth=5,
        edge_diffraction=True,
    )
    scene = configure_scene(scene_xml, bootstrap_cfg)

    completed: list[Dict[str, Any]] = []
    failures: list[Dict[str, Any]] = []

    for _, summary_row in summary.iterrows():
        sid = int(summary_row["station_id"])
        print(f"\n{'=' * 35} Station {sid:02d} {'=' * 35}")
        try:
            metadata = process_station(
                station=stations[sid],
                summary_row=summary_row,
                observations=observations,
                terrain=terrain,
                scene=scene,
                scene_xml=scene_xml,
                building_triangles_xy=building_triangles_xy,
                output_root=output_root,
                args=args,
            )
            completed.append(metadata)
            print(
                f"Station {sid:02d}完成："
                f"measurement-hit={metadata['measurement_hit_rate']:.1%}, "
            )
        except Exception as exc:
            failure = {
                "station_id": sid,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            failures.append(failure)
            print(f"Station {sid:02d}失败：{type(exc).__name__}: {exc}")
            if not args.continue_on_error:
                raise

    if completed:
        summary_rows = []
        for item in completed:
            summary_rows.append(
                {
                    "station_id": item["station_id"],
                    "station_label": item["station_label"],
                    "pcis": ";".join(map(str, item["pcis"])),
                    "height_agl_m": item["height_agl_m"],
                    "tx_absolute_z_m": item["tx_absolute_z_m"],
                    "shared_power_dbm": item["shared_power_dbm"],
                    "azimuth_offset_deg": item["azimuth_offset_deg"],
                    "absolute_downtilt_deg": item["absolute_downtilt_deg"],
                    "final_dense_map_rmse_db": item["final_dense_map_rmse_db"],
                    "measurement_hit_rate": item["measurement_hit_rate"],
                    "measurement_cell_count": item["measurement_cell_count"],
                    "simulated_hit_count_at_measurement_cells": item[
                        "simulated_hit_count_at_measurement_cells"
                    ],
                    "final_batches": item["final_batches"],
                    "final_samples_per_batch": item["final_samples_per_batch"],
                    "final_total_samples_per_tx": item["final_total_samples_per_tx"],
                    "final_max_depth": item["final_max_depth"],
                    "final_edge_diffraction": item["final_edge_diffraction"],
                }
            )
        pd.DataFrame(summary_rows).to_csv(
            output_root / "bestparam_radio_map_export_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )

    if failures:
        pd.DataFrame(failures).to_csv(
            output_root / "bestparam_radio_map_export_failures.csv",
            index=False,
            encoding="utf-8-sig",
        )

    print("\n" + "=" * 100)
    print(f"全部处理结束：成功={len(completed)}, 失败={len(failures)}")
    print(f"输出目录：{output_root}")
    print("=" * 100)


if __name__ == "__main__":
    main()
