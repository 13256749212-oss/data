#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_joint_best_server_4000x3000.py

读取27站参数校准结果，直接生成一张4000 m × 3000 m、1 m网格的
27站/79 PCI联合best-server无线电地图。

关键原则
--------
1. 不重新导出或保存27张单站无线电地图；
2. 所有物理基站使用各自调参后的最佳高度、共享功率、方向角偏移和绝对下倾角；
3. 接收面为真实DEM+1.5 m，建筑参与传播但建筑内部不属于室外接收域；
4. 4000×3000 m区域按500×500 m tile运行，支持断点续跑；
5. 每个tile内部按基站批次计算，立即更新best-server并释放中间结果；
6. 最终只保存联合RSRP、最佳物理站、最佳PCI和最佳扇区索引；正式PNG中的建筑统一显示为白色实体块。

默认输入
--------
outputs/parameter_calibration/all_27stations_summary.csv
outputs/parameter_calibration/estimated_initial_directions_27stations.csv
assets/ground.ply
assets/ynu_chenggong_campus-001.ply

默认输出
--------
outputs/joint_best_server_4000x3000/
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import export_bestparam_radio_maps as base
from core_dem15m import (
    RuntimeStationConfig,
    build_dense_outdoor_measurement_surface,
    build_scene_xml_multi,
    create_dense_grid_with_building_mask,
    load_building_projection_triangles,
)
from src.angles import apply_common_azimuth_offset, downtilt_to_sionna_beta_rad
from src.simulator import clear_transmitters, configure_scene
from src.terrain import TerrainModel

DEFAULT_CENTER_X = 267.5
DEFAULT_CENTER_Y = 69.53
DEFAULT_SIZE_X = 4000
DEFAULT_SIZE_Y = 3000
DEFAULT_CELL_SIZE = 1.0
DEFAULT_TILE_SIZE = 500
DEFAULT_RX_HEIGHT_AGL = 1.5
DEFAULT_SEED = 20260805
DEFAULT_SEED_STEP = 1009
DEFAULT_STATION_BATCH_SIZE = 4
DEFAULT_DISPLAY_MIN_DBM = -120.0
DEFAULT_DISPLAY_MAX_DBM = -40.0
EXPECTED_STATION_COUNT = 27
EXPECTED_SECTOR_COUNT = 79


@dataclass(frozen=True)
class TileSpec:
    ix: int
    iy: int
    center_x: float
    center_y: float
    size_x_m: int
    size_y_m: int
    global_ix0: int
    global_ix1: int
    global_iy0: int
    global_iy1: int

    @property
    def key(self) -> str:
        return f"tile_y{self.iy:02d}_x{self.ix:02d}"


@dataclass(frozen=True)
class TxDescriptor:
    global_sector_index: int
    station_id: int
    pci: int
    tx_name: str
    tx_x_m: float
    tx_y_m: float
    tx_z_m: float
    alpha_rad: float
    beta_rad: float
    power_dbm: float
    is_omnidirectional: bool


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "使用27站调参最佳参数，直接生成4000×3000 m联合best-server无线电地图；"
            "不生成27张单站地图。"
        )
    )
    p.add_argument("--project-root", default=str(Path(__file__).resolve().parents[2]))
    p.add_argument("--summary-csv", default=None)
    p.add_argument("--directions-csv", default=None)
    p.add_argument("--ground", default=None)
    p.add_argument("--buildings", nargs="*", default=None)
    p.add_argument("--output-root", default=None)
    p.add_argument("--center-x", type=float, default=DEFAULT_CENTER_X)
    p.add_argument("--center-y", type=float, default=DEFAULT_CENTER_Y)
    p.add_argument("--size-x", type=int, default=DEFAULT_SIZE_X)
    p.add_argument("--size-y", type=int, default=DEFAULT_SIZE_Y)
    p.add_argument("--cell-size", type=float, default=DEFAULT_CELL_SIZE)
    p.add_argument("--tile-size", type=int, default=DEFAULT_TILE_SIZE)
    p.add_argument("--rx-height-agl-m", type=float, default=DEFAULT_RX_HEIGHT_AGL)
    p.add_argument("--building-mask-buffer-cells", type=int, default=1)
    p.add_argument("--station-batch-size", type=int, default=DEFAULT_STATION_BATCH_SIZE)
    p.add_argument("--batch-count", type=int, default=None,
                   help="独立seed批次数；默认读取调参汇总final_batch_count")
    p.add_argument("--samples-per-tx", type=int, default=None,
                   help="每批每TX采样数；默认读取调参汇总final_samples_per_batch")
    p.add_argument("--max-depth", type=int, default=None,
                   help="传播最大深度；默认读取调参汇总final_max_depth")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--seed-step", type=int, default=DEFAULT_SEED_STEP)
    p.add_argument("--display-min-dbm", type=float, default=DEFAULT_DISPLAY_MIN_DBM)
    p.add_argument("--display-max-dbm", type=float, default=DEFAULT_DISPLAY_MAX_DBM)
    p.add_argument("--dpi", type=int, default=1000)
    p.add_argument("--quick", action="store_true",
                   help="流程试跑：1个seed、每TX最多100000样本、max_depth最多3")
    p.add_argument("--dry-run", action="store_true",
                   help="只检查最佳参数、方向文件、地图范围和tile划分，不调用Sionna")
    p.add_argument("--only-tile", default=None,
                   help="只测试一个tile，例如3,2；不组装完整地图")
    p.add_argument("--force", action="store_true",
                   help="忽略已完成tile，全部重新运行")
    p.add_argument("--continue-on-error", action="store_true")
    p.add_argument("--remove-tile-files-after-assembly", action="store_true")
    p.add_argument("--worker-tile", default=None, help=argparse.SUPPRESS)
    return p.parse_args()


def _resolve_file(explicit: str | None, default: Path, description: str) -> Path:
    path = Path(explicit).expanduser() if explicit else default
    path = path.resolve()
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"找不到{description}: {path}")
    return path


def _resolve_paths(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.project_root).expanduser().resolve()
    summary = _resolve_file(
        args.summary_csv,
        root / "outputs" / "parameter_calibration" / "all_27stations_summary.csv",
        "27站最佳参数汇总",
    )
    directions = _resolve_file(
        args.directions_csv,
        root / "outputs" / "parameter_calibration" / "estimated_initial_directions_27stations.csv",
        "27站初始方向文件",
    )
    ground = _resolve_file(args.ground, root / "assets" / "ground.ply", "ground.ply")
    if args.buildings:
        buildings = [Path(v).expanduser().resolve() for v in args.buildings]
    else:
        buildings = [root / "assets" / "ynu_chenggong_campus-001.ply"]
    buildings = [p for p in buildings if p.exists() and p.is_file() and p.stat().st_size > 0]
    if not buildings:
        raise FileNotFoundError("没有找到有效建筑PLY；默认需要assets/ynu_chenggong_campus-001.ply")
    output_root = (
        Path(args.output_root).expanduser().resolve()
        if args.output_root
        else root / "outputs" / "joint_best_server_4000x3000"
    )
    return {
        "root": root,
        "summary": summary,
        "directions": directions,
        "ground": ground,
        "buildings": buildings,
        "output_root": output_root,
    }


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _parse_pcis(value: Any) -> tuple[int, ...]:
    if pd.isna(value):
        return tuple()
    text = str(value).replace(",", ";")
    values = [int(float(v.strip())) for v in text.split(";") if v.strip()]
    return tuple(values)


def _read_summary(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    frame.columns = [str(c).strip().lstrip("\ufeff") for c in frame.columns]
    required = {
        "station_id", "x_m", "y_m", "height_agl_m", "azimuth_offset_deg",
        "absolute_downtilt_deg", "shared_power_dbm", "pcis",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"all_27stations_summary.csv缺少字段: {missing}")
    for c in ["station_id", "x_m", "y_m", "height_agl_m", "azimuth_offset_deg",
              "absolute_downtilt_deg", "shared_power_dbm"]:
        frame[c] = pd.to_numeric(frame[c], errors="coerce")
    frame = frame.loc[np.isfinite(frame["station_id"])].copy()
    frame["station_id"] = frame["station_id"].astype(int)
    if frame["station_id"].duplicated().any():
        duplicates = frame.loc[frame["station_id"].duplicated(False), "station_id"].tolist()
        raise ValueError(f"最佳参数汇总存在重复站号: {duplicates}")
    frame = frame.sort_values("station_id").reset_index(drop=True)
    if len(frame) != EXPECTED_STATION_COUNT:
        raise ValueError(
            f"联合地图要求27站全部调参完成，但summary中只有{len(frame)}站。"
            "请先运行 python run_pipeline.py calibrate --stations all，并处理失败站。"
        )
    invalid = frame[["x_m", "y_m", "height_agl_m", "azimuth_offset_deg",
                     "absolute_downtilt_deg", "shared_power_dbm"]].isna().any(axis=1)
    if invalid.any():
        raise ValueError(f"summary存在无效最佳参数站号: {frame.loc[invalid, 'station_id'].tolist()}")
    return frame


def _read_directions(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    frame.columns = [str(c).strip().lstrip("\ufeff") for c in frame.columns]
    required = {"station_id", "pcis", "alpha_1_rad", "antenna_type"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"estimated_initial_directions_27stations.csv缺少字段: {missing}")
    frame["station_id"] = pd.to_numeric(frame["station_id"], errors="coerce")
    frame = frame.loc[np.isfinite(frame["station_id"])].copy()
    frame["station_id"] = frame["station_id"].astype(int)
    frame = frame.drop_duplicates("station_id", keep="first").sort_values("station_id")
    return frame.reset_index(drop=True)


def _build_runtime_stations(
    summary: pd.DataFrame,
    directions: pd.DataFrame,
) -> tuple[dict[int, RuntimeStationConfig], pd.DataFrame]:
    d_by = {int(r.station_id): r for r in directions.itertuples(index=False)}
    stations: dict[int, RuntimeStationConfig] = {}
    used_rows: list[dict[str, Any]] = []
    global_sector_index = 0

    for row in summary.itertuples(index=False):
        sid = int(row.station_id)
        if sid not in d_by:
            raise ValueError(f"方向文件缺少{sid}号站")
        drow = d_by[sid]
        summary_pcis = _parse_pcis(getattr(row, "pcis"))
        direction_pcis = _parse_pcis(getattr(drow, "pcis"))
        pcis = summary_pcis or direction_pcis
        if direction_pcis and pcis != direction_pcis:
            raise ValueError(f"{sid}号站summary与方向文件PCI不一致: {pcis} vs {direction_pcis}")

        antenna_type = str(getattr(row, "antenna_type", getattr(drow, "antenna_type", ""))).lower()
        omni = sid == 22 or "omni" in antenna_type
        if omni:
            if len(pcis) != 1:
                raise ValueError(f"{sid}号全向站必须只有1个PCI，当前={pcis}")
            alphas = (float(getattr(drow, "alpha_1_rad")),)
        else:
            if len(pcis) != 3:
                raise ValueError(f"{sid}号三扇区站必须有3个PCI，当前={pcis}")
            needed = ["alpha_1_rad", "alpha_2_rad", "alpha_3_rad"]
            if any(not hasattr(drow, name) for name in needed):
                raise KeyError(f"方向文件缺少{sid}号站的alpha_1/2/3_rad")
            alphas = tuple(float(getattr(drow, name)) for name in needed)

        station = RuntimeStationConfig(
            station_id=sid,
            label=str(getattr(row, "label", f"station-{sid}")),
            x_m=float(row.x_m),
            y_m=float(row.y_m),
            pcis=pcis,
            initial_alphas_rad=alphas,
            original_downtilt_deg=0.0,
            initial_power_dbm=float(row.shared_power_dbm),
            is_omnidirectional=omni,
        )
        station.validate()
        stations[sid] = station

        final_alphas = apply_common_azimuth_offset(alphas, float(row.azimuth_offset_deg))
        for pci, initial_alpha, final_alpha in zip(pcis, alphas, final_alphas):
            used_rows.append(
                {
                    "global_sector_index": global_sector_index,
                    "station_id": sid,
                    "label": station.label,
                    "antenna_type": "omnidirectional" if omni else "three_sector",
                    "pci": int(pci),
                    "x_m": float(row.x_m),
                    "y_m": float(row.y_m),
                    "height_agl_m": float(row.height_agl_m),
                    "tx_absolute_z_m_from_summary": float(getattr(row, "tx_absolute_z_m", np.nan)),
                    "shared_power_dbm": float(row.shared_power_dbm),
                    "initial_alpha_rad": float(initial_alpha),
                    "initial_alpha_deg": float(math.degrees(initial_alpha)),
                    "azimuth_offset_deg": float(row.azimuth_offset_deg),
                    "final_alpha_rad": float(final_alpha),
                    "final_alpha_deg": float(math.degrees(final_alpha)),
                    "absolute_downtilt_deg": float(row.absolute_downtilt_deg),
                    "beta_rad": float(downtilt_to_sionna_beta_rad(float(row.absolute_downtilt_deg))),
                    "calibration_rmse_db": float(getattr(row, "final_dense_map_rmse_db", np.nan)),
                }
            )
            global_sector_index += 1

    used = pd.DataFrame(used_rows)
    if len(stations) != EXPECTED_STATION_COUNT:
        raise ValueError(f"运行时基站数应为27，实际={len(stations)}")
    if len(used) != EXPECTED_SECTOR_COUNT:
        raise ValueError(f"运行时PCI扇区数应为79，实际={len(used)}")
    return stations, used


def _bool_series_value(frame: pd.DataFrame, column: str, default: bool) -> bool:
    if column not in frame.columns:
        return default
    values = frame[column].dropna().astype(str).str.strip().str.lower()
    if values.empty:
        return default
    parsed = values.map(lambda v: v in {"1", "true", "yes", "y"})
    return bool(parsed.max())


def _positive_int_from_summary(frame: pd.DataFrame, column: str, default: int) -> int:
    if column not in frame.columns:
        return int(default)
    v = pd.to_numeric(frame[column], errors="coerce")
    v = v[np.isfinite(v) & (v > 0)]
    if v.empty:
        return int(default)
    unique = sorted(set(int(round(x)) for x in v.tolist()))
    if len(unique) > 1:
        print(f"[提示] summary中的{column}不完全一致{unique}，联合图使用最大值{max(unique)}")
    return max(unique)


def _resolve_simulation_settings(args: argparse.Namespace, summary: pd.DataFrame) -> dict[str, Any]:
    batch_count = args.batch_count or _positive_int_from_summary(summary, "final_batch_count", 5)
    samples = args.samples_per_tx or _positive_int_from_summary(summary, "final_samples_per_batch", 10_000_000)
    max_depth = args.max_depth or _positive_int_from_summary(summary, "final_max_depth", 5)
    edge = _bool_series_value(summary, "final_edge_diffraction", True)
    if args.quick:
        batch_count = 1
        samples = min(int(samples), 100_000)
        max_depth = min(int(max_depth), 3)
    if batch_count < 1 or samples < 1 or max_depth < 1:
        raise ValueError("batch-count、samples-per-tx和max-depth必须为正数")
    return {
        "batch_count": int(batch_count),
        "samples_per_tx": int(samples),
        "max_depth": int(max_depth),
        "edge_diffraction": bool(edge),
        "seed": int(args.seed),
        "seed_step": int(args.seed_step),
    }


def _validate_map_geometry(args: argparse.Namespace, terrain: TerrainModel) -> tuple[int, int, int, int]:
    if args.cell_size <= 0 or args.tile_size <= 0:
        raise ValueError("cell-size和tile-size必须>0")
    nx = int(round(args.size_x / args.cell_size))
    ny = int(round(args.size_y / args.cell_size))
    tile_n = int(round(args.tile_size / args.cell_size))
    if abs(nx * args.cell_size - args.size_x) > 1e-6 or abs(ny * args.cell_size - args.size_y) > 1e-6:
        raise ValueError("size-x和size-y必须能被cell-size整除")
    if abs(tile_n * args.cell_size - args.tile_size) > 1e-6:
        raise ValueError("tile-size必须能被cell-size整除")
    if args.size_x % args.tile_size != 0 or args.size_y % args.tile_size != 0:
        raise ValueError("4000×3000地图要求size-x和size-y均能被tile-size整除")
    ntx = args.size_x // args.tile_size
    nty = args.size_y // args.tile_size

    x0 = args.center_x - args.size_x / 2.0
    x1 = args.center_x + args.size_x / 2.0
    y0 = args.center_y - args.size_y / 2.0
    y1 = args.center_y + args.size_y / 2.0
    bx0, by0 = terrain.bounds[0, :2]
    bx1, by1 = terrain.bounds[1, :2]
    if x0 < bx0 or x1 > bx1 or y0 < by0 or y1 > by1:
        raise ValueError(
            "联合地图超出ground.ply范围，禁止使用边界外最近邻地形替代。\n"
            f"map: X={x0:.3f}..{x1:.3f}, Y={y0:.3f}..{y1:.3f}\n"
            f"ground: X={bx0:.3f}..{bx1:.3f}, Y={by0:.3f}..{by1:.3f}"
        )
    return nx, ny, ntx, nty


def _tile_specs(args: argparse.Namespace, nx: int, ny: int, ntx: int, nty: int) -> list[TileSpec]:
    tile_n = int(round(args.tile_size / args.cell_size))
    x_min = args.center_x - args.size_x / 2.0
    y_min = args.center_y - args.size_y / 2.0
    specs: list[TileSpec] = []
    for iy in range(nty):
        for ix in range(ntx):
            cx = x_min + (ix + 0.5) * args.tile_size
            cy = y_min + (iy + 0.5) * args.tile_size
            specs.append(
                TileSpec(
                    ix=ix,
                    iy=iy,
                    center_x=float(cx),
                    center_y=float(cy),
                    size_x_m=int(args.tile_size),
                    size_y_m=int(args.tile_size),
                    global_ix0=ix * tile_n,
                    global_ix1=(ix + 1) * tile_n,
                    global_iy0=iy * tile_n,
                    global_iy1=(iy + 1) * tile_n,
                )
            )
    return specs


def _run_signature(
    args: argparse.Namespace,
    paths: dict[str, Any],
    settings: dict[str, Any],
) -> str:
    payload = {
        "summary_sha256": _sha256(paths["summary"]),
        "directions_sha256": _sha256(paths["directions"]),
        "ground_sha256": _sha256(paths["ground"]),
        "building_sha256": [_sha256(p) for p in paths["buildings"]],
        "center_x": args.center_x,
        "center_y": args.center_y,
        "size_x": args.size_x,
        "size_y": args.size_y,
        "cell_size": args.cell_size,
        "tile_size": args.tile_size,
        "rx_height_agl_m": args.rx_height_agl_m,
        "building_mask_buffer_cells": args.building_mask_buffer_cells,
        "station_batch_size": args.station_batch_size,
        "simulation": settings,
        "algorithm": "streaming_joint_best_server_v1",
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _tile_output_path(output_root: Path, spec: TileSpec) -> Path:
    return output_root / "tiles" / f"{spec.key}.npz"


def _tile_is_reusable(path: Path, signature: str) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            return str(data["run_signature"][0]) == signature
    except Exception:
        return False


def _chunks(values: Sequence[int], size: int) -> Iterable[list[int]]:
    if size < 1:
        raise ValueError("station-batch-size必须>=1")
    for i in range(0, len(values), size):
        yield list(values[i:i + size])


def _lazy_sionna():
    try:
        from sionna.rt import PlanarArray, RadioMapSolver, Transmitter, load_mesh
    except Exception as exc:
        raise RuntimeError(
            "无法导入Sionna RT。请在sionna_env中运行，并确认Sionna RT及CUDA环境正常。"
        ) from exc
    return PlanarArray, RadioMapSolver, Transmitter, load_mesh


def _set_tx_array(scene: Any, omnidirectional: bool) -> None:
    PlanarArray, _, _, _ = _lazy_sionna()
    if omnidirectional:
        scene.tx_array = PlanarArray(
            num_rows=1, num_cols=1, vertical_spacing=0.5, horizontal_spacing=0.5,
            pattern="iso", polarization="V",
        )
    else:
        scene.tx_array = PlanarArray(
            num_rows=8, num_cols=4, vertical_spacing=0.5, horizontal_spacing=0.5,
            pattern="tr38901", polarization="VH",
        )


def _tx_descriptors_for_batch(
    batch_ids: Sequence[int],
    stations: dict[int, RuntimeStationConfig],
    summary_by_station: dict[int, pd.Series],
    terrain: TerrainModel,
    sector_catalog: pd.DataFrame,
) -> list[TxDescriptor]:
    global_index_by = {
        (int(r.station_id), int(r.pci)): int(r.global_sector_index)
        for r in sector_catalog.itertuples(index=False)
    }
    result: list[TxDescriptor] = []
    for sid in batch_ids:
        station = stations[int(sid)]
        row = summary_by_station[int(sid)]
        ground_z = float(terrain.query(station.x_m, station.y_m))
        tx_z = ground_z + float(row["height_agl_m"])
        alphas = apply_common_azimuth_offset(
            station.initial_alphas_rad, float(row["azimuth_offset_deg"])
        )
        beta = downtilt_to_sionna_beta_rad(float(row["absolute_downtilt_deg"]))
        for pci, alpha in zip(station.pcis, alphas):
            result.append(
                TxDescriptor(
                    global_sector_index=global_index_by[(int(sid), int(pci))],
                    station_id=int(sid),
                    pci=int(pci),
                    tx_name=f"st{int(sid)}_pci{int(pci)}",
                    tx_x_m=float(station.x_m),
                    tx_y_m=float(station.y_m),
                    tx_z_m=float(tx_z),
                    alpha_rad=float(alpha),
                    beta_rad=float(beta),
                    power_dbm=float(row["shared_power_dbm"]),
                    is_omnidirectional=bool(station.is_omnidirectional),
                )
            )
    return result


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _simulate_tx_batch_multiseed(
    scene: Any,
    surface: Any,
    descriptors: list[TxDescriptor],
    cfg: dict[str, Any],
    settings: dict[str, Any],
) -> np.ndarray:
    if not descriptors:
        return np.empty((0, surface.n_cells), dtype=np.float32)
    omni_values = {d.is_omnidirectional for d in descriptors}
    if len(omni_values) != 1:
        raise ValueError("同一TX批次不能混合全向和三扇区阵列")
    omnidirectional = next(iter(omni_values))
    _set_tx_array(scene, omnidirectional=omnidirectional)
    _, RadioMapSolver, Transmitter, load_mesh = _lazy_sionna()
    radio = cfg["radio"]
    n_tx = len(descriptors)
    accumulated_w = np.zeros((n_tx, surface.n_cells), dtype=np.float64)
    any_hit = np.zeros((n_tx, surface.n_cells), dtype=bool)

    for batch_index in range(settings["batch_count"]):
        clear_transmitters(scene)
        for d in descriptors:
            scene.add(
                Transmitter(
                    name=d.tx_name,
                    position=[d.tx_x_m, d.tx_y_m, d.tx_z_m],
                    orientation=[d.alpha_rad, d.beta_rad, 0.0],
                    power_dbm=d.power_dbm,
                )
            )
        measurement_surface = load_mesh(str(surface.path), flip_normals=False)
        seed = int(settings["seed"] + batch_index * settings["seed_step"])
        print(
            f"      seed {batch_index + 1}/{settings['batch_count']}: "
            f"TX={n_tx}, samples/tx={settings['samples_per_tx']:,}, seed={seed}"
        )
        solver = RadioMapSolver()
        rm = solver(
            scene,
            measurement_surface=measurement_surface,
            samples_per_tx=int(settings["samples_per_tx"]),
            max_depth=int(settings["max_depth"]),
            los=bool(radio["los"]),
            specular_reflection=bool(radio["specular_reflection"]),
            diffuse_reflection=bool(radio["diffuse_reflection"]),
            refraction=bool(radio["refraction"]),
            diffraction=bool(radio["diffraction"]),
            edge_diffraction=bool(settings["edge_diffraction"]),
            seed=seed,
        )
        rss_w = _to_numpy(rm.rss).astype(np.float64, copy=False)
        expected = n_tx * surface.n_faces
        if rss_w.size != expected:
            raise RuntimeError(
                f"MeshRadioMap输出尺寸异常: shape={rss_w.shape}, size={rss_w.size}, expected={expected}"
            )
        rss_cell_w = rss_w.reshape(n_tx, surface.n_faces).reshape(n_tx, surface.n_cells, 2).mean(axis=2)
        finite_positive = np.isfinite(rss_cell_w) & (rss_cell_w > 0.0)
        accumulated_w[finite_positive] += rss_cell_w[finite_positive]
        any_hit |= finite_positive
        del rm, rss_w, rss_cell_w, measurement_surface
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    average_w = accumulated_w / float(settings["batch_count"])
    rss_dbm = np.full(average_w.shape, np.nan, dtype=np.float64)
    positive = any_hit & np.isfinite(average_w) & (average_w > 0.0)
    rss_dbm[positive] = 10.0 * np.log10(average_w[positive] * 1000.0)
    re_count = int(radio["n_rb"]) * int(radio["subcarriers_per_rb"])
    rsrp = rss_dbm - 10.0 * np.log10(float(re_count)) + float(radio.get("rsrp_calibration_offset_db", 0.0))
    return rsrp.astype(np.float32)


def _update_best(
    best_rsrp: np.ndarray,
    best_station: np.ndarray,
    best_pci: np.ndarray,
    best_sector: np.ndarray,
    sector_rsrp: np.ndarray,
    descriptors: list[TxDescriptor],
) -> None:
    if sector_rsrp.size == 0:
        return
    safe = np.where(np.isfinite(sector_rsrp), sector_rsrp, -np.inf)
    local_index = np.argmax(safe, axis=0)
    local_best = safe[local_index, np.arange(safe.shape[1])]
    finite = np.isfinite(local_best) & (local_best > -np.inf)
    replace = finite & (~np.isfinite(best_rsrp) | (local_best > best_rsrp))
    if not np.any(replace):
        return
    descriptor_array = np.asarray(descriptors, dtype=object)
    chosen = descriptor_array[local_index[replace]]
    best_rsrp[replace] = local_best[replace].astype(np.float32)
    best_station[replace] = np.asarray([d.station_id for d in chosen], dtype=np.int16)
    best_pci[replace] = np.asarray([d.pci for d in chosen], dtype=np.int16)
    best_sector[replace] = np.asarray([d.global_sector_index for d in chosen], dtype=np.int16)


def _worker(args: argparse.Namespace, paths: dict[str, Any]) -> int:
    summary = _read_summary(paths["summary"])
    directions = _read_directions(paths["directions"])
    stations, sector_catalog = _build_runtime_stations(summary, directions)
    settings = _resolve_simulation_settings(args, summary)
    terrain = TerrainModel.load(paths["ground"])
    nx, ny, ntx, nty = _validate_map_geometry(args, terrain)
    specs = _tile_specs(args, nx, ny, ntx, nty)
    try:
        ix_text, iy_text = str(args.worker_tile).split(",")
        wanted_ix, wanted_iy = int(ix_text), int(iy_text)
    except Exception as exc:
        raise ValueError("--worker-tile格式必须为ix,iy，例如3,2") from exc
    selected = [s for s in specs if s.ix == wanted_ix and s.iy == wanted_iy]
    if not selected:
        raise ValueError(f"tile不存在: {wanted_ix},{wanted_iy}")
    spec = selected[0]
    output_root = paths["output_root"]
    output_root.mkdir(parents=True, exist_ok=True)
    tile_path = _tile_output_path(output_root, spec)
    tile_path.parent.mkdir(parents=True, exist_ok=True)
    signature = _run_signature(args, paths, settings)

    print("=" * 96)
    print(f"联合地图worker: {spec.key}, center=({spec.center_x:.3f},{spec.center_y:.3f})")
    print(f"最佳参数来源: {paths['summary']}")
    print("本worker只输出联合tile，不输出任何单站地图。")

    work = output_root / "work"
    scene_report = build_scene_xml_multi(
        ground_ply=paths["ground"],
        building_candidates=paths["buildings"],
        output_xml=work / "generated_joint_scene.xml",
        cleaned_dir=work / "scene_mesh_cache",
        allow_no_buildings=False,
    )
    triangles, triangle_diag = load_building_projection_triangles(paths["buildings"])
    grid, mask_diag = create_dense_grid_with_building_mask(
        terrain=terrain,
        building_triangles_xy=triangles,
        center_x=spec.center_x,
        center_y=spec.center_y,
        size_x_m=spec.size_x_m,
        size_y_m=spec.size_y_m,
        cell_size_m=float(args.cell_size),
        rx_height_agl_m=float(args.rx_height_agl_m),
        buffer_cells=int(args.building_mask_buffer_cells),
    )
    surface = build_dense_outdoor_measurement_surface(
        terrain=terrain,
        grid=grid,
        rx_height_agl_m=float(args.rx_height_agl_m),
        output_path=work / "tile_surfaces" / f"{spec.key}_dem_plus_1p5m_outdoor.ply",
    )

    representative = next(s for s in stations.values() if not s.is_omnidirectional)
    cfg = base.make_station_cfg(
        scene_xml=Path(scene_report["generated_xml"]),
        station=representative,
        rx_height_agl_m=float(args.rx_height_agl_m),
        max_depth=int(settings["max_depth"]),
        edge_diffraction=bool(settings["edge_diffraction"]),
    )
    cfg["radio"]["seed"] = int(settings["seed"])
    scene = configure_scene(Path(scene_report["generated_xml"]), cfg)
    summary_by = {int(r.station_id): summary.loc[summary["station_id"].eq(int(r.station_id))].iloc[0]
                  for r in summary.itertuples(index=False)}

    n_surface = surface.n_cells
    best_rsrp_surface = np.full(n_surface, np.nan, dtype=np.float32)
    best_station_surface = np.full(n_surface, -1, dtype=np.int16)
    best_pci_surface = np.full(n_surface, -1, dtype=np.int16)
    best_sector_surface = np.full(n_surface, -1, dtype=np.int16)

    directional_ids = sorted(sid for sid, st in stations.items() if not st.is_omnidirectional)
    omni_ids = sorted(sid for sid, st in stations.items() if st.is_omnidirectional)
    groups = [(False, batch) for batch in _chunks(directional_ids, int(args.station_batch_size))]
    groups += [(True, batch) for batch in _chunks(omni_ids, int(args.station_batch_size))]

    for group_index, (omni, batch_ids) in enumerate(groups, start=1):
        print(
            f"    TX批次 {group_index}/{len(groups)}: "
            f"站={batch_ids}, 类型={'全向' if omni else '三扇区'}"
        )
        descriptors = _tx_descriptors_for_batch(
            batch_ids=batch_ids,
            stations=stations,
            summary_by_station=summary_by,
            terrain=terrain,
            sector_catalog=sector_catalog,
        )
        sector_rsrp = _simulate_tx_batch_multiseed(
            scene=scene,
            surface=surface,
            descriptors=descriptors,
            cfg=cfg,
            settings=settings,
        )
        _update_best(
            best_rsrp_surface,
            best_station_surface,
            best_pci_surface,
            best_sector_surface,
            sector_rsrp,
            descriptors,
        )
        del sector_rsrp
        gc.collect()

    shape = (grid.ny, grid.nx)
    best_rsrp = np.full(shape, np.nan, dtype=np.float32)
    best_station = np.full(shape, -1, dtype=np.int16)
    best_pci = np.full(shape, -1, dtype=np.int16)
    best_sector = np.full(shape, -1, dtype=np.int16)
    best_rsrp[surface.cell_iy, surface.cell_ix] = best_rsrp_surface
    best_station[surface.cell_iy, surface.cell_ix] = best_station_surface
    best_pci[surface.cell_iy, surface.cell_ix] = best_pci_surface
    best_sector[surface.cell_iy, surface.cell_ix] = best_sector_surface
    best_rsrp[grid.building_mask] = np.nan
    best_station[grid.building_mask] = -1
    best_pci[grid.building_mask] = -1
    best_sector[grid.building_mask] = -1

    tile_meta = {
        "tile_key": spec.key,
        "tile_ix": spec.ix,
        "tile_iy": spec.iy,
        "center_x": spec.center_x,
        "center_y": spec.center_y,
        "size_x_m": spec.size_x_m,
        "size_y_m": spec.size_y_m,
        "outdoor_cell_count": int((~grid.building_mask).sum()),
        "building_cell_count": int(grid.building_mask.sum()),
        "finite_best_server_cell_count": int(np.isfinite(best_rsrp).sum()),
        "simulation": settings,
        "station_count": len(stations),
        "sector_count": len(sector_catalog),
        "run_signature": signature,
        "scene_report": scene_report,
        "building_triangle_diagnostics": triangle_diag,
        "building_mask_diagnostics": mask_diag,
    }
    np.savez_compressed(
        tile_path,
        run_signature=np.asarray([signature]),
        best_rsrp_dbm=best_rsrp,
        best_station_id=best_station,
        best_pci=best_pci,
        best_sector_index=best_sector,
        ground_z_m=np.asarray(grid.ground_z_m, dtype=np.float32),
        receiver_z_m=np.asarray(grid.receiver_z_m, dtype=np.float32),
        building_mask=np.asarray(grid.building_mask, dtype=bool),
        x_centers_m=np.asarray(grid.x_centers_m, dtype=np.float64),
        y_centers_m=np.asarray(grid.y_centers_m, dtype=np.float64),
        metadata_json=np.asarray([json.dumps(tile_meta, ensure_ascii=False)]),
    )
    tile_path.with_suffix(".json").write_text(
        json.dumps(tile_meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"完成tile: {tile_path}")
    return 0


def _worker_command(
    args: argparse.Namespace,
    paths: dict[str, Any],
    spec: TileSpec,
) -> list[str]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--project-root", str(paths["root"]),
        "--summary-csv", str(paths["summary"]),
        "--directions-csv", str(paths["directions"]),
        "--ground", str(paths["ground"]),
        "--buildings", *[str(p) for p in paths["buildings"]],
        "--output-root", str(paths["output_root"]),
        "--center-x", str(args.center_x),
        "--center-y", str(args.center_y),
        "--size-x", str(args.size_x),
        "--size-y", str(args.size_y),
        "--cell-size", str(args.cell_size),
        "--tile-size", str(args.tile_size),
        "--rx-height-agl-m", str(args.rx_height_agl_m),
        "--building-mask-buffer-cells", str(args.building_mask_buffer_cells),
        "--station-batch-size", str(args.station_batch_size),
        "--seed", str(args.seed),
        "--seed-step", str(args.seed_step),
        "--display-min-dbm", str(args.display_min_dbm),
        "--display-max-dbm", str(args.display_max_dbm),
        "--dpi", str(args.dpi),
        "--worker-tile", f"{spec.ix},{spec.iy}",
    ]
    if args.batch_count is not None:
        cmd += ["--batch-count", str(args.batch_count)]
    if args.samples_per_tx is not None:
        cmd += ["--samples-per-tx", str(args.samples_per_tx)]
    if args.max_depth is not None:
        cmd += ["--max-depth", str(args.max_depth)]
    if args.quick:
        cmd.append("--quick")
    return cmd


def _write_status(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def _assemble(
    args: argparse.Namespace,
    paths: dict[str, Any],
    specs: list[TileSpec],
    nx: int,
    ny: int,
    signature: str,
    sector_catalog: pd.DataFrame,
    settings: dict[str, Any],
) -> None:
    output_root = paths["output_root"]
    work = output_root / "work" / "assembly"
    work.mkdir(parents=True, exist_ok=True)
    shape = (ny, nx)
    arrays = {
        "best_rsrp_dbm": np.memmap(work / "best_rsrp_dbm.dat", mode="w+", dtype=np.float32, shape=shape),
        "best_station_id": np.memmap(work / "best_station_id.dat", mode="w+", dtype=np.int16, shape=shape),
        "best_pci": np.memmap(work / "best_pci.dat", mode="w+", dtype=np.int16, shape=shape),
        "best_sector_index": np.memmap(work / "best_sector_index.dat", mode="w+", dtype=np.int16, shape=shape),
        "ground_z_m": np.memmap(work / "ground_z_m.dat", mode="w+", dtype=np.float32, shape=shape),
        "receiver_z_m": np.memmap(work / "receiver_z_m.dat", mode="w+", dtype=np.float32, shape=shape),
        "building_mask": np.memmap(work / "building_mask.dat", mode="w+", dtype=np.bool_, shape=shape),
    }
    arrays["best_rsrp_dbm"][:] = np.nan
    arrays["best_station_id"][:] = -1
    arrays["best_pci"][:] = -1
    arrays["best_sector_index"][:] = -1
    arrays["ground_z_m"][:] = np.nan
    arrays["receiver_z_m"][:] = np.nan
    arrays["building_mask"][:] = False

    for spec in specs:
        tile_path = _tile_output_path(output_root, spec)
        if not _tile_is_reusable(tile_path, signature):
            raise RuntimeError(f"缺失或签名不匹配的tile，不能组装: {tile_path}")
        with np.load(tile_path, allow_pickle=False) as data:
            sl = np.s_[spec.global_iy0:spec.global_iy1, spec.global_ix0:spec.global_ix1]
            for name in arrays:
                arrays[name][sl] = data[name]

    for arr in arrays.values():
        arr.flush()

    x_min = args.center_x - args.size_x / 2.0
    y_min = args.center_y - args.size_y / 2.0
    x_centers = x_min + (np.arange(nx, dtype=np.float64) + 0.5) * args.cell_size
    y_centers = y_min + (np.arange(ny, dtype=np.float64) + 0.5) * args.cell_size
    outdoor = ~np.asarray(arrays["building_mask"])
    finite = outdoor & np.isfinite(np.asarray(arrays["best_rsrp_dbm"]))

    metadata = {
        "product": "27-station joint best-server radio map",
        "individual_station_maps_generated": False,
        "station_count": EXPECTED_STATION_COUNT,
        "sector_count": EXPECTED_SECTOR_COUNT,
        "map_center_x_m": args.center_x,
        "map_center_y_m": args.center_y,
        "map_size_x_m": args.size_x,
        "map_size_y_m": args.size_y,
        "map_bounds": [x_min, x_min + args.size_x, y_min, y_min + args.size_y],
        "cell_size_m": args.cell_size,
        "array_shape": [ny, nx],
        "tile_size_m": args.tile_size,
        "tile_count": len(specs),
        "receiver_surface": f"outdoor DEM+{args.rx_height_agl_m}m",
        "building_interior_removed_from_receiver_domain": True,
        "building_still_participates_in_propagation": True,
        "best_parameter_summary": str(paths["summary"]),
        "initial_direction_source": str(paths["directions"]),
        "simulation": settings,
        "run_signature": signature,
        "outdoor_cell_count": int(outdoor.sum()),
        "finite_best_server_cell_count": int(finite.sum()),
        "outdoor_hit_rate": float(finite.sum() / outdoor.sum()) if outdoor.any() else float("nan"),
        "display_min_dbm": args.display_min_dbm,
        "display_max_dbm": args.display_max_dbm,
    }

    final_npz = output_root / "joint_best_server_27stations_4000x3000.npz"
    print(f"正在压缩保存完整联合NPZ（约1200万网格）: {final_npz}")
    np.savez_compressed(
        final_npz,
        best_rsrp_dbm=np.asarray(arrays["best_rsrp_dbm"]),
        best_station_id=np.asarray(arrays["best_station_id"]),
        best_pci=np.asarray(arrays["best_pci"]),
        best_sector_index=np.asarray(arrays["best_sector_index"]),
        ground_z_m=np.asarray(arrays["ground_z_m"]),
        receiver_z_m=np.asarray(arrays["receiver_z_m"]),
        building_mask=np.asarray(arrays["building_mask"]),
        outdoor_valid_mask=outdoor,
        no_hit_outdoor_mask=outdoor & ~finite,
        x_centers_m=x_centers,
        y_centers_m=y_centers,
        sector_station_id=sector_catalog["station_id"].to_numpy(dtype=np.int16),
        sector_pci=sector_catalog["pci"].to_numpy(dtype=np.int16),
        metadata_json=np.asarray([json.dumps(metadata, ensure_ascii=False)]),
    )

    thresholds = [-80, -90, -100, -110, -120]
    coverage_rows: list[dict[str, Any]] = []
    rsrp = np.asarray(arrays["best_rsrp_dbm"])
    outdoor_count = int(outdoor.sum())
    for threshold in thresholds:
        covered = outdoor & np.isfinite(rsrp) & (rsrp >= float(threshold))
        count = int(covered.sum())
        coverage_rows.append({
            "threshold_dbm": threshold,
            "covered_cell_count": count,
            "covered_area_km2": count * args.cell_size * args.cell_size / 1_000_000.0,
            "outdoor_coverage_percent": 100.0 * count / outdoor_count if outdoor_count else np.nan,
        })
    pd.DataFrame(coverage_rows).to_csv(
        output_root / "coverage_statistics.csv", index=False, encoding="utf-8-sig"
    )

    station_rows = []
    station_map = np.asarray(arrays["best_station_id"])
    for sid in sorted(int(v) for v in np.unique(station_map) if int(v) >= 0):
        count = int((station_map == sid).sum())
        station_rows.append({
            "station_id": sid,
            "best_server_cell_count": count,
            "best_server_area_km2": count * args.cell_size * args.cell_size / 1_000_000.0,
            "percent_of_outdoor_cells": 100.0 * count / outdoor_count if outdoor_count else np.nan,
        })
    pd.DataFrame(station_rows).to_csv(
        output_root / "physical_station_best_server_area.csv", index=False, encoding="utf-8-sig"
    )

    pci_rows = []
    pci_map = np.asarray(arrays["best_pci"])
    for pci in sorted(int(v) for v in np.unique(pci_map) if int(v) >= 0):
        count = int((pci_map == pci).sum())
        pci_rows.append({
            "pci": pci,
            "best_server_cell_count": count,
            "best_server_area_km2": count * args.cell_size * args.cell_size / 1_000_000.0,
            "percent_of_outdoor_cells": 100.0 * count / outdoor_count if outdoor_count else np.nan,
        })
    pd.DataFrame(pci_rows).to_csv(
        output_root / "pci_best_server_area.csv", index=False, encoding="utf-8-sig"
    )

    (output_root / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _plot_outputs(args, output_root, arrays, sector_catalog, x_min, y_min)

    for arr in arrays.values():
        del arr
    gc.collect()
    if args.remove_tile_files_after_assembly:
        for spec in specs:
            p = _tile_output_path(output_root, spec)
            p.unlink(missing_ok=True)
            p.with_suffix(".json").unlink(missing_ok=True)


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


def _create_fixed_map_axes(fig, extent: list[float], left: float = 0.090, bottom: float = 0.180, top: float = 0.820, right_margin: float = 0.105, cbar_pad: float = 0.012, cbar_width: float = 0.022):
    x0, x1, y0, y1 = map(float, extent)
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


def _downsample_bool_mask_any(mask: np.ndarray, stride: int) -> np.ndarray:
    """Downsample a boolean mask while preserving any building cell in each block."""
    arr = np.asarray(mask, dtype=bool)
    stride = max(1, int(stride))
    if stride == 1:
        return arr.copy()
    ny, nx = arr.shape
    out_ny = (ny + stride - 1) // stride
    out_nx = (nx + stride - 1) // stride
    pad_y = out_ny * stride - ny
    pad_x = out_nx * stride - nx
    padded = np.pad(arr, ((0, pad_y), (0, pad_x)), mode="constant", constant_values=False)
    return padded.reshape(out_ny, stride, out_nx, stride).any(axis=(1, 3))


def _white_building_rgba(mask: np.ndarray) -> np.ndarray:
    """Create a compact uint8 RGBA overlay for solid white building footprints."""
    arr = np.asarray(mask, dtype=bool)
    rgba = np.zeros(arr.shape + (4,), dtype=np.uint8)
    rgba[arr] = np.asarray([255, 255, 255, 255], dtype=np.uint8)
    return rgba


def _overlay_white_buildings(ax, building_mask: np.ndarray, extent) -> None:
    """Overlay solid white building blocks on a joint-map axes."""
    mask = np.asarray(building_mask, dtype=bool)
    if mask.size == 0 or not np.any(mask):
        return
    ax.imshow(
        _white_building_rgba(mask),
        origin="lower",
        extent=extent,
        interpolation="nearest",
        aspect="equal",
        zorder=6,
    )


def _plot_outputs(
    args: argparse.Namespace,
    output_root: Path,
    arrays: dict[str, np.memmap],
    sector_catalog: pd.DataFrame,
    x_min: float,
    y_min: float,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ny, nx = arrays["best_rsrp_dbm"].shape
    stride = max(1, int(math.ceil(max(nx / 2400.0, ny / 1800.0))))
    extent = [x_min, x_min + args.size_x, y_min, y_min + args.size_y]
    rsrp = np.asarray(arrays["best_rsrp_dbm"])[::stride, ::stride]
    building = _downsample_bool_mask_any(np.asarray(arrays["building_mask"]), stride)
    display = np.where(np.isfinite(rsrp), rsrp, args.display_min_dbm)

    fig = plt.figure(figsize=_publication_figsize_inches(), dpi=MAP_DPI)
    ax, cax = _create_fixed_map_axes(fig, extent)
    im = ax.imshow(
        np.clip(display, args.display_min_dbm, args.display_max_dbm),
        origin="lower", extent=extent, interpolation="nearest", aspect="equal",
        cmap="viridis", vmin=args.display_min_dbm, vmax=args.display_max_dbm,
    )
    _overlay_white_buildings(ax, building, extent)
    station_points = sector_catalog.drop_duplicates("station_id")
    ax.scatter(station_points["x_m"], station_points["y_m"], marker="^", s=18,
               facecolors="none", edgecolors="red", linewidths=0.7)
    ax.set_title("27-station joint best-server RSRP | calibrated parameters | DEM+1.5 m")
    ax.set_xlabel("Blender X (m)")
    ax.set_ylabel("Blender Y (m)")
    _add_fixed_colorbar(fig, cax, im, "Best-server RSRP (dBm)")
    _save_png(fig, output_root / "joint_best_server_rsrp_4000x3000.png")
    plt.close(fig)

    station_map = np.asarray(arrays["best_station_id"])[::stride, ::stride].astype(float)
    station_map[station_map < 0] = np.nan
    fig = plt.figure(figsize=_publication_figsize_inches(), dpi=MAP_DPI)
    ax, cax = _create_fixed_map_axes(fig, extent)
    im = ax.imshow(station_map, origin="lower", extent=extent, interpolation="nearest",
                   aspect="equal", cmap="tab20")
    _overlay_white_buildings(ax, building, extent)
    ax.set_title("Joint best physical station ID")
    ax.set_xlabel("Blender X (m)")
    ax.set_ylabel("Blender Y (m)")
    _add_fixed_colorbar(fig, cax, im, "Station ID")
    _save_png(fig, output_root / "joint_best_station_id_4000x3000.png")
    plt.close(fig)

    pci_map = np.asarray(arrays["best_pci"])[::stride, ::stride].astype(float)
    pci_map[pci_map < 0] = np.nan
    fig = plt.figure(figsize=_publication_figsize_inches(), dpi=MAP_DPI)
    ax, cax = _create_fixed_map_axes(fig, extent)
    im = ax.imshow(pci_map, origin="lower", extent=extent, interpolation="nearest",
                   aspect="equal", cmap="turbo")
    _overlay_white_buildings(ax, building, extent)
    ax.set_title("Joint best PCI")
    ax.set_xlabel("Blender X (m)")
    ax.set_ylabel("Blender Y (m)")
    _add_fixed_colorbar(fig, cax, im, "PCI")
    _save_png(fig, output_root / "joint_best_pci_4000x3000.png")
    plt.close(fig)


def _orchestrator(args: argparse.Namespace, paths: dict[str, Any]) -> int:
    summary = _read_summary(paths["summary"])
    directions = _read_directions(paths["directions"])
    stations, sector_catalog = _build_runtime_stations(summary, directions)
    settings = _resolve_simulation_settings(args, summary)
    terrain = TerrainModel.load(paths["ground"])
    nx, ny, ntx, nty = _validate_map_geometry(args, terrain)
    specs = _tile_specs(args, nx, ny, ntx, nty)
    signature = _run_signature(args, paths, settings)
    output_root = paths["output_root"]
    output_root.mkdir(parents=True, exist_ok=True)

    sector_catalog.to_csv(
        output_root / "parameters_used_27stations_79sectors.csv",
        index=False, encoding="utf-8-sig",
    )
    summary.to_csv(
        output_root / "best_parameter_summary_used_27stations.csv",
        index=False, encoding="utf-8-sig",
    )

    x0 = args.center_x - args.size_x / 2.0
    x1 = args.center_x + args.size_x / 2.0
    y0 = args.center_y - args.size_y / 2.0
    y1 = args.center_y + args.size_y / 2.0
    validation = {
        "status": "ok",
        "station_count": len(stations),
        "sector_count": len(sector_catalog),
        "station_ids": sorted(stations),
        "summary_csv": str(paths["summary"]),
        "directions_csv": str(paths["directions"]),
        "ground": str(paths["ground"]),
        "buildings": [str(p) for p in paths["buildings"]],
        "map_center": [args.center_x, args.center_y],
        "map_bounds": [x0, x1, y0, y1],
        "map_shape": [ny, nx],
        "tile_grid": [nty, ntx],
        "tile_count": len(specs),
        "simulation": settings,
        "run_signature": signature,
        "individual_station_maps_generated": False,
    }
    (output_root / "joint_map_input_validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("=" * 96)
    print("27站联合best-server无线电地图")
    print(f"最佳参数汇总: {paths['summary']}")
    print(f"初始方向文件: {paths['directions']}")
    print(f"物理站={len(stations)}, PCI扇区={len(sector_catalog)}")
    print(f"地图: center=({args.center_x},{args.center_y}), X={x0}..{x1}, Y={y0}..{y1}")
    print(f"网格: {nx}×{ny}, tile={args.tile_size}m, 共{len(specs)}个tile")
    print(f"仿真设置: {settings}")
    print("不会重新生成或保存27张单站地图；单站结果只在tile内存中参与max运算。")

    if args.dry_run:
        print(f"[DRY-RUN通过] 检查报告: {output_root / 'joint_map_input_validation.json'}")
        return 0

    selected_specs = specs
    if args.only_tile:
        try:
            ix_text, iy_text = str(args.only_tile).split(",")
            ix, iy = int(ix_text), int(iy_text)
        except Exception as exc:
            raise ValueError("--only-tile格式必须为ix,iy，例如3,2") from exc
        selected_specs = [s for s in specs if s.ix == ix and s.iy == iy]
        if not selected_specs:
            raise ValueError(f"指定tile不存在: {args.only_tile}")

    status_rows: list[dict[str, Any]] = []
    failures = 0
    for index, spec in enumerate(selected_specs, start=1):
        tile_path = _tile_output_path(output_root, spec)
        started = time.time()
        if not args.force and _tile_is_reusable(tile_path, signature):
            print(f"[{index}/{len(selected_specs)}] 复用已完成 {spec.key}")
            status = "cached"
            return_code = 0
        else:
            print(f"[{index}/{len(selected_specs)}] 开始 {spec.key}")
            cmd = _worker_command(args, paths, spec)
            print("[RUN]", " ".join(f'\"{v}\"' if " " in str(v) else str(v) for v in cmd), flush=True)
            return_code = subprocess.run(cmd, cwd=paths["root"]).returncode
            status = "completed" if return_code == 0 else "failed"
        status_rows.append({
            "tile": spec.key,
            "tile_ix": spec.ix,
            "tile_iy": spec.iy,
            "status": status,
            "return_code": return_code,
            "elapsed_s": time.time() - started,
            "tile_output": str(tile_path),
        })
        _write_status(output_root / "tile_status.csv", status_rows)
        if return_code != 0:
            failures += 1
            if not args.continue_on_error:
                return return_code

    if args.only_tile:
        print("单tile测试完成，不执行完整地图组装。")
        return 0 if failures == 0 else 1

    missing = [s.key for s in specs if not _tile_is_reusable(_tile_output_path(output_root, s), signature)]
    if missing:
        print(f"仍有{len(missing)}个tile缺失或失败，暂不组装完整地图: {missing}")
        return 1

    _assemble(
        args=args,
        paths=paths,
        specs=specs,
        nx=nx,
        ny=ny,
        signature=signature,
        sector_catalog=sector_catalog,
        settings=settings,
    )
    print("完成27站联合best-server地图:", output_root)
    return 0


def main() -> int:
    args = parse_args()
    paths = _resolve_paths(args)
    if args.worker_tile:
        return _worker(args, paths)
    return _orchestrator(args, paths)


if __name__ == "__main__":
    raise SystemExit(main())
