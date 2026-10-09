#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import copy
import hashlib
import html
import json
import math
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.angles import validate_sector_spacing
from src.configuration import load_yaml, resolve_path
from src.measurement_io import prepare_station_measurements, read_measurements
import src.optimizer as optimizer_module
import src.simulator as simulator_module
from src.optimizer import evaluate_prediction, optimize_station
from src.plotting import plot_comparison, plot_final_maps, save_final_npz
from src.simulator import Candidate, configure_scene, run_candidate
from src.terrain import (
    SurfaceInfo,
    TerrainModel,
    build_dense_measurement_surface,
    build_sparse_measurement_surface,
    inspect_mesh,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Sionna RT全部27站校准：真实DEM+1.5m室外曲面、建筑掩膜、" "多seed功率平均、最终depth=5")
    )
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument("--measurements", default=None, help="坐标转换后的实测CSV")
    parser.add_argument(
        "--stations",
        default="all",
        help="默认运行全部27个基站；也可指定例如 3,7,18",
    )
    parser.add_argument("--quick", action="store_true", help="仅做流程试跑，降低采样数和搜索网格")
    parser.add_argument("--force", action="store_true", help="忽略缓存重新仿真")
    parser.add_argument(
        "--ground",
        default=None,
        help="覆盖config.yaml中的地形PLY路径，例如 assets\\ground(1).ply",
    )
    parser.add_argument(
        "--buildings",
        nargs="*",
        default=None,
        help=(
            "显式指定一个或多个建筑PLY。未指定时会读取config.yaml并自动扫描"
            "assets/ynu_chenggong_campus*.ply"
        ),
    )
    parser.add_argument(
        "--allow-no-buildings",
        action="store_true",
        help="所有建筑PLY均无效时仍允许只使用地形运行；默认直接停止，避免误跑",
    )
    parser.add_argument(
        "--direction-top-fraction",
        type=float,
        default=0.25,
        help="用每个PCI最强的前多少比例实测点估计初始方向角，默认0.25",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="任一基站失败时立即停止；默认记录失败后继续其余基站",
    )
    parser.add_argument(
        "--final-batches",
        type=int,
        default=5,
        help=(
            "最终1m DEM+1.5m无线电地图的独立蒙特卡洛批次数，默认5。"
            "每批使用不同seed，并在线性功率域平均。"
        ),
    )
    parser.add_argument(
        "--final-samples-per-batch",
        type=int,
        default=None,
        help=(
            "最终地图每批每TX射线数；默认沿用config中的"
            "radio.samples_per_tx_final（正式配置通常为10000000）。"
        ),
    )
    parser.add_argument(
        "--final-seed-step",
        type=int,
        default=1009,
        help="最终多批次seed递增间隔，默认1009。",
    )
    # Backward-compatible arguments are accepted but ignored. The reviewed method uses
    # one propagation configuration everywhere: max_depth=5, diffraction=True,
    # edge_diffraction=True. They are intentionally hidden from normal --help output.
    parser.add_argument("--search-max-depth", type=int, default=5, help=argparse.SUPPRESS)
    parser.add_argument("--final-max-depth", type=int, default=5, help=argparse.SUPPRESS)
    parser.add_argument("--no-final-edge-diffraction", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.25,
        help="按完整source_file/测量轨迹划分独立validation集合的目标比例，默认0.25。",
    )
    parser.add_argument(
        "--validation-seed",
        type=int,
        default=20261003,
        help="独立轨迹划分随机种子。",
    )
    parser.add_argument(
        "--min-calibration-rows-per-pci",
        type=int,
        default=6,
        help="划出validation轨迹后，每个PCI在calibration集合至少保留的原始长表记录数。",
    )
    parser.add_argument(
        "--disable-independent-validation-split",
        action="store_true",
        help="关闭按source_file独立划分；仅用于兼容旧数据，正式复现不建议使用。",
    )
    parser.add_argument(
        "--no-final-refinement",
        action="store_true",
        help="关闭depth=5+edge diffraction下的最终候选细化；正式复现不建议关闭。",
    )
    parser.add_argument("--refine-height-radius-m", type=float, default=1.0)
    parser.add_argument("--refine-height-step-m", type=float, default=1.0)
    parser.add_argument("--refine-azimuth-radius-deg", type=float, default=2.0)
    parser.add_argument("--refine-azimuth-step-deg", type=float, default=2.0)
    parser.add_argument("--refine-downtilt-radius-deg", type=float, default=1.0)
    parser.add_argument("--refine-downtilt-step-deg", type=float, default=1.0)
    parser.add_argument(
        "--building-mask-buffer-cells",
        type=int,
        default=1,
        help=(
            "建筑占地掩膜向外扩张的1m网格数，默认1。"
            "用于避免DEM+1.5m接收单元跨过建筑墙体。"
        ),
    )
    parser.add_argument(
        "--global-epre-calibration-file",
        default=None,
        help="全网统一EPRE校准JSON；默认使用 config/global_epre_calibration.json。",
    )
    parser.add_argument(
        "--global-epre-offset-db",
        type=float,
        default=None,
        help="显式覆盖全网统一EPRE offset；仅用于方法学复现/敏感性分析。",
    )
    parser.add_argument("--output-root", default=None, help="覆盖校准输出目录。")
    parser.add_argument("--objective-mode", choices=["pooled_rmse", "trajectory_balanced_pci_rmse", "reliability_balanced_pci_rmse"], default=None)
    parser.add_argument("--reliability-min-trajectory-points", type=int, default=None, help="Minimum matched cells for a trajectory to influence calibration selection.")
    parser.add_argument("--reliability-trajectory-weight-cap", type=int, default=None, help="Cap before square-root support weighting across trajectories.")
    parser.add_argument("--reliability-min-pci-points", type=int, default=None, help="Minimum matched cells for a PCI group in the reliability objective.")
    parser.add_argument("--reliability-pci-weight-cap", type=int, default=None, help="Cap before square-root support weighting across PCI groups.")
    parser.add_argument("--power-regularization-lambda", type=float, default=None)
    parser.add_argument("--power-regularization-anchor-dbm", type=float, default=None)
    parser.add_argument("--reference-geometry-file", default=None, help="Reference station geometry CSV used for constrained calibration.")
    parser.add_argument("--constrain-geometry-to-reference", action="store_true", help="Restrict height/azimuth/downtilt to local windows around the reference solution.")
    parser.add_argument("--reference-height-radius-m", type=float, default=3.0)
    parser.add_argument("--reference-azimuth-radius-deg", type=float, default=15.0)
    parser.add_argument("--reference-downtilt-radius-deg", type=float, default=3.0)
    parser.add_argument("--geometry-regularization-lambda-db2", type=float, default=1.0)
    parser.add_argument("--anchor-power-to-reference", action="store_true", help="Use each station's reference shared power as the power-regularization anchor.")
    return parser.parse_args()



def _resolve_cli_path(value: str | Path) -> Path:
    """Resolve a CLI path relative to this project directory."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _safe_xml_id(text: str, fallback: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", text).strip("-.")
    return value or fallback


def _mesh_candidates(
    cfg: Dict[str, Any],
    explicit_buildings: Sequence[str] | None,
    ground_ply: Path,
) -> list[Path]:
    """
    Return building candidates in deterministic order.

    Priority:
      1. --buildings
      2. scene.building_plys (list)
      3. scene.building_ply (single-path fallback)
      4. automatic assets/ynu_chenggong_campus*.ply discovery
    """
    raw: list[Path] = []
    if explicit_buildings:
        raw.extend(_resolve_cli_path(v) for v in explicit_buildings)
    else:
        scene_cfg = cfg.get("scene", {})
        configured_many = scene_cfg.get("building_plys", [])
        if isinstance(configured_many, (str, Path)):
            configured_many = [configured_many]
        for value in configured_many or []:
            raw.append(resolve_path(cfg, value))
        configured_one = scene_cfg.get("building_ply")
        if configured_one:
            raw.append(resolve_path(cfg, configured_one))

        assets_dir = ground_ply.parent
        raw.extend(sorted(assets_dir.glob("ynu_chenggong_campus*.ply")))

    result: list[Path] = []
    seen: set[str] = set()
    ground_key = str(ground_ply.resolve()).casefold()
    for path in raw:
        resolved = Path(path).expanduser().resolve()
        key = str(resolved).casefold()
        if key == ground_key or key in seen:
            continue
        seen.add(key)
        result.append(resolved)
    return result


def _face_max_edge_lengths(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tri = vertices[faces]
    e01 = np.linalg.norm(tri[:, 0] - tri[:, 1], axis=1)
    e12 = np.linalg.norm(tri[:, 1] - tri[:, 2], axis=1)
    e20 = np.linalg.norm(tri[:, 2] - tri[:, 0], axis=1)
    return np.maximum(np.maximum(e01, e12), e20)


def _prepare_building_mesh(
    source_path: Path,
    ground_bounds: np.ndarray,
    cleaned_dir: Path,
    **_: Any,
) -> tuple[Path | None, Dict[str, Any]]:
    """
    用户已明确要求：所有存在且非空的建筑PLY均视为正确建筑物并加入场景。

    本函数不做以下操作：
      - 不按ground.ply的XY范围排除；
      - 不按Z高程排除；
      - 不删除顶点或三角面；
      - 不重建、缩放或平移建筑；
      - 不检查三角面边长。

    仅把原始文件逐字节复制到ASCII文件名缓存，避免Windows/Mitsuba
    无法正确解析中文PLY文件名。原始几何内容保持不变。
    """
    import shutil

    source_path = Path(source_path).expanduser().resolve()
    report: Dict[str, Any] = {
        "source_path": str(source_path),
        "included": False,
        "filtered": False,
        "forced_include": True,
        "geometry_modified": False,
    }

    if not source_path.exists():
        report["exclusion_reason"] = "文件不存在"
        return None, report
    if not source_path.is_file():
        report["exclusion_reason"] = "路径不是文件"
        return None, report
    if source_path.stat().st_size <= 0:
        report["exclusion_reason"] = "文件大小为0"
        return None, report

    cleaned_dir.mkdir(parents=True, exist_ok=True)
    digest = _sha256_file(source_path)
    effective_path = (cleaned_dir / f"building_{digest[:20]}.ply").resolve()

    if (
        not effective_path.exists()
        or effective_path.stat().st_size != source_path.stat().st_size
        or _sha256_file(effective_path) != digest
    ):
        shutil.copyfile(source_path, effective_path)

    report.update(
        {
            "size_bytes": int(source_path.stat().st_size),
            "effective_path": str(effective_path),
            "source_sha256": digest,
            "effective_sha256": _sha256_file(effective_path),
            "included": True,
            "note": "用户指定为正确建筑物；原样强制加入，未执行任何坐标或三角面过滤。",
        }
    )
    return effective_path, report


def build_scene_xml_multi(
    ground_ply: Path,
    building_candidates: Sequence[Path],
    output_xml: Path,
    cleaned_dir: Path,
    ground_material: str = "itu_wet_ground",
    building_material: str = "itu_concrete",
    allow_no_buildings: bool = False,
) -> Dict[str, Any]:
    """Build one Sionna scene XML containing one ground and multiple building PLYs."""
    ground_ply = Path(ground_ply).expanduser().resolve()
    output_xml = Path(output_xml).expanduser().resolve()
    output_xml.parent.mkdir(parents=True, exist_ok=True)

    ground_info = inspect_mesh(ground_ply)
    if not ground_ply.exists():
        raise FileNotFoundError(f"找不到地形PLY: {ground_ply}")
    if ground_info.get("empty", True) or not ground_info.get("bounds"):
        raise ValueError(f"地形PLY为空或无效: {ground_info}")
    ground_bounds = np.asarray(ground_info["bounds"], dtype=float)
    ground_hash = _sha256_file(ground_ply)

    included: list[Path] = []
    building_reports: list[Dict[str, Any]] = []
    for candidate in building_candidates:
        effective, report = _prepare_building_mesh(
            candidate,
            ground_bounds,
            cleaned_dir,
        )
        building_reports.append(report)
        if effective is not None:
            included.append(effective)

    if not included and not allow_no_buildings:
        details = "\n".join(
            f"- {r.get('source_path')}: {r.get('exclusion_reason', '无有效面')}"
            for r in building_reports
        )
        raise RuntimeError(
            "没有找到可安全加入场景的建筑PLY。为避免误生成无建筑无线电地图，程序已停止。\n"
            + details
            + "\n可用 --allow-no-buildings 明确允许只加载地形。"
        )

    ground_mat_id = f"mat-{_safe_xml_id(ground_material, 'itu-wet-ground')}"
    building_mat_id = f"mat-{_safe_xml_id(building_material, 'itu-concrete')}"
    lines = [
        '<?xml version="1.0" encoding="utf-8"?>',
        '<scene version="3.0.0">',
        f'  <!-- ground_sha256={ground_hash} -->',
        f'  <bsdf type="diffuse" id="{html.escape(ground_mat_id)}" name="{html.escape(ground_mat_id)}">',
        '    <rgb name="reflectance" value="0.35,0.28,0.20"/>',
        '  </bsdf>',
        f'  <bsdf type="diffuse" id="{html.escape(building_mat_id)}" name="{html.escape(building_mat_id)}">',
        '    <rgb name="reflectance" value="0.65,0.65,0.65"/>',
        '  </bsdf>',
        '  <shape type="ply" id="mesh-ground" name="mesh-ground">',
        f'    <string name="filename" value="{html.escape(ground_ply.as_posix())}"/>',
        f'    <ref name="bsdf" id="{html.escape(ground_mat_id)}"/>',
        '  </shape>',
    ]
    for index, path in enumerate(included, start=1):
        mesh_hash = _sha256_file(path)
        shape_id = f"mesh-building-{index:03d}"
        lines.extend(
            [
                f'  <!-- building_{index:03d}_sha256={mesh_hash} source={html.escape(path.name)} -->',
                f'  <shape type="ply" id="{shape_id}" name="{shape_id}">',
                f'    <string name="filename" value="{html.escape(path.as_posix())}"/>',
                f'    <ref name="bsdf" id="{html.escape(building_mat_id)}"/>',
                '  </shape>',
            ]
        )
    lines.append('</scene>')
    output_xml.write_text("\n".join(lines) + "\n", encoding="utf-8")

    report: Dict[str, Any] = {
        "generated_xml": str(output_xml),
        "ground": ground_info,
        "ground_sha256": ground_hash,
        "building_candidates": [str(p) for p in building_candidates],
        "buildings": building_reports,
        "building_included_count": len(included),
        "building_included_paths": [str(p) for p in included],
        "building_included": bool(included),
        "warning": None if included else "本次场景按用户显式允许，仅加载地形。",
    }
    report_path = output_xml.with_suffix(".diagnostics.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report



@dataclass(frozen=True)
class RuntimeStationConfig:
    """支持三扇区站以及单PCI全向站的运行时基站配置。"""

    station_id: int
    label: str
    x_m: float
    y_m: float
    pcis: tuple[int, ...]
    initial_alphas_rad: tuple[float, ...]
    original_downtilt_deg: float
    initial_power_dbm: float
    is_omnidirectional: bool = False

    def validate(self, tolerance_deg: float = 1.0) -> None:
        if self.is_omnidirectional:
            if len(self.pcis) != 1 or len(self.initial_alphas_rad) != 1:
                raise ValueError(
                    f"{self.station_id}号全向站必须恰好有1个PCI和1个alpha: "
                    f"pcis={self.pcis}, alphas={self.initial_alphas_rad}"
                )
            return

        if len(self.pcis) != 3 or len(set(self.pcis)) != 3:
            raise ValueError(
                f"{self.station_id}号三扇区站必须有3个互不相同的PCI: {self.pcis}"
            )
        validate_sector_spacing(self.initial_alphas_rad, tolerance_deg=tolerance_deg)


LONG_REQUIRED_COLUMNS = {
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



def _repair_27station_measurement_schema(frame: pd.DataFrame) -> pd.DataFrame:
    """兼容原始长表及旧版1 m/2.77 m聚合表。

    旧版聚合脚本会丢失校准器需要的 ``rx_point_id``、站点真值坐标、
    DEM标志和几何方位。本函数仅利用当前行坐标与代码包内PCI映射表补齐
    这些确定性字段，不修改RSRP数值。
    """
    out = frame.copy()
    out.columns = [str(c).replace("\ufeff", "").strip() for c in out.columns]

    aliases = {
        "blender_x": ["x_m", "x", "receiver_x_m"],
        "blender_y": ["y_m", "y", "receiver_y_m"],
        "measured_rsrp_dbm": ["rsrp_dbm", "cell_rsrp_dbm", "NR5G SS RSRP"],
    }
    lower = {c.lower(): c for c in out.columns}
    for target, candidates in aliases.items():
        if target in out.columns:
            continue
        source = next((lower.get(str(v).lower()) for v in candidates if lower.get(str(v).lower()) is not None), None)
        if source is not None:
            out[target] = out[source]

    if "pci" in out.columns:
        out["pci"] = pd.to_numeric(out["pci"], errors="coerce")

    # 按PCI从代码包内正式映射表补齐基站和扇区元数据。
    mapping_path = ROOT.parents[1] / "config" / "base_station_pci_mapping.csv"
    if mapping_path.exists() and "pci" in out.columns:
        mapping = pd.read_csv(mapping_path, encoding="utf-8-sig", low_memory=False)
        mapping.columns = [str(c).replace("\ufeff", "").strip() for c in mapping.columns]
        mapping["pci"] = pd.to_numeric(mapping["pci"], errors="coerce")
        mapping = mapping.dropna(subset=["pci"]).copy()
        mapping["pci"] = mapping["pci"].astype(int)
        wanted = [
            "pci", "station_id", "station_name", "station_label", "sector_index",
            "antenna_type", "is_omnidirectional", "tx_x_initial_m",
            "tx_y_initial_m", "tx_z_initial_m",
        ]
        mapping = mapping[[c for c in wanted if c in mapping.columns]].drop_duplicates("pci")
        out = out.merge(mapping, on="pci", how="left", suffixes=("", "__map"))
        for column in wanted:
            if column == "pci":
                continue
            mapped = f"{column}__map"
            if mapped not in out.columns:
                continue
            if column not in out.columns:
                out[column] = out[mapped]
            else:
                current = out[column]
                missing = current.isna() | current.astype(str).str.strip().isin(["", "nan", "None"])
                out.loc[missing, column] = out.loc[missing, mapped]
            out.drop(columns=[mapped], inplace=True)

    # 基础数值字段。
    for column in [
        "blender_x", "blender_y", "ground_z_m", "receiver_z_m", "pci",
        "measured_rsrp_dbm", "station_id", "sector_index", "is_omnidirectional",
        "tx_x_initial_m", "tx_y_initial_m", "tx_z_initial_m",
    ]:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")

    if "rsrp_unit" not in out.columns:
        out["rsrp_unit"] = "dBm"
    else:
        blank = out["rsrp_unit"].isna() | out["rsrp_unit"].astype(str).str.strip().eq("")
        out.loc[blank, "rsrp_unit"] = "dBm"

    if "is_target_27station_pci" not in out.columns:
        out["is_target_27station_pci"] = 1
    if "rsrp_plausible_flag" not in out.columns and "measured_rsrp_dbm" in out.columns:
        out["rsrp_plausible_flag"] = out["measured_rsrp_dbm"].between(-200.0, 0.0).astype(int)
    if "dem_hit" not in out.columns:
        out["dem_hit"] = np.isfinite(pd.to_numeric(out.get("ground_z_m"), errors="coerce")).astype(int)

    # 旧聚合表没有RX ID，按已有网格编号或坐标构建稳定ID。
    if "rx_point_id" not in out.columns:
        if {"grid_ix", "grid_iy"}.issubset(out.columns):
            gx = pd.to_numeric(out["grid_ix"], errors="coerce").fillna(0).astype(np.int64)
            gy = pd.to_numeric(out["grid_iy"], errors="coerce").fillna(0).astype(np.int64)
            out["rx_point_id"] = "grid_x" + gx.astype(str) + "_y" + gy.astype(str)
        elif {"blender_x", "blender_y"}.issubset(out.columns):
            out["rx_point_id"] = (
                "xy_" + out["blender_x"].round(3).astype(str)
                + "_" + out["blender_y"].round(3).astype(str)
            )

    # 聚合坐标对应的方位必须重新计算。
    geom = {"blender_x", "blender_y", "tx_x_initial_m", "tx_y_initial_m"}
    if geom.issubset(out.columns):
        dx = out["blender_x"] - out["tx_x_initial_m"]
        dy = out["blender_y"] - out["tx_y_initial_m"]
        out["bearing_from_tx_math_deg"] = (np.degrees(np.arctan2(dy, dx)) + 360.0) % 360.0
        if "horizontal_distance_initial_m" not in out.columns:
            out["horizontal_distance_initial_m"] = np.hypot(dx, dy)

    return out

def read_27station_long_measurements(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    严格读取 cell_pci_rsrp_long_27stations.csv。

    返回：
      raw_long: 保留用于构建27站配置的正式长表；
      observations: 适配现有 prepare_station_measurements() 的标准字段。
    """
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(
            f"找不到27站处理后实测长表: {path}\n"
            "默认文件应为 data/processed/cell_pci_rsrp_long_27stations.csv"
        )

    frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    frame = _repair_27station_measurement_schema(frame)
    if frame.empty:
        # Header-only formal CSVs can be left behind when an earlier extraction
        # produced zero target observations.  When possible, recover from the
        # unaggregated extracted long table, which preserves source_file and is
        # therefore suitable for the independent trajectory-level split.
        project_root = ROOT.parents[1]
        candidates = [
            project_root / "data" / "processed" / "extracted" / "cell_pci_rsrp_long_27stations.csv",
            project_root / "data" / "cell_pci_rsrp_long_27stations.csv",
        ]
        recovered = None
        for candidate in candidates:
            candidate = candidate.resolve()
            if candidate == path or not candidate.exists() or candidate.stat().st_size <= 3:
                continue
            try:
                fallback = pd.read_csv(candidate, encoding="utf-8-sig", low_memory=False)
                fallback = _repair_27station_measurement_schema(fallback)
            except Exception:
                continue
            if not fallback.empty:
                recovered = (candidate, fallback)
                break
        if recovered is None:
            extraction_report = project_root / "data" / "processed" / "extracted" / "extraction_report.json"
            raise ValueError(
                f"27站处理后实测长表为空: {path}\n"
                "该CSV只有表头或没有目标PCI观测。请先执行 `python run_pipeline.py prepare-data --force`；"
                "新版 run_pipeline.py 在 calibrate 前也会自动尝试从 aligned/raw measurements 重建。\n"
                f"若仍为空，请检查: {extraction_report}"
            )
        path, frame = recovered
        print(f"[DATA] 正式长表为空，自动改用非空的未聚合27站长表: {path}")

    missing = sorted(LONG_REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise KeyError(
            "输入文件不是预期的27站坐标转换长表。"
            f"缺少字段: {missing}\n实际字段: {list(frame.columns)}"
        )

    numeric_columns = [
        "blender_x", "blender_y", "ground_z_m", "receiver_z_m",
        "pci", "measured_rsrp_dbm", "station_id", "sector_index",
        "is_omnidirectional", "is_target_27station_pci",
        "rsrp_plausible_flag", "dem_hit", "tx_x_initial_m",
        "tx_y_initial_m", "bearing_from_tx_math_deg",
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    units = frame["rsrp_unit"].astype(str).str.strip().str.lower()
    if not units.eq("dbm").all():
        bad = frame.loc[~units.eq("dbm"), "rsrp_unit"].value_counts(dropna=False).to_dict()
        raise ValueError(f"RSRP单位不是统一的dBm: {bad}")

    valid = (
        np.isfinite(frame["blender_x"])
        & np.isfinite(frame["blender_y"])
        & np.isfinite(frame["pci"])
        & np.isfinite(frame["measured_rsrp_dbm"])
        & np.isfinite(frame["station_id"])
        & frame["measured_rsrp_dbm"].between(-160.0, -20.0)
        & frame["is_target_27station_pci"].eq(1)
        & frame["rsrp_plausible_flag"].eq(1)
        & frame["dem_hit"].eq(1)
    )
    dropped = int((~valid).sum())
    frame = frame.loc[valid].copy()
    if dropped:
        print(f"实测长表中有{dropped:,}行未通过有效性/目标站/DEM检查，已排除。")

    frame["pci"] = frame["pci"].astype(int)
    frame["station_id"] = frame["station_id"].astype(int)
    frame["sector_index"] = frame["sector_index"].astype(int)
    frame["is_omnidirectional"] = frame["is_omnidirectional"].astype(int)

    observations = pd.DataFrame(
        {
            "source_row": np.arange(len(frame), dtype=np.int64),
            "x_m": frame["blender_x"].to_numpy(dtype=float),
            "y_m": frame["blender_y"].to_numpy(dtype=float),
            "station_id": frame["station_id"].to_numpy(dtype=int),
            "pci": frame["pci"].to_numpy(dtype=int),
            "rsrp_dbm": frame["measured_rsrp_dbm"].to_numpy(dtype=float),
            "ground_z_input_m": frame["ground_z_m"].to_numpy(dtype=float),
            "receiver_z_input_m": frame["receiver_z_m"].to_numpy(dtype=float),
            "rx_point_id": frame["rx_point_id"].astype(str).to_numpy(),
            "source_file": (frame["source_file"].astype(str).to_numpy() if "source_file" in frame.columns else np.asarray(["unknown"] * len(frame), dtype=object)),
            "source_row_original": (pd.to_numeric(frame["source_row"], errors="coerce").to_numpy() if "source_row" in frame.columns else np.arange(len(frame), dtype=np.int64)),
        }
    )

    station_count = int(frame["station_id"].nunique())
    if station_count != 27:
        raise ValueError(
            f"正式长表中检测到{station_count}个物理基站，不是要求的27个。"
            f"station_id={sorted(frame['station_id'].unique().tolist())}"
        )

    print(
        f"读取27站处理后实测长表: {path}\n"
        f"有效长表记录={len(frame):,}, 唯一RX位置={frame['rx_point_id'].nunique():,}, "
        f"基站={station_count}, PCI={frame['pci'].nunique()}"
    )
    return frame, observations



def build_independent_source_split(
    raw_long: pd.DataFrame,
    *,
    validation_fraction: float,
    seed: int,
    min_calibration_rows_per_pci: int,
) -> dict[str, Any]:
    """Create a trajectory/file-level calibration/validation split.

    Whole source files are held out; individual points are never randomly mixed
    across calibration and validation. A candidate validation file is accepted
    only if every PCI still has at least ``min_calibration_rows_per_pci`` rows
    in the calibration subset.
    """
    if "source_file" not in raw_long.columns:
        return {
            "enabled": False,
            "reason": "source_file column is unavailable; independent trajectory split cannot be constructed",
            "calibration_source_files": [],
            "validation_source_files": [],
        }
    files = sorted(str(v) for v in raw_long["source_file"].dropna().astype(str).unique())
    if len(files) < 2:
        return {
            "enabled": False,
            "reason": "fewer than two source files are available",
            "calibration_source_files": files,
            "validation_source_files": [],
        }
    fraction = float(np.clip(validation_fraction, 0.0, 0.8))
    target = max(1, min(len(files)-1, int(round(len(files) * fraction))))
    counts = raw_long.groupby(["source_file", "pci"]).size().unstack(fill_value=0)
    counts.index = counts.index.astype(str)
    total = counts.sum(axis=0)
    rng = np.random.default_rng(int(seed))
    order = [files[i] for i in rng.permutation(len(files))]
    held: list[str] = []
    held_counts = pd.Series(0, index=counts.columns, dtype=np.int64)
    for source in order:
        if len(held) >= target:
            break
        candidate = held_counts + counts.loc[source]
        remaining = total - candidate
        if bool((remaining >= int(min_calibration_rows_per_pci)).all()):
            held.append(source)
            held_counts = candidate
    if not held:
        return {
            "enabled": False,
            "reason": "no complete source file could be held out while preserving PCI calibration coverage",
            "calibration_source_files": files,
            "validation_source_files": [],
        }
    calibration = [f for f in files if f not in set(held)]
    return {
        "enabled": True,
        "split_unit": "source_file_complete_trajectory",
        "seed": int(seed),
        "target_validation_fraction": fraction,
        "source_file_count": len(files),
        "calibration_source_files": calibration,
        "validation_source_files": sorted(held),
        "calibration_row_count": int(raw_long["source_file"].astype(str).isin(calibration).sum()),
        "validation_row_count": int(raw_long["source_file"].astype(str).isin(held).sum()),
        "min_calibration_rows_per_pci": int(min_calibration_rows_per_pci),
    }


def _inclusive_centered_values(center: float, radius: float, step: float, low: float, high: float) -> list[float]:
    if radius <= 0 or step <= 0:
        return [float(np.clip(center, low, high))]
    values = np.arange(center-radius, center+radius+0.5*step, step, dtype=float)
    values = np.clip(values, low, high)
    return sorted({round(float(v), 8) for v in values} | {round(float(np.clip(center, low, high)), 8)})


def refine_candidate_with_final_propagation_settings(
    *,
    scene: Any,
    terrain: TerrainModel,
    station: Any,
    measurements: pd.DataFrame,
    sparse_surface: SurfaceInfo,
    cfg: dict[str, Any],
    coarse_best: dict[str, Any],
    output_dir: Path,
    force: bool,
    final_max_depth: int,
    final_edge_diffraction: bool,
    samples_per_tx: int,
    height_radius_m: float,
    height_step_m: float,
    azimuth_radius_deg: float,
    azimuth_step_deg: float,
    downtilt_radius_deg: float,
    downtilt_step_deg: float,
) -> dict[str, Any]:
    """Locally refine station parameters under the same fixed propagation model.

    ``final_max_depth`` and ``final_edge_diffraction`` are retained only for backward
    compatibility with older command lines; the reviewed method always uses depth=5
    with diffraction and edge diffraction enabled.
    """
    final_max_depth = 5
    final_edge_diffraction = True
    final_cfg = copy.deepcopy(cfg)
    final_cfg["radio"]["max_depth"] = 5
    final_cfg["radio"]["edge_diffraction"] = True
    final_cfg["radio"]["diffraction"] = True
    final_cfg["radio"]["diffuse_reflection"] = True

    search_cfg = final_cfg["search"]
    power_grid = _inclusive_grid_exact(
        float(search_cfg["power_min_dbm"]),
        float(search_cfg["power_max_dbm"]),
        float(search_cfg["power_step_db"]),
    )
    reference_power = float(station.initial_power_dbm)
    state = {
        "height_agl_m": float(coarse_best["height_agl_m"]),
        "azimuth_offset_deg": float(coarse_best["azimuth_offset_deg"]),
        "downtilt_delta_deg": float(coarse_best["downtilt_delta_deg"]),
    }
    cache_dir = output_dir / "cache_final_setting_refinement"
    history: list[dict[str, Any]] = []

    def eval_one(h: float, az: float, tilt: float, stage: str, idx: int) -> dict[str, Any]:
        candidate = Candidate(
            height_agl_m=float(h),
            azimuth_offset_deg=float(az),
            downtilt_delta_deg=float(tilt),
            reference_power_dbm=reference_power,
        )
        sim = run_candidate(
            scene=scene,
            terrain=terrain,
            station=station,
            candidate=candidate,
            surface=sparse_surface,
            cfg=final_cfg,
            samples_per_tx=int(samples_per_tx),
            cache_dir=cache_dir,
            force=force,
        )
        ev = evaluate_prediction(
            station=station,
            measurements=measurements,
            surface=sparse_surface,
            sector_rsrp_at_reference_dbm=sim["sector_rsrp_dbm"],
            reference_power_dbm=reference_power,
            power_candidates_dbm=power_grid,
            objective_mode=str(final_cfg["search"].get("objective_mode", "trajectory_balanced_pci_rmse")),
            power_regularization_lambda=float(final_cfg["search"].get("power_regularization_lambda", 0.0)),
            power_regularization_anchor_dbm=float(final_cfg["search"].get("power_regularization_anchor_dbm", 52.5)),
            min_trajectory_points=int(final_cfg["search"].get("reliability_min_trajectory_points", 20)),
            trajectory_weight_cap=int(final_cfg["search"].get("reliability_trajectory_weight_cap", 250)),
            min_pci_points=int(final_cfg["search"].get("reliability_min_pci_points", 3)),
            pci_weight_cap=int(final_cfg["search"].get("reliability_pci_weight_cap", 100)),
        )
        combined_score = _combined_selection_score_db2(candidate, ev, final_cfg["search"])
        row = {
            "stage": stage, "candidate_index": int(idx),
            "height_agl_m": float(h), "azimuth_offset_deg": float(az),
            "downtilt_delta_deg": float(tilt), "best_power_dbm": float(ev["best_power_dbm"]),
            "rmse_db": float(ev["rmse_db"]), "mae_db": float(ev["mae_db"]),
            "bias_sim_minus_meas_db": float(ev["bias_sim_minus_meas_db"]),
            "max_depth": int(final_max_depth), "edge_diffraction": bool(final_edge_diffraction),
            "samples_per_tx": int(samples_per_tx),
            "combined_selection_score_db2": float(combined_score),
            "geometry_regularization_penalty_db2": float(_geometry_regularization_penalty_db2(candidate, final_cfg["search"])),
        }
        history.append(row)
        return {"candidate": candidate, "simulation": sim, "evaluation": ev, "selection_score_db2": float(combined_score)}

    stages: list[tuple[str, list[float]]] = [
        ("height_final_physics", _inclusive_centered_values(state["height_agl_m"], height_radius_m, height_step_m, float(search_cfg["height_min_m"]), float(search_cfg["height_max_m"]))),
    ]
    if not station.is_omnidirectional:
        stages += [
            ("azimuth_final_physics", _inclusive_centered_values(state["azimuth_offset_deg"], azimuth_radius_deg, azimuth_step_deg, float(search_cfg["azimuth_offset_min_deg"]), float(search_cfg["azimuth_offset_max_deg"]))),
            ("downtilt_final_physics", _inclusive_centered_values(state["downtilt_delta_deg"], downtilt_radius_deg, downtilt_step_deg, float(search_cfg["downtilt_delta_min_deg"]), float(search_cfg["downtilt_delta_max_deg"]))),
        ]

    best_payload = None
    for stage, values in stages:
        candidates=[]
        for idx, value in enumerate(values,1):
            h,az,tilt=state["height_agl_m"],state["azimuth_offset_deg"],state["downtilt_delta_deg"]
            if stage.startswith("height"): h=value
            elif stage.startswith("azimuth"): az=value
            else: tilt=value
            candidates.append(eval_one(h,az,tilt,stage,idx))
        selected=min(candidates,key=lambda x:x["selection_score_db2"])
        c=selected["candidate"]
        state.update(height_agl_m=float(c.height_agl_m),azimuth_offset_deg=float(c.azimuth_offset_deg),downtilt_delta_deg=float(c.downtilt_delta_deg))
        best_payload=selected

    if best_payload is None:
        best_payload=eval_one(state["height_agl_m"],state["azimuth_offset_deg"],state["downtilt_delta_deg"],"final_physics_baseline",1)
    pd.DataFrame(history).to_csv(output_dir / "final_setting_refinement_history.csv", index=False, encoding="utf-8-sig")
    c=best_payload["candidate"]; ev=best_payload["evaluation"]; sim=best_payload["simulation"]
    return {
        "height_agl_m": float(c.height_agl_m),
        "azimuth_offset_deg": float(c.azimuth_offset_deg),
        "azimuth_offset_rad": math.radians(float(c.azimuth_offset_deg)),
        "downtilt_delta_deg": float(c.downtilt_delta_deg),
        "absolute_downtilt_deg": float(c.downtilt_delta_deg),
        "shared_power_dbm": float(ev["best_power_dbm"]),
        "refinement_rmse_db": float(ev["rmse_db"]),
        "refinement_mae_db": float(ev["mae_db"]),
        "refinement_bias_db": float(ev["bias_sim_minus_meas_db"]),
        "paired_point_count": int(ev["paired_point_count"]),
        "per_pci": ev["per_pci"],
        "ground_z_m": float(sim["ground_z_m"]),
        "tx_absolute_z_m": float(sim["tx_z_m"]),
        "alphas_rad": sim["alphas_rad"].tolist(),
        "beta_rad": float(sim["beta_rad"]),
        "final_refinement_max_depth": int(final_max_depth),
        "final_refinement_edge_diffraction": bool(final_edge_diffraction),
        "final_refinement_samples_per_tx": int(samples_per_tx),
    }


def _wrap_rad(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def _weighted_circular_mean_rad(angles_rad: np.ndarray, weights: np.ndarray | None = None) -> float:
    angles = np.asarray(angles_rad, dtype=float)
    mask = np.isfinite(angles)
    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        mask &= np.isfinite(weights) & (weights > 0.0)
    if not np.any(mask):
        raise ValueError("没有有限角度可用于圆均值")
    angles = angles[mask]
    if weights is None:
        weights = np.ones(len(angles), dtype=float)
    else:
        weights = weights[mask]
    vector = np.sum(weights * np.exp(1j * angles))
    if abs(vector) < 1e-12:
        return float(angles[0])
    return _wrap_rad(float(np.angle(vector)))


def _estimate_sector_alpha_rad(part: pd.DataFrame, top_fraction: float) -> tuple[float, int]:
    data = part.loc[
        np.isfinite(part["bearing_from_tx_math_deg"])
        & np.isfinite(part["measured_rsrp_dbm"]),
        ["bearing_from_tx_math_deg", "measured_rsrp_dbm"],
    ].copy()
    if data.empty:
        raise ValueError("该PCI没有可用于方向估计的方位角/RSRP数据")

    fraction = float(np.clip(top_fraction, 0.02, 1.0))
    keep_count = min(len(data), max(20, int(math.ceil(len(data) * fraction))))
    data = data.nlargest(keep_count, "measured_rsrp_dbm")

    angles = np.radians(data["bearing_from_tx_math_deg"].to_numpy(dtype=float))
    strengths = data["measured_rsrp_dbm"].to_numpy(dtype=float)
    weights = np.exp(np.clip((strengths - np.nanmax(strengths)) / 6.0, -12.0, 0.0))
    return _weighted_circular_mean_rad(angles, weights), int(len(data))


def _fit_sector_rotation_order(
    estimates: dict[int, float],
    estimate_weights: dict[int, int],
) -> tuple[str, float, dict[int, float], float, list[float]]:
    """
    sector_index只表示PCI扇区编号，未必天然对应顺时针或逆时针排列。
    对两种120°排列都进行拟合，并选择与实测强信号方向更一致的一种：

      A: sector2 = base - 120°, sector3 = base + 120°
      B: sector2 = base + 120°, sector3 = base - 120°
    """
    candidates = {
        "sector2_minus120": {
            1: 0.0,
            2: -2.0 * math.pi / 3.0,
            3: 2.0 * math.pi / 3.0,
        },
        "sector2_plus120": {
            1: 0.0,
            2: 2.0 * math.pi / 3.0,
            3: -2.0 * math.pi / 3.0,
        },
    }

    weights = np.asarray(
        [max(int(estimate_weights[i]), 1) for i in (1, 2, 3)],
        dtype=float,
    )
    fitted: list[tuple[float, str, float, dict[int, float], list[float]]] = []

    for order_name, offsets in candidates.items():
        base_candidates = np.asarray(
            [_wrap_rad(estimates[i] - offsets[i]) for i in (1, 2, 3)],
            dtype=float,
        )
        base = _weighted_circular_mean_rad(base_candidates, weights)
        residuals = np.asarray(
            [_wrap_rad(value - base) for value in base_candidates],
            dtype=float,
        )
        weighted_rms_rad = float(
            np.sqrt(np.average(residuals ** 2, weights=weights))
        )
        fitted.append(
            (
                weighted_rms_rad,
                order_name,
                base,
                offsets,
                np.degrees(residuals).tolist(),
            )
        )

    score_rad, order_name, base, offsets, residuals_deg = min(
        fitted,
        key=lambda item: item[0],
    )
    return (
        order_name,
        base,
        offsets,
        math.degrees(score_rad),
        residuals_deg,
    )


def build_all_27_station_configs(
    raw_long: pd.DataFrame,
    top_fraction: float,
    initial_power_dbm: float = 53.5,
) -> tuple[dict[int, RuntimeStationConfig], list[dict[str, Any]]]:
    """
    从正式27站长表生成全部基站配置。

    关键修正：
      1. 三扇区顺/逆时针排列不再写死，自动比较两种120°排列；
      2. 下倾角以0°为基准，搜索变量就是绝对下倾角，不再使用统一假设12°；
      3. 全向站保持单PCI和0°下倾。
    """
    result: dict[int, RuntimeStationConfig] = {}
    diagnostics: list[dict[str, Any]] = []

    for station_id, group in raw_long.groupby("station_id", sort=True):
        sid = int(station_id)
        label_series = group["station_label"].dropna().astype(str)
        label = label_series.mode().iat[0] if not label_series.empty else f"station-{sid}"
        x_m = float(group["tx_x_initial_m"].median())
        y_m = float(group["tx_y_initial_m"].median())
        omni = bool(group["is_omnidirectional"].fillna(0).astype(int).max())

        sector_rows = (
            group[["sector_index", "pci"]]
            .dropna()
            .drop_duplicates()
            .sort_values(["sector_index", "pci"])
        )

        if omni:
            pcis = tuple(sorted(group["pci"].dropna().astype(int).unique().tolist()))
            if len(pcis) != 1:
                raise ValueError(f"{sid}号全向站应只有1个PCI，实际为{pcis}")

            config = RuntimeStationConfig(
                station_id=sid,
                label=label,
                x_m=x_m,
                y_m=y_m,
                pcis=pcis,
                initial_alphas_rad=(0.0,),
                original_downtilt_deg=0.0,
                initial_power_dbm=float(initial_power_dbm),
                is_omnidirectional=True,
            )
            diagnostics.append(
                {
                    "station_id": sid,
                    "label": label,
                    "antenna_type": "omnidirectional",
                    "pcis": ";".join(map(str, pcis)),
                    "selected_sector_order": "omnidirectional",
                    "base_alpha_rad": 0.0,
                    "base_alpha_deg": 0.0,
                    "direction_fit_rms_deg": 0.0,
                    "direction_points_used": 0,
                }
            )
        else:
            mapping = {
                int(row.sector_index): int(row.pci)
                for row in sector_rows.itertuples(index=False)
            }
            if set(mapping) != {1, 2, 3}:
                raise ValueError(
                    f"{sid}号三扇区站缺少完整sector_index=1,2,3映射: {mapping}"
                )

            estimates: dict[int, float] = {}
            estimate_weights: dict[int, int] = {}
            for sector_index in (1, 2, 3):
                pci = mapping[sector_index]
                estimate, used = _estimate_sector_alpha_rad(
                    group.loc[group["pci"].eq(pci)],
                    top_fraction=top_fraction,
                )
                estimates[sector_index] = estimate
                estimate_weights[sector_index] = used

            (
                selected_order,
                base,
                offsets,
                fit_rms_deg,
                fit_residuals_deg,
            ) = _fit_sector_rotation_order(estimates, estimate_weights)

            alphas = tuple(
                _wrap_rad(base + offsets[i])
                for i in (1, 2, 3)
            )
            pcis = tuple(mapping[i] for i in (1, 2, 3))

            config = RuntimeStationConfig(
                station_id=sid,
                label=label,
                x_m=x_m,
                y_m=y_m,
                pcis=pcis,
                initial_alphas_rad=alphas,
                # 搜索量直接表示绝对下倾角，范围由main设置为[-15°, +15°]
                original_downtilt_deg=0.0,
                initial_power_dbm=float(initial_power_dbm),
                is_omnidirectional=False,
            )
            diagnostics.append(
                {
                    "station_id": sid,
                    "label": label,
                    "antenna_type": "three_sector",
                    "pcis": ";".join(map(str, pcis)),
                    "selected_sector_order": selected_order,
                    "base_alpha_rad": base,
                    "base_alpha_deg": math.degrees(base),
                    "alpha_1_rad": alphas[0],
                    "alpha_2_rad": alphas[1],
                    "alpha_3_rad": alphas[2],
                    "estimated_alpha_1_deg": math.degrees(estimates[1]),
                    "estimated_alpha_2_deg": math.degrees(estimates[2]),
                    "estimated_alpha_3_deg": math.degrees(estimates[3]),
                    "direction_fit_rms_deg": fit_rms_deg,
                    "direction_fit_residual_1_deg": fit_residuals_deg[0],
                    "direction_fit_residual_2_deg": fit_residuals_deg[1],
                    "direction_fit_residual_3_deg": fit_residuals_deg[2],
                    "direction_points_used": int(sum(estimate_weights.values())),
                }
            )

        config.validate(tolerance_deg=1.0)
        result[sid] = config

    if len(result) != 27:
        raise ValueError(f"生成的基站配置数量为{len(result)}，不是27")

    return result, diagnostics


def parse_selected_station_ids(text: str | None, available: Sequence[int]) -> list[int]:
    all_ids = sorted(int(v) for v in available)
    if text is None or str(text).strip().lower() in {"", "all", "*", "27"}:
        return all_ids
    selected = sorted({int(v.strip()) for v in str(text).split(",") if v.strip()})
    unknown = sorted(set(selected) - set(all_ids))
    if unknown:
        raise ValueError(f"27站配置中不存在这些station_id: {unknown}")
    return selected


def _apply_common_azimuth_offset_general(
    initial_alphas_rad: Sequence[float],
    offset_deg: float,
) -> tuple[float, ...]:
    values = tuple(
        _wrap_rad(float(alpha) + math.radians(float(offset_deg)))
        for alpha in initial_alphas_rad
    )
    if len(values) == 1:
        return values
    if len(values) == 3:
        validate_sector_spacing(values, tolerance_deg=1.0)
        return values
    raise ValueError(f"只支持1扇区或3扇区，实际alpha数量={len(values)}")


def _candidate_row_general(
    station: RuntimeStationConfig,
    candidate: Candidate,
    evaluation: Dict[str, Any],
    simulation: Dict[str, Any],
    pass_index: int,
    stage: str,
    candidate_index: int,
    elapsed_s: float,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "station_id": station.station_id,
        "pass_index": pass_index,
        "stage": stage,
        "candidate_index": candidate_index,
        "height_agl_m": candidate.height_agl_m,
        "tx_ground_z_m": simulation["ground_z_m"],
        "tx_absolute_z_m": simulation["tx_z_m"],
        "azimuth_offset_deg": candidate.azimuth_offset_deg,
        "azimuth_offset_rad": math.radians(candidate.azimuth_offset_deg),
        "downtilt_delta_deg": candidate.downtilt_delta_deg,
        "downtilt_delta_rad": math.radians(candidate.downtilt_delta_deg),
        "absolute_downtilt_deg": station.original_downtilt_deg + candidate.downtilt_delta_deg,
        "absolute_beta_rad": simulation["beta_rad"],
        "reference_power_dbm": candidate.reference_power_dbm,
        "optimized_shared_power_dbm": evaluation["best_power_dbm"],
        "pooled_equal_pci_rmse_db": evaluation["rmse_db"],
        "mae_db": evaluation["mae_db"],
        "bias_sim_minus_meas_db": evaluation["bias_sim_minus_meas_db"],
        "paired_point_count": evaluation["paired_point_count"],
        "cache_hit": simulation["cache_hit"],
        "cache_key": simulation["cache_key"],
        "elapsed_s": elapsed_s,
    }
    for index in range(3):
        row[f"alpha_{index + 1}_rad"] = (
            float(simulation["alphas_rad"][index])
            if index < len(simulation["alphas_rad"])
            else np.nan
        )
    return row


def install_general_sector_support() -> None:
    """让现有 simulator/optimizer 同时支持3扇区与单PCI全向站。"""
    simulator_module.apply_common_azimuth_offset = _apply_common_azimuth_offset_general
    optimizer_module._candidate_row = _candidate_row_general


def configure_tx_array_for_station(scene: Any, cfg: Dict[str, Any], station: RuntimeStationConfig) -> None:
    try:
        from sionna.rt import PlanarArray
    except Exception as exc:
        raise RuntimeError("无法导入Sionna RT PlanarArray") from exc

    antenna = cfg["antenna"]
    scene.tx_array = PlanarArray(
        num_rows=int(antenna["num_rows"]),
        num_cols=int(antenna["num_cols"]),
        vertical_spacing=float(antenna.get("vertical_spacing", 0.5)),
        horizontal_spacing=float(antenna.get("horizontal_spacing", 0.5)),
        pattern=str(antenna["pattern"]),
        polarization=str(antenna["polarization"]),
    )



@dataclass(frozen=True)
class DenseGridDefinition:
    """512m规则XY网格及其DEM、接收高度和建筑占地掩膜。"""

    nx: int
    ny: int
    cell_size_m: float
    x_min_m: float
    y_min_m: float
    x_centers_m: np.ndarray
    y_centers_m: np.ndarray
    x_m: np.ndarray
    y_m: np.ndarray
    ground_z_m: np.ndarray
    receiver_z_m: np.ndarray
    building_mask: np.ndarray

    @property
    def extent(self) -> list[float]:
        return [
            self.x_min_m,
            self.x_min_m + self.nx * self.cell_size_m,
            self.y_min_m,
            self.y_min_m + self.ny * self.cell_size_m,
        ]


def _trimesh_geometries(path: Path) -> list[Any]:
    """原样读取PLY中的所有三角网格，不修改坐标和三角面。"""
    import trimesh

    loaded = trimesh.load(str(Path(path).resolve()), process=False)
    if isinstance(loaded, trimesh.Trimesh):
        return [loaded]
    if isinstance(loaded, trimesh.Scene):
        return [
            geometry
            for geometry in loaded.geometry.values()
            if isinstance(geometry, trimesh.Trimesh)
        ]
    return []


def load_building_projection_triangles(
    building_paths: Sequence[str | Path],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """
    提取全部建筑PLY三角面的XY投影，用于生成建筑内部掩膜。

    这里不会按ground范围、Z范围或文件名排除任何建筑。垂直墙面投影为
    零面积线段，对二维占地掩膜没有面积贡献，因此只忽略零面积投影面；
    屋顶、地板和斜屋面会形成建筑占地。
    """
    triangle_parts: list[np.ndarray] = []
    diagnostics: list[dict[str, Any]] = []

    for raw_path in building_paths:
        path = Path(raw_path).expanduser().resolve()
        report: dict[str, Any] = {
            "path": str(path),
            "loaded": False,
            "mesh_count": 0,
            "face_count": 0,
            "nondegenerate_xy_triangle_count": 0,
        }
        try:
            geometries = _trimesh_geometries(path)
            report["mesh_count"] = len(geometries)
            for mesh in geometries:
                vertices = np.asarray(mesh.vertices, dtype=np.float64)
                faces = np.asarray(mesh.faces, dtype=np.int64)
                if (
                    vertices.ndim != 2
                    or vertices.shape[1] < 3
                    or faces.ndim != 2
                    or faces.shape[1] != 3
                    or len(vertices) == 0
                    or len(faces) == 0
                ):
                    continue
                if faces.min() < 0 or faces.max() >= len(vertices):
                    raise IndexError(
                        f"建筑PLY面索引越界: vertices={len(vertices)}, "
                        f"faces=[{faces.min()}, {faces.max()}]"
                    )
                triangles = vertices[faces][:, :, :2]
                edge1 = triangles[:, 1] - triangles[:, 0]
                edge2 = triangles[:, 2] - triangles[:, 0]
                twice_area = np.abs(
                    edge1[:, 0] * edge2[:, 1]
                    - edge1[:, 1] * edge2[:, 0]
                )
                valid = np.isfinite(triangles).all(axis=(1, 2)) & (twice_area > 1e-8)
                if np.any(valid):
                    triangle_parts.append(triangles[valid])
                report["face_count"] += int(len(faces))
                report["nondegenerate_xy_triangle_count"] += int(valid.sum())
                if len(vertices):
                    bounds = np.vstack([vertices.min(axis=0), vertices.max(axis=0)])
                    report.setdefault("bounds", []).append(bounds.tolist())
            report["loaded"] = bool(report["mesh_count"])
        except Exception as exc:
            report["error"] = f"{type(exc).__name__}: {exc}"
        diagnostics.append(report)

    if not triangle_parts:
        raise RuntimeError(
            "所有已加入场景的建筑PLY都没有可用于二维建筑占地掩膜的非退化XY三角面。"
        )

    return np.concatenate(triangle_parts, axis=0), diagnostics


def _rasterize_triangle_into_mask(
    mask: np.ndarray,
    triangle_xy: np.ndarray,
    x_centers: np.ndarray,
    y_centers: np.ndarray,
) -> bool:
    """把一个XY三角形栅格化到规则网格；返回它是否与当前地图相交。"""
    tri = np.asarray(triangle_xy, dtype=np.float64)
    min_x = float(np.min(tri[:, 0]))
    max_x = float(np.max(tri[:, 0]))
    min_y = float(np.min(tri[:, 1]))
    max_y = float(np.max(tri[:, 1]))

    if (
        max_x < float(x_centers[0])
        or min_x > float(x_centers[-1])
        or max_y < float(y_centers[0])
        or min_y > float(y_centers[-1])
    ):
        return False

    ix0 = max(0, int(np.searchsorted(x_centers, min_x, side="left")))
    ix1 = min(len(x_centers), int(np.searchsorted(x_centers, max_x, side="right")))
    iy0 = max(0, int(np.searchsorted(y_centers, min_y, side="left")))
    iy1 = min(len(y_centers), int(np.searchsorted(y_centers, max_y, side="right")))
    if ix1 <= ix0 or iy1 <= iy0:
        return False

    xx, yy = np.meshgrid(x_centers[ix0:ix1], y_centers[iy0:iy1])
    a, b, c = tri
    v0 = b - a
    v1 = c - a
    denominator = v0[0] * v1[1] - v0[1] * v1[0]
    if not np.isfinite(denominator) or abs(denominator) <= 1e-12:
        return False

    px = xx - a[0]
    py = yy - a[1]
    u = (px * v1[1] - py * v1[0]) / denominator
    v = (v0[0] * py - v0[1] * px) / denominator
    tolerance = 1e-9
    inside = (
        (u >= -tolerance)
        & (v >= -tolerance)
        & ((u + v) <= 1.0 + tolerance)
    )
    mask[iy0:iy1, ix0:ix1] |= inside
    return True


def create_dense_grid_with_building_mask(
    terrain: TerrainModel,
    building_triangles_xy: np.ndarray,
    center_x: float,
    center_y: float,
    size_x_m: int,
    size_y_m: int,
    cell_size_m: float,
    rx_height_agl_m: float,
    buffer_cells: int,
) -> tuple[DenseGridDefinition, dict[str, Any]]:
    """创建完整XY网格，并根据建筑PLY生成建筑内部掩膜。"""
    nx = int(round(size_x_m / cell_size_m))
    ny = int(round(size_y_m / cell_size_m))
    if (
        abs(nx * cell_size_m - size_x_m) > 1e-6
        or abs(ny * cell_size_m - size_y_m) > 1e-6
    ):
        raise ValueError("地图尺寸必须能被cell_size整除")

    x_min = float(center_x) - float(size_x_m) / 2.0
    y_min = float(center_y) - float(size_y_m) / 2.0
    x_centers = x_min + (np.arange(nx, dtype=np.float64) + 0.5) * cell_size_m
    y_centers = y_min + (np.arange(ny, dtype=np.float64) + 0.5) * cell_size_m
    x_grid, y_grid = np.meshgrid(x_centers, y_centers)

    mask = np.zeros((ny, nx), dtype=bool)
    intersecting_triangles = 0
    for triangle in np.asarray(building_triangles_xy, dtype=np.float64):
        if _rasterize_triangle_into_mask(mask, triangle, x_centers, y_centers):
            intersecting_triangles += 1

    requested_buffer = max(int(buffer_cells), 0)
    if requested_buffer > 0 and np.any(mask):
        from scipy.ndimage import binary_dilation

        mask = binary_dilation(mask, iterations=requested_buffer)

    ground = terrain.query(x_grid, y_grid)
    receiver = ground + float(rx_height_agl_m)
    grid = DenseGridDefinition(
        nx=nx,
        ny=ny,
        cell_size_m=float(cell_size_m),
        x_min_m=x_min,
        y_min_m=y_min,
        x_centers_m=x_centers,
        y_centers_m=y_centers,
        x_m=x_grid,
        y_m=y_grid,
        ground_z_m=ground,
        receiver_z_m=receiver,
        building_mask=mask,
    )
    diagnostics = {
        "nx": nx,
        "ny": ny,
        "cell_size_m": float(cell_size_m),
        "map_cell_count": int(nx * ny),
        "input_building_xy_triangle_count": int(len(building_triangles_xy)),
        "building_triangles_intersecting_map_bbox": int(intersecting_triangles),
        "building_mask_buffer_cells": requested_buffer,
        "building_cell_count": int(mask.sum()),
        "outdoor_cell_count": int((~mask).sum()),
        "building_fraction": float(mask.mean()),
        "receiver_height_minus_dem_min_m": float(np.nanmin(receiver - ground)),
        "receiver_height_minus_dem_max_m": float(np.nanmax(receiver - ground)),
        "receiver_surface_z_min_m": float(np.nanmin(receiver)),
        "receiver_surface_z_max_m": float(np.nanmax(receiver)),
        "receiver_surface_relief_m": float(np.nanmax(receiver) - np.nanmin(receiver)),
    }
    return grid, diagnostics


def _export_outdoor_surface_mesh(
    output_path: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
) -> None:
    import trimesh

    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        process=False,
        validate=False,
    )
    if len(mesh.face_normals) and np.nanmedian(mesh.face_normals[:, 2]) < 0:
        mesh.faces = mesh.faces[:, ::-1]
    mesh.export(output_path, file_type="ply", encoding="binary")


def build_dense_outdoor_measurement_surface(
    terrain: TerrainModel,
    grid: DenseGridDefinition,
    rx_height_agl_m: float,
    output_path: Path,
) -> SurfaceInfo:
    """
    构建真实DEM+1.5m不平坦接收面，但不为建筑内部单元生成三角面。

    顶点Z仍逐点取DEM+接收高度；建筑内部不属于测量面，因此不会再把
    室外车载接收面穿入建筑内部，也减少了无意义的射线采样消耗。
    """
    nx, ny = grid.nx, grid.ny
    x_edges = grid.x_min_m + np.arange(nx + 1, dtype=np.float64) * grid.cell_size_m
    y_edges = grid.y_min_m + np.arange(ny + 1, dtype=np.float64) * grid.cell_size_m
    xx, yy = np.meshgrid(x_edges, y_edges)
    zz = terrain.query(xx, yy) + float(rx_height_agl_m)
    vertices = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()]).astype(np.float32)

    outdoor_iy, outdoor_ix = np.nonzero(~grid.building_mask)
    if len(outdoor_ix) == 0:
        raise RuntimeError("建筑掩膜覆盖了整张地图，没有室外接收单元")

    v00 = outdoor_iy.astype(np.int64) * (nx + 1) + outdoor_ix.astype(np.int64)
    v10 = v00 + 1
    v01 = v00 + (nx + 1)
    v11 = v01 + 1
    faces = np.empty((len(outdoor_ix) * 2, 3), dtype=np.int32)
    faces[0::2] = np.column_stack([v00, v10, v11]).astype(np.int32)
    faces[1::2] = np.column_stack([v00, v11, v01]).astype(np.int32)
    _export_outdoor_surface_mesh(output_path, vertices, faces)

    center_x = grid.x_m[outdoor_iy, outdoor_ix]
    center_y = grid.y_m[outdoor_iy, outdoor_ix]
    ground = grid.ground_z_m[outdoor_iy, outdoor_ix]
    return SurfaceInfo(
        path=Path(output_path).expanduser().resolve(),
        n_cells=int(len(outdoor_ix)),
        n_faces=int(2 * len(outdoor_ix)),
        cell_ix=outdoor_ix.astype(np.int32),
        cell_iy=outdoor_iy.astype(np.int32),
        cell_center_x=center_x.astype(np.float64),
        cell_center_y=center_y.astype(np.float64),
        cell_ground_z=ground.astype(np.float64),
        cell_rx_z=(ground + float(rx_height_agl_m)).astype(np.float64),
        nx=nx,
        ny=ny,
    )


def remove_measurements_inside_buildings(
    measurements: pd.DataFrame,
    building_mask: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """从室外校准数据中移除落在建筑占地掩膜内的1m聚合点。"""
    ix = pd.to_numeric(measurements["ix"], errors="coerce").to_numpy(dtype=float)
    iy = pd.to_numeric(measurements["iy"], errors="coerce").to_numpy(dtype=float)
    valid_index = np.isfinite(ix) & np.isfinite(iy)
    if not valid_index.all():
        raise ValueError("实测聚合数据中存在无效ix/iy")
    ix_int = ix.astype(np.int64)
    iy_int = iy.astype(np.int64)
    ny, nx = building_mask.shape
    in_bounds = (ix_int >= 0) & (ix_int < nx) & (iy_int >= 0) & (iy_int < ny)
    if not in_bounds.all():
        raise ValueError("实测聚合点ix/iy超出当前512m地图范围")
    inside = building_mask[iy_int, ix_int]
    outdoor = measurements.loc[~inside].copy().reset_index(drop=True)
    excluded = measurements.loc[inside].copy().reset_index(drop=True)
    return outdoor, excluded


def run_candidate_multibatch_linear_average(
    scene: Any,
    terrain: TerrainModel,
    station: RuntimeStationConfig,
    candidate: Candidate,
    surface: SurfaceInfo,
    cfg: Dict[str, Any],
    samples_per_batch: int,
    batch_count: int,
    seed_step: int,
    cache_dir: Path | None,
    force: bool,
) -> Dict[str, Any]:
    """
    使用多个独立seed运行最终无线电地图，并在线性功率域平均。

    单批次未命中以0功率参与平均；只有所有批次都未命中时，最终单元才为NaN。
    这样既不直接在dBm域错误平均，也不会用插值伪造建筑后方信号。
    """
    if int(batch_count) < 1:
        raise ValueError("final batch_count必须>=1")
    if int(samples_per_batch) < 1:
        raise ValueError("final samples_per_batch必须>=1")
    if int(seed_step) == 0:
        raise ValueError("final seed_step不能为0")

    base_seed = int(cfg["radio"]["seed"])
    accumulated_mw: np.ndarray | None = None
    hit_counts: np.ndarray | None = None
    batch_results: list[dict[str, Any]] = []

    for batch_index in range(int(batch_count)):
        batch_cfg = copy.deepcopy(cfg)
        batch_seed = base_seed + batch_index * int(seed_step)
        batch_cfg["radio"]["seed"] = int(batch_seed)
        print(
            f"    最终地图批次 {batch_index + 1}/{batch_count}: "
            f"seed={batch_seed}, samples/tx={int(samples_per_batch):,}, "
            f"max_depth={batch_cfg['radio']['max_depth']}, "
            f"edge_diffraction={batch_cfg['radio'].get('edge_diffraction', False)}"
        )
        result = run_candidate(
            scene=scene,
            terrain=terrain,
            station=station,
            candidate=candidate,
            surface=surface,
            cfg=batch_cfg,
            samples_per_tx=int(samples_per_batch),
            cache_dir=cache_dir,
            force=force,
        )
        rsrp_dbm = np.asarray(result["sector_rsrp_dbm"], dtype=np.float64)
        finite = np.isfinite(rsrp_dbm)
        batch_mw = np.zeros(rsrp_dbm.shape, dtype=np.float64)
        batch_mw[finite] = np.power(10.0, rsrp_dbm[finite] / 10.0)
        if accumulated_mw is None:
            accumulated_mw = np.zeros_like(batch_mw)
            hit_counts = np.zeros(batch_mw.shape, dtype=np.uint16)
        accumulated_mw += batch_mw
        assert hit_counts is not None
        hit_counts += finite.astype(np.uint16)
        batch_results.append(
            {
                "batch_index": batch_index + 1,
                "seed": int(batch_seed),
                "cache_hit": bool(result.get("cache_hit", False)),
                "cache_key": str(result.get("cache_key", "")),
                "finite_cell_sector_count": int(finite.sum()),
            }
        )

    assert accumulated_mw is not None and hit_counts is not None
    average_mw = accumulated_mw / float(batch_count)
    averaged_dbm = np.full(average_mw.shape, np.nan, dtype=np.float64)
    positive = np.isfinite(average_mw) & (average_mw > 0.0)
    averaged_dbm[positive] = 10.0 * np.log10(average_mw[positive])

    first = batch_results[0]
    combined_key = hashlib.sha256(
        json.dumps(batch_results, sort_keys=True).encode("utf-8")
    ).hexdigest()[:24]
    # 几何/方向元数据由最后一次run_candidate返回；各批几何完全相同。
    result_metadata = result
    return {
        "sector_rsrp_dbm": averaged_dbm.astype(np.float32),
        "batch_hit_counts": hit_counts,
        "alphas_rad": np.asarray(result_metadata["alphas_rad"], dtype=np.float64),
        "beta_rad": float(result_metadata["beta_rad"]),
        "ground_z_m": float(result_metadata["ground_z_m"]),
        "tx_z_m": float(result_metadata["tx_z_m"]),
        "cache_hit": bool(all(item["cache_hit"] for item in batch_results)),
        "cache_key": combined_key,
        "batch_count": int(batch_count),
        "samples_per_batch": int(samples_per_batch),
        "total_samples_per_tx": int(batch_count) * int(samples_per_batch),
        "batch_seeds": [item["seed"] for item in batch_results],
        "batch_results": batch_results,
        "first_batch_cache_key": first["cache_key"],
    }


def sector_values_to_full_maps(
    surface: SurfaceInfo,
    sector_values: np.ndarray,
    grid: DenseGridDefinition,
    fill_value: float | int = np.nan,
) -> np.ndarray:
    values = np.asarray(sector_values)
    if values.ndim != 2 or values.shape[1] != surface.n_cells:
        raise ValueError(
            f"扇区结果形状异常: {values.shape}, expected (*,{surface.n_cells})"
        )
    # NaN填充值要求浮点类型；批次命中次数则使用整数0填充。
    if isinstance(fill_value, float) and np.isnan(fill_value):
        output_dtype = np.result_type(values.dtype, np.float32)
    else:
        output_dtype = values.dtype
    full = np.full(
        (values.shape[0], grid.ny, grid.nx),
        fill_value,
        dtype=output_dtype,
    )
    full[:, surface.cell_iy, surface.cell_ix] = values
    full[:, grid.building_mask] = fill_value
    return full


def save_final_npz_general(
    path: Path,
    station: RuntimeStationConfig,
    surface: SurfaceInfo,
    grid: DenseGridDefinition,
    sector_rsrp_dbm: np.ndarray,
    batch_hit_counts: np.ndarray,
    best: Dict[str, Any],
) -> None:
    sector_maps = sector_values_to_full_maps(surface, sector_rsrp_dbm, grid)
    hit_maps = sector_values_to_full_maps(
        surface, batch_hit_counts, grid, fill_value=0
    )
    finite_any = np.any(np.isfinite(sector_maps), axis=0)
    best_index = np.argmax(
        np.where(np.isfinite(sector_maps), sector_maps, -np.inf), axis=0
    )
    best_rsrp = np.take_along_axis(sector_maps, best_index[None, ...], axis=0)[0]
    best_rsrp = np.where(finite_any, best_rsrp, np.nan)
    pci_values = np.asarray(station.pcis, dtype=np.int32)
    best_pci = np.where(finite_any, pci_values[best_index], -1).astype(np.int32)
    outdoor_mask = ~grid.building_mask
    no_hit_outdoor_mask = outdoor_mask & ~finite_any

    np.savez_compressed(
        path,
        station_id=np.asarray([station.station_id], dtype=np.int32),
        pcis=pci_values,
        x_m=grid.x_m,
        y_m=grid.y_m,
        ground_z_m=grid.ground_z_m,
        receiver_z_m=grid.receiver_z_m,
        building_mask=grid.building_mask,
        outdoor_valid_mask=outdoor_mask,
        no_hit_outdoor_mask=no_hit_outdoor_mask,
        sector_rsrp_dbm=sector_maps,
        sector_batch_hit_counts=hit_maps.astype(np.uint16),
        best_server_rsrp_dbm=best_rsrp,
        best_server_pci=best_pci,
        height_agl_m=np.asarray([best["height_agl_m"]], dtype=np.float32),
        tx_absolute_z_m=np.asarray([best["tx_absolute_z_m"]], dtype=np.float32),
        shared_power_dbm=np.asarray([best["shared_power_dbm"]], dtype=np.float32),
        alphas_rad=np.asarray(best["alphas_rad"], dtype=np.float64),
        beta_rad=np.asarray([best["beta_rad"]], dtype=np.float64),
        final_batch_count=np.asarray([best["final_batch_count"]], dtype=np.int32),
        final_samples_per_batch=np.asarray(
            [best["final_samples_per_batch"]], dtype=np.int64
        ),
        final_total_samples_per_tx=np.asarray(
            [best["final_total_samples_per_tx"]], dtype=np.int64
        ),
        final_max_depth=np.asarray([best["final_max_depth"]], dtype=np.int32),
        final_edge_diffraction=np.asarray(
            [best["final_edge_diffraction"]], dtype=np.bool_
        ),
    )


def compute_radio_map_coverage_diagnostics(
    station: RuntimeStationConfig,
    surface: SurfaceInfo,
    grid: DenseGridDefinition,
    sector_rsrp_dbm: np.ndarray,
    batch_hit_counts: np.ndarray,
    measurements: pd.DataFrame,
    final_evaluation: Dict[str, Any],
    final_simulation: Dict[str, Any],
) -> Dict[str, Any]:
    maps = sector_values_to_full_maps(surface, sector_rsrp_dbm, grid)
    hit_maps = sector_values_to_full_maps(
        surface, batch_hit_counts, grid, fill_value=0
    )
    outdoor = ~grid.building_mask
    best_hit = np.any(np.isfinite(maps), axis=0) & outdoor
    outdoor_count = int(outdoor.sum())
    building_count = int(grid.building_mask.sum())
    map_hit_count = int(best_hit.sum())

    paired_per_pci = {
        int(pci): int(info.get("count", 0))
        for pci, info in final_evaluation.get("per_pci", {}).items()
    }
    measurement_per_pci: dict[int, dict[str, Any]] = {}
    for pci in station.pcis:
        total = int(measurements["pci"].eq(int(pci)).sum())
        hit = int(paired_per_pci.get(int(pci), 0))
        measurement_per_pci[int(pci)] = {
            "outdoor_measurement_count": total,
            "simulated_hit_count": hit,
            "simulated_no_hit_count": total - hit,
            "measurement_hit_rate": float(hit / total) if total else float("nan"),
        }

    sector_map_stats = {}
    for index, pci in enumerate(station.pcis):
        finite = np.isfinite(maps[index]) & outdoor
        sector_map_stats[int(pci)] = {
            "outdoor_hit_cell_count": int(finite.sum()),
            "outdoor_hit_rate": float(finite.sum() / outdoor_count) if outdoor_count else 0.0,
            "mean_batches_hitting_outdoor_cell": float(
                np.nanmean(hit_maps[index][outdoor])
            ) if outdoor_count else 0.0,
        }

    total_measurements = int(len(measurements))
    paired_measurements = int(final_evaluation["paired_point_count"])
    return {
        "station_id": int(station.station_id),
        "building_cell_count": building_count,
        "outdoor_cell_count": outdoor_count,
        "building_fraction": float(building_count / (building_count + outdoor_count)),
        "best_server_outdoor_hit_cell_count": map_hit_count,
        "best_server_outdoor_no_hit_cell_count": outdoor_count - map_hit_count,
        "best_server_outdoor_hit_rate": float(map_hit_count / outdoor_count)
        if outdoor_count else 0.0,
        "outdoor_measurement_count": total_measurements,
        "measurement_simulated_hit_count": paired_measurements,
        "measurement_simulated_no_hit_count": total_measurements - paired_measurements,
        "measurement_hit_rate": float(paired_measurements / total_measurements)
        if total_measurements else 0.0,
        "per_pci_measurement": measurement_per_pci,
        "per_pci_map": sector_map_stats,
        "final_batch_count": int(final_simulation["batch_count"]),
        "final_samples_per_batch": int(final_simulation["samples_per_batch"]),
        "final_total_samples_per_tx": int(final_simulation["total_samples_per_tx"]),
        "final_batch_seeds": list(final_simulation["batch_seeds"]),
    }


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


def _create_fixed_map_axes(fig, extent: list[float], left: float = 0.100, bottom: float = 0.245, top: float = 0.820, right_margin: float = 0.110, cbar_pad: float = 0.012, cbar_width: float = 0.022):
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


def plot_building_mask(
    output_path: Path,
    station: RuntimeStationConfig,
    grid: DenseGridDefinition,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    fig = plt.figure(figsize=_publication_figsize_inches(), dpi=MAP_DPI)
    ax = fig.add_axes([0.10, 0.10, 0.80, 0.80])
    image = ax.imshow(
        grid.building_mask.astype(np.uint8),
        origin="lower",
        extent=grid.extent,
        cmap=ListedColormap(["white", "0.45"]),
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    ax.scatter([station.x_m], [station.y_m], marker="^", s=70, c="red")
    ax.set_title(
        f"Station {station.station_id}: building-interior mask\n"
        f"gray={int(grid.building_mask.sum()):,} cells, "
        f"outdoor={int((~grid.building_mask).sum()):,} cells"
    )
    ax.set_xlabel("Blender X (m)")
    ax.set_ylabel("Blender Y (m)")
    ax.set_aspect("equal")
    _save_png(fig, output_path)
    plt.close(fig)


def plot_final_maps_general(
    output_dir: Path,
    station: RuntimeStationConfig,
    surface: SurfaceInfo,
    grid: DenseGridDefinition,
    sector_rsrp_dbm: np.ndarray,
    measurements: pd.DataFrame,
    best: Dict[str, Any],
    coverage: Dict[str, Any],
    display_min_dbm: float,
    display_max_dbm: float,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    output_dir.mkdir(parents=True, exist_ok=True)
    maps = sector_values_to_full_maps(surface, sector_rsrp_dbm, grid)
    extent = grid.extent
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("white")
    building_layer = np.ma.masked_where(
        ~grid.building_mask,
        grid.building_mask.astype(float),
    )
    building_cmap = ListedColormap(["0.88"])

    for idx, pci in enumerate(station.pcis):
        fig = plt.figure(figsize=_publication_figsize_inches(), dpi=MAP_DPI)
        ax, cax = _create_fixed_map_axes(fig, extent)
        image = ax.imshow(
            maps[idx],
            origin="lower",
            extent=extent,
            vmin=display_min_dbm,
            vmax=display_max_dbm,
            cmap=cmap,
            interpolation="nearest",
            zorder=1,
        )
        ax.imshow(
            building_layer,
            origin="lower",
            extent=extent,
            cmap=building_cmap,
            vmin=0,
            vmax=1,
            interpolation="nearest",
            zorder=2,
        )
        ax.contour(
            grid.x_m, grid.y_m, grid.building_mask.astype(float),
            levels=[0.5], colors="0.35", linewidths=0.45, zorder=3,
        )
        subset = measurements.loc[measurements["pci"].eq(pci)]
        if not subset.empty:
            ax.scatter(
                subset["cell_x_m"],
                subset["cell_y_m"],
                s=7,
                facecolors="none",
                edgecolors="white",
                linewidths=0.45,
                label=f"Measured PCI {pci}",
                zorder=4,
            )
        ax.scatter(
            [station.x_m],
            [station.y_m],
            marker="^",
            s=70,
            c="red",
            label="TX",
            zorder=5,
        )
        pci_cov = coverage["per_pci_map"][int(pci)]
        ax.set_title(
            f"Station {station.station_id} / PCI {pci}\n"
            f"h={best['height_agl_m']:.1f} m, P={best['shared_power_dbm']:.2f} dBm, "
            f"RMSE={best['final_dense_map_rmse_db']:.2f} dB\n"
            f"DEM+1.5m outdoor map, depth={best['final_max_depth']}, "
            f"batches={best['final_batch_count']}, "
            f"outdoor hit={pci_cov['outdoor_hit_rate']:.1%}"
        )
        ax.set_xlabel("Blender X (m)")
        ax.set_ylabel("Blender Y (m)")
        ax.set_aspect("equal", adjustable="box")
        handles, labels = ax.get_legend_handles_labels()
        handles.append(
            Patch(facecolor="0.88", edgecolor="0.35", label="Building interior (masked)")
        )
        ax.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.115),
            ncol=min(3, len(handles)),
            framealpha=1.0,
            facecolor="white",
            borderaxespad=0.0,
        )
        _add_fixed_colorbar(fig, cax, image, "Aligned simulated received power (dBm)")
        _save_png(fig, output_dir / f"station_{station.station_id}_pci_{pci}_rsrp.png")
        plt.close(fig)

    finite_any = np.any(np.isfinite(maps), axis=0)
    best_map = np.max(np.where(np.isfinite(maps), maps, -np.inf), axis=0)
    best_map = np.where(finite_any, best_map, np.nan)
    fig = plt.figure(figsize=_publication_figsize_inches(), dpi=MAP_DPI)
    ax, cax = _create_fixed_map_axes(fig, extent)
    image = ax.imshow(
        best_map,
        origin="lower",
        extent=extent,
        vmin=display_min_dbm,
        vmax=display_max_dbm,
        cmap=cmap,
        interpolation="nearest",
        zorder=1,
    )
    ax.imshow(
        building_layer,
        origin="lower",
        extent=extent,
        cmap=building_cmap,
        vmin=0,
        vmax=1,
        interpolation="nearest",
        zorder=2,
    )
    ax.contour(
        grid.x_m, grid.y_m, grid.building_mask.astype(float),
        levels=[0.5], colors="0.35", linewidths=0.45, zorder=3,
    )
    ax.scatter([station.x_m], [station.y_m], marker="^", s=70, c="red", zorder=4)
    kind = "omnidirectional" if station.is_omnidirectional else "3-sector best-server"
    ax.set_title(
        f"Station {station.station_id}: {kind} RSRP\n"
        f"true DEM+1.5m outdoor surface, depth={best['final_max_depth']}, "
        f"batches={best['final_batch_count']}, "
        f"outdoor hit={coverage['best_server_outdoor_hit_rate']:.1%}"
    )
    ax.set_xlabel("Blender X (m)")
    ax.set_ylabel("Blender Y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(
        handles=[Patch(facecolor="0.88", edgecolor="0.35", label="Building interior (masked)")],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.115),
        framealpha=1.0,
        facecolor="white",
        borderaxespad=0.0,
    )
    _add_fixed_colorbar(fig, cax, image, "Aligned simulated received power (dBm)")
    _save_png(fig, output_dir / f"station_{station.station_id}_best_server_rsrp.png")
    plt.close(fig)


def _inclusive_grid_exact(start: float, stop: float, step: float) -> np.ndarray:
    if step <= 0:
        raise ValueError(f"step必须>0，实际={step}")
    if stop < start:
        raise ValueError(f"stop必须>=start，实际start={start}, stop={stop}")
    count = int(math.floor((stop - start) / step + 1e-9))
    values = start + np.arange(count + 1, dtype=float) * step
    if values.size == 0 or values[-1] < stop - 1e-8:
        values = np.r_[values, float(stop)]
    else:
        values[-1] = min(values[-1], float(stop))
    values = np.clip(values, start, stop)
    return np.unique(np.round(values, 10))


def _local_grid(
    center: float,
    radius: float,
    step: float,
    global_min: float,
    global_max: float,
) -> np.ndarray:
    lo = max(float(global_min), float(center) - float(radius))
    hi = min(float(global_max), float(center) + float(radius))
    values = _inclusive_grid_exact(lo, hi, step)
    values = np.unique(
        np.round(
            np.r_[values, float(center), float(global_min), float(global_max)],
            10,
        )
    )
    return values[(values >= lo - 1e-9) & (values <= hi + 1e-9)]


def _power_diagnostics(
    evaluation: Dict[str, Any],
    reference_power_dbm: float,
    power_min_dbm: float,
    power_max_dbm: float,
    power_step_db: float,
) -> Dict[str, Any]:
    measured = np.asarray(evaluation["measured_rsrp_dbm"], dtype=float)
    predicted_reference = np.asarray(
        evaluation["predicted_reference_dbm"],
        dtype=float,
    )
    # 对固定几何，最小二乘最优的加性功率偏移有解析解。
    unconstrained_power = float(
        reference_power_dbm
        + np.mean(measured - predicted_reference)
    )
    best_power = float(evaluation["best_power_dbm"])
    half_step = max(abs(float(power_step_db)) * 0.51, 1e-6)
    return {
        "unconstrained_shared_power_dbm": unconstrained_power,
        "power_at_min_boundary": bool(best_power <= power_min_dbm + half_step),
        "power_at_max_boundary": bool(best_power >= power_max_dbm - half_step),
        "power_clipped_low_db": float(max(power_min_dbm - unconstrained_power, 0.0)),
        "power_clipped_high_db": float(max(unconstrained_power - power_max_dbm, 0.0)),
    }




def _active_subcarrier_count(cfg: Dict[str, Any]) -> int:
    radio = cfg["radio"]
    value = int(radio["n_rb"]) * int(radio["subcarriers_per_rb"])
    if value <= 0:
        raise ValueError("n_rb * subcarriers_per_rb必须为正数")
    return value


def carrier_power_to_sss_re_epre_dbm(
    carrier_power_dbm: float, cfg: Dict[str, Any]
) -> float:
    """Map modeled sector-wide carrier power to SSS-RE EPRE.

    The mapping uses uniform power allocation over the configured active
    subcarriers. This is a TX-side allocation assumption, not the SS-RSRP
    receive-side averaging definition. No station-specific fitted offset is used.
    """
    n_active = _active_subcarrier_count(cfg)
    fixed_offset_db = float(cfg["radio"].get("sss_re_epre_offset_db", 0.0))
    return float(carrier_power_dbm) - 10.0 * math.log10(float(n_active)) + fixed_offset_db


def write_ss_rsrp_method_diagnostics(
    *, station: RuntimeStationConfig, cfg: Dict[str, Any], output_dir: Path
) -> Dict[str, Any]:
    """Write the fixed, non-fitted SS-RSRP/EPRE method metadata for one station."""
    n_active = _active_subcarrier_count(cfg)
    diagnostics = {
        "station_id": int(station.station_id),
        "method": "3gpp_ts_38_215_sss_re_uniform_epre_narrowband_global_calibrated",
        "standard_definition": "3GPP TS 38.215 clause 5.1.1: linear average of power contributions of REs carrying SSS",
        "sss_mapping_reference": "3GPP TS 38.211 clauses 7.4.2.3 and 7.4.3.1.2",
        "sss_re_count": 127,
        "ss_pbch_block_width_subcarriers": 240,
        "linear_average_over_sss_re": True,
        "frequency_flat_over_sss_re": True,
        "active_subcarrier_count": int(n_active),
        "tx_power_allocation": "uniform_over_configured_active_subcarriers",
        "tx_power_to_sss_re_epre_formula": "P_SSS_RE_EPRE_dBm = P_carrier_dBm - 10log10(n_rb*subcarriers_per_rb) + network_global_epre_offset_db",
        "network_global_epre_offset_db": float(cfg["radio"].get("sss_re_epre_offset_db", 0.0)),
        "global_epre_offset_scope": "one value shared by all stations; calibration trajectories only; frozen before validation",
        "station_specific_sss_re_offset_fitted": False,
        "division_by_127_or_240_for_ss_rsrp": False,
        "use_of_3276": "TX-side carrier-power allocation only; not an SS-RSRP averaging divisor",
        "exact_frequency_selective_ue_measurement": False,
        "max_depth": 5,
        "diffraction": True,
        "edge_diffraction": True,
        "note": (
            "The only station-level absolute-power parameter fitted by the propagation "
            "calibration is shared_power_dbm within 50-55 dBm. A single network-global EPRE "
            "alignment offset is calibrated from calibration trajectories only and then frozen; "
            "no station-specific SS-RSRP or SSS-RE offset is fitted."
        ),
    }
    (output_dir / "ss_rsrp_method.json").write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return diagnostics


def _geometry_regularization_penalty_db2(candidate: Candidate, search_cfg: Dict[str, Any]) -> float:
    """Softly discourage drift away from the frozen reference geometry.

    The penalty is dimensionless displacement normalized by the hard search radii,
    multiplied by a single dB^2-scale lambda.  It is deliberately secondary to the
    trajectory-balanced data term; the hard windows provide the primary physical guard.
    """
    if not bool(search_cfg.get("geometry_reference_enabled", False)):
        return 0.0
    lam = max(0.0, float(search_cfg.get("geometry_regularization_lambda_db2", 0.0)))
    if lam <= 0.0:
        return 0.0
    rh = max(float(search_cfg.get("reference_height_radius_m", 3.0)), 1e-9)
    ra = max(float(search_cfg.get("reference_azimuth_radius_deg", 15.0)), 1e-9)
    rt = max(float(search_cfg.get("reference_downtilt_radius_deg", 3.0)), 1e-9)
    h0 = float(search_cfg["reference_height_agl_m"])
    a0 = float(search_cfg["reference_azimuth_offset_deg"])
    t0 = float(search_cfg["reference_downtilt_deg"])
    dh = (float(candidate.height_agl_m) - h0) / rh
    da_raw = (float(candidate.azimuth_offset_deg) - a0 + 180.0) % 360.0 - 180.0
    da = da_raw / ra
    dt = (float(candidate.downtilt_delta_deg) - t0) / rt
    return float(lam * (dh * dh + da * da + dt * dt))


def _combined_selection_score_db2(candidate: Candidate, evaluation: Dict[str, Any], search_cfg: Dict[str, Any]) -> float:
    data_score = float(evaluation.get("selection_score_db2", float(evaluation["rmse_db"]) ** 2))
    return float(data_score + _geometry_regularization_penalty_db2(candidate, search_cfg))


def optimize_station_improved(
    scene: Any,
    terrain: TerrainModel,
    station: RuntimeStationConfig,
    measurements: pd.DataFrame,
    sparse_surface: Any,
    cfg: Dict[str, Any],
    output_dir: Path,
    force: bool = False,
) -> Dict[str, Any]:
    """
    改进的两阶段坐标搜索。

    与原始单轮搜索相比：
      - 方位搜索覆盖完整[-180°, +180°]，避免初始方向偏差导致卡在±90°；
      - 三扇区先自动选择顺/逆时针PCI排列；
      - 下倾搜索直接使用绝对角[-15°, +15°]；
      - 先粗搜索，再围绕当前最优做局部细化；
      - 功率严格使用50～55 dBm、1 dB间隔；
      - 同一轮内重复几何参数只仿真一次。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    search_cfg = cfg["search"]
    radio_cfg = cfg["radio"]
    cache_dir = (
        output_dir / "cache_sparse_maps"
        if bool(search_cfg.get("cache_maps", True))
        else None
    )

    height_min = float(search_cfg["height_min_m"])
    height_max = float(search_cfg["height_max_m"])
    height_step = float(search_cfg["height_step_m"])
    az_min = float(search_cfg["azimuth_offset_min_deg"])
    az_max = float(search_cfg["azimuth_offset_max_deg"])
    az_coarse_step = float(search_cfg.get("azimuth_coarse_step_deg", 10.0))
    az_fine_radius = float(search_cfg.get("azimuth_fine_radius_deg", 12.0))
    az_fine_step = float(search_cfg.get("azimuth_fine_step_deg", 2.0))
    tilt_min = float(search_cfg["downtilt_delta_min_deg"])
    tilt_max = float(search_cfg["downtilt_delta_max_deg"])
    tilt_coarse_step = float(search_cfg.get("downtilt_coarse_step_deg", 2.0))
    tilt_fine_radius = float(search_cfg.get("downtilt_fine_radius_deg", 3.0))
    tilt_fine_step = float(search_cfg.get("downtilt_fine_step_deg", 0.5))
    height_fine_radius = float(search_cfg.get("height_fine_radius_m", 2.0))
    height_fine_step = float(search_cfg.get("height_fine_step_m", 0.5))

    power_min = float(search_cfg["power_min_dbm"])
    power_max = float(search_cfg["power_max_dbm"])
    power_step = float(search_cfg["power_step_db"])
    power_grid = _inclusive_grid_exact(power_min, power_max, power_step)

    samples = int(radio_cfg["samples_per_tx_search"])
    reference_power = float(station.initial_power_dbm)

    state = {
        "height_agl_m": float(
            np.clip(
                float(search_cfg.get("reference_height_agl_m", search_cfg.get("initial_height_agl_m", 30.0))),
                height_min,
                height_max,
            )
        ),
        "azimuth_offset_deg": float(search_cfg.get("reference_azimuth_offset_deg", 0.0)),
        "downtilt_delta_deg": float(search_cfg.get("reference_downtilt_deg", 0.0)),
        "shared_power_dbm": float(
            np.clip(reference_power, power_min, power_max)
        ),
        "rmse_db": float("inf"),
    }

    history: list[dict[str, Any]] = []
    best_payload: Dict[str, Any] | None = None
    in_memory_cache: dict[
        tuple[float, float, float],
        Dict[str, Any],
    ] = {}

    def evaluate_geometry(
        height_agl_m: float,
        azimuth_offset_deg: float,
        downtilt_deg: float,
    ) -> Dict[str, Any]:
        key = (
            round(float(height_agl_m), 6),
            round(float(azimuth_offset_deg), 6),
            round(float(downtilt_deg), 6),
        )
        if key in in_memory_cache:
            return in_memory_cache[key]

        candidate = Candidate(
            height_agl_m=float(height_agl_m),
            azimuth_offset_deg=float(azimuth_offset_deg),
            # station.original_downtilt_deg=0，所以此处就是绝对下倾角
            downtilt_delta_deg=float(downtilt_deg),
            reference_power_dbm=reference_power,
        )
        started = time.time()
        simulation = run_candidate(
            scene=scene,
            terrain=terrain,
            station=station,
            candidate=candidate,
            surface=sparse_surface,
            cfg=cfg,
            samples_per_tx=samples,
            cache_dir=cache_dir,
            force=force,
        )
        evaluation = evaluate_prediction(
            station=station,
            measurements=measurements,
            surface=sparse_surface,
            sector_rsrp_at_reference_dbm=simulation["sector_rsrp_dbm"],
            reference_power_dbm=reference_power,
            power_candidates_dbm=power_grid,
            objective_mode=str(cfg["search"].get("objective_mode", "trajectory_balanced_pci_rmse")),
            power_regularization_lambda=float(cfg["search"].get("power_regularization_lambda", 0.0)),
            power_regularization_anchor_dbm=float(cfg["search"].get("power_regularization_anchor_dbm", 52.5)),
            min_trajectory_points=int(cfg["search"].get("reliability_min_trajectory_points", 20)),
            trajectory_weight_cap=int(cfg["search"].get("reliability_trajectory_weight_cap", 250)),
            min_pci_points=int(cfg["search"].get("reliability_min_pci_points", 3)),
            pci_weight_cap=int(cfg["search"].get("reliability_pci_weight_cap", 100)),
        )
        payload = {
            "candidate": candidate,
            "simulation": simulation,
            "evaluation": evaluation,
            "selection_score_db2": _combined_selection_score_db2(candidate, evaluation, search_cfg),
            "geometry_regularization_penalty_db2": _geometry_regularization_penalty_db2(candidate, search_cfg),
            "elapsed_s": time.time() - started,
        }
        in_memory_cache[key] = payload
        return payload

    def run_stage(stage: str, values: Sequence[float]) -> None:
        nonlocal state, best_payload
        payloads: list[Dict[str, Any]] = []

        for candidate_index, value in enumerate(values, start=1):
            params = dict(state)
            if stage.startswith("azimuth"):
                params["azimuth_offset_deg"] = float(value)
            elif stage.startswith("height"):
                params["height_agl_m"] = float(value)
            elif stage.startswith("downtilt"):
                params["downtilt_delta_deg"] = float(value)
            else:
                raise ValueError(stage)

            payload = evaluate_geometry(
                params["height_agl_m"],
                params["azimuth_offset_deg"],
                params["downtilt_delta_deg"],
            )
            candidate = payload["candidate"]
            evaluation = payload["evaluation"]
            simulation = payload["simulation"]

            row = _candidate_row_general(
                station=station,
                candidate=candidate,
                evaluation=evaluation,
                simulation=simulation,
                pass_index=1,
                stage=stage,
                candidate_index=candidate_index,
                elapsed_s=payload["elapsed_s"],
            )
            row.update(
                _power_diagnostics(
                    evaluation=evaluation,
                    reference_power_dbm=reference_power,
                    power_min_dbm=power_min,
                    power_max_dbm=power_max,
                    power_step_db=power_step,
                )
            )
            row["combined_selection_score_db2"] = float(payload["selection_score_db2"])
            row["geometry_regularization_penalty_db2"] = float(payload["geometry_regularization_penalty_db2"])
            history.append(row)
            payloads.append(payload)

            print(
                f"  [{stage} {candidate_index:03d}] "
                f"h={candidate.height_agl_m:.1f}m, "
                f"az={candidate.azimuth_offset_deg:+.1f}°, "
                f"tilt={candidate.downtilt_delta_deg:+.1f}°, "
                f"P={evaluation['best_power_dbm']:.1f}dBm, "
                f"RMSE={evaluation['rmse_db']:.3f}dB"
            )

        selected = min(
            payloads,
            key=lambda item: item["selection_score_db2"],
        )
        candidate = selected["candidate"]
        evaluation = selected["evaluation"]
        state.update(
            {
                "height_agl_m": float(candidate.height_agl_m),
                "azimuth_offset_deg": float(candidate.azimuth_offset_deg),
                "downtilt_delta_deg": float(candidate.downtilt_delta_deg),
                "shared_power_dbm": float(evaluation["best_power_dbm"]),
                "rmse_db": float(evaluation["rmse_db"]),
            }
        )
        best_payload = selected
        print(
            f"  -> {stage}最优: "
            f"h={state['height_agl_m']:.1f}m, "
            f"az={state['azimuth_offset_deg']:+.1f}°, "
            f"tilt={state['downtilt_delta_deg']:+.1f}°, "
            f"P={state['shared_power_dbm']:.1f}dBm, "
            f"RMSE={state['rmse_db']:.3f}dB"
        )

    # 全向站没有方位和下倾意义。
    if station.is_omnidirectional:
        run_stage(
            "height_coarse",
            _inclusive_grid_exact(height_min, height_max, height_step),
        )
        run_stage(
            "height_fine",
            _local_grid(
                state["height_agl_m"],
                height_fine_radius,
                height_fine_step,
                height_min,
                height_max,
            ),
        )
    else:
        run_stage(
            "azimuth_coarse",
            _inclusive_grid_exact(az_min, az_max, az_coarse_step),
        )
        run_stage(
            "height_coarse",
            _inclusive_grid_exact(height_min, height_max, height_step),
        )
        run_stage(
            "downtilt_coarse",
            np.unique(
                np.r_[
                    _inclusive_grid_exact(
                        tilt_min,
                        tilt_max,
                        tilt_coarse_step,
                    ),
                    0.0,
                ]
            ),
        )
        run_stage(
            "azimuth_fine",
            _local_grid(
                state["azimuth_offset_deg"],
                az_fine_radius,
                az_fine_step,
                az_min,
                az_max,
            ),
        )
        run_stage(
            "height_fine",
            _local_grid(
                state["height_agl_m"],
                height_fine_radius,
                height_fine_step,
                height_min,
                height_max,
            ),
        )
        run_stage(
            "downtilt_fine",
            _local_grid(
                state["downtilt_delta_deg"],
                tilt_fine_radius,
                tilt_fine_step,
                tilt_min,
                tilt_max,
            ),
        )

    history_frame = pd.DataFrame(history)
    history_frame.to_csv(
        output_dir / "search_history.csv",
        index=False,
        encoding="utf-8-sig",
    )
    if best_payload is None:
        raise RuntimeError("搜索没有产生候选结果")

    final_evaluation = best_payload["evaluation"]
    final_simulation = best_payload["simulation"]
    power_info = _power_diagnostics(
        evaluation=final_evaluation,
        reference_power_dbm=reference_power,
        power_min_dbm=power_min,
        power_max_dbm=power_max,
        power_step_db=power_step,
    )

    half_height_step = max(height_fine_step * 0.51, 1e-6)
    half_tilt_step = max(tilt_fine_step * 0.51, 1e-6)

    best = {
        "station_id": station.station_id,
        "label": station.label,
        "pcis": list(station.pcis),
        "x_m": station.x_m,
        "y_m": station.y_m,
        "height_agl_m": state["height_agl_m"],
        "ground_z_m": final_simulation["ground_z_m"],
        "tx_absolute_z_m": final_simulation["tx_z_m"],
        "azimuth_offset_deg": state["azimuth_offset_deg"],
        "azimuth_offset_rad": math.radians(state["azimuth_offset_deg"]),
        "alphas_rad": final_simulation["alphas_rad"].tolist(),
        "original_downtilt_deg": 0.0,
        "downtilt_delta_deg": state["downtilt_delta_deg"],
        "absolute_downtilt_deg": state["downtilt_delta_deg"],
        "beta_rad": final_simulation["beta_rad"],
        "gamma_rad": 0.0,
        "shared_power_dbm": state["shared_power_dbm"],
        "pooled_equal_pci_rmse_db": state["rmse_db"],
        "search_mae_db": final_evaluation["mae_db"],
        "search_bias_sim_minus_meas_db": final_evaluation[
            "bias_sim_minus_meas_db"
        ],
        "paired_point_count": final_evaluation["paired_point_count"],
        "per_pci": final_evaluation["per_pci"],
        "search_samples_per_tx": samples,
        "rsrp_definition": (
            "3GPP TS 38.215 SSS-RE linear-average SS-RSRP using the Sionna narrowband "
            "path-gain approximation; 127 SSS REs per TS 38.211; unknown SSS-RE transmit "
            "power represented by a calibration-only station offset; no 127/240/3276 divisor."
        ),
        "selection_metric": str(search_cfg["metric"]),
        "combined_selection_score_db2": float(best_payload.get("selection_score_db2", final_evaluation.get("selection_score_db2", final_evaluation["rmse_db"] ** 2))),
        "geometry_regularization_penalty_db2": float(best_payload.get("geometry_regularization_penalty_db2", 0.0)),
        "reference_geometry_enabled": bool(search_cfg.get("geometry_reference_enabled", False)),
        "reference_height_agl_m": search_cfg.get("reference_height_agl_m"),
        "reference_azimuth_offset_deg": search_cfg.get("reference_azimuth_offset_deg"),
        "reference_downtilt_deg": search_cfg.get("reference_downtilt_deg"),
        "reference_power_dbm": search_cfg.get("reference_power_dbm"),
        "height_at_min_boundary": bool(
            state["height_agl_m"] <= height_min + half_height_step
        ),
        "height_at_max_boundary": bool(
            state["height_agl_m"] >= height_max - half_height_step
        ),
        "downtilt_at_min_boundary": bool(
            (not station.is_omnidirectional)
            and state["downtilt_delta_deg"] <= tilt_min + half_tilt_step
        ),
        "downtilt_at_max_boundary": bool(
            (not station.is_omnidirectional)
            and state["downtilt_delta_deg"] >= tilt_max - half_tilt_step
        ),
        **power_info,
    }

    (output_dir / "best_parameters.json").write_text(
        json.dumps(best, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "best": best,
        "history": history_frame,
        "best_payload": best_payload,
    }


def write_aggregate_result_analysis(
    summary: pd.DataFrame,
    output_root: Path,
) -> None:
    if summary.empty:
        return

    counts = summary["paired_point_count"].to_numpy(dtype=float)
    rmse = summary["final_dense_map_rmse_db"].to_numpy(dtype=float)
    weighted_global_rmse = float(
        np.sqrt(np.sum(counts * rmse ** 2) / np.sum(counts))
    )

    _epre_values = pd.to_numeric(summary.get("sss_re_epre_offset_db", pd.Series(dtype=float)), errors="coerce")
    _epre_values = _epre_values[np.isfinite(_epre_values)]
    _unique_epre = np.unique(np.round(_epre_values.to_numpy(float), 9)) if not _epre_values.empty else np.asarray([], dtype=float)

    result = {
        "station_count": int(len(summary)),
        "network_global_epre_offset_db": (float(_unique_epre[0]) if len(_unique_epre) == 1 else None),
        "network_global_epre_offset_unique": bool(len(_unique_epre) == 1),
        "station_specific_epre_offset_fitted": False,
        "total_paired_point_count": int(summary["paired_point_count"].sum()),
        "mean_station_rmse_db": float(summary["final_dense_map_rmse_db"].mean()),
        "median_station_rmse_db": float(summary["final_dense_map_rmse_db"].median()),
        "p90_station_rmse_db": float(
            summary["final_dense_map_rmse_db"].quantile(0.90)
        ),
        "weighted_global_rmse_db": weighted_global_rmse,
        "minimum_station_rmse_db": float(summary["final_dense_map_rmse_db"].min()),
        "maximum_station_rmse_db": float(summary["final_dense_map_rmse_db"].max()),
        "station_count_rmse_lt_15_db": int(
            (summary["final_dense_map_rmse_db"] < 15.0).sum()
        ),
        "station_count_rmse_lt_18_db": int(
            (summary["final_dense_map_rmse_db"] < 18.0).sum()
        ),
        "station_count_rmse_ge_20_db": int(
            (summary["final_dense_map_rmse_db"] >= 20.0).sum()
        ),
        "power_at_min_boundary_count": int(
            summary["power_at_min_boundary"].astype(bool).sum()
        ),
        "power_at_max_boundary_count": int(
            summary["power_at_max_boundary"].astype(bool).sum()
        ),
        "height_at_min_boundary_count": int(
            summary["height_at_min_boundary"].astype(bool).sum()
        ),
        "height_at_max_boundary_count": int(
            summary["height_at_max_boundary"].astype(bool).sum()
        ),
        "downtilt_at_min_boundary_count": int(
            summary["downtilt_at_min_boundary"].astype(bool).sum()
        ),
        "downtilt_at_max_boundary_count": int(
            summary["downtilt_at_max_boundary"].astype(bool).sum()
        ),
        "worst_five_stations": (
            summary.nlargest(5, "final_dense_map_rmse_db")[
                [
                    "station_id",
                    "label",
                    "final_dense_map_rmse_db",
                    "paired_point_count",
                    "shared_power_dbm",
                    "height_agl_m",
                    "azimuth_offset_deg",
                    "absolute_downtilt_deg",
                ]
            ].to_dict(orient="records")
        ),
    }

    (output_root / "all_27stations_result_analysis.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    flags = summary.loc[
        summary[
            [
                "power_at_min_boundary",
                "power_at_max_boundary",
                "height_at_min_boundary",
                "height_at_max_boundary",
                "downtilt_at_min_boundary",
                "downtilt_at_max_boundary",
            ]
        ].astype(bool).any(axis=1)
        | summary["final_dense_map_rmse_db"].ge(20.0)
        | summary["paired_point_count"].lt(100)
    ].copy()
    flags.to_csv(
        output_root / "all_27stations_quality_flags.csv",
        index=False,
        encoding="utf-8-sig",
    )

def apply_quick_mode(cfg: Dict[str, Any]) -> None:
    # Quick mode is only for verifying paths, units, mesh loading, and output files.
    cfg["radio"]["samples_per_tx_search"] = 200_000
    cfg["radio"]["samples_per_tx_final"] = 500_000
    cfg["search"]["height_step_m"] = 5.0
    cfg["search"]["azimuth_offset_step_deg"] = 30.0
    cfg["search"]["downtilt_delta_step_deg"] = 5.0
    cfg["search"]["passes"] = 1
    cfg["search"]["max_measurement_cells_per_pci"] = 100


def comparison_frame(
    measurements: pd.DataFrame,
    evaluation: Dict[str, Any],
) -> pd.DataFrame:
    kept = evaluation["kept_measurement_indices"]
    result = measurements.loc[kept].copy().reset_index(drop=True)
    result["measured_rsrp_dbm"] = evaluation["measured_rsrp_dbm"]
    result["simulated_rsrp_dbm"] = evaluation["predicted_best_power_dbm"]
    result["error_sim_minus_meas_db"] = evaluation["error_db"]
    return result


def clear_stale_failure_reports(output_root: Path) -> None:
    """Remove failure manifests from previous runs before starting a new batch."""
    for name in ("all_27stations_failures.json", "all_27stations_failures.csv"):
        path = output_root / name
        try:
            path.unlink(missing_ok=True)
        except TypeError:  # Python < 3.8 compatibility fallback
            if path.exists():
                path.unlink()


def load_global_epre_calibration(args: argparse.Namespace, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Load one network-global EPRE offset calibrated without validation data."""
    if args.global_epre_offset_db is not None:
        offset = float(args.global_epre_offset_db)
        if not np.isfinite(offset):
            raise ValueError("--global-epre-offset-db必须为有限数")
        meta = {
            "method": "explicit_cli_override",
            "global_epre_offset_db": offset,
            "source_subset": "user_override",
            "validation_data_used": False,
            "station_specific_offset_fitted": False,
        }
        cfg["radio"]["sss_re_epre_offset_db"] = offset
        return meta

    if args.global_epre_calibration_file:
        path = _resolve_cli_path(args.global_epre_calibration_file)
    else:
        path = PROJECT_ROOT / "config" / "global_epre_calibration.json"
    if not path.exists():
        offset = float(cfg["radio"].get("sss_re_epre_offset_db", 0.0))
        return {
            "method": "config_yaml_fallback",
            "global_epre_offset_db": offset,
            "source_subset": "config",
            "validation_data_used": False,
            "station_specific_offset_fitted": False,
            "warning": f"global EPRE calibration file not found: {path}",
        }
    meta = json.loads(path.read_text(encoding="utf-8"))
    if bool(meta.get("validation_data_used", False)):
        raise RuntimeError("全网EPRE offset校准文件声明使用了validation数据，正式流程拒绝加载")
    if bool(meta.get("station_specific_offset_fitted", False)):
        raise RuntimeError("全网EPRE校准文件不能包含站级独立offset")
    offset = float(meta["global_epre_offset_db"])
    if not np.isfinite(offset):
        raise ValueError("global_epre_offset_db必须为有限数")
    meta = dict(meta)
    meta["loaded_from"] = str(path)
    cfg["radio"]["sss_re_epre_offset_db"] = offset
    if "power_regularization_lambda" in meta:
        cfg["search"]["power_regularization_lambda"] = max(0.0, float(meta["power_regularization_lambda"]))
    if "power_regularization_anchor_dbm" in meta:
        cfg["search"]["power_regularization_anchor_dbm"] = float(meta["power_regularization_anchor_dbm"])
    for key in [
        "reliability_min_trajectory_points",
        "reliability_trajectory_weight_cap",
        "reliability_min_pci_points",
        "reliability_pci_weight_cap",
    ]:
        if key in meta:
            cfg["search"][key] = int(meta[key])
    return meta


def load_reference_geometry(args: argparse.Namespace) -> dict[int, dict[str, float]]:
    if not bool(args.constrain_geometry_to_reference):
        return {}
    path = _resolve_cli_path(args.reference_geometry_file or (PROJECT_ROOT / "config/reference_geometry.csv"))
    if not path.exists():
        raise FileNotFoundError(f"Reference geometry CSV not found: {path}")
    frame = pd.read_csv(path)
    required = {"station_id", "height_agl_m", "azimuth_offset_deg", "absolute_downtilt_deg", "shared_power_dbm"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Reference geometry CSV missing columns: {sorted(missing)}")
    refs: dict[int, dict[str, float]] = {}
    for _, row in frame.iterrows():
        sid = int(row["station_id"])
        refs[sid] = {
            "height_agl_m": float(row["height_agl_m"]),
            "azimuth_offset_deg": float(row["azimuth_offset_deg"]),
            "downtilt_deg": float(row["absolute_downtilt_deg"]),
            "shared_power_dbm": float(row["shared_power_dbm"]),
        }
    return refs


def apply_station_reference_constraints(station_cfg: Dict[str, Any], ref: dict[str, float], args: argparse.Namespace, is_omnidirectional: bool) -> None:
    search = station_cfg["search"]
    hr = max(0.0, float(args.reference_height_radius_m))
    ar = max(0.0, float(args.reference_azimuth_radius_deg))
    tr = max(0.0, float(args.reference_downtilt_radius_deg))
    h0 = float(ref["height_agl_m"]); a0 = float(ref["azimuth_offset_deg"]); t0 = float(ref["downtilt_deg"])
    search["geometry_reference_enabled"] = True
    search["reference_height_agl_m"] = h0
    search["reference_azimuth_offset_deg"] = 0.0 if is_omnidirectional else a0
    search["reference_downtilt_deg"] = 0.0 if is_omnidirectional else t0
    search["reference_power_dbm"] = float(ref["shared_power_dbm"])
    search["reference_height_radius_m"] = hr
    search["reference_azimuth_radius_deg"] = ar
    search["reference_downtilt_radius_deg"] = tr
    search["geometry_regularization_lambda_db2"] = max(0.0, float(args.geometry_regularization_lambda_db2))
    search["height_min_m"] = max(20.0, h0 - hr)
    search["height_max_m"] = min(35.0, h0 + hr)
    if is_omnidirectional:
        search["azimuth_offset_min_deg"] = search["azimuth_offset_max_deg"] = 0.0
        search["downtilt_delta_min_deg"] = search["downtilt_delta_max_deg"] = 0.0
    else:
        search["azimuth_offset_min_deg"] = max(-180.0, a0 - ar)
        search["azimuth_offset_max_deg"] = min(180.0, a0 + ar)
        search["downtilt_delta_min_deg"] = max(-15.0, t0 - tr)
        search["downtilt_delta_max_deg"] = min(15.0, t0 + tr)
    if bool(args.anchor_power_to_reference):
        search["power_regularization_anchor_dbm"] = float(ref["shared_power_dbm"])


def main() -> None:
    args = parse_args()
    cfg = load_yaml(Path(args.config))
    global_epre_calibration = load_global_epre_calibration(args, cfg)
    reference_geometry = load_reference_geometry(args)

    # 本正式版本固定使用27站处理后长表，以及用户指定的共享功率搜索网格。
    cfg["project"]["measurement_csv"] = "../../data/processed/cell_pci_rsrp_1m_calibration.csv"
    cfg["search"]["power_min_dbm"] = 50.0
    cfg["search"]["power_max_dbm"] = 55.0
    cfg["search"]["power_step_db"] = 1.0
    if args.objective_mode is not None:
        cfg["search"]["objective_mode"] = str(args.objective_mode)
    cfg["search"].setdefault("objective_mode", "trajectory_balanced_pci_rmse")
    if args.power_regularization_lambda is not None:
        cfg["search"]["power_regularization_lambda"] = max(0.0, float(args.power_regularization_lambda))
    cfg["search"].setdefault("power_regularization_lambda", 0.0)
    if args.power_regularization_anchor_dbm is not None:
        cfg["search"]["power_regularization_anchor_dbm"] = float(args.power_regularization_anchor_dbm)
    cfg["search"].setdefault("power_regularization_anchor_dbm", 52.5)
    if args.reliability_min_trajectory_points is not None:
        cfg["search"]["reliability_min_trajectory_points"] = max(1, int(args.reliability_min_trajectory_points))
    if args.reliability_trajectory_weight_cap is not None:
        cfg["search"]["reliability_trajectory_weight_cap"] = max(1, int(args.reliability_trajectory_weight_cap))
    if args.reliability_min_pci_points is not None:
        cfg["search"]["reliability_min_pci_points"] = max(1, int(args.reliability_min_pci_points))
    if args.reliability_pci_weight_cap is not None:
        cfg["search"]["reliability_pci_weight_cap"] = max(1, int(args.reliability_pci_weight_cap))
    cfg["search"].setdefault("reliability_min_trajectory_points", 20)
    cfg["search"].setdefault("reliability_trajectory_weight_cap", 250)
    cfg["search"].setdefault("reliability_min_pci_points", 3)
    cfg["search"].setdefault("reliability_pci_weight_cap", 100)


    # One propagation configuration is used throughout calibration and map generation.
    # Search-stage depth overrides are ignored so one propagation configuration is used throughout.
    args.search_max_depth = 5
    args.final_max_depth = 5
    args.no_final_edge_diffraction = False
    if int(args.final_batches) < 1:
        raise ValueError("--final-batches必须>=1")
    if int(args.building_mask_buffer_cells) < 0:
        raise ValueError("--building-mask-buffer-cells必须>=0")
    cfg["radio"]["max_depth"] = 5
    cfg["radio"]["edge_diffraction"] = True
    cfg["radio"]["diffraction"] = True
    cfg["radio"]["diffuse_reflection"] = True

    # 改进搜索：全方位粗搜 + 局部细化；下倾角直接表示绝对角。
    cfg["search"]["azimuth_offset_min_deg"] = -180.0
    cfg["search"]["azimuth_offset_max_deg"] = 180.0
    cfg["search"]["azimuth_coarse_step_deg"] = 10.0
    cfg["search"]["azimuth_fine_radius_deg"] = 12.0
    cfg["search"]["azimuth_fine_step_deg"] = 2.0

    cfg["search"]["height_min_m"] = 20.0
    cfg["search"]["height_max_m"] = 35.0
    cfg["search"]["height_step_m"] = 1.0
    cfg["search"]["height_fine_radius_m"] = 2.0
    cfg["search"]["height_fine_step_m"] = 0.5

    cfg["search"]["downtilt_delta_min_deg"] = -15.0
    cfg["search"]["downtilt_delta_max_deg"] = 15.0
    cfg["search"]["downtilt_coarse_step_deg"] = 2.0
    cfg["search"]["downtilt_fine_radius_deg"] = 3.0
    cfg["search"]["downtilt_fine_step_deg"] = 0.5

    # 当前正式数据每个PCI在512m范围内的1m聚合点不超过约1250；
    # 设为2000即可使用全部聚合点，不再按原配置截断到600点。
    cfg["search"]["max_measurement_cells_per_pci"] = 2000

    if args.quick:
        apply_quick_mode(cfg)
        # quick模式也不得改变用户指定的功率网格。
        cfg["search"]["power_min_dbm"] = 50.0
        cfg["search"]["power_max_dbm"] = 55.0
        cfg["search"]["power_step_db"] = 1.0

    measurement_csv = (
        Path(args.measurements).expanduser().resolve()
        if args.measurements
        else resolve_path(cfg, cfg["project"]["measurement_csv"])
    )
    output_root = (_resolve_cli_path(args.output_root) if args.output_root else resolve_path(cfg, cfg["project"]["output_dir"]))
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "global_epre_calibration_used.json").write_text(
        json.dumps(global_epre_calibration, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    clear_stale_failure_reports(output_root)
    work_root = ROOT / "work"
    work_root.mkdir(parents=True, exist_ok=True)

    configured_ground = resolve_path(cfg, cfg["scene"]["ground_ply"])
    ground_ply = _resolve_cli_path(args.ground) if args.ground else configured_ground
    if not ground_ply.exists() and not args.ground:
        fallback = configured_ground.parent / "ground(1).ply"
        if fallback.exists():
            ground_ply = fallback.resolve()
            print(f"config中的ground.ply不存在，自动改用: {ground_ply}")

    building_candidates = _mesh_candidates(
        cfg=cfg,
        explicit_buildings=args.buildings,
        ground_ply=ground_ply,
    )
    scene_xml = resolve_path(cfg, cfg["scene"]["generated_xml"])
    scene_report = build_scene_xml_multi(
        ground_ply=ground_ply,
        building_candidates=building_candidates,
        output_xml=scene_xml,
        cleaned_dir=work_root / "scene_mesh_cache",
        ground_material=str(cfg["scene"]["ground_material"]),
        building_material=str(cfg["scene"]["building_material"]),
        allow_no_buildings=bool(args.allow_no_buildings),
    )
    print(json.dumps(scene_report, ensure_ascii=False, indent=2))
    print(f"有效建筑PLY数量: {scene_report['building_included_count']}")
    for item in scene_report["buildings"]:
        if item.get("included", False):
            print(
                "  已强制加入（原样、未做坐标或三角面过滤）:",
                item.get("source_path"), "->", item.get("effective_path")
            )
        else:
            print("  未加入:", item.get("source_path"), "原因:", item.get("exclusion_reason"))

    cfg["_resolved_scene_xml"] = str(scene_xml)

    building_triangles_xy, building_projection_diagnostics = (
        load_building_projection_triangles(
            [Path(value) for value in scene_report["building_included_paths"]]
        )
    )
    (output_root / "building_projection_diagnostics.json").write_text(
        json.dumps(
            {
                "total_nondegenerate_xy_triangles": int(len(building_triangles_xy)),
                "files": building_projection_diagnostics,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"用于建筑内部掩膜的非退化XY三角面: "
        f"{len(building_triangles_xy):,}"
    )

    raw_long, observations = read_27station_long_measurements(measurement_csv)
    if args.disable_independent_validation_split:
        split = {
            "enabled": False,
            "reason": "disabled by --disable-independent-validation-split",
            "calibration_source_files": sorted(raw_long.get("source_file", pd.Series(dtype=str)).dropna().astype(str).unique().tolist()),
            "validation_source_files": [],
        }
        calibration_long = raw_long.copy()
        calibration_observations = observations.copy()
    else:
        split = build_independent_source_split(
            raw_long,
            validation_fraction=float(args.validation_fraction),
            seed=int(args.validation_seed),
            min_calibration_rows_per_pci=int(args.min_calibration_rows_per_pci),
        )
        if not split.get("enabled"):
            raise RuntimeError(
                "无法构建独立calibration/validation轨迹划分: " + str(split.get("reason")) +
                "。正式复现必须保留source_file元数据；若仅需兼容旧数据可显式使用--disable-independent-validation-split。"
            )
        cal_files=set(split["calibration_source_files"])
        calibration_long=raw_long.loc[raw_long["source_file"].astype(str).isin(cal_files)].copy()
        calibration_observations=observations.loc[observations["source_file"].astype(str).isin(cal_files)].copy()
    (output_root / "calibration_validation_split.json").write_text(json.dumps(split,ensure_ascii=False,indent=2),encoding="utf-8")
    if "source_file" in raw_long.columns:
        manifest = pd.DataFrame({"source_file": sorted(raw_long["source_file"].astype(str).unique())})
        val=set(split.get("validation_source_files",[]))
        manifest["subset"] = manifest["source_file"].map(lambda x: "validation" if x in val else "calibration")
        manifest.to_csv(output_root / "calibration_validation_split.csv", index=False, encoding="utf-8-sig")
    stations, direction_diagnostics = build_all_27_station_configs(
        raw_long=calibration_long,
        top_fraction=float(args.direction_top_fraction),
        initial_power_dbm=53.5,
    )
    observations = calibration_observations
    selected_ids = parse_selected_station_ids(args.stations, list(stations.keys()))

    pd.DataFrame(direction_diagnostics).to_csv(
        output_root / "estimated_initial_directions_27stations.csv",
        index=False,
        encoding="utf-8-sig",
    )

    direction_by_station = {
        int(item["station_id"]): item
        for item in direction_diagnostics
    }

    print("\n将运行的物理基站数量:", len(selected_ids))
    print("station_id:", selected_ids)
    power_grid = np.arange(50.0, 55.0 + 0.5, 1.0)
    power_grid[-1] = 55.0
    print("三扇区共享功率候选(dBm):", [round(float(v), 1) for v in power_grid])
    print("说明：每个三扇区站的3个PCI始终使用同一个候选发射功率。")
    print("方位：先搜索-180°～+180°，再局部细化。")
    print("下倾：直接搜索绝对下倾角-15°～+15°，不再统一加12°。")
    print("实测：每PCI最多2000个1m聚合点，本数据可基本使用全部点。")
    print(
        "全网统一EPRE offset: "
        f"{float(cfg['radio'].get('sss_re_epre_offset_db', 0.0)):+.6f} dB；"
        "仅由calibration轨迹估计并在held-out validation前冻结。"
    )

    print("传播：全流程统一 max_depth=5, diffraction=True, edge_diffraction=True。")
    print(
        f"最终1m地图：{int(args.final_batches)}个独立seed批次，"
        "在线性功率域平均；建筑内部从DEM+1.5m接收面中移除。"
    )
    print(
        "独立验证划分：" + (
            f"calibration={len(split.get('calibration_source_files', []))}条轨迹, validation={len(split.get('validation_source_files', []))}条轨迹"
            if split.get("enabled") else f"未启用（{split.get('reason','unknown')}）"
        )
    )
    print(
        "局部参数细化：" + ("关闭" if args.no_final_refinement else "开启（传播物理设置与搜索/制图完全相同）")
    )

    terrain = TerrainModel.load(ground_ply)
    scene = configure_scene(scene_xml, cfg)
    install_general_sector_support()

    map_cfg = cfg["map"]
    search_cfg = cfg["search"]
    radio_cfg = cfg["radio"]
    map_x = int(map_cfg["size_x_m"])
    map_y = int(map_cfg["size_y_m"])
    cell = float(map_cfg["cell_size_m"])
    rx_height = float(radio_cfg["rx_height_agl_m"])
    summaries: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for station_id in selected_ids:
        station = stations[station_id]
        station_started = time.time()
        try:
            station.validate(tolerance_deg=1.0)
            terrain.assert_map_inside(station.x_m, station.y_m, map_x, map_y)

            station_cfg = copy.deepcopy(cfg)
            if args.constrain_geometry_to_reference:
                if station_id not in reference_geometry:
                    raise KeyError(f"Reference geometry missing station {station_id}")
                apply_station_reference_constraints(station_cfg, reference_geometry[station_id], args, station.is_omnidirectional)
            if station.is_omnidirectional:
                station_cfg["antenna"] = {
                    "num_rows": 1,
                    "num_cols": 1,
                    "vertical_spacing": 0.5,
                    "horizontal_spacing": 0.5,
                    "pattern": "iso",
                    "polarization": "V",
                }
                # 全向站没有方位和下倾意义，只搜索高度与共享功率。
                station_cfg["search"]["azimuth_offset_min_deg"] = 0.0
                station_cfg["search"]["azimuth_offset_max_deg"] = 0.0
                station_cfg["search"]["azimuth_offset_step_deg"] = 5.0
                station_cfg["search"]["downtilt_delta_min_deg"] = 0.0
                station_cfg["search"]["downtilt_delta_max_deg"] = 0.0
                station_cfg["search"]["downtilt_delta_step_deg"] = 1.0
            configure_tx_array_for_station(scene, station_cfg, station)

            station_dir = output_root / f"station_{station_id}"
            station_dir.mkdir(parents=True, exist_ok=True)
            work_dir = work_root / f"station_{station_id}"
            work_dir.mkdir(parents=True, exist_ok=True)

            grid, building_mask_diagnostics = create_dense_grid_with_building_mask(
                terrain=terrain,
                building_triangles_xy=building_triangles_xy,
                center_x=station.x_m,
                center_y=station.y_m,
                size_x_m=map_x,
                size_y_m=map_y,
                cell_size_m=cell,
                rx_height_agl_m=rx_height,
                buffer_cells=int(args.building_mask_buffer_cells),
            )
            np.save(station_dir / "building_mask.npy", grid.building_mask)
            (station_dir / "building_mask_diagnostics.json").write_text(
                json.dumps(building_mask_diagnostics, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            plot_building_mask(
                station_dir / f"station_{station_id}_building_mask.png",
                station,
                grid,
            )

            print("\n" + "=" * 96)
            print(
                f"开始 {station_id}号站 {station.label}, PCI={station.pcis}, "
                f"类型={'全向' if station.is_omnidirectional else '三扇区'}"
            )
            print(f"地图: {map_x}m×{map_y}m, RX=DEM+{rx_height}m")
            print(f"初始alpha(rad): {station.initial_alphas_rad}")
            print("共享功率搜索: 50.0～55.0 dBm，步长1.0 dB")
            if args.constrain_geometry_to_reference:
                ref = reference_geometry[station_id]
                print(f"参考几何约束: h={ref['height_agl_m']:.1f}±{args.reference_height_radius_m:.1f}m, "
                      f"az={ref['azimuth_offset_deg']:+.1f}±{args.reference_azimuth_radius_deg:.1f}°, "
                      f"tilt={ref['downtilt_deg']:+.1f}±{args.reference_downtilt_radius_deg:.1f}°, "
                      f"P_anchor={ref['shared_power_dbm']:.1f}dBm")

            measured = prepare_station_measurements(
                observations=observations,
                station=station,
                map_size_x_m=map_x,
                map_size_y_m=map_y,
                cell_size_m=cell,
                min_points_per_pci=int(search_cfg["min_points_per_pci"]),
                max_cells_per_pci=int(search_cfg["max_measurement_cells_per_pci"]),
                strong_signal_sampling_fraction=float(search_cfg["strong_signal_sampling_fraction"]),
            )
            measured_before_mask = measured.copy()
            measured, measurements_inside_buildings = remove_measurements_inside_buildings(
                measurements=measured_before_mask,
                building_mask=grid.building_mask,
            )
            measurements_inside_buildings.to_csv(
                station_dir / "measurement_cells_excluded_inside_buildings.csv",
                index=False,
                encoding="utf-8-sig",
            )
            print(
                f"建筑掩膜: building={int(grid.building_mask.sum()):,}格, "
                f"outdoor={int((~grid.building_mask).sum()):,}格；"
                f"实测聚合点移除建筑内部={len(measurements_inside_buildings):,}，"
                f"保留室外={len(measured):,}。"
            )
            if len(measured) < int(search_cfg["min_points_total"]):
                raise ValueError(
                    f"{station_id}号站建筑掩膜后有效室外1m实测点总数{len(measured)}，"
                    f"小于{search_cfg['min_points_total']}"
                )
            measured.to_csv(
                station_dir / "measurement_cells_used.csv",
                index=False,
                encoding="utf-8-sig",
            )

            x_min = station.x_m - map_x / 2.0
            y_min = station.y_m - map_y / 2.0
            sparse_surface = build_sparse_measurement_surface(
                terrain=terrain,
                cells=zip(measured["ix"], measured["iy"]),
                x_min=x_min,
                y_min=y_min,
                cell_size_m=float(search_cfg["sparse_surface_cell_size_m"]),
                rx_height_agl_m=rx_height,
                output_path=work_dir / "measurement_surface_sparse_dem_plus_1p5m.ply",
            )

            ss_rsrp_method = write_ss_rsrp_method_diagnostics(
                station=station, cfg=station_cfg, output_dir=station_dir
            )
            print(
                "SS-RSRP方法: 127个SSS RE在线性功率域平均；"
                f"载波功率按N_active={ss_rsrp_method['active_subcarrier_count']}个活动子载波均匀分配到每RE；"
                f"全网统一EPRE offset={float(station_cfg['radio'].get('sss_re_epre_offset_db',0.0)):+.6f} dB；"
                "不拟合任何站级SSS-RE/SS-RSRP额外偏置。"
            )

            optimization = optimize_station_improved(
                scene=scene,
                terrain=terrain,
                station=station,
                measurements=measured,
                sparse_surface=sparse_surface,
                cfg=station_cfg,
                output_dir=station_dir,
                force=args.force,
            )
            best = optimization["best"]
            best["ss_rsrp_method"] = "3gpp_ts_38_215_sss_re_uniform_epre_narrowband_global_calibrated"
            best["sss_re_count"] = 127
            best["active_subcarrier_count"] = _active_subcarrier_count(station_cfg)
            best["sss_re_epre_offset_db"] = float(station_cfg["radio"].get("sss_re_epre_offset_db", 0.0))
            best["global_epre_offset_calibrated"] = True
            best["global_epre_offset_source_subset"] = "calibration_trajectories_only"
            best["sss_re_tx_epre_dbm"] = carrier_power_to_sss_re_epre_dbm(
                float(best.get("shared_power_dbm", station.initial_power_dbm)), station_cfg
            )
            best["station_specific_sss_re_offset_fitted"] = False
            best["coarse_search_height_agl_m"] = float(best["height_agl_m"])
            best["coarse_search_azimuth_offset_deg"] = float(best["azimuth_offset_deg"])
            best["coarse_search_downtilt_deg"] = float(best["downtilt_delta_deg"])
            best["coarse_search_power_dbm"] = float(best["shared_power_dbm"])
            best["coarse_search_rmse_db"] = float(best["pooled_equal_pci_rmse_db"])
            if not args.no_final_refinement:
                refinement = refine_candidate_with_final_propagation_settings(
                    scene=scene, terrain=terrain, station=station, measurements=measured,
                    sparse_surface=sparse_surface, cfg=station_cfg, coarse_best=best,
                    output_dir=station_dir, force=args.force,
                    final_max_depth=int(args.final_max_depth),
                    final_edge_diffraction=bool(not args.no_final_edge_diffraction),
                    samples_per_tx=(int(args.final_samples_per_batch) if args.final_samples_per_batch is not None else int(radio_cfg["samples_per_tx_final"])),
                    height_radius_m=float(args.refine_height_radius_m), height_step_m=float(args.refine_height_step_m),
                    azimuth_radius_deg=float(args.refine_azimuth_radius_deg), azimuth_step_deg=float(args.refine_azimuth_step_deg),
                    downtilt_radius_deg=float(args.refine_downtilt_radius_deg), downtilt_step_deg=float(args.refine_downtilt_step_deg),
                )
                best.update(refinement)
                best["pooled_equal_pci_rmse_db"] = float(refinement["refinement_rmse_db"])
                best["search_mae_db"] = float(refinement["refinement_mae_db"])
                best["search_bias_sim_minus_meas_db"] = float(refinement["refinement_bias_db"])
                best["final_refinement_applied"] = True
            else:
                best["final_refinement_applied"] = False

            dense_surface = build_dense_outdoor_measurement_surface(
                terrain=terrain,
                grid=grid,
                rx_height_agl_m=rx_height,
                output_path=(
                    work_dir
                    / "measurement_surface_dense_512m_dem_plus_1p5m_outdoor_only.ply"
                ),
            )
            print(
                f"最终接收面: DEM+{rx_height}m真实不平坦曲面，"
                f"仅室外单元={dense_surface.n_cells:,}，"
                f"三角面={dense_surface.n_faces:,}。"
            )
            final_candidate = Candidate(
                height_agl_m=float(best["height_agl_m"]),
                azimuth_offset_deg=float(best["azimuth_offset_deg"]),
                downtilt_delta_deg=float(best["downtilt_delta_deg"]),
                reference_power_dbm=float(best["shared_power_dbm"]),
            )
            final_cfg = copy.deepcopy(station_cfg)
            final_cfg["radio"]["max_depth"] = 5
            final_cfg["radio"]["edge_diffraction"] = True
            final_cfg["radio"]["diffraction"] = True
            final_cfg["radio"]["diffuse_reflection"] = True
            final_samples_per_batch = (
                int(args.final_samples_per_batch)
                if args.final_samples_per_batch is not None
                else int(radio_cfg["samples_per_tx_final"])
            )
            final_sim = run_candidate_multibatch_linear_average(
                scene=scene,
                terrain=terrain,
                station=station,
                candidate=final_candidate,
                surface=dense_surface,
                cfg=final_cfg,
                samples_per_batch=final_samples_per_batch,
                batch_count=int(args.final_batches),
                seed_step=int(args.final_seed_step),
                cache_dir=station_dir / "cache_final_map_multibatch",
                force=args.force,
            )

            final_eval = evaluate_prediction(
                station=station,
                measurements=measured,
                surface=dense_surface,
                sector_rsrp_at_reference_dbm=final_sim["sector_rsrp_dbm"],
                reference_power_dbm=float(best["shared_power_dbm"]),
                power_candidates_dbm=np.asarray([best["shared_power_dbm"]], dtype=float),
                objective_mode="pooled_rmse",
                power_regularization_lambda=0.0,
                power_regularization_anchor_dbm=float(best["shared_power_dbm"]),
            )
            coverage = compute_radio_map_coverage_diagnostics(
                station=station,
                surface=dense_surface,
                grid=grid,
                sector_rsrp_dbm=final_sim["sector_rsrp_dbm"],
                batch_hit_counts=final_sim["batch_hit_counts"],
                measurements=measured,
                final_evaluation=final_eval,
                final_simulation=final_sim,
            )
            (station_dir / "radio_map_coverage_diagnostics.json").write_text(
                json.dumps(coverage, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            comparison = comparison_frame(measured, final_eval)
            comparison["station_id"] = int(station_id)
            comparison["selected_shared_power_dbm"] = float(best["shared_power_dbm"])
            comparison["network_global_epre_offset_db"] = float(station_cfg["radio"].get("sss_re_epre_offset_db", 0.0))
            comparison.to_csv(
                station_dir / "final_measured_vs_simulated.csv",
                index=False,
                encoding="utf-8-sig",
            )
            traj_rows = []
            if "source_file" in comparison.columns:
                for sf, grp in comparison.groupby("source_file", sort=True):
                    err = grp["error_sim_minus_meas_db"].to_numpy(dtype=float)
                    diag = final_eval.get("per_trajectory", {}).get(str(sf), {})
                    traj_rows.append({
                        "station_id": int(station_id), "source_file": str(sf),
                        "matched_count": int(len(grp)),
                        "mse_db2": float(np.mean(err ** 2)),
                        "rmse_db": float(np.sqrt(np.mean(err ** 2))),
                        "mae_db": float(np.mean(np.abs(err))),
                        "bias_sim_minus_measured_db": float(np.mean(err)),
                        "eligible_for_selection": bool(diag.get("eligible_for_selection", int(len(grp)) >= int(station_cfg["search"].get("reliability_min_trajectory_points", 20)))),
                        "selection_weight": float(diag.get("selection_weight", 0.0)),
                        "reliability_rmse_db": float(diag.get("reliability_rmse_db", np.sqrt(np.mean(err ** 2)))),
                        "selected_shared_power_dbm": float(best["shared_power_dbm"]),
                        "network_global_epre_offset_db": float(station_cfg["radio"].get("sss_re_epre_offset_db", 0.0)),
                    })
            pd.DataFrame(traj_rows).to_csv(station_dir / "calibration_trajectory_metrics.csv", index=False, encoding="utf-8-sig")
            plot_comparison(
                station_dir / "final_measured_vs_simulated.png",
                comparison,
                station_id,
            )

            best["final_dense_map_rmse_db"] = final_eval["rmse_db"]
            best["final_dense_map_mae_db"] = final_eval["mae_db"]
            best["final_dense_map_bias_db"] = final_eval["bias_sim_minus_meas_db"]
            best["final_dense_trajectory_balanced_rmse_db"] = float(final_eval.get("trajectory_balanced_rmse_db", final_eval["rmse_db"]))
            best["final_dense_reliability_balanced_rmse_db"] = float(final_eval.get("reliability_balanced_rmse_db", final_eval["rmse_db"]))
            best["eligible_trajectory_count"] = int(final_eval.get("eligible_trajectory_count", 0))
            best["excluded_low_support_trajectory_count"] = int(final_eval.get("excluded_low_support_trajectory_count", 0))
            best["reliability_fallback_used"] = bool(final_eval.get("reliability_fallback_used", False))
            best["calibration_objective_mode"] = str(station_cfg["search"].get("objective_mode", "reliability_balanced_pci_rmse"))
            best["power_regularization_lambda"] = float(station_cfg["search"].get("power_regularization_lambda", 0.0))
            best["power_regularization_anchor_dbm"] = float(station_cfg["search"].get("power_regularization_anchor_dbm", 52.5))
            if bool(station_cfg["search"].get("geometry_reference_enabled", False)):
                best["reference_height_delta_m"] = float(best["height_agl_m"]) - float(station_cfg["search"]["reference_height_agl_m"])
                best["reference_azimuth_delta_deg"] = (float(best["azimuth_offset_deg"]) - float(station_cfg["search"]["reference_azimuth_offset_deg"]) + 180.0) % 360.0 - 180.0
                best["reference_downtilt_delta_deg"] = float(best["absolute_downtilt_deg"]) - float(station_cfg["search"]["reference_downtilt_deg"])
                best["reference_power_delta_db"] = float(best["shared_power_dbm"]) - float(station_cfg["search"]["reference_power_dbm"])
            best["paired_point_count"] = int(final_eval["paired_point_count"])
            best["per_pci"] = final_eval["per_pci"]
            best["final_samples_per_tx"] = int(
                final_sim["total_samples_per_tx"]
            )
            best["final_batch_count"] = int(final_sim["batch_count"])
            best["final_samples_per_batch"] = int(
                final_sim["samples_per_batch"]
            )
            best["final_total_samples_per_tx"] = int(
                final_sim["total_samples_per_tx"]
            )
            best["final_batch_seeds"] = list(final_sim["batch_seeds"])
            best["search_max_depth"] = int(args.search_max_depth)
            best["final_max_depth"] = int(args.final_max_depth)
            best["final_edge_diffraction"] = bool(
                not args.no_final_edge_diffraction
            )
            best["building_mask_buffer_cells"] = int(
                args.building_mask_buffer_cells
            )
            best["building_cell_count"] = int(coverage["building_cell_count"])
            best["outdoor_cell_count"] = int(coverage["outdoor_cell_count"])
            best["best_server_outdoor_hit_rate"] = float(
                coverage["best_server_outdoor_hit_rate"]
            )
            best["measurement_hit_rate"] = float(
                coverage["measurement_hit_rate"]
            )
            best["measurement_cells_excluded_inside_buildings"] = int(
                len(measurements_inside_buildings)
            )
            best["receiver_surface_z_min_m"] = float(
                building_mask_diagnostics["receiver_surface_z_min_m"]
            )
            best["receiver_surface_z_max_m"] = float(
                building_mask_diagnostics["receiver_surface_z_max_m"]
            )
            best["receiver_surface_relief_m"] = float(
                building_mask_diagnostics["receiver_surface_relief_m"]
            )
            best["is_omnidirectional"] = bool(station.is_omnidirectional)
            best["power_search_min_dbm"] = 50.0
            best["power_search_max_dbm"] = 55.0
            best["power_search_step_db"] = 1.0
            best["active_subcarrier_count"] = _active_subcarrier_count(final_cfg)
            best["sss_re_epre_offset_db"] = float(final_cfg["radio"].get("sss_re_epre_offset_db", 0.0))
            best["global_epre_offset_calibrated"] = True
            best["global_epre_offset_source_subset"] = "calibration_trajectories_only"
            best["sss_re_tx_epre_dbm"] = carrier_power_to_sss_re_epre_dbm(
                float(best["shared_power_dbm"]), final_cfg
            )
            best["station_specific_sss_re_offset_fitted"] = False
            (station_dir / "best_parameters.json").write_text(
                json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8"
            )

            save_final_npz_general(
                path=station_dir / "final_radio_map_512m.npz",
                station=station,
                surface=dense_surface,
                grid=grid,
                sector_rsrp_dbm=final_sim["sector_rsrp_dbm"],
                batch_hit_counts=final_sim["batch_hit_counts"],
                best=best,
            )
            plot_final_maps_general(
                output_dir=station_dir,
                station=station,
                surface=dense_surface,
                grid=grid,
                sector_rsrp_dbm=final_sim["sector_rsrp_dbm"],
                measurements=measured,
                best=best,
                coverage=coverage,
                display_min_dbm=float(map_cfg["display_min_dbm"]),
                display_max_dbm=float(map_cfg["display_max_dbm"]),
            )

            summaries.append(
                {
                    "station_id": station_id,
                    "label": station.label,
                    "antenna_type": "omnidirectional" if station.is_omnidirectional else "three_sector",
                    "pcis": ";".join(map(str, station.pcis)),
                    "selected_sector_order": direction_by_station[station_id].get(
                        "selected_sector_order"
                    ),
                    "direction_fit_rms_deg": direction_by_station[station_id].get(
                        "direction_fit_rms_deg"
                    ),
                    "x_m": station.x_m,
                    "y_m": station.y_m,
                    "ground_z_m": best["ground_z_m"],
                    "height_agl_m": best["height_agl_m"],
                    "tx_absolute_z_m": best["tx_absolute_z_m"],
                    "azimuth_offset_deg": best["azimuth_offset_deg"],
                    "azimuth_offset_rad": best["azimuth_offset_rad"],
                    "absolute_downtilt_deg": best["absolute_downtilt_deg"],
                    "beta_rad": best["beta_rad"],
                    "shared_power_dbm": best["shared_power_dbm"],
                    "unconstrained_shared_power_dbm": best.get(
                        "unconstrained_shared_power_dbm"
                    ),
                    "coarse_search_rmse_db": best.get("coarse_search_rmse_db"),
                    "final_refinement_applied": best.get("final_refinement_applied", False),
                    "final_refinement_rmse_db": best.get("refinement_rmse_db"),
                    "search_rmse_db": best["pooled_equal_pci_rmse_db"],
                    "search_mae_db": best.get("search_mae_db"),
                    "search_bias_sim_minus_meas_db": best.get(
                        "search_bias_sim_minus_meas_db"
                    ),
                    "final_dense_map_rmse_db": best["final_dense_map_rmse_db"],
                    "final_dense_map_mae_db": best["final_dense_map_mae_db"],
                    "final_dense_map_bias_db": best["final_dense_map_bias_db"],
                    "final_dense_trajectory_balanced_rmse_db": best.get("final_dense_trajectory_balanced_rmse_db"),
                    "calibration_objective_mode": best.get("calibration_objective_mode"),
                    "power_regularization_lambda": best.get("power_regularization_lambda"),
                    "power_regularization_anchor_dbm": best.get("power_regularization_anchor_dbm"),
                    "reference_geometry_enabled": best.get("reference_geometry_enabled", False),
                    "reference_height_agl_m": best.get("reference_height_agl_m"),
                    "reference_azimuth_offset_deg": best.get("reference_azimuth_offset_deg"),
                    "reference_downtilt_deg": best.get("reference_downtilt_deg"),
                    "reference_power_dbm": best.get("reference_power_dbm"),
                    "reference_height_delta_m": best.get("reference_height_delta_m"),
                    "reference_azimuth_delta_deg": best.get("reference_azimuth_delta_deg"),
                    "reference_downtilt_delta_deg": best.get("reference_downtilt_delta_deg"),
                    "reference_power_delta_db": best.get("reference_power_delta_db"),
                    "geometry_regularization_penalty_db2": best.get("geometry_regularization_penalty_db2"),
                    "combined_selection_score_db2": best.get("combined_selection_score_db2"),
                    "paired_point_count": best["paired_point_count"],
                    "measurement_cells_excluded_inside_buildings": int(
                        len(measurements_inside_buildings)
                    ),
                    "building_cell_count": coverage["building_cell_count"],
                    "outdoor_cell_count": coverage["outdoor_cell_count"],
                    "best_server_outdoor_hit_rate": coverage[
                        "best_server_outdoor_hit_rate"
                    ],
                    "measurement_hit_rate": coverage["measurement_hit_rate"],
                    "final_batch_count": final_sim["batch_count"],
                    "final_samples_per_batch": final_sim["samples_per_batch"],
                    "final_total_samples_per_tx": final_sim[
                        "total_samples_per_tx"
                    ],
                    "max_depth": 5,
                    "diffraction": True,
                    "edge_diffraction": True,
                    "ss_rsrp_method": "3gpp_ts_38_215_sss_re_uniform_epre_narrowband_global_calibrated",
                    "sss_re_count": 127,
                    "active_subcarrier_count": best.get("active_subcarrier_count", 3276),
                    "sss_re_epre_offset_db": best.get("sss_re_epre_offset_db", 0.0),
                    "global_epre_offset_calibrated": True,
                    "global_epre_offset_source_subset": "calibration_trajectories_only",
                    "sss_re_tx_epre_dbm": best.get("sss_re_tx_epre_dbm"),
                    "station_specific_sss_re_offset_fitted": False,
                    "tx_power_allocation_mode": "uniform_over_configured_active_subcarriers",
                    "power_at_min_boundary": best.get(
                        "power_at_min_boundary", False
                    ),
                    "power_at_max_boundary": best.get(
                        "power_at_max_boundary", False
                    ),
                    "height_at_min_boundary": best.get(
                        "height_at_min_boundary", False
                    ),
                    "height_at_max_boundary": best.get(
                        "height_at_max_boundary", False
                    ),
                    "downtilt_at_min_boundary": best.get(
                        "downtilt_at_min_boundary", False
                    ),
                    "downtilt_at_max_boundary": best.get(
                        "downtilt_at_max_boundary", False
                    ),
                    "elapsed_s": time.time() - station_started,
                }
            )
            pd.DataFrame(summaries).to_csv(
                output_root / "all_27stations_summary.csv",
                index=False,
                encoding="utf-8-sig",
            )
            print(
                f"完成 {station_id}号站: h={best['height_agl_m']:.0f}m, "
                f"P={best['shared_power_dbm']:.2f}dBm, "
                f"az_off={best['azimuth_offset_deg']:+.0f}°, "
                f"downtilt={best['absolute_downtilt_deg']:.0f}°, "
                f"dense RMSE={best['final_dense_map_rmse_db']:.3f}dB, "
                f"outdoor hit={coverage['best_server_outdoor_hit_rate']:.1%}, "
                f"measurement hit={coverage['measurement_hit_rate']:.1%}"
            )

        except Exception as exc:
            failure = {
                "station_id": station_id,
                "label": station.label,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "elapsed_s": time.time() - station_started,
            }
            failures.append(failure)
            (output_root / "all_27stations_failures.json").write_text(
                json.dumps(failures, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"\n[失败] {station_id}号站 {station.label}: {type(exc).__name__}: {exc}")
            if args.stop_on_error:
                raise

    final_summary = pd.DataFrame(summaries)
    final_summary.to_csv(
        output_root / "all_27stations_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    write_aggregate_result_analysis(
        summary=final_summary,
        output_root=output_root,
    )
    trajectory_parts = []
    for path in sorted(output_root.glob("station_*/calibration_trajectory_metrics.csv")):
        try:
            part = pd.read_csv(path, encoding="utf-8-sig")
        except Exception:
            continue
        if not part.empty:
            trajectory_parts.append(part)
    if trajectory_parts:
        trajectory_all = pd.concat(trajectory_parts, ignore_index=True)
        trajectory_all.to_csv(output_root / "all_station_calibration_trajectory_metrics.csv", index=False, encoding="utf-8-sig")
        macro_rows = []
        for sf, grp in trajectory_all.groupby("source_file", sort=True):
            macro_rows.append({
                "source_file": str(sf),
                "station_count": int(grp["station_id"].nunique()),
                "matched_count": int(grp["matched_count"].sum()),
                "station_equal_mse_db2": float(grp["mse_db2"].mean()),
                "station_equal_rmse_db": float(np.sqrt(grp["mse_db2"].mean())),
                "station_equal_bias_db": float(grp["bias_sim_minus_measured_db"].mean()),
            })
        pd.DataFrame(macro_rows).to_csv(output_root / "calibration_trajectory_macro_metrics.csv", index=False, encoding="utf-8-sig")
    if failures:
        pd.DataFrame(failures).to_csv(
            output_root / "all_27stations_failures.csv",
            index=False,
            encoding="utf-8-sig",
        )
    else:
        clear_stale_failure_reports(output_root)

    print("\n全部基站任务结束:", output_root)
    print(f"成功={len(summaries)}, 失败={len(failures)}")


if __name__ == "__main__":
    main()
