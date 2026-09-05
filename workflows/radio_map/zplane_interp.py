#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
zplane_interp.py

固定Z平面插值方法：
- 水平平面垂直间隔固定为1 m；
- 每个平面只保留室外网格单元，建筑内部不创建接收面；
- 建筑物仍保留在Sionna场景中，继续参与遮挡、反射、散射和绕射；
- 各平面使用相同最佳TX参数与传播参数；
- 在“线性功率域”插值到每个室外网格的真实 DEM+1.5 m 高度；
- 只改变接收面算法，便于与直接DEM+1.5 m地形跟随方法公平对照。
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict

import numpy as np
import trimesh

from core_dem15m import (
    DenseGridDefinition,
    RuntimeStationConfig,
    run_candidate_multibatch_linear_average,
)
from src.simulator import Candidate
from src.terrain import SurfaceInfo, TerrainModel


Z_PLANE_STEP_M = 1.0


def _export_surface_mesh(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh = trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=np.float32),
        faces=np.asarray(faces, dtype=np.int32),
        process=False,
        validate=False,
    )
    if len(mesh.face_normals) and np.nanmedian(mesh.face_normals[:, 2]) < 0:
        mesh.faces = mesh.faces[:, ::-1]
    mesh.export(path, file_type="ply", encoding="binary")


def build_dense_outdoor_horizontal_surface(
    grid: DenseGridDefinition,
    z_value_m: float,
    output_path: Path,
) -> SurfaceInfo:
    """构建固定绝对Z高度的室外水平接收面，建筑内部单元不生成三角面。"""
    nx, ny = int(grid.nx), int(grid.ny)
    x_edges = grid.x_min_m + np.arange(nx + 1, dtype=np.float64) * grid.cell_size_m
    y_edges = grid.y_min_m + np.arange(ny + 1, dtype=np.float64) * grid.cell_size_m
    xx, yy = np.meshgrid(x_edges, y_edges)
    zz = np.full(xx.shape, float(z_value_m), dtype=np.float64)
    vertices = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()]).astype(np.float32)

    outdoor_iy, outdoor_ix = np.nonzero(~grid.building_mask)
    if len(outdoor_ix) == 0:
        raise RuntimeError("建筑掩膜覆盖了整张地图，没有室外固定Z接收单元")

    v00 = outdoor_iy.astype(np.int64) * (nx + 1) + outdoor_ix.astype(np.int64)
    v10 = v00 + 1
    v01 = v00 + (nx + 1)
    v11 = v01 + 1
    faces = np.empty((len(outdoor_ix) * 2, 3), dtype=np.int32)
    faces[0::2] = np.column_stack([v00, v10, v11]).astype(np.int32)
    faces[1::2] = np.column_stack([v00, v11, v01]).astype(np.int32)
    _export_surface_mesh(output_path, vertices, faces)

    return SurfaceInfo(
        path=Path(output_path).expanduser().resolve(),
        n_cells=int(len(outdoor_ix)),
        n_faces=int(2 * len(outdoor_ix)),
        cell_ix=outdoor_ix.astype(np.int32),
        cell_iy=outdoor_iy.astype(np.int32),
        cell_center_x=grid.x_m[outdoor_iy, outdoor_ix].astype(np.float64),
        cell_center_y=grid.y_m[outdoor_iy, outdoor_ix].astype(np.float64),
        cell_ground_z=grid.ground_z_m[outdoor_iy, outdoor_ix].astype(np.float64),
        cell_rx_z=np.full(len(outdoor_ix), float(z_value_m), dtype=np.float64),
        nx=nx,
        ny=ny,
    )


def build_z_plane_levels(
    target_receiver_z_m: np.ndarray,
    step_m: float = Z_PLANE_STEP_M,
) -> np.ndarray:
    """构造完整覆盖目标接收高度的固定Z平面序列。"""
    step = float(step_m)
    if not math.isclose(step, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"本对照实验的Z平面垂直间隔固定为1 m，实际传入{step_m}")
    values = np.asarray(target_receiver_z_m, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("目标DEM+1.5m接收高度为空")
    start = math.floor(float(values.min()) / step) * step
    stop = math.ceil(float(values.max()) / step) * step
    count = int(round((stop - start) / step)) + 1
    levels = start + np.arange(count, dtype=np.float64) * step
    return np.round(levels, 9)


def _dbm_to_mw_zero_for_no_hit(dbm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(dbm, dtype=np.float64)
    finite = np.isfinite(arr)
    mw = np.zeros(arr.shape, dtype=np.float64)
    mw[finite] = np.power(10.0, arr[finite] / 10.0)
    return mw, finite


def run_zplane_stack_multibatch_linear_interpolation(
    scene: Any,
    terrain: TerrainModel,
    station: RuntimeStationConfig,
    candidate: Candidate,
    target_surface: SurfaceInfo,
    grid: DenseGridDefinition,
    cfg: Dict[str, Any],
    samples_per_batch: int,
    batch_count: int,
    seed_step: int,
    cache_dir: Path | None,
    work_dir: Path,
    force: bool,
    delete_plane_ply: bool = True,
) -> Dict[str, Any]:
    """
    多固定Z平面仿真，并在线性功率域插值到DEM+1.5m目标高度。

    所有平面：
    - 共用相同的XY网格、建筑室内掩膜、TX最佳参数、射线参数与随机种子序列；
    - 仅平面的绝对Z值不同。
    """
    work_dir = Path(work_dir).expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    if target_surface.n_cells <= 0:
        raise ValueError("目标室外接收面没有单元")

    target_z = np.asarray(grid.receiver_z_m[
        target_surface.cell_iy,
        target_surface.cell_ix,
    ], dtype=np.float64)
    levels = build_z_plane_levels(target_z, Z_PLANE_STEP_M)
    z0 = float(levels[0])
    ratio = (target_z - z0) / Z_PLANE_STEP_M
    nearest = np.rint(ratio)
    exact = np.isclose(ratio, nearest, rtol=0.0, atol=1e-7)
    lower_index = np.floor(ratio + 1e-10).astype(np.int32)
    upper_index = lower_index + 1
    lower_index[exact] = nearest[exact].astype(np.int32)
    upper_index[exact] = nearest[exact].astype(np.int32)
    lower_index = np.clip(lower_index, 0, len(levels) - 1)
    upper_index = np.clip(upper_index, 0, len(levels) - 1)

    n_sector = len(station.pcis)
    output = np.full((n_sector, target_surface.n_cells), np.nan, dtype=np.float32)
    geometrically_assigned = np.zeros(target_surface.n_cells, dtype=bool)

    previous_dbm: np.ndarray | None = None
    previous_z: float | None = None
    previous_plane_index: int | None = None
    plane_records: list[dict[str, Any]] = []
    last_result: Dict[str, Any] | None = None

    for plane_index, z_value in enumerate(levels):
        z_value = float(z_value)
        plane_path = work_dir / f"zplane_{plane_index:04d}_{z_value:.3f}m.ply"
        surface = build_dense_outdoor_horizontal_surface(
            grid=grid,
            z_value_m=z_value,
            output_path=plane_path,
        )
        print(
            f"    Z平面 {plane_index + 1}/{len(levels)}: "
            f"z={z_value:.3f} m, step={Z_PLANE_STEP_M:.1f} m"
        )
        result = run_candidate_multibatch_linear_average(
            scene=scene,
            terrain=terrain,
            station=station,
            candidate=candidate,
            surface=surface,
            cfg=cfg,
            samples_per_batch=int(samples_per_batch),
            batch_count=int(batch_count),
            seed_step=int(seed_step),
            cache_dir=(None if cache_dir is None else Path(cache_dir) / f"plane_{plane_index:04d}"),
            force=bool(force),
        )
        current_dbm = np.asarray(result["sector_rsrp_dbm"], dtype=np.float64)
        if current_dbm.shape != (n_sector, target_surface.n_cells):
            raise RuntimeError(
                f"Z平面结果形状错误: {current_dbm.shape}, "
                f"expected {(n_sector, target_surface.n_cells)}"
            )

        exact_mask = exact & (lower_index == plane_index)
        if np.any(exact_mask):
            output[:, exact_mask] = current_dbm[:, exact_mask].astype(np.float32)
            geometrically_assigned[exact_mask] = True

        if previous_dbm is not None and previous_z is not None and previous_plane_index is not None:
            interval_mask = (
                (~exact)
                & (lower_index == previous_plane_index)
                & (upper_index == plane_index)
            )
            if np.any(interval_mask):
                weight = (
                    (target_z[interval_mask] - previous_z)
                    / max(z_value - previous_z, 1e-12)
                )
                weight = np.clip(weight, 0.0, 1.0)

                lower_dbm = previous_dbm[:, interval_mask]
                upper_dbm = current_dbm[:, interval_mask]
                lower_mw, lower_finite = _dbm_to_mw_zero_for_no_hit(lower_dbm)
                upper_mw, upper_finite = _dbm_to_mw_zero_for_no_hit(upper_dbm)
                interp_mw = (
                    (1.0 - weight[None, :]) * lower_mw
                    + weight[None, :] * upper_mw
                )
                any_hit = lower_finite | upper_finite
                interpolated_dbm = np.full(interp_mw.shape, np.nan, dtype=np.float64)
                positive = any_hit & np.isfinite(interp_mw) & (interp_mw > 0.0)
                interpolated_dbm[positive] = 10.0 * np.log10(interp_mw[positive])
                output[:, interval_mask] = interpolated_dbm.astype(np.float32)
                geometrically_assigned[interval_mask] = True

        plane_records.append({
            "plane_index": int(plane_index),
            "z_m": z_value,
            "batch_count": int(result["batch_count"]),
            "samples_per_batch": int(result["samples_per_batch"]),
            "total_samples_per_tx_for_plane": int(result["total_samples_per_tx"]),
            "cache_hit": bool(result.get("cache_hit", False)),
            "cache_key": str(result.get("cache_key", "")),
            "finite_sector_cell_count": int(np.isfinite(current_dbm).sum()),
        })

        previous_dbm = current_dbm
        previous_z = z_value
        previous_plane_index = plane_index
        last_result = result

        if delete_plane_ply:
            try:
                plane_path.unlink(missing_ok=True)
            except Exception:
                pass
        del surface, current_dbm, result
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    if not np.all(geometrically_assigned):
        missing = int((~geometrically_assigned).sum())
        raise RuntimeError(f"Z平面插值有{missing}个室外网格未被任何上下平面区间覆盖")
    if last_result is None:
        raise RuntimeError("没有执行任何Z平面仿真")

    combined_key = hashlib.sha256(
        json.dumps(plane_records, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:24]
    return {
        "sector_rsrp_dbm": output,
        "alphas_rad": np.asarray(last_result["alphas_rad"], dtype=np.float64),
        "beta_rad": float(last_result["beta_rad"]),
        "ground_z_m": float(last_result["ground_z_m"]),
        "tx_z_m": float(last_result["tx_z_m"]),
        "cache_hit": bool(all(item["cache_hit"] for item in plane_records)),
        "cache_key": combined_key,
        "z_plane_step_m": float(Z_PLANE_STEP_M),
        "z_plane_levels_m": levels.astype(np.float64),
        "z_plane_count": int(len(levels)),
        "z_plane_min_m": float(levels.min()),
        "z_plane_max_m": float(levels.max()),
        "batch_count_per_plane": int(batch_count),
        "samples_per_batch": int(samples_per_batch),
        "samples_per_tx_per_plane": int(batch_count) * int(samples_per_batch),
        "total_samples_per_tx_all_planes": int(len(levels)) * int(batch_count) * int(samples_per_batch),
        "plane_records": plane_records,
        "interpolation_domain": "linear_milliwatt",
        "target_receiver_surface": "DEM_plus_1.5m_outdoor_cells",
        "building_interior_removed_from_each_plane": True,
    }
