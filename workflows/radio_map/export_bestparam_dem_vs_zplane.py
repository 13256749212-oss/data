#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
export_bestparam_dem_vs_zplane.py

使用同一组调参最佳参数，对同一512m×512m区域生成两种可公平对照的无线电地图：

A. DEM-following
   直接在真实DEM+1.5m不平坦室外接收面上进行Sionna RT仿真。

B. Z-plane interpolation
   在多个固定绝对Z水平面上进行Sionna RT仿真，垂直间隔固定1m，
   然后在线性功率域插值到每个室外网格的DEM+1.5m目标高度。

两种方法除接收面算法外保持一致：
- 同一建筑/地形场景；
- 建筑内部均从接收面中排除，但建筑仍参与传播；
- 同一最佳TX高度、功率、方位角和下倾角；
- 同一频率、天线阵列、传播机制、最大深度；
- 同一多批次样本数和seed序列；
- 同一512m×512m、1m XY网格；
- 同一实测残差IDW重构流程。

输出结构：
station_XX/
  01_dem_following/
    01_pure_simulation/{figures,npz}
    02_measurement_reconstructed/{figures,npz}
  02_zplane_interp_1m/
    01_pure_simulation/{figures,npz}
    02_measurement_reconstructed/{figures,npz}
  03_method_comparison/
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

import export_bestparam_radio_maps as base
from core_dem15m import (
    build_all_27_station_configs,
    build_dense_outdoor_measurement_surface,
    build_scene_xml_multi,
    configure_tx_array_for_station,
    create_dense_grid_with_building_mask,
    install_general_sector_support,
    load_building_projection_triangles,
    read_27station_long_measurements,
    run_candidate_multibatch_linear_average,
    sector_values_to_full_maps,
)
from src.optimizer import evaluate_prediction
from src.simulator import Candidate, configure_scene
from src.terrain import TerrainModel
from zplane_interp import (
    Z_PLANE_STEP_M,
    run_zplane_stack_multibatch_linear_interpolation,
)

install_general_sector_support()

MAP_SIZE_M = 512
CELL_SIZE_M = 1.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="按最佳参数生成DEM+1.5m与1m Z平面插值两种512m无线电地图。"
    )
    p.add_argument("--project-root", default=str(Path(__file__).resolve().parents[2]))
    p.add_argument("--measurements", default=None)
    p.add_argument("--summary-csv", default=None)
    p.add_argument("--ground", default=None)
    p.add_argument("--buildings", nargs="*", default=None)
    p.add_argument("--output-root", default=None)
    p.add_argument("--stations", default="all")
    p.add_argument(
        "--methods",
        choices=["both", "dem_following", "zplane_interp"],
        default="both",
        help="默认同时运行两种方法",
    )
    p.add_argument("--force", action="store_true")
    p.add_argument("--continue-on-error", action="store_true")
    p.add_argument("--final-batches", type=int, default=None)
    p.add_argument("--final-samples-per-batch", type=int, default=None)
    p.add_argument("--final-seed-step", type=int, default=1009)
    p.add_argument("--final-max-depth", type=int, default=None)
    p.add_argument("--rx-height-agl-m", type=float, default=1.5)
    p.add_argument("--building-mask-buffer-cells", type=int, default=1)
    p.add_argument("--direction-top-fraction", type=float, default=0.25)
    p.add_argument("--display-min-dbm", type=float, default=-120.0)
    p.add_argument("--display-max-dbm", type=float, default=-40.0)
    p.add_argument("--dpi", type=int, default=1000)
    p.add_argument("--idw-neighbors", type=int, default=12)
    p.add_argument("--idw-power", type=float, default=2.0)
    p.add_argument("--residual-clip-db", type=float, default=30.0)
    return p.parse_args()


def _method_selected(args: argparse.Namespace, name: str) -> bool:
    return args.methods == "both" or args.methods == name


def _metadata(
    station: Any,
    summary_row: pd.Series,
    simulation: Dict[str, Any],
    hit_stats: Dict[str, Any],
    args: argparse.Namespace,
    method_key: str,
    method_description: str,
    final_batches: int,
    final_samples_per_batch: int,
    final_max_depth: int,
    final_edge_diffraction: bool,
) -> Dict[str, Any]:
    return {
        "station_id": int(station.station_id),
        "station_label": station.label,
        "pcis": [int(v) for v in station.pcis],
        "is_omnidirectional": bool(station.is_omnidirectional),
        "simulation_method": method_key,
        "simulation_method_description": method_description,
        "height_agl_m": base._float_value(summary_row, "height_agl_m", 30.0),
        "tx_absolute_z_m": float(simulation["tx_z_m"]),
        "shared_power_dbm": base._float_value(summary_row, "shared_power_dbm", 53.5),
        "azimuth_offset_deg": base._float_value(summary_row, "azimuth_offset_deg", 0.0),
        "absolute_downtilt_deg": base._float_value(summary_row, "absolute_downtilt_deg", 0.0),
        "alphas_rad": [float(v) for v in simulation["alphas_rad"]],
        "beta_rad": float(simulation["beta_rad"]),
        "final_dense_map_rmse_db": float(hit_stats["evaluation_rmse_db"]),
        "measurement_hit_rate": float(hit_stats["measurement_hit_rate"]),
        "measurement_cell_count": int(hit_stats["measurement_cell_count"]),
        "simulated_hit_count_at_measurement_cells": int(hit_stats["simulated_hit_count"]),
        "final_batches": int(final_batches),
        "final_samples_per_batch": int(final_samples_per_batch),
        "final_max_depth": int(final_max_depth),
        "final_edge_diffraction": bool(final_edge_diffraction),
        "display_min_dbm": float(args.display_min_dbm),
        "display_max_dbm": float(args.display_max_dbm),
        "receiver_target": "outdoor DEM+1.5m",
        "building_interior_removed_from_receiver_domain": True,
    }


def _evaluate(
    station: Any,
    measurements: pd.DataFrame,
    target_surface: Any,
    sector_values: np.ndarray,
    shared_power_dbm: float,
) -> Dict[str, Any]:
    evaluation = evaluate_prediction(
        station=station,
        measurements=measurements,
        surface=target_surface,
        sector_rsrp_at_reference_dbm=np.asarray(sector_values, dtype=np.float32),
        reference_power_dbm=float(shared_power_dbm),
        power_candidates_dbm=np.asarray([float(shared_power_dbm)], dtype=np.float64),
    )
    stats = base.compute_measurement_hit_statistics(station, measurements, evaluation)
    stats["evaluation_rmse_db"] = float(evaluation["rmse_db"])
    stats["evaluation_mae_db"] = float(evaluation["mae_db"])
    stats["evaluation_bias_db"] = float(evaluation["bias_sim_minus_meas_db"])
    return stats


def _save_method(
    method_root: Path,
    method_key: str,
    method_label: str,
    station: Any,
    grid: Any,
    measurements: pd.DataFrame,
    pure_sector_maps: np.ndarray,
    metadata: Dict[str, Any],
    hit_stats: Dict[str, Any],
    args: argparse.Namespace,
    method_extra: Dict[str, Any],
) -> Dict[str, Any]:
    method_root.mkdir(parents=True, exist_ok=True)
    pure_png = method_root / "01_pure_simulation" / "figures"
    pure_npz = method_root / "01_pure_simulation" / "npz"
    rec_png = method_root / "02_measurement_reconstructed" / "figures"
    rec_npz = method_root / "02_measurement_reconstructed" / "npz"
    for d in (pure_png, pure_npz, rec_png, rec_npz):
        d.mkdir(parents=True, exist_ok=True)

    pure_best, pure_best_pci = base.compute_best_server(station, pure_sector_maps)
    reconstruction = base.reconstruct_all_outdoor_cells_from_measurements(
        station=station,
        grid=grid,
        pure_sector_maps=pure_sector_maps,
        measurements=measurements,
        display_min_dbm=float(args.display_min_dbm),
        display_max_dbm=float(args.display_max_dbm),
        idw_neighbors=int(args.idw_neighbors),
        idw_power=float(args.idw_power),
        residual_clip_db=float(args.residual_clip_db),
    )
    rec_sector_maps = reconstruction["reconstructed_sector_maps"]
    rec_best, rec_best_pci = base.compute_best_server(station, rec_sector_maps)
    outdoor_count = int((~grid.building_mask).sum())

    exact_counts: dict[int, int] = {}
    for idx, pci_value in enumerate(station.pcis):
        pci = int(pci_value)
        sector_meta = dict(metadata)
        sector_meta["measurement_hit_rate"] = float(
            hit_stats["per_pci"][pci]["measurement_hit_rate"]
        )
        exact_count = int(reconstruction["exact_measurement_mask"][idx].sum())
        exact_counts[pci] = exact_count

        pure_stem = f"station_{station.station_id:02d}_pci_{pci}_{method_key}_pure"
        base.plot_rsrp_map(
            path=pure_png / f"{pure_stem}.png",
            station=station,
            pci=pci,
            map_version_label=f"{method_label} | Pure",
            rsrp_map=pure_sector_maps[idx],
            grid=grid,
            metadata=sector_meta,
            dpi=int(args.dpi),
        )
        base.save_sector_npz(
            path=pure_npz / f"{pure_stem}.npz",
            station=station,
            pci=pci,
            map_version=f"{method_key}_pure_simulation",
            rsrp_map=pure_sector_maps[idx],
            grid=grid,
            metadata=sector_meta,
            extra_arrays={
                "simulation_method": np.asarray([method_key]),
                "raw_no_hit_mask": (~np.isfinite(pure_sector_maps[idx])) & (~grid.building_mask),
                **method_extra,
            },
        )

        rec_stem = f"station_{station.station_id:02d}_pci_{pci}_{method_key}_measurement_reconstructed"
        base.plot_rsrp_map(
            path=rec_png / f"{rec_stem}.png",
            station=station,
            pci=pci,
            map_version_label=f"{method_label} | Measurement-reconstructed",
            rsrp_map=rec_sector_maps[idx],
            grid=grid,
            metadata=sector_meta,
            dpi=int(args.dpi),
            reconstruction_count=outdoor_count,
        )
        base.save_sector_npz(
            path=rec_npz / f"{rec_stem}.npz",
            station=station,
            pci=pci,
            map_version=f"{method_key}_measurement_reconstructed",
            rsrp_map=rec_sector_maps[idx],
            grid=grid,
            metadata=sector_meta,
            extra_arrays={
                "simulation_method": np.asarray([method_key]),
                "pure_sim_rsrp_dbm": pure_sector_maps[idx],
                "simulation_baseline_rsrp_dbm": reconstruction["simulation_baselines"][idx],
                "measured_rsrp_grid_dbm": reconstruction["measured_grid"][idx],
                "exact_measurement_mask": reconstruction["exact_measurement_mask"][idx],
                "residual_correction_db": reconstruction["residual_correction_db"][idx],
                "measurement_support_distance_m": reconstruction["measurement_support_distance_m"][idx],
                "reconstruction_method": np.asarray(["simulation_residual_idw"]),
                **method_extra,
            },
        )

    pure_best_stem = f"station_{station.station_id:02d}_{method_key}_best_server_pure"
    base.plot_rsrp_map(
        path=pure_png / f"{pure_best_stem}.png",
        station=station,
        pci=None,
        map_version_label=f"{method_label} | Pure",
        rsrp_map=pure_best,
        grid=grid,
        metadata=metadata,
        dpi=int(args.dpi),
    )
    base.save_station_combined_npz(
        path=pure_npz / f"{pure_best_stem}.npz",
        station=station,
        map_version=f"{method_key}_pure_simulation",
        sector_maps=pure_sector_maps,
        best_server_map=pure_best,
        best_server_pci=pure_best_pci,
        grid=grid,
        metadata=metadata,
        extra_arrays={
            "simulation_method": np.asarray([method_key]),
            "raw_no_hit_mask_per_sector": (~np.isfinite(pure_sector_maps)) & (~grid.building_mask[None, ...]),
            **method_extra,
        },
    )

    rec_best_stem = f"station_{station.station_id:02d}_{method_key}_best_server_measurement_reconstructed"
    base.plot_rsrp_map(
        path=rec_png / f"{rec_best_stem}.png",
        station=station,
        pci=None,
        map_version_label=f"{method_label} | Measurement-reconstructed",
        rsrp_map=rec_best,
        grid=grid,
        metadata=metadata,
        dpi=int(args.dpi),
        reconstruction_count=outdoor_count,
    )
    base.save_station_combined_npz(
        path=rec_npz / f"{rec_best_stem}.npz",
        station=station,
        map_version=f"{method_key}_measurement_reconstructed",
        sector_maps=rec_sector_maps,
        best_server_map=rec_best,
        best_server_pci=rec_best_pci,
        grid=grid,
        metadata=metadata,
        extra_arrays={
            "simulation_method": np.asarray([method_key]),
            "pure_sector_rsrp_dbm": pure_sector_maps,
            "simulation_baseline_sector_rsrp_dbm": reconstruction["simulation_baselines"],
            "measured_rsrp_grid_dbm": reconstruction["measured_grid"],
            "exact_measurement_mask": reconstruction["exact_measurement_mask"],
            "residual_correction_db": reconstruction["residual_correction_db"],
            "measurement_support_distance_m": reconstruction["measurement_support_distance_m"],
            "reconstruction_method": np.asarray(["simulation_residual_idw"]),
            **method_extra,
        },
    )

    method_metadata = dict(metadata)
    method_metadata.update({
        "method_output_root": str(method_root),
        "exact_measurement_cells_per_pci": exact_counts,
        "measurement_hit_per_pci": hit_stats["per_pci"],
        "evaluation_mae_db": hit_stats["evaluation_mae_db"],
        "evaluation_bias_sim_minus_meas_db": hit_stats["evaluation_bias_db"],
    })
    (method_root / "method_metadata.json").write_text(
        json.dumps(method_metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "pure_sector_maps": pure_sector_maps,
        "pure_best_map": pure_best,
        "pure_best_pci": pure_best_pci,
        "reconstructed_sector_maps": rec_sector_maps,
        "reconstructed_best_map": rec_best,
        "metadata": method_metadata,
    }


def _comparison_metrics(
    station: Any,
    dem_result: Dict[str, Any],
    z_result: Dict[str, Any],
    grid: Any,
    output_dir: Path,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    outdoor = ~grid.building_mask

    def add_row(label: str, a: np.ndarray, b: np.ndarray) -> None:
        valid = outdoor & np.isfinite(a) & np.isfinite(b)
        diff = b[valid] - a[valid]
        rows.append({
            "station_id": int(station.station_id),
            "map": label,
            "overlap_cell_count": int(valid.sum()),
            "dem_finite_outdoor_count": int((outdoor & np.isfinite(a)).sum()),
            "zplane_finite_outdoor_count": int((outdoor & np.isfinite(b)).sum()),
            "mean_zplane_minus_dem_db": float(np.mean(diff)) if diff.size else np.nan,
            "mae_between_methods_db": float(np.mean(np.abs(diff))) if diff.size else np.nan,
            "rmse_between_methods_db": float(np.sqrt(np.mean(diff ** 2))) if diff.size else np.nan,
            "p95_absolute_difference_db": float(np.percentile(np.abs(diff), 95)) if diff.size else np.nan,
            "max_absolute_difference_db": float(np.max(np.abs(diff))) if diff.size else np.nan,
        })

    for idx, pci in enumerate(station.pcis):
        add_row(
            f"PCI_{int(pci)}_pure",
            dem_result["pure_sector_maps"][idx],
            z_result["pure_sector_maps"][idx],
        )
    add_row("best_server_pure", dem_result["pure_best_map"], z_result["pure_best_map"])
    for idx, pci in enumerate(station.pcis):
        add_row(
            f"PCI_{int(pci)}_measurement_reconstructed",
            dem_result["reconstructed_sector_maps"][idx],
            z_result["reconstructed_sector_maps"][idx],
        )
    add_row(
        "best_server_measurement_reconstructed",
        dem_result["reconstructed_best_map"],
        z_result["reconstructed_best_map"],
    )

    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "dem_vs_zplane_metrics.csv", index=False, encoding="utf-8-sig")
    np.savez_compressed(
        output_dir / "dem_vs_zplane_maps.npz",
        station_id=np.asarray([station.station_id], dtype=np.int32),
        pcis=np.asarray(station.pcis, dtype=np.int32),
        building_mask=grid.building_mask.astype(bool),
        x_m=grid.x_m.astype(np.float32),
        y_m=grid.y_m.astype(np.float32),
        receiver_z_m=grid.receiver_z_m.astype(np.float32),
        dem_pure_sector_rsrp_dbm=dem_result["pure_sector_maps"].astype(np.float32),
        zplane_pure_sector_rsrp_dbm=z_result["pure_sector_maps"].astype(np.float32),
        zplane_minus_dem_pure_sector_db=(z_result["pure_sector_maps"] - dem_result["pure_sector_maps"]).astype(np.float32),
        dem_pure_best_server_rsrp_dbm=dem_result["pure_best_map"].astype(np.float32),
        zplane_pure_best_server_rsrp_dbm=z_result["pure_best_map"].astype(np.float32),
        zplane_minus_dem_pure_best_server_db=(z_result["pure_best_map"] - dem_result["pure_best_map"]).astype(np.float32),
        z_plane_step_m=np.asarray([Z_PLANE_STEP_M], dtype=np.float32),
    )
    return frame


def process_station(
    station: Any,
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
    station_root = output_root / f"station_{sid:02d}"
    work = station_root / "work"
    work.mkdir(parents=True, exist_ok=True)

    grid, grid_diag = create_dense_grid_with_building_mask(
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
    target_surface = build_dense_outdoor_measurement_surface(
        terrain=terrain,
        grid=grid,
        rx_height_agl_m=float(args.rx_height_agl_m),
        output_path=work / f"station_{sid:02d}_dem_plus_1p5m_outdoor_surface.ply",
    )
    measurements, inside = base.prepare_station_measurement_grid(
        observations=observations,
        station=station,
        building_mask=grid.building_mask,
    )
    measurements.to_csv(station_root / "measurement_cells_outdoor_1m.csv", index=False, encoding="utf-8-sig")
    inside.to_csv(station_root / "measurement_cells_inside_buildings_excluded.csv", index=False, encoding="utf-8-sig")

    height = base._float_value(summary_row, "height_agl_m", 30.0)
    azimuth = base._float_value(summary_row, "azimuth_offset_deg", 0.0)
    tilt = base._float_value(summary_row, "absolute_downtilt_deg", 0.0)
    power = base._float_value(summary_row, "shared_power_dbm", 53.5)
    batches = int(args.final_batches) if args.final_batches is not None else base._int_value(summary_row, "final_batch_count", 5)
    samples = int(args.final_samples_per_batch) if args.final_samples_per_batch is not None else base._int_value(summary_row, "final_samples_per_batch", 10_000_000)
    max_depth = int(args.final_max_depth) if args.final_max_depth is not None else base._int_value(summary_row, "final_max_depth", 5)
    edge_diff = base._bool_value(summary_row, "final_edge_diffraction", True)

    cfg = base.make_station_cfg(
        scene_xml=scene_xml,
        station=station,
        rx_height_agl_m=float(args.rx_height_agl_m),
        max_depth=max_depth,
        edge_diffraction=edge_diff,
    )
    configure_tx_array_for_station(scene, cfg, station)
    candidate = Candidate(
        height_agl_m=height,
        azimuth_offset_deg=azimuth,
        downtilt_delta_deg=tilt,
        reference_power_dbm=power,
    )

    results: Dict[str, Any] = {}

    if _method_selected(args, "dem_following"):
        print(f"\nStation {sid:02d}: 方法A 直接DEM+1.5m地形跟随")
        dem_sim = run_candidate_multibatch_linear_average(
            scene=scene,
            terrain=terrain,
            station=station,
            candidate=candidate,
            surface=target_surface,
            cfg=cfg,
            samples_per_batch=samples,
            batch_count=batches,
            seed_step=int(args.final_seed_step),
            cache_dir=station_root / "01_dem_following" / "cache",
            force=bool(args.force),
        )
        dem_hit = _evaluate(station, measurements, target_surface, dem_sim["sector_rsrp_dbm"], power)
        dem_meta = _metadata(
            station, summary_row, dem_sim, dem_hit, args,
            "dem_following", "Direct outdoor DEM+1.5m terrain-following surface",
            batches, samples, max_depth, edge_diff,
        )
        dem_meta["grid_diagnostics"] = grid_diag
        dem_maps = sector_values_to_full_maps(target_surface, dem_sim["sector_rsrp_dbm"], grid)
        results["dem_following"] = _save_method(
            method_root=station_root / "01_dem_following",
            method_key="dem_following",
            method_label="DEM+1.5m",
            station=station,
            grid=grid,
            measurements=measurements,
            pure_sector_maps=dem_maps,
            metadata=dem_meta,
            hit_stats=dem_hit,
            args=args,
            method_extra={
                "receiver_surface_method": np.asarray(["direct_dem_following"]),
                "rx_height_agl_m": np.asarray([args.rx_height_agl_m], dtype=np.float32),
                "batch_count": np.asarray([batches], dtype=np.int32),
                "samples_per_batch": np.asarray([samples], dtype=np.int64),
            },
        )

    if _method_selected(args, "zplane_interp"):
        print(f"\nStation {sid:02d}: 方法B 固定Z平面1m间隔线性功率插值")
        configure_tx_array_for_station(scene, cfg, station)
        z_sim = run_zplane_stack_multibatch_linear_interpolation(
            scene=scene,
            terrain=terrain,
            station=station,
            candidate=candidate,
            target_surface=target_surface,
            grid=grid,
            cfg=cfg,
            samples_per_batch=samples,
            batch_count=batches,
            seed_step=int(args.final_seed_step),
            cache_dir=station_root / "02_zplane_interp_1m" / "cache",
            work_dir=work / "zplane_surfaces",
            force=bool(args.force),
            delete_plane_ply=True,
        )
        z_hit = _evaluate(station, measurements, target_surface, z_sim["sector_rsrp_dbm"], power)
        z_meta = _metadata(
            station, summary_row, z_sim, z_hit, args,
            "zplane_interp_1m", "Fixed Z planes at 1m intervals; linear-power interpolation to outdoor DEM+1.5m",
            batches, samples, max_depth, edge_diff,
        )
        z_meta.update({
            "grid_diagnostics": grid_diag,
            "z_plane_step_m": float(z_sim["z_plane_step_m"]),
            "z_plane_count": int(z_sim["z_plane_count"]),
            "z_plane_min_m": float(z_sim["z_plane_min_m"]),
            "z_plane_max_m": float(z_sim["z_plane_max_m"]),
            "z_plane_total_samples_per_tx_all_planes": int(z_sim["total_samples_per_tx_all_planes"]),
            "z_interpolation_domain": "linear_milliwatt",
        })
        z_maps = sector_values_to_full_maps(target_surface, z_sim["sector_rsrp_dbm"], grid)
        results["zplane_interp"] = _save_method(
            method_root=station_root / "02_zplane_interp_1m",
            method_key="zplane_interp_1m",
            method_label="Z-plane 1m → DEM+1.5m",
            station=station,
            grid=grid,
            measurements=measurements,
            pure_sector_maps=z_maps,
            metadata=z_meta,
            hit_stats=z_hit,
            args=args,
            method_extra={
                "receiver_surface_method": np.asarray(["fixed_z_planes_linear_power_interpolation"]),
                "z_plane_step_m": np.asarray([Z_PLANE_STEP_M], dtype=np.float32),
                "z_plane_levels_m": np.asarray(z_sim["z_plane_levels_m"], dtype=np.float64),
                "z_plane_count": np.asarray([z_sim["z_plane_count"]], dtype=np.int32),
                "interpolation_domain": np.asarray(["linear_milliwatt"]),
                "batch_count_per_plane": np.asarray([batches], dtype=np.int32),
                "samples_per_batch": np.asarray([samples], dtype=np.int64),
            },
        )
        (station_root / "02_zplane_interp_1m" / "zplane_run_diagnostics.json").write_text(
            json.dumps({
                key: value for key, value in z_sim.items()
                if key not in {"sector_rsrp_dbm", "alphas_rad"}
            }, ensure_ascii=False, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x)),
            encoding="utf-8",
        )

    comparison = None
    if "dem_following" in results and "zplane_interp" in results:
        comparison = _comparison_metrics(
            station=station,
            dem_result=results["dem_following"],
            z_result=results["zplane_interp"],
            grid=grid,
            output_dir=station_root / "03_method_comparison",
        )

    station_summary = {
        "station_id": sid,
        "station_label": station.label,
        "pcis": [int(v) for v in station.pcis],
        "is_omnidirectional": bool(station.is_omnidirectional),
        "methods_completed": list(results.keys()),
        "z_plane_step_m": float(Z_PLANE_STEP_M),
        "same_best_parameters_for_both_methods": True,
        "same_scene_propagation_and_random_seed_settings": True,
        "only_receiver_surface_algorithm_differs": True,
    }
    if comparison is not None:
        row = comparison.loc[comparison["map"].eq("best_server_pure")]
        if not row.empty:
            station_summary["pure_best_server_method_rmse_db"] = float(row.iloc[0]["rmse_between_methods_db"])
            station_summary["pure_best_server_method_mae_db"] = float(row.iloc[0]["mae_between_methods_db"])
    (station_root / "station_comparison_metadata.json").write_text(
        json.dumps(station_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return station_summary


def main() -> None:
    args = parse_args()
    if not math.isclose(Z_PLANE_STEP_M, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError("Z_PLANE_STEP_M必须固定为1.0 m")

    project_root = Path(args.project_root).expanduser().resolve()
    measurements_csv = base.resolve_measurements(project_root, args.measurements)
    summary_csv = base.resolve_summary(project_root, args.summary_csv)
    ground_ply = base.resolve_ground(project_root, args.ground)
    building_paths = base.resolve_buildings(project_root, args.buildings)
    output_root = (
        Path(args.output_root).expanduser().resolve()
        if args.output_root
        else (project_root / "outputs" / "bestparam_dem_vs_zplane_512m").resolve()
    )
    output_root.mkdir(parents=True, exist_ok=True)
    work_root = output_root / "work"
    work_root.mkdir(parents=True, exist_ok=True)

    summary = base.load_best_parameter_summary(summary_csv)
    selected = base.select_station_ids(args.stations, summary["station_id"].tolist())
    summary = summary.loc[summary["station_id"].isin(selected)].copy()

    raw_long, observations = read_27station_long_measurements(measurements_csv)
    stations, direction_diag = build_all_27_station_configs(
        raw_long=raw_long,
        top_fraction=float(args.direction_top_fraction),
        initial_power_dbm=53.5,
    )
    pd.DataFrame(direction_diag).to_csv(
        output_root / "estimated_initial_directions_27stations.csv",
        index=False,
        encoding="utf-8-sig",
    )

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
        json.dumps(scene_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    building_triangles, building_diag = load_building_projection_triangles(
        scene_report["building_included_paths"]
    )
    (output_root / "building_projection_diagnostics.json").write_text(
        json.dumps({
            "total_nondegenerate_xy_triangles": int(len(building_triangles)),
            "files": building_diag,
        }, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    bootstrap_station = next(
        (stations[sid] for sid in selected if not stations[sid].is_omnidirectional),
        stations[selected[0]],
    )
    bootstrap_cfg = base.make_station_cfg(
        scene_xml=scene_xml,
        station=bootstrap_station,
        rx_height_agl_m=float(args.rx_height_agl_m),
        max_depth=5,
        edge_diffraction=True,
    )
    scene = configure_scene(scene_xml, bootstrap_cfg)

    completed: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for _, row in summary.iterrows():
        sid = int(row["station_id"])
        print("\n" + "=" * 100)
        print(f"Station {sid:02d}: DEM+1.5m 与 Z-plane 1m 对照")
        try:
            item = process_station(
                station=stations[sid],
                summary_row=row,
                observations=observations,
                terrain=terrain,
                scene=scene,
                scene_xml=scene_xml,
                building_triangles_xy=building_triangles,
                output_root=output_root,
                args=args,
            )
            completed.append(item)
        except Exception as exc:
            failures.append({
                "station_id": sid,
                "error_type": type(exc).__name__,
                "error": str(exc),
            })
            print(f"Station {sid:02d}失败: {type(exc).__name__}: {exc}")
            if not args.continue_on_error:
                raise

    if completed:
        pd.DataFrame(completed).to_csv(
            output_root / "completed_station_methods.csv",
            index=False,
            encoding="utf-8-sig",
        )
    if failures:
        pd.DataFrame(failures).to_csv(
            output_root / "failures.csv",
            index=False,
            encoding="utf-8-sig",
        )
    print(f"\n完成。成功={len(completed)}，失败={len(failures)}，输出={output_root}")


if __name__ == "__main__":
    main()
