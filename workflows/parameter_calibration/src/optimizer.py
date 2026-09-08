from __future__ import annotations

import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import pandas as pd

from .configuration import StationConfig, inclusive_grid
from .simulator import Candidate, run_candidate
from .terrain import SurfaceInfo, TerrainModel


def _surface_cell_lookup(surface: SurfaceInfo) -> dict[tuple[int, int], int]:
    return {
        (int(ix), int(iy)): idx
        for idx, (ix, iy) in enumerate(zip(surface.cell_ix, surface.cell_iy))
    }


def evaluate_prediction(
    station: StationConfig,
    measurements: pd.DataFrame,
    surface: SurfaceInfo,
    sector_rsrp_at_reference_dbm: np.ndarray,
    reference_power_dbm: float,
    power_candidates_dbm: np.ndarray,
) -> Dict[str, Any]:
    lookup = _surface_cell_lookup(surface)
    sector_by_pci = {int(pci): idx for idx, pci in enumerate(station.pcis)}
    measured: list[float] = []
    predicted_reference: list[float] = []
    pcis: list[int] = []
    rows_kept: list[int] = []

    for row_index, row in measurements.iterrows():
        pci = int(row["pci"])
        cell_index = lookup.get((int(row["ix"]), int(row["iy"])))
        sector_index = sector_by_pci.get(pci)
        if cell_index is None or sector_index is None:
            continue
        pred = float(sector_rsrp_at_reference_dbm[sector_index, cell_index])
        meas = float(row["measured_rsrp_dbm"])
        if np.isfinite(pred) and np.isfinite(meas):
            measured.append(meas)
            predicted_reference.append(pred)
            pcis.append(pci)
            rows_kept.append(int(row_index))

    if not measured:
        raise RuntimeError("该候选参数没有形成任何有限的同PCI仿真-实测配对")
    y = np.asarray(measured, dtype=np.float64)
    p0 = np.asarray(predicted_reference, dtype=np.float64)
    pci_arr = np.asarray(pcis, dtype=np.int32)

    rmse_values = []
    for power_dbm in power_candidates_dbm:
        pred = p0 + (float(power_dbm) - float(reference_power_dbm))
        rmse_values.append(float(np.sqrt(np.mean((pred - y) ** 2))))
    rmse_values_arr = np.asarray(rmse_values)
    best_index = int(np.nanargmin(rmse_values_arr))
    best_power = float(power_candidates_dbm[best_index])
    pred_best = p0 + (best_power - float(reference_power_dbm))
    error = pred_best - y

    per_pci = {}
    for pci in station.pcis:
        mask = pci_arr == int(pci)
        if np.any(mask):
            per_pci[int(pci)] = {
                "count": int(mask.sum()),
                "rmse_db": float(np.sqrt(np.mean(error[mask] ** 2))),
                "mae_db": float(np.mean(np.abs(error[mask]))),
                "bias_sim_minus_meas_db": float(np.mean(error[mask])),
            }

    return {
        "rmse_db": float(rmse_values_arr[best_index]),
        "best_power_dbm": best_power,
        "mae_db": float(np.mean(np.abs(error))),
        "bias_sim_minus_meas_db": float(np.mean(error)),
        "paired_point_count": int(len(y)),
        "per_pci": per_pci,
        "kept_measurement_indices": rows_kept,
        "measured_rsrp_dbm": y,
        "predicted_reference_dbm": p0,
        "predicted_best_power_dbm": pred_best,
        "pci": pci_arr,
        "error_db": error,
    }


def _candidate_row(
    station: StationConfig,
    candidate: Candidate,
    evaluation: Dict[str, Any],
    simulation: Dict[str, Any],
    pass_index: int,
    stage: str,
    candidate_index: int,
    elapsed_s: float,
) -> Dict[str, Any]:
    return {
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
        "alpha_1_rad": float(simulation["alphas_rad"][0]),
        "alpha_2_rad": float(simulation["alphas_rad"][1]),
        "alpha_3_rad": float(simulation["alphas_rad"][2]),
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


def optimize_station(
    scene: Any,
    terrain: TerrainModel,
    station: StationConfig,
    measurements: pd.DataFrame,
    sparse_surface: SurfaceInfo,
    cfg: Dict[str, Any],
    output_dir: Path,
    force: bool = False,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    search_cfg = cfg["search"]
    radio_cfg = cfg["radio"]
    cache_dir = output_dir / "cache_sparse_maps" if bool(search_cfg.get("cache_maps", True)) else None

    heights = inclusive_grid(
        float(search_cfg["height_min_m"]),
        float(search_cfg["height_max_m"]),
        float(search_cfg["height_step_m"]),
    )
    az_offsets = inclusive_grid(
        float(search_cfg["azimuth_offset_min_deg"]),
        float(search_cfg["azimuth_offset_max_deg"]),
        float(search_cfg["azimuth_offset_step_deg"]),
    )
    tilt_deltas = inclusive_grid(
        float(search_cfg["downtilt_delta_min_deg"]),
        float(search_cfg["downtilt_delta_max_deg"]),
        float(search_cfg["downtilt_delta_step_deg"]),
    )
    power_grid = inclusive_grid(
        float(search_cfg["power_min_dbm"]),
        float(search_cfg["power_max_dbm"]),
        float(search_cfg["power_step_db"]),
    )
    samples = int(radio_cfg["samples_per_tx_search"])
    reference_power = float(station.initial_power_dbm)

    state = {
        "height_agl_m": float(search_cfg["initial_height_agl_m"]),
        "azimuth_offset_deg": 0.0,
        "downtilt_delta_deg": 0.0,
        "shared_power_dbm": float(np.clip(station.initial_power_dbm, power_grid.min(), power_grid.max())),
        "rmse_db": float("inf"),
    }
    history: list[dict[str, Any]] = []
    best_payload: Dict[str, Any] | None = None

    def run_stage(pass_index: int, stage: str, values: Iterable[float]) -> None:
        nonlocal state, best_payload
        stage_payloads = []
        for candidate_index, value in enumerate(values, start=1):
            params = dict(state)
            if stage == "azimuth":
                params["azimuth_offset_deg"] = float(value)
            elif stage == "height":
                params["height_agl_m"] = float(value)
            elif stage == "downtilt":
                params["downtilt_delta_deg"] = float(value)
            else:
                raise ValueError(stage)
            candidate = Candidate(
                height_agl_m=float(params["height_agl_m"]),
                azimuth_offset_deg=float(params["azimuth_offset_deg"]),
                downtilt_delta_deg=float(params["downtilt_delta_deg"]),
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
            )
            row = _candidate_row(
                station,
                candidate,
                evaluation,
                simulation,
                pass_index,
                stage,
                candidate_index,
                time.time() - started,
            )
            history.append(row)
            stage_payloads.append(
                {
                    "row": row,
                    "candidate": candidate,
                    "simulation": simulation,
                    "evaluation": evaluation,
                }
            )
            print(
                f"  [{stage} {candidate_index:02d}] h={candidate.height_agl_m:.0f}m, "
                f"az_off={candidate.azimuth_offset_deg:+.0f}°, "
                f"tilt_delta={candidate.downtilt_delta_deg:+.0f}°, "
                f"P={evaluation['best_power_dbm']:.2f}dBm, "
                f"RMSE={evaluation['rmse_db']:.3f}dB"
            )

        selected = min(stage_payloads, key=lambda p: p["evaluation"]["rmse_db"])
        c = selected["candidate"]
        e = selected["evaluation"]
        state.update(
            {
                "height_agl_m": c.height_agl_m,
                "azimuth_offset_deg": c.azimuth_offset_deg,
                "downtilt_delta_deg": c.downtilt_delta_deg,
                "shared_power_dbm": e["best_power_dbm"],
                "rmse_db": e["rmse_db"],
            }
        )
        best_payload = selected
        print(
            f"  -> {stage}阶段最优: h={state['height_agl_m']:.0f}m, "
            f"az_off={state['azimuth_offset_deg']:+.0f}°, "
            f"tilt_delta={state['downtilt_delta_deg']:+.0f}°, "
            f"P={state['shared_power_dbm']:.2f}dBm, RMSE={state['rmse_db']:.3f}dB"
        )

    for pass_index in range(1, int(search_cfg.get("passes", 1)) + 1):
        print(f"\n{station.station_id}号站，第{pass_index}轮坐标搜索")
        run_stage(pass_index, "azimuth", az_offsets)
        run_stage(pass_index, "height", heights)
        run_stage(pass_index, "downtilt", tilt_deltas)

    history_frame = pd.DataFrame(history)
    history_frame.to_csv(output_dir / "search_history.csv", index=False, encoding="utf-8-sig")
    if best_payload is None:
        raise RuntimeError("搜索没有产生候选结果")

    best = {
        "station_id": station.station_id,
        "label": station.label,
        "pcis": list(station.pcis),
        "x_m": station.x_m,
        "y_m": station.y_m,
        "height_agl_m": state["height_agl_m"],
        "ground_z_m": best_payload["simulation"]["ground_z_m"],
        "tx_absolute_z_m": best_payload["simulation"]["tx_z_m"],
        "azimuth_offset_deg": state["azimuth_offset_deg"],
        "azimuth_offset_rad": math.radians(state["azimuth_offset_deg"]),
        "alphas_rad": best_payload["simulation"]["alphas_rad"].tolist(),
        "original_downtilt_deg": station.original_downtilt_deg,
        "downtilt_delta_deg": state["downtilt_delta_deg"],
        "absolute_downtilt_deg": station.original_downtilt_deg + state["downtilt_delta_deg"],
        "beta_rad": best_payload["simulation"]["beta_rad"],
        "gamma_rad": 0.0,
        "shared_power_dbm": state["shared_power_dbm"],
        "pooled_equal_pci_rmse_db": state["rmse_db"],
        "paired_point_count": best_payload["evaluation"]["paired_point_count"],
        "per_pci": best_payload["evaluation"]["per_pci"],
        "search_samples_per_tx": samples,
        "rsrp_definition": "RSS_dBm - 10log10(273*12) + calibration_offset",
        "selection_metric": str(search_cfg["metric"]),
    }
    (output_dir / "best_parameters.json").write_text(
        json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"best": best, "history": history_frame, "best_payload": best_payload}
