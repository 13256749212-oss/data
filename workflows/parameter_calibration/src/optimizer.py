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
    objective_mode: str = "pooled_rmse",
    power_regularization_lambda: float = 0.0,
    power_regularization_anchor_dbm: float | None = None,
    min_trajectory_points: int = 20,
    trajectory_weight_cap: int = 250,
    min_pci_points: int = 3,
    pci_weight_cap: int = 100,
) -> Dict[str, Any]:
    """Evaluate one geometry/power candidate.

    adds a reliability-balanced hierarchical objective.  Earlier
    trajectory-balanced code gave a trajectory with only a handful of matched
    cells exactly the same weight as a trajectory with hundreds or thousands
    of cells.  That made tiny, noisy fragments dominate power/geometry fitting.

    ``reliability_balanced_pci_rmse`` therefore:
      1. excludes trajectory fragments with fewer than ``min_trajectory_points``
         from *selection* (they remain in diagnostics),
      2. combines PCI groups inside a trajectory with sqrt-capped support weights,
      3. combines trajectories with sqrt-capped support weights,
      4. falls back deterministically to all available groups if a station has
         no trajectory meeting the minimum support threshold.

    The cap prevents the longest drive from overwhelming the objective while
    the square-root support term prevents a 3-cell fragment from receiving the
    same leverage as a 1000-cell trajectory.
    """
    lookup = _surface_cell_lookup(surface)
    sector_by_pci = {int(pci): idx for idx, pci in enumerate(station.pcis)}
    measured, predicted_reference, pcis, source_files, rows_kept = [], [], [], [], []
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
            source_files.append(str(row.get("source_file", "unknown")))
            rows_kept.append(int(row_index))
    if not measured:
        raise RuntimeError("该候选参数没有形成任何有限的同PCI仿真-实测配对")

    y = np.asarray(measured, np.float64)
    p0 = np.asarray(predicted_reference, np.float64)
    pci_arr = np.asarray(pcis, np.int32)
    source_arr = np.asarray(source_files, dtype=object)
    mode = str(objective_mode or "pooled_rmse").strip().lower()
    reliability_mode = mode in {
        "reliability_balanced_pci_rmse",
        "reliability_balanced_rmse",
        "hierarchical_reliability_rmse",
    }
    trajectory_mode_compat = mode in {
        "trajectory_balanced_pci_rmse",
        "trajectory_balanced_rmse",
        "trajectory_macro_rmse",
    }
    anchor = (
        float(power_regularization_anchor_dbm)
        if power_regularization_anchor_dbm is not None
        else float(np.mean(power_candidates_dbm))
    )
    lam = max(0.0, float(power_regularization_lambda))
    min_traj = max(1, int(min_trajectory_points))
    traj_cap = max(min_traj, int(trajectory_weight_cap))
    min_pci = max(1, int(min_pci_points))
    pci_cap = max(min_pci, int(pci_weight_cap))

    def _sqrt_cap_weight(count: int, cap: int) -> float:
        return float(np.sqrt(float(max(1, min(int(count), int(cap))))))

    def _metric_parts(err: np.ndarray):
        pooled_mse = float(np.mean(err ** 2))
        per_trajectory: dict[str, dict[str, Any]] = {}
        trajectory_mses_for_compat: list[float] = []
        reliable_mses: list[float] = []
        reliable_weights: list[float] = []

        source_values = sorted(set(map(str, source_arr.tolist())))
        for sf in source_values:
            sf_mask = source_arr == sf
            sf_count = int(sf_mask.sum())
            pci_records = []
            for pci in sorted(set(pci_arr[sf_mask].tolist())):
                mask = sf_mask & (pci_arr == int(pci))
                n = int(mask.sum())
                if n <= 0:
                    continue
                mse = float(np.mean(err[mask] ** 2))
                pci_records.append((int(pci), n, mse))

            if not pci_records:
                continue

            trajectory_mse_for_compat = float(np.mean([x[2] for x in pci_records]))
            trajectory_mses_for_compat.append(trajectory_mse_for_compat)

            eligible_pci = [x for x in pci_records if x[1] >= min_pci]
            if not eligible_pci:
                eligible_pci = pci_records
            pci_weights = np.asarray(
                [_sqrt_cap_weight(x[1], pci_cap) for x in eligible_pci], dtype=float
            )
            pci_mses = np.asarray([x[2] for x in eligible_pci], dtype=float)
            reliable_traj_mse = float(np.average(pci_mses, weights=pci_weights))
            eligible_for_selection = sf_count >= min_traj
            traj_weight = _sqrt_cap_weight(sf_count, traj_cap) if eligible_for_selection else 0.0
            if eligible_for_selection:
                reliable_mses.append(reliable_traj_mse)
                reliable_weights.append(traj_weight)

            per_trajectory[sf] = {
                "count": sf_count,
                "mse_db2": trajectory_mse_for_compat,
                "rmse_db": float(np.sqrt(trajectory_mse_for_compat)),
                "reliability_mse_db2": reliable_traj_mse,
                "reliability_rmse_db": float(np.sqrt(reliable_traj_mse)),
                "mae_db": float(np.mean(np.abs(err[sf_mask]))),
                "bias_sim_minus_meas_db": float(np.mean(err[sf_mask])),
                "pci_group_count": int(len(pci_records)),
                "eligible_pci_group_count": int(len(eligible_pci)),
                "eligible_for_selection": bool(eligible_for_selection),
                "selection_weight": float(traj_weight),
            }

        trajectory_mean_mse_for_compat = (
            float(np.mean(trajectory_mses_for_compat))
            if trajectory_mses_for_compat
            else pooled_mse
        )

        # If all trajectory fragments are below the support threshold, do not
        # fail a sparse station. Fall back to reliability-weighted use of all
        # available trajectories and mark this in diagnostics.
        reliability_fallback = False
        if not reliable_mses:
            reliability_fallback = True
            all_mses, all_weights = [], []
            for item in per_trajectory.values():
                all_mses.append(float(item["reliability_mse_db2"]))
                w = _sqrt_cap_weight(int(item["count"]), traj_cap)
                all_weights.append(w)
                item["selection_weight"] = float(w)
            reliability_mse = (
                float(np.average(np.asarray(all_mses), weights=np.asarray(all_weights)))
                if all_mses
                else pooled_mse
            )
            reliability_weight_sum = float(np.sum(all_weights)) if all_weights else 0.0
        else:
            reliability_mse = float(
                np.average(np.asarray(reliable_mses), weights=np.asarray(reliable_weights))
            )
            reliability_weight_sum = float(np.sum(reliable_weights))

        diagnostics = {
            "eligible_trajectory_count": int(
                sum(bool(x["eligible_for_selection"]) for x in per_trajectory.values())
            ),
            "excluded_low_support_trajectory_count": int(
                sum(not bool(x["eligible_for_selection"]) for x in per_trajectory.values())
            ),
            "reliability_fallback_used": bool(reliability_fallback),
            "reliability_weight_sum": reliability_weight_sum,
        }
        return (
            pooled_mse,
            trajectory_mean_mse_for_compat,
            reliability_mse,
            per_trajectory,
            diagnostics,
        )

    candidate_rows = []
    for power_dbm in np.asarray(power_candidates_dbm, dtype=float):
        err = p0 + (float(power_dbm) - float(reference_power_dbm)) - y
        pooled_mse, trajectory_mse, reliability_mse, _, _ = _metric_parts(err)
        if reliability_mode:
            base_mse = reliability_mse
        elif trajectory_mode_compat:
            base_mse = trajectory_mse
        else:
            base_mse = pooled_mse
        score = base_mse + lam * (float(power_dbm) - anchor) ** 2
        candidate_rows.append((score, float(power_dbm)))

    best_index = int(np.nanargmin([x[0] for x in candidate_rows]))
    selection_score_db2, best_power = candidate_rows[best_index]
    pred_best = p0 + (best_power - float(reference_power_dbm))
    error = pred_best - y
    pooled_mse, trajectory_mse, reliability_mse, per_trajectory, reliability_diag = _metric_parts(error)

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

    if reliability_mode:
        selected_rmse = float(np.sqrt(reliability_mse))
    elif trajectory_mode_compat:
        selected_rmse = float(np.sqrt(trajectory_mse))
    else:
        selected_rmse = float(np.sqrt(pooled_mse))

    return {
        "rmse_db": selected_rmse,
        "pooled_rmse_db": float(np.sqrt(pooled_mse)),
        "trajectory_balanced_rmse_db": float(np.sqrt(trajectory_mse)),
        "reliability_balanced_rmse_db": float(np.sqrt(reliability_mse)),
        "selection_objective_mode": mode,
        "selection_score_db2": float(selection_score_db2),
        "power_regularization_lambda": lam,
        "power_regularization_anchor_dbm": anchor,
        "best_power_dbm": best_power,
        "mae_db": float(np.mean(np.abs(error))),
        "bias_sim_minus_meas_db": float(np.mean(error)),
        "paired_point_count": int(len(y)),
        "per_pci": per_pci,
        "per_trajectory": per_trajectory,
        "kept_measurement_indices": rows_kept,
        "measured_rsrp_dbm": y,
        "predicted_reference_dbm": p0,
        "predicted_best_power_dbm": pred_best,
        "pci": pci_arr,
        "source_file": source_arr,
        "error_db": error,
        "min_trajectory_points": min_traj,
        "trajectory_weight_cap": traj_cap,
        "min_pci_points": min_pci,
        "pci_weight_cap": pci_cap,
        **reliability_diag,
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
                objective_mode=str(search_cfg.get("objective_mode", "pooled_rmse")),
                power_regularization_lambda=float(search_cfg.get("power_regularization_lambda", 0.0)),
                power_regularization_anchor_dbm=float(search_cfg.get("power_regularization_anchor_dbm", 52.5)),
                min_trajectory_points=int(search_cfg.get("reliability_min_trajectory_points", 20)),
                trajectory_weight_cap=int(search_cfg.get("reliability_trajectory_weight_cap", 250)),
                min_pci_points=int(search_cfg.get("reliability_min_pci_points", 3)),
                pci_weight_cap=int(search_cfg.get("reliability_pci_weight_cap", 100)),
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
        "rsrp_definition": "3GPP TS 38.215 SSS-RE linear-average SS-RSRP under a narrowband path-gain approximation; 127 SSS REs from TS 38.211; sector-wide carrier power is mapped to per-RE EPRE by uniform allocation over 273*12 active subcarriers; 3276 is a TX-side allocation count, not an SS-RSRP averaging count; no station-specific SSS-RE offset is fitted",
        "selection_metric": str(search_cfg["metric"]),
    }
    (output_dir / "best_parameters.json").write_text(
        json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"best": best, "history": history_frame, "best_payload": best_payload}
