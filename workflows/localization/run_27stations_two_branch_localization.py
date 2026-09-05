#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Strict two-branch sparse base-station localization.

Formal comparison
-----------------
M   : measurement-only localization.
M+S : localization from the exact same selected measured receiver locations plus
      RSRP sampled from one fixed, pre-generated Sionna RT map. The 1 m map is
      sampled by nearest grid cell; simulation-side missing/no-path cells are
      lower-censored to -120 dBm so sparse RT coverage cannot silently remove
      a measured receiver location from the paired experiment.

The formal output contains only the requested two accuracy metrics for every
receiver-location count:

* Measurement-only RMSE (m)
* Measurement–simulation RMSE (m)

No mean/median/P90/percentage/CV metric is written to the formal result table.
Ground-truth transmitter coordinates are used only after all position estimates
have been formed, solely to compute the two RMSE values.

To keep the ablation clean, the formal two-branch workflow does not load a
calibrated direction-prior CSV. Candidate generation in the measurement-only
branch uses only selected receiver coordinates and measured PCI-RSRP. The M+S
branch receives exactly those same measurements and additionally receives
collocated Sionna PCI-RSRP at those receiver locations. It never uses the
Sionna map peak, map center, map extent, or transmitter metadata as a candidate
location.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import legacy_pgrmsbil as common  # noqa: E402
import run_27stations_multicandidate_cv_localization as mcvl  # noqa: E402

ALGORITHM_NAME = "Strict two-branch progressive nested MCVL localization"


def parse_counts(text: str | None, single: int | None) -> list[int]:
    if single is not None:
        if int(single) <= 0:
            raise ValueError("--points-per-station must be positive")
        return [int(single)]
    raw = "10,11,12,13,14,15" if text is None else str(text)
    counts = sorted({int(x.strip()) for x in raw.split(",") if x.strip()})
    if not counts or min(counts) <= 0:
        raise ValueError("--point-counts must contain positive integers")
    return counts


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Strict measurement-only vs measurement+simulation localization; RMSE-only output"
    )
    p.add_argument("--project-root", type=Path, default=ROOT)
    p.add_argument("--measurements", type=Path, default=None)
    p.add_argument("--output-root", "--output-dir", dest="output_root", type=Path, default=None)
    p.add_argument("--point-counts", default=None, help="Default: 10,11,12,13,14,15")
    p.add_argument("--points-per-station", type=int, default=None, help="Run one receiver-location count")
    p.add_argument("--random-trials", type=int, default=10)
    p.add_argument("--random-seed", type=int, default=20260805)
    p.add_argument("--station-ids", default="all")
    p.add_argument("--simulation-root", type=Path, default=None)
    p.add_argument(
        "--simulation-weight", type=float, default=0.50,
        help="Fixed weight of the collocated Sionna validation channel; not tuned from ground truth",
    )
    p.add_argument(
        "--progressive-min-improvement-db", type=float, default=0.25,
        help="Truth-free score improvement required before moving away from the previous-count estimate",
    )
    p.add_argument("--x-min", type=float, default=mcvl.DEFAULT_BOUNDS[0])
    p.add_argument("--x-max", type=float, default=mcvl.DEFAULT_BOUNDS[1])
    p.add_argument("--y-min", type=float, default=mcvl.DEFAULT_BOUNDS[2])
    p.add_argument("--y-max", type=float, default=mcvl.DEFAULT_BOUNDS[3])
    # Accepted only for backwards-compatible run_pipeline invocations. The formal
    # workflow is always the strict two-branch comparison and uses no direction prior.
    p.add_argument("--simulation-mode", choices=["compare"], default="compare")
    p.add_argument("--strict-simulation-data", action="store_true", default=True)
    p.add_argument("--allow-missing-simulation", action="store_true")
    p.add_argument("--direction-prior-mode", choices=["fixed", "soft", "off"], default="off")
    p.add_argument("--directions", type=Path, default=None)
    p.add_argument("--bootstrap", type=int, default=0)
    p.add_argument("--de-maxiter", type=int, default=100)
    p.add_argument("--de-popsize", type=int, default=10)
    p.add_argument("--dpi", type=int, default=1000)
    p.add_argument("--skip-figures", action="store_true")
    p.add_argument("--keep-trial-figures", action="store_true")
    p.add_argument("--continue-on-error", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--consensus-mad-z", type=float, default=2.0)
    p.add_argument("--consensus-min-inlier-fraction", type=float, default=0.60)
    p.add_argument("--consensus-min-scale-m", type=float, default=5.0)
    p.add_argument("--skip-per-station-figures", action="store_true")
    return p.parse_args()


def _strict_simulation_selected(
    *,
    project_root: Path,
    simulation_root: Path,
    station: pd.DataFrame,
    selected: pd.DataFrame,
    station_id: int,
    omni: bool,
    allow_missing: bool,
) -> pd.DataFrame | None:
    sim_raw, _, diagnostics = mcvl.build_collocated_simulation_dataset(
        project_root=project_root,
        simulation_root=simulation_root,
        station=station,
        selected=selected,
        station_id=int(station_id),
        omni=bool(omni),
        strict=not bool(allow_missing),
        simulation_sampling="nearest",
        fill_missing_with_floor=True,
        simulation_floor_dbm=mcvl.MIN_RSRP_DBM,
    )
    support = mcvl.lolo_support_diagnostics(sim_raw, False, omni)
    minimum_observations = 5 if omni else 7
    usable = (
        int(diagnostics.get("simulation_matched_observation_count", 0)) >= minimum_observations
        and bool(support.get("simulation_lolo_usable", False))
    )
    if usable:
        return sim_raw
    if allow_missing:
        return None
    raise RuntimeError(
        f"Station {station_id} still cannot form a usable Sionna validation channel after nearest-grid "
        f"sampling and simulation-only lower-floor completion: "
        f"matched={diagnostics.get('simulation_matched_observation_count', 0)}, "
        f"valid_LOLO_folds={support.get('simulation_lolo_valid_fold_count', 0)}. "
        "This normally indicates a missing station/PCI simulation map rather than sparse RT cells."
    )


def _progressive_station_estimates(raw: pd.DataFrame) -> pd.DataFrame:
    """Truth-free progressive trajectory fusion, then trial averaging per station.

    Within each station/trial, estimates from the smallest count through the
    current count are cumulatively averaged with weight=point_count. Across
    random trials, those trajectory estimates are arithmetically averaged per
    station. Ground truth is attached only after the fused estimate is complete.
    """
    states: list[dict[str, float | int]] = []
    for (station_id, trial_index), group in raw.groupby(["station_id", "trial_index"], sort=True):
        g = group.sort_values("point_count")
        weighted_xy = np.zeros(2, dtype=float)
        weight_sum = 0.0
        for row in g.itertuples(index=False):
            xy = np.asarray([row.predicted_x_m, row.predicted_y_m], dtype=float)
            weight = float(row.point_count)
            if not np.isfinite(xy).all() or weight <= 0:
                continue
            weighted_xy += weight * xy
            weight_sum += weight
            fused = weighted_xy / weight_sum
            states.append({
                "station_id": int(station_id),
                "trial_index": int(trial_index),
                "point_count": int(row.point_count),
                "fused_x_m": float(fused[0]),
                "fused_y_m": float(fused[1]),
                "true_x_m": float(row.true_x_m),
                "true_y_m": float(row.true_y_m),
            })
    state = pd.DataFrame(states)
    station_rows: list[dict[str, float | int]] = []
    for (point_count, station_id), group in state.groupby(["point_count", "station_id"], sort=True):
        x = float(pd.to_numeric(group["fused_x_m"], errors="coerce").mean())
        y = float(pd.to_numeric(group["fused_y_m"], errors="coerce").mean())
        tx = float(group["true_x_m"].iloc[0])
        ty = float(group["true_y_m"].iloc[0])
        station_rows.append({
            "point_count": int(point_count),
            "station_id": int(station_id),
            "predicted_x_m": x,
            "predicted_y_m": y,
            "true_x_m": tx,
            "true_y_m": ty,
            "horizontal_error_m": float(math.hypot(x - tx, y - ty)),
        })
    return pd.DataFrame(station_rows)


def _rmse_by_count(station_results: pd.DataFrame) -> dict[int, float]:
    out: dict[int, float] = {}
    for point_count, group in station_results.groupby("point_count", sort=True):
        err = pd.to_numeric(group["horizontal_error_m"], errors="coerce").to_numpy(float)
        err = err[np.isfinite(err)]
        if len(err) == 0:
            raise RuntimeError(f"No finite localization errors at point_count={point_count}")
        out[int(point_count)] = float(np.sqrt(np.mean(err ** 2)))
    return out


def build_formal_rmse_table(
    measurement_station_results: pd.DataFrame,
    joint_station_results: pd.DataFrame,
    counts: Sequence[int],
) -> pd.DataFrame:
    """Return the formal table with exactly two accuracy metrics."""
    rmse_m = _rmse_by_count(measurement_station_results)
    rmse_ms = _rmse_by_count(joint_station_results)
    rows = []
    for count in counts:
        rows.append({
            "Receiver locations per station": int(count),
            "Measurement-only RMSE (m)": float(rmse_m[int(count)]),
            "Measurement–simulation RMSE (m)": float(rmse_ms[int(count)]),
        })
    return pd.DataFrame(rows)


def run_experiment(args: argparse.Namespace) -> pd.DataFrame:
    project = args.project_root.expanduser().resolve()
    counts = parse_counts(args.point_counts, args.points_per_station)
    if int(args.random_trials) <= 0:
        raise ValueError("--random-trials must be positive")
    if not (0.0 < float(args.simulation_weight) < 1.0):
        raise ValueError("--simulation-weight must be strictly between 0 and 1 for the M+S branch")

    measurement_path = common.resolve_measurement_csv(project, args.measurements)
    localization, truth = common.load_and_filter(measurement_path)
    localization = localization[
        localization["rsrp_dbm"].between(mcvl.MIN_RSRP_DBM, mcvl.MAX_RSRP_DBM, inclusive="both")
    ].copy()
    available = sorted(localization.station_id.unique().astype(int))
    station_ids = common.parse_station_ids(args.station_ids, available)
    truth_index = truth.set_index("station_id")
    simulation_root = (
        args.simulation_root.expanduser().resolve()
        if args.simulation_root is not None
        else project / "outputs" / "bestparam_dem_vs_zplane_512m"
    )
    bounds = (float(args.x_min), float(args.x_max), float(args.y_min), float(args.y_max))

    # Formal experiment deliberately disables external direction priors in both
    # branches so that M is genuinely measurement-only and the only extra input
    # in M+S is the collocated Sionna RSRP channel.
    angles, direction_used = mcvl.sector_angles(None, "off")
    assert direction_used is False

    measured_rows: list[dict[str, float | int]] = []
    joint_rows: list[dict[str, float | int]] = []

    for trial_index in range(1, int(args.random_trials) + 1):
        trial_seed = int(args.random_seed) + int(trial_index) * 10007
        previous_m: dict[int, np.ndarray] = {}
        previous_ms: dict[int, np.ndarray] = {}

        for point_count in counts:
            for station_id in station_ids:
                station = localization[localization.station_id.eq(station_id)].copy()
                truth_row = truth_index.loc[station_id]
                omni = bool(int(truth_row.is_omnidirectional)) or int(station_id) == 22
                points = common.point_table(station)
                # Identical deterministic prefix for the two branches.
                selected = mcvl.random_points(
                    points,
                    int(point_count),
                    int(trial_seed) + int(station_id) * 7919,
                )

                m_solution = mcvl.solve(
                    selected,
                    angles,
                    False,
                    omni,
                    bounds,
                    previous_xy=previous_m.get(int(station_id)),
                    progressive_min_improvement_db=float(args.progressive_min_improvement_db),
                    simulation_selected=None,
                    simulation_weight=0.0,
                )
                m_xy = np.asarray(m_solution["final_xy"], dtype=float)
                previous_m[int(station_id)] = m_xy.copy()

                simulation_selected = _strict_simulation_selected(
                    project_root=project,
                    simulation_root=simulation_root,
                    station=station,
                    selected=selected,
                    station_id=int(station_id),
                    omni=omni,
                    allow_missing=bool(args.allow_missing_simulation),
                )
                if simulation_selected is None:
                    raise RuntimeError(
                        f"Station {station_id}: M+S branch would fall back to M. "
                        "This is disabled in the formal two-branch experiment."
                    )
                ms_solution = mcvl.solve(
                    selected,
                    angles,
                    False,
                    omni,
                    bounds,
                    previous_xy=previous_ms.get(int(station_id)),
                    progressive_min_improvement_db=float(args.progressive_min_improvement_db),
                    simulation_selected=simulation_selected,
                    simulation_weight=float(args.simulation_weight),
                )
                ms_xy = np.asarray(ms_solution["final_xy"], dtype=float)
                previous_ms[int(station_id)] = ms_xy.copy()

                # Ground truth enters only here, after both estimates exist.
                tx = float(truth_row.true_x_m)
                ty = float(truth_row.true_y_m)
                measured_rows.append({
                    "station_id": int(station_id),
                    "trial_index": int(trial_index),
                    "point_count": int(point_count),
                    "predicted_x_m": float(m_xy[0]),
                    "predicted_y_m": float(m_xy[1]),
                    "true_x_m": tx,
                    "true_y_m": ty,
                })
                joint_rows.append({
                    "station_id": int(station_id),
                    "trial_index": int(trial_index),
                    "point_count": int(point_count),
                    "predicted_x_m": float(ms_xy[0]),
                    "predicted_y_m": float(ms_xy[1]),
                    "true_x_m": tx,
                    "true_y_m": ty,
                })

        print(f"Completed localization trial {trial_index}/{args.random_trials}", flush=True)

    measured_station = _progressive_station_estimates(pd.DataFrame(measured_rows))
    joint_station = _progressive_station_estimates(pd.DataFrame(joint_rows))
    return build_formal_rmse_table(measured_station, joint_station, counts)


def main() -> int:
    args = parse_args()
    project = args.project_root.expanduser().resolve()
    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root is not None
        else project / "outputs" / "localization_two_branch_rmse_only"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    table = run_experiment(args)
    # Formal output: one CSV, two metrics only. No station-error, CV, MAE,
    # median, percentile, hit-rate, consensus, or ablation-delta tables.
    out = output_root / "localization_rmse_comparison.csv"
    table.to_csv(out, index=False, encoding="utf-8-sig")
    print("\n" + table.to_string(index=False))
    print(f"\nSaved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
