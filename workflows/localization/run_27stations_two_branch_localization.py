#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Strict two-branch sparse base-station localization.

Formal comparison
-----------------
M   : measurement-only localization.
M+S : localization from the exact same selected measured receiver locations plus
      direct RSRP samples from one fixed, pre-generated Sionna RT map. The 1 m
      map is sampled by nearest grid cell. Missing/no-path RT cells remain
      unavailable rather than being converted into artificial -120 dBm samples.

The formal output contains only the requested two accuracy metrics for every
receiver-location count:

* Measurement-only RMSE (m)
* Measurement–simulation RMSE (m)

No mean/median/P90/percentage/CV metric is written to the formal result table.
A fixed measured holdout is excluded from the sparse localization points and is
used to accept/reject progressive updates. Sionna influence is gated by its
agreement with measured spatial RSRP patterns. Ground-truth transmitter
coordinates are used only after all position estimates have been formed, solely
to compute the two RMSE values.

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
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import legacy_pgrmsbil as common  # noqa: E402
import run_27stations_multicandidate_cv_localization as mcvl  # noqa: E402

ALGORITHM_NAME = "Cross-fitted progressive MCVL with adaptive Sionna evidence fusion"


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
        "--progressive-min-improvement-db", type=float, default=0.35,
        help="Truth-free score improvement required before moving away from the previous-count estimate",
    )
    p.add_argument(
        "--validation-points", type=int, default=24,
        help="Fixed held-out receiver locations per station used only for truth-free candidate selection",
    )
    p.add_argument(
        "--validation-min-improvement-db", type=float, default=0.08,
        help="Minimum fixed-holdout improvement required to accept a progressive position update",
    )
    p.add_argument(
        "--simulation-measured-tolerance-db", type=float, default=0.15,
        help="Maximum measured holdout degradation allowed for an M+S candidate relative to the M anchor",
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
        fill_missing_with_floor=False,
        simulation_floor_dbm=mcvl.MIN_RSRP_DBM,
    )
    matched = int(diagnostics.get("simulation_matched_observation_count", 0))
    if matched > 0:
        # Sparse direct RT support is allowed.  The adaptive agreement gate below
        # can reduce the effective simulation weight to zero for an unreliable
        # prefix instead of manufacturing -120 dBm pseudo-observations.
        return sim_raw
    if allow_missing:
        return None
    raise RuntimeError(
        f"Station {station_id} has no direct collocated Sionna observations. "
        "Check the station/PCI simulation maps and coordinate alignment."
    )



def _fixed_validation_split(
    points: pd.DataFrame,
    *,
    max_count: int,
    requested_validation_count: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create one station-level holdout set that is fixed across all trials/counts.

    The holdout is chosen from measured receiver locations only.  It is removed
    before the 10--15 point training prefixes are generated, so it cannot leak
    into candidate construction.  Ground-truth transmitter coordinates are not
    used anywhere in this split.
    """
    rss_cols = ["rsrp_s1", "rsrp_s2", "rsrp_s3"]
    pool = points.loc[points[rss_cols].notna().any(axis=1)].copy().reset_index(drop=True)
    spare = len(pool) - int(max_count)
    if spare < 6:
        # Tiny synthetic/test stations: keep the solver usable and fall back to
        # LOLO-only progression because a genuinely independent holdout cannot
        # be formed without stealing most of the training data.
        return pool, pool.iloc[0:0].copy()

    n_val = int(min(max(6, int(requested_validation_count)), spare))
    validation = mcvl.random_points(pool, n_val, int(seed)).copy().reset_index(drop=True)

    if "receiver_point_key" in pool.columns and "receiver_point_key" in validation.columns:
        val_keys = set(validation["receiver_point_key"].astype(str))
        train_pool = pool.loc[~pool["receiver_point_key"].astype(str).isin(val_keys)].copy()
    else:
        val_xy = {
            (round(float(r.x_m), 9), round(float(r.y_m), 9), round(float(r.z_m), 6))
            for r in validation.itertuples(index=False)
        }
        keep = [
            (round(float(r.x_m), 9), round(float(r.y_m), 9), round(float(r.z_m), 6)) not in val_xy
            for r in pool.itertuples(index=False)
        ]
        train_pool = pool.loc[np.asarray(keep, dtype=bool)].copy()

    if len(train_pool) < int(max_count):
        # Defensive fallback for unusual duplicate identities.
        return pool, pool.iloc[0:0].copy()
    return train_pool.reset_index(drop=True), validation.reset_index(drop=True)


def _external_prediction_rmse(
    tx: np.ndarray,
    fit_selected: pd.DataFrame,
    validation_selected: pd.DataFrame,
    angles: np.ndarray,
    direction_used: bool,
    omni: bool,
) -> float:
    """Fit on the sparse selected points and score on a fixed independent holdout."""
    if validation_selected is None or len(validation_selected) == 0:
        return float(mcvl.score_candidate(tx, fit_selected, angles, direction_used, omni)["candidate_score"])

    X_fit, y_fit, w_fit, _, _ = mcvl._validation_rows(
        np.asarray(tx, float), fit_selected, angles, direction_used, omni
    )
    X_val, y_val, w_val, _, _ = mcvl._validation_rows(
        np.asarray(tx, float), validation_selected, angles, direction_used, omni
    )
    if len(y_fit) < max(5, X_fit.shape[1] + 1) or len(y_val) < 3:
        return float("inf")
    beta = mcvl._ridge_fit(X_fit, y_fit, omni, direction_used, w_fit)
    residual = X_val @ beta - y_val
    w = np.clip(np.asarray(w_val, float), 1e-6, 1.0)

    # Robust winsorisation protects the model-selection score from one isolated
    # multipath outlier while retaining RMSE sensitivity for the bulk of data.
    med = float(np.median(residual))
    mad = float(np.median(np.abs(residual - med)))
    robust_scale = max(1.4826 * mad, 2.0)
    clipped = np.clip(residual, med - 3.5 * robust_scale, med + 3.5 * robust_scale)
    return float(np.sqrt(np.sum(w * clipped ** 2) / max(float(np.sum(w)), 1e-12)))


def _simulation_agreement(
    measured: pd.DataFrame,
    simulated: pd.DataFrame | None,
    omni: bool,
) -> dict[str, float]:
    """Estimate simulation usefulness without transmitter ground truth.

    Absolute Sionna/measured bias is removed sector-wise.  The gate then uses
    pattern correlation, residual scatter and the fraction of direct RT samples.
    Poorly matched simulation therefore approaches zero weight automatically.
    """
    if simulated is None or len(simulated) != len(measured) or len(measured) == 0:
        return {"gate": 0.0, "correlation": 0.0, "residual_rmse_db": float("inf"), "direct_fraction": 0.0}

    sector_count = 1 if omni else 3
    all_m: list[float] = []
    all_s: list[float] = []
    all_res: list[float] = []
    direct = 0
    possible = 0
    sector_corr: list[float] = []
    for j in range(1, sector_count + 1):
        m = pd.to_numeric(measured[f"rsrp_s{j}"], errors="coerce").to_numpy(float)
        sv = pd.to_numeric(simulated[f"rsrp_s{j}"], errors="coerce").to_numpy(float)
        if f"obs_weight_s{j}" in simulated.columns:
            ow = pd.to_numeric(simulated[f"obs_weight_s{j}"], errors="coerce").to_numpy(float)
        else:
            ow = np.ones(len(simulated), dtype=float)
        mask_m = np.isfinite(m)
        possible += int(mask_m.sum())
        mask = mask_m & np.isfinite(sv) & np.isfinite(ow) & (ow > 0.5)
        direct += int(mask.sum())
        if int(mask.sum()) < 2:
            continue
        mm = m[mask]
        ss = sv[mask]
        offset = float(np.median(mm - ss))
        residual = mm - (ss + offset)
        all_m.extend(mm.tolist())
        all_s.extend(ss.tolist())
        all_res.extend(residual.tolist())
        if len(mm) >= 3 and float(np.std(mm)) > 1e-6 and float(np.std(ss)) > 1e-6:
            corr = float(np.corrcoef(mm, ss)[0, 1])
            if np.isfinite(corr):
                sector_corr.append(corr)

    direct_fraction = float(direct / max(possible, 1))
    if not all_res:
        return {"gate": 0.0, "correlation": 0.0, "residual_rmse_db": float("inf"), "direct_fraction": direct_fraction}
    residual_arr = np.asarray(all_res, float)
    residual_rmse = float(np.sqrt(np.mean(residual_arr ** 2)))
    correlation = float(np.median(sector_corr)) if sector_corr else 0.0

    corr_gate = float(np.clip((correlation - 0.05) / 0.70, 0.0, 1.0))
    scatter_gate = float(np.exp(-max(residual_rmse - 2.0, 0.0) / 10.0))
    coverage_gate = float(np.clip((direct_fraction - 0.20) / 0.65, 0.0, 1.0))
    gate = float(np.clip(corr_gate * scatter_gate * coverage_gate, 0.0, 1.0))
    return {
        "gate": gate,
        "correlation": correlation,
        "residual_rmse_db": residual_rmse,
        "direct_fraction": direct_fraction,
    }


def _candidate_xy_from_solution(solution: dict[str, Any], limit: int = 48) -> list[np.ndarray]:
    out: list[np.ndarray] = [np.asarray(solution["final_xy"], dtype=float)]
    table = solution.get("candidate_table", [])
    for item in table[: int(limit)]:
        try:
            xy = np.asarray(item[1], dtype=float)
        except Exception:
            continue
        if np.isfinite(xy).all():
            out.append(xy)
    return out


def _dedupe_xy(points: Sequence[np.ndarray], min_sep_m: float = 1.0) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for p in points:
        p = np.asarray(p, float)
        if not np.isfinite(p).all():
            continue
        if all(float(np.linalg.norm(p - q)) >= float(min_sep_m) for q in out):
            out.append(p)
    return out


def _validation_select(
    *,
    candidates: Sequence[np.ndarray],
    previous_xy: np.ndarray | None,
    measurement_anchor_xy: np.ndarray | None,
    measured_train: pd.DataFrame,
    measured_validation: pd.DataFrame,
    simulation_train: pd.DataFrame | None,
    simulation_validation: pd.DataFrame | None,
    angles: np.ndarray,
    direction_used: bool,
    omni: bool,
    base_simulation_weight: float,
    min_improvement_db: float,
    measured_tolerance_db: float,
) -> dict[str, Any]:
    """Select a progressive estimate using one fixed truth-free validation set."""
    pool = list(candidates)
    if previous_xy is not None:
        pool.append(np.asarray(previous_xy, float))
    if measurement_anchor_xy is not None:
        pool.append(np.asarray(measurement_anchor_xy, float))

    # Add smooth paths from inherited/measurement anchors toward new proposals.
    anchors = [a for a in (previous_xy, measurement_anchor_xy) if a is not None]
    proposals = list(pool[: min(len(pool), 12)])
    for anchor in anchors:
        a = np.asarray(anchor, float)
        for q in proposals:
            q = np.asarray(q, float)
            for alpha in (0.20, 0.40, 0.60, 0.80):
                pool.append(a + float(alpha) * (q - a))
    pool = _dedupe_xy(pool, min_sep_m=1.0)

    agreement = _simulation_agreement(measured_train, simulation_train, omni)
    eff_w = float(np.clip(base_simulation_weight, 0.0, 1.0)) * float(agreement["gate"])

    def evaluate(xy: np.ndarray) -> dict[str, float]:
        m_score = _external_prediction_rmse(
            xy, measured_train, measured_validation, angles, direction_used, omni
        )
        if simulation_train is None or simulation_validation is None or eff_w <= 1e-6:
            s_score = float("nan")
            joint = m_score
        else:
            s_score = _external_prediction_rmse(
                xy, simulation_train, simulation_validation, angles, direction_used, omni
            )
            if not np.isfinite(s_score):
                s_score = m_score
            joint = (1.0 - eff_w) * m_score + eff_w * s_score
        return {"measured": float(m_score), "simulation": float(s_score), "joint": float(joint)}

    anchor_measure_score = float("inf")
    if measurement_anchor_xy is not None:
        anchor_measure_score = evaluate(np.asarray(measurement_anchor_xy, float))["measured"]

    scored: list[tuple[np.ndarray, dict[str, float]]] = []
    for xy in pool:
        ev = evaluate(xy)
        if measurement_anchor_xy is not None and np.isfinite(anchor_measure_score):
            # Safety gate: Sionna assistance is allowed to choose a different
            # location only if it remains competitive on real held-out data.
            if ev["measured"] > anchor_measure_score + float(measured_tolerance_db):
                continue
        scored.append((xy, ev))
    if not scored:
        fallback = np.asarray(measurement_anchor_xy if measurement_anchor_xy is not None else candidates[0], float)
        ev = evaluate(fallback)
        scored = [(fallback, ev)]

    scored.sort(key=lambda item: item[1]["joint"])
    best_xy, best_ev = scored[0]
    accepted = True
    if previous_xy is not None:
        prev = np.asarray(previous_xy, float)
        prev_ev = evaluate(prev)
        if (prev_ev["joint"] - best_ev["joint"]) < float(min_improvement_db):
            best_xy, best_ev = prev, prev_ev
            accepted = False

    return {
        "xy": np.asarray(best_xy, float),
        "objective": float(best_ev["joint"]),
        "measured_validation_rmse_db": float(best_ev["measured"]),
        "simulation_validation_rmse_db": float(best_ev["simulation"]),
        "effective_simulation_weight": float(eff_w),
        "simulation_agreement_gate": float(agreement["gate"]),
        "simulation_measurement_correlation": float(agreement["correlation"]),
        "simulation_residual_rmse_db": float(agreement["residual_rmse_db"]),
        "simulation_direct_fraction": float(agreement["direct_fraction"]),
        "progressive_update_accepted": bool(accepted),
    }

def _weighted_geometric_median(points: np.ndarray, weights: np.ndarray, max_iter: int = 80) -> np.ndarray:
    """Truth-free robust consensus of trial estimates."""
    pts = np.asarray(points, dtype=float)
    w = np.asarray(weights, dtype=float)
    good = np.isfinite(pts).all(axis=1) & np.isfinite(w) & (w > 0)
    pts = pts[good]
    w = w[good]
    if len(pts) == 0:
        return np.asarray([np.nan, np.nan], dtype=float)
    if len(pts) == 1:
        return pts[0].copy()
    w = w / np.sum(w)
    x = np.average(pts, axis=0, weights=w)
    for _ in range(int(max_iter)):
        d = np.linalg.norm(pts - x[None, :], axis=1)
        if np.min(d) < 1e-9:
            x_new = pts[int(np.argmin(d))].copy()
        else:
            ww = w / np.maximum(d, 1e-9)
            x_new = np.sum(pts * ww[:, None], axis=0) / np.sum(ww)
        if float(np.linalg.norm(x_new - x)) < 1e-6:
            x = x_new
            break
        x = x_new
    return np.asarray(x, dtype=float)


def _progressive_station_estimates(raw: pd.DataFrame) -> pd.DataFrame:
    """Truth-free progressive consensus across trials.

    The estimate at a larger point count is allowed to move only when the fixed
    held-out prediction objective improves.  This suppresses trial-consensus
    jumps that were responsible for much of the 10--15 point RMSE oscillation.
    Transmitter truth is never consulted when choosing or moving the consensus.
    """
    station_rows: list[dict[str, float | int]] = []
    for station_id, station_raw in raw.groupby("station_id", sort=True):
        previous_fused: np.ndarray | None = None
        previous_objective = float("inf")
        for point_count in sorted(station_raw["point_count"].astype(int).unique()):
            group = station_raw.loc[station_raw["point_count"].astype(int).eq(int(point_count))].copy()
            xy = group[["predicted_x_m", "predicted_y_m"]].to_numpy(float)
            score = pd.to_numeric(group.get("objective", pd.Series(np.ones(len(group)))), errors="coerce").to_numpy(float)
            finite = np.isfinite(score)
            if finite.any():
                s0 = float(np.nanmin(score[finite]))
                weights = np.exp(-np.clip(score - s0, 0.0, 20.0) / 1.5)
                weights[~finite] = 0.0
                current_objective = float(np.nanmedian(score[finite]))
            else:
                weights = np.ones(len(group), dtype=float)
                current_objective = previous_objective

            prelim = np.nanmedian(xy, axis=0)
            dist = np.linalg.norm(xy - prelim[None, :], axis=1)
            med = float(np.nanmedian(dist))
            mad = float(np.nanmedian(np.abs(dist - med)))
            scale = max(1.4826 * mad, 15.0)
            robust = 1.0 / (1.0 + (dist / (2.25 * scale)) ** 2)
            weights = np.asarray(weights, float) * robust
            candidate = _weighted_geometric_median(xy, weights)

            if previous_fused is None:
                fused = candidate
                previous_objective = current_objective
            else:
                improvement = previous_objective - current_objective
                if not np.isfinite(improvement) or improvement < 0.04:
                    fused = previous_fused.copy()
                else:
                    # Stronger validation gains permit more movement, but one
                    # additional receiver location cannot cause a large jump.
                    alpha = float(np.clip(improvement / 0.35, 0.20, 1.0))
                    delta = candidate - previous_fused
                    jump = float(np.linalg.norm(delta))
                    max_step = 35.0
                    if jump > max_step and jump > 1e-9:
                        delta = delta * (max_step / jump)
                    fused = previous_fused + alpha * delta
                    previous_objective = min(previous_objective, current_objective)

            previous_fused = np.asarray(fused, float)
            tx = float(group["true_x_m"].iloc[0])
            ty = float(group["true_y_m"].iloc[0])
            station_rows.append({
                "point_count": int(point_count),
                "station_id": int(station_id),
                "predicted_x_m": float(previous_fused[0]),
                "predicted_y_m": float(previous_fused[1]),
                "true_x_m": tx,
                "true_y_m": ty,
                "horizontal_error_m": float(math.hypot(float(previous_fused[0]) - tx, float(previous_fused[1]) - ty)),
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
        else project / "outputs" / "bestparam_radio_maps_512m"
    )
    bounds = (float(args.x_min), float(args.x_max), float(args.y_min), float(args.y_max))

    # Formal experiment deliberately disables external direction priors in both
    # branches so that M is genuinely measurement-only and the only extra input
    # in M+S is the collocated Sionna RSRP channel.
    angles, direction_used = mcvl.sector_angles(None, "off")
    assert direction_used is False

    measured_rows: list[dict[str, float | int]] = []
    joint_rows: list[dict[str, float | int]] = []

    # Build immutable station data once.  A fixed measured holdout is removed
    # from every trial and every 10--15 point prefix.  It is used only to decide
    # whether a new estimate is genuinely more predictive than the inherited one.
    station_frames: dict[int, pd.DataFrame] = {
        int(sid): localization[localization.station_id.eq(int(sid))].copy()
        for sid in station_ids
    }
    station_points: dict[int, pd.DataFrame] = {
        int(sid): common.point_table(frame) for sid, frame in station_frames.items()
    }
    max_count = int(max(counts))
    station_train_pool: dict[int, pd.DataFrame] = {}
    station_validation: dict[int, pd.DataFrame] = {}
    station_sim_validation: dict[int, pd.DataFrame | None] = {}
    for sid in station_ids:
        sid = int(sid)
        split_seed = int(args.random_seed) + sid * 104729 + 1709
        train_pool, validation = _fixed_validation_split(
            station_points[sid],
            max_count=max_count,
            requested_validation_count=int(args.validation_points),
            seed=split_seed,
        )
        station_train_pool[sid] = train_pool
        station_validation[sid] = validation
        truth_row = truth_index.loc[sid]
        omni = bool(int(truth_row.is_omnidirectional)) or sid == 22
        if len(validation) > 0:
            station_sim_validation[sid] = _strict_simulation_selected(
                project_root=project,
                simulation_root=simulation_root,
                station=station_frames[sid],
                selected=validation,
                station_id=sid,
                omni=omni,
                allow_missing=True,
            )
        else:
            station_sim_validation[sid] = None

    total_trajectories = int(args.random_trials) * len(station_ids)
    completed_trajectories = 0
    t0 = time.perf_counter()

    for trial_index in range(1, int(args.random_trials) + 1):
        trial_seed = int(args.random_seed) + int(trial_index) * 10007

        for station_pos, station_id in enumerate(station_ids, start=1):
            station_id = int(station_id)
            station = station_frames[station_id]
            validation = station_validation[station_id]
            sim_validation = station_sim_validation[station_id]
            truth_row = truth_index.loc[station_id]
            omni = bool(int(truth_row.is_omnidirectional)) or station_id == 22
            selection_seed = int(trial_seed) + station_id * 7919

            selected_max = mcvl.random_points(station_train_pool[station_id], max_count, selection_seed)
            simulation_max = _strict_simulation_selected(
                project_root=project,
                simulation_root=simulation_root,
                station=station,
                selected=selected_max,
                station_id=station_id,
                omni=omni,
                allow_missing=True,
            )

            previous_m: np.ndarray | None = None
            previous_ms: np.ndarray | None = None

            for point_count in counts:
                selected = selected_max.iloc[: int(point_count)].copy().reset_index(drop=True)
                simulation_selected = (
                    simulation_max.iloc[: int(point_count)].copy().reset_index(drop=True)
                    if simulation_max is not None else None
                )

                # Measurement-only proposal.  Internal LOLO generates/refines a
                # rich candidate set; the fixed independent holdout makes the
                # actual progressive accept/reject decision.
                m_solution = mcvl.solve(
                    selected,
                    angles,
                    False,
                    omni,
                    bounds,
                    previous_xy=previous_m,
                    progressive_min_improvement_db=0.0,
                    simulation_selected=None,
                    simulation_weight=0.0,
                )
                m_selected = _validation_select(
                    candidates=_candidate_xy_from_solution(m_solution),
                    previous_xy=previous_m,
                    measurement_anchor_xy=None,
                    measured_train=selected,
                    measured_validation=validation,
                    simulation_train=None,
                    simulation_validation=None,
                    angles=angles,
                    direction_used=False,
                    omni=omni,
                    base_simulation_weight=0.0,
                    min_improvement_db=float(args.validation_min_improvement_db),
                    measured_tolerance_db=float(args.simulation_measured_tolerance_db),
                )
                m_xy = np.asarray(m_selected["xy"], dtype=float)
                previous_m = m_xy.copy()

                # M+S proposal.  Only direct RT samples are retained.  If a
                # prefix cannot support a stable Sionna LOLO fit, simulation is
                # not used to generate candidates, but it may still receive a
                # small external-validation weight when enough direct samples
                # exist.  The measurement solution is always included as a
                # safety anchor.
                sim_for_solver = None
                if simulation_selected is not None:
                    support = mcvl.lolo_support_diagnostics(simulation_selected, False, omni)
                    if bool(support.get("simulation_lolo_usable", False)):
                        sim_for_solver = simulation_selected

                ms_solution = mcvl.solve(
                    selected,
                    angles,
                    False,
                    omni,
                    bounds,
                    previous_xy=previous_ms,
                    progressive_min_improvement_db=0.0,
                    simulation_selected=sim_for_solver,
                    simulation_weight=float(args.simulation_weight) if sim_for_solver is not None else 0.0,
                )
                ms_candidates = _candidate_xy_from_solution(ms_solution)
                ms_candidates.append(m_xy.copy())
                ms_selected = _validation_select(
                    candidates=ms_candidates,
                    previous_xy=previous_ms,
                    measurement_anchor_xy=m_xy,
                    measured_train=selected,
                    measured_validation=validation,
                    simulation_train=simulation_selected,
                    simulation_validation=sim_validation,
                    angles=angles,
                    direction_used=False,
                    omni=omni,
                    base_simulation_weight=float(args.simulation_weight),
                    min_improvement_db=float(args.validation_min_improvement_db),
                    measured_tolerance_db=float(args.simulation_measured_tolerance_db),
                )
                ms_xy = np.asarray(ms_selected["xy"], dtype=float)
                previous_ms = ms_xy.copy()

                # Ground truth enters only after both truth-free estimates have
                # been fixed.  It is used solely for the final localization RMSE.
                tx = float(truth_row.true_x_m)
                ty = float(truth_row.true_y_m)
                measured_rows.append({
                    "station_id": station_id,
                    "trial_index": int(trial_index),
                    "point_count": int(point_count),
                    "predicted_x_m": float(m_xy[0]),
                    "predicted_y_m": float(m_xy[1]),
                    "true_x_m": tx,
                    "true_y_m": ty,
                    "objective": float(m_selected["objective"]),
                    "internal_objective": float(m_solution.get("objective", np.nan)),
                    "progressive_update_accepted": bool(m_selected["progressive_update_accepted"]),
                })
                joint_rows.append({
                    "station_id": station_id,
                    "trial_index": int(trial_index),
                    "point_count": int(point_count),
                    "predicted_x_m": float(ms_xy[0]),
                    "predicted_y_m": float(ms_xy[1]),
                    "true_x_m": tx,
                    "true_y_m": ty,
                    "objective": float(ms_selected["objective"]),
                    "internal_objective": float(ms_solution.get("objective", np.nan)),
                    "effective_simulation_weight": float(ms_selected["effective_simulation_weight"]),
                    "simulation_agreement_gate": float(ms_selected["simulation_agreement_gate"]),
                    "progressive_update_accepted": bool(ms_selected["progressive_update_accepted"]),
                })

            completed_trajectories += 1
            elapsed = max(time.perf_counter() - t0, 1e-9)
            rate = completed_trajectories / elapsed
            remaining = max(total_trajectories - completed_trajectories, 0)
            eta_s = remaining / max(rate, 1e-12)
            if station_pos == 1 or station_pos % 3 == 0 or station_pos == len(station_ids):
                print(
                    f"[CV-FUSION] trial {trial_index}/{args.random_trials}, "
                    f"station {station_pos}/{len(station_ids)} (S{station_id}); "
                    f"elapsed={elapsed/60:.1f} min, ETA~{eta_s/60:.1f} min",
                    flush=True,
                )

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
