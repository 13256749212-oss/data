#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v1.15.5 strict two-branch radio-map reconstruction with full leave-selected-out evaluation.

Formal experiment required by the dataset application example:
* Start from ALL eligible 1-m measured cells for the target station/PCI. No
  permanent 80/20 train-test split is made.
* Every Monte-Carlo trial uses one uniform random permutation; 1%--10% subsets
  are strict nested prefixes and never use RSRP/simulation/evaluation metrics for
  sample selection.
* At each sampling ratio p, the selected subset S_p is used for reconstruction
  and EVERY other eligible measured cell D\\S_p is used for quantitative
  evaluation. Therefore selected_count + evaluation_count == total_count for
  every trial and sampling ratio; no measured cells are silently discarded.
* Branch M (measurement-only): reconstruct from S_p only using robust k-nearest
  IDW. No Sionna RSRP value is read by this branch.
* Branch M+S (measurement+simulation): use exactly the same S_p plus one fixed,
  pre-generated Sionna RT map. The Sionna map is not recalibrated or relabelled
  inside this reconstruction workflow.
* Any fusion hyperparameter is selected only from leave-one-out predictions
  inside S_p. The evaluation measurements D\\S_p are never used to choose the
  sample order, fusion weight, gate, or fallback.
* Results are averaged across repeated nested trials (default 50). No monotonic
  projection or evaluation-label-based fallback is applied.
* The filled radio-map values are not used as RMSE targets and are not used to
  decide which measured cells are eligible. Only its grid axes/building mask are
  used to define the common spatial plotting domain.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent

from radio_reconstruction.data import read_processed_long_table
from radio_reconstruction.simulation_prior import discover_simulation_prior, load_simulation_prior
from run_reconstruction_legacy_v1151 import (
    _json_safe,
    _resample_prior_building_mask,
    basic_random_nested_ranking,
    bias,
    coverage_aware_nested_ranking,
    discover_filled_reference,
    grid_edges,
    mae,
    measured_pool_coverage_metrics,
    plot_map,
    plot_paired_reconstruction_composite,
    rmse,
    robust_affine_calibration,
    save_map_npz,
    write_ablation_analysis,
)

DEFAULT_PERCENTAGES = list(range(1, 11))


def parse_percentages(text: str) -> list[int]:
    values: list[int] = []
    for token in str(text).replace(";", ",").split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if not (1 <= value <= 100):
            raise ValueError(f"采样百分比必须位于1--100，当前得到{value}")
        if value not in values:
            values.append(value)
    if not values:
        raise ValueError("至少需要一个采样百分比")
    return sorted(values)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="v1.15.5 strict two-branch reconstruction: full leave-selected-out evaluation")
    p.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument("--measurements", type=Path, default=None)
    p.add_argument("--filled-reference-npz", type=Path, default=None)
    p.add_argument("--output-root", type=Path, default=None)
    p.add_argument("--station-id", type=int, default=3)
    p.add_argument("--pci", type=int, default=558)
    p.add_argument("--percentages", default=",".join(map(str, DEFAULT_PERCENTAGES)))
    p.add_argument("--random-seed", type=int, default=20260805)
    p.add_argument("--random-trials", type=int, default=50)
    p.add_argument("--test-fraction", type=float, default=None, help=argparse.SUPPRESS)
    p.add_argument("--test-seed", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--idw-k", type=int, default=8)
    p.add_argument("--idw-power", type=float, default=2.0)
    p.add_argument("--huber-c", type=float, default=1.5)
    p.add_argument("--residual-clip-mad", type=float, default=3.0)
    p.add_argument("--fusion-weight-grid", default="0.02,0.05,0.10,0.15,0.20,0.30,0.40,0.50,0.60,0.75",
                   help="Positive training-only CV weights for the strict measured+simulation branch. Weight 0 is intentionally excluded.")
    p.add_argument("--fusion-distance-scales", default="0,0.5,1,2",
                   help="Distance-gate tau multipliers relative to median leave-one-out nearest distance; 0 disables gating.")
    p.add_argument("--fusion-min-cv-gain-db", type=float, default=0.0,
                   help="Diagnostic only in strict M+S mode; the branch does not collapse to measurement-only when CV gain is small.")
    p.add_argument("--fusion-min-valid-points", type=int, default=6,
                   help="Minimum selected points with valid Sionna values required to fit simulation fusion.")
    # Compatibility arguments retained so old command lines do not fail.
    p.add_argument("--selection-mode", choices=["random", "coverage_aware"], default="random")
    p.add_argument(
        "--simulation-correction",
        choices=["strict_ms_fusion", "cv_safe_fusion", "robust_residual_idw", "additive_residual", "affine_residual"],
        default="strict_ms_fusion",
    )
    p.add_argument("--min-rsrp-dbm", type=float, default=-120.0)
    p.add_argument("--max-rsrp-dbm", type=float, default=-40.0)
    p.add_argument("--display-min-dbm", type=float, default=-120.0)
    p.add_argument("--display-max-dbm", type=float, default=-40.0)
    p.add_argument("--dpi", type=int, default=1000)
    p.add_argument("--skip-figures", action="store_true")
    p.add_argument("--simulation-mode", choices=["without", "with", "compare"], default="compare")
    p.add_argument("--simulation-npz", type=Path, default=None)
    return p.parse_args()


def _finite_metric(reference: np.ndarray, prediction: np.ndarray, kind: str) -> float:
    reference = np.asarray(reference, dtype=float).reshape(-1)
    prediction = np.asarray(prediction, dtype=float).reshape(-1)
    valid = np.isfinite(reference) & np.isfinite(prediction)
    if not np.any(valid):
        return float("nan")
    error = prediction[valid] - reference[valid]
    if kind == "rmse":
        return float(np.sqrt(np.mean(error ** 2)))
    if kind == "mae":
        return float(np.mean(np.abs(error)))
    if kind == "bias":
        return float(np.mean(error))
    raise ValueError(kind)


def robust_idw_predict(
    train_xy: np.ndarray,
    train_values: np.ndarray,
    query_xy: np.ndarray,
    *,
    k: int = 8,
    power: float = 2.0,
    huber_c: float = 1.5,
    chunk_size: int = 50000,
) -> np.ndarray:
    """Robust local k-NN inverse-distance interpolation.

    Distance weights are combined with a Huber-style local consistency weight
    computed from the median/MAD of the k neighbour values for each query.  This
    prevents a newly added isolated sample from taking over a large Voronoi cell,
    which is the main mechanism behind the previous 1-NN RMSE reversals.
    """
    xy = np.asarray(train_xy, dtype=float)
    values = np.asarray(train_values, dtype=float).reshape(-1)
    query = np.asarray(query_xy, dtype=float)
    valid = np.isfinite(values) & np.isfinite(xy).all(axis=1)
    xy, values = xy[valid], values[valid]
    if len(xy) == 0:
        return np.full(len(query), np.nan, dtype=float)
    k_eff = max(1, min(int(k), len(xy)))
    p = max(float(power), 0.1)
    hc = max(float(huber_c), 0.1)
    tree = cKDTree(xy)
    out = np.full(len(query), np.nan, dtype=float)

    for start in range(0, len(query), int(chunk_size)):
        stop = min(start + int(chunk_size), len(query))
        d, idx = tree.query(query[start:stop], k=k_eff)
        d = np.asarray(d, dtype=float)
        idx = np.asarray(idx, dtype=int)
        if k_eff == 1:
            d = d[:, None]
            idx = idx[:, None]
        neigh = values[idx]

        # Exact co-located point: preserve the measured value exactly.
        exact = d <= 1e-9
        has_exact = np.any(exact, axis=1)

        med = np.nanmedian(neigh, axis=1, keepdims=True)
        mad = np.nanmedian(np.abs(neigh - med), axis=1, keepdims=True)
        scale = np.maximum(1.4826 * mad, 1.0)
        deviation = np.abs(neigh - med)
        cutoff = hc * scale
        robust_w = np.minimum(1.0, cutoff / np.maximum(deviation, 1e-12))
        dist_w = 1.0 / np.maximum(d, 0.25) ** p
        w = dist_w * robust_w
        denom = np.sum(w, axis=1)
        pred = np.sum(w * neigh, axis=1) / np.maximum(denom, 1e-12)

        if np.any(has_exact):
            first_exact = np.argmax(exact, axis=1)
            rows = np.flatnonzero(has_exact)
            pred[rows] = neigh[rows, first_exact[rows]]
        out[start:stop] = pred
    return out


def _mad_clip(values: np.ndarray, z: float = 3.0) -> tuple[np.ndarray, dict[str, float]]:
    arr = np.asarray(values, dtype=float).copy()
    finite = np.isfinite(arr)
    if not np.any(finite):
        return arr, {"median": float("nan"), "mad": float("nan"), "clip_low": float("nan"), "clip_high": float("nan")}
    med = float(np.median(arr[finite]))
    mad = float(np.median(np.abs(arr[finite] - med)))
    robust_sigma = max(1.4826 * mad, 1.0)
    radius = max(float(z), 0.0) * robust_sigma
    lo, hi = med - radius, med + radius
    arr[finite] = np.clip(arr[finite], lo, hi)
    return arr, {"median": med, "mad": mad, "clip_low": lo, "clip_high": hi}


def simulation_residual_robust_idw(
    *,
    prior,
    train_xy: np.ndarray,
    train_y: np.ndarray,
    query_xy: np.ndarray,
    baseline_prediction: np.ndarray,
    k: int = 8,
    power: float = 2.0,
    huber_c: float = 1.5,
    residual_clip_mad: float = 3.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fixed Sionna trend + robust-IDW interpolation of measured residuals."""
    train_xy = np.asarray(train_xy, dtype=float)
    train_y = np.asarray(train_y, dtype=float).reshape(-1)
    query_xy = np.asarray(query_xy, dtype=float)
    baseline = np.asarray(baseline_prediction, dtype=float).reshape(-1)

    prior_train = np.asarray(prior.sample(train_xy), dtype=float).reshape(-1)
    valid_train = np.isfinite(prior_train) & np.isfinite(train_y)
    if not np.any(valid_train):
        return baseline.copy(), {
            "simulation_prior_training_count": 0,
            "simulation_fallback_fraction": 1.0,
            "simulation_correction_mode": "robust_residual_idw",
        }

    residual = train_y[valid_train] - prior_train[valid_train]
    clipped_residual, clip_diag = _mad_clip(residual, z=float(residual_clip_mad))
    prior_query = np.asarray(prior.sample(query_xy), dtype=float).reshape(-1)
    residual_query = robust_idw_predict(
        train_xy[valid_train], clipped_residual, query_xy,
        k=k, power=power, huber_c=huber_c,
    )
    corrected = prior_query + residual_query
    fallback = ~np.isfinite(corrected)
    corrected[fallback] = baseline[fallback]
    corrected = np.clip(corrected, -120.0, -40.0)

    return corrected, {
        "simulation_prior_training_count": int(valid_train.sum()),
        "simulation_prior_training_fraction": float(valid_train.mean()),
        "simulation_fallback_fraction": float(np.mean(fallback)),
        "simulation_correction_mode": "robust_residual_idw",
        "training_residual_median_db": float(np.median(residual)),
        "training_residual_mad_db": float(np.median(np.abs(residual - np.median(residual)))),
        "residual_clip_low_db": float(clip_diag["clip_low"]),
        "residual_clip_high_db": float(clip_diag["clip_high"]),
        "idw_k": int(k),
        "idw_power": float(power),
        "huber_c": float(huber_c),
    }



def _parse_float_grid(text: str, *, name: str, minimum: float | None = None, maximum: float | None = None) -> list[float]:
    values: list[float] = []
    for token in str(text).replace(";", ",").split(","):
        token = token.strip()
        if not token:
            continue
        value = float(token)
        if minimum is not None and value < minimum:
            raise ValueError(f"{name} contains {value}, below minimum {minimum}")
        if maximum is not None and value > maximum:
            raise ValueError(f"{name} contains {value}, above maximum {maximum}")
        if value not in values:
            values.append(value)
    if not values:
        raise ValueError(f"{name} cannot be empty")
    return sorted(values)


def robust_idw_loo_predict(
    train_xy: np.ndarray,
    train_values: np.ndarray,
    *,
    k: int = 8,
    power: float = 2.0,
    huber_c: float = 1.5,
) -> np.ndarray:
    """Leave-one-out robust IDW predictions at the training coordinates.

    This is used only for training-internal model selection.  The held-out test
    set is never accessed.  A vectorized neighbour query avoids rebuilding one
    KD-tree per left-out point.
    """
    xy = np.asarray(train_xy, dtype=float)
    values = np.asarray(train_values, dtype=float).reshape(-1)
    valid = np.isfinite(values) & np.isfinite(xy).all(axis=1)
    out = np.full(len(values), np.nan, dtype=float)
    valid_idx = np.flatnonzero(valid)
    if len(valid_idx) < 2:
        return out

    xyv = xy[valid_idx]
    vv = values[valid_idx]
    n = len(xyv)
    k_eff = max(1, min(int(k), n - 1))
    k_query = min(n, k_eff + 1)
    p = max(float(power), 0.1)
    hc = max(float(huber_c), 0.1)
    tree = cKDTree(xyv)
    d_all, idx_all = tree.query(xyv, k=k_query)
    d_all = np.asarray(d_all, dtype=float)
    idx_all = np.asarray(idx_all, dtype=int)
    if k_query == 1:
        d_all = d_all[:, None]
        idx_all = idx_all[:, None]

    pred = np.full(n, np.nan, dtype=float)
    for row in range(n):
        keep = idx_all[row] != row
        ids = idx_all[row][keep][:k_eff]
        ds = d_all[row][keep][:k_eff]
        if len(ids) == 0:
            continue
        neigh = vv[ids]
        med = float(np.nanmedian(neigh))
        mad = float(np.nanmedian(np.abs(neigh - med)))
        scale = max(1.4826 * mad, 1.0)
        deviation = np.abs(neigh - med)
        cutoff = hc * scale
        robust_w = np.minimum(1.0, cutoff / np.maximum(deviation, 1e-12))
        dist_w = 1.0 / np.maximum(ds, 0.25) ** p
        w = dist_w * robust_w
        denom = float(np.sum(w))
        if denom > 0.0:
            pred[row] = float(np.sum(w * neigh) / denom)
    out[valid_idx] = pred
    return out


def _nearest_other_distance(train_xy: np.ndarray) -> np.ndarray:
    xy = np.asarray(train_xy, dtype=float)
    out = np.full(len(xy), np.nan, dtype=float)
    valid = np.isfinite(xy).all(axis=1)
    valid_idx = np.flatnonzero(valid)
    if len(valid_idx) < 2:
        return out
    tree = cKDTree(xy[valid_idx])
    d, _ = tree.query(xy[valid_idx], k=2)
    d = np.asarray(d, dtype=float)
    out[valid_idx] = d[:, 1]
    return out


def _nearest_training_distance(train_xy: np.ndarray, query_xy: np.ndarray) -> np.ndarray:
    xy = np.asarray(train_xy, dtype=float)
    query = np.asarray(query_xy, dtype=float)
    valid = np.isfinite(xy).all(axis=1)
    if not np.any(valid):
        return np.full(len(query), np.nan, dtype=float)
    tree = cKDTree(xy[valid])
    d, _ = tree.query(query, k=1)
    return np.asarray(d, dtype=float).reshape(-1)


def _simulation_residual_candidate(
    *,
    prior,
    train_xy: np.ndarray,
    train_y: np.ndarray,
    query_xy: np.ndarray,
    baseline_prediction: np.ndarray,
    k: int,
    power: float,
    huber_c: float,
    residual_clip_mad: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build the full-strength Sionna + residual-IDW candidate.

    This candidate is *not* used directly by the new default method.  It is one
    branch of a convex ensemble whose weight is selected by training-only LOO CV.
    """
    return simulation_residual_robust_idw(
        prior=prior,
        train_xy=train_xy,
        train_y=train_y,
        query_xy=query_xy,
        baseline_prediction=baseline_prediction,
        k=k,
        power=power,
        huber_c=huber_c,
        residual_clip_mad=residual_clip_mad,
    )


def simulation_cv_safe_fusion(
    *,
    prior,
    train_xy: np.ndarray,
    train_y: np.ndarray,
    query_xy: np.ndarray,
    baseline_prediction: np.ndarray,
    k: int = 8,
    power: float = 2.0,
    huber_c: float = 1.5,
    residual_clip_mad: float = 3.0,
    fusion_weight_grid: list[float] | tuple[float, ...] = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.75, 1.0),
    fusion_distance_scales: list[float] | tuple[float, ...] = (0.0, 0.5, 1.0, 2.0),
    min_cv_gain_db: float = 0.05,
    min_valid_points: int = 6,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Cross-validated shrinkage fusion of measured IDW and Sionna residual IDW.

    The key correction over v1.15.2 is that simulation no longer has forced
    weight 1.  The measurement-only prediction is an explicit nested submodel
    (fusion weight 0).  The current sparse training subset alone determines
    whether and how strongly the Sionna branch is used.

    For each selected training point, leave-one-out predictions are formed for
    both branches.  A small deterministic grid search selects a convex weight
    and an optional distance gate.  If the best LOO gain is below
    ``min_cv_gain_db``, the method safely falls back to measurement-only.
    """
    train_xy = np.asarray(train_xy, dtype=float)
    train_y = np.asarray(train_y, dtype=float).reshape(-1)
    query_xy = np.asarray(query_xy, dtype=float)
    baseline = np.asarray(baseline_prediction, dtype=float).reshape(-1)

    weight_grid = sorted({float(v) for v in fusion_weight_grid if 0.0 <= float(v) <= 1.0})
    if 0.0 not in weight_grid:
        weight_grid = [0.0] + weight_grid
    distance_scales = sorted({max(0.0, float(v)) for v in fusion_distance_scales})
    if not distance_scales:
        distance_scales = [0.0]

    prior_train = np.asarray(prior.sample(train_xy), dtype=float).reshape(-1)
    valid_sim = np.isfinite(prior_train) & np.isfinite(train_y) & np.isfinite(train_xy).all(axis=1)
    valid_count = int(valid_sim.sum())
    base_diag: dict[str, Any] = {
        "simulation_prior_training_count": valid_count,
        "simulation_prior_training_fraction": float(valid_count / max(len(train_y), 1)),
        "simulation_correction_mode": "cv_safe_fusion",
        "fusion_weight": 0.0,
        "fusion_distance_scale_multiplier": 0.0,
        "fusion_tau_m": 0.0,
        "fusion_cv_rmse_baseline_db": float("nan"),
        "fusion_cv_rmse_candidate_db": float("nan"),
        "fusion_cv_rmse_selected_db": float("nan"),
        "fusion_cv_gain_db": 0.0,
        "fusion_activated": False,
        "fusion_reason": "insufficient_valid_simulation_points",
    }
    if valid_count < max(3, int(min_valid_points)) or len(train_y) < 3:
        base_diag["simulation_fallback_fraction"] = 1.0
        base_diag["simulation_active_fraction"] = 0.0
        return baseline.copy(), base_diag

    # Training-only leave-one-out baseline predictions.
    baseline_loo = robust_idw_loo_predict(
        train_xy, train_y, k=k, power=power, huber_c=huber_c,
    )

    # Training-only leave-one-out Sionna residual candidate.  Raw residuals are
    # used in LOO selection to avoid clipping thresholds that depend on the
    # held-out training target.  Robust-IDW already limits local outlier impact.
    vxy = train_xy[valid_sim]
    vy = train_y[valid_sim]
    vsim = prior_train[valid_sim]
    residual = vy - vsim
    residual_loo = robust_idw_loo_predict(
        vxy, residual, k=k, power=power, huber_c=huber_c,
    )
    candidate_loo = vsim + residual_loo
    baseline_loo_valid = baseline_loo[valid_sim]
    nearest_other = _nearest_other_distance(train_xy)[valid_sim]

    cv_mask = np.isfinite(vy) & np.isfinite(baseline_loo_valid) & np.isfinite(candidate_loo) & np.isfinite(nearest_other)
    if int(cv_mask.sum()) < max(3, int(min_valid_points) - 1):
        base_diag["fusion_reason"] = "insufficient_valid_loo_predictions"
        base_diag["simulation_fallback_fraction"] = 1.0
        base_diag["simulation_active_fraction"] = 0.0
        return baseline.copy(), base_diag

    y_cv = vy[cv_mask]
    b_cv = baseline_loo_valid[cv_mask]
    s_cv = candidate_loo[cv_mask]
    d_cv = nearest_other[cv_mask]
    baseline_cv_rmse = _finite_metric(y_cv, b_cv, "rmse")
    candidate_cv_rmse = _finite_metric(y_cv, s_cv, "rmse")
    finite_d = d_cv[np.isfinite(d_cv) & (d_cv > 0)]
    median_nn = float(np.median(finite_d)) if len(finite_d) else 1.0
    median_nn = max(median_nn, 0.5)

    best = {
        "rmse": baseline_cv_rmse,
        "weight": 0.0,
        "scale": 0.0,
        "tau": 0.0,
    }
    for scale in distance_scales:
        tau = float(scale) * median_nn
        if tau <= 0.0:
            gate = np.ones_like(d_cv, dtype=float)
        else:
            gate = d_cv / np.maximum(d_cv + tau, 1e-12)
        for weight in weight_grid:
            alpha = float(weight) * gate
            pred = b_cv + alpha * (s_cv - b_cv)
            score = _finite_metric(y_cv, pred, "rmse")
            # Tie-break toward smaller simulation weight, then simpler no-gate
            # solutions.  This makes sparse stages conservative.
            key = (score, float(weight), float(scale))
            best_key = (float(best["rmse"]), float(best["weight"]), float(best["scale"]))
            if key < best_key:
                best = {"rmse": score, "weight": float(weight), "scale": float(scale), "tau": tau}

    cv_gain = float(baseline_cv_rmse - float(best["rmse"]))
    if (not np.isfinite(cv_gain)) or cv_gain < max(0.0, float(min_cv_gain_db)):
        best = {"rmse": baseline_cv_rmse, "weight": 0.0, "scale": 0.0, "tau": 0.0}
        cv_gain = 0.0
        reason = "training_cv_gain_below_threshold"
    else:
        reason = "training_cv_supports_simulation"

    # Fit the full-strength candidate on all selected training points.  Only now
    # is MAD clipping allowed because there is no held-out training label inside
    # this final fit.
    candidate_query, candidate_diag = _simulation_residual_candidate(
        prior=prior,
        train_xy=train_xy,
        train_y=train_y,
        query_xy=query_xy,
        baseline_prediction=baseline,
        k=k,
        power=power,
        huber_c=huber_c,
        residual_clip_mad=residual_clip_mad,
    )

    query_distance = _nearest_training_distance(train_xy, query_xy)
    tau = float(best["tau"])
    if tau <= 0.0:
        gate_q = np.ones(len(query_xy), dtype=float)
    else:
        gate_q = query_distance / np.maximum(query_distance + tau, 1e-12)
    alpha_q = float(best["weight"]) * gate_q
    alpha_q[~np.isfinite(candidate_query)] = 0.0
    alpha_q[~np.isfinite(baseline)] = 1.0
    prediction = baseline + alpha_q * (candidate_query - baseline)
    fallback = ~np.isfinite(prediction)
    prediction[fallback] = baseline[fallback]
    prediction = np.clip(prediction, -120.0, -40.0)

    # Correlation of the branch innovation with the baseline LOO error is a useful
    # diagnostic only; it does not decide the test result.
    innovation = s_cv - b_cv
    target_residual = y_cv - b_cv
    corr = float("nan")
    if len(innovation) >= 3 and np.nanstd(innovation) > 1e-9 and np.nanstd(target_residual) > 1e-9:
        corr = float(np.corrcoef(innovation, target_residual)[0, 1])

    diag = {
        **candidate_diag,
        **base_diag,
        "simulation_correction_mode": "cv_safe_fusion",
        "simulation_fallback_fraction": float(np.mean(fallback)) if len(fallback) else 0.0,
        "fusion_weight": float(best["weight"]),
        "fusion_distance_scale_multiplier": float(best["scale"]),
        "fusion_tau_m": float(tau),
        "fusion_cv_point_count": int(cv_mask.sum()),
        "fusion_cv_rmse_baseline_db": float(baseline_cv_rmse),
        "fusion_cv_rmse_candidate_db": float(candidate_cv_rmse),
        "fusion_cv_rmse_selected_db": float(best["rmse"]),
        "fusion_cv_gain_db": float(cv_gain),
        "fusion_activated": bool(float(best["weight"]) > 0.0),
        "fusion_reason": reason,
        "fusion_innovation_error_correlation": corr,
        "fusion_median_loo_nn_distance_m": float(median_nn),
        "simulation_active_fraction": float(np.mean(alpha_q > 1e-12)) if len(alpha_q) else 0.0,
        "simulation_mean_effective_weight": float(np.mean(alpha_q[np.isfinite(alpha_q)])) if np.any(np.isfinite(alpha_q)) else 0.0,
        "simulation_max_effective_weight": float(np.max(alpha_q[np.isfinite(alpha_q)])) if np.any(np.isfinite(alpha_q)) else 0.0,
    }
    return prediction, diag


def simulation_strict_ms_fusion(
    *,
    prior,
    train_xy: np.ndarray,
    train_y: np.ndarray,
    query_xy: np.ndarray,
    baseline_prediction: np.ndarray,
    k: int = 8,
    power: float = 2.0,
    huber_c: float = 1.5,
    residual_clip_mad: float = 3.0,
    fusion_weight_grid: list[float] | tuple[float, ...] = (0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.75),
    fusion_distance_scales: list[float] | tuple[float, ...] = (0.0, 0.5, 1.0, 2.0),
    min_valid_points: int = 6,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Strict measurement+simulation reconstruction.

    Input contract:
      M branch   = selected measured points only.
      M+S branch = the SAME selected measured points + one fixed Sionna RT map.

    The Sionna map is treated as an already-generated auxiliary spatial field.
    This function never calibrates/relabels Sionna and never reads held-out test
    labels.  Unlike ``simulation_cv_safe_fusion``, weight 0 is forbidden: the
    M+S branch therefore remains a genuine two-source reconstruction instead of
    silently collapsing into the M branch in difficult trials.

    Positive fusion strength and the optional distance gate are selected only
    from leave-one-out predictions of the currently selected measured points.
    """
    train_xy = np.asarray(train_xy, dtype=float)
    train_y = np.asarray(train_y, dtype=float).reshape(-1)
    query_xy = np.asarray(query_xy, dtype=float)
    baseline = np.asarray(baseline_prediction, dtype=float).reshape(-1)

    weight_grid = sorted({float(v) for v in fusion_weight_grid if 0.0 < float(v) <= 1.0})
    if not weight_grid:
        raise ValueError("strict M+S reconstruction requires at least one positive fusion weight")
    distance_scales = sorted({max(0.0, float(v)) for v in fusion_distance_scales}) or [0.0]

    prior_train = np.asarray(prior.sample(train_xy), dtype=float).reshape(-1)
    valid_sim = np.isfinite(prior_train) & np.isfinite(train_y) & np.isfinite(train_xy).all(axis=1)
    valid_count = int(valid_sim.sum())
    if valid_count < max(3, int(min_valid_points)):
        raise ValueError(
            f"strict M+S reconstruction needs >= {max(3, int(min_valid_points))} selected measured points "
            f"with valid Sionna values, but only {valid_count} are available"
        )

    # LOO measured-only baseline, using selected measured points only.
    baseline_loo = robust_idw_loo_predict(train_xy, train_y, k=k, power=power, huber_c=huber_c)

    # LOO simulation-assisted candidate.  Simulation is fixed; only residuals
    # computed from the selected measured points are interpolated.
    vxy = train_xy[valid_sim]
    vy = train_y[valid_sim]
    vsim = prior_train[valid_sim]
    residual = vy - vsim
    residual_loo = robust_idw_loo_predict(vxy, residual, k=k, power=power, huber_c=huber_c)
    candidate_loo = vsim + residual_loo
    baseline_loo_valid = baseline_loo[valid_sim]
    nearest_other = _nearest_other_distance(train_xy)[valid_sim]

    cv_mask = np.isfinite(vy) & np.isfinite(baseline_loo_valid) & np.isfinite(candidate_loo) & np.isfinite(nearest_other)
    if int(cv_mask.sum()) < max(3, int(min_valid_points) - 1):
        raise ValueError("strict M+S reconstruction has too few valid training-only LOO predictions")

    y_cv = vy[cv_mask]
    b_cv = baseline_loo_valid[cv_mask]
    s_cv = candidate_loo[cv_mask]
    d_cv = nearest_other[cv_mask]
    baseline_cv_rmse = _finite_metric(y_cv, b_cv, "rmse")
    candidate_cv_rmse = _finite_metric(y_cv, s_cv, "rmse")
    finite_d = d_cv[np.isfinite(d_cv) & (d_cv > 0)]
    median_nn = max(float(np.median(finite_d)) if len(finite_d) else 1.0, 0.5)

    # Strict branch: search POSITIVE weights only.  Test labels never participate.
    best = None
    for scale in distance_scales:
        tau = float(scale) * median_nn
        gate = np.ones_like(d_cv, dtype=float) if tau <= 0.0 else d_cv / np.maximum(d_cv + tau, 1e-12)
        for weight in weight_grid:
            alpha = float(weight) * gate
            pred = b_cv + alpha * (s_cv - b_cv)
            score = _finite_metric(y_cv, pred, "rmse")
            key = (score, float(weight), float(scale))
            if best is None or key < (float(best["rmse"]), float(best["weight"]), float(best["scale"])):
                best = {"rmse": score, "weight": float(weight), "scale": float(scale), "tau": tau}
    assert best is not None

    candidate_query, candidate_diag = _simulation_residual_candidate(
        prior=prior,
        train_xy=train_xy,
        train_y=train_y,
        query_xy=query_xy,
        baseline_prediction=baseline,
        k=k,
        power=power,
        huber_c=huber_c,
        residual_clip_mad=residual_clip_mad,
    )

    query_distance = _nearest_training_distance(train_xy, query_xy)
    tau = float(best["tau"])
    gate_q = np.ones(len(query_xy), dtype=float) if tau <= 0.0 else query_distance / np.maximum(query_distance + tau, 1e-12)
    alpha_q = float(best["weight"]) * gate_q
    # If Sionna is locally unavailable, the M+S method must use the measured
    # reconstruction at that grid cell rather than inventing a simulation value.
    alpha_q[~np.isfinite(candidate_query)] = 0.0
    alpha_q[~np.isfinite(baseline)] = 1.0
    prediction = baseline + alpha_q * (candidate_query - baseline)
    fallback = ~np.isfinite(prediction)
    prediction[fallback] = baseline[fallback]
    prediction = np.clip(prediction, -120.0, -40.0)

    innovation = s_cv - b_cv
    target_residual = y_cv - b_cv
    corr = float("nan")
    if len(innovation) >= 3 and np.nanstd(innovation) > 1e-9 and np.nanstd(target_residual) > 1e-9:
        corr = float(np.corrcoef(innovation, target_residual)[0, 1])

    cv_gain = float(baseline_cv_rmse - float(best["rmse"]))
    diag = {
        **candidate_diag,
        "simulation_correction_mode": "strict_ms_fusion",
        "simulation_prior_training_count": valid_count,
        "simulation_prior_training_fraction": float(valid_count / max(len(train_y), 1)),
        "fusion_weight": float(best["weight"]),
        "fusion_distance_scale_multiplier": float(best["scale"]),
        "fusion_tau_m": float(tau),
        "fusion_cv_point_count": int(cv_mask.sum()),
        "fusion_cv_rmse_baseline_db": float(baseline_cv_rmse),
        "fusion_cv_rmse_candidate_db": float(candidate_cv_rmse),
        "fusion_cv_rmse_selected_db": float(best["rmse"]),
        "fusion_cv_gain_db": float(cv_gain),
        "fusion_activated": True,
        "fusion_reason": "strict_measured_plus_fixed_sionna",
        "fusion_innovation_error_correlation": corr,
        "fusion_median_loo_nn_distance_m": float(median_nn),
        "simulation_fallback_fraction": float(np.mean(fallback)) if len(fallback) else 0.0,
        "simulation_active_fraction": float(np.mean(alpha_q > 1e-12)) if len(alpha_q) else 0.0,
        "simulation_mean_effective_weight": float(np.mean(alpha_q[np.isfinite(alpha_q)])) if np.any(np.isfinite(alpha_q)) else 0.0,
        "simulation_max_effective_weight": float(np.max(alpha_q[np.isfinite(alpha_q)])) if np.any(np.isfinite(alpha_q)) else 0.0,
    }
    return prediction, diag

def simulation_residual_nearest_neighbor(
    *, prior, train_xy: np.ndarray, train_y: np.ndarray, query_xy: np.ndarray,
    baseline_prediction: np.ndarray, correction_mode: str = "additive_residual",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Backward-compatible API used by the existing test suite.

    v1.15.2 intentionally routes this historical function name to the robust-IDW
    residual implementation.  For a constant residual field the result is exactly
    identical to the former nearest-neighbour implementation.
    """
    return simulation_residual_robust_idw(
        prior=prior, train_xy=train_xy, train_y=train_y,
        query_xy=query_xy, baseline_prediction=baseline_prediction,
        k=8, power=2.0, huber_c=1.5, residual_clip_mad=3.0,
    )


def prepare_all_measured_cells(
    measurements_csv: Path,
    station_id: int,
    pci: int,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    building_mask: np.ndarray,
    min_rsrp_dbm: float,
    max_rsrp_dbm: float,
) -> pd.DataFrame:
    """Prepare every eligible measured 1-m cell without consulting map RSRP values.

    Eligibility is based only on the target station/PCI, measured-RSRP range,
    common map bounds, 1-m cell aggregation, and the building geometry mask.
    In particular, no finite/non-finite value from the filled map or Sionna map
    is used to include or exclude a measured cell.
    """
    frame = read_processed_long_table(measurements_csv)
    selected = frame.loc[
        frame["station_id"].eq(int(station_id))
        & frame["pci"].eq(int(pci))
        & frame["measured_rsrp_dbm"].between(float(min_rsrp_dbm), float(max_rsrp_dbm), inclusive="both")
    ].copy()
    if selected.empty:
        raise ValueError(f"station={station_id}, PCI={pci}没有可用实测记录")

    x_min, x_max, dx = grid_edges(x_axis)
    y_min, y_max, dy = grid_edges(y_axis)
    selected = selected.loc[
        selected["blender_x"].ge(x_min) & selected["blender_x"].lt(x_max)
        & selected["blender_y"].ge(y_min) & selected["blender_y"].lt(y_max)
    ].copy()
    if selected.empty:
        raise ValueError("重构地图窗口内没有实测点")

    selected["ix"] = np.floor((selected["blender_x"] - x_min) / dx).astype(int)
    selected["iy"] = np.floor((selected["blender_y"] - y_min) / dy).astype(int)
    nx, ny = len(x_axis), len(y_axis)
    selected = selected.loc[selected["ix"].between(0, nx - 1) & selected["iy"].between(0, ny - 1)].copy()

    grouped = (
        selected.groupby(["station_id", "pci", "ix", "iy"], as_index=False)
        .agg(
            x_m=("blender_x", "median"),
            y_m=("blender_y", "median"),
            measured_rsrp_dbm=("measured_rsrp_dbm", "median"),
            raw_record_count=("measured_rsrp_dbm", "size"),
        )
        .sort_values(["iy", "ix"])
        .reset_index(drop=True)
    )
    iy = grouped["iy"].to_numpy(dtype=int)
    ix = grouped["ix"].to_numpy(dtype=int)
    valid = ~np.asarray(building_mask, dtype=bool)[iy, ix]
    grouped = grouped.loc[valid].copy().reset_index(drop=True)
    if grouped.empty:
        raise ValueError("地图范围内没有有效室外实测栅格")
    grouped["measurement_point_id"] = np.arange(len(grouped), dtype=int)
    return grouped


def _trial_ranking(train_pool: pd.DataFrame, seed: int, selection_mode: str, max_count: int, *, x_axis=None, y_axis=None, mask=None) -> pd.DataFrame:
    if selection_mode == "coverage_aware":
        # Retained only for backwards compatibility. Formal v1.15.2 results use random.
        return coverage_aware_nested_ranking(train_pool, x_axis, y_axis, mask, max_count=max_count, seed=seed)
    return basic_random_nested_ranking(train_pool, max_count=max_count, seed=seed)


def _aggregate_trial_metrics(trials: pd.DataFrame, percentages: list[int]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for pct in percentages:
        sub = trials.loc[trials["sampling_percent"].eq(int(pct))]
        row: dict[str, Any] = {
            "sampling_percent": int(pct),
            "selected_measured_point_count": int(sub["selected_measured_point_count"].iloc[0]),
            "random_trial_count": int(sub["trial_id"].nunique()),
            "evaluation_measured_point_count": int(sub["evaluation_measured_point_count"].iloc[0]),
            "all_measured_accounted_for": bool(sub["all_measured_accounted_for"].all()),
        }
        for variant, suffix in [("without_simulation", "without_simulation"), ("with_simulation", "with_simulation")]:
            v = sub.loc[sub["variant"].eq(variant)]
            if v.empty:
                continue
            for metric in ("rmse_db", "mae_db", "bias_db"):
                vals = pd.to_numeric(v[metric], errors="coerce")
                row[f"{metric}_{suffix}"] = float(vals.mean())
                row[f"{metric}_std_{suffix}"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
                row[f"{metric}_median_{suffix}"] = float(vals.median())
            row[f"rmse_db_p90_{suffix}"] = float(pd.to_numeric(v["rmse_db"], errors="coerce").quantile(0.90))
            if variant == "with_simulation":
                for diag_col in (
                    "fusion_weight", "fusion_cv_gain_db", "fusion_cv_rmse_baseline_db",
                    "fusion_cv_rmse_selected_db", "simulation_mean_effective_weight",
                    "simulation_active_fraction", "fusion_innovation_error_correlation",
                ):
                    if diag_col in v.columns:
                        vals = pd.to_numeric(v[diag_col], errors="coerce")
                        row[f"{diag_col}_mean"] = float(vals.mean())
                        row[f"{diag_col}_median"] = float(vals.median())
                if "fusion_activated" in v.columns:
                    active = v["fusion_activated"].astype(bool)
                    row["fusion_activation_rate"] = float(active.mean())
        if "rmse_db_without_simulation" in row and "rmse_db_with_simulation" in row:
            row["rmse_gain_with_simulation_db"] = row["rmse_db_without_simulation"] - row["rmse_db_with_simulation"]
            row["simulation_improved_mean_rmse"] = bool(row["rmse_gain_with_simulation_db"] > 0.0)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("sampling_percent").reset_index(drop=True)


def _write_trend_audit(path: Path, paired: pd.DataFrame) -> None:
    lines = ["# Reconstruction trend audit (v1.15.5 full leave-selected-out protocol)", ""]
    labels = [
        ("without_simulation", "Measurement-only robust IDW"),
        ("with_simulation", "Measurement + fixed Sionna RT"),
    ]
    for suffix, label in labels:
        col = f"rmse_db_{suffix}"
        if col not in paired.columns:
            continue
        vals = paired[col].to_numpy(dtype=float)
        diffs = np.diff(vals)
        violations = np.flatnonzero(diffs > 0)
        lines.append(f"## {label}")
        lines.append(f"Mean RMSE start -> end: {vals[0]:.4f} -> {vals[-1]:.4f} dB")
        lines.append(f"Local increases in mean curve: {len(violations)}")
        if len(violations):
            for i in violations:
                lines.append(f"- {int(paired.iloc[i]['sampling_percent'])}% -> {int(paired.iloc[i+1]['sampling_percent'])}%: {diffs[i]:+.4f} dB")
        else:
            lines.append("- none")
        lines.append("")

    if "rmse_gain_with_simulation_db" in paired.columns:
        gain = pd.to_numeric(paired["rmse_gain_with_simulation_db"], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(gain)
        positive = int(np.sum(gain[finite] > 0.0))
        lines.append("## Simulation contribution")
        lines.append(f"Sampling ratios with lower mean RMSE after adding simulation: {positive}/{int(np.sum(finite))}")
        if np.any(finite):
            lines.append(f"Mean RMSE gain range (measurement-only minus measured+simulation): {np.nanmin(gain):+.4f} to {np.nanmax(gain):+.4f} dB")
        if "fusion_activation_rate" in paired.columns:
            ar = pd.to_numeric(paired["fusion_activation_rate"], errors="coerce")
            lines.append(f"Simulation-active trial rate: {float(ar.min()):.3f} to {float(ar.max()):.3f}")
        if "fusion_weight_mean" in paired.columns:
            fw = pd.to_numeric(paired["fusion_weight_mean"], errors="coerce")
            lines.append(f"Mean positive simulation-weight range: {float(fw.min()):.3f} to {float(fw.max()):.3f}")
        lines.append("")

    lines += [
        "No monotonic post-processing is applied.",
        "Branch M uses only the selected measured points.",
        "Branch M+S uses the exact same selected measured points plus one fixed, pre-generated Sionna RT map.",
        "The reconstruction workflow does not recalibrate or relabel the Sionna RT map.",
        "At each sampling ratio, every eligible measured point not selected for reconstruction is used for common evaluation of both branches.",
        "For every trial/stage: selected measured points + evaluation measured points = all eligible measured points; no measured point is discarded.",
        "In strict M+S mode, the simulation fusion weight is always positive when valid simulation values exist; weight 0 is not a candidate.",
        "Filled-map RSRP values are not used for measured-point eligibility or RMSE; only common grid axes/building geometry are used for the spatial domain.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    project_root = args.project_root.expanduser().resolve()
    measurements_csv = args.measurements.expanduser().resolve() if args.measurements is not None else project_root / "data" / "processed" / "cell_pci_rsrp_1m_calibration.csv"
    output_root = args.output_root.expanduser().resolve() if args.output_root is not None else project_root / "outputs" / "radio_map_reconstruction_two_branch_v1155"
    percentages = parse_percentages(args.percentages)
    fusion_weight_grid = _parse_float_grid(args.fusion_weight_grid, name="fusion-weight-grid", minimum=0.0, maximum=1.0)
    fusion_distance_scales = _parse_float_grid(args.fusion_distance_scales, name="fusion-distance-scales", minimum=0.0)
    if args.simulation_correction == "strict_ms_fusion":
        fusion_weight_grid = [v for v in fusion_weight_grid if v > 0.0]
        if not fusion_weight_grid:
            raise ValueError("strict M+S mode requires positive --fusion-weight-grid values")
    elif 0.0 not in fusion_weight_grid:
        fusion_weight_grid = [0.0] + fusion_weight_grid
    if int(args.random_trials) < 1:
        raise ValueError("--random-trials至少为1")

    reference_path = discover_filled_reference(project_root, args.station_id, args.pci, args.filled_reference_npz)
    reference = load_simulation_prior(reference_path, station_id=int(args.station_id), pci=int(args.pci), fallback_extent=None)
    reference_map = np.asarray(reference.rsrp_dbm, dtype=float).copy()
    finite = np.isfinite(reference_map)
    reference_map[finite] = np.clip(reference_map[finite], -120.0, -40.0)
    building_mask = np.asarray(reference.building_mask, dtype=bool) if reference.building_mask is not None else np.zeros(reference_map.shape, dtype=bool)
    reference_eval_mask = ~building_mask

    simulation_prior = None
    simulation_path = None
    if args.simulation_mode in {"with", "compare"}:
        simulation_path = discover_simulation_prior(project_root, int(args.station_id), int(args.pci), args.simulation_npz)
        if simulation_path.resolve() == reference_path.resolve():
            raise ValueError("纯仿真NPZ不能与filled reference使用同一文件")
        simulation_prior = load_simulation_prior(simulation_path, int(args.station_id), int(args.pci), fallback_extent=None)

    measured = prepare_all_measured_cells(
        measurements_csv, int(args.station_id), int(args.pci),
        reference.x_axis_m, reference.y_axis_m, building_mask,
        float(args.min_rsrp_dbm), float(args.max_rsrp_dbm),
    )
    n_total = int(len(measured))
    max_count = max(max(1, int(math.ceil(n_total * pct / 100.0))) for pct in percentages)
    if max_count >= n_total:
        raise ValueError("最大采样比例必须保留至少1个未选实测点用于评价")

    station_root = output_root / f"station_{int(args.station_id):02d}_pci_{int(args.pci)}"
    station_root.mkdir(parents=True, exist_ok=True)
    measured.to_csv(station_root / "eligible_measured_points.csv", index=False, encoding="utf-8-sig")

    print("=" * 88)
    print("v1.15.5 strict two-branch radio-map reconstruction")
    print(f"Station={args.station_id}, PCI={args.pci}")
    print(f"Eligible measured cells={n_total}; all are used at every stage: selected for reconstruction or unselected for evaluation")
    print(f"Trials={args.random_trials}; percentages={percentages}")
    print(f"Measured-only=robust IDW(k={args.idw_k}, p={args.idw_power})")
    if args.simulation_correction == "strict_ms_fusion":
        print("Branch M=selected measured points only -> robust IDW")
        print("Branch M+S=same selected measured points + fixed Sionna RT map -> strict positive-weight fusion")
        print(f"Positive simulation weights={fusion_weight_grid}; distance scales={fusion_distance_scales}")
        print("Sionna RT is read as a fixed input map; no Sionna recalibration/relabeling occurs in reconstruction.")
    elif args.simulation_correction == "cv_safe_fusion":
        print("With simulation=legacy CV-safe fusion (may collapse to measurement-only)")
    else:
        print("With simulation=legacy full-strength Sionna trend + robust residual IDW")
    print("Primary metric=ALL unselected measured cells at each sampling ratio; no permanent test split and no discarded measured cells")
    print("=" * 88)

    trial_rows: list[dict[str, Any]] = []
    representative_maps: dict[int, tuple[np.ndarray, np.ndarray, float, float]] = {}
    xx, yy = np.meshgrid(reference.x_axis_m, reference.y_axis_m)
    map_query_xy = np.column_stack([xx.ravel(), yy.ravel()])

    for trial_id in range(int(args.random_trials)):
        trial_seed = int(args.random_seed) + 1000 + trial_id
        ranked = _trial_ranking(
            measured, trial_seed, args.selection_mode, max_count,
            x_axis=reference.x_axis_m, y_axis=reference.y_axis_m, mask=reference_eval_mask,
        )
        if trial_id == 0:
            ranked.to_csv(station_root / "trial000_nested_ranking.csv", index=False, encoding="utf-8-sig")

        for pct in percentages:
            sample_count = min(max(1, int(math.ceil(n_total * pct / 100.0))), len(ranked))
            selected = ranked.iloc[:sample_count].copy().reset_index(drop=True)
            train_xy = selected[["x_m", "y_m"]].to_numpy(dtype=float)
            train_y = selected["measured_rsrp_dbm"].to_numpy(dtype=float)

            selected_ids = set(selected["measurement_point_id"].astype(int).tolist())
            evaluation = measured.loc[~measured["measurement_point_id"].isin(selected_ids)].copy().reset_index(drop=True)
            expected_eval_count = int(n_total - sample_count)
            if len(evaluation) != expected_eval_count:
                raise RuntimeError(
                    f"实测点记账错误: total={n_total}, selected={sample_count}, "
                    f"evaluation={len(evaluation)}, expected={expected_eval_count}"
                )
            eval_xy = evaluation[["x_m", "y_m"]].to_numpy(dtype=float)
            eval_y = evaluation["measured_rsrp_dbm"].to_numpy(dtype=float)

            baseline_test = robust_idw_predict(train_xy, train_y, eval_xy, k=int(args.idw_k), power=float(args.idw_power), huber_c=float(args.huber_c))
            base_row = {
                "trial_id": int(trial_id), "trial_seed": int(trial_seed),
                "sampling_percent": int(pct), "selected_measured_point_count": int(sample_count),
                "total_eligible_measured_points": int(n_total),
                "evaluation_measured_point_count": int(len(evaluation)),
                "all_measured_accounted_for": bool(sample_count + len(evaluation) == n_total),
                "variant": "without_simulation", "simulation_data_used": False,
                "method": "robust_idw", "rmse_db": _finite_metric(eval_y, baseline_test, "rmse"),
                "mae_db": _finite_metric(eval_y, baseline_test, "mae"), "bias_db": _finite_metric(eval_y, baseline_test, "bias"),
            }
            trial_rows.append(base_row)

            assisted_test = None
            sim_diag: dict[str, Any] = {}
            if simulation_prior is not None:
                if args.simulation_correction == "strict_ms_fusion":
                    assisted_test, sim_diag = simulation_strict_ms_fusion(
                        prior=simulation_prior, train_xy=train_xy, train_y=train_y,
                        query_xy=eval_xy, baseline_prediction=baseline_test,
                        k=int(args.idw_k), power=float(args.idw_power), huber_c=float(args.huber_c),
                        residual_clip_mad=float(args.residual_clip_mad),
                        fusion_weight_grid=fusion_weight_grid,
                        fusion_distance_scales=fusion_distance_scales,
                        min_valid_points=int(args.fusion_min_valid_points),
                    )
                    assisted_method = "strict_measured_plus_fixed_sionna"
                elif args.simulation_correction == "cv_safe_fusion":
                    assisted_test, sim_diag = simulation_cv_safe_fusion(
                        prior=simulation_prior, train_xy=train_xy, train_y=train_y,
                        query_xy=eval_xy, baseline_prediction=baseline_test,
                        k=int(args.idw_k), power=float(args.idw_power), huber_c=float(args.huber_c),
                        residual_clip_mad=float(args.residual_clip_mad),
                        fusion_weight_grid=fusion_weight_grid,
                        fusion_distance_scales=fusion_distance_scales,
                        min_cv_gain_db=float(args.fusion_min_cv_gain_db),
                        min_valid_points=int(args.fusion_min_valid_points),
                    )
                    assisted_method = "cv_safe_measured_sionna_fusion"
                else:
                    assisted_test, sim_diag = simulation_residual_robust_idw(
                        prior=simulation_prior, train_xy=train_xy, train_y=train_y,
                        query_xy=eval_xy, baseline_prediction=baseline_test,
                        k=int(args.idw_k), power=float(args.idw_power), huber_c=float(args.huber_c),
                        residual_clip_mad=float(args.residual_clip_mad),
                    )
                    assisted_method = "legacy_full_strength_sionna_residual_idw"
                trial_rows.append({
                    **{k: v for k, v in base_row.items() if k not in {"variant", "simulation_data_used", "method", "rmse_db", "mae_db", "bias_db"}},
                    "variant": "with_simulation", "simulation_data_used": True,
                    "method": assisted_method,
                    "rmse_db": _finite_metric(eval_y, assisted_test, "rmse"),
                    "mae_db": _finite_metric(eval_y, assisted_test, "mae"),
                    "bias_db": _finite_metric(eval_y, assisted_test, "bias"),
                    **sim_diag,
                })

            # Full 512x512 maps are generated only for trial 0 and 1/5/10%, so
            # Monte-Carlo evaluation remains fast while preserving paper figures.
            if trial_id == 0 and int(pct) in {1, 5, 10}:
                baseline_map = robust_idw_predict(
                    train_xy, train_y, map_query_xy,
                    k=int(args.idw_k), power=float(args.idw_power), huber_c=float(args.huber_c),
                ).reshape(reference_map.shape)
                baseline_map = np.clip(baseline_map, -120.0, -40.0)
                baseline_map[building_mask] = np.nan
                assisted_map = baseline_map.copy()
                if simulation_prior is not None:
                    if args.simulation_correction == "strict_ms_fusion":
                        assisted_flat, _ = simulation_strict_ms_fusion(
                            prior=simulation_prior, train_xy=train_xy, train_y=train_y,
                            query_xy=map_query_xy, baseline_prediction=baseline_map.ravel(),
                            k=int(args.idw_k), power=float(args.idw_power), huber_c=float(args.huber_c),
                            residual_clip_mad=float(args.residual_clip_mad),
                            fusion_weight_grid=fusion_weight_grid,
                            fusion_distance_scales=fusion_distance_scales,
                            min_valid_points=int(args.fusion_min_valid_points),
                        )
                    elif args.simulation_correction == "cv_safe_fusion":
                        assisted_flat, _ = simulation_cv_safe_fusion(
                            prior=simulation_prior, train_xy=train_xy, train_y=train_y,
                            query_xy=map_query_xy, baseline_prediction=baseline_map.ravel(),
                            k=int(args.idw_k), power=float(args.idw_power), huber_c=float(args.huber_c),
                            residual_clip_mad=float(args.residual_clip_mad),
                            fusion_weight_grid=fusion_weight_grid,
                            fusion_distance_scales=fusion_distance_scales,
                            min_cv_gain_db=float(args.fusion_min_cv_gain_db),
                            min_valid_points=int(args.fusion_min_valid_points),
                        )
                    else:
                        assisted_flat, _ = simulation_residual_robust_idw(
                            prior=simulation_prior, train_xy=train_xy, train_y=train_y,
                            query_xy=map_query_xy, baseline_prediction=baseline_map.ravel(),
                            k=int(args.idw_k), power=float(args.idw_power), huber_c=float(args.huber_c),
                            residual_clip_mad=float(args.residual_clip_mad),
                        )
                    assisted_map = assisted_flat.reshape(reference_map.shape)
                    assisted_map[building_mask] = np.nan
                representative_maps[int(pct)] = (
                    baseline_map.copy(), assisted_map.copy(),
                    float(base_row["rmse_db"]),
                    float(_finite_metric(eval_y, assisted_test, "rmse")) if assisted_test is not None else float("nan"),
                )

                pct_dir = station_root / f"percent_{int(pct):02d}_trial000"
                pct_dir.mkdir(parents=True, exist_ok=True)
                selected.to_csv(pct_dir / "selected_measured_points.csv", index=False, encoding="utf-8-sig")
                evaluation.to_csv(pct_dir / "evaluation_all_unselected_measured_points.csv", index=False, encoding="utf-8-sig")
                save_map_npz(
                    pct_dir / f"station_{int(args.station_id):02d}_pci_{int(args.pci)}_robust_idw_{int(pct):02d}pct.npz",
                    station_id=args.station_id, pci=args.pci, x_axis=reference.x_axis_m, y_axis=reference.y_axis_m,
                    rsrp_map=baseline_map, building_mask=building_mask, map_role=f"robust_idw_{pct}pct",
                    metadata={"primary_rmse_all_unselected_measured_db": base_row["rmse_db"], "evaluation_point_count": len(evaluation), "trial_id": 0, "trial_seed": trial_seed}, selected_points=selected,
                )
                if simulation_prior is not None:
                    save_map_npz(
                        pct_dir / f"station_{int(args.station_id):02d}_pci_{int(args.pci)}_measured_sionna_fusion_{int(pct):02d}pct.npz",
                        station_id=args.station_id, pci=args.pci, x_axis=reference.x_axis_m, y_axis=reference.y_axis_m,
                        rsrp_map=assisted_map, building_mask=building_mask, map_role=f"measured_sionna_fusion_{pct}pct",
                        metadata={"primary_rmse_all_unselected_measured_db": representative_maps[int(pct)][3], "evaluation_point_count": len(evaluation), "trial_id": 0, "trial_seed": trial_seed}, selected_points=selected,
                    )
                if not args.skip_figures:
                    plot_map(
                        pct_dir / f"robust_idw_{int(pct):02d}pct.png", rsrp_map=baseline_map,
                        x_axis=reference.x_axis_m, y_axis=reference.y_axis_m, building_mask=building_mask,
                        title=f"Measured-only robust IDW ({pct}% measured points)", min_dbm=args.display_min_dbm,
                        max_dbm=args.display_max_dbm, dpi=args.dpi, selected_points=selected, show_selected_points=True,
                    )
                    if simulation_prior is not None:
                        plot_map(
                            pct_dir / f"measured_sionna_fusion_{int(pct):02d}pct.png", rsrp_map=assisted_map,
                            x_axis=reference.x_axis_m, y_axis=reference.y_axis_m, building_mask=building_mask,
                            title=f"Measured + fixed Sionna RT ({pct}% measured points)", min_dbm=args.display_min_dbm,
                            max_dbm=args.display_max_dbm, dpi=args.dpi, selected_points=selected, show_selected_points=True,
                        )

    trials = pd.DataFrame(trial_rows).sort_values(["trial_id", "sampling_percent", "simulation_data_used"]).reset_index(drop=True)
    trials.to_csv(station_root / "reconstruction_trial_metrics.csv", index=False, encoding="utf-8-sig")
    paired = _aggregate_trial_metrics(trials, percentages)
    paired.to_csv(station_root / "reconstruction_simulation_ablation_comparison.csv", index=False, encoding="utf-8-sig")
    accounting = paired[["sampling_percent", "selected_measured_point_count", "evaluation_measured_point_count", "all_measured_accounted_for"]].copy()
    accounting["total_eligible_measured_points"] = int(n_total)
    accounting["selected_plus_evaluation"] = accounting["selected_measured_point_count"] + accounting["evaluation_measured_point_count"]
    accounting["all_points_used_exactly_once_per_stage"] = accounting["selected_plus_evaluation"].eq(int(n_total))
    accounting.to_csv(station_root / "reconstruction_measurement_usage_audit.csv", index=False, encoding="utf-8-sig")
    _write_trend_audit(station_root / "reconstruction_trend_audit.md", paired)

    # Compatibility output: one row per percentage / variant using Monte-Carlo means.
    metric_rows: list[dict[str, Any]] = []
    for _, r in paired.iterrows():
        assisted_metric_method = ("strict_measured_plus_fixed_sionna" if args.simulation_correction == "strict_ms_fusion" else ("cv_safe_measured_sionna_fusion" if args.simulation_correction == "cv_safe_fusion" else "legacy_full_strength_sionna_residual_idw"))
        for suffix, used, method in [
            ("without_simulation", False, "robust_idw"),
            ("with_simulation", True, assisted_metric_method),
        ]:
            key = f"rmse_db_{suffix}"
            if key not in r or not np.isfinite(r.get(key, np.nan)):
                continue
            metric_rows.append({
                "station_id": int(args.station_id), "pci": int(args.pci),
                "sampling_percent": int(r["sampling_percent"]),
                "selected_measured_point_count": int(r["selected_measured_point_count"]),
                "rmse_db": float(r[key]), "rmse_std_db": float(r.get(f"rmse_db_std_{suffix}", np.nan)),
                "mae_db": float(r.get(f"mae_db_{suffix}", np.nan)), "bias_db": float(r.get(f"bias_db_{suffix}", np.nan)),
                "method": method, "variant": suffix, "simulation_data_used": used,
                "evaluation_reference": "all_unselected_measured_points_at_each_sampling_ratio",
                "random_trial_count": int(args.random_trials),
            })
    pd.DataFrame(metric_rows).to_csv(station_root / "reconstruction_simulation_ablation_metrics.csv", index=False, encoding="utf-8-sig")

    # Fusion diagnostics are intentionally training-side diagnostics. They make it
    # possible to verify that a result is not obtained by looking at the current unselected
    # evaluation labels when choosing simulation strength.
    with_rows = trials.loc[trials["variant"].eq("with_simulation")].copy() if not trials.empty else pd.DataFrame()
    diag_cols = [c for c in [
        "trial_id", "trial_seed", "sampling_percent", "selected_measured_point_count",
        "fusion_activated", "fusion_reason", "fusion_weight", "fusion_distance_scale_multiplier",
        "fusion_tau_m", "fusion_cv_point_count", "fusion_cv_rmse_baseline_db",
        "fusion_cv_rmse_candidate_db", "fusion_cv_rmse_selected_db", "fusion_cv_gain_db",
        "fusion_innovation_error_correlation", "simulation_mean_effective_weight",
        "simulation_max_effective_weight", "simulation_active_fraction",
        "simulation_prior_training_count", "simulation_prior_training_fraction",
        "rmse_db", "mae_db", "bias_db",
    ] if c in with_rows.columns]
    if diag_cols:
        with_rows[diag_cols].to_csv(station_root / "reconstruction_fusion_diagnostics.csv", index=False, encoding="utf-8-sig")

    if not args.skip_figures:
        if simulation_prior is not None and all(p in representative_maps for p in (1, 5, 10)):
            plot_paired_reconstruction_composite(
                station_root / "reconstruction_simulation_ablation_representative_maps.png",
                representative_maps=representative_maps, x_axis=reference.x_axis_m, y_axis=reference.y_axis_m,
                building_mask=building_mask, min_dbm=args.display_min_dbm, max_dbm=args.display_max_dbm, dpi=args.dpi,
            )
        fig, ax = plt.subplots(figsize=(7.48, 5.2), dpi=int(args.dpi))
        x = paired["sampling_percent"].to_numpy(dtype=float)
        if "rmse_db_without_simulation" in paired:
            y = paired["rmse_db_without_simulation"].to_numpy(dtype=float)
            s = paired["rmse_db_std_without_simulation"].to_numpy(dtype=float)
            ax.plot(x, y, marker="o", label="Measured only: robust IDW")
            ax.fill_between(x, y - s, y + s, alpha=0.15)
        if "rmse_db_with_simulation" in paired:
            y = paired["rmse_db_with_simulation"].to_numpy(dtype=float)
            s = paired["rmse_db_std_with_simulation"].to_numpy(dtype=float)
            ax.plot(x, y, marker="s", label="Measured + fixed Sionna RT")
            ax.fill_between(x, y - s, y + s, alpha=0.15)
        ax.set_xlabel("Measured sampling ratio [%]")
        ax.set_ylabel("RMSE on all unselected measurements [dB]")
        ax.set_title(f"Radio-map reconstruction ({int(args.random_trials)} nested random trials)")
        ax.set_xticks(x.astype(int))
        ax.grid(True, alpha=0.28)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(station_root / "reconstruction_simulation_ablation_rmse.png", dpi=int(args.dpi), bbox_inches="tight", facecolor="white")
        plt.close(fig)

    metadata = {
        "version": "1.15.5",
        "experiment": "strict two-branch radio-map reconstruction with full leave-selected-out measurement evaluation",
        "station_id": int(args.station_id), "pci": int(args.pci),
        "measurement_csv": str(measurements_csv), "filled_reference_npz": str(reference_path),
        "simulation_prior_npz": str(simulation_path) if simulation_path else None,
        "eligible_measured_point_count": int(n_total),
        "measurement_usage_rule": "at each sampling ratio, selected points reconstruct and every unselected measured point evaluates; selected+evaluation=all eligible measured points",
        "random_trials": int(args.random_trials), "base_random_seed": int(args.random_seed),
        "percentages": percentages,
        "sampling_definition": "nested prefixes of one uniform-random permutation per trial; sample count is percentage of the full eligible measured-cell population",
        "primary_evaluation_reference": "all eligible real measured RSRP points not selected for reconstruction at the current sampling ratio",
        "filled_map_role": "grid axes/building-mask visualization domain only; filled-map RSRP values are not used for measured eligibility or quantitative RMSE",
        "measurement_only_method": {"name": "robust_idw", "k": int(args.idw_k), "power": float(args.idw_power), "huber_c": float(args.huber_c)},
        "measurement_simulation_method": {
            "name": ("strict_measured_plus_fixed_sionna" if args.simulation_correction == "strict_ms_fusion" else ("cv_safe_measured_sionna_fusion" if args.simulation_correction == "cv_safe_fusion" else "legacy_full_strength_sionna_residual_idw")),
            "candidate": "Sionna RT + robust IDW of measurement-minus-simulation residuals",
            "fusion": "same measured subset + fixed Sionna residual field; strictly positive weight/distance gate selected by measured-training-only leave-one-out CV",
            "fusion_weight_grid": fusion_weight_grid,
            "fusion_distance_scales": fusion_distance_scales,
            "fusion_min_cv_gain_db": float(args.fusion_min_cv_gain_db),
            "fusion_min_valid_points": int(args.fusion_min_valid_points),
            "residual_clip_mad": float(args.residual_clip_mad),
        },
        "simulation_branch_contains_measurement_only_submodel": False,
        "evaluation_labels_used_for_fusion_selection": False,
        "sionna_recalibrated_inside_reconstruction": False,
        "branch_M_inputs": "selected measured points only",
        "branch_MS_inputs": "same selected measured points + fixed pre-generated Sionna RT map",
        "monotonic_postprocessing": False,
        "seed_selection_by_metric": False,
    }
    (station_root / "experiment_metadata.json").write_text(json.dumps(_json_safe(metadata), ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n完成。主结果：", station_root / "reconstruction_simulation_ablation_comparison.csv")
    print("逐trial结果：", station_root / "reconstruction_trial_metrics.csv")
    if simulation_prior is not None:
        print("融合诊断：", station_root / "reconstruction_fusion_diagnostics.csv")
    print("趋势审计：", station_root / "reconstruction_trend_audit.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
