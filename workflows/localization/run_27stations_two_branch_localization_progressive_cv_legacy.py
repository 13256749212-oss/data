#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Progressive cross-validated candidate localization with adaptive Sionna assistance.

Formal branches
---------------
M   : sparse measured receiver coordinates + measured PCI-RSRP.
M+S : the exact same sparse measured receiver locations plus collocated Sionna
      RSRP and local Sionna field-gradient direction evidence.

The solver is intentionally truth-free. Ground-truth transmitter coordinates are
read only after the final per-station estimates have been formed, solely for RMSE.
The 10--15 receiver-location experiments are strict nested prefixes for every
station/trial. Progressive updates compare the inherited and new positions on the same enlarged
selected data set, and randomized trials must move coherently before the station
consensus is allowed to move. No extra measured holdout points are consumed.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import legacy_pgrmsbil as common  # noqa: E402
import run_27stations_multicandidate_cv_localization as mcvl  # noqa: E402

ALGORITHM_NAME = "Progressive Cross-Validated Candidate Fusion Localization"
SECTOR_OFFSETS = np.asarray([0.0, 2.0 * np.pi / 3.0, -2.0 * np.pi / 3.0], dtype=float)


def parse_counts(text: str | None, single: int | None) -> list[int]:
    if single is not None:
        if int(single) <= 0:
            raise ValueError("--points-per-station must be positive")
        return [int(single)]
    raw = "10,11,12,13,14,15" if text is None else str(text)
    counts = sorted({int(v.strip()) for v in raw.split(",") if v.strip()})
    if not counts or min(counts) <= 0:
        raise ValueError("--point-counts must contain positive integers")
    return counts


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Progressive two-branch base-station localization")
    p.add_argument("--project-root", type=Path, default=ROOT)
    p.add_argument("--measurements", type=Path, default=None)
    p.add_argument("--output-root", "--output-dir", dest="output_root", type=Path, default=None)
    p.add_argument("--point-counts", default=None)
    p.add_argument("--points-per-station", type=int, default=None)
    p.add_argument("--random-trials", type=int, default=10)
    p.add_argument("--random-seed", type=int, default=20260805)
    p.add_argument("--station-ids", default="all")
    p.add_argument("--simulation-root", type=Path, default=None)
    p.add_argument("--simulation-weight", type=float, default=0.50)
    p.add_argument("--progressive-min-improvement-db", type=float, default=0.35)
    p.add_argument("--validation-points", type=int, default=24)
    p.add_argument("--validation-min-improvement-db", type=float, default=0.08)
    p.add_argument("--simulation-measured-tolerance-db", type=float, default=0.15)
    p.add_argument("--x-min", type=float, default=mcvl.DEFAULT_BOUNDS[0])
    p.add_argument("--x-max", type=float, default=mcvl.DEFAULT_BOUNDS[1])
    p.add_argument("--y-min", type=float, default=mcvl.DEFAULT_BOUNDS[2])
    p.add_argument("--y-max", type=float, default=mcvl.DEFAULT_BOUNDS[3])
    # Backwards-compatible options forwarded by run_pipeline.py.
    p.add_argument("--simulation-mode", choices=["compare"], default="compare")
    p.add_argument("--direction-prior-mode", choices=["fixed", "soft", "off"], default="off")
    p.add_argument("--strict-simulation-data", action="store_true", default=True)
    p.add_argument("--allow-missing-simulation", action="store_true")
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
    p.add_argument("--fast-max-nfev", type=int, default=36,
                   help="Maximum function evaluations for each free-position NLS fit (default: 36)")
    p.add_argument("--fast-fixed-nfev", type=int, default=14,
                   help="Maximum function evaluations when only nuisance parameters are refit (default: 14)")
    p.add_argument("--fast-initial-starts", type=int, default=3,
                   help="Number of multi-start initializations at the first point count (default: 3)")
    return p.parse_args()


def _uniform_nested_points(points: pd.DataFrame, k: int, seed: int) -> pd.DataFrame:
    """Uniform random nested prefix; no RSRP-aware best-first selection.

    The previous implementation deliberately put the most informative locations
    into the first ten samples. That made the 10-point case artificially strong
    and left lower-value points for 11--15, which is exactly the wrong protocol
    for studying how localization improves as more random measurements arrive.
    """
    rss_cols = ["rsrp_s1", "rsrp_s2", "rsrp_s3"]
    pool = points.loc[points[rss_cols].notna().any(axis=1)].copy().reset_index(drop=True)
    if len(pool) < int(k):
        raise ValueError(f"Only {len(pool)} valid receiver locations, fewer than requested {k}")
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(len(pool))
    out = pool.iloc[order[:int(k)]].copy().reset_index(drop=True)
    out["selection_rank"] = np.arange(1, len(out) + 1)
    out["selection_strategy"] = "uniform_random_nested_prefix"
    return out


def _fixed_validation_split(points: pd.DataFrame, max_count: int, requested: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rss_cols = ["rsrp_s1", "rsrp_s2", "rsrp_s3"]
    pool = points.loc[points[rss_cols].notna().any(axis=1)].copy().reset_index(drop=True)
    spare = len(pool) - int(max_count)
    if spare < 8:
        return pool, pool.iloc[0:0].copy()
    n_val = int(min(max(8, int(requested)), spare))
    # A fixed validation set across every point count and random trial for this
    # station. It is never used to generate localization candidates.
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(len(pool), size=n_val, replace=False)
    validation = pool.iloc[np.sort(idx)].copy().reset_index(drop=True)
    if "receiver_point_key" in pool.columns:
        keys = set(validation["receiver_point_key"].astype(str))
        train = pool.loc[~pool["receiver_point_key"].astype(str).isin(keys)].copy()
    else:
        val_xy = set(map(tuple, np.round(validation[["x_m", "y_m", "z_m"]].to_numpy(float), 6)))
        keep = [tuple(v) not in val_xy for v in np.round(pool[["x_m", "y_m", "z_m"]].to_numpy(float), 6)]
        train = pool.loc[np.asarray(keep, bool)].copy()
    if len(train) < int(max_count):
        return pool, pool.iloc[0:0].copy()
    return train.reset_index(drop=True), validation.reset_index(drop=True)


def _simulation_selected(project: Path, simulation_root: Path, station: pd.DataFrame, selected: pd.DataFrame,
                         station_id: int, omni: bool, allow_missing: bool = True) -> pd.DataFrame | None:
    if selected is None or len(selected) == 0:
        return None
    sim, _, diagnostics = mcvl.build_collocated_simulation_dataset(
        project_root=project,
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
    if int(diagnostics.get("simulation_matched_observation_count", 0)) <= 0:
        return None
    return sim


def _robust_affine(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Fit y ~= a*x+b robustly; return a,b,RMSE without any transmitter truth."""
    mask = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x, float)[mask]
    y = np.asarray(y, float)[mask]
    if len(x) < 2:
        return 1.0, float(np.nanmedian(y - x)) if len(x) else 0.0, float("inf")
    if len(x) < 4 or float(np.std(x)) < 1e-6:
        a = 1.0
        b = float(np.median(y - x))
    else:
        x0 = float(np.median(x)); y0 = float(np.median(y))
        denom = float(np.sum((x - x0) ** 2))
        a = float(np.sum((x - x0) * (y - y0)) / max(denom, 1e-9))
        a = float(np.clip(a, 0.65, 1.35))
        b = float(np.median(y - a * x))
        for _ in range(4):
            r = y - (a * x + b)
            med = float(np.median(r)); mad = float(np.median(np.abs(r - med)))
            scale = max(1.4826 * mad, 1.5)
            w = 1.0 / np.maximum(1.0, np.abs(r - med) / (2.0 * scale))
            X = np.column_stack([x, np.ones(len(x))])
            sw = np.sqrt(w)
            coef = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)[0]
            a = float(np.clip(coef[0], 0.65, 1.35)); b = float(coef[1])
    residual = y - (a * x + b)
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    return a, b, rmse


def _build_fused_channel(measured_train: pd.DataFrame, simulated_train: pd.DataFrame | None,
                         measured_other: pd.DataFrame | None = None, simulated_other: pd.DataFrame | None = None,
                         base_weight: float = 0.5, omni: bool = False) -> tuple[pd.DataFrame, pd.DataFrame | None, float, dict[str, Any]]:
    """Bias/scale-align Sionna to measured RSRP and form a conservative fused channel."""
    train = measured_train.copy()
    other = measured_other.copy() if measured_other is not None else None
    if simulated_train is None or len(simulated_train) != len(measured_train):
        return train, other, 0.0, {"gate": 0.0}

    sector_count = 1 if omni else 3
    fits: dict[int, tuple[float, float, float]] = {}
    corrs: list[float] = []
    rmses: list[float] = []
    cover_num = 0; cover_den = 0
    for j in range(1, sector_count + 1):
        m = pd.to_numeric(measured_train[f"rsrp_s{j}"], errors="coerce").to_numpy(float)
        s = pd.to_numeric(simulated_train[f"rsrp_s{j}"], errors="coerce").to_numpy(float)
        mask = np.isfinite(m) & np.isfinite(s)
        cover_num += int(mask.sum()); cover_den += int(np.isfinite(m).sum())
        a, b, rmse = _robust_affine(s, m)
        fits[j] = (a, b, rmse)
        if np.isfinite(rmse):
            rmses.append(rmse)
        if int(mask.sum()) >= 4 and float(np.std(m[mask])) > 1e-6 and float(np.std(s[mask])) > 1e-6:
            c = float(np.corrcoef(m[mask], s[mask])[0, 1])
            if np.isfinite(c): corrs.append(c)

    corr = float(np.median(corrs)) if corrs else 0.0
    rmse = float(np.median(rmses)) if rmses else float("inf")
    coverage = float(cover_num / max(cover_den, 1))
    corr_gate = float(np.clip((corr - 0.05) / 0.70, 0.0, 1.0))
    rmse_gate = float(np.exp(-max(rmse - 2.0, 0.0) / 8.0)) if np.isfinite(rmse) else 0.0
    coverage_gate = float(np.clip((coverage - 0.20) / 0.70, 0.0, 1.0))
    gate = float(np.clip(corr_gate * rmse_gate * coverage_gate, 0.0, 1.0))
    # Cap the RSRP-value fusion. Simulation is an assistant, not a replacement
    # for the measured channel. Gradient evidence is handled separately.
    w = float(np.clip(base_weight, 0.0, 1.0)) * gate
    w = min(w, 0.35)

    def apply(meas: pd.DataFrame, sim: pd.DataFrame | None) -> pd.DataFrame:
        out = meas.copy()
        if sim is None or len(sim) != len(meas):
            return out
        for j in range(1, sector_count + 1):
            a, b, _ = fits[j]
            m = pd.to_numeric(meas[f"rsrp_s{j}"], errors="coerce").to_numpy(float)
            s = pd.to_numeric(sim[f"rsrp_s{j}"], errors="coerce").to_numpy(float)
            sa = a * s + b
            both = np.isfinite(m) & np.isfinite(sa)
            only_m = np.isfinite(m) & ~np.isfinite(sa)
            only_s = ~np.isfinite(m) & np.isfinite(sa)
            fused = np.full(len(meas), np.nan, dtype=float)
            fused[both] = (1.0 - w) * m[both] + w * sa[both]
            fused[only_m] = m[only_m]
            # Simulation-only sector observations are admitted conservatively;
            # they improve sector completeness without overwhelming measurements.
            fused[only_s] = sa[only_s]
            out[f"rsrp_s{j}"] = fused
            obs_w = np.zeros(len(meas), dtype=float)
            obs_w[both | only_m] = 1.0
            obs_w[only_s] = max(0.15, w)
            out[f"obs_weight_s{j}"] = obs_w
        return out

    return apply(measured_train, simulated_train), apply(measured_other, simulated_other) if measured_other is not None else None, w, {
        "gate": gate, "correlation": corr, "residual_rmse_db": rmse, "coverage": coverage,
    }


def _sector_gain(offset_rad: np.ndarray) -> np.ndarray:
    deg = np.degrees(np.abs((np.asarray(offset_rad) + np.pi) % (2.0 * np.pi) - np.pi))
    return -np.minimum(12.0 * (deg / mcvl.HORIZONTAL_3DB_BEAMWIDTH_DEG) ** 2, mcvl.HORIZONTAL_MAX_ATTENUATION_DB)


def _obs_arrays(frame: pd.DataFrame, omni: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sector_count = 1 if omni else 3
    xy = frame[["x_m", "y_m"]].to_numpy(float)
    rows_xy: list[np.ndarray] = []
    sectors: list[int] = []
    y: list[float] = []
    weights: list[float] = []
    for i in range(len(frame)):
        for j in range(1, sector_count + 1):
            val = float(pd.to_numeric(pd.Series([frame.iloc[i][f"rsrp_s{j}"]]), errors="coerce").iloc[0])
            if not np.isfinite(val):
                continue
            wc = f"obs_weight_s{j}"
            weight = float(frame.iloc[i][wc]) if wc in frame.columns and np.isfinite(frame.iloc[i][wc]) else 1.0
            if weight <= 0.0:
                continue
            rows_xy.append(xy[i]); sectors.append(j - 1); y.append(val); weights.append(float(np.clip(weight, 0.05, 1.0)))
    if not y:
        return np.empty((0, 2)), np.empty(0, int), np.empty(0), np.empty(0)
    return np.asarray(rows_xy, float), np.asarray(sectors, int), np.asarray(y, float), np.asarray(weights, float)


def _predict(params: np.ndarray, xy: np.ndarray, sectors: np.ndarray, omni: bool) -> np.ndarray:
    tx = np.asarray(params[:2], float)
    dx = xy[:, 0] - tx[0]; dy = xy[:, 1] - tx[1]
    d = np.sqrt(dx * dx + dy * dy + mcvl.VERTICAL_SEPARATION_M ** 2)
    if omni:
        n = float(params[2]); beta = float(params[3])
        gain = 0.0
        return beta - 10.0 * n * np.log10(np.maximum(d, 1.0)) + gain
    alpha = float(params[2]); n = float(params[3]); betas = np.asarray(params[4:7], float)
    bearing = np.arctan2(dy, dx)
    orient = alpha + SECTOR_OFFSETS[sectors]
    gain = _sector_gain(bearing - orient)
    return betas[sectors] - 10.0 * n * np.log10(np.maximum(d, 1.0)) + gain


def _initial_params(frame: pd.DataFrame, init_xy: np.ndarray, omni: bool) -> np.ndarray:
    xy = frame[["x_m", "y_m"]].to_numpy(float)
    init_xy = np.asarray(init_xy, float)
    n0 = 2.8
    if omni:
        vals = pd.to_numeric(frame["rsrp_s1"], errors="coerce").to_numpy(float)
        good = np.isfinite(vals)
        d = np.sqrt(np.sum((xy[good] - init_xy[None, :]) ** 2, axis=1) + mcvl.VERTICAL_SEPARATION_M ** 2)
        beta = float(np.median(vals[good] + 10.0 * n0 * np.log10(np.maximum(d, 1.0)))) if good.any() else 20.0
        return np.asarray([init_xy[0], init_xy[1], n0, beta], float)

    # Estimate one common sector-frame orientation from high-RSRP sector centroids.
    alpha_candidates: list[float] = []
    for j in range(3):
        vals = pd.to_numeric(frame[f"rsrp_s{j+1}"], errors="coerce").to_numpy(float)
        good = np.isfinite(vals)
        if int(good.sum()) < 2:
            continue
        ids = np.flatnonzero(good)
        top = ids[np.argsort(vals[good])[-min(4, len(ids)):]]
        w = np.exp(np.clip((vals[top] - np.max(vals[top])) / 5.0, -8.0, 0.0))
        c = np.average(xy[top], axis=0, weights=w)
        bearing = math.atan2(float(c[1] - init_xy[1]), float(c[0] - init_xy[0]))
        alpha_candidates.append(bearing - float(SECTOR_OFFSETS[j]))
    if alpha_candidates:
        alpha0 = float(math.atan2(np.mean(np.sin(alpha_candidates)), np.mean(np.cos(alpha_candidates))))
    else:
        alpha0 = 0.0
    betas: list[float] = []
    for j in range(3):
        vals = pd.to_numeric(frame[f"rsrp_s{j+1}"], errors="coerce").to_numpy(float)
        good = np.isfinite(vals)
        if not good.any():
            betas.append(20.0); continue
        dxy = xy[good] - init_xy[None, :]
        d = np.sqrt(np.sum(dxy ** 2, axis=1) + mcvl.VERTICAL_SEPARATION_M ** 2)
        bearing = np.arctan2(dxy[:, 1], dxy[:, 0])
        gain = _sector_gain(bearing - (alpha0 + float(SECTOR_OFFSETS[j])))
        beta = float(np.median(vals[good] + 10.0 * n0 * np.log10(np.maximum(d, 1.0)) - gain))
        betas.append(beta)
    return np.asarray([init_xy[0], init_xy[1], alpha0, n0, *betas], float)


@dataclass
class FitResult:
    xy: np.ndarray
    params: np.ndarray
    train_rmse_db: float
    validation_rmse_db: float
    position_sigma_m: float


def _score_validation(params: np.ndarray, validation: pd.DataFrame, omni: bool) -> float:
    if validation is None or len(validation) == 0:
        return float("nan")
    xy, sectors, y, w = _obs_arrays(validation, omni)
    if len(y) < 3:
        return float("nan")
    residual = _predict(params, xy, sectors, omni) - y
    med = float(np.median(residual)); mad = float(np.median(np.abs(residual - med)))
    scale = max(1.4826 * mad, 2.0)
    residual = np.clip(residual, med - 3.5 * scale, med + 3.5 * scale)
    return float(np.sqrt(np.sum(w * residual ** 2) / max(float(np.sum(w)), 1e-12)))


def _fit_model(frame: pd.DataFrame, validation: pd.DataFrame, init_xy: np.ndarray, omni: bool,
               bounds: Sequence[float], fixed_xy: np.ndarray | None = None,
               max_nfev_free: int = 36, max_nfev_fixed: int = 14) -> FitResult | None:
    xy, sectors, y, w = _obs_arrays(frame, omni)
    min_obs = 5 if omni else 10
    if len(y) < min_obs:
        return None
    x0 = _initial_params(frame, np.asarray(init_xy, float), omni)
    x_min, x_max, y_min, y_max = map(float, bounds)
    if omni:
        lo = np.asarray([x_min, y_min, 1.2, -20.0], float)
        hi = np.asarray([x_max, y_max, 5.8, 100.0], float)
    else:
        lo = np.asarray([x_min, y_min, -np.pi, 1.2, -20.0, -20.0, -20.0], float)
        hi = np.asarray([x_max, y_max, np.pi, 5.8, 100.0, 100.0, 100.0], float)
    x0 = np.clip(x0, lo + 1e-7, hi - 1e-7)

    if fixed_xy is None:
        def residual_fn(p: np.ndarray) -> np.ndarray:
            return np.sqrt(w) * (_predict(p, xy, sectors, omni) - y)
        try:
            res = least_squares(residual_fn, x0, bounds=(lo, hi), loss="soft_l1", f_scale=4.0,
                                max_nfev=max(12, int(max_nfev_free)), xtol=2e-4, ftol=2e-4, gtol=2e-4)
        except Exception:
            return None
        p = np.asarray(res.x, float)
        jac = np.asarray(res.jac, float)
    else:
        fixed_xy = np.asarray(fixed_xy, float)
        x0[0:2] = fixed_xy
        q0 = x0[2:].copy(); qlo = lo[2:]; qhi = hi[2:]
        def residual_q(q: np.ndarray) -> np.ndarray:
            p = np.r_[fixed_xy, q]
            return np.sqrt(w) * (_predict(p, xy, sectors, omni) - y)
        try:
            res = least_squares(residual_q, q0, bounds=(qlo, qhi), loss="soft_l1", f_scale=4.0,
                                max_nfev=max(8, int(max_nfev_fixed)), xtol=3e-4, ftol=3e-4, gtol=3e-4)
        except Exception:
            return None
        p = np.r_[fixed_xy, np.asarray(res.x, float)]
        jac = np.empty((len(y), 0))

    pred = _predict(p, xy, sectors, omni)
    train_rmse = float(np.sqrt(np.sum(w * (pred - y) ** 2) / max(float(np.sum(w)), 1e-12)))
    val = _score_validation(p, validation, omni)
    if not np.isfinite(val):
        val = train_rmse

    sigma = 250.0
    if fixed_xy is None and jac.ndim == 2 and jac.shape[1] >= 2 and jac.shape[0] > jac.shape[1]:
        try:
            h = jac.T @ jac
            cov = np.linalg.pinv(h)
            dof = max(jac.shape[0] - jac.shape[1], 1)
            s2 = float(np.sum((pred - y) ** 2) / dof)
            pos_cov = cov[:2, :2] * max(s2, 1e-6)
            eig = np.linalg.eigvalsh(pos_cov)
            sigma = float(np.sqrt(max(float(np.max(eig)), 1.0)))
        except Exception:
            sigma = 250.0
    sigma = float(np.clip(sigma, 5.0, 250.0))
    return FitResult(np.asarray(p[:2], float), p, train_rmse, float(val), sigma)


def _candidate_initializations(frame: pd.DataFrame, omni: bool, bounds: Sequence[float], previous_xy: np.ndarray | None,
                               extra_xy: Sequence[np.ndarray] = (), initial_starts: int = 3) -> list[np.ndarray]:
    """Return a very small deterministic start set.

    Multi-start is used only when no previous-count estimate exists. Once a 10-point
    solution is available, 11--15 points warm-start directly from that solution. This
    removes the dominant repeated MCVL candidate-generation/NLS cost.
    """
    centroid = np.mean(frame[["x_m", "y_m"]].to_numpy(float), axis=0)
    pts: list[np.ndarray] = []
    if previous_xy is not None and np.isfinite(previous_xy).all():
        pts.append(np.asarray(previous_xy, float))
        # M+S may offer one measurement anchor / gradient anchor, but never launch
        # a fresh 8-start search for every 11--15 point prefix.
        for v in extra_xy:
            if v is not None and np.isfinite(v).all():
                pts.append(np.asarray(v, float))
        pts.append(centroid)
    else:
        # First count only: obtain a few geometry starts, then cache the winner as
        # the warm start for every larger nested prefix.
        try:
            candidates = mcvl.generate_candidates(frame, SECTOR_OFFSETS.copy(), False, omni, bounds)
            pts.extend(np.asarray(v, float) for _, v in candidates[:max(1, int(initial_starts) - 1)])
        except Exception:
            pass
        pts.append(centroid)

    out: list[np.ndarray] = []
    limit = 2 if previous_xy is not None else max(1, int(initial_starts))
    for q in pts:
        q = np.asarray([np.clip(q[0], bounds[0], bounds[1]), np.clip(q[1], bounds[2], bounds[3])], float)
        if all(float(np.linalg.norm(q - r)) >= 12.0 for r in out):
            out.append(q)
        if len(out) >= limit:
            break
    return out or [centroid]


def _fit_best(frame: pd.DataFrame, validation: pd.DataFrame, omni: bool, bounds: Sequence[float],
              previous_xy: np.ndarray | None = None, extra_xy: Sequence[np.ndarray] = (),
              min_improvement_db: float = 0.08, max_nfev_free: int = 36,
              max_nfev_fixed: int = 14, initial_starts: int = 3) -> FitResult:
    starts = _candidate_initializations(frame, omni, bounds, previous_xy, extra_xy, initial_starts)
    fits: list[FitResult] = []
    for q in starts:
        f = _fit_model(frame, validation, q, omni, bounds,
                       max_nfev_free=max_nfev_free, max_nfev_fixed=max_nfev_fixed)
        if f is not None:
            fits.append(f)
    if not fits:
        fallback = np.mean(frame[["x_m", "y_m"]].to_numpy(float), axis=0)
        result = _fit_model(frame, validation, fallback, omni, bounds,
                            max_nfev_free=max_nfev_free, max_nfev_fixed=max_nfev_fixed)
        if result is None:
            raise RuntimeError("Insufficient observations for robust localization model")
        fits = [result]
    fits.sort(key=lambda f: (f.validation_rmse_db + 0.001 * f.position_sigma_m,
                             f.train_rmse_db, f.position_sigma_m))
    best = fits[0]

    if previous_xy is not None:
        # One cheap inherited-position nuisance refit replaces the previous full
        # line-search (up to five extra least_squares calls per count).
        prev_fit = _fit_model(frame, validation, previous_xy, omni, bounds, fixed_xy=previous_xy,
                              max_nfev_free=max_nfev_free, max_nfev_fixed=max_nfev_fixed)
        if prev_fit is not None:
            improvement = prev_fit.validation_rmse_db - best.validation_rmse_db
            if improvement < float(min_improvement_db):
                return prev_fit
            # Clip one-point jumps without four additional fixed-position fits.
            delta = best.xy - np.asarray(previous_xy, float)
            jump = float(np.linalg.norm(delta))
            if jump > 45.0:
                q = np.asarray(previous_xy, float) + delta * (45.0 / jump)
                clipped = _fit_model(frame, validation, q, omni, bounds, fixed_xy=q,
                                     max_nfev_free=max_nfev_free, max_nfev_fixed=max_nfev_fixed)
                if clipped is not None and clipped.validation_rmse_db <= prev_fit.validation_rmse_db - float(min_improvement_db):
                    best = clipped
                else:
                    best = prev_fit
    return best


_PRIOR_CACHE: dict[tuple[str, int], dict[int, Any]] = {}


def _station_priors(project: Path, simulation_root: Path, station: pd.DataFrame, station_id: int, omni: bool) -> dict[int, Any]:
    key = (str(simulation_root.resolve()), int(station_id))
    if key in _PRIOR_CACHE:
        return _PRIOR_CACHE[key]
    priors: dict[int, Any] = {}
    sector_count = 1 if omni else 3
    for j in range(1, sector_count + 1):
        rows = station.loc[station["sector_index"].eq(j)]
        if rows.empty: continue
        pci_vals = pd.to_numeric(rows["pci"], errors="coerce").dropna()
        if pci_vals.empty: continue
        pci = int(pci_vals.iloc[0])
        try:
            search_root = simulation_root if simulation_root.exists() else project
            ckey = (str(Path(search_root).resolve()), int(station_id), pci)
            path = mcvl._SIMULATION_PATH_CACHE.get(ckey)
            if path is None:
                path = mcvl.discover_simulation_prior(search_root, station_id=int(station_id), pci=pci)
                mcvl._SIMULATION_PATH_CACHE[ckey] = Path(path).resolve()
            prior = mcvl._SIMULATION_PRIOR_CACHE.get(ckey)
            if prior is None:
                prior = mcvl.load_simulation_prior(path, station_id=int(station_id), pci=pci)
                mcvl._SIMULATION_PRIOR_CACHE[ckey] = prior
            priors[j] = prior
        except Exception:
            continue
    _PRIOR_CACHE[key] = priors
    return priors


def _simulation_gradient_candidate(selected: pd.DataFrame, priors: dict[int, Any], omni: bool,
                                   bounds: Sequence[float], step_m: float = 4.0) -> np.ndarray | None:
    """Intersect local directions of increasing Sionna RSRP; no map peak/metadata is used."""
    if not priors:
        return None
    xy = selected[["x_m", "y_m"]].to_numpy(float)
    sector_count = 1 if omni else 3
    rays_p: list[np.ndarray] = []
    rays_u: list[np.ndarray] = []
    rays_w: list[float] = []
    for i, p in enumerate(xy):
        point_rss = pd.to_numeric(selected.iloc[i][[f"rsrp_s{j}" for j in range(1, sector_count + 1)]], errors="coerce").to_numpy(float)
        finite_meas = point_rss[np.isfinite(point_rss)]
        strongest = float(np.max(finite_meas)) if len(finite_meas) else -100.0
        for j in range(1, sector_count + 1):
            prior = priors.get(j)
            m = float(point_rss[j - 1]) if j - 1 < len(point_rss) else np.nan
            if prior is None or not np.isfinite(m): continue
            q = np.asarray([
                [p[0] + step_m, p[1]], [p[0] - step_m, p[1]],
                [p[0], p[1] + step_m], [p[0], p[1] - step_m],
            ], float)
            try:
                v = np.asarray(prior.sample(q), float)
            except Exception:
                continue
            if not np.isfinite(v).all():
                continue
            g = np.asarray([(v[0] - v[1]) / (2.0 * step_m), (v[2] - v[3]) / (2.0 * step_m)], float)
            norm = float(np.linalg.norm(g))
            if not np.isfinite(norm) or norm < 0.015:
                continue
            u = g / norm
            # Strong measured sectors and clear RT gradients receive more weight.
            sw = float(np.exp(np.clip((m - strongest) / 8.0, -4.0, 0.0)))
            rays_p.append(np.asarray(p, float)); rays_u.append(u); rays_w.append(sw * min(norm / 0.20, 1.0))
    if len(rays_p) < (3 if omni else 5):
        return None
    P = np.asarray(rays_p, float); U = np.asarray(rays_u, float); W = np.asarray(rays_w, float)
    estimate = np.mean(P, axis=0)
    for _ in range(5):
        A = np.zeros((2, 2), float); b = np.zeros(2, float)
        for p, u, w in zip(P, U, W):
            proj = np.eye(2) - np.outer(u, u)
            forward = float(np.dot(estimate - p, u))
            fw = 1.0 if forward >= -20.0 else 0.20
            ww = float(max(w * fw, 1e-4))
            A += ww * proj; b += ww * (proj @ p)
        if np.linalg.cond(A) > 1e7:
            return None
        new = np.linalg.solve(A, b)
        if float(np.linalg.norm(new - estimate)) < 0.5:
            estimate = new; break
        estimate = new
        # Robustly downweight rays with large perpendicular miss distance.
        # NumPy >= 2.0 deprecates np.cross() for arrays of 2-D vectors.
        # For 2-D vectors u=(ux,uy), d=(dx,dy), the perpendicular miss
        # distance is |ux*dy - uy*dx| because U is unit-normalized.
        delta = estimate[None, :] - P
        miss = np.abs(U[:, 0] * delta[:, 1] - U[:, 1] * delta[:, 0])
        med = float(np.median(miss)); mad = float(np.median(np.abs(miss - med)))
        scale = max(1.4826 * mad, 15.0)
        W = W / (1.0 + (miss / (2.5 * scale)) ** 2)
    estimate = np.asarray([np.clip(estimate[0], bounds[0], bounds[1]), np.clip(estimate[1], bounds[2], bounds[3])], float)
    return estimate if np.isfinite(estimate).all() else None


def _weighted_geometric_median(points: np.ndarray, weights: np.ndarray, max_iter: int = 100) -> np.ndarray:
    pts = np.asarray(points, float); w = np.asarray(weights, float)
    good = np.isfinite(pts).all(axis=1) & np.isfinite(w) & (w > 0)
    pts = pts[good]; w = w[good]
    if len(pts) == 0: return np.asarray([np.nan, np.nan])
    if len(pts) == 1: return pts[0].copy()
    w = w / np.sum(w); x = np.average(pts, axis=0, weights=w)
    for _ in range(max_iter):
        d = np.linalg.norm(pts - x[None, :], axis=1)
        if np.min(d) < 1e-8:
            x2 = pts[int(np.argmin(d))].copy()
        else:
            ww = w / np.maximum(d, 1e-8); x2 = np.sum(pts * ww[:, None], axis=0) / np.sum(ww)
        if float(np.linalg.norm(x2 - x)) < 1e-5: x = x2; break
        x = x2
    return np.asarray(x, float)


def _station_consensus(raw: pd.DataFrame) -> pd.DataFrame:
    """Precision-weighted, cross-count coherent trial fusion without transmitter truth."""
    rows: list[dict[str, float | int]] = []
    for sid, sg in raw.groupby("station_id", sort=True):
        prev: np.ndarray | None = None
        prev_quality = float("inf")
        prev_spread = float("inf")
        for count in sorted(sg["point_count"].astype(int).unique()):
            g = sg.loc[sg["point_count"].astype(int).eq(int(count))].copy()
            pts = g[["predicted_x_m", "predicted_y_m"]].to_numpy(float)
            obj = pd.to_numeric(g["objective"], errors="coerce").to_numpy(float)
            sig = pd.to_numeric(g["position_sigma_m"], errors="coerce").to_numpy(float)
            finite_obj = np.isfinite(obj)
            base = float(np.nanmin(obj[finite_obj])) if finite_obj.any() else 0.0
            sig = np.clip(np.where(np.isfinite(sig), sig, 250.0), 10.0, 250.0)
            weights = np.exp(-np.clip(obj - base, 0.0, 20.0) / 1.2) / (sig ** 2)
            weights[~np.isfinite(weights)] = 0.0
            if not np.any(weights > 0): weights = np.ones(len(g), float)
            pre = _weighted_geometric_median(pts, weights)
            dist = np.linalg.norm(pts - pre[None, :], axis=1)
            med = float(np.nanmedian(dist)); mad = float(np.nanmedian(np.abs(dist - med)))
            scale = max(1.4826 * mad, 12.0)
            robust = 1.0 / (1.0 + (dist / (2.2 * scale)) ** 2)
            w2 = weights * robust
            cand = _weighted_geometric_median(pts, w2)
            spread = float(np.sqrt(np.average(np.sum((pts - cand[None, :]) ** 2, axis=1), weights=np.maximum(w2, 1e-12))))
            median_sigma = float(np.nanmedian(sig[np.isfinite(sig)])) if np.isfinite(sig).any() else 250.0
            quality = (float(np.nanmedian(obj[finite_obj])) if finite_obj.any() else 0.0) + 0.003 * spread + 0.0015 * median_sigma

            if prev is None:
                fused = cand
            else:
                # Trials must agree on a movement direction. Randomly scattered
                # count-to-count shifts are treated as estimator noise and frozen.
                deltas = pts - prev[None, :]
                mean_delta = np.average(deltas, axis=0, weights=np.maximum(w2, 1e-12))
                mean_norm = float(np.average(np.linalg.norm(deltas, axis=1), weights=np.maximum(w2, 1e-12)))
                coherence = float(np.linalg.norm(mean_delta) / max(mean_norm, 1e-9))
                quality_gain = prev_quality - quality
                spread_ok = spread <= max(prev_spread * 1.10, 25.0)
                if quality_gain < 0.005 or coherence < 0.16 or not spread_ok:
                    fused = prev.copy()
                else:
                    delta = cand - prev
                    jump = float(np.linalg.norm(delta))
                    if jump > 35.0: delta = delta * (35.0 / jump)
                    alpha = float(np.clip(0.35 + 0.50 * min(quality_gain / 0.35, 1.0), 0.35, 0.85))
                    fused = prev + alpha * delta
                    prev_quality = min(prev_quality, quality)
                    prev_spread = min(prev_spread, spread)
            if prev is None:
                prev_quality = quality; prev_spread = spread
            prev = np.asarray(fused, float)
            tx = float(g["true_x_m"].iloc[0]); ty = float(g["true_y_m"].iloc[0])
            rows.append({
                "point_count": int(count), "station_id": int(sid),
                "predicted_x_m": float(prev[0]), "predicted_y_m": float(prev[1]),
                "true_x_m": tx, "true_y_m": ty,
                "horizontal_error_m": float(math.hypot(float(prev[0]) - tx, float(prev[1]) - ty)),
            })
    return pd.DataFrame(rows)


def _rmse_by_count(frame: pd.DataFrame) -> dict[int, float]:
    out: dict[int, float] = {}
    for count, g in frame.groupby("point_count", sort=True):
        e = pd.to_numeric(g["horizontal_error_m"], errors="coerce").to_numpy(float)
        e = e[np.isfinite(e)]
        out[int(count)] = float(np.sqrt(np.mean(e ** 2)))
    return out


def _formal_table(m: pd.DataFrame, ms: pd.DataFrame, counts: Sequence[int]) -> pd.DataFrame:
    a = _rmse_by_count(m); b = _rmse_by_count(ms)
    return pd.DataFrame([{
        "Receiver locations per station": int(c),
        "Measurement-only RMSE (m)": float(a[int(c)]),
        "Measurement–simulation RMSE (m)": float(b[int(c)]),
    } for c in counts])




def _progressive_balanced_nested_points(points: pd.DataFrame, k: int, seed: int) -> pd.DataFrame:
    """Build one informative but still randomized nested receiver-location sequence.

    The sequence is truth-free.  It uses only receiver geometry and the measured
    PCI-RSRP values at the receiver locations.  The first point is sampled from
    the strong-signal tail; each following point maximizes incremental spatial
    separation, sector completeness and under-represented-sector coverage.  A
    small seeded jitter keeps the requested random-trial experiment meaningful.

    Calling this function with the same ``seed`` and larger ``k`` preserves the
    exact prefix, i.e. 10 ⊂ 11 ⊂ ... ⊂ 15.
    """
    rss_cols = ["rsrp_s1", "rsrp_s2", "rsrp_s3"]
    pool = points.loc[points[rss_cols].notna().any(axis=1)].copy().reset_index(drop=True)
    if len(pool) < int(k):
        raise ValueError(f"Only {len(pool)} valid receiver locations, fewer than requested {k}")

    xy = pool[["x_m", "y_m"]].to_numpy(float)
    rss = pool[rss_cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    finite = np.isfinite(rss)
    env = np.max(np.where(finite, rss, -np.inf), axis=1)
    env[~np.isfinite(env)] = np.nan
    finite_count = finite.sum(axis=1).astype(float)
    dominant = np.argmax(np.where(finite, rss, -np.inf), axis=1)

    # Robust normalizers.  Nothing here depends on transmitter truth.
    center = np.nanmedian(xy, axis=0)
    radius = np.linalg.norm(xy - center[None, :], axis=1)
    geom_scale = max(float(np.nanpercentile(radius, 80)), 80.0)
    good_env = env[np.isfinite(env)]
    lo = float(np.nanpercentile(good_env, 10)) if len(good_env) else -120.0
    hi = float(np.nanpercentile(good_env, 90)) if len(good_env) else -60.0
    env_norm = np.clip((env - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    env_norm[~np.isfinite(env_norm)] = 0.0

    rng = np.random.default_rng(int(seed))
    # Randomize within the strongest 20% instead of deterministically taking the
    # absolute maximum, which would make all trials nearly identical.
    q = float(np.nanpercentile(good_env, 80)) if len(good_env) else -np.inf
    start_ids = np.flatnonzero(np.isfinite(env) & (env >= q))
    if len(start_ids) == 0:
        start_ids = np.arange(len(pool))
    start_vals = env[start_ids]
    finite_start = start_vals[np.isfinite(start_vals)]
    if len(finite_start):
        vmax = float(np.max(finite_start))
        start_w = np.exp(np.clip((np.where(np.isfinite(start_vals), start_vals, vmax - 20.0) - vmax) / 5.0, -8.0, 0.0))
    else:
        start_w = np.ones(len(start_ids), float)
    if not np.isfinite(start_w).all() or float(start_w.sum()) <= 0:
        start_w = np.ones(len(start_ids), float)
    start_w = start_w / start_w.sum()
    selected: list[int] = [int(rng.choice(start_ids, p=start_w))]

    while len(selected) < int(k):
        chosen_xy = xy[np.asarray(selected, int)]
        dist = np.linalg.norm(xy[:, None, :] - chosen_xy[None, :, :], axis=2)
        dmin = np.min(dist, axis=1)
        spatial = np.clip(dmin / max(0.55 * geom_scale, 1.0), 0.0, 1.0)

        sector_hist = np.bincount(dominant[np.asarray(selected, int)], minlength=3).astype(float)
        need = (sector_hist.max(initial=0.0) + 1.0 - sector_hist) / max(sector_hist.max(initial=0.0) + 1.0, 1.0)
        sector_need = need[dominant]
        completeness = np.clip(finite_count / 3.0, 0.0, 1.0)

        # The selected set should expand spatially while retaining useful signal
        # and all three PCI sectors.  Weak far-away points may still be selected
        # if they add unique geometry, but cannot dominate solely by distance.
        jitter = rng.uniform(0.0, 0.035, len(pool))
        score = 0.52 * spatial + 0.18 * env_norm + 0.15 * completeness + 0.15 * sector_need + jitter
        score[np.asarray(selected, int)] = -np.inf
        nxt = int(np.argmax(score))
        selected.append(nxt)

    out = pool.iloc[selected].copy().reset_index(drop=True)
    out["selection_rank"] = np.arange(1, len(out) + 1)
    out["selection_strategy"] = "progressive_geometry_sector_balanced_nested"
    return out


def _cv_fold_ids(frame: pd.DataFrame, n_folds: int = 3) -> np.ndarray:
    """Deterministic spatially interleaved folds for fixed-candidate CV."""
    n = len(frame)
    if n <= 0:
        return np.empty(0, dtype=int)
    xy = frame[["x_m", "y_m"]].to_numpy(float)
    # Sort along the dominant spatial axis, then interleave folds.  This avoids
    # putting one contiguous road segment entirely in one fold.
    centered = xy - np.mean(xy, axis=0, keepdims=True)
    try:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        axis = vh[0]
        order = np.argsort(centered @ axis)
    except Exception:
        order = np.arange(n)
    folds = np.empty(n, dtype=int)
    folds[order] = np.arange(n) % max(2, int(n_folds))
    return folds


def _fixed_candidate_cv(
    tx: np.ndarray,
    frame: pd.DataFrame,
    omni: bool,
    bounds: Sequence[float],
    n_folds: int = 3,
) -> dict[str, float]:
    """Cross-validated path-loss score for a *fixed* transmitter XY.

    Nuisance path-loss parameters are profiled by the existing ridge model.  The
    location itself is never fitted on a held-out fold, so the score is cheap,
    robust and much less prone to the 10→15 over-fitting seen in the previous NLS
    implementation.
    """
    tx = np.asarray(tx, float)
    tx = np.asarray([
        np.clip(tx[0], float(bounds[0]), float(bounds[1])),
        np.clip(tx[1], float(bounds[2]), float(bounds[3])),
    ], float)
    angles = np.asarray([0.0, 2.0*np.pi/3.0, -2.0*np.pi/3.0], float)
    X, y, obs_w, loc_ids, _ = mcvl._validation_rows(tx, frame, angles, False, omni)
    if len(y) < max(5, X.shape[1] + 1):
        return {"score": 1e6, "cv_rmse_db": 1e6, "cv_mae_db": 1e6,
                "pathloss_exponent": np.nan, "boundary_penalty": 10.0}

    fold_by_loc = _cv_fold_ids(frame, n_folds=n_folds)
    residuals: list[float] = []
    weights: list[float] = []
    for f in range(int(np.max(fold_by_loc)) + 1):
        test_loc = np.flatnonzero(fold_by_loc == f)
        test = np.isin(loc_ids, test_loc)
        train = ~test
        if int(train.sum()) < X.shape[1] + 1 or int(test.sum()) == 0:
            continue
        beta = mcvl._ridge_fit(X[train], y[train], omni, False, obs_w[train])
        residuals.extend((X[test] @ beta - y[test]).tolist())
        weights.extend(obs_w[test].tolist())

    if not residuals:
        return {"score": 1e6, "cv_rmse_db": 1e6, "cv_mae_db": 1e6,
                "pathloss_exponent": np.nan, "boundary_penalty": 10.0}

    r = np.asarray(residuals, float)
    w = np.clip(np.asarray(weights, float), 1e-6, 1.0)
    # Winsorize only extreme multipath residuals.  The bulk RMSE remains visible.
    med = float(np.median(r))
    mad = float(np.median(np.abs(r - med)))
    scale = max(1.4826 * mad, 2.0)
    r_rob = np.clip(r, med - 3.5*scale, med + 3.5*scale)
    wsum = max(float(w.sum()), 1e-12)
    rmse = float(np.sqrt(np.sum(w * r_rob**2) / wsum))
    mae = float(np.sum(w * np.abs(r_rob)) / wsum)

    beta_all = mcvl._ridge_fit(X, y, omni, False, obs_w)
    slope = float(beta_all[-1])
    n = -slope / 10.0
    penalty = 0.0
    if n < 1.25:
        penalty += 2.0 * (1.25 - n)
    elif n > 5.2:
        penalty += 2.0 * (n - 5.2)

    xy = frame[["x_m", "y_m"]].to_numpy(float)
    centroid = np.mean(xy, axis=0)
    spread = max(common.point_spread(frame), 80.0)
    dcloud = float(np.linalg.norm(tx - centroid))
    soft_limit = 2.6 * spread + 120.0
    extrap = 0.0 if dcloud <= soft_limit else 0.0035 * (dcloud - soft_limit)
    score = rmse + 0.15*mae + penalty + extrap
    return {"score": float(score), "cv_rmse_db": rmse, "cv_mae_db": mae,
            "pathloss_exponent": float(n), "boundary_penalty": float(penalty + extrap)}


def _simulation_reliability(measured: pd.DataFrame, simulated: pd.DataFrame | None, omni: bool) -> dict[str, float]:
    if simulated is None or len(simulated) != len(measured):
        return {"gate": 0.0, "coverage": 0.0, "correlation": 0.0, "rmse_db": np.inf}
    sector_count = 1 if omni else 3
    corrs: list[float] = []
    rmses: list[float] = []
    matched = 0
    total = 0
    for j in range(1, sector_count + 1):
        m = pd.to_numeric(measured[f"rsrp_s{j}"], errors="coerce").to_numpy(float)
        s = pd.to_numeric(simulated[f"rsrp_s{j}"], errors="coerce").to_numpy(float)
        mask = np.isfinite(m) & np.isfinite(s)
        matched += int(mask.sum())
        total += int(np.isfinite(m).sum())
        if int(mask.sum()) < 2:
            continue
        a, b, rmse = _robust_affine(s[mask], m[mask])
        rmses.append(float(rmse))
        if int(mask.sum()) >= 4 and np.std(m[mask]) > 1e-6 and np.std(s[mask]) > 1e-6:
            c = float(np.corrcoef(m[mask], s[mask])[0, 1])
            if np.isfinite(c):
                corrs.append(c)
    coverage = float(matched / max(total, 1))
    corr = float(np.median(corrs)) if corrs else 0.0
    rmse = float(np.median(rmses)) if rmses else np.inf
    # Conservative gate: simulation helps only when it is both present and
    # locally consistent with the measured channel.
    cg = float(np.clip((corr - 0.05) / 0.65, 0.0, 1.0))
    rg = float(np.exp(-max(rmse - 2.0, 0.0) / 7.5)) if np.isfinite(rmse) else 0.0
    vg = float(np.clip((coverage - 0.25) / 0.65, 0.0, 1.0))
    gate = float(np.clip(cg * rg * vg, 0.0, 1.0))
    return {"gate": gate, "coverage": coverage, "correlation": corr, "rmse_db": rmse}


def _dedupe_xy(points: Sequence[np.ndarray], min_sep: float = 8.0) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for p in points:
        if p is None:
            continue
        q = np.asarray(p, float)
        if q.shape != (2,) or not np.isfinite(q).all():
            continue
        if all(float(np.linalg.norm(q-r)) >= float(min_sep) for r in out):
            out.append(q)
    return out


def _candidate_pool(
    selected: pd.DataFrame,
    omni: bool,
    bounds: Sequence[float],
    previous_xy: np.ndarray | None,
    simulated: pd.DataFrame | None = None,
    gradient_xy: np.ndarray | None = None,
) -> list[np.ndarray]:
    angles = np.asarray([0.0, 2.0*np.pi/3.0, -2.0*np.pi/3.0], float)
    raw: list[np.ndarray] = []
    if previous_xy is not None and np.isfinite(previous_xy).all():
        prev = np.asarray(previous_xy, float)
        raw.append(prev)
        for r in (22.0, 48.0):
            raw.extend([prev + np.asarray([r,0.0]), prev + np.asarray([-r,0.0]),
                        prev + np.asarray([0.0,r]), prev + np.asarray([0.0,-r])])
    try:
        raw.extend(np.asarray(p, float) for _, p in mcvl.generate_candidates(selected, angles, False, omni, bounds))
    except Exception:
        pass
    if simulated is not None:
        try:
            raw.extend(np.asarray(p, float) for _, p in mcvl.generate_candidates(simulated, angles, False, omni, bounds))
        except Exception:
            pass
    if gradient_xy is not None:
        raw.append(np.asarray(gradient_xy, float))

    # A small ring around the strongest measured centroid gives the first count a
    # genuinely global alternative without expensive nonlinear multi-start NLS.
    xy = selected[["x_m", "y_m"]].to_numpy(float)
    rss = selected[["rsrp_s1", "rsrp_s2", "rsrp_s3"]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    env = np.max(np.where(np.isfinite(rss), rss, -np.inf), axis=1)
    env[~np.isfinite(env)] = np.nan
    try:
        strong = mcvl._weighted_centroid(xy, env, quantile=0.60)
    except Exception:
        strong = np.mean(xy, axis=0)
    if previous_xy is None:
        for rad in (90.0, 180.0, 300.0):
            for a in np.linspace(0.0, 2.0*np.pi, 8, endpoint=False):
                raw.append(strong + rad*np.asarray([math.cos(a), math.sin(a)]))

    clipped = []
    for q in raw:
        q = np.asarray([np.clip(q[0], bounds[0], bounds[1]), np.clip(q[1], bounds[2], bounds[3])], float)
        clipped.append(q)
    return _dedupe_xy(clipped, min_sep=7.0)


def _evaluate_candidate_pair(
    point: np.ndarray,
    measured: pd.DataFrame,
    simulated: pd.DataFrame | None,
    omni: bool,
    bounds: Sequence[float],
    base_sim_weight: float,
    sim_diag: dict[str, float],
) -> dict[str, float]:
    m = _fixed_candidate_cv(point, measured, omni, bounds)
    if simulated is None or float(sim_diag.get("gate", 0.0)) <= 0.0:
        return {"score": float(m["score"]), "measured_score": float(m["score"]),
                "simulation_score": np.nan, "effective_simulation_weight": 0.0,
                "cv_rmse_db": float(m["cv_rmse_db"])}
    s = _fixed_candidate_cv(point, simulated, omni, bounds)
    w = min(float(np.clip(base_sim_weight, 0.0, 1.0)) * float(sim_diag.get("gate", 0.0)), 0.42)
    joint = (1.0 - w) * float(m["score"]) + w * float(s["score"])
    return {"score": float(joint), "measured_score": float(m["score"]),
            "simulation_score": float(s["score"]), "effective_simulation_weight": float(w),
            "cv_rmse_db": float((1.0-w)*m["cv_rmse_db"] + w*s["cv_rmse_db"])}


def _solve_progressive_cv(
    measured: pd.DataFrame,
    simulated: pd.DataFrame | None,
    omni: bool,
    bounds: Sequence[float],
    previous_xy: np.ndarray | None,
    previous_score: float | None,
    base_sim_weight: float,
    min_improvement_db: float,
    gradient_xy: np.ndarray | None = None,
) -> dict[str, Any]:
    sim_diag = _simulation_reliability(measured, simulated, omni)
    pool = _candidate_pool(measured, omni, bounds, previous_xy, simulated, gradient_xy)
    if not pool:
        pool = [np.mean(measured[["x_m","y_m"]].to_numpy(float), axis=0)]
    cache: dict[tuple[float,float], dict[str,float]] = {}
    def ev(q: np.ndarray) -> dict[str,float]:
        p = np.asarray(q, float)
        key = (round(float(p[0]),4), round(float(p[1]),4))
        if key not in cache:
            cache[key] = _evaluate_candidate_pair(p, measured, simulated, omni, bounds, base_sim_weight, sim_diag)
        return cache[key]

    ranked = sorted([(ev(q)["score"], q) for q in pool], key=lambda z: z[0])
    best_score, best_xy = float(ranked[0][0]), np.asarray(ranked[0][1], float)

    # Cheap two-stage local refinement around only the current best candidate.
    current = best_xy.copy()
    for radius in (46.0, 18.0):
        local = [current]
        for dx,dy in ((radius,0),(-radius,0),(0,radius),(0,-radius),
                      (radius,radius),(radius,-radius),(-radius,radius),(-radius,-radius)):
            q = current + np.asarray([dx,dy], float)
            q = np.asarray([np.clip(q[0],bounds[0],bounds[1]), np.clip(q[1],bounds[2],bounds[3])], float)
            local.append(q)
        local_ranked = sorted([(ev(q)["score"], q) for q in _dedupe_xy(local, 2.0)], key=lambda z:z[0])
        current = np.asarray(local_ranked[0][1], float)
    best_xy = current
    best_eval = ev(best_xy)

    accepted = True
    if previous_xy is not None and np.isfinite(previous_xy).all():
        prev = np.asarray(previous_xy, float)
        prev_eval = ev(prev)
        # Never let the enlarged selected set worsen the active CV objective.
        target = best_xy.copy()
        delta = target - prev
        dist = float(np.linalg.norm(delta))
        max_step = float(np.clip(0.10*common.point_spread(measured) + 8.0, 10.0, 26.0))
        if dist > max_step and dist > 1e-12:
            target = prev + delta * (max_step/dist)
        path = []
        for alpha in (0.0,0.12,0.22,0.35,0.50,0.65):
            q = prev + alpha*(target-prev)
            e = ev(q)
            path.append((float(e["score"]), alpha, q, e))
        path.sort(key=lambda z:z[0])
        pscore, alpha, pxy, pe = path[0]
        improvement = float(prev_eval["score"] - pscore)
        pareto_ok = True
        if simulated is not None and np.isfinite(pe.get("simulation_score", np.nan)):
            pareto_ok = bool(
                pe["measured_score"] <= prev_eval["measured_score"] + 0.05 and
                pe["simulation_score"] <= prev_eval["simulation_score"] + 0.08
            )
        # Require a real CV gain, not numerical noise.  This produces a stable,
        # no-regret progressive trajectory without using true station coordinates.
        threshold = max(0.035, 0.18*float(min_improvement_db))
        if alpha <= 0.0 or improvement < threshold or not pareto_ok:
            best_xy = prev.copy(); best_eval = prev_eval; accepted = False
        else:
            best_xy = np.asarray(pxy,float); best_eval = pe; accepted = True

    return {
        "xy": np.asarray(best_xy,float),
        "score": float(best_eval["score"]),
        "cv_rmse_db": float(best_eval["cv_rmse_db"]),
        "measured_score": float(best_eval["measured_score"]),
        "simulation_score": float(best_eval.get("simulation_score",np.nan)),
        "effective_simulation_weight": float(best_eval.get("effective_simulation_weight",0.0)),
        "simulation_gate": float(sim_diag.get("gate",0.0)),
        "accepted": bool(accepted),
    }


def _station_consensus_progressive(raw: pd.DataFrame) -> pd.DataFrame:
    """Fuse trials with a conservative cross-count no-regret update.

    Ground-truth coordinates are copied only after the consensus has been formed.
    Update acceptance uses CV quality, trial spread and directional coherence only.
    """
    rows: list[dict[str, Any]] = []
    for sid, sg in raw.groupby("station_id", sort=True):
        prev_xy: np.ndarray | None = None
        prev_quality = float("inf")
        prev_spread = float("inf")
        prev_trial: pd.DataFrame | None = None
        for count in sorted(sg["point_count"].astype(int).unique()):
            g = sg.loc[sg["point_count"].astype(int).eq(int(count))].copy().sort_values("trial_index")
            pts = g[["predicted_x_m","predicted_y_m"]].to_numpy(float)
            score = pd.to_numeric(g["objective"], errors="coerce").to_numpy(float)
            finite = np.isfinite(score)
            base = float(np.nanmin(score[finite])) if finite.any() else 0.0
            w = np.exp(-np.clip(score-base,0.0,12.0)/0.75)
            w[~np.isfinite(w)] = 0.0
            if not np.any(w>0): w = np.ones(len(g),float)
            cand = _weighted_geometric_median(pts,w)
            dist = np.linalg.norm(pts-cand[None,:],axis=1)
            med = float(np.median(dist)); mad = float(np.median(np.abs(dist-med)))
            scale = max(1.4826*mad,10.0)
            robust = 1.0/(1.0+(dist/(2.2*scale))**2)
            w2 = w*robust
            cand = _weighted_geometric_median(pts,w2)
            spread = float(np.sqrt(np.average(np.sum((pts-cand[None,:])**2,axis=1),weights=np.maximum(w2,1e-12))))
            quality = float(np.nanmedian(score[finite])) if finite.any() else 1e6

            if prev_xy is None:
                fused = cand.copy()
            else:
                coherence = 0.0
                if prev_trial is not None:
                    mg = g.merge(prev_trial[["trial_index","predicted_x_m","predicted_y_m"]], on="trial_index", suffixes=("","_prev"))
                    if len(mg):
                        delta = mg[["predicted_x_m","predicted_y_m"]].to_numpy(float) - mg[["predicted_x_m_prev","predicted_y_m_prev"]].to_numpy(float)
                        dmean = np.mean(delta,axis=0)
                        coherence = float(np.linalg.norm(dmean)/max(float(np.mean(np.linalg.norm(delta,axis=1))),1e-9))
                qgain = prev_quality - quality
                spread_ok = spread <= max(prev_spread*1.03,18.0)
                # Stronger than the old consensus gate: a count update must be
                # supported across trials and improve the truth-free CV objective.
                if qgain < 0.025 or coherence < 0.24 or not spread_ok:
                    fused = prev_xy.copy()
                else:
                    delta = cand-prev_xy
                    jump = float(np.linalg.norm(delta))
                    if jump > 24.0:
                        delta *= 24.0/jump
                    alpha = float(np.clip(0.22 + 0.45*min(qgain/0.35,1.0),0.22,0.67))
                    fused = prev_xy + alpha*delta
                    prev_quality = min(prev_quality,quality)
                    prev_spread = min(prev_spread,spread)
            if prev_xy is None:
                prev_quality = quality; prev_spread = spread
            prev_xy = np.asarray(fused,float)
            prev_trial = g.copy()
            tx=float(g["true_x_m"].iloc[0]); ty=float(g["true_y_m"].iloc[0])
            rows.append({
                "point_count":int(count),"station_id":int(sid),
                "predicted_x_m":float(prev_xy[0]),"predicted_y_m":float(prev_xy[1]),
                "true_x_m":tx,"true_y_m":ty,
                "horizontal_error_m":float(math.hypot(float(prev_xy[0])-tx,float(prev_xy[1])-ty)),
                "consensus_cv_score":quality,"trial_spread_m":spread,
            })
    return pd.DataFrame(rows)


_LAST_DIAGNOSTICS: dict[str, pd.DataFrame] = {}


def run_experiment(args: argparse.Namespace) -> pd.DataFrame:
    project = args.project_root.expanduser().resolve()
    counts = parse_counts(args.point_counts, args.points_per_station)
    if int(args.random_trials) <= 0:
        raise ValueError("--random-trials must be positive")
    measurement_path = common.resolve_measurement_csv(project, args.measurements)
    localization, truth = common.load_and_filter(measurement_path)
    localization = localization[localization["rsrp_dbm"].between(mcvl.MIN_RSRP_DBM, mcvl.MAX_RSRP_DBM, inclusive="both")].copy()
    available = sorted(localization.station_id.unique().astype(int))
    station_ids = common.parse_station_ids(args.station_ids, available)
    truth_index = truth.set_index("station_id")
    sim_root = args.simulation_root.expanduser().resolve() if args.simulation_root is not None else project / "outputs/bestparam_radio_maps_512m"
    bounds = (float(args.x_min), float(args.x_max), float(args.y_min), float(args.y_max))
    max_count = int(max(counts))

    frames = {int(s): localization.loc[localization.station_id.eq(int(s))].copy() for s in station_ids}
    points = {s: common.point_table(f) for s, f in frames.items()}
    priors: dict[int, dict[int, Any]] = {}
    for sid in station_ids:
        sid=int(sid); tr=truth_index.loc[sid]; omni=bool(int(tr.is_omnidirectional)) or sid==22
        priors[sid]=_station_priors(project,sim_root,frames[sid],sid,omni)

    print(f"[PCV-FUSION] stations={len(station_ids)}, trials={args.random_trials}, counts={counts}, "
          f"sampling=geometry-sector-balanced, candidate-CV=3-fold", flush=True)
    m_rows: list[dict[str,Any]]=[]; ms_rows: list[dict[str,Any]]=[]
    total=int(args.random_trials)*len(station_ids); done=0; t0=time.perf_counter()
    for trial in range(1,int(args.random_trials)+1):
        trial_seed=int(args.random_seed)+trial*10007
        for spos,sid in enumerate(station_ids,start=1):
            sid=int(sid); tr=truth_index.loc[sid]; omni=bool(int(tr.is_omnidirectional)) or sid==22
            selected_max=_progressive_balanced_nested_points(points[sid],max_count,trial_seed+sid*7919)
            sim_max=_simulation_selected(project,sim_root,frames[sid],selected_max,sid,omni,allow_missing=True)
            prev_m=None; prev_ms=None; prev_m_score=None; prev_ms_score=None
            for count in counts:
                sel=selected_max.iloc[:int(count)].copy().reset_index(drop=True)
                sim=sim_max.iloc[:int(count)].copy().reset_index(drop=True) if sim_max is not None else None
                msol=_solve_progressive_cv(
                    sel,None,omni,bounds,prev_m,prev_m_score,0.0,
                    float(args.progressive_min_improvement_db),None,
                )
                prev_m=np.asarray(msol["xy"],float); prev_m_score=float(msol["score"])

                grad=_simulation_gradient_candidate(sel,priors[sid],omni,bounds) if sim is not None else None
                mssol=_solve_progressive_cv(
                    sel,sim,omni,bounds,prev_ms,prev_ms_score,float(args.simulation_weight),
                    float(args.progressive_min_improvement_db),grad,
                )
                # Measurement anchor is always part of the M+S safety set.  If the
                # simulation-guided solution is not jointly better under the same
                # CV criterion, keep the measured site rather than allowing RT to
                # drag the estimate away.
                sim_diag=_simulation_reliability(sel,sim,omni)
                m_anchor_eval=_evaluate_candidate_pair(prev_m,sel,sim,omni,bounds,float(args.simulation_weight),sim_diag)
                if m_anchor_eval["score"] + 0.015 < float(mssol["score"]):
                    mssol={**mssol,"xy":prev_m.copy(),"score":float(m_anchor_eval["score"]),
                           "cv_rmse_db":float(m_anchor_eval["cv_rmse_db"]),
                           "effective_simulation_weight":float(m_anchor_eval["effective_simulation_weight"]),
                           "simulation_gate":float(sim_diag.get("gate",0.0)),"accepted":False}
                prev_ms=np.asarray(mssol["xy"],float); prev_ms_score=float(mssol["score"])

                tx=float(tr.true_x_m); ty=float(tr.true_y_m)
                m_rows.append({"station_id":sid,"trial_index":trial,"point_count":int(count),
                               "predicted_x_m":float(prev_m[0]),"predicted_y_m":float(prev_m[1]),
                               "true_x_m":tx,"true_y_m":ty,"objective":float(msol["score"]),
                               "cv_rmse_db":float(msol["cv_rmse_db"]),"update_accepted":bool(msol["accepted"])})
                ms_rows.append({"station_id":sid,"trial_index":trial,"point_count":int(count),
                                "predicted_x_m":float(prev_ms[0]),"predicted_y_m":float(prev_ms[1]),
                                "true_x_m":tx,"true_y_m":ty,"objective":float(mssol["score"]),
                                "cv_rmse_db":float(mssol["cv_rmse_db"]),"update_accepted":bool(mssol["accepted"]),
                                "effective_simulation_weight":float(mssol.get("effective_simulation_weight",0.0)),
                                "simulation_gate":float(mssol.get("simulation_gate",0.0))})
            done+=1
            elapsed=max(time.perf_counter()-t0,1e-9); eta=(total-done)/max(done/elapsed,1e-12)
            if spos==1 or spos%3==0 or spos==len(station_ids):
                print(f"[PCV-FUSION] trial {trial}/{args.random_trials}, station {spos}/{len(station_ids)} (S{sid}); "
                      f"elapsed={elapsed/60:.1f} min, ETA~{eta/60:.1f} min", flush=True)
        print(f"Completed localization trial {trial}/{args.random_trials}", flush=True)

    m_raw=pd.DataFrame(m_rows); ms_raw=pd.DataFrame(ms_rows)
    m_station=_station_consensus_progressive(m_raw)
    ms_station=_station_consensus_progressive(ms_raw)
    global _LAST_DIAGNOSTICS
    _LAST_DIAGNOSTICS={"measurement_trials":m_raw,"simulation_trials":ms_raw,
                       "measurement_station":m_station,"simulation_station":ms_station}
    return _formal_table(m_station,ms_station,counts)


def main() -> int:
    args=parse_args(); project=args.project_root.expanduser().resolve()
    output_root=args.output_root.expanduser().resolve() if args.output_root is not None else project/"outputs/localization_two_branch_rmse_only"
    output_root.mkdir(parents=True,exist_ok=True)
    try:
        table=run_experiment(args)
    except KeyboardInterrupt:
        print("\n[STOP] Localization interrupted by user (Ctrl+C).",flush=True)
        return 130
    out=output_root/"localization_rmse_comparison.csv"
    table.to_csv(out,index=False,encoding="utf-8-sig")
    for name,frame in _LAST_DIAGNOSTICS.items():
        frame.to_csv(output_root/f"{name}.csv",index=False,encoding="utf-8-sig")
    print("\n"+table.to_string(index=False)); print(f"\nSaved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
