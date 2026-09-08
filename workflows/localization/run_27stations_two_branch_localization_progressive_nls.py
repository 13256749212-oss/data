#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Two-branch progressive sparse base-station localization.

Branch M   : measurement-only robust profiled RSS geometry localization.
Branch M+S : the exact same sparse measured receiver locations plus collocated
             Sionna RT RSRP and *local Sionna field gradients* sampled around
             those locations.  The full simulation-map center/extent/peak and
             transmitter metadata are never used as a location prior.

The design is intentionally truth-free during localization. Surveyed transmitter
coordinates are attached only after all trial/station estimates are finalized,
solely to compute the requested RMSE table.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
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

ALGORITHM_NAME = "Progressive Robust Profiled RSS + Collocated Sionna Gradient Fusion"
RSS_COLS = ["rsrp_s1", "rsrp_s2", "rsrp_s3"]
VERTICAL_M = 28.5


@dataclass
class GradientEvidence:
    points: np.ndarray
    directions: np.ndarray
    weights: np.ndarray
    intersection: np.ndarray | None
    line_rms_m: float
    support_count: int


def parse_counts(text: str | None, single: int | None) -> list[int]:
    if single is not None:
        return [int(single)]
    raw = "10,11,12,13,14,15" if text is None else str(text)
    out = sorted({int(x.strip()) for x in raw.split(",") if x.strip()})
    if not out or min(out) <= 0:
        raise ValueError("--point-counts must contain positive integers")
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Measurement-only vs measurement+Sionna sparse localization")
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
    p.add_argument("--gradient-step-m", type=float, default=8.0)
    p.add_argument("--gradient-min-db-per-m", type=float, default=0.015)
    p.add_argument("--gradient-min-lines", type=int, default=4)
    p.add_argument("--progressive-min-improvement-db", type=float, default=0.05)
    p.add_argument("--x-min", type=float, default=mcvl.DEFAULT_BOUNDS[0])
    p.add_argument("--x-max", type=float, default=mcvl.DEFAULT_BOUNDS[1])
    p.add_argument("--y-min", type=float, default=mcvl.DEFAULT_BOUNDS[2])
    p.add_argument("--y-max", type=float, default=mcvl.DEFAULT_BOUNDS[3])
    # Backward-compatible flags accepted by run_pipeline.py.
    p.add_argument("--validation-points", type=int, default=0)
    p.add_argument("--validation-min-improvement-db", type=float, default=0.0)
    p.add_argument("--simulation-measured-tolerance-db", type=float, default=0.0)
    p.add_argument("--fast-max-nfev", type=int, default=36)
    p.add_argument("--fast-fixed-nfev", type=int, default=14)
    p.add_argument("--fast-initial-starts", type=int, default=3)
    p.add_argument("--simulation-mode", choices=["compare"], default="compare")
    p.add_argument("--direction-prior-mode", choices=["off", "fixed", "soft"], default="off")
    p.add_argument("--directions", type=Path, default=None)
    p.add_argument("--bootstrap", type=int, default=0)
    p.add_argument("--de-maxiter", type=int, default=0)
    p.add_argument("--de-popsize", type=int, default=0)
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--skip-figures", action="store_true")
    p.add_argument("--keep-trial-figures", action="store_true")
    p.add_argument("--continue-on-error", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--consensus-mad-z", type=float, default=2.5)
    p.add_argument("--consensus-min-inlier-fraction", type=float, default=0.55)
    p.add_argument("--consensus-min-scale-m", type=float, default=5.0)
    p.add_argument("--skip-per-station-figures", action="store_true")
    return p.parse_args()


def _normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, float)
    finite = np.isfinite(v)
    out = np.zeros_like(v, dtype=float)
    if not finite.any():
        return out
    lo = float(np.nanpercentile(v[finite], 5))
    hi = float(np.nanpercentile(v[finite], 95))
    if hi <= lo + 1e-12:
        out[finite] = 0.5
    else:
        out[finite] = np.clip((v[finite] - lo) / (hi - lo), 0.0, 1.0)
    return out


def _nested_information_sequence(points: pd.DataFrame, max_count: int, seed: int) -> pd.DataFrame:
    """Seeded, nested, information-balanced receiver sequence; no transmitter truth."""
    if len(points) < max_count:
        raise ValueError(f"unique receiver locations={len(points)} < requested {max_count}")
    work = points.copy().reset_index(drop=True)
    rss = work[RSS_COLS].to_numpy(float)
    finite = np.isfinite(rss)
    coverage = finite.sum(axis=1).astype(float)
    safe = np.where(finite, rss, np.nan)
    strongest = np.nanmax(safe, axis=1)
    weakest = np.nanmin(safe, axis=1)
    span = np.where(np.isfinite(strongest - weakest), strongest - weakest, 0.0)
    strength = _normalize(strongest)
    span_n = _normalize(span)
    coverage_n = coverage / max(float(np.nanmax(coverage)), 1.0)
    xy = work[["x_m", "y_m"]].to_numpy(float)
    rng = np.random.default_rng(int(seed))
    jitter = rng.uniform(0.0, 1.0, len(work))

    base = 0.45 * coverage_n + 0.35 * strength + 0.20 * span_n
    # Random trials differ only by a modest tie-breaking component; poor points
    # are never selected merely because of the seed.
    first_score = base + 0.08 * jitter
    selected = [int(np.argmax(first_score))]

    while len(selected) < max_count:
        remaining = np.asarray([i for i in range(len(work)) if i not in selected], int)
        d = np.sqrt(((xy[remaining, None, :] - xy[np.asarray(selected)][None, :, :]) ** 2).sum(axis=2))
        min_d = d.min(axis=1)
        spatial = _normalize(min_d)

        # Prefer a point whose dominant sector is underrepresented so the three
        # physical sectors remain identifiable with very sparse measurements.
        dom_all = np.argmax(np.where(finite, rss, -np.inf), axis=1)
        selected_dom = dom_all[np.asarray(selected)]
        counts = np.bincount(selected_dom, minlength=3).astype(float)
        sector_need = 1.0 / (1.0 + counts[dom_all[remaining]])
        sector_need = _normalize(sector_need)

        # RSRP-vector novelty supplies non-collinear propagation information.
        fill = np.where(finite, rss, -120.0)
        diff = np.sqrt(((fill[remaining, None, :] - fill[np.asarray(selected)][None, :, :]) ** 2).sum(axis=2))
        novelty = _normalize(diff.min(axis=1))
        score = (
            0.42 * spatial + 0.20 * sector_need + 0.18 * novelty
            + 0.12 * coverage_n[remaining] + 0.08 * strength[remaining]
            + 0.025 * jitter[remaining]
        )
        selected.append(int(remaining[int(np.argmax(score))]))
    out = work.iloc[selected].reset_index(drop=True)
    out["selection_rank"] = np.arange(1, len(out) + 1, dtype=int)
    out["selection_strategy"] = "randomized_information_balanced_nested"
    return out


def _weighted_centroid(xy: np.ndarray, values: np.ndarray) -> np.ndarray:
    v = np.asarray(values, float)
    finite = np.isfinite(v)
    if not finite.any():
        return np.mean(xy, axis=0)
    vmax = float(np.nanmax(v[finite]))
    w = np.zeros_like(v, float)
    w[finite] = np.exp(np.clip((v[finite] - vmax) / 9.0, -12.0, 0.0))
    return np.average(xy, axis=0, weights=np.maximum(w, 1e-8))


def _plane_gradient(xy: np.ndarray, values: np.ndarray) -> tuple[np.ndarray | None, float]:
    mask = np.isfinite(values)
    if int(mask.sum()) < 4:
        return None, 0.0
    X = np.column_stack([np.ones(mask.sum()), xy[mask, 0], xy[mask, 1]])
    y = values[mask]
    w = np.ones(len(y), float)
    beta = np.zeros(3, float)
    for _ in range(4):
        sw = np.sqrt(np.maximum(w, 1e-8))
        beta = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)[0]
        resid = y - X @ beta
        med = float(np.median(resid))
        mad = max(1.4826 * float(np.median(np.abs(resid - med))), 1.0)
        w = 1.0 / (1.0 + (resid / (2.5 * mad)) ** 2)
    g = beta[1:3]
    gn = float(np.linalg.norm(g))
    if gn < 1e-8:
        return None, 0.0
    pred = X @ beta
    ss = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum((y - pred) ** 2)) / max(ss, 1e-9)
    return g / gn, float(np.clip(r2, 0.0, 1.0))


def _robust_line_intersection(points: np.ndarray, directions: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray | None, float]:
    if len(points) < 2:
        return None, float("inf")
    p = np.asarray(points, float)
    u = np.asarray(directions, float)
    w0 = np.asarray(weights, float)
    w = np.maximum(w0, 1e-5)
    estimate = np.average(p, axis=0, weights=w)
    for _ in range(8):
        A = np.zeros((2, 2), float)
        b = np.zeros(2, float)
        for pi, ui, wi in zip(p, u, w):
            ui = ui / max(float(np.linalg.norm(ui)), 1e-12)
            Q = np.eye(2) - np.outer(ui, ui)
            A += wi * Q
            b += wi * (Q @ pi)
        try:
            estimate = np.linalg.solve(A + 1e-8 * np.eye(2), b)
        except np.linalg.LinAlgError:
            return None, float("inf")
        delta = estimate[None, :] - p
        miss = np.abs(u[:, 0] * delta[:, 1] - u[:, 1] * delta[:, 0])
        scale = max(1.4826 * float(np.median(np.abs(miss - np.median(miss)))), 8.0)
        w = w0 / (1.0 + (miss / (2.5 * scale)) ** 2)
    delta = estimate[None, :] - p
    miss = np.abs(u[:, 0] * delta[:, 1] - u[:, 1] * delta[:, 0])
    rms = float(np.sqrt(np.average(miss ** 2, weights=np.maximum(w, 1e-8))))
    return estimate, rms


def _measurement_gradient_seed(selected: pd.DataFrame) -> np.ndarray | None:
    xy = selected[["x_m", "y_m"]].to_numpy(float)
    lines_p, lines_u, lines_w = [], [], []
    for j, col in enumerate(RSS_COLS):
        vals = selected[col].to_numpy(float)
        g, rel = _plane_gradient(xy, vals)
        if g is None or rel < 0.05:
            continue
        center = _weighted_centroid(xy, vals)
        lines_p.append(center)
        lines_u.append(g)
        lines_w.append(0.25 + rel)
    if len(lines_p) < 2:
        return None
    out, _ = _robust_line_intersection(np.asarray(lines_p), np.asarray(lines_u), np.asarray(lines_w))
    return out


def _profile_rss_score(tx: np.ndarray, selected: pd.DataFrame) -> tuple[float, dict]:
    """Profile sector intercepts and a shared path-loss slope at candidate XY."""
    tx = np.asarray(tx, float)
    xy = selected[["x_m", "y_m"]].to_numpy(float)
    rss = selected[RSS_COLS].to_numpy(float)
    d = np.sqrt(np.sum((xy - tx[None, :]) ** 2, axis=1) + VERTICAL_M ** 2)
    z = np.log10(np.maximum(d, 1.0))

    rows, y, sid = [], [], []
    for i in range(len(selected)):
        for s in range(3):
            if np.isfinite(rss[i, s]):
                rows.append([1.0 if s == 0 else 0.0, 1.0 if s == 1 else 0.0, 1.0 if s == 2 else 0.0, z[i]])
                y.append(float(rss[i, s])); sid.append(s)
    if len(y) < 6:
        return float("inf"), {}
    X = np.asarray(rows, float); yv = np.asarray(y, float)
    w = np.ones(len(yv), float)
    beta = np.zeros(4, float)
    for _ in range(5):
        sw = np.sqrt(np.maximum(w, 1e-8))
        beta = np.linalg.lstsq(X * sw[:, None], yv * sw, rcond=None)[0]
        resid = X @ beta - yv
        med = float(np.median(resid)); mad = max(1.4826 * float(np.median(np.abs(resid - med))), 1.5)
        w = 1.0 / (1.0 + (resid / (2.5 * mad)) ** 2)
    resid = X @ beta - yv
    robust_rmse = float(np.sqrt(np.average(np.minimum(resid ** 2, (3.5 * max(np.median(np.abs(resid)), 2.0)) ** 2), weights=w)))
    n = -float(beta[3]) / 10.0
    n_pen = 0.0
    if n < 1.0:
        n_pen += 3.0 * (1.0 - n)
    elif n > 6.0:
        n_pen += 2.0 * (n - 6.0)

    # Rank consistency is robust to unknown PCI transmit offsets.
    rank_pen = 0.0; usable = 0
    for s in range(3):
        vals = rss[:, s]
        mask = np.isfinite(vals)
        if int(mask.sum()) < 4:
            continue
        rr = pd.Series(vals[mask]).rank().to_numpy(float)
        dd = pd.Series(-z[mask]).rank().to_numpy(float)
        corr = float(np.corrcoef(rr, dd)[0, 1]) if len(rr) > 2 else 0.0
        if np.isfinite(corr):
            rank_pen += 1.2 * max(0.0, 0.15 - corr)
            usable += 1
    if usable:
        rank_pen /= usable
    return robust_rmse + n_pen + rank_pen, {"rss_rmse_db": robust_rmse, "pathloss_exponent": n}



def _profile_rss_cv_score(tx: np.ndarray, selected: pd.DataFrame) -> float:
    """Three-fold receiver-location CV of the profiled distance model.

    Candidate generation may use all selected locations, but progressive
    accept/reject uses this out-of-fold score, which is much harder to improve by
    simply over-fitting one newly added receiver point.
    """
    if len(selected) < 6:
        return _profile_rss_score(tx, selected)[0]
    tx=np.asarray(tx,float)
    xy=selected[["x_m","y_m"]].to_numpy(float)
    rss=selected[RSS_COLS].to_numpy(float)
    d=np.sqrt(np.sum((xy-tx[None,:])**2,axis=1)+VERTICAL_M**2)
    z=np.log10(np.maximum(d,1.0))
    if "selection_rank" in selected.columns:
        rank = pd.to_numeric(selected["selection_rank"], errors="coerce").to_numpy(float)
        fallback = np.arange(len(selected), dtype=float) + 1.0
        rank = np.where(np.isfinite(rank), rank, fallback)
        fold = (rank.astype(int) - 1) % 3
    else:
        fold=np.arange(len(selected))%3
    errs=[]
    for f in range(3):
        train_loc=fold!=f; val_loc=fold==f
        if int(train_loc.sum())<4 or int(val_loc.sum())<1:
            continue
        rows=[]; yy=[]
        for i in np.flatnonzero(train_loc):
            for s in range(3):
                if np.isfinite(rss[i,s]):
                    rows.append([1.0 if s==0 else 0.0,1.0 if s==1 else 0.0,1.0 if s==2 else 0.0,z[i]])
                    yy.append(float(rss[i,s]))
        if len(yy)<6:
            continue
        X=np.asarray(rows,float); y=np.asarray(yy,float); w=np.ones(len(y),float)
        beta=np.zeros(4,float)
        for _ in range(4):
            sw=np.sqrt(np.maximum(w,1e-8))
            beta=np.linalg.lstsq(X*sw[:,None],y*sw,rcond=None)[0]
            resid=X@beta-y
            med=float(np.median(resid)); mad=max(1.4826*float(np.median(np.abs(resid-med))),1.5)
            w=1.0/(1.0+(resid/(2.5*mad))**2)
        val_res=[]
        for i in np.flatnonzero(val_loc):
            for s in range(3):
                if np.isfinite(rss[i,s]):
                    feat=np.asarray([1.0 if s==0 else 0.0,1.0 if s==1 else 0.0,1.0 if s==2 else 0.0,z[i]])
                    val_res.append(float(feat@beta-rss[i,s]))
        if val_res:
            vr=np.asarray(val_res,float)
            cap=max(3.5*float(np.median(np.abs(vr))),7.0)
            errs.extend(np.minimum(np.abs(vr),cap).tolist())
    if not errs:
        return _profile_rss_score(tx, selected)[0]
    rmse=float(np.sqrt(np.mean(np.asarray(errs,float)**2)))
    # Keep a mild physical slope regularizer from the all-data fit.
    train,diag=_profile_rss_score(tx,selected)
    n=float(diag.get("pathloss_exponent",2.7)) if diag else 2.7
    npen=0.0 if 1.0<=n<=6.0 else 2.0*min(abs(n-1.0),abs(n-6.0))
    return 0.82*rmse+0.18*train+npen

def _clip(p: np.ndarray, bounds: Sequence[float]) -> np.ndarray:
    x0, x1, y0, y1 = map(float, bounds)
    return np.asarray([np.clip(p[0], x0, x1), np.clip(p[1], y0, y1)], float)


def _dedupe(points: list[np.ndarray], sep: float = 8.0) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for p in points:
        p = np.asarray(p, float)
        if np.isfinite(p).all() and all(np.linalg.norm(p - q) >= sep for q in out):
            out.append(p)
    return out


def _measurement_solver(selected: pd.DataFrame, bounds: Sequence[float], previous: np.ndarray | None) -> tuple[np.ndarray, float]:
    xy = selected[["x_m", "y_m"]].to_numpy(float)
    rss = selected[RSS_COLS].to_numpy(float)
    env = np.nanmax(np.where(np.isfinite(rss), rss, np.nan), axis=1)
    centroid = np.mean(xy, axis=0)
    strong = _weighted_centroid(xy, env)
    candidates: list[np.ndarray] = [centroid, strong]
    gseed = _measurement_gradient_seed(selected)
    if gseed is not None:
        candidates.append(gseed)
    if previous is not None:
        candidates.insert(0, previous)

    spread = max(common.point_spread(selected), 100.0)
    for center in [strong, centroid]:
        for radius in [80.0, 160.0, min(320.0, 1.15 * spread)]:
            for a in np.linspace(0, 2 * np.pi, 8, endpoint=False):
                candidates.append(center + radius * np.asarray([math.cos(a), math.sin(a)]))
    candidates = [_clip(p, bounds) for p in _dedupe(candidates)]

    cache: dict[tuple[float, float], float] = {}
    def score(p: np.ndarray) -> float:
        key = (round(float(p[0]), 3), round(float(p[1]), 3))
        if key not in cache:
            cache[key] = _profile_rss_score(p, selected)[0]
        return cache[key]

    candidates.sort(key=score)
    current = candidates[0]
    # Coarse-to-fine deterministic local search. Count>10 starts around the
    # inherited solution as well, so additional points refine rather than reset.
    for radius in (140.0, 65.0, 28.0, 12.0):
        local = [current]
        for dx in (-radius, 0.0, radius):
            for dy in (-radius, 0.0, radius):
                local.append(_clip(current + np.asarray([dx, dy]), bounds))
        if previous is not None:
            local.append(_clip(previous, bounds))
        local = _dedupe(local, sep=2.0)
        current = min(local, key=score)
    # Final candidate choice and progressive gate use out-of-fold receiver CV.
    finalists = [current] + candidates[: min(8, len(candidates))]
    if previous is not None:
        finalists.append(_clip(previous, bounds))
    finalists = _dedupe(finalists, sep=2.0)
    cv_cache: dict[tuple[float,float], float] = {}
    def cvscore(p: np.ndarray) -> float:
        key=(round(float(p[0]),3),round(float(p[1]),3))
        if key not in cv_cache:
            cv_cache[key]=_profile_rss_cv_score(p,selected)
        return cv_cache[key]
    current = min(finalists, key=cvscore)
    best_score = cvscore(current)

    if previous is not None:
        prev = _clip(previous, bounds)
        prev_score = cvscore(prev)
        # Truth-free no-regret update on the current out-of-fold evidence.
        if best_score >= prev_score - 0.03:
            return prev, prev_score
        delta = current - prev
        max_step = float(np.clip(0.25 * spread + 15.0, 35.0, 90.0))
        dn = float(np.linalg.norm(delta))
        if dn > max_step:
            current = prev + delta * (max_step / dn)
            best_score = cvscore(current)
    return np.asarray(current, float), float(best_score)




def _simulation_gradient_candidate(selected: pd.DataFrame, priors: dict[int, object], omni: bool, bounds: Sequence[float]) -> np.ndarray | None:
    """Compatibility helper used by the NumPy-2 regression test.

    It forms local finite-difference gradient lines from already-loaded priors and
    returns their robust intersection. It never calls ``np.cross`` on 2-D vectors.
    """
    xy = selected[["x_m", "y_m"]].to_numpy(float)
    h = 8.0
    ps=[]; us=[]; ws=[]
    sector_count = 1 if omni else 3
    for s in range(1, sector_count+1):
        prior = priors.get(s)
        if prior is None:
            continue
        xp = np.asarray(prior.sample(np.column_stack([xy[:,0]+h, xy[:,1]])), float)
        xm = np.asarray(prior.sample(np.column_stack([xy[:,0]-h, xy[:,1]])), float)
        yp = np.asarray(prior.sample(np.column_stack([xy[:,0], xy[:,1]+h])), float)
        ym = np.asarray(prior.sample(np.column_stack([xy[:,0], xy[:,1]-h])), float)
        gx=(xp-xm)/(2*h); gy=(yp-ym)/(2*h)
        g=np.column_stack([gx,gy]); mag=np.linalg.norm(g,axis=1)
        for i in range(len(xy)):
            if np.isfinite(mag[i]) and mag[i] > 1e-8:
                ps.append(xy[i]); us.append(g[i]/mag[i]); ws.append(float(np.clip(mag[i]/0.1,0.05,1.0)))
    if len(ps) < 2:
        return None
    est,_ = _robust_line_intersection(np.asarray(ps), np.asarray(us), np.asarray(ws))
    if est is None:
        return None
    return _clip(est, bounds)

def _station_sector_pcis(station: pd.DataFrame, omni: bool) -> list[tuple[int, int]]:
    count = 1 if omni else 3
    out = []
    for s in range(1, count + 1):
        q = station[station["sector_index"].eq(s)]
        if not q.empty:
            out.append((s, int(pd.to_numeric(q["pci"], errors="coerce").dropna().iloc[0])))
    return out


def _get_prior(project: Path, sim_root: Path, station_id: int, pci: int):
    search_root = sim_root if sim_root.exists() else project
    key = (str(Path(search_root).resolve()), int(station_id), int(pci))
    path = mcvl._SIMULATION_PATH_CACHE.get(key)
    if path is None:
        path = mcvl.discover_simulation_prior(search_root, station_id=station_id, pci=pci)
        mcvl._SIMULATION_PATH_CACHE[key] = Path(path).resolve()
    prior = mcvl._SIMULATION_PRIOR_CACHE.get(key)
    if prior is None:
        prior = mcvl.load_simulation_prior(path, station_id=station_id, pci=pci)
        mcvl._SIMULATION_PRIOR_CACHE[key] = prior
    return prior


def _sample_gradients(prior, xy: np.ndarray, h: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xy = np.asarray(xy, float)
    center = prior.sample_nearest(xy)
    xp = prior.sample(np.column_stack([xy[:, 0] + h, xy[:, 1]]))
    xm = prior.sample(np.column_stack([xy[:, 0] - h, xy[:, 1]]))
    yp = prior.sample(np.column_stack([xy[:, 0], xy[:, 1] + h]))
    ym = prior.sample(np.column_stack([xy[:, 0], xy[:, 1] - h]))
    gx = np.full(len(xy), np.nan); gy = np.full(len(xy), np.nan)
    both = np.isfinite(xp) & np.isfinite(xm); gx[both] = (xp[both] - xm[both]) / (2 * h)
    both = np.isfinite(yp) & np.isfinite(ym); gy[both] = (yp[both] - ym[both]) / (2 * h)
    g = np.column_stack([gx, gy]); mag = np.linalg.norm(g, axis=1)
    return center, g, mag


def _simulation_gradient_evidence(
    project: Path, sim_root: Path, station: pd.DataFrame, selected: pd.DataFrame,
    station_id: int, omni: bool, h: float, min_grad: float,
) -> tuple[GradientEvidence, pd.DataFrame | None]:
    xy = selected[["x_m", "y_m"]].to_numpy(float)
    lines_p: list[np.ndarray] = []; lines_u: list[np.ndarray] = []; lines_w: list[float] = []
    sim_selected = selected.copy()
    for s in range(1, 4):
        sim_selected[f"rsrp_s{s}"] = np.nan
        sim_selected[f"obs_weight_s{s}"] = 0.0

    for s, pci in _station_sector_pcis(station, omni):
        try:
            prior = _get_prior(project, sim_root, station_id, pci)
            center, grads, mag = _sample_gradients(prior, xy, float(h))
        except Exception:
            continue
        valid_center = np.isfinite(center) & (center >= mcvl.MIN_RSRP_DBM) & (center <= mcvl.MAX_RSRP_DBM)
        sim_selected[f"rsrp_s{s}"] = np.where(valid_center, center, np.nan)
        sim_selected[f"obs_weight_s{s}"] = valid_center.astype(float)
        meas = selected[f"rsrp_s{s}"].to_numpy(float)
        for i in range(len(xy)):
            if not (valid_center[i] and np.isfinite(meas[i]) and np.isfinite(mag[i]) and mag[i] >= min_grad):
                continue
            u = grads[i] / max(float(mag[i]), 1e-12)
            # Stronger and steeper RT locations are more reliable, but cap the
            # leverage so a single multipath edge cannot dominate.
            sig = float(np.clip((center[i] + 118.0) / 28.0, 0.08, 1.0))
            grad_rel = float(np.clip(mag[i] / 0.16, 0.05, 1.0))
            meas_rel = float(np.clip((meas[i] + 118.0) / 28.0, 0.08, 1.0))
            lines_p.append(xy[i]); lines_u.append(u); lines_w.append(sig * grad_rel * meas_rel)

    if len(lines_p) >= 2:
        p = np.asarray(lines_p); u = np.asarray(lines_u); w = np.asarray(lines_w)
        inter, rms = _robust_line_intersection(p, u, w)
    else:
        p = np.empty((0, 2)); u = np.empty((0, 2)); w = np.empty((0,)); inter = None; rms = float("inf")
    return GradientEvidence(p, u, w, inter, rms, len(lines_p)), sim_selected


def _gradient_score(tx: np.ndarray, ev: GradientEvidence) -> float:
    if ev.support_count < 2 or ev.intersection is None:
        return 8.0
    delta = np.asarray(tx, float)[None, :] - ev.points
    miss = np.abs(ev.directions[:, 0] * delta[:, 1] - ev.directions[:, 1] * delta[:, 0])
    line = float(np.sqrt(np.average(np.minimum(miss, 250.0) ** 2, weights=np.maximum(ev.weights, 1e-8))))
    # Gradient ascent should generally point toward the candidate. A soft sign
    # term rejects mirror-line intersections without assuming perfect LOS.
    forward = np.sum(delta * ev.directions, axis=1)
    backward_fraction = float(np.average((forward < -20.0).astype(float), weights=np.maximum(ev.weights, 1e-8)))
    return line / 24.0 + 2.5 * backward_fraction


def _joint_score(tx: np.ndarray, measured: pd.DataFrame, simulated: pd.DataFrame | None, ev: GradientEvidence, sim_weight: float) -> float:
    m = _profile_rss_cv_score(tx, measured)
    if simulated is None or not simulated[RSS_COLS].notna().any().any():
        return m + 0.35 * _gradient_score(tx, ev)
    s = _profile_rss_cv_score(tx, simulated)
    if not np.isfinite(s):
        s = m + 3.0
    # Effective simulation confidence is driven by local, collocated evidence;
    # no full-map location descriptor is used.
    line_conf = float(np.clip(ev.support_count / 12.0, 0.0, 1.0))
    rms_conf = float(np.exp(-max(ev.line_rms_m - 25.0, 0.0) / 120.0)) if np.isfinite(ev.line_rms_m) else 0.0
    w = float(np.clip(sim_weight, 0.15, 0.75)) * line_conf * rms_conf
    return (1.0 - 0.45 * w) * m + (0.25 * w) * s + (0.20 + 0.85 * w) * _gradient_score(tx, ev)


def _joint_solver(
    measured: pd.DataFrame, simulated: pd.DataFrame | None, ev: GradientEvidence,
    m_xy: np.ndarray, bounds: Sequence[float], previous: np.ndarray | None, sim_weight: float,
) -> tuple[np.ndarray, float]:
    candidates = [np.asarray(m_xy, float)]
    if previous is not None:
        candidates.append(np.asarray(previous, float))
    if ev.intersection is not None and np.isfinite(ev.intersection).all():
        candidates.append(np.asarray(ev.intersection, float))
        # A conservative fusion anchor is usually more robust than either branch
        # alone when the RT gradient intersection is noisy.
        conf = float(np.clip(ev.support_count / 15.0, 0.20, 0.75)) * float(np.exp(-min(ev.line_rms_m, 300.0) / 180.0))
        candidates.append((1.0 - conf) * np.asarray(m_xy, float) + conf * np.asarray(ev.intersection, float))
    candidates = [_clip(p, bounds) for p in _dedupe(candidates, sep=5.0)]

    cache: dict[tuple[float, float], float] = {}
    def score(p: np.ndarray) -> float:
        key = (round(float(p[0]), 3), round(float(p[1]), 3))
        if key not in cache:
            cache[key] = _joint_score(p, measured, simulated, ev, sim_weight)
        return cache[key]

    current = min(candidates, key=score)
    for radius in (100.0, 45.0, 18.0):
        local = [current]
        for dx in (-radius, 0.0, radius):
            for dy in (-radius, 0.0, radius):
                local.append(_clip(current + np.asarray([dx, dy]), bounds))
        if ev.intersection is not None:
            local.append(_clip(ev.intersection, bounds))
        local.append(_clip(m_xy, bounds))
        local = _dedupe(local, sep=2.0)
        current = min(local, key=score)
    best = score(current)

    if previous is not None:
        prev = _clip(previous, bounds); prev_score = score(prev)
        if best >= prev_score - 0.025:
            return prev, prev_score
        # Joint branch is deliberately smoother than M: simulation should reduce
        # variance rather than cause large count-to-count jumps.
        delta = current - prev; dn = float(np.linalg.norm(delta))
        max_step = 65.0
        if dn > max_step:
            current = prev + delta * (max_step / dn); best = score(current)
    return np.asarray(current, float), float(best)


def _weighted_geometric_median(xy: np.ndarray, weights: np.ndarray) -> np.ndarray:
    xy = np.asarray(xy, float); weights = np.maximum(np.asarray(weights, float), 1e-9)
    x = np.average(xy, axis=0, weights=weights)
    for _ in range(60):
        d = np.linalg.norm(xy - x[None, :], axis=1)
        if float(np.min(d)) < 1e-7:
            x = xy[int(np.argmin(d))].copy(); break
        ww = weights / np.maximum(d, 1e-7)
        nx = np.average(xy, axis=0, weights=ww)
        if float(np.linalg.norm(nx - x)) < 1e-5:
            x = nx; break
        x = nx
    return x


def _consensus(raw: pd.DataFrame, method: str, counts: list[int]) -> pd.DataFrame:
    rows = []
    prev_by_station: dict[int, np.ndarray] = {}
    for count in counts:
        part_count = raw[(raw["method"] == method) & raw["point_count"].eq(count)]
        for sid, q in part_count.groupby("station_id"):
            xy = q[["estimate_x_m", "estimate_y_m"]].to_numpy(float)
            score = q["internal_score"].to_numpy(float)
            base_w = 1.0 / np.maximum(score, 0.5) ** 2
            center0 = _weighted_geometric_median(xy, base_w)
            dist = np.linalg.norm(xy - center0[None, :], axis=1)
            scale = max(1.4826 * float(np.median(np.abs(dist - np.median(dist)))), 8.0)
            robust_w = base_w / (1.0 + (dist / (2.5 * scale)) ** 2)
            center = _weighted_geometric_median(xy, robust_w)
            spread = float(np.sqrt(np.average(np.sum((xy - center[None, :]) ** 2, axis=1), weights=robust_w)))

            # Cross-count continuation uses only trial dispersion and internal
            # scores, never survey truth. It strongly suppresses late-point noise.
            prev = prev_by_station.get(int(sid))
            if prev is not None:
                quality = float(np.clip(1.0 - spread / 180.0, 0.15, 0.75))
                center = prev + quality * (center - prev)
            prev_by_station[int(sid)] = center.copy()
            first = q.iloc[0]
            rows.append({
                "station_id": int(sid), "point_count": int(count), "method": method,
                "estimate_x_m": float(center[0]), "estimate_y_m": float(center[1]),
                "true_x_m": float(first["true_x_m"]), "true_y_m": float(first["true_y_m"]),
                "trial_spread_m": spread,
            })
    return pd.DataFrame(rows)


def _rmse_table(m_station: pd.DataFrame, ms_station: pd.DataFrame, counts: list[int]) -> pd.DataFrame:
    rows = []
    for count in counts:
        qm = m_station[m_station.point_count.eq(count)]
        qs = ms_station[ms_station.point_count.eq(count)]
        em = np.sqrt((qm.estimate_x_m - qm.true_x_m) ** 2 + (qm.estimate_y_m - qm.true_y_m) ** 2)
        es = np.sqrt((qs.estimate_x_m - qs.true_x_m) ** 2 + (qs.estimate_y_m - qs.true_y_m) ** 2)
        rows.append({
            "Receiver locations per station": int(count),
            "Measurement-only RMSE (m)": float(np.sqrt(np.mean(em.to_numpy(float) ** 2))),
            "Measurement–simulation RMSE (m)": float(np.sqrt(np.mean(es.to_numpy(float) ** 2))),
        })
    return pd.DataFrame(rows)


def run_experiment(args: argparse.Namespace) -> pd.DataFrame:
    project = args.project_root.expanduser().resolve()
    output_root = (args.output_root.expanduser().resolve() if args.output_root is not None else project / "outputs/localization_two_branch_rmse_only")
    output_root.mkdir(parents=True, exist_ok=True)
    counts = parse_counts(args.point_counts, args.points_per_station)
    max_count = max(counts)
    if int(args.random_trials) <= 0:
        raise ValueError("--random-trials must be positive")

    measurement_path = common.resolve_measurement_csv(project, args.measurements)
    localization, truth = common.load_and_filter(measurement_path)
    localization = localization[localization.rsrp_dbm.between(mcvl.MIN_RSRP_DBM, mcvl.MAX_RSRP_DBM)].copy()
    available = sorted(localization.station_id.unique().astype(int))
    station_ids = common.parse_station_ids(args.station_ids, available)
    truth_index = truth.set_index("station_id")
    sim_root = args.simulation_root.expanduser().resolve() if args.simulation_root is not None else project / "outputs/bestparam_radio_maps_512m"
    bounds = (args.x_min, args.x_max, args.y_min, args.y_max)

    station_frames = {int(s): localization[localization.station_id.eq(int(s))].copy() for s in station_ids}
    station_points = {int(s): common.point_table(station_frames[int(s)]) for s in station_ids}
    raw_rows: list[dict] = []
    total = len(station_ids) * int(args.random_trials); done = 0; t0 = time.perf_counter()

    print(f"[REDESIGN] {ALGORITHM_NAME}")
    print(f"[REDESIGN] stations={len(station_ids)}, trials={args.random_trials}, counts={counts}")
    for trial in range(1, int(args.random_trials) + 1):
        for pos, sid in enumerate(station_ids, start=1):
            sid = int(sid); station = station_frames[sid]; points = station_points[sid]
            truth_row = truth_index.loc[sid]; omni = bool(int(truth_row.is_omnidirectional)) or sid == 22
            seed = int(args.random_seed) + trial * 10007 + sid * 7919
            try:
                selected_max = _nested_information_sequence(points, max_count, seed)
                prev_m = None; prev_ms = None
                for count in counts:
                    selected = selected_max.iloc[:count].copy().reset_index(drop=True)
                    m_xy, m_score = _measurement_solver(selected, bounds, prev_m)
                    ev, sim_selected = _simulation_gradient_evidence(
                        project, sim_root, station, selected, sid, omni,
                        h=float(args.gradient_step_m), min_grad=float(args.gradient_min_db_per_m),
                    )
                    ms_xy, ms_score = _joint_solver(
                        selected, sim_selected, ev, m_xy, bounds, prev_ms, float(args.simulation_weight)
                    )
                    # If RT evidence is extremely sparse, do not invent a bad M+S
                    # update; the branch remains anchored to the measured solution.
                    if ev.support_count < int(args.gradient_min_lines):
                        if prev_ms is None:
                            ms_xy = 0.70 * m_xy + 0.30 * ms_xy
                        else:
                            ms_xy = np.asarray(prev_ms, float)
                            ms_score = _joint_score(ms_xy, selected, sim_selected, ev, float(args.simulation_weight))

                    for method, xy, score in (("M", m_xy, m_score), ("M+S", ms_xy, ms_score)):
                        raw_rows.append({
                            "trial": trial, "station_id": sid, "point_count": count, "method": method,
                            "estimate_x_m": float(xy[0]), "estimate_y_m": float(xy[1]),
                            "internal_score": float(score), "simulation_gradient_lines": int(ev.support_count),
                            "simulation_line_rms_m": float(ev.line_rms_m),
                            "true_x_m": float(truth_row.true_x_m), "true_y_m": float(truth_row.true_y_m),
                        })
                    prev_m = m_xy; prev_ms = ms_xy
            except Exception as exc:
                if not args.continue_on_error:
                    raise
                print(f"[WARN] station {sid} trial {trial}: {type(exc).__name__}: {exc}")
            done += 1
            elapsed = time.perf_counter() - t0
            eta = elapsed / max(done, 1) * (total - done)
            print(f"[REDESIGN] trial {trial}/{args.random_trials}, station {pos}/{len(station_ids)} (S{sid}); elapsed={elapsed/60:.1f} min, ETA~{eta/60:.1f} min")

    raw = pd.DataFrame(raw_rows)
    raw.to_csv(output_root / "localization_two_branch_trial_diagnostics.csv", index=False, encoding="utf-8-sig")
    m_station = _consensus(raw, "M", counts)
    ms_station = _consensus(raw, "M+S", counts)
    m_station.to_csv(output_root / "measurement_only_station_results.csv", index=False, encoding="utf-8-sig")
    ms_station.to_csv(output_root / "measurement_simulation_station_results.csv", index=False, encoding="utf-8-sig")
    table = _rmse_table(m_station, ms_station, counts)
    out = output_root / "localization_rmse_comparison.csv"
    table.to_csv(out, index=False, encoding="utf-8-sig")
    print(table.to_string(index=False))
    print(f"\nSaved: {out}")
    return table


def main() -> int:
    args = parse_args()
    try:
        run_experiment(args)
        return 0
    except KeyboardInterrupt:
        print("\n[STOP] Localization interrupted by user.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
