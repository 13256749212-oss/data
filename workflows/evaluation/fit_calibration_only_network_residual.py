# -*- coding: utf-8 -*-
"""
Calibration-only network residual calibration for the 27-station joint best-server map.

Design goals
------------
1. Never read the held-out validation trajectories during parameter selection.
2. Preserve the Sionna RT best-station / best-PCI decision surface.
3. Fit only one NETWORK-WIDE residual-transfer model from the nine calibration
   trajectories, selected by leave-one-calibration-trajectory-out (LOTO) CV.
4. Allow a small family of physically interpretable low-capacity models:
      identity, offset, affine_rsrp, distance, affine_rsrp_distance
5. Use trajectory-balanced training weights and Huber IRLS to avoid a long drive
   or a few large residuals dominating the fit.
6. Clip the learned correction to a conservative range before applying it to the
   dense map. This prevents extrapolation from creating unrealistic powers.

This is a post-RT calibration layer. It does not alter ray-tracing physics,
geometry, transmit powers, antenna orientation, station/PCI winner identity, or
station-specific EPRE. It produces a separate calibrated NPZ and keeps the raw
joint map unchanged.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from workflows.evaluation.compare_joint_map_with_measurements import (  # noqa: E402
    ComparisonError,
    load_joint_map,
    match_measurements_to_map,
    read_measurement_best_server,
)

RAW_MAP_REL = Path("outputs/joint_best_server_4000x3000/joint_best_server_27stations_4000x3000.npz")
CAL_MAP_REL = Path("outputs/joint_best_server_4000x3000/joint_best_server_27stations_4000x3000_calibrated.npz")
MEAS_REL = Path("data/processed/cell_pci_rsrp_long_27stations.csv")
SPLIT_REL = Path("outputs/parameter_calibration/calibration_validation_split.json")
MAPPING_REL = Path("config/base_station_pci_mapping.csv")
MODEL_REL = Path("config/network_residual_calibration.json")
CV_REL = Path("outputs/network_residual_calibration/cv_candidates.csv")
MATCH_REL = Path("outputs/network_residual_calibration/calibration_matched_cells.csv")


def _resolve(root: Path, value: str | Path | None, default: Path) -> Path:
    if value is None:
        return (root / default).resolve()
    p = Path(value)
    if not p.is_absolute():
        p = root / p
    return p.resolve()


def _load_station_xy(path: Path) -> dict[int, tuple[float, float]]:
    if not path.is_file():
        raise ComparisonError(f"找不到基站映射CSV: {path}")
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
    needed = {"station_id", "tx_x_initial_m", "tx_y_initial_m"}
    if not needed.issubset(df.columns):
        raise ComparisonError(f"基站映射缺少字段 {sorted(needed)}")
    out: dict[int, tuple[float, float]] = {}
    for sid, g in df.groupby("station_id", sort=False):
        x = pd.to_numeric(g["tx_x_initial_m"], errors="coerce").dropna()
        y = pd.to_numeric(g["tx_y_initial_m"], errors="coerce").dropna()
        if len(x) and len(y):
            out[int(sid)] = (float(x.iloc[0]), float(y.iloc[0]))
    if not out:
        raise ComparisonError("基站映射中没有有效TX XY坐标")
    return out


def _append_predicted_distance(matched: pd.DataFrame, station_xy: dict[int, tuple[float, float]]) -> pd.DataFrame:
    if "simulated_best_station_id" not in matched.columns:
        raise ComparisonError("联合地图没有simulated_best_station_id，无法使用distance候选模型")
    out = matched.copy()
    xs = out["map_x_m"].to_numpy(float)
    ys = out["map_y_m"].to_numpy(float)
    sids = pd.to_numeric(out["simulated_best_station_id"], errors="coerce").fillna(-1).astype(int).to_numpy()
    d = np.full(len(out), np.nan, dtype=float)
    for sid in np.unique(sids):
        if sid not in station_xy:
            continue
        mask = sids == sid
        tx_x, tx_y = station_xy[sid]
        d[mask] = np.hypot(xs[mask] - tx_x, ys[mask] - tx_y)
    out["predicted_tx_distance_m"] = d
    return out


def _feature_matrix(df: pd.DataFrame, model: str) -> np.ndarray:
    sim = pd.to_numeric(df["simulated_best_rsrp_dbm"], errors="coerce").to_numpy(float)
    s = np.clip((sim + 80.0) / 20.0, -2.5, 2.5)
    if model == "identity":
        return np.zeros((len(df), 0), dtype=float)
    if model == "offset":
        return np.ones((len(df), 1), dtype=float)
    if model == "affine_rsrp":
        return np.column_stack([np.ones(len(df)), s])
    dist = pd.to_numeric(df["predicted_tx_distance_m"], errors="coerce").to_numpy(float)
    d = np.log10(np.clip(dist, 30.0, 1500.0) / 100.0)
    if model == "distance":
        return np.column_stack([np.ones(len(df)), d])
    if model == "affine_rsrp_distance":
        return np.column_stack([np.ones(len(df)), s, d])
    raise ValueError(f"未知模型: {model}")


def _trajectory_balanced_base_weights(df: pd.DataFrame) -> np.ndarray:
    if "source_file" not in df.columns:
        return np.ones(len(df), dtype=float)
    counts = df.groupby("source_file")["source_file"].transform("size").to_numpy(float)
    w = 1.0 / np.maximum(counts, 1.0)
    # normalize so mean weight is one; this keeps ridge scale stable
    return w / max(float(np.mean(w)), 1e-12)


def _fit_huber_ridge(
    X: np.ndarray,
    y: np.ndarray,
    base_w: np.ndarray,
    ridge: float,
    huber_k: float = 1.5,
    max_iter: int = 30,
) -> np.ndarray:
    if X.shape[1] == 0:
        return np.empty(0, dtype=float)
    finite = np.isfinite(y) & np.all(np.isfinite(X), axis=1) & np.isfinite(base_w) & (base_w > 0)
    X = X[finite]
    y = y[finite]
    base_w = base_w[finite]
    if len(y) < max(5, X.shape[1] + 2):
        raise ComparisonError("网络残差校准训练样本不足")
    beta = np.zeros(X.shape[1], dtype=float)
    robust_w = np.ones(len(y), dtype=float)
    penalty = np.eye(X.shape[1], dtype=float) * float(ridge)
    penalty[0, 0] = 0.0  # never shrink the intercept
    for _ in range(max_iter):
        w = base_w * robust_w
        Xw = X * np.sqrt(w)[:, None]
        yw = y * np.sqrt(w)
        A = Xw.T @ Xw + penalty
        b = Xw.T @ yw
        try:
            new_beta = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            new_beta = np.linalg.pinv(A) @ b
        resid = y - X @ new_beta
        med = float(np.median(resid))
        mad = float(np.median(np.abs(resid - med)))
        sigma = max(1.4826 * mad, 0.5)
        cutoff = huber_k * sigma
        abs_r = np.abs(resid)
        robust_w_new = np.ones_like(abs_r)
        high = abs_r > cutoff
        robust_w_new[high] = cutoff / np.maximum(abs_r[high], 1e-9)
        if np.max(np.abs(new_beta - beta)) < 1e-7:
            beta = new_beta
            break
        beta = new_beta
        robust_w = robust_w_new
    return beta


def _predict_correction(df: pd.DataFrame, model: str, beta: np.ndarray, clip_db: float) -> np.ndarray:
    if model == "identity":
        return np.zeros(len(df), dtype=float)
    X = _feature_matrix(df, model)
    corr = X @ beta
    return np.clip(corr, -float(clip_db), float(clip_db))


def _fold_metrics(df: pd.DataFrame, correction: np.ndarray) -> tuple[float, float, float]:
    sim = df["simulated_best_rsrp_dbm"].to_numpy(float) + correction
    meas = df["measured_best_rsrp_dbm"].to_numpy(float)
    e = sim - meas
    return float(np.sqrt(np.mean(e * e))), float(np.mean(e)), float(np.mean(np.abs(e)))


def select_model_loto(
    matched: pd.DataFrame,
    candidate_models: list[str],
    ridge_grid: list[float],
    correction_clip_db: float,
    min_cv_gain_db: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    if "source_file" not in matched.columns:
        raise ComparisonError("calibration matched table缺少source_file，无法执行LOTO")
    source_files = sorted(matched["source_file"].astype(str).unique().tolist())
    if len(source_files) < 3:
        raise ComparisonError("至少需要3条calibration trajectories执行LOTO")
    rows: list[dict[str, Any]] = []
    for model in candidate_models:
        ridges = [0.0] if model in {"identity", "offset"} else ridge_grid
        for ridge in ridges:
            fold_rmse: list[float] = []
            fold_bias: list[float] = []
            fold_mae: list[float] = []
            for heldout in source_files:
                train = matched.loc[matched["source_file"].astype(str) != heldout].copy()
                test = matched.loc[matched["source_file"].astype(str) == heldout].copy()
                y = train["measured_best_rsrp_dbm"].to_numpy(float) - train["simulated_best_rsrp_dbm"].to_numpy(float)
                if model == "identity":
                    beta = np.empty(0, dtype=float)
                else:
                    X = _feature_matrix(train, model)
                    beta = _fit_huber_ridge(X, y, _trajectory_balanced_base_weights(train), ridge)
                corr = _predict_correction(test, model, beta, correction_clip_db)
                rmse, bias, mae = _fold_metrics(test, corr)
                fold_rmse.append(rmse)
                fold_bias.append(bias)
                fold_mae.append(mae)
            rows.append({
                "model": model,
                "ridge": float(ridge),
                "mean_fold_rmse_db": float(np.mean(fold_rmse)),
                "median_fold_rmse_db": float(np.median(fold_rmse)),
                "mean_abs_fold_bias_db": float(np.mean(np.abs(fold_bias))),
                "mean_fold_mae_db": float(np.mean(fold_mae)),
                "fold_count": len(source_files),
            })
    cv = pd.DataFrame(rows).sort_values(
        ["mean_fold_rmse_db", "mean_abs_fold_bias_db", "model", "ridge"], kind="mergesort"
    ).reset_index(drop=True)
    identity_rmse = float(cv.loc[cv["model"].eq("identity"), "mean_fold_rmse_db"].iloc[0])
    best = cv.iloc[0].to_dict()
    gain = identity_rmse - float(best["mean_fold_rmse_db"])
    if best["model"] != "identity" and gain < float(min_cv_gain_db):
        best = cv.loc[cv["model"].eq("identity")].iloc[0].to_dict()
        gain = 0.0
    best["identity_cv_rmse_db"] = identity_rmse
    best["cv_gain_vs_identity_db"] = float(gain)
    return best, cv


def fit_final_model(matched: pd.DataFrame, selected: dict[str, Any], correction_clip_db: float) -> dict[str, Any]:
    model = str(selected["model"])
    ridge = float(selected["ridge"])
    y = matched["measured_best_rsrp_dbm"].to_numpy(float) - matched["simulated_best_rsrp_dbm"].to_numpy(float)
    if model == "identity":
        beta = np.empty(0, dtype=float)
    else:
        beta = _fit_huber_ridge(_feature_matrix(matched, model), y, _trajectory_balanced_base_weights(matched), ridge)
    corr = _predict_correction(matched, model, beta, correction_clip_db)
    rmse, bias, mae = _fold_metrics(matched, corr)
    raw_rmse, raw_bias, raw_mae = _fold_metrics(matched, np.zeros(len(matched)))
    return {
        "model": model,
        "ridge": ridge,
        "coefficients": [float(v) for v in beta],
        "correction_clip_db": float(correction_clip_db),
        "calibration_raw_rmse_db": raw_rmse,
        "calibration_raw_bias_db": raw_bias,
        "calibration_raw_mae_db": raw_mae,
        "calibration_corrected_rmse_db": rmse,
        "calibration_corrected_bias_db": bias,
        "calibration_corrected_mae_db": mae,
    }


def _map_correction_chunk(
    rsrp: np.ndarray,
    station: np.ndarray,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    row0: int,
    station_xy: dict[int, tuple[float, float]],
    model: str,
    beta: np.ndarray,
    clip_db: float,
) -> np.ndarray:
    if model == "identity":
        return np.zeros_like(rsrp, dtype=np.float32)
    finite = np.isfinite(rsrp)
    s = np.clip((rsrp.astype(float) + 80.0) / 20.0, -2.5, 2.5)
    corr = np.zeros(rsrp.shape, dtype=float)
    if model == "offset":
        corr[finite] = beta[0]
    elif model == "affine_rsrp":
        corr[finite] = beta[0] + beta[1] * s[finite]
    else:
        dfeat = np.full(rsrp.shape, np.nan, dtype=float)
        rows, cols = np.indices(rsrp.shape)
        global_rows = rows + row0
        for sid in np.unique(station[finite]):
            sid = int(sid)
            if sid not in station_xy:
                continue
            mask = finite & (station == sid)
            tx_x, tx_y = station_xy[sid]
            dist = np.hypot(x_axis[cols[mask]] - tx_x, y_axis[global_rows[mask]] - tx_y)
            dfeat[mask] = np.log10(np.clip(dist, 30.0, 1500.0) / 100.0)
        valid = finite & np.isfinite(dfeat)
        if model == "distance":
            corr[valid] = beta[0] + beta[1] * dfeat[valid]
        elif model == "affine_rsrp_distance":
            corr[valid] = beta[0] + beta[1] * s[valid] + beta[2] * dfeat[valid]
        # if distance is unavailable, use intercept only rather than inventing distance
        fallback = finite & ~valid
        corr[fallback] = beta[0]
    return np.clip(corr, -clip_db, clip_db).astype(np.float32)


def apply_to_npz(
    raw_map_path: Path,
    output_path: Path,
    model_meta: dict[str, Any],
    station_xy: dict[int, tuple[float, float]],
    chunk_rows: int = 256,
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with np.load(raw_map_path, allow_pickle=False) as z:
        arrays = {name: np.asarray(z[name]) for name in z.files}
    if "best_rsrp_dbm" not in arrays or "best_station_id" not in arrays:
        raise ComparisonError("联合地图NPZ缺少best_rsrp_dbm或best_station_id")
    rsrp = np.asarray(arrays["best_rsrp_dbm"], dtype=np.float32).copy()
    station = np.asarray(arrays["best_station_id"], dtype=np.int32)
    x_axis = np.asarray(arrays["x_centers_m"], dtype=float).reshape(-1)
    y_axis = np.asarray(arrays["y_centers_m"], dtype=float).reshape(-1)
    model = str(model_meta["model"])
    beta = np.asarray(model_meta.get("coefficients", []), dtype=float)
    clip_db = float(model_meta["correction_clip_db"])
    finite_before = np.isfinite(rsrp)
    correction_values: list[np.ndarray] = []
    for r0 in range(0, rsrp.shape[0], int(chunk_rows)):
        r1 = min(r0 + int(chunk_rows), rsrp.shape[0])
        corr = _map_correction_chunk(
            rsrp[r0:r1], station[r0:r1], x_axis, y_axis, r0,
            station_xy, model, beta, clip_db,
        )
        valid = np.isfinite(rsrp[r0:r1])
        rsrp[r0:r1][valid] = rsrp[r0:r1][valid] + corr[valid]
        if np.any(valid):
            correction_values.append(corr[valid].astype(np.float32))
    arrays["best_rsrp_dbm"] = rsrp
    # Explicit cell-level value provenance for the calibrated network map.
    # Code 2 is reserved for per-PCI measurement-filled products and is never
    # used by this network residual layer because no field measurement is copied
    # into the dense joint map.
    value_source_mask = np.zeros(rsrp.shape, dtype=np.uint8)
    value_source_mask[finite_before] = 3
    arrays["value_source_mask"] = value_source_mask
    arrays["provenance_codebook"] = np.asarray([
        "0=missing_or_invalid;1=original_sionna_rt;2=measurement_filled_same_grid;3=sionna_rt_plus_calibration_only_network_residual"
    ])
    metadata: dict[str, Any] = {}
    if "metadata_json" in arrays and np.size(arrays["metadata_json"]):
        try:
            metadata = json.loads(str(np.ravel(arrays["metadata_json"])[0]))
        except Exception:
            metadata = {}
    calibration_block = {
        "role": "calibration_only_network_residual_transfer",
        "validation_data_used": False,
        **model_meta,
    }
    metadata["network_residual_calibration"] = calibration_block
    arrays["metadata_json"] = np.asarray([json.dumps(metadata, ensure_ascii=False)])
    np.savez_compressed(output_path, **arrays)
    corr_all = np.concatenate(correction_values) if correction_values else np.asarray([], dtype=float)
    stats = {
        "output_npz": str(output_path),
        "finite_cell_count": int(finite_before.sum()),
        "model": model,
        "correction_mean_db": float(np.mean(corr_all)) if len(corr_all) else 0.0,
        "correction_std_db": float(np.std(corr_all)) if len(corr_all) else 0.0,
        "correction_min_db": float(np.min(corr_all)) if len(corr_all) else 0.0,
        "correction_max_db": float(np.max(corr_all)) if len(corr_all) else 0.0,
        "best_station_and_pci_arrays_unchanged": True,
        "value_source_mask_added": True,
        "value_source_code_for_finite_cells": 3,
        "measurement_values_copied_into_dense_joint_map": False,
    }
    return stats


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Calibration-only network residual calibration")
    p.add_argument("--project-root", default=None)
    p.add_argument("--raw-map", default=None)
    p.add_argument("--measurements", default=None)
    p.add_argument("--split-manifest", default=None)
    p.add_argument("--station-mapping", default=None)
    p.add_argument("--output-model", default=None)
    p.add_argument("--output-cv", default=None)
    p.add_argument("--output-matched", default=None)
    p.add_argument("--output-map", default=None)
    p.add_argument("--models", default="identity,offset,affine_rsrp,distance,affine_rsrp_distance")
    p.add_argument("--ridge-grid", default="0,0.1,1,10")
    p.add_argument("--correction-clip-db", type=float, default=12.0)
    p.add_argument("--min-cv-gain-db", type=float, default=0.05)
    p.add_argument("--center-arfcn", type=int, default=513000)
    p.add_argument("--bandwidth-mhz", type=float, default=100.0)
    p.add_argument("--rsrp-min-dbm", type=float, default=-140.0)
    p.add_argument("--rsrp-max-dbm", type=float, default=-40.0)
    p.add_argument("--chunk-rows", type=int, default=256)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.project_root).resolve() if args.project_root else ROOT
    raw_map_path = _resolve(root, args.raw_map, RAW_MAP_REL)
    measurement_path = _resolve(root, args.measurements, MEAS_REL)
    split_path = _resolve(root, args.split_manifest, SPLIT_REL)
    mapping_path = _resolve(root, args.station_mapping, MAPPING_REL)
    model_path = _resolve(root, args.output_model, MODEL_REL)
    cv_path = _resolve(root, args.output_cv, CV_REL)
    matched_path = _resolve(root, args.output_matched, MATCH_REL)
    output_map = _resolve(root, args.output_map, CAL_MAP_REL)

    if not split_path.is_file():
        raise ComparisonError(f"找不到split manifest: {split_path}")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    calibration_files = sorted(set(split.get("calibration_source_files", [])))
    validation_files = sorted(set(split.get("validation_source_files", [])))
    if not calibration_files:
        raise ComparisonError("split manifest没有calibration_source_files")
    if set(calibration_files) & set(validation_files):
        raise ComparisonError("calibration与validation source files发生重叠")

    print("[1/6] 读取raw joint map")
    radio_map = load_joint_map(raw_map_path)
    print("[2/6] 构造实测best-server，并严格保留calibration trajectories")
    read_args = SimpleNamespace(
        filter_n41=True,
        filter_center_arfcn=True,
        filter_bandwidth=True,
        center_arfcn=args.center_arfcn,
        bandwidth_mhz=args.bandwidth_mhz,
        rsrp_min_dbm=args.rsrp_min_dbm,
        rsrp_max_dbm=args.rsrp_max_dbm,
    )
    receiver_best, _ = read_measurement_best_server(measurement_path, read_args)
    if "source_file" not in receiver_best.columns:
        raise ComparisonError("实测表缺少source_file")
    receiver_best = receiver_best.loc[receiver_best["source_file"].astype(str).isin(calibration_files)].copy()
    if receiver_best.empty:
        raise ComparisonError("calibration trajectories筛选后为空")
    if receiver_best["source_file"].astype(str).isin(validation_files).any():
        raise ComparisonError("内部错误：validation trajectory进入了网络残差校准")

    print("[3/6] 与raw joint map进行1 m共址匹配")
    matched, _ = match_measurements_to_map(receiver_best, radio_map, aggregate_map_cells=True)
    station_xy = _load_station_xy(mapping_path)
    matched = _append_predicted_distance(matched, station_xy)
    # Distance-based models require finite predicted TX distance. Keep all rows for
    # identity/offset/affine; model-specific feature code handles unavailable distance.
    matched_path.parent.mkdir(parents=True, exist_ok=True)
    matched.to_csv(matched_path, index=False, encoding="utf-8-sig")

    print("[4/6] calibration-only LOTO选择低容量网络残差模型")
    models = [x.strip() for x in args.models.split(",") if x.strip()]
    valid_models = {"identity", "offset", "affine_rsrp", "distance", "affine_rsrp_distance"}
    unknown = [m for m in models if m not in valid_models]
    if unknown:
        raise ComparisonError(f"未知候选模型: {unknown}")
    ridge_grid = [float(x.strip()) for x in args.ridge_grid.split(",") if x.strip()]
    selected, cv = select_model_loto(
        matched, models, ridge_grid,
        correction_clip_db=args.correction_clip_db,
        min_cv_gain_db=args.min_cv_gain_db,
    )
    cv_path.parent.mkdir(parents=True, exist_ok=True)
    cv.to_csv(cv_path, index=False, encoding="utf-8-sig")

    print("[5/6] 用全部calibration trajectories拟合最终网络残差模型")
    final_model = fit_final_model(matched, selected, args.correction_clip_db)
    metadata = {
        "schema_version": 1,
        "method": "calibration_only_network_residual_transfer_loto",
        "source_subset": "calibration_trajectories_only",
        "validation_data_used": False,
        "calibration_trajectory_count": len(calibration_files),
        "calibration_source_files": calibration_files,
        "validation_source_files_not_used": validation_files,
        "selection_rule": "minimum equal-trajectory mean LOTO RMSE; identity retained unless gain >= min_cv_gain_db",
        "min_cv_gain_db": float(args.min_cv_gain_db),
        "selected_cv": selected,
        "final_model": final_model,
        "best_station_and_pci_are_not_reselected": True,
        "station_specific_offset_fitted": False,
        "raw_joint_map": str(raw_map_path),
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[6/6] 应用到dense joint map；raw map保留不覆盖")
    stats = apply_to_npz(raw_map_path, output_map, final_model, station_xy, chunk_rows=args.chunk_rows)
    stats_path = model_path.with_name("network_residual_calibration_map_stats.json")
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n完成 network residual calibration")
    print(f"  Selected model: {final_model['model']}")
    print(f"  CV RMSE: {float(selected['mean_fold_rmse_db']):.6f} dB")
    print(f"  Identity CV RMSE: {float(selected['identity_cv_rmse_db']):.6f} dB")
    print(f"  CV gain: {float(selected['cv_gain_vs_identity_db']):.6f} dB")
    print(f"  Calibration raw RMSE: {final_model['calibration_raw_rmse_db']:.6f} dB")
    print(f"  Calibration corrected RMSE: {final_model['calibration_corrected_rmse_db']:.6f} dB")
    print(f"  Calibration corrected bias: {final_model['calibration_corrected_bias_db']:+.6f} dB")
    print(f"  Calibrated map: {output_map}")
    print("  validation_data_used: false")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ComparisonError as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(2)
