# -*- coding: utf-8 -*-
"""Build raw-vs-calibrated evaluation tables.

The script regenerates raw and calibrated comparisons on the *same* trajectory
subsets and then exports side-by-side primary and stratified tables.  It also
copies the calibration-only LOTO candidate table used to select the residual
model.  No model parameter is fitted here.

Important: the three held-out trajectories are excluded from residual fitting and model selection. They were collected in the same measurement campaign, so the resulting assessment is described as a trajectory-disjoint internal evaluation rather than an independently acquired external test.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from workflows.evaluation.compare_joint_map_with_measurements import main as compare_main  # noqa: E402

RAW_MAP = Path("outputs/joint_best_server_4000x3000/joint_best_server_27stations_4000x3000.npz")
CAL_MAP = Path("outputs/joint_best_server_4000x3000/joint_best_server_27stations_4000x3000_calibrated.npz")
MEAS = Path("data/processed/cell_pci_rsrp_long_27stations.csv")
SPLIT = Path("outputs/parameter_calibration/calibration_validation_split.json")
MAPPING = Path("config/base_station_pci_mapping.csv")
MODEL = Path("config/network_residual_calibration.json")
CV = Path("outputs/network_residual_calibration/cv_candidates.csv")
OUT = Path("outputs/evaluation")

GROUP_FILES = {
    "by_station": "per_station_validation_metrics.csv",
    "by_trajectory": "rmse_by_source_file.csv",
    "by_pci": "validation_metrics_by_measured_best_pci.csv",
    "by_elevation_quartile": "validation_metrics_by_ground_elevation_quartile.csv",
    "by_tx_distance": "validation_metrics_by_tx_distance.csv",
}


def _resolve(root: Path, value: str | Path | None, default: Path) -> Path:
    p = default if value is None else Path(value)
    if not p.is_absolute():
        p = root / p
    return p.resolve()


def _run_compare(root: Path, map_path: Path, measurements: Path, split: Path, mapping: Path,
                 outdir: Path, subset: str, skip_figures: bool) -> None:
    argv = [
        "--project-root", str(root),
        "--map-npz", str(map_path),
        "--measurements", str(measurements),
        "--output-dir", str(outdir),
        "--split-manifest", str(split),
        "--station-mapping", str(mapping),
        "--evaluation-subset", subset,
    ]
    if skip_figures:
        argv.append("--skip-figures")
    rc = compare_main(argv)
    if rc != 0:
        raise RuntimeError(f"comparison failed for {map_path} subset={subset}: rc={rc}")


def _read_one_row(path: Path) -> dict[str, Any]:
    df = pd.read_csv(path, encoding="utf-8-sig")
    if len(df) != 1:
        raise RuntimeError(f"expected one-row metrics CSV: {path}")
    return df.iloc[0].to_dict()


def _primary_table(raw_metrics: dict[str, Any], cal_metrics: dict[str, Any], subset: str) -> pd.DataFrame:
    keys = [
        "matched_map_cell_count", "primary_rmse_db", "mae_db", "median_absolute_error_db",
        "bias_sim_minus_measured_db", "error_standard_deviation_db", "bias_corrected_rmse_db",
        "p90_absolute_error_db", "p95_absolute_error_db", "pearson_correlation",
        "within_5db_percent", "within_10db_percent", "within_15db_percent", "within_20db_percent",
        "best_station_match_percent", "best_pci_match_percent",
    ]
    rows = []
    for key in keys:
        if key not in raw_metrics and key not in cal_metrics:
            continue
        raw = raw_metrics.get(key)
        cal = cal_metrics.get(key)
        delta = None
        try:
            delta = float(cal) - float(raw)
        except Exception:
            pass
        rows.append({"evaluation_subset": subset, "metric": key, "raw": raw, "calibrated": cal, "delta_calibrated_minus_raw": delta})
    return pd.DataFrame(rows)


def _group_key_columns(df: pd.DataFrame) -> list[str]:
    metric_cols = {
        "matched_count", "rmse_db", "mae_db", "bias_sim_minus_measured_db",
        "error_standard_deviation_db", "pearson_correlation", "p90_absolute_error_db",
    }
    return [c for c in df.columns if c not in metric_cols]


def _merge_group(raw_path: Path, cal_path: Path) -> pd.DataFrame:
    raw = pd.read_csv(raw_path, encoding="utf-8-sig")
    cal = pd.read_csv(cal_path, encoding="utf-8-sig")
    keys = _group_key_columns(raw)
    if keys != _group_key_columns(cal):
        raise RuntimeError(f"raw/calibrated group table keys differ: {raw_path.name}")
    raw = raw.rename(columns={c: f"raw_{c}" for c in raw.columns if c not in keys})
    cal = cal.rename(columns={c: f"calibrated_{c}" for c in cal.columns if c not in keys})
    merged = raw.merge(cal, on=keys, how="outer", validate="one_to_one")
    if {"raw_rmse_db", "calibrated_rmse_db"}.issubset(merged.columns):
        merged["rmse_delta_db"] = merged["calibrated_rmse_db"] - merged["raw_rmse_db"]
    if {"raw_mae_db", "calibrated_mae_db"}.issubset(merged.columns):
        merged["mae_delta_db"] = merged["calibrated_mae_db"] - merged["raw_mae_db"]
    if {"raw_bias_sim_minus_measured_db", "calibrated_bias_sim_minus_measured_db"}.issubset(merged.columns):
        merged["bias_delta_db"] = merged["calibrated_bias_sim_minus_measured_db"] - merged["raw_bias_sim_minus_measured_db"]
    return merged


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Export raw-vs-calibrated evaluation package")
    p.add_argument("--project-root", default=None)
    p.add_argument("--raw-map", default=None)
    p.add_argument("--calibrated-map", default=None)
    p.add_argument("--measurements", default=None)
    p.add_argument("--split-manifest", default=None)
    p.add_argument("--station-mapping", default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--cv-candidates", default=None)
    p.add_argument("--output-root", default=None)
    p.add_argument("--skip-validation-figures", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.project_root).resolve() if args.project_root else ROOT
    raw_map = _resolve(root, args.raw_map, RAW_MAP)
    cal_map = _resolve(root, args.calibrated_map, CAL_MAP)
    measurements = _resolve(root, args.measurements, MEAS)
    split = _resolve(root, args.split_manifest, SPLIT)
    mapping = _resolve(root, args.station_mapping, MAPPING)
    model_path = _resolve(root, args.model, MODEL)
    cv_path = _resolve(root, args.cv_candidates, CV)
    out = _resolve(root, args.output_root, OUT)
    tables = out / "summary_tables"
    tables.mkdir(parents=True, exist_ok=True)

    split_meta = json.loads(split.read_text(encoding="utf-8"))
    cal_files = set(map(str, split_meta.get("calibration_source_files", [])))
    val_files = set(map(str, split_meta.get("validation_source_files", [])))
    if not cal_files or not val_files or cal_files & val_files:
        raise RuntimeError("invalid calibration/validation trajectory split")
    model = json.loads(model_path.read_text(encoding="utf-8"))
    if bool(model.get("validation_data_used", True)):
        raise RuntimeError("network residual model indicates validation_data_used=true")

    jobs = [
        ("raw_validation", raw_map, "validation", bool(args.skip_validation_figures)),
        ("calibrated_validation", cal_map, "validation", bool(args.skip_validation_figures)),
        ("raw_calibration", raw_map, "calibration", True),
        ("calibrated_calibration", cal_map, "calibration", True),
    ]
    for name, map_path, subset, skip_figures in jobs:
        print(f"[evaluation] {name}")
        _run_compare(root, map_path, measurements, split, mapping, out / name, subset, skip_figures)

    raw_val = _read_one_row(out / "raw_validation" / "comparison_metrics.csv")
    cal_val = _read_one_row(out / "calibrated_validation" / "comparison_metrics.csv")
    raw_cal = _read_one_row(out / "raw_calibration" / "comparison_metrics.csv")
    cal_cal = _read_one_row(out / "calibrated_calibration" / "comparison_metrics.csv")
    _primary_table(raw_val, cal_val, "validation").to_csv(
        tables / "validation_primary_metrics_raw_vs_calibrated.csv", index=False, encoding="utf-8-sig"
    )
    _primary_table(raw_cal, cal_cal, "calibration").to_csv(
        tables / "calibration_primary_metrics_raw_vs_calibrated.csv", index=False, encoding="utf-8-sig"
    )

    for label, filename in GROUP_FILES.items():
        raw_path = out / "raw_validation" / filename
        cal_path = out / "calibrated_validation" / filename
        if raw_path.is_file() and cal_path.is_file():
            _merge_group(raw_path, cal_path).to_csv(
                tables / f"validation_{label}_raw_vs_calibrated.csv", index=False, encoding="utf-8-sig"
            )

    if cv_path.is_file():
        shutil.copy2(cv_path, tables / "calibration_only_loto_cv_candidates.csv")
    shutil.copy2(model_path, tables / "network_residual_model.json")

    validation_rmse_gain = float(raw_val["primary_rmse_db"]) - float(cal_val["primary_rmse_db"])
    manifest = {
        "schema_version": 1,
        "status": "ok",
        "raw_map": str(raw_map),
        "calibrated_map": str(cal_map),
        "network_residual_model": str(model_path),
        "network_residual_model_validation_data_used": False,
        "calibration_validation_disjoint": True,
        "calibration_trajectory_count": len(cal_files),
        "validation_trajectory_count": len(val_files),
        "validation_role": "trajectory_disjoint_internal_evaluation",
        "newly_pristine_test_set": False,
        "validation_was_not_read_by_network_residual_fitter": True,
        "raw_validation_rmse_db": float(raw_val["primary_rmse_db"]),
        "calibrated_validation_rmse_db": float(cal_val["primary_rmse_db"]),
        "validation_rmse_gain_db": validation_rmse_gain,
        "raw_validation_bias_db": float(raw_val["bias_sim_minus_measured_db"]),
        "calibrated_validation_bias_db": float(cal_val["bias_sim_minus_measured_db"]),
        "best_station_and_pci_not_reselected_by_network_residual_model": bool(model.get("best_station_and_pci_are_not_reselected", False)),
        "recommended_dataset_roles": {
            "raw_joint_map": "independent propagation-model baseline and physics-facing analysis",
            "calibrated_joint_map": "measurement-calibrated RSRP benchmark; not a replacement for raw RT provenance",
            "measurement_filled_per_pci_maps": "reconstruction/dense-reference studies; do not treat filled cells as independent propagation ground truth",
        },
    }
    (out / "evaluation_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    readme = f"""# Evaluation outputs\n\nThis folder keeps raw and calibrated evaluations separate.\n\n- Raw joint-map validation RMSE: {manifest['raw_validation_rmse_db']:.6f} dB\n- Calibrated joint-map validation RMSE: {manifest['calibrated_validation_rmse_db']:.6f} dB\n- RMSE reduction: {manifest['validation_rmse_gain_db']:.6f} dB\n\nThe network residual model was selected only from the nine calibration trajectories.\nThe three held-out trajectories were not read by the residual fitter or model selector.\nBecause all trajectories were collected in the same measurement campaign, the result is\ndescribed as a trajectory-disjoint internal evaluation rather than an independently\nacquired external test.\n\n`summary_tables/` contains side-by-side primary metrics and error stratifications by\nstation, trajectory, PCI, elevation quartile, and TX distance, plus the calibration-only\nLOTO model-selection table.\n"""
    (out / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
