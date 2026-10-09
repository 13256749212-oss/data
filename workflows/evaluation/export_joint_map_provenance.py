# -*- coding: utf-8 -*-
"""Export explicit cell-level provenance for the network-scale joint radio maps.

The exported masks let downstream users distinguish original ray-tracing values,
measurement-filled values, calibration-adjusted values, and invalid cells without
inferring provenance from filenames.

For the 4000 x 3000 m network joint map, measurements are *not* copied into the
map. Therefore the network-scale source masks use:

  0 = missing_or_invalid
  1 = original_sionna_rt
  2 = measurement_filled_same_grid (reserved; used by per-PCI filled products)
  3 = sionna_rt_plus_calibration_only_network_residual

A separate ``measurement_split_mask`` marks where a field measurement exists:

  0 = no_measurement
  1 = calibration_trajectory_measurement
  2 = validation_trajectory_measurement
  3 = both_calibration_and_validation_measurements_visit_this_cell

The measurement mask is diagnostic metadata only. It never replaces the RSRP
value in either the raw or calibrated network joint map.
"""
from __future__ import annotations

import argparse
import json
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
    _nearest_indices,
    load_joint_map,
    read_measurement_best_server,
)

RAW_DEFAULT = Path("outputs/joint_best_server_4000x3000/joint_best_server_27stations_4000x3000.npz")
CAL_DEFAULT = Path("outputs/joint_best_server_4000x3000/joint_best_server_27stations_4000x3000_calibrated.npz")
MEAS_DEFAULT = Path("data/processed/cell_pci_rsrp_long_27stations.csv")
SPLIT_DEFAULT = Path("outputs/parameter_calibration/calibration_validation_split.json")
OUT_DEFAULT = Path("outputs/joint_best_server_4000x3000/joint_best_server_27stations_4000x3000_provenance.npz")
SUMMARY_JSON_DEFAULT = Path("outputs/joint_best_server_4000x3000/joint_map_provenance_summary.json")
SUMMARY_CSV_DEFAULT = Path("outputs/joint_best_server_4000x3000/joint_map_provenance_counts.csv")

SOURCE_CODEBOOK: dict[int, str] = {
    0: "missing_or_invalid",
    1: "original_sionna_rt",
    2: "measurement_filled_same_grid_reserved_for_per_pci_products",
    3: "sionna_rt_plus_calibration_only_network_residual",
}
MEASUREMENT_SPLIT_CODEBOOK: dict[int, str] = {
    0: "no_measurement",
    1: "calibration_trajectory_measurement",
    2: "validation_trajectory_measurement",
    3: "calibration_and_validation_measurements_visit_same_cell",
}


def _resolve(root: Path, value: str | Path | None, default: Path) -> Path:
    p = default if value is None else Path(value)
    if not p.is_absolute():
        p = root / p
    return p.resolve()


def _assert_same_grid(raw: Any, cal: Any) -> None:
    if raw.best_rsrp_dbm.shape != cal.best_rsrp_dbm.shape:
        raise ComparisonError("raw/calibrated joint maps have different shapes")
    if not np.array_equal(raw.x_centers_m, cal.x_centers_m):
        raise ComparisonError("raw/calibrated joint maps have different x axes")
    if not np.array_equal(raw.y_centers_m, cal.y_centers_m):
        raise ComparisonError("raw/calibrated joint maps have different y axes")
    if raw.best_station_id is not None and cal.best_station_id is not None:
        if not np.array_equal(raw.best_station_id, cal.best_station_id):
            raise ComparisonError("best_station_id changed between raw and calibrated maps")
    if raw.best_pci is not None and cal.best_pci is not None:
        if not np.array_equal(raw.best_pci, cal.best_pci):
            raise ComparisonError("best_pci changed between raw and calibrated maps")
    if not np.array_equal(np.isfinite(raw.best_rsrp_dbm), np.isfinite(cal.best_rsrp_dbm)):
        raise ComparisonError("finite mask changed between raw and calibrated maps")


def _measurement_split_mask(
    raw_map: Any,
    measurements: Path,
    split_manifest: Path,
    center_arfcn: int,
    bandwidth_mhz: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    if not split_manifest.is_file():
        raise ComparisonError(f"split manifest not found: {split_manifest}")
    split = json.loads(split_manifest.read_text(encoding="utf-8"))
    calibration_files = set(map(str, split.get("calibration_source_files", [])))
    validation_files = set(map(str, split.get("validation_source_files", [])))
    if not calibration_files or not validation_files:
        raise ComparisonError("split manifest must contain calibration_source_files and validation_source_files")
    if calibration_files & validation_files:
        raise ComparisonError("calibration and validation source files overlap")

    read_args = SimpleNamespace(
        filter_n41=True,
        filter_center_arfcn=True,
        filter_bandwidth=True,
        center_arfcn=center_arfcn,
        bandwidth_mhz=bandwidth_mhz,
        rsrp_min_dbm=-140.0,
        rsrp_max_dbm=-40.0,
    )
    receiver_best, report = read_measurement_best_server(measurements, read_args)
    if "source_file" not in receiver_best.columns:
        raise ComparisonError("measurement table lacks source_file; provenance split cannot be exported")

    ix, inside_x = _nearest_indices(raw_map.x_centers_m, receiver_best["x_m"].to_numpy(float))
    iy, inside_y = _nearest_indices(raw_map.y_centers_m, receiver_best["y_m"].to_numpy(float))
    inside = inside_x & inside_y
    mask = np.zeros(raw_map.best_rsrp_dbm.shape, dtype=np.uint8)

    source_files = receiver_best["source_file"].astype(str).to_numpy()
    calibration_points = 0
    validation_points = 0
    unknown_points = 0
    for idx in np.flatnonzero(inside):
        src = source_files[idx]
        bit = 0
        if src in calibration_files:
            bit |= 1
            calibration_points += 1
        if src in validation_files:
            bit |= 2
            validation_points += 1
        if bit == 0:
            unknown_points += 1
            continue
        mask[iy[idx], ix[idx]] |= np.uint8(bit)

    report = dict(report)
    report.update({
        "receiver_instances_inside_joint_map": int(np.sum(inside)),
        "calibration_receiver_instances_inside_joint_map": int(calibration_points),
        "validation_receiver_instances_inside_joint_map": int(validation_points),
        "unknown_split_receiver_instances_inside_joint_map": int(unknown_points),
        "calibration_observed_cell_count": int(np.sum((mask & 1) > 0)),
        "validation_observed_cell_count": int(np.sum((mask & 2) > 0)),
        "overlap_observed_cell_count": int(np.sum(mask == 3)),
    })
    return mask, report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Export explicit cell-level provenance for raw and calibrated joint maps")
    p.add_argument("--project-root", default=None)
    p.add_argument("--raw-map", default=None)
    p.add_argument("--calibrated-map", default=None)
    p.add_argument("--measurements", default=None)
    p.add_argument("--split-manifest", default=None)
    p.add_argument("--output", default=None)
    p.add_argument("--summary-json", default=None)
    p.add_argument("--summary-csv", default=None)
    p.add_argument("--center-arfcn", type=int, default=513000)
    p.add_argument("--bandwidth-mhz", type=float, default=100.0)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.project_root).resolve() if args.project_root else ROOT
    raw_path = _resolve(root, args.raw_map, RAW_DEFAULT)
    cal_path = _resolve(root, args.calibrated_map, CAL_DEFAULT)
    measurements = _resolve(root, args.measurements, MEAS_DEFAULT)
    split_manifest = _resolve(root, args.split_manifest, SPLIT_DEFAULT)
    output = _resolve(root, args.output, OUT_DEFAULT)
    summary_json = _resolve(root, args.summary_json, SUMMARY_JSON_DEFAULT)
    summary_csv = _resolve(root, args.summary_csv, SUMMARY_CSV_DEFAULT)

    raw = load_joint_map(raw_path)
    cal = load_joint_map(cal_path)
    _assert_same_grid(raw, cal)

    raw_finite = np.isfinite(raw.best_rsrp_dbm)
    cal_finite = np.isfinite(cal.best_rsrp_dbm)
    raw_source = np.zeros(raw_finite.shape, dtype=np.uint8)
    cal_source = np.zeros(cal_finite.shape, dtype=np.uint8)
    raw_source[raw_finite] = 1
    cal_source[cal_finite] = 3

    measurement_split_mask, measurement_report = _measurement_split_mask(
        raw, measurements, split_manifest, args.center_arfcn, args.bandwidth_mhz
    )
    measurement_observed_mask = measurement_split_mask > 0

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        x_centers_m=raw.x_centers_m.astype(np.float64),
        y_centers_m=raw.y_centers_m.astype(np.float64),
        raw_value_source_mask=raw_source,
        calibrated_value_source_mask=cal_source,
        measurement_observed_mask=measurement_observed_mask,
        measurement_split_mask=measurement_split_mask,
        source_codebook_json=np.asarray([json.dumps(SOURCE_CODEBOOK, ensure_ascii=False)]),
        measurement_split_codebook_json=np.asarray([json.dumps(MEASUREMENT_SPLIT_CODEBOOK, ensure_ascii=False)]),
        metadata_json=np.asarray([json.dumps({
            "schema_version": 1,
            "raw_map": str(raw_path),
            "calibrated_map": str(cal_path),
            "measurements": str(measurements),
            "split_manifest": str(split_manifest),
            "measurement_values_copied_into_network_joint_map": False,
            "source_codebook": SOURCE_CODEBOOK,
            "measurement_split_codebook": MEASUREMENT_SPLIT_CODEBOOK,
        }, ensure_ascii=False)]),
    )

    outdoor = raw.outdoor_valid_mask if raw.outdoor_valid_mask is not None else np.ones(raw_finite.shape, dtype=bool)
    counts = []
    for variant, source_mask in (("raw", raw_source), ("calibrated", cal_source)):
        for code, label in SOURCE_CODEBOOK.items():
            counts.append({
                "variant": variant,
                "provenance_code": int(code),
                "provenance_label": label,
                "cell_count": int(np.sum(source_mask == code)),
            })
    for code, label in MEASUREMENT_SPLIT_CODEBOOK.items():
        counts.append({
            "variant": "measurement_observation_overlay",
            "provenance_code": int(code),
            "provenance_label": label,
            "cell_count": int(np.sum(measurement_split_mask == code)),
        })
    pd.DataFrame(counts).to_csv(summary_csv, index=False, encoding="utf-8-sig")

    summary = {
        "schema_version": 1,
        "status": "ok",
        "output_npz": str(output),
        "map_shape": list(raw.best_rsrp_dbm.shape),
        "raw_finite_simulated_cell_count": int(raw_finite.sum()),
        "calibrated_finite_cell_count": int(cal_finite.sum()),
        "outdoor_cell_count": int(np.sum(outdoor)),
        "outdoor_missing_or_invalid_cell_count": int(np.sum(outdoor & ~raw_finite)),
        "measurement_observed_cell_count": int(np.sum(measurement_observed_mask)),
        "calibration_observed_cell_count": int(np.sum((measurement_split_mask & 1) > 0)),
        "validation_observed_cell_count": int(np.sum((measurement_split_mask & 2) > 0)),
        "same_cell_calibration_validation_overlap_count": int(np.sum(measurement_split_mask == 3)),
        "measurement_values_copied_into_network_joint_map": False,
        "best_station_and_pci_are_not_changed_by_provenance_export": True,
        "source_codebook": SOURCE_CODEBOOK,
        "measurement_split_codebook": MEASUREMENT_SPLIT_CODEBOOK,
        "measurement_processing": measurement_report,
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
