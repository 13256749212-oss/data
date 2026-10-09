#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Single entry point for the campus 5G NR radio-map dataset processing pipeline."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent
PYTHON = Path(sys.executable)

SCRIPTS = {
    "align": ROOT / "workflows/preprocessing/01_align_measurements_to_blender.py",
    "extract": ROOT / "workflows/preprocessing/02_extract_multi_pci_rsrp.py",
    "preprocess": ROOT / "workflows/preprocessing/03_build_analysis_tables.py",
    "calibrate": ROOT / "workflows/parameter_calibration/run_27station_parameter_calibration.py",
    "export-dem": ROOT / "workflows/radio_map/export_bestparam_radio_maps.py",
    "export-joint-map": ROOT / "workflows/radio_map/generate_joint_best_server_4000x3000.py",
    "compare-joint-map": ROOT / "workflows/evaluation/compare_joint_map_with_measurements.py",
    "fit-network-residual": ROOT / "workflows/evaluation/fit_calibration_only_network_residual.py",
    "compare-calibrated-map": ROOT / "workflows/evaluation/compare_joint_map_with_measurements.py",
    "export-provenance": ROOT / "workflows/evaluation/export_joint_map_provenance.py",
    "export-evaluation": ROOT / "workflows/evaluation/export_evaluation.py",
    "export-metadata": ROOT / "tools/export_reproducibility_metadata.py",
}


def _project_path(value: Path | str | None, default: Path) -> Path:
    if value is None:
        return default.resolve()
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _display_command(command: Iterable[object]) -> str:
    parts = []
    for item in command:
        value = str(item)
        parts.append(f'"{value}"' if " " in value else value)
    return " ".join(parts)


def run(command: list[object]) -> int:
    print("[RUN]", _display_command(command), flush=True)
    proc = subprocess.Popen([str(x) for x in command], cwd=ROOT)
    try:
        return int(proc.wait())
    except KeyboardInterrupt:
        print("\n[STOP] Interrupted; stopping child process...", flush=True)
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        return 130


def _csv_has_data_rows(path: Path) -> bool:
    path = Path(path)
    if not path.exists() or path.stat().st_size <= 3:
        return False
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            with path.open("r", encoding=encoding, newline="") as handle:
                reader = csv.reader(handle)
                if not next(reader, None):
                    return False
                return any(any(str(value).strip() for value in row) for row in reader)
        except UnicodeDecodeError:
            continue
        except Exception:
            return False
    return False


def _ensure_calibration_measurement_table() -> int:
    """Build the long-format 27-station measurement table if it is missing."""
    target = ROOT / "data/processed/cell_pci_rsrp_long_27stations.csv"
    extracted = ROOT / "data/processed/extracted/cell_pci_rsrp_long_27stations.csv"
    aligned_dir = ROOT / "data/aligned_measurements"
    raw_dir = ROOT / "data/raw_measurements"

    if _csv_has_data_rows(target):
        return 0
    if _csv_has_data_rows(extracted):
        rc = run([PYTHON, SCRIPTS["preprocess"]])
        if rc == 0 and _csv_has_data_rows(target):
            return 0
    if aligned_dir.exists() and list(aligned_dir.glob("*_with_blender_xyz.csv")):
        rc = run([PYTHON, SCRIPTS["extract"], "--input-dir", aligned_dir,
                  "--mapping", ROOT / "config/base_station_pci_mapping.csv",
                  "--output-dir", ROOT / "data/processed/extracted"])
        if rc:
            return rc
        rc = run([PYTHON, SCRIPTS["preprocess"]])
        if rc == 0 and _csv_has_data_rows(target):
            return 0
    if raw_dir.exists() and list(raw_dir.glob("*.csv")):
        rc = run([PYTHON, SCRIPTS["align"], "--force"])
        if rc:
            return rc
        rc = run([PYTHON, SCRIPTS["extract"], "--input-dir", aligned_dir,
                  "--mapping", ROOT / "config/base_station_pci_mapping.csv",
                  "--output-dir", ROOT / "data/processed/extracted"])
        if rc:
            return rc
        rc = run([PYTHON, SCRIPTS["preprocess"]])
        if rc == 0 and _csv_has_data_rows(target):
            return 0
    print("[ERROR] Could not build data/processed/cell_pci_rsrp_long_27stations.csv", flush=True)
    return 2


def _add_compare_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--map-npz", type=Path, default=None)
    p.add_argument("--measurements", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--split-manifest", type=Path, default=None)
    p.add_argument("--evaluation-subset", choices=["validation", "calibration", "all"], default="validation")
    p.add_argument("--allow-overlap-evaluation", action="store_true")
    p.add_argument("--station-mapping", type=Path, default=None)
    p.add_argument("--keep-duplicate-map-cells", action="store_true")
    p.add_argument("--rsrp-min-dbm", type=float, default=-140.0)
    p.add_argument("--rsrp-max-dbm", type=float, default=-40.0)
    p.add_argument("--display-min-dbm", type=float, default=-120.0)
    p.add_argument("--display-max-dbm", type=float, default=-40.0)
    p.add_argument("--center-arfcn", type=int, default=513000)
    p.add_argument("--bandwidth-mhz", type=float, default=100.0)
    p.add_argument("--no-filter-n41", action="store_true")
    p.add_argument("--no-filter-center-arfcn", action="store_true")
    p.add_argument("--no-filter-bandwidth", action="store_true")
    p.add_argument("--skip-figures", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Campus-scale 5G NR radio-propagation dataset processing pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("check", help="Check required scene/configuration files and available data")
    sub.add_parser("export-metadata", help="Export reproducibility metadata")

    p = sub.add_parser("align", help="Align WGS84 measurements with the Blender-local scene")
    p.add_argument("--force", action="store_true")
    sub.add_parser("extract", help="Expand multi-PCI RSRP observations")
    sub.add_parser("preprocess", help="Build processed analysis tables")
    p = sub.add_parser("prepare-data", help="Run align, extract, and preprocess")
    p.add_argument("--force", action="store_true")

    p = sub.add_parser("calibrate", help="Calibrate station parameters using calibration trajectories")
    p.add_argument("--stations", default="all")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--stop-on-error", action="store_true")
    p.add_argument("--validation-fraction", type=float, default=0.25)
    p.add_argument("--validation-seed", type=int, default=20261003)
    p.add_argument("--min-calibration-rows-per-pci", type=int, default=6)
    p.add_argument("--disable-independent-validation-split", action="store_true")
    p.add_argument("--no-final-refinement", action="store_true")
    p.add_argument("--global-epre-calibration-file", type=Path, default=None)
    p.add_argument("--global-epre-offset-db", type=float, default=None)
    p.add_argument("--output-root", type=Path, default=None)
    p.add_argument("--objective-mode", choices=["pooled_rmse", "trajectory_balanced_pci_rmse", "reliability_balanced_pci_rmse"], default="reliability_balanced_pci_rmse")
    p.add_argument("--reliability-min-trajectory-points", type=int, default=20)
    p.add_argument("--reliability-trajectory-weight-cap", type=int, default=250)
    p.add_argument("--reliability-min-pci-points", type=int, default=3)
    p.add_argument("--reliability-pci-weight-cap", type=int, default=100)
    p.add_argument("--power-regularization-lambda", type=float, default=None)
    p.add_argument("--power-regularization-anchor-dbm", type=float, default=None)
    p.add_argument("--reference-geometry-file", type=Path, default=None)
    p.add_argument("--reestimate-geometry", action="store_true", help="Re-estimate station geometry instead of using the released reference geometry")

    p = sub.add_parser("export-dem", help="Generate terrain-following per-station radio maps")
    p.add_argument("--stations", default="all")
    p.add_argument("--force", action="store_true")
    p.add_argument("--continue-on-error", action="store_true")

    p = sub.add_parser("export-joint-map", help="Generate the 27-station 4000 x 3000 best-server map")
    p.add_argument("--center-x", type=float, default=267.5)
    p.add_argument("--center-y", type=float, default=69.53)
    p.add_argument("--size-x", type=int, default=4000)
    p.add_argument("--size-y", type=int, default=3000)
    p.add_argument("--cell-size", type=float, default=1.0)
    p.add_argument("--tile-size", type=int, default=500)
    p.add_argument("--station-batch-size", type=int, default=4)
    p.add_argument("--batch-count", type=int, default=None)
    p.add_argument("--samples-per-tx", type=int, default=None)
    p.add_argument("--max-depth", type=int, default=None)
    p.add_argument("--plot-only", action="store_true")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--only-tile", default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--continue-on-error", action="store_true")

    p = sub.add_parser("compare-joint-map", help="Evaluate the raw joint map against the selected trajectory subset")
    _add_compare_args(p)
    p = sub.add_parser("compare-calibrated-map", help="Evaluate the calibration-adjusted joint map")
    _add_compare_args(p)

    p = sub.add_parser("fit-network-residual", help="Fit the calibration-only network residual model")
    p.add_argument("--raw-map", type=Path, default=None)
    p.add_argument("--measurements", type=Path, default=None)
    p.add_argument("--split-manifest", type=Path, default=None)
    p.add_argument("--station-mapping", type=Path, default=None)
    p.add_argument("--output-model", type=Path, default=None)
    p.add_argument("--output-cv", type=Path, default=None)
    p.add_argument("--output-matched", type=Path, default=None)
    p.add_argument("--output-map", type=Path, default=None)
    p.add_argument("--models", default="identity,offset,affine_rsrp,distance,affine_rsrp_distance")
    p.add_argument("--ridge-grid", default="0,0.1,1,10")
    p.add_argument("--correction-clip-db", type=float, default=12.0)
    p.add_argument("--min-cv-gain-db", type=float, default=0.05)
    p.add_argument("--chunk-rows", type=int, default=256)

    p = sub.add_parser("export-provenance", help="Export joint-map cell provenance and measurement-observation masks")
    p.add_argument("--raw-map", type=Path, default=None)
    p.add_argument("--calibrated-map", type=Path, default=None)
    p.add_argument("--measurements", type=Path, default=None)
    p.add_argument("--split-manifest", type=Path, default=None)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--summary-json", type=Path, default=None)
    p.add_argument("--summary-csv", type=Path, default=None)

    p = sub.add_parser("export-evaluation", help="Export raw/calibrated evaluation tables and figures")
    p.add_argument("--raw-map", type=Path, default=None)
    p.add_argument("--calibrated-map", type=Path, default=None)
    p.add_argument("--measurements", type=Path, default=None)
    p.add_argument("--split-manifest", type=Path, default=None)
    p.add_argument("--station-mapping", type=Path, default=None)
    p.add_argument("--model", type=Path, default=None)
    p.add_argument("--cv-candidates", type=Path, default=None)
    p.add_argument("--output-root", type=Path, default=None)
    p.add_argument("--skip-validation-figures", action="store_true")

    return parser


def _compare_command(args, calibrated: bool) -> list[object]:
    default_map = ROOT / "outputs/joint_best_server_4000x3000" / (
        "joint_best_server_27stations_4000x3000_calibrated.npz" if calibrated
        else "joint_best_server_27stations_4000x3000.npz"
    )
    default_out = ROOT / ("outputs/joint_map_measurement_comparison_calibrated" if calibrated else "outputs/joint_map_measurement_comparison")
    cmd = [PYTHON, SCRIPTS["compare-calibrated-map" if calibrated else "compare-joint-map"],
           "--project-root", ROOT,
           "--map-npz", _project_path(args.map_npz, default_map),
           "--measurements", _project_path(args.measurements, ROOT / "data/processed/cell_pci_rsrp_long_27stations.csv"),
           "--output-dir", _project_path(args.output_dir, default_out),
           "--split-manifest", _project_path(args.split_manifest, ROOT / "outputs/parameter_calibration/calibration_validation_split.json"),
           "--station-mapping", _project_path(args.station_mapping, ROOT / "config/base_station_pci_mapping.csv"),
           "--evaluation-subset", args.evaluation_subset,
           "--rsrp-min-dbm", args.rsrp_min_dbm, "--rsrp-max-dbm", args.rsrp_max_dbm,
           "--display-min-dbm", args.display_min_dbm, "--display-max-dbm", args.display_max_dbm,
           "--center-arfcn", args.center_arfcn, "--bandwidth-mhz", args.bandwidth_mhz]
    for flag, enabled in (("--allow-overlap-evaluation", args.allow_overlap_evaluation),
                          ("--keep-duplicate-map-cells", args.keep_duplicate_map_cells),
                          ("--no-filter-n41", args.no_filter_n41),
                          ("--no-filter-center-arfcn", args.no_filter_center_arfcn),
                          ("--no-filter-bandwidth", args.no_filter_bandwidth),
                          ("--skip-figures", args.skip_figures)):
        if enabled:
            cmd.append(flag)
    return cmd


def main() -> int:
    parser = build_parser()
    args, extra = parser.parse_known_args()
    c = args.command

    if c == "check":
        return run([PYTHON, ROOT / "check_project_layout.py", *extra])
    if c == "prepare-data":
        align = [PYTHON, SCRIPTS["align"]] + (["--force"] if args.force else [])
        steps = [align,
                 [PYTHON, SCRIPTS["extract"], "--input-dir", ROOT / "data/aligned_measurements", "--mapping", ROOT / "config/base_station_pci_mapping.csv", "--output-dir", ROOT / "data/processed/extracted"],
                 [PYTHON, SCRIPTS["preprocess"]]]
        for step in steps:
            rc = run(step)
            if rc:
                return rc
        return 0
    if c == "export-metadata":
        return run([PYTHON, SCRIPTS[c], *extra])
    if c == "align":
        cmd = [PYTHON, SCRIPTS[c]] + (["--force"] if args.force else []) + extra
        return run(cmd)
    if c == "extract":
        return run([PYTHON, SCRIPTS[c], "--input-dir", ROOT / "data/aligned_measurements", "--mapping", ROOT / "config/base_station_pci_mapping.csv", "--output-dir", ROOT / "data/processed/extracted", *extra])
    if c == "preprocess":
        return run([PYTHON, SCRIPTS[c], *extra])

    if c == "calibrate":
        rc = _ensure_calibration_measurement_table()
        if rc:
            return rc
        cmd = [PYTHON, SCRIPTS[c],
               "--config", ROOT / "workflows/parameter_calibration/config.yaml",
               "--measurements", ROOT / "data/processed/cell_pci_rsrp_long_27stations.csv",
               "--stations", args.stations,
               "--ground", ROOT / "assets/ground.ply",
               "--buildings", ROOT / "assets/ynu_chenggong_campus-001.ply",
               "--validation-fraction", args.validation_fraction,
               "--validation-seed", args.validation_seed,
               "--min-calibration-rows-per-pci", args.min_calibration_rows_per_pci,
               "--global-epre-calibration-file", _project_path(args.global_epre_calibration_file, ROOT / "config/global_epre_calibration.json"),
               "--objective-mode", args.objective_mode,
               "--reliability-min-trajectory-points", args.reliability_min_trajectory_points,
               "--reliability-trajectory-weight-cap", args.reliability_trajectory_weight_cap,
               "--reliability-min-pci-points", args.reliability_min_pci_points,
               "--reliability-pci-weight-cap", args.reliability_pci_weight_cap]
        if args.quick: cmd.append("--quick")
        if args.force: cmd.append("--force")
        if args.stop_on_error: cmd.append("--stop-on-error")
        if args.disable_independent_validation_split: cmd.append("--disable-independent-validation-split")
        if args.no_final_refinement: cmd.append("--no-final-refinement")
        if args.global_epre_offset_db is not None: cmd += ["--global-epre-offset-db", args.global_epre_offset_db]
        if args.output_root is not None: cmd += ["--output-root", _project_path(args.output_root, ROOT / "outputs/parameter_calibration")]
        if args.power_regularization_lambda is not None: cmd += ["--power-regularization-lambda", args.power_regularization_lambda]
        if args.power_regularization_anchor_dbm is not None: cmd += ["--power-regularization-anchor-dbm", args.power_regularization_anchor_dbm]
        if args.reestimate_geometry:
            if args.reference_geometry_file is not None:
                cmd += ["--reference-geometry-file", _project_path(args.reference_geometry_file, ROOT / "config/reference_geometry.csv")]
        else:
            cmd += [
                "--reference-geometry-file", _project_path(args.reference_geometry_file, ROOT / "config/reference_geometry.csv"),
                "--constrain-geometry-to-reference",
                "--reference-height-radius-m", 0.0,
                "--reference-azimuth-radius-deg", 0.0,
                "--reference-downtilt-radius-deg", 0.0,
                "--geometry-regularization-lambda-db2", 1.0,
                "--anchor-power-to-reference",
            ]
        return run(cmd + extra)

    if c == "export-dem":
        cmd = [PYTHON, SCRIPTS[c],
               "--project-root", ROOT,
               "--measurements", ROOT / "data/processed/cell_pci_rsrp_long_27stations.csv",
               "--summary-csv", ROOT / "outputs/parameter_calibration/all_27stations_summary.csv",
               "--ground", ROOT / "assets/ground.ply",
               "--buildings", ROOT / "assets/ynu_chenggong_campus-001.ply",
               "--stations", args.stations]
        if args.force: cmd.append("--force")
        if args.continue_on_error: cmd.append("--continue-on-error")
        return run(cmd + extra)

    if c == "export-joint-map":
        cmd = [PYTHON, SCRIPTS[c], "--project-root", ROOT,
               "--summary-csv", ROOT / "outputs/parameter_calibration/all_27stations_summary.csv",
               "--directions-csv", ROOT / "outputs/parameter_calibration/estimated_initial_directions_27stations.csv",
               "--ground", ROOT / "assets/ground.ply",
               "--buildings", ROOT / "assets/ynu_chenggong_campus-001.ply",
               "--center-x", args.center_x, "--center-y", args.center_y,
               "--size-x", args.size_x, "--size-y", args.size_y,
               "--cell-size", args.cell_size, "--tile-size", args.tile_size,
               "--station-batch-size", args.station_batch_size]
        if args.batch_count is not None: cmd += ["--batch-count", args.batch_count]
        if args.samples_per_tx is not None: cmd += ["--samples-per-tx", args.samples_per_tx]
        if args.max_depth is not None: cmd += ["--max-depth", args.max_depth]
        for flag, enabled in (("--plot-only", args.plot_only), ("--quick", args.quick), ("--dry-run", args.dry_run), ("--force", args.force), ("--continue-on-error", args.continue_on_error)):
            if enabled: cmd.append(flag)
        if args.only_tile: cmd += ["--only-tile", args.only_tile]
        return run(cmd + extra)

    if c == "compare-joint-map":
        return run(_compare_command(args, False) + extra)
    if c == "compare-calibrated-map":
        return run(_compare_command(args, True) + extra)

    if c == "fit-network-residual":
        cmd = [PYTHON, SCRIPTS[c], "--project-root", ROOT,
               "--raw-map", _project_path(args.raw_map, ROOT / "outputs/joint_best_server_4000x3000/joint_best_server_27stations_4000x3000.npz"),
               "--measurements", _project_path(args.measurements, ROOT / "data/processed/cell_pci_rsrp_long_27stations.csv"),
               "--split-manifest", _project_path(args.split_manifest, ROOT / "outputs/parameter_calibration/calibration_validation_split.json"),
               "--station-mapping", _project_path(args.station_mapping, ROOT / "config/base_station_pci_mapping.csv"),
               "--output-model", _project_path(args.output_model, ROOT / "config/network_residual_calibration.json"),
               "--output-cv", _project_path(args.output_cv, ROOT / "outputs/network_residual_calibration/cv_candidates.csv"),
               "--output-matched", _project_path(args.output_matched, ROOT / "outputs/network_residual_calibration/calibration_matched_cells.csv"),
               "--output-map", _project_path(args.output_map, ROOT / "outputs/joint_best_server_4000x3000/joint_best_server_27stations_4000x3000_calibrated.npz"),
               "--models", args.models, "--ridge-grid", args.ridge_grid,
               "--correction-clip-db", args.correction_clip_db,
               "--min-cv-gain-db", args.min_cv_gain_db, "--chunk-rows", args.chunk_rows]
        return run(cmd + extra)

    if c == "export-provenance":
        cmd = [PYTHON, SCRIPTS[c], "--project-root", ROOT,
               "--raw-map", _project_path(args.raw_map, ROOT / "outputs/joint_best_server_4000x3000/joint_best_server_27stations_4000x3000.npz"),
               "--calibrated-map", _project_path(args.calibrated_map, ROOT / "outputs/joint_best_server_4000x3000/joint_best_server_27stations_4000x3000_calibrated.npz"),
               "--measurements", _project_path(args.measurements, ROOT / "data/processed/cell_pci_rsrp_long_27stations.csv"),
               "--split-manifest", _project_path(args.split_manifest, ROOT / "outputs/parameter_calibration/calibration_validation_split.json"),
               "--output", _project_path(args.output, ROOT / "outputs/joint_best_server_4000x3000/joint_best_server_27stations_4000x3000_provenance.npz"),
               "--summary-json", _project_path(args.summary_json, ROOT / "outputs/joint_best_server_4000x3000/joint_map_provenance_summary.json"),
               "--summary-csv", _project_path(args.summary_csv, ROOT / "outputs/joint_best_server_4000x3000/joint_map_provenance_counts.csv")]
        return run(cmd + extra)

    if c == "export-evaluation":
        cmd = [PYTHON, SCRIPTS[c], "--project-root", ROOT,
               "--raw-map", _project_path(args.raw_map, ROOT / "outputs/joint_best_server_4000x3000/joint_best_server_27stations_4000x3000.npz"),
               "--calibrated-map", _project_path(args.calibrated_map, ROOT / "outputs/joint_best_server_4000x3000/joint_best_server_27stations_4000x3000_calibrated.npz"),
               "--measurements", _project_path(args.measurements, ROOT / "data/processed/cell_pci_rsrp_long_27stations.csv"),
               "--split-manifest", _project_path(args.split_manifest, ROOT / "outputs/parameter_calibration/calibration_validation_split.json"),
               "--station-mapping", _project_path(args.station_mapping, ROOT / "config/base_station_pci_mapping.csv"),
               "--model", _project_path(args.model, ROOT / "config/network_residual_calibration.json"),
               "--cv-candidates", _project_path(args.cv_candidates, ROOT / "outputs/network_residual_calibration/cv_candidates.csv"),
               "--output-root", _project_path(args.output_root, ROOT / "outputs/evaluation")]
        if args.skip_validation_figures: cmd.append("--skip-validation-figures")
        return run(cmd + extra)

    parser.error(f"unsupported command: {c}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
