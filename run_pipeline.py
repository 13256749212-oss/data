#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Single entry point for the campus 5G radio-map dataset pipeline."""
from __future__ import annotations

import argparse
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
    "reconstruct": ROOT / "workflows/reconstruction/run_reconstruction.py",
    "localize": ROOT / "workflows/localization/run_27stations_two_branch_localization_progressive_nls.py",
    "localize-sweep": ROOT / "workflows/localization/run_27stations_two_branch_localization_progressive_nls.py",
    "localize-all-pci": ROOT / "workflows/localization/all_pci_cluster_localization.py",
    "visualize-measurements": ROOT / "workflows/visualization/plot_raw_measurement_dataset.py",
    "plot-output-structure": ROOT / "workflows/visualization/plot_dataset_output_structure.py",
}


def _project_path(value: Path | str | None, default: Path) -> Path:
    if value is None:
        return default.resolve()
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _display_command(command: Iterable[object]) -> str:
    parts: list[str] = []
    for item in command:
        text = str(item)
        parts.append(f'"{text}"' if " " in text else text)
    return " ".join(parts)


def run(command: list[object]) -> int:
    print("[RUN]", _display_command(command), flush=True)
    proc = subprocess.Popen([str(x) for x in command], cwd=ROOT)
    try:
        return int(proc.wait())
    except KeyboardInterrupt:
        print("\n[STOP] Ctrl+C received; stopping child process cleanly...", flush=True)
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        return 130


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="校园5G无线电地图数据集：统一流水线入口"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("check", help="检查场景、配置和当前数据阶段")
    sub.add_parser("migrate-output-names", help="将旧结果目录的04_/11_/12_前缀安全移除")

    p = sub.add_parser("align", help="WGS84实测CSV与BlenderGIS场景对齐")
    p.add_argument("--force", action="store_true")

    sub.add_parser("extract", help="展开同一行中的多PCI–RSRP观测")
    sub.add_parser("preprocess", help="生成正式长表、1 m表和2.77 m表")

    p = sub.add_parser("prepare-data", help="依次执行align、extract和preprocess")
    p.add_argument("--force", action="store_true", help="重新生成坐标对齐结果")

    p = sub.add_parser("calibrate", help="校准高度、功率、方向角和下倾角")
    p.add_argument("--stations", default="all")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--stop-on-error", action="store_true")

    p = sub.add_parser("export-dem", help="用最佳参数导出单站DEM+1.5 m地图")
    p.add_argument("--stations", default="all")
    p.add_argument("--force", action="store_true")
    p.add_argument("--continue-on-error", action="store_true")

    p = sub.add_parser("export-joint-map", help="生成27站联合4000×3000 m best-server地图")
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
    p.add_argument(
        "--campus-outline-source",
        type=Path,
        default=None,
        help="校园外围轮廓来源：对齐实测CSV或data/aligned_measurements目录",
    )
    p.add_argument("--no-campus-outline", action="store_true")
    p.add_argument("--campus-outline-linewidth", type=float, default=1.8)
    p.add_argument("--campus-outline-buffer-m", type=float, default=40.0)
    p.add_argument("--campus-outline-max-buffer-m", type=float, default=120.0)
    p.add_argument("--campus-outline-grid-m", type=float, default=5.0)
    p.add_argument("--campus-outline-smooth-m", type=float, default=18.0)
    p.add_argument("--campus-outline-min-coverage", type=float, default=0.97)
    p.add_argument(
        "--campus-outline-straight-segment-m",
        type=float,
        default=220.0,
        help="S38附近用户标出的小弯折直线化半跨度(m)，默认220；0关闭",
    )
    p.add_argument("--plot-only", action="store_true")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--only-tile", default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--continue-on-error", action="store_true")

    p = sub.add_parser("compare-joint-map", help="仅用已保存联合地图与实测轨迹计算RMSE")
    p.add_argument("--map-npz", type=Path, default=None)
    p.add_argument("--measurements", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
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

    p = sub.add_parser("reconstruct", help="单次双分支最近邻重构：1%%--10%%使用同一条严格嵌套空间序列，每个比例仅生成1次地图；RMSE按完整512m×512m有效栅格计算")
    p.add_argument("--station-id", type=int, default=3)
    p.add_argument("--pci", type=int, default=558)
    p.add_argument("--measurements", type=Path, default=None)
    p.add_argument("--filled-reference-npz", type=Path, default=None, help="measurement-filled单PCI NPZ；全部有限室外栅格作为RMSE参考域")
    p.add_argument("--output-root", type=Path, default=None)
    p.add_argument("--percentages", default="1,2,3,4,5,6,7,8,9,10")
    p.add_argument("--random-seed", type=int, default=20260805)
    p.add_argument("--random-trials", type=int, default=1, help="兼容旧参数；正式重构固定为1次")
    p.add_argument("--selection-mode", choices=["voronoi-safe-adaptive-nested", "adaptive-domain-nested", "hierarchical-nested", "adaptive-progressive", "stable-spatial", "progressive-coverage", "random"], default="voronoi-safe-adaptive-nested")
    p.add_argument("--min-rsrp-dbm", type=float, default=-120.0)
    p.add_argument("--max-rsrp-dbm", type=float, default=-40.0)
    p.add_argument("--display-min-dbm", type=float, default=-120.0)
    p.add_argument("--display-max-dbm", type=float, default=-40.0)
    p.add_argument("--dpi", type=int, default=1000)
    p.add_argument("--skip-figures", action="store_true")
    p.add_argument("--simulation-mode", choices=["without", "with", "compare"], default="compare")
    p.add_argument("--simulation-npz", type=Path, default=None, help="固定纯Sionna单PCI NPZ；默认自动搜索")

    p = sub.add_parser(
        "localize",
        help="严格双分支定位：相同实测点下比较Measurement-only与Measurement–simulation，只输出两项RMSE",
    )
    p.add_argument("--measurements", type=Path, default=None)
    p.add_argument("--points-per-station", type=int, default=10)
    p.add_argument("--random-trials", type=int, default=10)
    p.add_argument("--random-seed", type=int, default=20260805)
    p.add_argument("--station-ids", default="all")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--simulation-root", type=Path, default=None, help="固定Sionna RT单PCI地图搜索根目录")

    p = sub.add_parser(
        "localize-sweep",
        help="10--15点严格双分支渐进嵌套定位；正式输出仅Measurement-only RMSE和Measurement–simulation RMSE",
    )
    p.add_argument("--measurements", type=Path, default=None)
    p.add_argument("--output-root", type=Path, default=None)
    p.add_argument("--point-counts", default="10,11,12,13,14,15")
    p.add_argument("--random-trials", type=int, default=10)
    p.add_argument("--random-seed", type=int, default=20260805)
    p.add_argument("--station-ids", default="all")
    p.add_argument("--simulation-root", type=Path, default=None, help="固定Sionna RT单PCI地图搜索根目录")
    p.add_argument("--progressive-min-improvement-db", type=float, default=0.35)
    p.add_argument("--validation-points", type=int, default=24)
    p.add_argument("--validation-min-improvement-db", type=float, default=0.08)
    p.add_argument("--simulation-measured-tolerance-db", type=float, default=0.15)
    p.add_argument("--fast-max-nfev", type=int, default=36)
    p.add_argument("--fast-fixed-nfev", type=int, default=14)
    p.add_argument("--fast-initial-starts", type=int, default=3)

    p = sub.add_parser(
        "localize-all-pci",
        help="使用全部映射PCI原始实测点定位（不做空间聚合）；--station-id all可一次运行全部基站，22号自动按单PCI 800处理",
    )
    p.add_argument("--measurements", type=Path, default=None)
    p.add_argument("--station-id", default="2", help="基站编号，或 all")
    p.add_argument("--dbscan-eps-m", type=float, default=12.0)
    p.add_argument("--dbscan-min-samples", type=int, default=3)
    p.add_argument("--calibration-root", type=Path, default=None, help="默认outputs/parameter_calibration，用于绘制最终调参扇区方向")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--skip-figure", action="store_true")
    p.add_argument("--stop-on-error", action="store_true")

    p = sub.add_parser("visualize-measurements", help="生成原始采集数据的英文论文展示图（1000 dpi PNG）")
    p.add_argument("--input-dir", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--dpi", type=int, default=1000)
    p.add_argument("--top-pci", type=int, default=25)
    p.add_argument("--skip-session-figure", action="store_true")

    p = sub.add_parser("plot-output-structure", help="生成outputs数据产品的英文内容结构图（1000 dpi PNG）")
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--dpi", type=int, default=1000)

    p = sub.add_parser("visualize-dataset", help="依次生成原始采集数据展示图和数据集output内容结构图")
    p.add_argument("--input-dir", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--dpi", type=int, default=1000)
    p.add_argument("--top-pci", type=int, default=25)

    sub.add_parser("test", help="运行轻量单元测试")
    return parser


def main() -> int:
    parser = build_parser()
    args, extra = parser.parse_known_args()
    command = args.command

    if command == "check":
        return run([PYTHON, ROOT / "check_project_layout.py", *extra])

    if command == "migrate-output-names":
        return run([PYTHON, ROOT / "tools/migrate_output_folder_names.py", *extra])

    if command == "test":
        return run([PYTHON, "-m", "pytest", ROOT / "tests", *extra])

    if command == "visualize-dataset":
        input_dir = _project_path(args.input_dir, ROOT / "data/raw_measurements")
        output_dir = _project_path(args.output_dir, ROOT / "outputs/dataset_visualization")
        commands = [
            [PYTHON, SCRIPTS["visualize-measurements"], "--input-dir", input_dir, "--output-dir", output_dir, "--dpi", args.dpi, "--top-pci", args.top_pci],
            [PYTHON, SCRIPTS["plot-output-structure"], "--project-root", ROOT, "--output", output_dir / "05_dataset_output_structure.png", "--dpi", args.dpi],
        ]
        for item in commands:
            rc = run(item)
            if rc:
                return rc
        return 0

    if command == "prepare-data":
        align_cmd: list[object] = [PYTHON, SCRIPTS["align"]]
        if args.force:
            align_cmd.append("--force")
        commands = [
            align_cmd,
            [
                PYTHON,
                SCRIPTS["extract"],
                "--input-dir", ROOT / "data/aligned_measurements",
                "--mapping", ROOT / "config/base_station_pci_mapping.csv",
                "--output-dir", ROOT / "data/processed/extracted",
            ],
            [PYTHON, SCRIPTS["preprocess"]],
        ]
        for item in commands:
            rc = run(item)
            if rc:
                return rc
        return 0

    script = SCRIPTS[command]
    cmd: list[object] = [PYTHON, script]

    if command == "align":
        if args.force:
            cmd.append("--force")

    elif command == "extract":
        cmd += [
            "--input-dir", ROOT / "data/aligned_measurements",
            "--mapping", ROOT / "config/base_station_pci_mapping.csv",
            "--output-dir", ROOT / "data/processed/extracted",
        ]

    elif command == "preprocess":
        pass

    elif command == "calibrate":
        cmd += [
            "--config", ROOT / "workflows/parameter_calibration/config.yaml",
            "--measurements", ROOT / "data/processed/cell_pci_rsrp_long_27stations.csv",
            "--stations", args.stations,
            "--ground", ROOT / "assets/ground.ply",
            "--buildings", ROOT / "assets/ynu_chenggong_campus-001.ply",
        ]
        if args.quick:
            cmd.append("--quick")
        if args.force:
            cmd.append("--force")
        if args.stop_on_error:
            cmd.append("--stop-on-error")

    elif command == "export-dem":
        cmd += [
            "--project-root", ROOT,
            "--measurements", ROOT / "data/processed/cell_pci_rsrp_long_27stations.csv",
            "--summary-csv", ROOT / "outputs/parameter_calibration/all_27stations_summary.csv",
            "--ground", ROOT / "assets/ground.ply",
            "--buildings", ROOT / "assets/ynu_chenggong_campus-001.ply",
            "--stations", args.stations,
        ]
        if args.force:
            cmd.append("--force")
        if args.continue_on_error:
            cmd.append("--continue-on-error")

    elif command == "export-joint-map":
        cmd += [
            "--project-root", ROOT,
            "--summary-csv", ROOT / "outputs/parameter_calibration/all_27stations_summary.csv",
            "--directions-csv", ROOT / "outputs/parameter_calibration/estimated_initial_directions_27stations.csv",
            "--ground", ROOT / "assets/ground.ply",
            "--buildings", ROOT / "assets/ynu_chenggong_campus-001.ply",
            "--center-x", args.center_x,
            "--center-y", args.center_y,
            "--size-x", args.size_x,
            "--size-y", args.size_y,
            "--cell-size", args.cell_size,
            "--tile-size", args.tile_size,
            "--station-batch-size", args.station_batch_size,
            "--campus-outline-linewidth", args.campus_outline_linewidth,
            "--campus-outline-buffer-m", args.campus_outline_buffer_m,
            "--campus-outline-max-buffer-m", args.campus_outline_max_buffer_m,
            "--campus-outline-grid-m", args.campus_outline_grid_m,
            "--campus-outline-smooth-m", args.campus_outline_smooth_m,
            "--campus-outline-min-coverage", args.campus_outline_min_coverage,
            "--campus-outline-straight-segment-m", args.campus_outline_straight_segment_m,
        ]
        if args.campus_outline_source is not None:
            cmd += ["--campus-outline-source", _project_path(args.campus_outline_source, args.campus_outline_source)]
        if args.no_campus_outline:
            cmd.append("--no-campus-outline")
        if args.plot_only:
            cmd.append("--plot-only")
        if args.batch_count is not None:
            cmd += ["--batch-count", args.batch_count]
        if args.samples_per_tx is not None:
            cmd += ["--samples-per-tx", args.samples_per_tx]
        if args.max_depth is not None:
            cmd += ["--max-depth", args.max_depth]
        if args.quick:
            cmd.append("--quick")
        if args.dry_run:
            cmd.append("--dry-run")
        if args.only_tile:
            cmd += ["--only-tile", args.only_tile]
        if args.force:
            cmd.append("--force")
        if args.continue_on_error:
            cmd.append("--continue-on-error")

    elif command == "compare-joint-map":
        map_npz = _project_path(
            args.map_npz,
            ROOT / "outputs/joint_best_server_4000x3000/joint_best_server_27stations_4000x3000.npz",
        )
        measurements = _project_path(
            args.measurements,
            ROOT / "data/processed/cell_pci_rsrp_long_27stations.csv",
        )
        output_dir = _project_path(
            args.output_dir,
            ROOT / "outputs/joint_map_measurement_comparison",
        )
        cmd += [
            "--project-root", ROOT,
            "--map-npz", map_npz,
            "--measurements", measurements,
            "--output-dir", output_dir,
            "--rsrp-min-dbm", args.rsrp_min_dbm,
            "--rsrp-max-dbm", args.rsrp_max_dbm,
            "--display-min-dbm", args.display_min_dbm,
            "--display-max-dbm", args.display_max_dbm,
            "--center-arfcn", args.center_arfcn,
            "--bandwidth-mhz", args.bandwidth_mhz,
        ]
        if args.keep_duplicate_map_cells:
            cmd.append("--keep-duplicate-map-cells")
        if args.no_filter_n41:
            cmd.append("--no-filter-n41")
        if args.no_filter_center_arfcn:
            cmd.append("--no-filter-center-arfcn")
        if args.no_filter_bandwidth:
            cmd.append("--no-filter-bandwidth")

    elif command == "reconstruct":
        measurements = _project_path(
            args.measurements,
            ROOT / "data/processed/cell_pci_rsrp_1m_calibration.csv",
        )
        output_root = _project_path(
            args.output_root,
            ROOT / "outputs/radio_map_reconstruction_nn_fullgrid_two_branch",
        )
        cmd += [
            "--project-root", ROOT,
            "--measurements", measurements,
            "--output-root", output_root,
            "--station-id", args.station_id,
            "--pci", args.pci,
            "--percentages", args.percentages,
            "--random-seed", args.random_seed,
            "--random-trials", args.random_trials,
            "--selection-mode", args.selection_mode,
            "--min-rsrp-dbm", args.min_rsrp_dbm,
            "--max-rsrp-dbm", args.max_rsrp_dbm,
            "--display-min-dbm", args.display_min_dbm,
            "--display-max-dbm", args.display_max_dbm,
            "--dpi", args.dpi,
            "--simulation-mode", args.simulation_mode,
        ]
        if args.filled_reference_npz is not None:
            cmd += ["--filled-reference-npz", _project_path(args.filled_reference_npz, ROOT / "outputs")]
        if args.simulation_npz is not None:
            cmd += ["--simulation-npz", _project_path(args.simulation_npz, ROOT / "outputs")]
        if args.skip_figures:
            cmd.append("--skip-figures")

    elif command == "localize":
        measurements = _project_path(
            args.measurements,
            ROOT / "data/processed/cell_pci_rsrp_long_27stations.csv",
        )
        output_dir = _project_path(
            args.output_dir,
            ROOT / "outputs" / f"localization_two_branch_{args.points_per_station}points",
        )
        cmd += [
            "--project-root", ROOT,
            "--measurements", measurements,
            "--output-root", output_dir,
            "--points-per-station", args.points_per_station,
            "--random-trials", args.random_trials,
            "--random-seed", args.random_seed,
            "--station-ids", args.station_ids,
            "--simulation-weight", 0.50,
            "--simulation-mode", "compare",
            "--direction-prior-mode", "off",
        ]
        if args.simulation_root is not None:
            cmd += ["--simulation-root", _project_path(args.simulation_root, ROOT / "outputs")]

    elif command == "localize-sweep":
        measurements = _project_path(
            args.measurements,
            ROOT / "data/processed/cell_pci_rsrp_long_27stations.csv",
        )
        output_root = _project_path(
            args.output_root,
            ROOT / "outputs/localization_two_branch_rmse_only",
        )
        cmd += [
            "--project-root", ROOT,
            "--measurements", measurements,
            "--output-root", output_root,
            "--point-counts", args.point_counts,
            "--random-trials", args.random_trials,
            "--random-seed", args.random_seed,
            "--station-ids", args.station_ids,
            "--simulation-weight", 0.50,
            "--progressive-min-improvement-db", args.progressive_min_improvement_db,
            "--validation-points", args.validation_points,
            "--validation-min-improvement-db", args.validation_min_improvement_db,
            "--simulation-measured-tolerance-db", args.simulation_measured_tolerance_db,
            "--fast-max-nfev", args.fast_max_nfev,
            "--fast-fixed-nfev", args.fast_fixed_nfev,
            "--fast-initial-starts", args.fast_initial_starts,
            "--simulation-mode", "compare",
            "--direction-prior-mode", "off",
        ]
        if args.simulation_root is not None:
            cmd += ["--simulation-root", _project_path(args.simulation_root, ROOT / "outputs")]

    elif command == "visualize-measurements":
        input_dir = _project_path(args.input_dir, ROOT / "data/raw_measurements")
        output_dir = _project_path(args.output_dir, ROOT / "outputs/dataset_visualization")
        cmd += ["--input-dir", input_dir, "--output-dir", output_dir, "--dpi", args.dpi, "--top-pci", args.top_pci]
        if args.skip_session_figure:
            cmd.append("--skip-session-figure")

    elif command == "plot-output-structure":
        output = _project_path(args.output, ROOT / "outputs/dataset_visualization/05_dataset_output_structure.png")
        cmd += ["--project-root", ROOT, "--output", output, "--dpi", args.dpi]

    elif command == "localize-all-pci":
        measurements = _project_path(
            args.measurements,
            ROOT / "data/processed/cell_pci_rsrp_long_27stations.csv",
        )
        cmd += [
            "--project-root", ROOT,
            "--measurements", measurements,
            "--station-id", str(args.station_id),
            "--dbscan-eps-m", args.dbscan_eps_m,
            "--dbscan-min-samples", args.dbscan_min_samples,
        ]
        if args.output_dir is not None:
            cmd += ["--output-dir", _project_path(args.output_dir, ROOT / "outputs/localization_all_pci_clusters")]
        if args.skip_figure:
            cmd.append("--skip-figure")
        if args.stop_on_error:
            cmd.append("--stop-on-error")

    cmd += extra
    return run(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
