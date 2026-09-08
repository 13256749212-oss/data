#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Recompute MCVL-RC-v1.14 robust 10-trial consensus from an existing all-trials CSV.

This utility does not rerun any single-trial localization. It only fuses the
already-computed randomized trial positions using the same truth-free spatial
consensus used by the v1.14 localize-sweep workflow.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from run_27stations_localization_multicount import (
    build_robust_consensus,
    make_ensemble_comparison,
    plot_robust_consensus_locations,
    plot_robust_consensus_summary,
    summarize_mean_prediction_ensemble,
    summarize_robust_consensus,
    summarize_station_trials,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="从现有10次随机定位结果直接计算MCVL-RC-v1.14鲁棒共识")
    p.add_argument("--input-csv", type=Path, required=True, help="localization_random10_all_trials_station_results.csv")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--consensus-mad-z", type=float, default=2.0)
    p.add_argument("--consensus-min-inlier-fraction", type=float, default=0.60)
    p.add_argument("--consensus-min-scale-m", type=float, default=5.0)
    p.add_argument("--dpi", type=int, default=1000)
    p.add_argument("--skip-figures", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    input_csv = args.input_csv.expanduser().resolve()
    if not input_csv.exists():
        raise FileNotFoundError(input_csv)
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else input_csv.parent / "mcvl_rc_v114_consensus_recomputed"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = pd.read_csv(input_csv, encoding="utf-8-sig")
    required = {"point_count", "station_id", "predicted_x_m", "predicted_y_m"}
    missing = required - set(all_results.columns)
    if missing:
        raise ValueError(f"输入CSV缺少必要列: {sorted(missing)}")

    robust = build_robust_consensus(
        all_results,
        mad_z=float(args.consensus_mad_z),
        min_inlier_fraction=float(args.consensus_min_inlier_fraction),
        min_scale_m=float(args.consensus_min_scale_m),
    )
    robust_summary = summarize_robust_consensus(robust)

    arithmetic_station = summarize_station_trials(all_results)
    arithmetic_summary = summarize_mean_prediction_ensemble(arithmetic_station)
    comparison = make_ensemble_comparison(arithmetic_summary, robust_summary)

    robust.to_csv(output_dir / "localization_random10_robust_consensus_station_results.csv", index=False, encoding="utf-8-sig")
    robust_summary.to_csv(output_dir / "localization_random10_robust_consensus_summary.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(output_dir / "localization_random10_ensemble_comparison.csv", index=False, encoding="utf-8-sig")
    if len(robust_summary):
        best_idx = pd.to_numeric(robust_summary["rmse_m"], errors="coerce").idxmin()
        robust_summary.loc[[best_idx]].to_csv(output_dir / "localization_random10_best_point_count_by_robust_rmse.csv", index=False, encoding="utf-8-sig")

    if not args.skip_figures:
        plot_robust_consensus_summary(robust_summary, output_dir, int(args.dpi))
        plot_robust_consensus_locations(robust, all_results, output_dir, int(args.dpi))

    print("MCVL-RC-v1.14 robust consensus completed")
    print("Input:", input_csv)
    print("Output:", output_dir)
    print(robust_summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
