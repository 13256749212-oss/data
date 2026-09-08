#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Recompute PN-MCVL-RC-v1.15 progressive trajectory fusion from an existing all-trials CSV.

This utility does not rerun localization.  It is intended for quickly checking the
new progressive trajectory fusion on previously completed Monte-Carlo outputs.
The full v1.15 experiment should still be rerun to obtain the new nested 10--15
point selections and previous-count candidate inheritance.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from run_27stations_localization_multicount import build_progressive_trajectory_ensemble


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Recompute v1.15 progressive trajectory ensemble from all-trials CSV")
    p.add_argument("--input-csv", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    input_csv = args.input_csv.expanduser().resolve()
    if not input_csv.is_file():
        raise FileNotFoundError(input_csv)
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else input_csv.parent / "pn_mcvl_rc_v115_progressive_recomputed"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(input_csv, encoding="utf-8-sig")
    states, station, summary = build_progressive_trajectory_ensemble(frame)
    states.to_csv(output_dir / "localization_progressive_trajectory_trial_states.csv", index=False, encoding="utf-8-sig")
    station.to_csv(output_dir / "localization_progressive_trajectory_station_results.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output_dir / "localization_progressive_trajectory_summary.csv", index=False, encoding="utf-8-sig")
    if len(summary):
        best_idx = pd.to_numeric(summary["rmse_m"], errors="coerce").idxmin()
        summary.loc[[best_idx]].to_csv(output_dir / "localization_progressive_best_point_count.csv", index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))
    print("Output:", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
