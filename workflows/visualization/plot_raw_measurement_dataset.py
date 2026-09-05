#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Publication figures for the raw Cellular-Pro measurement dataset.

All visible labels are English.  The default publication export is PNG at
1000 dpi with a 7.48-inch full-page width (7480 pixels).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIG_WIDTH_IN = 7.48
FIG_HEIGHT_IN = 5.61
DPI = 1000
EARTH_RADIUS_M = 6371008.8

NUMERIC_COLUMNS = [
    "LATITUDE", "LONGITUDE", "SPEED(M/s)", "ALT(M)", "ACCURACY(M)",
    "NR5G PCI", "NR5G SS RSRP", "NR5G SS RSRQ", "NR5G SS SINR",
    "NR5G Center ARFCN DL", "NR5G SSB ARFCN DL", "NR5G BandWidth DL", "NR5G Band",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot English publication figures for raw 5G NR measurement CSV files.")
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--dpi", type=int, default=DPI)
    p.add_argument("--top-pci", type=int, default=25)
    p.add_argument("--skip-session-figure", action="store_true")
    return p.parse_args()


def _looks_like_metadata_row(row: list[str]) -> bool:
    if not row:
        return False
    first = row[0].strip()
    if first not in {"0", "0.0", "0.00"}:
        return False
    first_twenty = row[:20]
    zeros = sum(x.strip() in {"0", "0.0", "0.00", ""} for x in first_twenty)
    hex_like = sum(bool(re.fullmatch(r"[0-9A-Fa-f]{7,10}", x.strip())) for x in row[4:100])
    first_four_zero = all(x.strip() in {"0", "0.0", "0.00", ""} for x in row[:4])
    return (zeros >= 15 and hex_like >= 10) or (first_four_zero and hex_like >= 10)


def read_cellular_pro_csv(path: Path) -> pd.DataFrame:
    """Read the Cellular-Pro CSV format used in this dataset.

    The source files contain one metadata/hash row after the header and can
    contain one more data field than the header.  This reader preserves every
    field, skips the metadata row, and prevents the first timestamp field from
    being accidentally promoted to the pandas index.
    """
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"Empty CSV: {path}")
    header = [str(x).strip() for x in rows[0]]
    body = rows[1:]
    if body and _looks_like_metadata_row(body[0]):
        body = body[1:]
    max_len = max([len(header)] + [len(r) for r in body])
    if max_len > len(header):
        header = header + [f"_EXTRA_{i+1:02d}" for i in range(max_len - len(header))]
    normalized: list[list[str]] = []
    for row in body:
        if not row:
            continue
        if len(row) < len(header):
            row = row + [""] * (len(header) - len(row))
        elif len(row) > len(header):
            row = row[: len(header)]
        normalized.append(row)
    df = pd.DataFrame(normalized, columns=header)
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "TIME" in df.columns:
        df["time_parsed"] = pd.to_datetime(df["TIME"], format="%Y%m%d %H:%M:%S.%f", errors="coerce")
    else:
        df["time_parsed"] = pd.NaT
    df["source_file"] = path.name
    return df


def load_dataset(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    files = sorted(input_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")
    frames: list[pd.DataFrame] = []
    session_rows: list[dict[str, object]] = []
    for index, path in enumerate(files, start=1):
        df = read_cellular_pro_csv(path)
        df["session_index"] = index
        df["session_label"] = f"S{index:02d}"
        frames.append(df)
        valid_geo = _valid_geo(df)
        valid_rsrp = _valid_rsrp(df)
        session_rows.append({
            "session_index": index,
            "session_label": f"S{index:02d}",
            "source_file": path.name,
            "rows": int(len(df)),
            "valid_gps_rows": int(valid_geo.sum()),
            "valid_rsrp_rows": int(valid_rsrp.sum()),
            "median_rsrp_dbm": float(df.loc[valid_rsrp, "NR5G SS RSRP"].median()) if valid_rsrp.any() else math.nan,
            "median_rsrq_db": float(pd.to_numeric(df.get("NR5G SS RSRQ"), errors="coerce").median()),
            "median_sinr_db": float(pd.to_numeric(df.get("NR5G SS SINR"), errors="coerce").median()),
            "median_speed_mps": float(pd.to_numeric(df.get("SPEED(M/s)"), errors="coerce").median()),
            "unique_serving_pci": int(pd.to_numeric(df.get("NR5G PCI"), errors="coerce").dropna().nunique()),
        })
    all_df = pd.concat(frames, ignore_index=True)
    sessions = pd.DataFrame(session_rows)
    return all_df, sessions


def _valid_geo(df: pd.DataFrame) -> pd.Series:
    lat = pd.to_numeric(df.get("LATITUDE"), errors="coerce")
    lon = pd.to_numeric(df.get("LONGITUDE"), errors="coerce")
    return lat.between(-90, 90) & lon.between(-180, 180) & lat.ne(0) & lon.ne(0)


def _valid_rsrp(df: pd.DataFrame) -> pd.Series:
    x = pd.to_numeric(df.get("NR5G SS RSRP"), errors="coerce")
    return x.between(-140, -40)


def _local_xy(lat: np.ndarray, lon: np.ndarray, lat0: float, lon0: float) -> tuple[np.ndarray, np.ndarray]:
    lat_r = np.deg2rad(lat)
    lon_r = np.deg2rad(lon)
    lat0_r = math.radians(lat0)
    lon0_r = math.radians(lon0)
    x = EARTH_RADIUS_M * math.cos(lat0_r) * (lon_r - lon0_r)
    y = EARTH_RADIUS_M * (lat_r - lat0_r)
    return x, y


def _style_axis(ax) -> None:
    ax.tick_params(labelsize=7.2, width=0.8)
    ax.xaxis.label.set_fontsize(8.2)
    ax.yaxis.label.set_fontsize(8.2)
    ax.title.set_fontsize(9.0)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def _save(fig, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.set_size_inches(FIG_WIDTH_IN, FIG_HEIGHT_IN, forward=True)
    for ax in fig.axes:
        _style_axis(ax)
    fig.savefig(path, dpi=int(dpi), format="png", bbox_inches=None, pad_inches=0.0, facecolor="white")
    plt.close(fig)


def plot_trajectory_rsrp(df: pd.DataFrame, path: Path, dpi: int) -> None:
    valid = _valid_geo(df) & _valid_rsrp(df)
    d = df.loc[valid].copy()
    if d.empty:
        raise ValueError("No rows contain both valid GPS coordinates and plausible RSRP.")
    lat0 = float(d["LATITUDE"].median())
    lon0 = float(d["LONGITUDE"].median())
    x, y = _local_xy(d["LATITUDE"].to_numpy(float), d["LONGITUDE"].to_numpy(float), lat0, lon0)
    d["local_x_m"] = x
    d["local_y_m"] = y

    fig = plt.figure(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN), dpi=100)
    ax = fig.add_axes([0.10, 0.13, 0.76, 0.78])
    cax = fig.add_axes([0.885, 0.13, 0.026, 0.78])
    for _, g in d.groupby("session_index", sort=True):
        ax.plot(g["local_x_m"], g["local_y_m"], linewidth=0.45, alpha=0.22)
    sc = ax.scatter(
        d["local_x_m"], d["local_y_m"], c=d["NR5G SS RSRP"],
        s=4.0, vmin=-120, vmax=-40, cmap="viridis", linewidths=0,
    )
    cb = fig.colorbar(sc, cax=cax)
    cb.set_label("Serving-cell SS-RSRP (dBm)", fontsize=8.0)
    cb.ax.tick_params(labelsize=7.0)
    ax.set_xlabel("Local Easting (m)")
    ax.set_ylabel("Local Northing (m)")
    ax.set_title("Measured 5G NR collection trajectory")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.15, linewidth=0.4)
    _save(fig, path, dpi)


def plot_metric_distributions(df: pd.DataFrame, path: Path, dpi: int) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN), dpi=100)
    specs = [
        ("NR5G SS RSRP", (-120, -40), "SS-RSRP (dBm)", "Serving-cell RSRP"),
        ("NR5G SS RSRQ", (-25, -5), "SS-RSRQ (dB)", "Serving-cell RSRQ"),
        ("NR5G SS SINR", (-25, 40), "SS-SINR (dB)", "Serving-cell SINR"),
    ]
    for ax, (col, lim, xlabel, title) in zip(axes, specs):
        x = pd.to_numeric(df.get(col), errors="coerce")
        x = x[np.isfinite(x) & x.between(lim[0], lim[1])]
        ax.hist(x, bins=45, edgecolor="black", linewidth=0.35, alpha=0.85)
        if len(x):
            ax.axvline(float(np.median(x)), linestyle="--", linewidth=1.0, label=f"Median = {np.median(x):.1f}")
            ax.legend(fontsize=7.0, frameon=True)
        ax.set_xlim(*lim)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Number of samples")
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.18, linewidth=0.4)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.15, top=0.90, wspace=0.30)
    _save(fig, path, dpi)


def plot_pci_frequency(df: pd.DataFrame, path: Path, dpi: int, top_pci: int) -> None:
    pci = pd.to_numeric(df.get("NR5G PCI"), errors="coerce").dropna().astype(int)
    counts = pci.value_counts().head(int(top_pci)).sort_values()
    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN), dpi=100)
    ax.barh([str(x) for x in counts.index], counts.values)
    ax.set_xlabel("Number of serving-cell observations")
    ax.set_ylabel("Serving PCI")
    ax.set_title(f"Most frequently observed serving PCIs (top {len(counts)})")
    ax.grid(True, axis="x", alpha=0.18, linewidth=0.4)
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.14, top=0.90)
    _save(fig, path, dpi)


def plot_session_summary(sessions: pd.DataFrame, path: Path, dpi: int) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN), dpi=100)
    x = np.arange(len(sessions))
    labels = sessions["session_label"].tolist()
    panels = [
        ("rows", "Samples", "Samples per collection session"),
        ("median_rsrp_dbm", "Median SS-RSRP (dBm)", "Median RSRP by session"),
        ("median_sinr_db", "Median SS-SINR (dB)", "Median SINR by session"),
        ("unique_serving_pci", "Unique serving PCIs", "Serving-PCI diversity by session"),
    ]
    for ax, (col, ylabel, title) in zip(axes.flat, panels):
        ax.bar(x, sessions[col].to_numpy())
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.18, linewidth=0.4)
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.17, top=0.91, hspace=0.38, wspace=0.28)
    _save(fig, path, dpi)


def write_summaries(df: pd.DataFrame, sessions: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    sessions.to_csv(output_dir / "raw_measurement_session_summary.csv", index=False, encoding="utf-8-sig")
    valid_geo = _valid_geo(df)
    valid_rsrp = _valid_rsrp(df)
    pci = pd.to_numeric(df.get("NR5G PCI"), errors="coerce")
    overall = {
        "source_csv_files": int(df["source_file"].nunique()),
        "rows_after_metadata_row_removal": int(len(df)),
        "valid_gps_rows": int(valid_geo.sum()),
        "valid_rsrp_rows": int(valid_rsrp.sum()),
        "unique_serving_pci": int(pci.dropna().nunique()),
        "latitude_min": float(df.loc[valid_geo, "LATITUDE"].min()),
        "latitude_max": float(df.loc[valid_geo, "LATITUDE"].max()),
        "longitude_min": float(df.loc[valid_geo, "LONGITUDE"].min()),
        "longitude_max": float(df.loc[valid_geo, "LONGITUDE"].max()),
        "rsrp_mean_dbm": float(df.loc[valid_rsrp, "NR5G SS RSRP"].mean()),
        "rsrp_median_dbm": float(df.loc[valid_rsrp, "NR5G SS RSRP"].median()),
        "rsrp_min_dbm": float(df.loc[valid_rsrp, "NR5G SS RSRP"].min()),
        "rsrp_max_dbm": float(df.loc[valid_rsrp, "NR5G SS RSRP"].max()),
    }
    (output_dir / "raw_measurement_overall_summary.json").write_text(json.dumps(overall, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    df, sessions = load_dataset(input_dir)
    write_summaries(df, sessions, output_dir)
    plot_trajectory_rsrp(df, output_dir / "01_measured_collection_trajectory_rsrp.png", args.dpi)
    plot_metric_distributions(df, output_dir / "02_radio_metric_distributions.png", args.dpi)
    plot_pci_frequency(df, output_dir / "03_serving_pci_observation_frequency.png", args.dpi, args.top_pci)
    if not args.skip_session_figure:
        plot_session_summary(sessions, output_dir / "04_collection_session_summary.png", args.dpi)
    print(f"[OK] Raw measurement figures written to: {output_dir}")
    print(f"[OK] CSV files: {df['source_file'].nunique()}, rows: {len(df)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
