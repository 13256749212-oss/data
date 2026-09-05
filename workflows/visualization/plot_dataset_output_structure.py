#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate an English dataset-output content structure diagram."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

FIG_WIDTH_IN = 7.48
FIG_HEIGHT_IN = 6.60
DPI = 1000


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot the dataset output-product structure as a publication PNG.")
    p.add_argument("--project-root", type=Path, required=True)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--dpi", type=int, default=DPI)
    return p.parse_args()


def _box(ax, x: float, y: float, w: float, h: float, title: str, lines: list[str], title_size: float = 8.0) -> None:
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.012", facecolor="white", edgecolor="black", linewidth=1.0)
    ax.add_patch(patch)
    ax.text(x + 0.018, y + h - 0.032, title, ha="left", va="top", fontsize=title_size, fontweight="bold")
    body = "\n".join(lines)
    ax.text(x + 0.018, y + h - 0.078, body, ha="left", va="top", fontsize=6.5, linespacing=1.24)


def _connector(ax, x0: float, y0: float, x1: float, y1: float) -> None:
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-", linewidth=0.9, color="black"))


def plot_structure(project_root: Path, output: Path, dpi: int) -> None:
    fig = plt.figure(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN), dpi=100)
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.96])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    root_x, root_y, root_w, root_h = 0.31, 0.895, 0.38, 0.075
    _box(ax, root_x, root_y, root_w, root_h, "Dataset output products (outputs/)", [])

    left_x, right_x = 0.035, 0.535
    box_w, box_h = 0.43, 0.155
    ys = [0.69, 0.505, 0.320, 0.135]

    nodes = [
        (left_x, ys[0], "Parameter calibration", ["parameter_calibration/", "all_27stations_summary.csv", "per-site best_parameters.json"]),
        (right_x, ys[0], "Single-site radio maps", ["bestparam_radio_maps_512m/", "DEM + 1.5 m", "per-PCI and best-server maps"]),
        (left_x, ys[1], "Receiver-surface comparison", ["bestparam_dem_vs_zplane_512m/", "DEM-following maps", "fixed-Z interpolation maps"]),
        (right_x, ys[1], "Campus-scale joint radio map", ["joint_best_server_4000x3000/", "best RSRP / station / PCI", "coverage statistics and NPZ"]),
        (left_x, ys[2], "Measurement-based validation", ["joint_map_measurement_comparison/", "RMSE / MAE / bias / correlation", "matched trajectory tables"]),
        (right_x, ys[2], "Radio-map reconstruction", ["radio_map_reconstruction_10points_pgsrf/", "10-point reconstruction maps", "baseline and validation metrics"]),
        (left_x, ys[3], "Five-point base-station localization", ["localization_27stations_5points_dppgrsl/", "per-site estimates", "accuracy summary and figures"]),
        (right_x, ys[3], "All-PCI cluster localization", ["localization_all_pci_clusters/", "per-PCI clusters and centers", "site estimates and accuracy summary"]),
    ]

    root_center = (root_x + root_w / 2, root_y)
    trunk_y = 0.855
    _connector(ax, root_center[0], root_center[1], root_center[0], trunk_y)
    ax.plot([0.25, 0.75], [trunk_y, trunk_y], linewidth=0.9, color="black")

    for x, y, title, lines in nodes:
        center_x = x + box_w / 2
        top_y = y + box_h
        ax.plot([center_x, center_x], [top_y, trunk_y if y == ys[0] else top_y + 0.018], linewidth=0.7, color="black")
        _box(ax, x, y, box_w, box_h, title, lines)

    # Vertical family connectors make the two columns read as dataset-product streams.
    for x in (left_x + box_w / 2, right_x + box_w / 2):
        for upper, lower in zip(ys[:-1], ys[1:]):
            ax.plot([x, x], [lower + box_h, upper], linewidth=0.65, color="black")

    ax.text(0.50, 0.055, "Campus 5G radio-map dataset: derived data products and application outputs", ha="center", va="center", fontsize=6.5)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=int(dpi), format="png", bbox_inches=None, pad_inches=0.0, facecolor="white")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    project_root = args.project_root.expanduser().resolve()
    output = args.output.expanduser().resolve() if args.output else project_root / "outputs" / "dataset_visualization" / "05_dataset_output_structure.png"
    plot_structure(project_root, output, args.dpi)
    print(f"[OK] Dataset output structure figure: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
