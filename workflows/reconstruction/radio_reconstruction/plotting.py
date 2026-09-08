from __future__ import annotations

from pathlib import Path
import hashlib
import os
from typing import Sequence

import numpy as np
from matplotlib.collections import LineCollection


def _display_map(rsrp: np.ndarray, min_dbm: float, max_dbm: float) -> np.ndarray:
    result = np.asarray(rsrp, dtype=float).copy()
    result[~np.isfinite(result)] = float(min_dbm)
    return np.clip(result, float(min_dbm), float(max_dbm))


def _create_fixed_square_map_axes(
    fig,
    *,
    extent: Sequence[float],
    left: float = 0.105,
    bottom: float = 0.105,
    top: float = 0.835,
    right_margin: float = 0.155,
    cbar_pad: float = 0.020,
    cbar_width: float = 0.032,
):
    """Create a fixed 512 m × 512 m map axes and an equal-height colorbar axes."""
    x_min, x_max, y_min, y_max = map(float, extent)
    data_ratio = abs((y_max - y_min) / (x_max - x_min))
    fig_w, fig_h = fig.get_size_inches()
    avail_w = 1.0 - float(left) - float(right_margin) - float(cbar_pad) - float(cbar_width)
    avail_h = float(top) - float(bottom)
    normalized_h_if_full_w = avail_w * (fig_w / fig_h) * data_ratio
    if normalized_h_if_full_w <= avail_h:
        ax_w = avail_w
        ax_h = normalized_h_if_full_w
        ax_left = float(left)
        ax_bottom = float(bottom) + 0.5 * (avail_h - ax_h)
    else:
        ax_h = avail_h
        ax_w = avail_h / ((fig_w / fig_h) * data_ratio)
        ax_left = float(left) + 0.5 * (avail_w - ax_w)
        ax_bottom = float(bottom)
    cax_left = ax_left + ax_w + float(cbar_pad)
    ax = fig.add_axes([ax_left, ax_bottom, ax_w, ax_h])
    cax = fig.add_axes([cax_left, ax_bottom, float(cbar_width), ax_h])
    return ax, cax


def _add_aligned_colorbar(fig, cax, mappable, label: str):
    cbar = fig.colorbar(mappable, cax=cax)
    cbar.set_label(label)
    return cbar


def _lock_extent_512(ax, extent: Sequence[float]) -> None:
    """Lock plot limits to the 512 m × 512 m reconstruction domain."""
    x_min, x_max, y_min, y_max = map(float, extent)
    width = x_max - x_min
    height = y_max - y_min
    if not (np.isclose(width, 512.0, atol=1e-6) and np.isclose(height, 512.0, atol=1e-6)):
        raise ValueError(
            f"重构绘图范围必须严格为512 m × 512 m，当前为{width:.6f} m × {height:.6f} m。"
        )
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")


FULL_PAGE_WIDTH_IN = 7.48  # 7480 px at 1000 dpi
FIGURE_HEIGHT_IN = 5.61
MAP_DPI = 1000
COMPARISON_DPI = 1000


def _publication_figsize_inches() -> tuple[float, float]:
    return (FULL_PAGE_WIDTH_IN, FIGURE_HEIGHT_IN)


def _style_publication_text(fig) -> None:
    for _ax in fig.axes:
        try:
            _ax.title.set_fontsize(10.0)
            _ax.xaxis.label.set_fontsize(9.0)
            _ax.yaxis.label.set_fontsize(9.0)
            _ax.tick_params(axis="both", which="major", labelsize=8.2, pad=2.0)
            _legend = _ax.get_legend()
            if _legend is not None:
                for _txt in _legend.get_texts():
                    _txt.set_fontsize(8.0)
        except Exception:
            pass


def _save_png(fig, output_path: Path, dpi: int = MAP_DPI, facecolor: str | None = None) -> Path:
    """Save one publication PNG."""
    base = Path(output_path).with_suffix("")
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.set_size_inches(*_publication_figsize_inches(), forward=True)
    _style_publication_text(fig)
    fc = facecolor if facecolor is not None else fig.get_facecolor()
    out = base.with_suffix(".png")
    fig.savefig(out, format="png", dpi=int(dpi), bbox_inches=None, pad_inches=0.0, facecolor=fc)
    return out


def add_building_outlines(ax, outline_segments: Sequence[np.ndarray] | None) -> None:
    """Draw precomputed building-outline line segments on a radio-map axes."""
    if outline_segments is None:
        return
    segments = [np.asarray(seg, dtype=float) for seg in outline_segments if np.asarray(seg).size >= 4]
    if not segments:
        return
    ax.add_collection(
        LineCollection(
            segments,
            colors=[(0.96, 0.96, 0.96, 0.95)],
            linewidths=0.26,
            alpha=0.95,
            zorder=5,
        )
    )


def add_building_blocks(ax, building_mask: np.ndarray | None, extent: Sequence[float]) -> None:
    """Render buildings as solid white blocks for reconstruction figures."""
    if building_mask is None:
        return
    mask = np.asarray(building_mask, dtype=bool)
    if mask.size == 0 or not np.any(mask):
        return
    overlay = np.zeros(mask.shape + (4,), dtype=float)
    overlay[mask] = (1.0, 1.0, 1.0, 1.0)
    ax.imshow(overlay, origin="lower", extent=extent, interpolation="nearest", zorder=5)


RECONSTRUCTION_FIGURE_WIDTH_IN = 7.48
RECONSTRUCTION_FIGURE_HEIGHT_IN = 7.48


def _save_reconstruction_png(
    fig,
    output_path: Path,
    facecolor: str | None = None,
) -> Path:
    """Save a readable square reconstruction figure at 1000 dpi."""
    base = Path(output_path).with_suffix("")
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.set_size_inches(RECONSTRUCTION_FIGURE_WIDTH_IN, RECONSTRUCTION_FIGURE_HEIGHT_IN, forward=True)
    fc = facecolor if facecolor is not None else fig.get_facecolor()
    out = base.with_suffix(".png")
    # Windows commonly enforces the legacy MAX_PATH limit unless long paths are
    # enabled system-wide.  Keep a safety margin so publication PNG export does
    # not fail merely because an experiment root or method label is long.
    if os.name == "nt":
        try:
            absolute_text = str(out.absolute())
        except Exception:
            absolute_text = str(out)
        if len(absolute_text) >= 240:
            digest = hashlib.sha1(out.stem.encode("utf-8")).hexdigest()[:10]
            short_stem = out.stem[:64].rstrip("_- ")
            out = out.with_name(f"{short_stem}_{digest}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        out,
        format="png",
        dpi=MAP_DPI,
        bbox_inches=None,
        pad_inches=0.0,
        facecolor=fc,
    )
    return out


def plot_clean_map(
    output_path: Path,
    rsrp_map: np.ndarray,
    extent: Sequence[float],
    outline_segments: Sequence[np.ndarray] | None,
    tx_xy: tuple[float, float],
    method_label: str,
    station_id: int,
    pci: int,
    rmse_db: float,
    training_count: int,
    min_dbm: float = -120.0,
    max_dbm: float = -40.0,
    dpi: int = 1000,
    building_mask: np.ndarray | None = None,
) -> None:
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(
        figsize=(RECONSTRUCTION_FIGURE_WIDTH_IN, RECONSTRUCTION_FIGURE_HEIGHT_IN),
        dpi=MAP_DPI,
        facecolor="white",
    )
    ax, cax = _create_fixed_square_map_axes(fig, extent=extent)
    image = ax.imshow(
        _display_map(rsrp_map, min_dbm, max_dbm),
        origin="lower",
        extent=extent,
        cmap="viridis",
        vmin=min_dbm,
        vmax=max_dbm,
        interpolation="nearest",
        zorder=1,
    )
    if building_mask is not None:
        add_building_blocks(ax, building_mask, extent)
    else:
        add_building_outlines(ax, outline_segments)
    ax.scatter(
        [tx_xy[0]], [tx_xy[1]], marker="^", s=48,
        c="red", edgecolors="white", linewidths=0.6, zorder=8,
    )
    ax.set_title(
        f"Station {station_id} / PCI {pci} - {method_label}\n"
        f"Training points: {training_count} | Validation RMSE: {rmse_db:.2f} dB",
        fontsize=10.5,
        pad=8.0,
    )
    ax.set_xlabel("Blender X (m)", fontsize=9.5)
    ax.set_ylabel("Blender Y (m)", fontsize=9.5)
    ax.tick_params(axis="both", which="major", labelsize=8.5, pad=3.0)
    _lock_extent_512(ax, extent)
    cbar = _add_aligned_colorbar(fig, cax, image, "Reconstructed SS-RSRP (dBm)")
    cbar.ax.tick_params(labelsize=8.5)
    cbar.set_label("Reconstructed SS-RSRP (dBm)", fontsize=10.0)
    _save_reconstruction_png(fig, output_path)
    plt.close(fig)


def plot_map_with_training_points(
    output_path: Path,
    rsrp_map: np.ndarray,
    extent: Sequence[float],
    outline_segments: Sequence[np.ndarray] | None,
    tx_xy: tuple[float, float],
    training_xy: np.ndarray,
    method_label: str,
    station_id: int,
    pci: int,
    training_count: int,
    min_dbm: float = -120.0,
    max_dbm: float = -40.0,
    dpi: int = 1000,
    building_mask: np.ndarray | None = None,
) -> None:
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(
        figsize=(RECONSTRUCTION_FIGURE_WIDTH_IN, RECONSTRUCTION_FIGURE_HEIGHT_IN),
        dpi=MAP_DPI,
        facecolor="white",
    )
    ax, cax = _create_fixed_square_map_axes(fig, extent=extent)
    image = ax.imshow(
        _display_map(rsrp_map, min_dbm, max_dbm),
        origin="lower",
        extent=extent,
        cmap="viridis",
        vmin=min_dbm,
        vmax=max_dbm,
        interpolation="nearest",
        zorder=1,
    )
    if building_mask is not None:
        add_building_blocks(ax, building_mask, extent)
    else:
        add_building_outlines(ax, outline_segments)
    ax.scatter(
        training_xy[:, 0], training_xy[:, 1],
        s=36, facecolors="white", edgecolors="black", linewidths=0.75,
        label=f"{training_count} reconstruction points", zorder=8,
    )
    ax.scatter(
        [tx_xy[0]], [tx_xy[1]], marker="^", s=48,
        c="red", edgecolors="white", linewidths=0.6, label="TX", zorder=9,
    )
    ax.set_title(
        f"Station {station_id} / PCI {pci} - {method_label}\n"
        f"Training points: {training_count}",
        fontsize=10.5,
        pad=8.0,
    )
    ax.set_xlabel("Blender X (m)", fontsize=10.0)
    ax.set_ylabel("Blender Y (m)", fontsize=10.0)
    ax.tick_params(axis="both", which="major", labelsize=8.5, pad=3.0)
    _lock_extent_512(ax, extent)
    ax.legend(loc="upper right", fontsize=9.0, framealpha=0.96, borderpad=0.45)
    cbar = _add_aligned_colorbar(fig, cax, image, "Reconstructed SS-RSRP (dBm)")
    cbar.ax.tick_params(labelsize=8.5)
    cbar.set_label("Reconstructed SS-RSRP (dBm)", fontsize=10.0)
    _save_reconstruction_png(fig, output_path)
    plt.close(fig)


def plot_validation_scatter(
    output_path: Path,
    validation_frame,
    methods: Sequence[tuple[str, str]],
    dpi: int = 1000,
) -> None:
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_methods = len(methods)
    fig, axes = plt.subplots(1, n_methods, figsize=_publication_figsize_inches(), dpi=MAP_DPI, sharex=True, sharey=True)
    if n_methods == 1:
        axes = [axes]
    measured = validation_frame["measured_rsrp_dbm"].to_numpy(dtype=float)
    limits = [
        float(np.nanmin(measured)) - 3.0,
        float(np.nanmax(measured)) + 3.0,
    ]
    for ax, (column, label) in zip(axes, methods):
        predicted = validation_frame[f"predicted_{column}_dbm"].to_numpy(dtype=float)
        ax.scatter(measured, predicted, s=10, alpha=1.0, rasterized=True)
        ax.plot(limits, limits, linestyle="--", linewidth=1.0, color="black")
        error = predicted - measured
        rmse = float(np.sqrt(np.nanmean(error**2)))
        ax.set_title(f"{label}\nRMSE={rmse:.2f} dB")
        ax.set_xlabel("Measured SS-RSRP (dBm)")
        ax.grid(True, color="0.88", alpha=1.0)
        ax.set_xlim(limits)
        ax.set_ylim(limits)
    axes[0].set_ylabel("Predicted SS-RSRP (dBm)")
    fig.tight_layout()
    _save_png(fig, output_path, dpi=COMPARISON_DPI)
    plt.close(fig)
