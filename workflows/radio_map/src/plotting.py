from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .configuration import StationConfig
from .terrain import SurfaceInfo


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
            _ax.tick_params(axis='both', which='major', labelsize=8.2, pad=2.0)
            _legend = _ax.get_legend()
            if _legend is not None:
                for _txt in _legend.get_texts():
                    _txt.set_fontsize(8.0)
        except Exception:
            pass


def _save_png(fig, output_path: Path, dpi: int = MAP_DPI, facecolor: str | None = None) -> Path:
    """Save one publication PNG.

    Full-page canvas width is fixed to 7.48 in. At 1000 dpi this gives 7480 px. No PNG duplicate is produced.
    """
    base = Path(output_path).with_suffix('')
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.set_size_inches(*_publication_figsize_inches(), forward=True)
    _style_publication_text(fig)
    fc = facecolor if facecolor is not None else fig.get_facecolor()
    out = base.with_suffix('.png')
    fig.savefig(out, format='png', dpi=int(dpi), bbox_inches=None, pad_inches=0.0, facecolor=fc)
    return out


def save_final_npz(
    path: Path,
    station: StationConfig,
    surface: SurfaceInfo,
    sector_rsrp_dbm: np.ndarray,
    best: Dict[str, Any],
) -> None:
    if surface.nx is None or surface.ny is None:
        raise ValueError("最终地图必须使用dense surface")
    ny, nx = int(surface.ny), int(surface.nx)
    sector_maps = sector_rsrp_dbm.reshape(3, ny, nx)
    finite_any = np.any(np.isfinite(sector_maps), axis=0)
    best_index = np.argmax(
        np.where(np.isfinite(sector_maps), sector_maps, -np.inf), axis=0
    )
    best_rsrp = np.take_along_axis(sector_maps, best_index[None, ...], axis=0)[0]
    best_rsrp = np.where(finite_any, best_rsrp, np.nan)
    pci_values = np.asarray(station.pcis, dtype=np.int32)
    best_pci = np.where(finite_any, pci_values[best_index], -1).astype(np.int32)
    np.savez_compressed(
        path,
        station_id=np.asarray([station.station_id], dtype=np.int32),
        pcis=pci_values,
        x_m=surface.cell_center_x.reshape(ny, nx),
        y_m=surface.cell_center_y.reshape(ny, nx),
        ground_z_m=surface.cell_ground_z.reshape(ny, nx),
        receiver_z_m=surface.cell_rx_z.reshape(ny, nx),
        sector_rsrp_dbm=sector_maps,
        best_server_rsrp_dbm=best_rsrp,
        best_server_pci=best_pci,
        height_agl_m=np.asarray([best["height_agl_m"]], dtype=np.float32),
        tx_absolute_z_m=np.asarray([best["tx_absolute_z_m"]], dtype=np.float32),
        shared_power_dbm=np.asarray([best["shared_power_dbm"]], dtype=np.float32),
        alphas_rad=np.asarray(best["alphas_rad"], dtype=np.float64),
        beta_rad=np.asarray([best["beta_rad"]], dtype=np.float64),
    )


def _extent(surface: SurfaceInfo) -> list[float]:
    x = surface.cell_center_x
    y = surface.cell_center_y
    if surface.nx is None or surface.ny is None:
        raise ValueError("dense surface required")
    dx = (x.max() - x.min()) / max(surface.nx - 1, 1)
    dy = (y.max() - y.min()) / max(surface.ny - 1, 1)
    return [x.min() - dx / 2, x.max() + dx / 2, y.min() - dy / 2, y.max() + dy / 2]

def _create_fixed_map_axes(fig, extent: list[float], left: float = 0.100, bottom: float = 0.180, top: float = 0.820, right_margin: float = 0.110, cbar_pad: float = 0.012, cbar_width: float = 0.022):
    x0, x1, y0, y1 = map(float, extent)
    data_ratio = abs((y1 - y0) / max(x1 - x0, 1e-9))
    fig_w, fig_h = fig.get_size_inches()
    avail_w = 1.0 - left - right_margin - cbar_pad - cbar_width
    avail_h = top - bottom
    normalized_h_if_full_w = avail_w * (fig_w / fig_h) * data_ratio
    if normalized_h_if_full_w <= avail_h:
        ax_w = avail_w
        ax_h = normalized_h_if_full_w
        ax_left = left
        ax_bottom = bottom + 0.5 * (avail_h - ax_h)
    else:
        ax_h = avail_h
        ax_w = avail_h / ((fig_w / fig_h) * data_ratio)
        ax_left = left + 0.5 * (avail_w - ax_w)
        ax_bottom = bottom
    cax_left = ax_left + ax_w + cbar_pad
    ax = fig.add_axes([ax_left, ax_bottom, ax_w, ax_h])
    cax = fig.add_axes([cax_left, ax_bottom, cbar_width, ax_h])
    return ax, cax


def _add_fixed_colorbar(fig, cax, mappable, label: str):
    cbar = fig.colorbar(mappable, cax=cax)
    cbar.set_label(label)
    return cbar



def plot_final_maps(
    output_dir: Path,
    station: StationConfig,
    surface: SurfaceInfo,
    sector_rsrp_dbm: np.ndarray,
    measurements: pd.DataFrame,
    best: Dict[str, Any],
    display_min_dbm: float,
    display_max_dbm: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if surface.nx is None or surface.ny is None:
        raise ValueError("最终绘图需要dense surface")
    ny, nx = int(surface.ny), int(surface.nx)
    maps = sector_rsrp_dbm.reshape(3, ny, nx)
    extent = _extent(surface)

    for idx, pci in enumerate(station.pcis):
        fig = plt.figure(figsize=_publication_figsize_inches(), dpi=MAP_DPI)
        ax, cax = _create_fixed_map_axes(fig, extent)
        image = ax.imshow(
            maps[idx],
            origin="lower",
            extent=extent,
            vmin=display_min_dbm,
            vmax=display_max_dbm,
            cmap="viridis",
            interpolation="nearest",
        )
        subset = measurements.loc[measurements["pci"].eq(pci)]
        if not subset.empty:
            ax.scatter(
                subset["cell_x_m"], subset["cell_y_m"],
                s=7, facecolors="none", edgecolors="white", linewidths=0.45,
                label=f"Measured PCI {pci}",
            )
        ax.scatter([station.x_m], [station.y_m], marker="^", s=70, c="red", label="TX")
        ax.set_title(
            f"Station {station.station_id} / PCI {pci}\n"
            f"h={best['height_agl_m']:.0f} m, P={best['shared_power_dbm']:.2f} dBm, "
            f"RMSE={best['pooled_equal_pci_rmse_db']:.2f} dB"
        )
        ax.set_xlabel("Blender X (m)")
        ax.set_ylabel("Blender Y (m)")
        ax.set_aspect("equal", adjustable="box")
        ax.legend(loc="upper right", framealpha=1.0)
        _add_fixed_colorbar(fig, cax, image, "Simulated SS-RSRP approximation (dBm)")
        _save_png(fig, output_dir / f"station_{station.station_id}_pci_{pci}_rsrp.png")
        plt.close(fig)

    finite_any = np.any(np.isfinite(maps), axis=0)
    best_map = np.max(np.where(np.isfinite(maps), maps, -np.inf), axis=0)
    best_map = np.where(finite_any, best_map, np.nan)
    fig = plt.figure(figsize=_publication_figsize_inches(), dpi=MAP_DPI)
    ax, cax = _create_fixed_map_axes(fig, extent)
    image = ax.imshow(
        best_map,
        origin="lower",
        extent=extent,
        vmin=display_min_dbm,
        vmax=display_max_dbm,
        cmap="viridis",
        interpolation="nearest",
    )
    ax.scatter([station.x_m], [station.y_m], marker="^", s=70, c="red")
    ax.set_title(f"Station {station.station_id}: 3-sector best-server RSRP")
    ax.set_xlabel("Blender X (m)")
    ax.set_ylabel("Blender Y (m)")
    ax.set_aspect("equal", adjustable="box")
    _add_fixed_colorbar(fig, cax, image, "Best-sector SS-RSRP approximation (dBm)")
    _save_png(fig, output_dir / f"station_{station.station_id}_best_server_rsrp.png")
    plt.close(fig)


def plot_comparison(output_path: Path, comparison: pd.DataFrame, station_id: int) -> None:
    fig, ax = plt.subplots(figsize=_publication_figsize_inches(), dpi=MAP_DPI)
    for pci, part in comparison.groupby("pci"):
        ax.scatter(
            part["measured_rsrp_dbm"], part["simulated_rsrp_dbm"],
            s=12, alpha=0.65, label=f"PCI {int(pci)}",
        )
    values = np.r_[comparison["measured_rsrp_dbm"], comparison["simulated_rsrp_dbm"]]
    lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
    ax.set_xlabel("Measured SS-RSRP (dBm)")
    ax.set_ylabel("Simulated SS-RSRP approximation (dBm)")
    ax.set_title(f"Station {station_id}: measured vs simulated")
    ax.grid(True, linewidth=0.4, alpha=0.5)
    ax.legend()
    _save_png(fig, output_path, dpi=COMPARISON_DPI)
    plt.close(fig)
