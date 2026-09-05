from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.interpolate import RegularGridInterpolator


@dataclass(frozen=True)
class SimulationPrior:
    source_path: Path
    station_id: int
    pci: int
    x_axis_m: np.ndarray
    y_axis_m: np.ndarray
    rsrp_dbm: np.ndarray
    building_mask: np.ndarray | None
    selected_key: str
    metadata: dict[str, Any]

    def sample(self, query_xy: np.ndarray) -> np.ndarray:
        query_xy = np.asarray(query_xy, dtype=float)
        interpolator = RegularGridInterpolator(
            (self.y_axis_m, self.x_axis_m),
            self.rsrp_dbm,
            method="linear",
            bounds_error=False,
            fill_value=np.nan,
        )
        return np.asarray(interpolator(query_xy[:, [1, 0]]), dtype=float)

    def sample_nearest(self, query_xy: np.ndarray) -> np.ndarray:
        """Sample the fixed 1 m simulation map by nearest grid cell.

        Localization uses sparse receiver locations. Linear interpolation can turn
        an otherwise usable receiver location into NaN whenever one of the four
        surrounding Sionna cells is invalid. Nearest-grid sampling is therefore
        the appropriate collocation rule for the discrete radio-map channel and
        does not use any measurement value.
        """
        query_xy = np.asarray(query_xy, dtype=float)
        interpolator = RegularGridInterpolator(
            (self.y_axis_m, self.x_axis_m),
            self.rsrp_dbm,
            method="nearest",
            bounds_error=False,
            fill_value=np.nan,
        )
        return np.asarray(interpolator(query_xy[:, [1, 0]]), dtype=float)


def _scalar_int(data: Any, key: str) -> int | None:
    if key not in data.files:
        return None
    arr = np.asarray(data[key]).reshape(-1)
    if arr.size == 0:
        return None
    try:
        return int(arr[0])
    except Exception:
        return None


def _extract_axes(
    data: Any,
    map_shape: tuple[int, int],
    fallback_extent: tuple[float, float, float, float] | None,
) -> tuple[np.ndarray, np.ndarray]:
    ny, nx = map_shape
    if "x_m" in data.files and "y_m" in data.files:
        x = np.asarray(data["x_m"], dtype=float)
        y = np.asarray(data["y_m"], dtype=float)
        if x.ndim == 1 and y.ndim == 1:
            if len(x) == nx and len(y) == ny:
                return x.copy(), y.copy()
        if x.ndim == 2 and y.ndim == 2 and x.shape == map_shape and y.shape == map_shape:
            return x[0, :].copy(), y[:, 0].copy()

    if fallback_extent is None:
        raise ValueError(
            "仿真NPZ没有可识别的x_m/y_m坐标轴，且未提供地图extent。"
        )
    x_min, x_max, y_min, y_max = map(float, fallback_extent)
    x_axis = x_min + (np.arange(nx, dtype=float) + 0.5) * ((x_max - x_min) / nx)
    y_axis = y_min + (np.arange(ny, dtype=float) + 0.5) * ((y_max - y_min) / ny)
    return x_axis, y_axis


def _extract_map(data: Any, target_pci: int) -> tuple[np.ndarray, str]:
    # 优先单扇区纯仿真字段。
    preferred_2d = [
        "rsrp_dbm",
        "pure_sim_rsrp_dbm",
        "simulated_rsrp_dbm",
        "radio_map_rsrp_dbm",
    ]
    for key in preferred_2d:
        if key in data.files:
            arr = np.asarray(data[key], dtype=float)
            if arr.ndim == 2:
                return arr, key

    # 兼容联合NPZ：根据pcis选择对应扇区。
    candidate_3d = [
        "sector_rsrp_dbm",
        "pure_sector_rsrp_dbm",
        "sector_maps_dbm",
    ]
    for key in candidate_3d:
        if key not in data.files:
            continue
        arr = np.asarray(data[key], dtype=float)
        if arr.ndim != 3:
            continue
        if "pcis" in data.files:
            pcis = np.asarray(data["pcis"], dtype=int).reshape(-1)
            match = np.flatnonzero(pcis == int(target_pci))
            if len(match):
                return arr[int(match[0])], f"{key}[pci={target_pci}]"
        if arr.shape[0] == 1:
            return arr[0], f"{key}[0]"

    raise KeyError(
        "仿真NPZ中没有找到单PCI二维RSRP地图。支持字段："
        "rsrp_dbm、pure_sim_rsrp_dbm、simulated_rsrp_dbm、"
        "sector_rsrp_dbm+pcis。"
    )


def _normalize_axis_and_map(
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    rsrp_map: np.ndarray,
    building_mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    x_axis = np.asarray(x_axis, dtype=float).reshape(-1)
    y_axis = np.asarray(y_axis, dtype=float).reshape(-1)
    rsrp_map = np.asarray(rsrp_map, dtype=float)

    if rsrp_map.shape == (len(x_axis), len(y_axis)) and rsrp_map.shape != (len(y_axis), len(x_axis)):
        rsrp_map = rsrp_map.T
        if building_mask is not None and building_mask.shape == (len(x_axis), len(y_axis)):
            building_mask = building_mask.T

    expected = (len(y_axis), len(x_axis))
    if rsrp_map.shape != expected:
        raise ValueError(
            f"仿真地图shape={rsrp_map.shape}，坐标轴对应shape={expected}，无法对齐。"
        )

    if len(x_axis) > 1 and x_axis[1] < x_axis[0]:
        x_axis = x_axis[::-1]
        rsrp_map = rsrp_map[:, ::-1]
        if building_mask is not None:
            building_mask = building_mask[:, ::-1]
    if len(y_axis) > 1 and y_axis[1] < y_axis[0]:
        y_axis = y_axis[::-1]
        rsrp_map = rsrp_map[::-1, :]
        if building_mask is not None:
            building_mask = building_mask[::-1, :]

    return x_axis, y_axis, rsrp_map, building_mask


def load_simulation_prior(
    path: Path,
    station_id: int,
    pci: int,
    fallback_extent: tuple[float, float, float, float] | None = None,
) -> SimulationPrior:
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"找不到Sionna纯仿真NPZ：{path}")

    with np.load(path, allow_pickle=False) as data:
        file_station = _scalar_int(data, "station_id")
        file_pci = _scalar_int(data, "pci")
        if file_station is not None and file_station != int(station_id):
            raise ValueError(
                f"仿真NPZ station_id={file_station}，当前要求station_id={station_id}。"
            )
        if file_pci is not None and file_pci != int(pci):
            raise ValueError(f"仿真NPZ PCI={file_pci}，当前要求PCI={pci}。")

        rsrp_map, selected_key = _extract_map(data, target_pci=int(pci))
        building_mask = (
            np.asarray(data["building_mask"], dtype=bool)
            if "building_mask" in data.files else None
        )
        x_axis, y_axis = _extract_axes(data, rsrp_map.shape, fallback_extent)
        x_axis, y_axis, rsrp_map, building_mask = _normalize_axis_and_map(
            x_axis, y_axis, rsrp_map, building_mask
        )
        metadata = {
            "available_keys": list(data.files),
            "finite_cell_count": int(np.isfinite(rsrp_map).sum()),
            "total_cell_count": int(rsrp_map.size),
            "finite_fraction": float(np.isfinite(rsrp_map).mean()),
        }

    return SimulationPrior(
        source_path=path,
        station_id=int(station_id),
        pci=int(pci),
        x_axis_m=x_axis,
        y_axis_m=y_axis,
        rsrp_dbm=rsrp_map,
        building_mask=building_mask,
        selected_key=selected_key,
        metadata=metadata,
    )


def _candidate_score(path: Path, station_id: int, pci: int) -> tuple[int, float]:
    name = str(path).lower().replace("\\", "/")
    score = 0
    if f"station_{station_id:02d}" in name or f"station_{station_id}" in name:
        score += 30
    if f"pci_{pci}" in name or f"pci{pci}" in name:
        score += 30
    if "pure_simulation" in name or "pure_sim" in name:
        score += 25
    if "dem_following" in name or "dem_plus_1p5" in name:
        score += 20
    if "bestparam" in name:
        score += 10
    if "zplane" in name:
        score -= 8
    if "measurement_replaced" in name or "reconstructed" in name or "kriging" in name:
        score -= 50
    return score, path.stat().st_mtime


def discover_simulation_prior(
    project_root: Path,
    station_id: int,
    pci: int,
    explicit_path: Path | None = None,
) -> Path:
    if explicit_path is not None:
        explicit = Path(explicit_path).expanduser().resolve()
        if not explicit.exists():
            raise FileNotFoundError(f"--simulation-npz指定文件不存在：{explicit}")
        return explicit

    project_root = Path(project_root).expanduser().resolve()
    roots = [project_root / "outputs", project_root]
    candidates: list[Path] = []
    patterns = [
        f"**/*station_{station_id:02d}*pci_{pci}*pure*.npz",
        f"**/*station_{station_id:02d}*pci{pci}*pure*.npz",
        f"**/*station_{station_id}*pci_{pci}*pure*.npz",
    ]
    for root in roots:
        if not root.exists():
            continue
        for pattern in patterns:
            candidates.extend(root.glob(pattern))

    unique = sorted({path.resolve() for path in candidates if path.is_file()})
    viable = []
    for path in unique:
        score, mtime = _candidate_score(path, station_id, pci)
        if score > 0:
            viable.append((score, mtime, path))
    if not viable:
        raise FileNotFoundError(
            "未自动找到该站/PCI的纯Sionna仿真NPZ。请使用 --simulation-npz 显式指定。\n"
            "优先使用DEM+1.5m纯仿真单PCI文件，例如：\n"
            f"outputs/.../station_{station_id:02d}/.../station_{station_id:02d}_pci_{pci}_pure_simulation.npz"
        )
    viable.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return viable[0][2]
