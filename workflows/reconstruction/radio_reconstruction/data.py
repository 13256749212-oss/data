from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "rx_point_id",
    "blender_x",
    "blender_y",
    "pci",
    "measured_rsrp_dbm",
    "station_id",
    "tx_x_initial_m",
    "tx_y_initial_m",
}


@dataclass(frozen=True)
class ReconstructionDataset:
    station_id: int
    pci: int
    station_label: str
    tx_x_m: float
    tx_y_m: float
    x_min_m: float
    x_max_m: float
    y_min_m: float
    y_max_m: float
    aggregated: pd.DataFrame
    training: pd.DataFrame
    validation: pd.DataFrame


def read_processed_long_table(path: Path) -> pd.DataFrame:
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"找不到处理后实测长表：{path}")

    frame: pd.DataFrame | None = None
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            frame = pd.read_csv(path, encoding=encoding, low_memory=False)
            break
        except UnicodeDecodeError as exc:
            last_error = exc
    if frame is None:
        raise UnicodeError(f"CSV编码读取失败：{last_error}")

    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise KeyError(
            "输入文件不是预期的27站处理后长表。\n"
            f"缺少字段：{missing}\n"
            f"实际字段：{list(frame.columns)}"
        )

    numeric = [
        "blender_x",
        "blender_y",
        "pci",
        "measured_rsrp_dbm",
        "station_id",
        "tx_x_initial_m",
        "tx_y_initial_m",
    ]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    finite = np.ones(len(frame), dtype=bool)
    for column in numeric:
        finite &= np.isfinite(frame[column].to_numpy(dtype=float))
    frame = frame.loc[finite].copy()

    if "rsrp_unit" in frame.columns:
        units = frame["rsrp_unit"].astype(str).str.strip().str.lower()
        bad = ~units.eq("dbm")
        if bad.any():
            raise ValueError(
                "检测到非dBm的RSRP单位："
                f"{frame.loc[bad, 'rsrp_unit'].value_counts(dropna=False).to_dict()}"
            )

    frame["station_id"] = frame["station_id"].astype(int)
    frame["pci"] = frame["pci"].astype(int)
    return frame


def aggregate_one_meter_cells(
    frame: pd.DataFrame,
    station_id: int,
    pci: int,
    map_size_m: int,
    min_rsrp_dbm: float,
    max_rsrp_dbm: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    selected = frame.loc[
        frame["station_id"].eq(int(station_id))
        & frame["pci"].eq(int(pci))
    ].copy()
    if selected.empty:
        raise ValueError(f"station_id={station_id}, PCI={pci}没有实测记录")

    tx_x = float(selected["tx_x_initial_m"].median())
    tx_y = float(selected["tx_y_initial_m"].median())
    half = float(map_size_m) / 2.0
    x_min, x_max = tx_x - half, tx_x + half
    y_min, y_max = tx_y - half, tx_y + half

    selected = selected.loc[
        selected["blender_x"].ge(x_min)
        & selected["blender_x"].lt(x_max)
        & selected["blender_y"].ge(y_min)
        & selected["blender_y"].lt(y_max)
        & selected["measured_rsrp_dbm"].between(
            float(min_rsrp_dbm), float(max_rsrp_dbm), inclusive="both"
        )
    ].copy()
    if selected.empty:
        raise ValueError("地图范围和RSRP门限内没有可用实测点")

    # 与无线电地图的1m栅格严格对齐，而不是简单floor世界坐标。
    selected["ix"] = np.floor(selected["blender_x"] - x_min).astype(int)
    selected["iy"] = np.floor(selected["blender_y"] - y_min).astype(int)
    selected = selected.loc[
        selected["ix"].between(0, int(map_size_m) - 1)
        & selected["iy"].between(0, int(map_size_m) - 1)
    ].copy()

    aggregations: dict[str, tuple[str, str]] = {
        "x_m": ("blender_x", "median"),
        "y_m": ("blender_y", "median"),
        "measured_rsrp_dbm": ("measured_rsrp_dbm", "median"),
        "raw_record_count": ("measured_rsrp_dbm", "size"),
    }
    if "rx_point_id" in selected.columns:
        aggregations["rx_point_count"] = ("rx_point_id", "nunique")

    grouped = (
        selected.groupby(["station_id", "pci", "ix", "iy"], as_index=False)
        .agg(**aggregations)
        .sort_values(["iy", "ix"])
        .reset_index(drop=True)
    )
    if "rx_point_count" not in grouped.columns:
        grouped["rx_point_count"] = grouped["raw_record_count"]

    label = ""
    for column in ("station_label", "station_name"):
        if column in selected.columns:
            nonempty = selected[column].dropna().astype(str)
            if len(nonempty):
                label = str(nonempty.iloc[0])
                break

    metadata = {
        "station_id": int(station_id),
        "pci": int(pci),
        "station_label": label,
        "tx_x_m": tx_x,
        "tx_y_m": tx_y,
        "x_min_m": x_min,
        "x_max_m": x_max,
        "y_min_m": y_min,
        "y_max_m": y_max,
        "aggregated_cell_count": int(len(grouped)),
        "raw_record_count": int(len(selected)),
    }
    return grouped, metadata


def select_spatial_training_points(
    aggregated: pd.DataFrame,
    training_count: int,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select a deterministic nested maximin subset using coordinates only.

    Calling this function with the same ``aggregated`` table and seed produces a
    nested design: the 10-point set is exactly the first 10 points of the
    20/30/40/50-point sets.  This makes point-count ablations substantially more
    controlled than independent KMeans runs, while still avoiding any use of
    measured RSRP values during point selection.
    """
    k = int(training_count)
    if k < 3:
        raise ValueError("training_count至少为3")
    if len(aggregated) <= k:
        raise ValueError(
            f"可用1m实测网格仅{len(aggregated)}个，必须大于训练点数{training_count}"
        )

    xy = aggregated[["x_m", "y_m"]].to_numpy(dtype=float)
    if not np.isfinite(xy).all():
        raise ValueError("训练候选点坐标包含NaN或Inf")

    # Start near the geometric median/center so the first point is not an
    # arbitrary boundary sample.  Tiny deterministic jitter breaks exact ties
    # without changing the geometric objective.
    center = np.median(xy, axis=0)
    d2_center = np.sum((xy - center[None, :]) ** 2, axis=1)
    rng = np.random.default_rng(int(random_seed))
    tie_jitter = rng.uniform(0.0, 1e-9, size=len(xy))
    first = int(np.argmin(d2_center + tie_jitter))

    selected_indices: list[int] = [first]
    selected_mask = np.zeros(len(xy), dtype=bool)
    selected_mask[first] = True
    min_d2 = np.sum((xy - xy[first][None, :]) ** 2, axis=1)
    min_d2[first] = -np.inf

    # Greedy farthest-point / maximin ordering.  Because the ordering itself is
    # independent of k, all requested training counts share a common prefix.
    while len(selected_indices) < k:
        score = min_d2 + tie_jitter
        score[selected_mask] = -np.inf
        chosen = int(np.argmax(score))
        if not np.isfinite(score[chosen]):
            raise RuntimeError("嵌套maximin选点失败：没有可继续选择的唯一候选点")
        selected_indices.append(chosen)
        selected_mask[chosen] = True
        d2_new = np.sum((xy - xy[chosen][None, :]) ** 2, axis=1)
        min_d2 = np.minimum(min_d2, d2_new)
        min_d2[selected_mask] = -np.inf

    training = aggregated.iloc[selected_indices].copy()
    validation = aggregated.loc[~aggregated.index.isin(aggregated.index[selected_indices])].copy()
    training["split"] = "training"
    training["selection_rank"] = np.arange(1, len(training) + 1, dtype=int)
    validation["split"] = "validation"
    return training.reset_index(drop=True), validation.reset_index(drop=True)


def build_dataset(
    csv_path: Path,
    station_id: int = 3,
    pci: int = 558,
    map_size_m: int = 512,
    training_count: int = 20,
    random_seed: int = 20260805,
    min_rsrp_dbm: float = -140.0,
    max_rsrp_dbm: float = -40.0,
) -> ReconstructionDataset:
    frame = read_processed_long_table(csv_path)
    aggregated, metadata = aggregate_one_meter_cells(
        frame=frame,
        station_id=station_id,
        pci=pci,
        map_size_m=map_size_m,
        min_rsrp_dbm=min_rsrp_dbm,
        max_rsrp_dbm=max_rsrp_dbm,
    )
    training, validation = select_spatial_training_points(
        aggregated=aggregated,
        training_count=training_count,
        random_seed=random_seed,
    )
    return ReconstructionDataset(
        station_id=int(station_id),
        pci=int(pci),
        station_label=str(metadata["station_label"]),
        tx_x_m=float(metadata["tx_x_m"]),
        tx_y_m=float(metadata["tx_y_m"]),
        x_min_m=float(metadata["x_min_m"]),
        x_max_m=float(metadata["x_max_m"]),
        y_min_m=float(metadata["y_min_m"]),
        y_max_m=float(metadata["y_max_m"]),
        aggregated=aggregated,
        training=training,
        validation=validation,
    )
