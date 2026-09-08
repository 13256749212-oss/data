from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .configuration import StationConfig


_SPLIT_RE = re.compile(r"[;,|/\s]+")
_NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())


def _choose_column(columns: Sequence[str], candidates: Sequence[str]) -> str | None:
    normalized = {_norm(c): c for c in columns}
    for candidate in candidates:
        hit = normalized.get(_norm(candidate))
        if hit is not None:
            return hit
    return None


def _numeric_tokens(value: object) -> list[float]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    if isinstance(value, (int, float, np.integer, np.floating)):
        return [float(value)] if np.isfinite(value) else []
    text = str(value).strip()
    if not text:
        return []
    return [float(v) for v in _NUMBER_RE.findall(text)]


def _suffix_key(column: str, token: str) -> str:
    name = str(column).lower()
    pos = name.find(token)
    if pos < 0:
        return _norm(name)
    suffix = name[pos + len(token):]
    digits = re.findall(r"\d+", suffix)
    return digits[-1] if digits else "default"


def _detect_xy(frame: pd.DataFrame) -> tuple[str, str]:
    x_col = _choose_column(
        list(frame.columns),
        ["x_m", "x", "blender_x", "blender_x_m", "scene_x", "x_abs_m"],
    )
    y_col = _choose_column(
        list(frame.columns),
        ["y_m", "y", "blender_y", "blender_y_m", "scene_y", "y_abs_m"],
    )
    if x_col is None or y_col is None:
        raise ValueError(
            "实测CSV必须包含完成坐标转换后的X/Y列。支持列名: "
            "x_m/y_m, x/y, blender_x/blender_y。"
        )
    return x_col, y_col


def _detect_long_columns(frame: pd.DataFrame) -> tuple[str | None, str | None]:
    pci_col = _choose_column(list(frame.columns), ["pci", "nr5g pci", "nr5g_pci"])
    rsrp_col = _choose_column(
        list(frame.columns),
        ["rsrp_dbm", "ss_rsrp_dbm", "nr5g ss rsrp", "rsrp", "ssrsrp"],
    )
    return pci_col, rsrp_col


def _wide_pairs(frame: pd.DataFrame) -> list[tuple[str, str]]:
    pci_cols = [c for c in frame.columns if "pci" in _norm(c)]
    rsrp_cols = [
        c for c in frame.columns
        if "rsrp" in _norm(c) and "rsrq" not in _norm(c)
    ]
    if not pci_cols or not rsrp_cols:
        return []

    by_pci: dict[str, str] = {}
    by_rsrp: dict[str, str] = {}
    for c in pci_cols:
        by_pci[_suffix_key(c, "pci")] = c
    for c in rsrp_cols:
        key = _suffix_key(c, "rsrp")
        by_rsrp[key] = c

    common = sorted(set(by_pci) & set(by_rsrp))
    if common:
        return [(by_pci[k], by_rsrp[k]) for k in common]
    if len(pci_cols) == len(rsrp_cols):
        return list(zip(sorted(pci_cols), sorted(rsrp_cols)))
    if len(pci_cols) == 1 and len(rsrp_cols) == 1:
        return [(pci_cols[0], rsrp_cols[0])]
    return []


def _expand_row_pairs(pci_value: object, rsrp_value: object) -> list[tuple[int, float]]:
    pcis = [int(round(v)) for v in _numeric_tokens(pci_value)]
    rsrps = _numeric_tokens(rsrp_value)
    if not pcis or not rsrps:
        return []
    if len(pcis) == 1 and len(rsrps) == 1:
        return [(pcis[0], float(rsrps[0]))]
    if len(pcis) != len(rsrps):
        return []
    return list(zip(pcis, map(float, rsrps)))


def read_measurements(path: Path) -> pd.DataFrame:
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(
            f"找不到实测CSV: {path}\n"
            "请把坐标转换后的 all_measurements_blender_xyz.csv 放入 data/，"
            "或通过 --measurements 指定路径。"
        )
    frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    if frame.empty:
        raise ValueError(f"实测CSV为空: {path}")

    x_col, y_col = _detect_xy(frame)
    station_col = _choose_column(list(frame.columns), ["station_id", "StationID", "base_station_id"])
    ground_col = _choose_column(list(frame.columns), ["ground_z", "ground_z_m", "dem_z", "terrain_z"])
    receiver_col = _choose_column(list(frame.columns), ["receiver_z", "receiver_z_m", "rx_z", "rx_z_m"])

    pci_long, rsrp_long = _detect_long_columns(frame)
    pairs = _wide_pairs(frame)
    records: list[dict[str, object]] = []

    for row_idx, row in frame.iterrows():
        try:
            x = float(row[x_col])
            y = float(row[y_col])
        except (TypeError, ValueError):
            continue
        if not (np.isfinite(x) and np.isfinite(y)):
            continue

        row_pairs: list[tuple[int, float]] = []
        if pci_long is not None and rsrp_long is not None:
            row_pairs.extend(_expand_row_pairs(row[pci_long], row[rsrp_long]))
        if not row_pairs:
            for pci_col, rsrp_col in pairs:
                row_pairs.extend(_expand_row_pairs(row[pci_col], row[rsrp_col]))
        if not row_pairs:
            continue

        station_id = np.nan
        if station_col is not None:
            vals = _numeric_tokens(row[station_col])
            if vals:
                station_id = int(round(vals[0]))

        ground_z = np.nan
        if ground_col is not None:
            vals = _numeric_tokens(row[ground_col])
            if vals:
                ground_z = vals[0]
        receiver_z = np.nan
        if receiver_col is not None:
            vals = _numeric_tokens(row[receiver_col])
            if vals:
                receiver_z = vals[0]

        for pci, rsrp in row_pairs:
            # SS-RSRP is a power in dBm. Values outside this generous range are invalid.
            if not (-160.0 <= float(rsrp) <= -20.0):
                continue
            records.append(
                {
                    "source_row": int(row_idx),
                    "x_m": x,
                    "y_m": y,
                    "station_id": station_id,
                    "pci": int(pci),
                    "rsrp_dbm": float(rsrp),
                    "ground_z_input_m": ground_z,
                    "receiver_z_input_m": receiver_z,
                }
            )

    result = pd.DataFrame.from_records(records)
    if result.empty:
        raise ValueError(
            "没有解析到有效PCI-RSRP对。请检查列名和内容。"
            "支持长表格式，也支持同一行多个PCI/RSRP列或分隔字符串。"
        )
    return result


def prepare_station_measurements(
    observations: pd.DataFrame,
    station: StationConfig,
    map_size_x_m: float,
    map_size_y_m: float,
    cell_size_m: float,
    min_points_per_pci: int,
    max_cells_per_pci: int,
    strong_signal_sampling_fraction: float,
) -> pd.DataFrame:
    station.validate()
    obs = observations.loc[observations["pci"].isin(station.pcis)].copy()
    if "station_id" in obs and obs["station_id"].notna().any():
        matched = obs["station_id"].eq(station.station_id)
        # Keep PCI-matched rows if station_id is unavailable on that row.
        obs = obs.loc[matched | obs["station_id"].isna()].copy()

    x_min = station.x_m - map_size_x_m / 2.0
    y_min = station.y_m - map_size_y_m / 2.0
    x_max = x_min + map_size_x_m
    y_max = y_min + map_size_y_m
    obs = obs.loc[
        obs["x_m"].between(x_min, x_max, inclusive="left")
        & obs["y_m"].between(y_min, y_max, inclusive="left")
    ].copy()
    if obs.empty:
        raise ValueError(f"{station.station_id}号站512m范围内没有对应PCI实测点")

    obs["ix"] = np.floor((obs["x_m"] - x_min) / cell_size_m).astype(int)
    obs["iy"] = np.floor((obs["y_m"] - y_min) / cell_size_m).astype(int)
    obs["cell_x_m"] = x_min + (obs["ix"] + 0.5) * cell_size_m
    obs["cell_y_m"] = y_min + (obs["iy"] + 0.5) * cell_size_m

    aggregated = (
        obs.groupby(["pci", "ix", "iy", "cell_x_m", "cell_y_m"], as_index=False)
        .agg(
            measured_rsrp_dbm=("rsrp_dbm", "median"),
            raw_observation_count=("rsrp_dbm", "size"),
            ground_z_input_m=("ground_z_input_m", "median"),
            receiver_z_input_m=("receiver_z_input_m", "median"),
        )
    )

    sampled: list[pd.DataFrame] = []
    frac = float(np.clip(strong_signal_sampling_fraction, 0.0, 1.0))
    for pci in station.pcis:
        part = aggregated.loc[aggregated["pci"].eq(pci)].copy()
        if len(part) < min_points_per_pci:
            raise ValueError(
                f"{station.station_id}号站 PCI {pci} 只有{len(part)}个1m聚合点，"
                f"少于要求的{min_points_per_pci}个"
            )
        if len(part) > max_cells_per_pci:
            n_strong = max(1, int(round(max_cells_per_pci * frac)))
            strong = part.nlargest(n_strong, "measured_rsrp_dbm")
            remaining = part.drop(index=strong.index)
            n_random = max_cells_per_pci - len(strong)
            random = remaining.sample(
                min(n_random, len(remaining)), random_state=20260805
            )
            part = pd.concat([strong, random], ignore_index=True)
        sampled.append(part)

    result = pd.concat(sampled, ignore_index=True)
    result["station_id"] = station.station_id
    result["x_min_m"] = x_min
    result["y_min_m"] = y_min
    return result.sort_values(["pci", "iy", "ix"]).reset_index(drop=True)
