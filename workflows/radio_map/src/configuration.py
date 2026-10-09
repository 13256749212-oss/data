from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import yaml


@dataclass(frozen=True)
class StationConfig:
    station_id: int
    label: str
    x_m: float
    y_m: float
    pcis: tuple[int, int, int]
    initial_alphas_rad: tuple[float, float, float]
    original_downtilt_deg: float
    initial_power_dbm: float

    def validate(self, tolerance_deg: float = 1.0) -> None:
        if len(set(self.pcis)) != 3:
            raise ValueError(f"{self.station_id}号站的3个PCI必须互不相同: {self.pcis}")
        angles = np.mod(np.asarray(self.initial_alphas_rad, dtype=float), 2.0 * np.pi)
        ordered = np.sort(angles)
        gaps = np.diff(np.r_[ordered, ordered[0] + 2.0 * np.pi])
        gap_deg = np.degrees(gaps)
        if not np.all(np.abs(gap_deg - 120.0) <= tolerance_deg + 1e-8):
            raise ValueError(
                f"{self.station_id}号站初始三扇区间隔不是120°±{tolerance_deg}°: "
                f"{gap_deg.tolist()}"
            )


def load_yaml(path: Path) -> Dict[str, Any]:
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"找不到配置文件: {path}")
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("config.yaml 顶层必须是字典")
    cfg["_config_path"] = str(path)
    cfg["_root"] = str(path.parent)
    return cfg


def resolve_path(cfg: Dict[str, Any], value: str | Path) -> Path:
    p = Path(value).expanduser()
    if not p.is_absolute():
        p = Path(cfg["_root"]) / p
    return p.resolve()


def inclusive_grid(start: float, stop: float, step: float) -> np.ndarray:
    if step <= 0:
        raise ValueError("step必须大于0")
    n = int(round((stop - start) / step))
    values = start + np.arange(n + 1, dtype=float) * step
    if len(values) == 0 or abs(values[-1] - stop) > 1e-7:
        values = np.r_[values, stop]
    return values


def station_ids_arg(text: str | None, available: Iterable[int]) -> list[int]:
    all_ids = sorted(set(int(v) for v in available))
    if not text:
        return all_ids
    selected = sorted(set(int(v.strip()) for v in text.split(",") if v.strip()))
    unknown = sorted(set(selected) - set(all_ids))
    if unknown:
        raise ValueError(f"测试站配置中不存在这些station_id: {unknown}")
    return selected
