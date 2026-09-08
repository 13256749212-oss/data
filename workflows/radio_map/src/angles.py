from __future__ import annotations

import math
from typing import Iterable

import numpy as np

TWO_PI = 2.0 * math.pi


def wrap_rad(angle: float) -> float:
    """Normalize an angle to [-pi, pi)."""
    return (float(angle) + math.pi) % TWO_PI - math.pi


def wrap_deg(angle: float) -> float:
    return (float(angle) + 180.0) % 360.0 - 180.0


def sionna_alpha_to_north_clockwise_deg(alpha_rad: float) -> float:
    """
    Sionna boresight at alpha=0 points along +X.
    Project convention: azimuth 0° points +Y and increases clockwise.
    A_north_clockwise = 90° - alpha_deg.
    """
    return (90.0 - math.degrees(alpha_rad)) % 360.0


def north_clockwise_deg_to_sionna_alpha(azimuth_deg: float) -> float:
    return wrap_rad(math.radians(90.0 - float(azimuth_deg)))


def apply_common_azimuth_offset(
    initial_alphas_rad: Iterable[float], offset_deg: float
) -> tuple[float, ...]:
    """
    同时支持：
    - 单PCI全向站：1个alpha，不执行三扇区间隔校验；
    - 三扇区站：3个alpha，并校验120度间隔。

    22号站属于单PCI全向站，必须走第一种逻辑。
    """
    offset_rad = math.radians(float(offset_deg))
    values = tuple(wrap_rad(a + offset_rad) for a in initial_alphas_rad)
    if len(values) == 1:
        return values
    if len(values) == 3:
        validate_sector_spacing(values)
        return values
    raise ValueError(
        f"仅支持单PCI全向站或三扇区站，实际alpha数量={len(values)}"
    )


def validate_sector_spacing(
    alphas_rad: Iterable[float], tolerance_deg: float = 1.0
) -> np.ndarray:
    angles = np.mod(np.asarray(tuple(alphas_rad), dtype=float), TWO_PI)
    if angles.size != 3:
        raise ValueError("三扇区间隔校验需要3个角度")
    ordered = np.sort(angles)
    gaps = np.diff(np.r_[ordered, ordered[0] + TWO_PI])
    gaps_deg = np.degrees(gaps)
    if not np.all(np.abs(gaps_deg - 120.0) <= float(tolerance_deg) + 1e-8):
        raise ValueError(
            f"三扇区间隔必须为120°±{tolerance_deg}°，实际为 {gaps_deg.tolist()}"
        )
    return gaps_deg


def downtilt_to_sionna_beta_rad(downtilt_deg: float) -> float:
    """
    Sionna orientation=[alpha,beta,gamma] uses radians.
    With the local boresight initially along +X, positive beta rotates it toward -Z,
    hence positive beta means electrical/mechanical downtilt in this project.
    Negative beta is uptilt.
    """
    return math.radians(float(downtilt_deg))


def direction_vector(alpha_rad: float, beta_rad: float) -> np.ndarray:
    """Boresight direction after Z(alpha)-Y(beta) rotation."""
    return np.asarray(
        [
            math.cos(alpha_rad) * math.cos(beta_rad),
            math.sin(alpha_rad) * math.cos(beta_rad),
            -math.sin(beta_rad),
        ],
        dtype=float,
    )
