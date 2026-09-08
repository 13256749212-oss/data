from __future__ import annotations

import gc
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np

from .angles import apply_common_azimuth_offset, downtilt_to_sionna_beta_rad
from .configuration import StationConfig
from .terrain import SurfaceInfo, TerrainModel


@dataclass(frozen=True)
class Candidate:
    height_agl_m: float
    azimuth_offset_deg: float
    downtilt_delta_deg: float
    reference_power_dbm: float


def _lazy_sionna_imports():
    try:
        from sionna.rt import PlanarArray, RadioMapSolver, Transmitter, load_mesh, load_scene
    except Exception as exc:  # pragma: no cover - requires Sionna runtime
        raise RuntimeError(
            "无法导入Sionna RT 1.2.2。请在支持CUDA的Linux环境安装 requirements.txt。"
        ) from exc
    return PlanarArray, RadioMapSolver, Transmitter, load_mesh, load_scene


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def clear_transmitters(scene: Any) -> None:
    for name in list(scene.transmitters.keys()):
        scene.remove(name)


def configure_scene(scene_xml: Path, cfg: Dict[str, Any]) -> Any:
    PlanarArray, _, _, _, load_scene = _lazy_sionna_imports()
    scene = load_scene(str(Path(scene_xml).resolve()), merge_shapes=False)
    radio = cfg["radio"]
    antenna = cfg["antenna"]
    scene.frequency = float(radio["frequency_hz"])
    scene.bandwidth = float(radio["bandwidth_hz"])
    scene.tx_array = PlanarArray(
        num_rows=int(antenna["num_rows"]),
        num_cols=int(antenna["num_cols"]),
        vertical_spacing=float(antenna["vertical_spacing"]),
        horizontal_spacing=float(antenna["horizontal_spacing"]),
        pattern=str(antenna["pattern"]),
        polarization=str(antenna["polarization"]),
    )
    scene.rx_array = PlanarArray(
        num_rows=1,
        num_cols=1,
        pattern="iso",
        polarization="V",
    )
    return scene


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def candidate_key(
    station: StationConfig,
    candidate: Candidate,
    surface: SurfaceInfo,
    samples_per_tx: int,
    cfg: Dict[str, Any],
) -> str:
    radio = cfg["radio"]
    antenna = cfg["antenna"]
    scene_xml = Path(cfg.get("_resolved_scene_xml", ""))
    payload = {
        "station_id": station.station_id,
        "station_xy": [round(station.x_m, 6), round(station.y_m, 6)],
        "pcis": station.pcis,
        "initial_alphas_rad": [round(v, 12) for v in station.initial_alphas_rad],
        "original_downtilt_deg": round(station.original_downtilt_deg, 6),
        "height_agl_m": round(candidate.height_agl_m, 6),
        "azimuth_offset_deg": round(candidate.azimuth_offset_deg, 6),
        "downtilt_delta_deg": round(candidate.downtilt_delta_deg, 6),
        "reference_power_dbm": round(candidate.reference_power_dbm, 6),
        "surface_path": str(surface.path),
        "surface_sha256": _sha256_file(surface.path),
        "surface_faces": surface.n_faces,
        "scene_xml": str(scene_xml),
        "scene_xml_sha256": _sha256_file(scene_xml) if scene_xml.is_file() else None,
        "samples_per_tx": int(samples_per_tx),
        "frequency_hz": float(radio["frequency_hz"]),
        "bandwidth_hz": float(radio["bandwidth_hz"]),
        "n_rb": int(radio["n_rb"]),
        "subcarriers_per_rb": int(radio["subcarriers_per_rb"]),
        "rsrp_calibration_offset_db": float(radio.get("rsrp_calibration_offset_db", 0.0)),
        "max_depth": int(radio["max_depth"]),
        "los": bool(radio["los"]),
        "specular_reflection": bool(radio["specular_reflection"]),
        "diffuse_reflection": bool(radio["diffuse_reflection"]),
        "refraction": bool(radio["refraction"]),
        "diffraction": bool(radio["diffraction"]),
        "edge_diffraction": bool(radio.get("edge_diffraction", False)),
        "seed": int(radio["seed"]),
        "antenna": antenna,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:24]


def run_candidate(
    scene: Any,
    terrain: TerrainModel,
    station: StationConfig,
    candidate: Candidate,
    surface: SurfaceInfo,
    cfg: Dict[str, Any],
    samples_per_tx: int,
    cache_dir: Path | None = None,
    force: bool = False,
) -> Dict[str, Any]:
    _, RadioMapSolver, Transmitter, load_mesh, _ = _lazy_sionna_imports()
    radio = cfg["radio"]
    seed = int(radio["seed"])
    key = candidate_key(station, candidate, surface, samples_per_tx, cfg)
    cache_path = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{key}.npz"
        if cache_path.exists() and not force:
            data = np.load(cache_path, allow_pickle=False)
            return {
                "sector_rsrp_dbm": data["sector_rsrp_dbm"],
                "alphas_rad": data["alphas_rad"],
                "beta_rad": float(data["beta_rad"][0]),
                "ground_z_m": float(data["ground_z_m"][0]),
                "tx_z_m": float(data["tx_z_m"][0]),
                "cache_hit": True,
                "cache_key": key,
            }

    clear_transmitters(scene)
    ground_z = float(terrain.query(station.x_m, station.y_m))
    tx_z = ground_z + float(candidate.height_agl_m)
    alphas = apply_common_azimuth_offset(
        station.initial_alphas_rad, candidate.azimuth_offset_deg
    )
    absolute_downtilt_deg = station.original_downtilt_deg + candidate.downtilt_delta_deg
    beta = downtilt_to_sionna_beta_rad(absolute_downtilt_deg)

    for sector_index, (pci, alpha) in enumerate(zip(station.pcis, alphas), start=1):
        scene.add(
            Transmitter(
                name=f"st{station.station_id}_pci{pci}",
                position=[station.x_m, station.y_m, tx_z],
                orientation=[float(alpha), float(beta), 0.0],
                power_dbm=float(candidate.reference_power_dbm),
            )
        )

    measurement_surface = load_mesh(str(surface.path), flip_normals=False)
    solver = RadioMapSolver()
    rm = solver(
        scene,
        measurement_surface=measurement_surface,
        samples_per_tx=int(samples_per_tx),
        max_depth=int(radio["max_depth"]),
        los=bool(radio["los"]),
        specular_reflection=bool(radio["specular_reflection"]),
        diffuse_reflection=bool(radio["diffuse_reflection"]),
        refraction=bool(radio["refraction"]),
        diffraction=bool(radio["diffraction"]),
        edge_diffraction=bool(radio.get("edge_diffraction", False)),
        seed=seed,
    )
    rss_w = _to_numpy(rm.rss).astype(np.float64, copy=False)
    n_tx = len(station.pcis)
    if rss_w.size != n_tx * surface.n_faces:
        raise RuntimeError(
            f"MeshRadioMap输出尺寸异常: rss.shape={rss_w.shape}, size={rss_w.size}, "
            f"期望 {n_tx}×{surface.n_faces}"
        )
    rss_w = rss_w.reshape(n_tx, surface.n_faces)
    # Every 1 m square is represented by two consecutive triangles.
    rss_cell_w = rss_w.reshape(n_tx, surface.n_cells, 2).mean(axis=2)
    # A zero value means no simulated path hit the cell for this Monte Carlo run.
    # Keep it as NaN rather than inventing an extremely weak received power.
    rss_dbm = np.full(rss_cell_w.shape, np.nan, dtype=np.float64)
    positive = np.isfinite(rss_cell_w) & (rss_cell_w > 0.0)
    rss_dbm[positive] = 10.0 * np.log10(rss_cell_w[positive] * 1000.0)
    re_count = int(radio["n_rb"]) * int(radio["subcarriers_per_rb"])
    rsrp_offset_db = 10.0 * np.log10(float(re_count))
    sector_rsrp_dbm = (
        rss_dbm
        - rsrp_offset_db
        + float(radio.get("rsrp_calibration_offset_db", 0.0))
    ).astype(np.float32)

    result = {
        "sector_rsrp_dbm": sector_rsrp_dbm,
        "alphas_rad": np.asarray(alphas, dtype=np.float64),
        "beta_rad": float(beta),
        "ground_z_m": ground_z,
        "tx_z_m": tx_z,
        "cache_hit": False,
        "cache_key": key,
    }
    if cache_path is not None:
        np.savez_compressed(
            cache_path,
            sector_rsrp_dbm=sector_rsrp_dbm,
            alphas_rad=np.asarray(alphas, dtype=np.float64),
            beta_rad=np.asarray([beta], dtype=np.float64),
            ground_z_m=np.asarray([ground_z], dtype=np.float64),
            tx_z_m=np.asarray([tx_z], dtype=np.float64),
        )

    del rm, rss_w, rss_cell_w, measurement_surface
    gc.collect()
    try:  # pragma: no cover - depends on torch availability
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return result
