#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export reproducibility metadata for the released dataset and processing workflow.

The script records exact values from the released code/assets and author-specified
terrain/building provenance from ``config/source_metadata.json``.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

import numpy as np
import trimesh
import yaml

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def mesh_info(path: Path) -> dict:
    mesh = trimesh.load_mesh(path, process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        return {"path": str(path.relative_to(ROOT)), "type": type(mesh).__name__, "sha256": sha256(path)}
    bounds = np.asarray(mesh.bounds, float)
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256(path),
        "vertex_count": int(len(mesh.vertices)),
        "face_count": int(len(mesh.faces)),
        "bounds_xyz": bounds.tolist(),
        "extent_xyz": (bounds[1]-bounds[0]).tolist(),
    }


def main() -> int:
    out_dir = ROOT / 'metadata'
    out_dir.mkdir(parents=True, exist_ok=True)
    coord = json.loads((ROOT/'config/coordinate_alignment.json').read_text(encoding='utf-8'))
    with (ROOT/'workflows/parameter_calibration/config.yaml').open('r', encoding='utf-8') as f:
        cal = yaml.safe_load(f)
    source_path = ROOT/'config/source_metadata.json'
    source = json.loads(source_path.read_text(encoding='utf-8')) if source_path.exists() else {}
    global_epre_path = ROOT/'config/global_epre_calibration.json'
    global_epre = json.loads(global_epre_path.read_text(encoding='utf-8')) if global_epre_path.exists() else {}
    global_epre_offset_db = float(global_epre.get('global_epre_offset_db', cal['radio'].get('sss_re_epre_offset_db', 0.0)))
    evaluation_reference_path = ROOT/'metadata/evaluation_reference.json'
    evaluation_reference = json.loads(evaluation_reference_path.read_text(encoding='utf-8')) if evaluation_reference_path.exists() else {}
    residual_path = ROOT/'config/network_residual_calibration.json'
    residual = json.loads(residual_path.read_text(encoding='utf-8')) if residual_path.exists() else {}

    spatial = {
        "source_crs": coord.get("source_crs"),
        "projected_crs": coord.get("projected_crs"),
        "scene_origin_projected_x": coord.get("scene_origin_projected_x"),
        "scene_origin_projected_y": coord.get("scene_origin_projected_y"),
        "blender_x_scale": coord.get("blender_x_scale", 1.0),
        "blender_y_scale": coord.get("blender_y_scale", 1.0),
        "blender_x_offset_m": coord.get("blender_x_offset_m", 0.0),
        "blender_y_offset_m": coord.get("blender_y_offset_m", 0.0),
        "receiver_height_agl_m": coord.get("receiver_height_agl_m", 1.5),
        "recommended_metric_crs_for_external_distance_analysis": coord.get("recommended_metric_crs_for_external_distance_analysis", "EPSG:32648"),
        "grid_interval_interpretation": coord.get("grid_interval_interpretation"),
        "axis_definition": coord.get("axis_definition"),
        "transform_formula": {
            "x_local": "(x_EPSG3857 - scene_origin_projected_x) * blender_x_scale + blender_x_offset_m",
            "y_local": "(y_EPSG3857 - scene_origin_projected_y) * blender_y_scale + blender_y_offset_m",
            "receiver_z": "terrain_z_from_ground_mesh + receiver_height_agl_m",
        },
    }
    (out_dir/'spatial_reference.json').write_text(json.dumps(spatial, ensure_ascii=False, indent=2), encoding='utf-8')

    scene = {
        "terrain_mesh": mesh_info(ROOT/'assets/ground.ply'),
        "building_mesh": mesh_info(ROOT/'assets/ynu_chenggong_campus-001.ply'),
        "ground_material": cal['scene']['ground_material'],
        "building_material": cal['scene']['building_material'],
        "source_provenance": source,
    }
    (out_dir/'scene_metadata.json').write_text(json.dumps(scene, ensure_ascii=False, indent=2), encoding='utf-8')

    propagation = {
        "frequency_hz": cal['radio']['frequency_hz'],
        "bandwidth_hz": cal['radio']['bandwidth_hz'],
        "nr_band": cal['radio']['nr_band'],
        "center_arfcn_dl": cal['radio']['center_arfcn_dl'],
        "ssb_arfcn_dl": cal['radio']['ssb_arfcn_dl'],
        "n_rb": cal['radio']['n_rb'],
        "subcarriers_per_rb": cal['radio']['subcarriers_per_rb'],
        "receiver_height_agl_m": cal['radio']['rx_height_agl_m'],
        "propagation_model": {
            "max_depth": 5,
            "diffraction": True,
            "edge_diffraction": True,
            "scope": "calibration + local refinement + per-station maps + joint map",
        },
        "antenna": cal['antenna'],
        "generic_antenna_limitation": "The built-in Sionna/3GPP-style generic array model is not a vendor-specific commercial antenna pattern.",
        "ss_rsrp_method": {
            "name": "3gpp_ts_38_215_sss_re_uniform_epre_narrowband_global_calibrated",
            "definition": "Linear average of received power contributions of resource elements carrying SSS, following 3GPP TS 38.215 clause 5.1.1.",
            "sss_sequence_re_count": 127,
            "sss_sequence_reference": "3GPP TS 38.211 clauses 7.4.2.3 and 7.4.3.1.2",
            "ss_pbch_block_width_subcarriers": 240,
            "averaging_domain": "linear watts",
            "narrowband_path_gain_assumption": True,
            "configured_active_subcarrier_count": int(cal['radio']['n_rb']) * int(cal['radio']['subcarriers_per_rb']),
            "tx_power_allocation_assumption": "uniform over n_rb * subcarriers_per_rb configured active subcarriers",
            "tx_power_to_sss_re_epre": "P_SSS_RE_EPRE_dBm = P_carrier_dBm - 10log10(n_rb*subcarriers_per_rb) + network_global_epre_offset_db",
            "network_global_epre_offset_db": global_epre_offset_db,
            "global_epre_calibration": global_epre,
            "use_of_3276": "TX-side carrier-power allocation only; not an SS-RSRP averaging divisor",
            "division_by_127_or_240_for_ss_rsrp": False,
            "station_specific_sss_re_offset_fitted": False,
            "station_level_absolute_power_parameter": "shared_power_dbm only (50-55 dBm)",
            "station_specific_epre_offset_fitted": False,
            "exact_frequency_selective_ue_measurement": False,
        },
        "search_ranges": cal['search'],
    }
    propagation['evaluation_reference'] = evaluation_reference
    propagation['network_residual_calibration'] = residual
    propagation['joint_map_reference'] = evaluation_reference.get('joint_map', {}) if evaluation_reference else {}
    (out_dir/'propagation_configuration.json').write_text(json.dumps(propagation, ensure_ascii=False, indent=2), encoding='utf-8')
    if evaluation_reference:
        (out_dir/'evaluation_reference.json').write_text(json.dumps(evaluation_reference, ensure_ascii=False, indent=2), encoding='utf-8')

    software = {
        "python": "3.10",
        "sionna_rt": "1.2.2",
        "blender": "4.5 LTS",
        "requirements_file": "requirements.txt",
        "environment_file": "environment.yml",
        "note": "Declared reproduction environment used by the manuscript workflow; this metadata does not report the interpreter of the machine that regenerates the JSON files.",
    }
    (out_dir/'software_environment.json').write_text(json.dumps(software, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"[OK] reproducibility metadata -> {out_dir}")
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
