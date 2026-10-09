#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""将原始 Cellular-Pro 经纬度数据转换到 BlenderGIS 局部坐标并采样 DEM。

正式流程仍为：WGS84(EPSG:4326) -> EPSG:3857 -> Blender local。
本版增加两项防护：
1) 如果项目中保留了历史有效 processed 表（含经纬度与 blender_x/y），优先从这些已验证数据反推运行时 scene origin，避免错误/过期的 origin 使全部 DEM 采样落空；
2) 自动识别并修复本批 Cellular-Pro CSV 的“TIME 实际缺失导致整行相对表头左移一列”问题；
3) 在完成整行字段修复后，再检查 LONGITUDE/LATITUDE 是否真的被源 CSV 对调。

不会通过“最近 DEM 点”或任意平移强行制造 DEM 命中；若标准模式、历史参考恢复和经纬度交换都不能与 ground.ply 对齐，脚本直接停止并输出坐标范围诊断，避免继续生成空的多 PCI 长表。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib.tri as mtri
import numpy as np
import pandas as pd
import trimesh
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[2]


def _norm_name(value: object) -> str:
    return str(value).strip().lower().replace(" ", "").replace("_", "")


def _find_column(columns: Iterable[object], aliases: Iterable[str], required: bool = True):
    norm = {_norm_name(c): c for c in columns}
    for alias in aliases:
        hit = norm.get(_norm_name(alias))
        if hit is not None:
            return hit
    if required:
        raise KeyError(f"找不到字段，候选={list(aliases)}，实际字段={list(columns)}")
    return None


def read_csv_robust(path: Path) -> pd.DataFrame:
    last = None
    for enc in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except UnicodeDecodeError as exc:
            last = exc
    if last is not None:
        raise last
    raise RuntimeError(f"无法读取CSV: {path}")


def _numeric(values) -> np.ndarray:
    return pd.to_numeric(values, errors="coerce").to_numpy(float)


def _repair_cellularpro_left_shift_if_needed(frame: pd.DataFrame):
    """修复本数据集 Cellular-Pro CSV 的一列左移导出布局。

    已核验的 2026-07-25 原始 CSV 存在如下结构：表头包含 TIME，但实际数据行
    没有 TIME 值，因此整行数据相对表头左移一列，并在末尾保留一个空字段。
    例如：
      TIME=24.831435, LATITUDE=102.858915, LONGITUDE=0.0, SPEED=1951
    实际应解释为：
      LATITUDE=24.831435, LONGITUDE=102.858915, SPEED=0.0, ALT=1951

    这不仅影响坐标，还会使 NR5G PCI / SS-RSRP / Cells PCI List / Cells RSRP List
    等所有无线字段整体错位。因此必须在坐标转换和多 PCI 提取之前修复整个原始表。

    仅在“标准 LATITUDE/LONGITUDE 几乎无效，而 TIME/LATITUDE 作为 lat/lon
    大量有效”的情况下自动修复，避免误改正常 CSV。
    """
    required = ["TIME", "LATITUDE", "LONGITUDE"]
    if not all(c in frame.columns for c in required):
        return frame, {"mode": "normal", "reason": "required_columns_missing"}

    lat_normal = _numeric(frame["LATITUDE"])
    lon_normal = _numeric(frame["LONGITUDE"])
    lat_shift = _numeric(frame["TIME"])
    lon_shift = _numeric(frame["LATITUDE"])

    normal_valid = _valid_lonlat(lon_normal, lat_normal)
    shifted_valid = _valid_lonlat(lon_shift, lat_shift)
    n = max(len(frame), 1)
    normal_ratio = float(normal_valid.sum() / n)
    shifted_ratio = float(shifted_valid.sum() / n)

    # 本数据集正常经纬度解释应接近全有效。只有非常明确时才执行整行右移修复。
    should_repair = (normal_ratio < 0.10) and (shifted_ratio > 0.80) and (shifted_valid.sum() >= 20)
    if not should_repair:
        return frame, {
            "mode": "normal",
            "normal_valid_lonlat_rows": int(normal_valid.sum()),
            "shifted_candidate_valid_lonlat_rows": int(shifted_valid.sum()),
            "normal_valid_ratio": normal_ratio,
            "shifted_candidate_valid_ratio": shifted_ratio,
        }

    cols = list(frame.columns)
    arr = frame.to_numpy(dtype=object, copy=True)
    repaired = np.empty(arr.shape, dtype=object)
    repaired[:, 0] = np.nan  # 原始 TIME 值在此导出中缺失，不能伪造。
    repaired[:, 1:] = arr[:, :-1]
    fixed = pd.DataFrame(repaired, columns=cols, index=frame.index)

    # 关键字段语义自检。修复后纬度应来自旧 TIME，经度来自旧 LATITUDE。
    fixed_lat = _numeric(fixed["LATITUDE"])
    fixed_lon = _numeric(fixed["LONGITUDE"])
    fixed_valid = _valid_lonlat(fixed_lon, fixed_lat)
    return fixed, {
        "mode": "cellularpro_missing_time_left_shift_repaired",
        "normal_valid_lonlat_rows_before_repair": int(normal_valid.sum()),
        "shifted_candidate_valid_lonlat_rows_before_repair": int(shifted_valid.sum()),
        "valid_lonlat_rows_after_repair": int(fixed_valid.sum()),
        "rows": int(len(frame)),
        "repair": "new_col[1:] = old_col[:-1]; TIME set to missing",
    }


def load_ground(path: Path):
    mesh = trimesh.load_mesh(path, process=False)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) < 3 or len(mesh.faces) < 1:
        raise ValueError(f"ground.ply不是有效三角网格: {path}")
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=int)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("ground.ply必须为三角面")
    tri = mtri.Triangulation(vertices[:, 0], vertices[:, 1], faces)
    interp = mtri.LinearTriInterpolator(tri, vertices[:, 2])
    trifinder = tri.get_trifinder()
    return mesh, interp, trifinder


def parse_args():
    parser = argparse.ArgumentParser(description="Cellular-Pro坐标转换、BlenderGIS对齐和DEM+1.5m高度提取")
    parser.add_argument("--input-dir", default=str(ROOT / "data/raw_measurements"))
    parser.add_argument("--output-dir", default=str(ROOT / "data/aligned_measurements"))
    parser.add_argument("--ground", default=str(ROOT / "assets/ground.ply"))
    parser.add_argument("--alignment-config", default=str(ROOT / "config/coordinate_alignment.json"))
    parser.add_argument("--glob", default="*.csv")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _valid_lonlat(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    return (
        np.isfinite(lon)
        & np.isfinite(lat)
        & (np.abs(lon) <= 180.0)
        & (np.abs(lat) <= 90.0)
        & ~((lon == 0.0) & (lat == 0.0))
    )


def _project_to_local(
    lon: np.ndarray,
    lat: np.ndarray,
    transformer: Transformer,
    origin_x: float,
    origin_y: float,
    scale_x: float,
    scale_y: float,
    offset_x: float,
    offset_y: float,
):
    valid = _valid_lonlat(lon, lat)
    px = np.full(len(lon), np.nan, dtype=float)
    py = np.full(len(lon), np.nan, dtype=float)
    if valid.any():
        tx, ty = transformer.transform(lon[valid], lat[valid])
        px[valid] = np.asarray(tx, dtype=float)
        py[valid] = np.asarray(ty, dtype=float)
    bx = (px - origin_x) * scale_x + offset_x
    by = (py - origin_y) * scale_y + offset_y
    return valid, px, py, bx, by


def _terrain_hit_mask(bx: np.ndarray, by: np.ndarray, trifinder) -> np.ndarray:
    valid = np.isfinite(bx) & np.isfinite(by)
    hit = np.zeros(len(bx), dtype=bool)
    if valid.any():
        tri_index = np.asarray(trifinder(bx[valid], by[valid]), dtype=int)
        hit[valid] = tri_index >= 0
    return hit


def _safe_range(values: np.ndarray):
    good = np.asarray(values, dtype=float)
    good = good[np.isfinite(good)]
    if good.size == 0:
        return None
    return [float(np.min(good)), float(np.max(good))]


def _reference_candidates() -> list[Path]:
    # 优先使用历史上最稳定的已处理表。prepare-data 在 extract 成功之前不会重写这些文件。
    return [
        ROOT / "data/processed/cell_pci_rsrp_long_27stations.csv",
        ROOT / "data/processed/cell_pci_rsrp_1m_calibration.csv",
        ROOT / "data/processed/cell_pci_rsrp_2p77m_localization.csv",
    ]


def _infer_origin_from_reference(
    transformer: Transformer,
    sx: float,
    sy: float,
    xoff: float,
    yoff: float,
):
    """从历史有效表恢复 EPSG:3857 -> Blender local 的 scene origin。

    只使用同时包含经纬度和 blender_x/blender_y 的非空表。返回值经过中位数稳健估计，
    并报告重建残差；残差过大时不会自动采用。
    """
    for path in _reference_candidates():
        if not path.exists() or path.stat().st_size <= 32:
            continue
        try:
            frame = read_csv_robust(path)
        except Exception:
            continue
        if frame.empty:
            continue
        lon_col = _find_column(frame.columns, ["longitude", "LONGITUDE", "lon"], required=False)
        lat_col = _find_column(frame.columns, ["latitude", "LATITUDE", "lat"], required=False)
        x_col = _find_column(frame.columns, ["blender_x", "x_m", "receiver_x_m", "x"], required=False)
        y_col = _find_column(frame.columns, ["blender_y", "y_m", "receiver_y_m", "y"], required=False)
        if None in (lon_col, lat_col, x_col, y_col):
            continue

        lon = pd.to_numeric(frame[lon_col], errors="coerce").to_numpy(float)
        lat = pd.to_numeric(frame[lat_col], errors="coerce").to_numpy(float)
        bx_ref = pd.to_numeric(frame[x_col], errors="coerce").to_numpy(float)
        by_ref = pd.to_numeric(frame[y_col], errors="coerce").to_numpy(float)
        valid = _valid_lonlat(lon, lat) & np.isfinite(bx_ref) & np.isfinite(by_ref)
        if int(valid.sum()) < 20:
            continue
        px, py = transformer.transform(lon[valid], lat[valid])
        px = np.asarray(px, dtype=float)
        py = np.asarray(py, dtype=float)
        # bx=(px-origin)*scale+offset -> origin=px-(bx-offset)/scale
        origin_x_samples = px - (bx_ref[valid] - xoff) / sx
        origin_y_samples = py - (by_ref[valid] - yoff) / sy
        ox = float(np.nanmedian(origin_x_samples))
        oy = float(np.nanmedian(origin_y_samples))
        bx_hat = (px - ox) * sx + xoff
        by_hat = (py - oy) * sy + yoff
        residual = np.hypot(bx_hat - bx_ref[valid], by_hat - by_ref[valid])
        med_res = float(np.nanmedian(residual))
        p95_res = float(np.nanpercentile(residual, 95.0))
        # 聚合表会引入亚网格级误差；这里给足容差，但拒绝明显不一致的参考表。
        if np.isfinite(med_res) and med_res <= 10.0 and np.isfinite(p95_res) and p95_res <= 30.0:
            return {
                "origin_x": ox,
                "origin_y": oy,
                "source": str(path),
                "rows": int(valid.sum()),
                "median_xy_residual_m": med_res,
                "p95_xy_residual_m": p95_res,
            }
    return None


def _read_all_input_lonlat(files: list[Path]):
    parts = []
    for path in files:
        frame_raw = read_csv_robust(path)
        frame, layout_info = _repair_cellularpro_left_shift_if_needed(frame_raw)
        lon_col = _find_column(frame.columns, ["LONGITUDE", "longitude", "lon"])
        lat_col = _find_column(frame.columns, ["LATITUDE", "latitude", "lat"])
        lon = pd.to_numeric(frame[lon_col], errors="coerce").to_numpy(float)
        lat = pd.to_numeric(frame[lat_col], errors="coerce").to_numpy(float)
        parts.append((path, frame, lon_col, lat_col, lon, lat, layout_info))
    return parts


def main():
    args = parse_args()
    inp = Path(args.input_dir)
    out = Path(args.output_dir)
    ground = Path(args.ground)
    cfg_path = Path(args.alignment_config)
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    out.mkdir(parents=True, exist_ok=True)

    files = [p for p in sorted(inp.glob(args.glob)) if p.is_file() and "_with_blender_xyz" not in p.stem]
    if not files:
        raise FileNotFoundError(f"在 {inp} 中没有找到原始CSV")

    mesh, interp, trifinder = load_ground(ground)
    transformer = Transformer.from_crs(
        cfg.get("source_crs", "EPSG:4326"),
        cfg.get("projected_crs", "EPSG:3857"),
        always_xy=True,
    )
    cfg_ox = float(cfg["scene_origin_projected_x"])
    cfg_oy = float(cfg["scene_origin_projected_y"])
    sx = float(cfg.get("blender_x_scale", 1.0))
    sy = float(cfg.get("blender_y_scale", 1.0))
    xoff = float(cfg.get("blender_x_offset_m", 0.0))
    yoff = float(cfg.get("blender_y_offset_m", 0.0))
    rxh = float(cfg.get("receiver_height_agl_m", 1.5))

    loaded = _read_all_input_lonlat(files)
    all_lon = np.concatenate([item[4] for item in loaded])
    all_lat = np.concatenate([item[5] for item in loaded])
    layout_reports = [{"source_file": item[0].name, **item[6]} for item in loaded]

    reference = _infer_origin_from_reference(transformer, sx, sy, xoff, yoff)
    if reference is not None:
        ox = float(reference["origin_x"])
        oy = float(reference["origin_y"])
        origin_source = "historical_processed_reference"
    else:
        ox = cfg_ox
        oy = cfg_oy
        origin_source = "config/coordinate_alignment.json"

    # 先按标准 LONGITUDE/LATITUDE 解释检查 DEM 覆盖。
    normal_valid, _, _, normal_bx, normal_by = _project_to_local(
        all_lon, all_lat, transformer, ox, oy, sx, sy, xoff, yoff
    )
    normal_hit = _terrain_hit_mask(normal_bx, normal_by, trifinder)

    # 仅作为防错候选检查经纬度列是否对调。
    swapped_valid, _, _, swapped_bx, swapped_by = _project_to_local(
        all_lat, all_lon, transformer, ox, oy, sx, sy, xoff, yoff
    )
    swapped_hit = _terrain_hit_mask(swapped_bx, swapped_by, trifinder)

    normal_hit_count = int(normal_hit.sum())
    swapped_hit_count = int(swapped_hit.sum())
    coordinate_mode = "normal"
    if normal_hit_count == 0 and swapped_hit_count > 0:
        coordinate_mode = "swapped_lon_lat"

    selected_bx = normal_bx if coordinate_mode == "normal" else swapped_bx
    selected_by = normal_by if coordinate_mode == "normal" else swapped_by
    selected_hit_count = normal_hit_count if coordinate_mode == "normal" else swapped_hit_count

    ground_bounds = np.asarray(mesh.bounds, dtype=float)
    preflight = {
        "ground_bounds": ground_bounds.tolist(),
        "configured_origin_epsg3857": [cfg_ox, cfg_oy],
        "runtime_origin_epsg3857": [ox, oy],
        "origin_source": origin_source,
        "reference_origin_recovery": reference,
        "coordinate_mode": coordinate_mode,
        "raw_row_count": int(len(all_lon)),
        "normal_valid_lonlat_rows": int(normal_valid.sum()),
        "normal_dem_hit_rows": normal_hit_count,
        "swapped_valid_lonlat_rows": int(swapped_valid.sum()),
        "swapped_dem_hit_rows": swapped_hit_count,
        "selected_dem_hit_rows": selected_hit_count,
        "selected_local_x_range": _safe_range(selected_bx),
        "selected_local_y_range": _safe_range(selected_by),
        "normal_local_x_range": _safe_range(normal_bx),
        "normal_local_y_range": _safe_range(normal_by),
        "swapped_local_x_range": _safe_range(swapped_bx),
        "swapped_local_y_range": _safe_range(swapped_by),
        "raw_layout_reports": layout_reports,
    }

    print("[ALIGN] ground XY bounds:", ground_bounds[:, :2].tolist())
    print("[ALIGN] origin source:", origin_source)
    print("[ALIGN] configured origin:", [cfg_ox, cfg_oy])
    print("[ALIGN] runtime origin:", [ox, oy])
    if reference is not None:
        print("[ALIGN] recovered from:", reference["source"])
        print("[ALIGN] reference residual median/p95 (m):", reference["median_xy_residual_m"], reference["p95_xy_residual_m"])
    print("[ALIGN] normal DEM hits:", normal_hit_count, "/", len(all_lon))
    print("[ALIGN] swapped-lon/lat DEM hits:", swapped_hit_count, "/", len(all_lon))
    print("[ALIGN] selected coordinate mode:", coordinate_mode)
    print("[ALIGN] selected local x range:", preflight["selected_local_x_range"])
    print("[ALIGN] selected local y range:", preflight["selected_local_y_range"])

    if selected_hit_count == 0:
        report = {
            "alignment_config": cfg,
            "preflight": preflight,
            "error": (
                "All measurement rows miss ground.ply after coordinate conversion. "
                "The multi-PCI extractor is not the cause; ground_z_m/receiver_z_m would be NaN for every row."
            ),
        }
        report_path = out / "alignment_report.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        raise RuntimeError(
            "坐标对齐阶段0个DEM命中，已停止在多PCI提取之前。"
            f" 详细坐标范围见 {report_path}。"
            " 请核对 scene origin / 原始经纬度；程序不会用最近高程或任意平移伪造 DEM+1.5 m 高度。"
        )

    reports = []
    all_parts = []
    all_valid_lon = []
    all_valid_lat = []

    for path, df, lon_col, lat_col, lon_raw, lat_raw, layout_info in loaded:
        target = out / f"{path.stem}_with_blender_xyz.csv"
        if target.exists() and not args.force:
            print("[SKIP]", target)
            continue

        if coordinate_mode == "normal":
            lon = lon_raw
            lat = lat_raw
        else:
            lon = lat_raw
            lat = lon_raw

        valid, px, py, bx, by = _project_to_local(lon, lat, transformer, ox, oy, sx, sy, xoff, yoff)
        if valid.any():
            all_valid_lon.extend(lon[valid].tolist())
            all_valid_lat.extend(lat[valid].tolist())

        gz = np.full(len(df), np.nan, dtype=float)
        hit = _terrain_hit_mask(bx, by, trifinder)
        if hit.any():
            z = np.ma.asarray(interp(bx[hit], by[hit]))
            gz[hit] = z.filled(np.nan)
        # 理论上 trifinder 命中而线性插值应有值；若极少数退化点失败，只将这些点保留为 miss。
        hit = np.isfinite(gz)

        df = df.copy()
        df["epsg3857_x"] = px
        df["epsg3857_y"] = py
        df["blender_x"] = bx
        df["blender_y"] = by
        df["ground_z_m"] = gz
        df["receiver_z_m"] = gz + rxh
        df["dem_hit"] = hit.astype(int)
        df["dem_object"] = "ground"
        df["coordinate_mode"] = coordinate_mode
        df["alignment_origin_source"] = origin_source
        df.to_csv(target, index=False, encoding="utf-8-sig")

        part = df.copy()
        part.insert(0, "source_row", np.arange(2, len(df) + 2))
        part.insert(0, "source_file", path.name)
        all_parts.append(part)
        reports.append(
            {
                "source_file": path.name,
                "row_count": int(len(df)),
                "valid_lonlat_count": int(valid.sum()),
                "dem_hit_count": int(hit.sum()),
                "dem_hit_rate": float(hit.mean()) if len(df) else 0.0,
                "local_x_range": _safe_range(bx),
                "local_y_range": _safe_range(by),
                "raw_layout": layout_info,
            }
        )
        print("[OK]", path.name, "->", target.name, "有效经纬度", int(valid.sum()), "/", len(df), "DEM命中", int(hit.sum()), "/", len(df))

    if all_parts:
        pd.concat(all_parts, ignore_index=True).to_csv(
            out / "all_measurements_blender_xyz.csv", index=False, encoding="utf-8-sig"
        )

    spatial_diagnostics = {}
    if all_valid_lat:
        lon0 = float(np.median(np.asarray(all_valid_lon, float)))
        lat0 = float(np.median(np.asarray(all_valid_lat, float)))
        scale = float(1.0 / max(np.cos(np.deg2rad(lat0)), 1e-12))
        ground_per_projected = float(1.0 / scale)
        utm = Transformer.from_crs(
            "EPSG:4326",
            cfg.get("recommended_metric_crs_for_external_distance_analysis", "EPSG:32648"),
            always_xy=True,
        )
        ux, uy = utm.transform(lon0, lat0)
        spatial_diagnostics = {
            "dataset_centroid_wgs84": [lon0, lat0],
            "web_mercator_local_scale_factor_approx": scale,
            "approx_ground_metres_per_projected_metre": ground_per_projected,
            "nominal_1m_grid_approx_ground_interval_m": ground_per_projected,
            "recommended_metric_crs_for_external_distance_analysis": cfg.get(
                "recommended_metric_crs_for_external_distance_analysis", "EPSG:32648"
            ),
            "centroid_in_recommended_metric_crs_m": [float(ux), float(uy)],
            "warning": (
                "EPSG:3857 is retained for internal consistency with the released Blender scene. "
                "Nominal 1 m spacing is not an exact geodetic ground-distance interval."
            ),
        }

    report = {
        "alignment_config": cfg,
        "preflight": preflight,
        "runtime_alignment": {
            "scene_origin_projected_x": ox,
            "scene_origin_projected_y": oy,
            "origin_source": origin_source,
            "coordinate_mode": coordinate_mode,
        },
        "ground_bounds": ground_bounds.tolist(),
        "spatial_reference_diagnostics": spatial_diagnostics,
        "files": reports,
    }
    (out / "alignment_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "spatial_reference_metadata.json").write_text(
        json.dumps(
            {
                "alignment_config": cfg,
                "runtime_alignment": report["runtime_alignment"],
                "diagnostics": spatial_diagnostics,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("完成，输出目录:", out)


if __name__ == "__main__":
    main()
