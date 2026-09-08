from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Dict

from .terrain import inspect_mesh


def build_scene_xml(
    ground_ply: Path,
    building_ply: Path | None,
    output_xml: Path,
    ground_material: str = "itu_wet_ground",
    building_material: str = "itu_concrete",
) -> Dict[str, Any]:
    ground_ply = Path(ground_ply).expanduser().resolve()
    if not ground_ply.exists():
        raise FileNotFoundError(f"找不到ground.ply: {ground_ply}")
    output_xml = Path(output_xml).expanduser().resolve()
    output_xml.parent.mkdir(parents=True, exist_ok=True)

    ground_info = inspect_mesh(ground_ply)
    if ground_info.get("empty", True):
        raise ValueError(f"ground.ply为空或无效: {ground_info}")

    building_info = None
    include_buildings = False
    if building_ply is not None:
        building_ply = Path(building_ply).expanduser().resolve()
        building_info = inspect_mesh(building_ply)
        include_buildings = bool(
            building_info.get("exists") and not building_info.get("empty", True)
        )

    # Sionna recognizes built-in ITU materials through the mat-itu_* ID convention.
    lines = [
        '<?xml version="1.0" encoding="utf-8"?>',
        '<scene version="3.0.0">',
        f'  <bsdf type="diffuse" id="mat-{html.escape(ground_material)}" name="mat-{html.escape(ground_material)}">',
        '    <rgb name="reflectance" value="0.35,0.28,0.20"/>',
        '  </bsdf>',
        f'  <bsdf type="diffuse" id="mat-{html.escape(building_material)}" name="mat-{html.escape(building_material)}">',
        '    <rgb name="reflectance" value="0.65,0.65,0.65"/>',
        '  </bsdf>',
        '  <shape type="ply" id="mesh-ground" name="mesh-ground">',
        f'    <string name="filename" value="{html.escape(ground_ply.as_posix())}"/>',
        f'    <ref name="bsdf" id="mat-{html.escape(ground_material)}"/>',
        '  </shape>',
    ]
    if include_buildings and building_ply is not None:
        lines.extend(
            [
                '  <shape type="ply" id="mesh-buildings" name="mesh-buildings">',
                f'    <string name="filename" value="{html.escape(building_ply.as_posix())}"/>',
                f'    <ref name="bsdf" id="mat-{html.escape(building_material)}"/>',
                '  </shape>',
            ]
        )
    lines.append('</scene>')
    output_xml.write_text("\n".join(lines) + "\n", encoding="utf-8")

    report = {
        "generated_xml": str(output_xml),
        "ground": ground_info,
        "building": building_info,
        "building_included": include_buildings,
        "warning": (
            None
            if include_buildings
            else "建筑PLY为空或无有效三角面，本次场景只加载地形。请用有效的ynu_chenggong_campus.ply替换后重跑。"
        ),
    }
    report_path = output_xml.with_suffix(".diagnostics.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
