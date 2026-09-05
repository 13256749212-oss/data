#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.configuration import load_yaml, resolve_path
from src.scene_builder import build_scene_xml
from src.terrain import inspect_mesh


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    args = parser.parse_args()
    cfg = load_yaml(Path(args.config))
    ground = resolve_path(cfg, cfg["scene"]["ground_ply"])
    building = resolve_path(cfg, cfg["scene"]["building_ply"])
    xml_path = resolve_path(cfg, cfg["scene"]["generated_xml"])
    report = {
        "ground": inspect_mesh(ground),
        "building": inspect_mesh(building),
    }
    report["scene"] = build_scene_xml(
        ground,
        building,
        xml_path,
        str(cfg["scene"]["ground_material"]),
        str(cfg["scene"]["building_material"]),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
