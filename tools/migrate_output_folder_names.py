#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Safely remove numeric prefixes from legacy result-folder names.

The script never overwrites an existing file. When both old and new folders
exist, non-conflicting files are merged and conflicting files remain in the
old folder for manual review.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
MAPPINGS = {
    "04_parameter_calibration": "parameter_calibration",
    "11_joint_best_server_4000x3000": "joint_best_server_4000x3000",
    "12_joint_map_measurement_comparison": "joint_map_measurement_comparison",
}


def merge_directory(source: Path, target: Path, dry_run: bool) -> tuple[int, list[str]]:
    moved = 0
    conflicts: list[str] = []
    for item in sorted(source.rglob("*")):
        relative = item.relative_to(source)
        destination = target / relative
        if item.is_dir():
            if not dry_run:
                destination.mkdir(parents=True, exist_ok=True)
            continue
        if destination.exists():
            conflicts.append(str(relative))
            continue
        print(f"[MOVE] {item.relative_to(ROOT)} -> {destination.relative_to(ROOT)}")
        if not dry_run:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(item), str(destination))
        moved += 1
    if not dry_run:
        # Remove empty legacy directories only.
        for folder in sorted(source.rglob("*"), reverse=True):
            if folder.is_dir():
                try: folder.rmdir()
                except OSError: pass
        try: source.rmdir()
        except OSError: pass
    return moved, conflicts


def main() -> int:
    parser = argparse.ArgumentParser(description="移除outputs结果目录的数字前缀")
    parser.add_argument("--dry-run", action="store_true", help="仅显示将执行的操作")
    args = parser.parse_args()
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    total_moved = 0
    all_conflicts: list[str] = []
    for old_name, new_name in MAPPINGS.items():
        old = OUTPUTS / old_name
        new = OUTPUTS / new_name
        if not old.exists():
            print(f"[SKIP] 不存在: {old.relative_to(ROOT)}")
            continue
        if not new.exists():
            print(f"[RENAME] {old.relative_to(ROOT)} -> {new.relative_to(ROOT)}")
            if not args.dry_run:
                old.rename(new)
            continue
        print(f"[MERGE] 新旧目录同时存在: {old.relative_to(ROOT)} -> {new.relative_to(ROOT)}")
        moved, conflicts = merge_directory(old, new, args.dry_run)
        total_moved += moved
        all_conflicts.extend(f"{old_name}/{x}" for x in conflicts)
    if all_conflicts:
        print("\n[WARNING] 以下同名文件未覆盖，仍保留在旧目录:")
        for item in all_conflicts:
            print(" -", item)
        return 2
    print(f"\n完成。迁移文件数: {total_moved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
