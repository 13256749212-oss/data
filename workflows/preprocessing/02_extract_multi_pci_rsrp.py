# -*- coding: utf-8 -*-
"""
extract_multi_pci_rsrp_from_blender_xyz.py

目标
----
从 Blender 已添加 XYZ/DEM 高度字段的完整 Cellular-Pro CSV 中，
逐行提取“同一采样时刻的多个 PCI 及其对应 RSRP”。

本脚本使用两组不同层级的列表字段：

A. 小区级列表（后续逐PCI仿真-RMSE比较的主数据）
    NR5G Cells PCI List
    NR5G Cells RSRP List
    NR5G Cells RSRQ List
    NR5G Cells SINR List
    NR5G Cells ARFCN List
    NR5G Cells SSB Index List
    NR5G Cells BeamNum List

   小区级列表通常在同一行中每个 PCI 只出现一次，
   PCI 与 RSRP 按分号分隔后的相同索引一一对应。

B. 波束级列表（单独输出，仅用于检查/后续波束分析）
    NR5G SSB Beam PCI List
    NR5G SSB Beam RSRP List
    NR5G SSB Beam RSRQ List
    NR5G SSB Beam SINR List
    NR5G SSB Beam Index List

   波束级列表中，同一 PCI 可能重复多次，因为不同 SSB Beam
   会分别给出 RSRP。不能把这些重复 PCI 当成不同扇区。

重要单位
--------
- RSRP：dBm（绝对接收功率）
- RSRQ：dB（功率比）
- SINR：dB（信号与干扰噪声功率比）
- 两个 dBm 数值相减后的 residual / delta：dB

本脚本不会：
- 划分训练集、验证集或测试集；
- 随机抽样；
- 对重复点求均值或中位数；
- 截断低于 -140 dBm 的原始 RSRP；
- 修改原始 RSRP；
- 运行 Sionna；
- 优化高度、方位角、下倾角或功率。

主要输出
--------
1. cell_pci_rsrp_long_all.csv
   所有小区级 PCI-RSRP 观测，一行一个 PCI。

2. cell_pci_rsrp_long_27stations.csv
   只保留 PCI 映射表中的27个研究基站/扇区。

3. ssb_beam_rsrp_long_all.csv
   波束级长表；同一 PCI 允许重复。

4. ssb_beam_best_per_pci.csv
   同一原始行、同一 PCI 的所有波束中，保留最强 RSRP，
   仅用于与 Cells 小区级列表交叉检查。

5. extraction_audit.csv
   列表长度不一致、无效坐标、无效PCI/RSRP等审计信息。

6. extraction_report.json
   文件数、原始行数、展开后的PCI观测数、单位和规则。

默认目录
--------
完整的12份文件应类似：
    *_with_blender_xyz.csv

默认放在：
    项目目录/data/aligned_measurements/

PCI映射文件默认放在：
    项目目录/data/aligned_measurements/base_station_pci_mapping(10).csv

运行
----
直接运行：
    python extract_multi_pci_rsrp_from_blender_xyz.py

或显式指定：
    python extract_multi_pci_rsrp_from_blender_xyz.py ^
      --input-dir "data\\aligned_measurements" ^
      --mapping "config\\base_station_pci_mapping.csv" ^
      --output-dir "data\\processed\\extracted"

依赖
----
仅使用 Python 标准库，不需要 pandas。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


# =============================================================================
# 1. 用户配置
# =============================================================================

# 所有输入和输出均位于本代码包内部。
PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = PACKAGE_ROOT / "data" / "aligned_measurements"
DEFAULT_MAPPING_CSV = PACKAGE_ROOT / "config" / "base_station_pci_mapping.csv"
DEFAULT_OUTPUT_DIR = PACKAGE_ROOT / "data" / "processed" / "extracted"

INPUT_GLOB = "*_with_blender_xyz.csv"

# 是否要求 DEM 命中。
REQUIRE_DEM_HIT = True

# 是否输出波束级列表。
WRITE_BEAM_LEVEL_OUTPUT = True

# 当 Cells 列表为空时，是否退回使用单值服务小区字段。
# 这只是防止少数行列表缺失，不会在 Cells 列表正常时重复添加服务小区。
USE_SERVING_FALLBACK_WHEN_CELL_LIST_EMPTY = True

# 对 PCI List 与 RSRP List：
# True = 数量不一致时整行不展开，并写入 audit；
# False = 只取两者共同的最短长度。
STRICT_CELL_PCI_RSRP_LENGTH = True

# 不按 -140 dBm 截断。这里只做非常宽松的物理合理性标记，
# 不会因为超出范围删除原始值。
PLAUSIBLE_RSRP_MIN_DBM = -200.0
PLAUSIBLE_RSRP_MAX_DBM = 0.0

# 接收点ID的小数位；仅生成ID，不修改原始坐标。
RX_ID_DECIMALS = 3

# CSV可能含很长的列表字段。
csv.field_size_limit(min(sys.maxsize, 2_000_000_000))


# =============================================================================
# 2. 27个基站的初始位置
#    用于给展开结果附加基站坐标和初始几何量；
#    本脚本不会调整这些参数。
# =============================================================================

SELECTED_STATIONS = [
    {"station_id":2,  "name":"tx-2",  "label":"图书馆-东北角",       "position":[-62.7873,-48.317,2009.28]},
    {"station_id":3,  "name":"tx-3",  "label":"农学院",             "position":[220.957,-22.6014,2013.2]},
    {"station_id":18, "name":"tx-18", "label":"食堂-西",             "position":[253.779,-328.187,2009.19]},
    {"station_id":41, "name":"tx-41", "label":"楠苑宿舍楼-西",       "position":[79.0692,-442.067,2005.64]},
    {"station_id":7,  "name":"tx-7",  "label":"图书馆-西南角",       "position":[-380.963,-177.432,1990.21]},
    {"station_id":6,  "name":"tx-6",  "label":"桦苑-西",             "position":[-417.217,-420.524,1994.57]},
    {"station_id":9,  "name":"tx-9",  "label":"诸子百家-西",         "position":[-731.812,76.9981,1989.89]},
    {"station_id":23, "name":"tx-23", "label":"梓苑食堂-南",         "position":[-708.457,420.289,1989.54]},
    {"station_id":26, "name":"tx-26", "label":"楠苑-后大路",         "position":[-601.935,-426.487,1982.05]},
    {"station_id":16, "name":"tx-16", "label":"楸苑",               "position":[540.712,-230.001,2005.25]},
    {"station_id":15, "name":"tx-15", "label":"楸苑7宿舍-西",        "position":[856.239,-122.77,2019.91]},
    {"station_id":14, "name":"tx-14", "label":"校医院",             "position":[916.187,213.785,2010.49]},
    {"station_id":20, "name":"tx-20", "label":"软件学院",           "position":[512.655,144.529,2028.89]},
    {"station_id":13, "name":"tx-13", "label":"国重大楼",           "position":[436.867,428.861,2025.31]},
    {"station_id":12, "name":"tx-12", "label":"明远楼",             "position":[85.5687,503.753,2020.15]},
    {"station_id":33, "name":"tx-33", "label":"北门东侧马路-西",     "position":[-5.83533,902.896,1978.67]},
    {"station_id":31, "name":"tx-31", "label":"北门西侧马路-对面",   "position":[-417.807,1049.26,1978.78]},
    {"station_id":35, "name":"tx-35", "label":"北门东侧马路-偏国重", "position":[366.611,761.907,1987.78]},
    {"station_id":37, "name":"tx-37", "label":"学校东北角-南",       "position":[1105.33,678.84,2002.88]},
    {"station_id":38, "name":"tx-38", "label":"学校东门对面",       "position":[1447.05,384.09,1978.78]},
    {"station_id":39, "name":"tx-39", "label":"学校东南角-东",       "position":[1794.33,24.778,2023.38]},
    {"station_id":27, "name":"tx-27", "label":"学校西北角",         "position":[-1284.87,548.908,1987.37]},
    {"station_id":10, "name":"tx-10", "label":"梓苑食堂后面",       "position":[-972.554,342.474,1987.55]},
    {"station_id":22, "name":"tx-22", "label":"化工学院-西",         "position":[688.004,250.823,2029.42]},
    {"station_id":11, "name":"tx-11", "label":"校史馆",             "position":[-382.4,627.376,1986.34]},
    {"station_id":25, "name":"tx-25", "label":"西南角",             "position":[-405.485,-944.095,1986.02]},
    {"station_id":30, "name":"tx-30", "label":"北门地铁马路-西",     "position":[-833.997,875.923,1976.43]},
]

STATION_BY_ID = {
    int(station["station_id"]): station
    for station in SELECTED_STATIONS
}

if len(STATION_BY_ID) != 27:
    raise RuntimeError(
        f"SELECTED_STATIONS数量应为27，当前为{len(STATION_BY_ID)}。"
    )


# =============================================================================
# 3. 字段定义
# =============================================================================

REQUIRED_INPUT_COLUMNS = [
    "TIME",
    "LATITUDE",
    "LONGITUDE",
    "NR5G PCI",
    "NR5G SS RSRP",
    "NR5G Cells PCI List",
    "NR5G Cells RSRP List",
    "blender_x",
    "blender_y",
    "ground_z_m",
    "receiver_z_m",
    "dem_hit",
]

CELL_LIST_COLUMNS = {
    "cell_pci": "NR5G Cells PCI List",
    "cell_rsrp_dbm": "NR5G Cells RSRP List",
    "cell_rsrq_db": "NR5G Cells RSRQ List",
    "cell_sinr_db": "NR5G Cells SINR List",
    "cell_ssb_arfcn": "NR5G Cells ARFCN List",
    "cell_ssb_index": "NR5G Cells SSB Index List",
    "cell_beam_num": "NR5G Cells BeamNum List",
    "cell_serving_type": "NR5G ServingType List",
}

BEAM_LIST_COLUMNS = {
    "beam_index": "NR5G SSB Beam Index List",
    "beam_pci": "NR5G SSB Beam PCI List",
    "beam_rsrp_dbm": "NR5G SSB Beam RSRP List",
    "beam_rsrq_db": "NR5G SSB Beam RSRQ List",
    "beam_sinr_db": "NR5G SSB Beam SINR List",
}

ROW_FIELDS_TO_KEEP = {
    "time": "TIME",
    "latitude": "LATITUDE",
    "longitude": "LONGITUDE",
    "speed_mps": "SPEED(M/s)",
    "original_altitude_m": "ALT(M)",
    "accuracy_m": "ACCURACY(M)",
    "network_type": "NETWORK_TYPE",
    "operator": "OPERATOR",
    "nr5g_nci": "NR5G NCI",
    "nr5g_gnodeb_id": "NR5G gNodeB ID",
    "nr5g_cell_id": "NR5G Cell ID",
    "nr5g_band": "NR5G Band",
    "nr5g_bandwidth_dl_mhz": "NR5G BandWidth DL",
    "nr5g_ssb_arfcn_dl": "NR5G SSB ARFCN DL",
    "nr5g_center_arfcn_dl": "NR5G Center ARFCN DL",
    "serving_pci": "NR5G PCI",
    "serving_ss_rsrp_dbm": "NR5G SS RSRP",
    "serving_ss_rsrq_db": "NR5G SS RSRQ",
    "serving_ss_sinr_db": "NR5G SS SINR",
    "blender_x": "blender_x",
    "blender_y": "blender_y",
    "ground_z_m": "ground_z_m",
    "receiver_z_m": "receiver_z_m",
    "dem_hit": "dem_hit",
    "dem_object": "dem_object",
}


CELL_OUTPUT_FIELDS = [
    "measurement_id",
    "rx_point_id",
    "source_file",
    "source_row",
    "csv_physical_row",
    "time",
    "latitude",
    "longitude",
    "speed_mps",
    "original_altitude_m",
    "accuracy_m",
    "network_type",
    "operator",
    "nr5g_nci",
    "nr5g_gnodeb_id",
    "nr5g_cell_id",
    "nr5g_band",
    "nr5g_bandwidth_dl_mhz",
    "nr5g_ssb_arfcn_dl",
    "nr5g_center_arfcn_dl",
    "blender_x",
    "blender_y",
    "ground_z_m",
    "receiver_z_m",
    "receiver_height_agl_m",
    "dem_hit",
    "dem_object",
    "cell_list_index",
    "pci",
    "measured_rsrp_dbm",
    "rsrp_unit",
    "measured_rsrq_db",
    "rsrq_unit",
    "measured_sinr_db",
    "sinr_unit",
    "cell_ssb_arfcn",
    "cell_ssb_index",
    "cell_beam_num",
    "cell_serving_type",
    "measurement_source",
    "serving_pci",
    "serving_ss_rsrp_dbm",
    "is_serving_pci",
    "serving_minus_cell_rsrp_db",
    "rsrp_plausible_flag",
    "is_target_27station_pci",
    "station_id",
    "station_name",
    "station_label",
    "sector_index",
    "antenna_type",
    "is_omnidirectional",
    "mapping_source",
    "tx_x_initial_m",
    "tx_y_initial_m",
    "tx_z_initial_m",
    "horizontal_distance_initial_m",
    "distance_3d_initial_m",
    "bearing_from_tx_math_deg",
    "bearing_from_tx_compass_deg",
    "geometric_downtilt_to_rx_deg",
]

BEAM_OUTPUT_FIELDS = [
    "beam_measurement_id",
    "rx_point_id",
    "source_file",
    "source_row",
    "csv_physical_row",
    "time",
    "latitude",
    "longitude",
    "blender_x",
    "blender_y",
    "ground_z_m",
    "receiver_z_m",
    "beam_list_index",
    "pci",
    "beam_index",
    "beam_rsrp_dbm",
    "rsrp_unit",
    "beam_rsrq_db",
    "rsrq_unit",
    "beam_sinr_db",
    "sinr_unit",
    "is_target_27station_pci",
    "station_id",
    "sector_index",
    "antenna_type",
]

AUDIT_FIELDS = [
    "source_file",
    "source_row",
    "csv_physical_row",
    "time",
    "audit_type",
    "detail",
    "cells_pci_count",
    "cells_rsrp_count",
    "cells_rsrq_count",
    "cells_sinr_count",
    "cells_arfcn_count",
    "cells_ssb_index_count",
    "cells_beam_num_count",
    "beam_pci_count",
    "beam_rsrp_count",
]


# =============================================================================
# 4. 基础工具
# =============================================================================

MISSING_TEXT = {
    "",
    "-",
    "--",
    "nan",
    "NaN",
    "NAN",
    "None",
    "NULL",
    "null",
}


def clean_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def is_missing(value: object) -> bool:
    return clean_text(value) in MISSING_TEXT


def safe_float(value: object) -> Optional[float]:
    text = clean_text(value)
    if text in MISSING_TEXT:
        return None

    try:
        number = float(text)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(number):
        return None

    return number


def safe_int(value: object) -> Optional[int]:
    number = safe_float(value)

    if number is None:
        return None

    rounded = round(number)

    if abs(number - rounded) > 1.0e-6:
        return None

    return int(rounded)


def parse_bool(value: object) -> bool:
    return clean_text(value).lower() in {
        "true",
        "1",
        "yes",
        "y",
    }


def split_semicolon_list(value: object) -> List[str]:
    text = clean_text(value)

    if text in MISSING_TEXT:
        return []

    return [
        item.strip()
        for item in text.split(";")
    ]


def list_value(
    values: Sequence[str],
    index: int,
) -> str:
    if 0 <= index < len(values):
        return values[index]
    return ""


def stable_id(prefix: str, text: str) -> str:
    digest = hashlib.sha1(
        text.encode("utf-8")
    ).hexdigest()[:16]
    return f"{prefix}_{digest}"


def make_rx_point_id(
    x: float,
    y: float,
    z: float,
) -> str:
    text = (
        f"{x:.{RX_ID_DECIMALS}f}|"
        f"{y:.{RX_ID_DECIMALS}f}|"
        f"{z:.{RX_ID_DECIMALS}f}"
    )
    return stable_id("rx", text)


def fix_csv_row(
    row: List[str],
    header_length: int,
) -> Tuple[List[str], str]:
    if (
        len(row) == header_length + 1
        and row[-1] == ""
    ):
        return row[:-1], "trimmed_trailing_empty"

    if len(row) > header_length:
        return row[:header_length], "trimmed_extra_columns"

    if len(row) < header_length:
        return (
            row + [""] * (header_length - len(row)),
            "padded_missing_columns",
        )

    return row, "ok"


def read_csv_fixed(
    path: Path,
) -> Iterable[
    Tuple[int, Dict[str, str], str]
]:
    """
    返回：
        csv物理行号,
        字段字典,
        行长修复状态

    表头为物理第1行，首条数据是物理第2行。
    """
    with path.open(
        "r",
        encoding="utf-8-sig",
        errors="replace",
        newline="",
    ) as file:
        reader = csv.reader(file)

        try:
            header = next(reader)
        except StopIteration:
            return

        header = [
            clean_text(column)
            for column in header
        ]

        if len(set(header)) != len(header):
            duplicates = [
                name
                for name, count
                in Counter(header).items()
                if count > 1
            ]
            raise RuntimeError(
                f"{path.name}存在重复表头：{duplicates}"
            )

        for physical_row, raw_row in enumerate(
            reader,
            start=2,
        ):
            row, status = fix_csv_row(
                raw_row,
                len(header),
            )
            yield (
                physical_row,
                dict(zip(header, row)),
                status,
            )


def extract_row_common(
    row: Mapping[str, str],
) -> Dict[str, object]:
    output: Dict[str, object] = {}

    numeric_float_fields = {
        "latitude",
        "longitude",
        "speed_mps",
        "original_altitude_m",
        "accuracy_m",
        "serving_ss_rsrp_dbm",
        "serving_ss_rsrq_db",
        "serving_ss_sinr_db",
        "blender_x",
        "blender_y",
        "ground_z_m",
        "receiver_z_m",
    }

    numeric_int_fields = {
        "nr5g_nci",
        "nr5g_gnodeb_id",
        "nr5g_cell_id",
        "nr5g_band",
        "nr5g_ssb_arfcn_dl",
        "nr5g_center_arfcn_dl",
        "serving_pci",
        "dem_hit",
    }

    for output_name, source_name in ROW_FIELDS_TO_KEEP.items():
        value = row.get(source_name, "")

        if output_name in numeric_float_fields:
            output[output_name] = safe_float(value)
        elif output_name in numeric_int_fields:
            output[output_name] = safe_int(value)
        elif output_name == "nr5g_bandwidth_dl_mhz":
            output[output_name] = safe_float(value)
        else:
            output[output_name] = clean_text(value)

    ground_z = output.get("ground_z_m")
    receiver_z = output.get("receiver_z_m")

    if (
        isinstance(ground_z, float)
        and isinstance(receiver_z, float)
    ):
        output["receiver_height_agl_m"] = (
            receiver_z - ground_z
        )
    else:
        output["receiver_height_agl_m"] = None

    return output


def valid_spatial_row(
    common: Mapping[str, object],
) -> Tuple[bool, str]:
    lat = common.get("latitude")
    lon = common.get("longitude")
    x = common.get("blender_x")
    y = common.get("blender_y")
    ground_z = common.get("ground_z_m")
    receiver_z = common.get("receiver_z_m")
    dem_hit = common.get("dem_hit")

    required = [
        lat,
        lon,
        x,
        y,
        ground_z,
        receiver_z,
    ]

    if any(value is None for value in required):
        return False, "missing_spatial_coordinate_or_height"

    assert isinstance(lat, float)
    assert isinstance(lon, float)

    if (
        abs(lat) < 1.0e-12
        and abs(lon) < 1.0e-12
    ):
        return False, "zero_longitude_latitude"

    if REQUIRE_DEM_HIT and dem_hit != 1:
        return False, "dem_not_hit"

    return True, ""


def load_mapping(
    path: Path,
) -> Dict[int, Dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(
            f"PCI映射文件不存在：{path}"
        )

    mapping: Dict[int, Dict[str, object]] = {}

    with path.open(
        "r",
        encoding="utf-8-sig",
        errors="replace",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        required = {
            "station_id",
            "pci",
            "sector_index",
            "antenna_type",
            "is_omnidirectional",
            "is_research_target",
        }

        missing = required - set(
            reader.fieldnames or []
        )

        if missing:
            raise KeyError(
                f"PCI映射缺少字段：{sorted(missing)}"
            )

        for row in reader:
            if not parse_bool(
                row.get(
                    "is_research_target",
                    "",
                )
            ):
                continue

            station_id = safe_int(
                row.get("station_id")
            )
            pci = safe_int(
                row.get("pci")
            )
            sector_index = safe_int(
                row.get("sector_index")
            )

            if (
                station_id is None
                or pci is None
                or sector_index is None
            ):
                raise RuntimeError(
                    f"PCI映射存在无效记录：{row}"
                )

            if station_id not in STATION_BY_ID:
                raise RuntimeError(
                    f"映射中station_id={station_id}"
                    "不属于当前27站。"
                )

            if pci in mapping:
                raise RuntimeError(
                    f"PCI {pci} 在映射表中重复。"
                )

            mapping[pci] = {
                "station_id": station_id,
                "sector_index": sector_index,
                "antenna_type": clean_text(
                    row.get("antenna_type")
                ),
                "is_omnidirectional": int(
                    parse_bool(
                        row.get(
                            "is_omnidirectional",
                            "",
                        )
                    )
                ),
                "mapping_source": clean_text(
                    row.get("mapping_source")
                ),
            }

    station_ids = {
        int(info["station_id"])
        for info in mapping.values()
    }

    missing_station_ids = (
        set(STATION_BY_ID)
        - station_ids
    )

    if missing_station_ids:
        raise RuntimeError(
            "映射表缺少27站中的站号："
            f"{sorted(missing_station_ids)}"
        )

    print(
        f"读取PCI映射："
        f"{len(station_ids)}个物理站，"
        f"{len(mapping)}个PCI。"
    )

    return mapping


def add_station_and_geometry(
    record: Dict[str, object],
    mapping_info: Optional[
        Mapping[str, object]
    ],
) -> None:
    if mapping_info is None:
        record.update({
            "is_target_27station_pci": 0,
            "station_id": "",
            "station_name": "",
            "station_label": "",
            "sector_index": "",
            "antenna_type": "",
            "is_omnidirectional": "",
            "mapping_source": "",
            "tx_x_initial_m": "",
            "tx_y_initial_m": "",
            "tx_z_initial_m": "",
            "horizontal_distance_initial_m": "",
            "distance_3d_initial_m": "",
            "bearing_from_tx_math_deg": "",
            "bearing_from_tx_compass_deg": "",
            "geometric_downtilt_to_rx_deg": "",
        })
        return

    station_id = int(
        mapping_info["station_id"]
    )
    station = STATION_BY_ID[station_id]

    tx_x, tx_y, tx_z = (
        float(station["position"][0]),
        float(station["position"][1]),
        float(station["position"][2]),
    )

    rx_x = record.get("blender_x")
    rx_y = record.get("blender_y")
    rx_z = record.get("receiver_z_m")

    record.update({
        "is_target_27station_pci": 1,
        "station_id": station_id,
        "station_name": station["name"],
        "station_label": station["label"],
        "sector_index": int(
            mapping_info["sector_index"]
        ),
        "antenna_type": mapping_info[
            "antenna_type"
        ],
        "is_omnidirectional": mapping_info[
            "is_omnidirectional"
        ],
        "mapping_source": mapping_info[
            "mapping_source"
        ],
        "tx_x_initial_m": tx_x,
        "tx_y_initial_m": tx_y,
        "tx_z_initial_m": tx_z,
    })

    if not all(
        isinstance(value, float)
        for value in (rx_x, rx_y, rx_z)
    ):
        record.update({
            "horizontal_distance_initial_m": "",
            "distance_3d_initial_m": "",
            "bearing_from_tx_math_deg": "",
            "bearing_from_tx_compass_deg": "",
            "geometric_downtilt_to_rx_deg": "",
        })
        return

    assert isinstance(rx_x, float)
    assert isinstance(rx_y, float)
    assert isinstance(rx_z, float)

    dx = rx_x - tx_x
    dy = rx_y - tx_y
    dz = rx_z - tx_z

    horizontal = math.hypot(dx, dy)
    distance_3d = math.sqrt(
        dx * dx
        + dy * dy
        + dz * dz
    )

    bearing_math = (
        math.degrees(
            math.atan2(dy, dx)
        )
        + 360.0
    ) % 360.0

    bearing_compass = (
        math.degrees(
            math.atan2(dx, dy)
        )
        + 360.0
    ) % 360.0

    geometric_downtilt = math.degrees(
        math.atan2(
            tx_z - rx_z,
            max(horizontal, 1.0e-9),
        )
    )

    record.update({
        "horizontal_distance_initial_m": horizontal,
        "distance_3d_initial_m": distance_3d,
        "bearing_from_tx_math_deg": bearing_math,
        "bearing_from_tx_compass_deg": bearing_compass,
        "geometric_downtilt_to_rx_deg": geometric_downtilt,
    })


def write_csv_header(
    file,
    fieldnames: Sequence[str],
) -> csv.DictWriter:
    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames,
        extrasaction="ignore",
    )
    writer.writeheader()
    return writer


def make_audit_row(
    source_file: str,
    source_row: object,
    physical_row: int,
    time_value: object,
    audit_type: str,
    detail: str,
    cell_lists: Optional[
        Mapping[str, Sequence[str]]
    ] = None,
    beam_lists: Optional[
        Mapping[str, Sequence[str]]
    ] = None,
) -> Dict[str, object]:
    cell_lists = cell_lists or {}
    beam_lists = beam_lists or {}

    return {
        "source_file": source_file,
        "source_row": source_row,
        "csv_physical_row": physical_row,
        "time": clean_text(time_value),
        "audit_type": audit_type,
        "detail": detail,
        "cells_pci_count": len(
            cell_lists.get("cell_pci", [])
        ),
        "cells_rsrp_count": len(
            cell_lists.get(
                "cell_rsrp_dbm",
                [],
            )
        ),
        "cells_rsrq_count": len(
            cell_lists.get(
                "cell_rsrq_db",
                [],
            )
        ),
        "cells_sinr_count": len(
            cell_lists.get(
                "cell_sinr_db",
                [],
            )
        ),
        "cells_arfcn_count": len(
            cell_lists.get(
                "cell_ssb_arfcn",
                [],
            )
        ),
        "cells_ssb_index_count": len(
            cell_lists.get(
                "cell_ssb_index",
                [],
            )
        ),
        "cells_beam_num_count": len(
            cell_lists.get(
                "cell_beam_num",
                [],
            )
        ),
        "beam_pci_count": len(
            beam_lists.get(
                "beam_pci",
                [],
            )
        ),
        "beam_rsrp_count": len(
            beam_lists.get(
                "beam_rsrp_dbm",
                [],
            )
        ),
    }


# =============================================================================
# 5. 主处理
# =============================================================================

def process_files(
    input_paths: Sequence[Path],
    mapping: Mapping[
        int,
        Mapping[str, object],
    ],
    output_dir: Path,
) -> Dict[str, object]:
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_cell_path = (
        output_dir
        / "cell_pci_rsrp_long_all.csv"
    )
    target_cell_path = (
        output_dir
        / "cell_pci_rsrp_long_27stations.csv"
    )
    beam_path = (
        output_dir
        / "ssb_beam_rsrp_long_all.csv"
    )
    beam_best_path = (
        output_dir
        / "ssb_beam_best_per_pci.csv"
    )
    audit_path = (
        output_dir
        / "extraction_audit.csv"
    )
    report_path = (
        output_dir
        / "extraction_report.json"
    )

    stats: Counter = Counter()
    per_file_stats: List[
        Dict[str, object]
    ] = []
    target_pci_counts: Counter = Counter()
    all_pci_counts: Counter = Counter()
    audit_type_counts: Counter = Counter()

    with (
        all_cell_path.open(
            "w",
            encoding="utf-8-sig",
            newline="",
        ) as all_cell_file,
        target_cell_path.open(
            "w",
            encoding="utf-8-sig",
            newline="",
        ) as target_cell_file,
        audit_path.open(
            "w",
            encoding="utf-8-sig",
            newline="",
        ) as audit_file,
        beam_path.open(
            "w",
            encoding="utf-8-sig",
            newline="",
        ) as beam_file,
        beam_best_path.open(
            "w",
            encoding="utf-8-sig",
            newline="",
        ) as beam_best_file,
    ):
        all_cell_writer = write_csv_header(
            all_cell_file,
            CELL_OUTPUT_FIELDS,
        )
        target_cell_writer = write_csv_header(
            target_cell_file,
            CELL_OUTPUT_FIELDS,
        )
        audit_writer = write_csv_header(
            audit_file,
            AUDIT_FIELDS,
        )
        beam_writer = write_csv_header(
            beam_file,
            BEAM_OUTPUT_FIELDS,
        )
        beam_best_writer = write_csv_header(
            beam_best_file,
            BEAM_OUTPUT_FIELDS,
        )

        for file_index, path in enumerate(
            input_paths,
            start=1,
        ):
            print(
                f"[{file_index:02d}/"
                f"{len(input_paths):02d}] "
                f"{path.name}"
            )

            file_stats: Counter = Counter()
            header_checked = False

            for (
                physical_row,
                row,
                row_fix_status,
            ) in read_csv_fixed(path):
                stats["raw_csv_rows"] += 1
                file_stats["raw_csv_rows"] += 1

                if not header_checked:
                    missing_columns = [
                        column
                        for column
                        in REQUIRED_INPUT_COLUMNS
                        if column not in row
                    ]

                    if missing_columns:
                        raise KeyError(
                            f"{path.name}缺少多PCI提取字段："
                            f"{missing_columns}\n"
                            "请使用完整的"
                            "*_with_blender_xyz.csv，"
                            "不要使用只保留单值字段的"
                            "all_measurements_blender_xyz.csv。"
                        )

                    header_checked = True

                source_file = clean_text(
                    row.get("source_file")
                ) or path.name

                source_row = (
                    safe_int(
                        row.get("source_row")
                    )
                    or physical_row
                )

                time_value = row.get("TIME", "")

                if row_fix_status != "ok":
                    audit = make_audit_row(
                        source_file,
                        source_row,
                        physical_row,
                        time_value,
                        "csv_row_length_fixed",
                        row_fix_status,
                    )
                    audit_writer.writerow(audit)
                    audit_type_counts[
                        "csv_row_length_fixed"
                    ] += 1

                common = extract_row_common(row)

                spatial_ok, spatial_reason = (
                    valid_spatial_row(common)
                )

                if not spatial_ok:
                    stats[
                        "rows_rejected_spatial"
                    ] += 1
                    file_stats[
                        "rows_rejected_spatial"
                    ] += 1

                    audit = make_audit_row(
                        source_file,
                        source_row,
                        physical_row,
                        time_value,
                        "row_rejected_spatial",
                        spatial_reason,
                    )
                    audit_writer.writerow(audit)
                    audit_type_counts[
                        "row_rejected_spatial"
                    ] += 1
                    continue

                stats["valid_spatial_rows"] += 1
                file_stats[
                    "valid_spatial_rows"
                ] += 1

                rx_x = common["blender_x"]
                rx_y = common["blender_y"]
                rx_z = common["receiver_z_m"]

                assert isinstance(rx_x, float)
                assert isinstance(rx_y, float)
                assert isinstance(rx_z, float)

                rx_point_id = make_rx_point_id(
                    rx_x,
                    rx_y,
                    rx_z,
                )

                # -------------------------------------------------------------
                # 5.1 小区级列表
                # -------------------------------------------------------------

                cell_lists = {
                    output_name:
                    split_semicolon_list(
                        row.get(source_name)
                    )
                    for (
                        output_name,
                        source_name,
                    ) in CELL_LIST_COLUMNS.items()
                }

                cell_pcis = cell_lists[
                    "cell_pci"
                ]
                cell_rsrps = cell_lists[
                    "cell_rsrp_dbm"
                ]

                if len(cell_pcis) != len(
                    cell_rsrps
                ):
                    audit = make_audit_row(
                        source_file,
                        source_row,
                        physical_row,
                        time_value,
                        "cell_pci_rsrp_length_mismatch",
                        (
                            f"PCI数量={len(cell_pcis)}, "
                            f"RSRP数量={len(cell_rsrps)}"
                        ),
                        cell_lists=cell_lists,
                    )
                    audit_writer.writerow(audit)
                    audit_type_counts[
                        "cell_pci_rsrp_length_mismatch"
                    ] += 1
                    stats[
                        "rows_cell_pci_rsrp_mismatch"
                    ] += 1

                    if STRICT_CELL_PCI_RSRP_LENGTH:
                        cell_pair_count = 0
                    else:
                        cell_pair_count = min(
                            len(cell_pcis),
                            len(cell_rsrps),
                        )
                else:
                    cell_pair_count = len(
                        cell_pcis
                    )

                # 可选列表长度不同只记录，不错位前面的对应项。
                optional_cell_lengths = {
                    key: len(values)
                    for key, values
                    in cell_lists.items()
                    if key not in {
                        "cell_pci",
                        "cell_rsrp_dbm",
                    }
                }

                mismatched_optional = {
                    key: length
                    for key, length
                    in optional_cell_lengths.items()
                    if (
                        length not in {
                            0,
                            cell_pair_count,
                        }
                    )
                }

                if mismatched_optional:
                    audit = make_audit_row(
                        source_file,
                        source_row,
                        physical_row,
                        time_value,
                        "optional_cell_list_length_mismatch",
                        json.dumps(
                            mismatched_optional,
                            ensure_ascii=False,
                        ),
                        cell_lists=cell_lists,
                    )
                    audit_writer.writerow(audit)
                    audit_type_counts[
                        "optional_cell_list_length_mismatch"
                    ] += 1

                # Cells列表为空时，单值服务小区作为fallback。
                use_serving_fallback = (
                    cell_pair_count == 0
                    and USE_SERVING_FALLBACK_WHEN_CELL_LIST_EMPTY
                    and common.get(
                        "serving_pci"
                    ) is not None
                    and common.get(
                        "serving_ss_rsrp_dbm"
                    ) is not None
                )

                if use_serving_fallback:
                    cell_pair_count = 1
                    cell_lists = {
                        "cell_pci": [
                            str(
                                common[
                                    "serving_pci"
                                ]
                            )
                        ],
                        "cell_rsrp_dbm": [
                            str(
                                common[
                                    "serving_ss_rsrp_dbm"
                                ]
                            )
                        ],
                        "cell_rsrq_db": [
                            str(
                                common.get(
                                    "serving_ss_rsrq_db",
                                    "",
                                )
                            )
                        ],
                        "cell_sinr_db": [
                            str(
                                common.get(
                                    "serving_ss_sinr_db",
                                    "",
                                )
                            )
                        ],
                        "cell_ssb_arfcn": [
                            str(
                                common.get(
                                    "nr5g_ssb_arfcn_dl",
                                    "",
                                )
                            )
                        ],
                        "cell_ssb_index": [""],
                        "cell_beam_num": [""],
                        "cell_serving_type": [
                            "serving_fallback"
                        ],
                    }

                    stats[
                        "rows_using_serving_fallback"
                    ] += 1

                if cell_pair_count == 0:
                    stats[
                        "rows_without_cell_pairs"
                    ] += 1
                    file_stats[
                        "rows_without_cell_pairs"
                    ] += 1

                row_seen_pcis: Counter = Counter()

                for list_index in range(
                    cell_pair_count
                ):
                    pci = safe_int(
                        list_value(
                            cell_lists[
                                "cell_pci"
                            ],
                            list_index,
                        )
                    )
                    rsrp_dbm = safe_float(
                        list_value(
                            cell_lists[
                                "cell_rsrp_dbm"
                            ],
                            list_index,
                        )
                    )

                    if (
                        pci is None
                        or rsrp_dbm is None
                    ):
                        audit = make_audit_row(
                            source_file,
                            source_row,
                            physical_row,
                            time_value,
                            "invalid_cell_pci_or_rsrp",
                            (
                                f"index={list_index}, "
                                f"pci={list_value(cell_lists['cell_pci'], list_index)!r}, "
                                f"rsrp={list_value(cell_lists['cell_rsrp_dbm'], list_index)!r}"
                            ),
                            cell_lists=cell_lists,
                        )
                        audit_writer.writerow(
                            audit
                        )
                        audit_type_counts[
                            "invalid_cell_pci_or_rsrp"
                        ] += 1
                        stats[
                            "invalid_cell_pairs"
                        ] += 1
                        continue

                    row_seen_pcis[pci] += 1

                    mapping_info = mapping.get(pci)

                    record: Dict[
                        str,
                        object,
                    ] = dict(common)

                    measurement_id = stable_id(
                        "cell",
                        (
                            f"{source_file}|"
                            f"{source_row}|"
                            f"{list_index}|"
                            f"{pci}"
                        ),
                    )

                    serving_pci = common.get(
                        "serving_pci"
                    )
                    serving_rsrp = common.get(
                        "serving_ss_rsrp_dbm"
                    )

                    is_serving = int(
                        isinstance(
                            serving_pci,
                            int,
                        )
                        and serving_pci == pci
                    )

                    if (
                        is_serving
                        and isinstance(
                            serving_rsrp,
                            float,
                        )
                    ):
                        serving_minus_cell = (
                            serving_rsrp
                            - rsrp_dbm
                        )
                    else:
                        serving_minus_cell = None

                    record.update({
                        "measurement_id": measurement_id,
                        "rx_point_id": rx_point_id,
                        "source_file": source_file,
                        "source_row": source_row,
                        "csv_physical_row": physical_row,
                        "cell_list_index": list_index,
                        "pci": pci,
                        "measured_rsrp_dbm": rsrp_dbm,
                        "rsrp_unit": "dBm",
                        "measured_rsrq_db": safe_float(
                            list_value(
                                cell_lists[
                                    "cell_rsrq_db"
                                ],
                                list_index,
                            )
                        ),
                        "rsrq_unit": "dB",
                        "measured_sinr_db": safe_float(
                            list_value(
                                cell_lists[
                                    "cell_sinr_db"
                                ],
                                list_index,
                            )
                        ),
                        "sinr_unit": "dB",
                        "cell_ssb_arfcn": safe_int(
                            list_value(
                                cell_lists[
                                    "cell_ssb_arfcn"
                                ],
                                list_index,
                            )
                        ),
                        "cell_ssb_index": safe_int(
                            list_value(
                                cell_lists[
                                    "cell_ssb_index"
                                ],
                                list_index,
                            )
                        ),
                        "cell_beam_num": safe_int(
                            list_value(
                                cell_lists[
                                    "cell_beam_num"
                                ],
                                list_index,
                            )
                        ),
                        "cell_serving_type": clean_text(
                            list_value(
                                cell_lists[
                                    "cell_serving_type"
                                ],
                                list_index,
                            )
                        ),
                        "measurement_source": (
                            "serving_fallback"
                            if use_serving_fallback
                            else "NR5G Cells PCI/RSRP List"
                        ),
                        "is_serving_pci": is_serving,
                        "serving_minus_cell_rsrp_db": serving_minus_cell,
                        "rsrp_plausible_flag": int(
                            PLAUSIBLE_RSRP_MIN_DBM
                            <= rsrp_dbm
                            <= PLAUSIBLE_RSRP_MAX_DBM
                        ),
                    })

                    add_station_and_geometry(
                        record,
                        mapping_info,
                    )

                    all_cell_writer.writerow(
                        record
                    )
                    stats[
                        "cell_observations_all"
                    ] += 1
                    file_stats[
                        "cell_observations_all"
                    ] += 1
                    all_pci_counts[pci] += 1

                    if mapping_info is not None:
                        target_cell_writer.writerow(
                            record
                        )
                        stats[
                            "cell_observations_target"
                        ] += 1
                        file_stats[
                            "cell_observations_target"
                        ] += 1
                        target_pci_counts[pci] += 1

                duplicate_pcis = [
                    pci
                    for pci, count
                    in row_seen_pcis.items()
                    if count > 1
                ]

                if duplicate_pcis:
                    audit = make_audit_row(
                        source_file,
                        source_row,
                        physical_row,
                        time_value,
                        "duplicate_pci_in_cells_list",
                        str(
                            sorted(
                                duplicate_pcis
                            )
                        ),
                        cell_lists=cell_lists,
                    )
                    audit_writer.writerow(audit)
                    audit_type_counts[
                        "duplicate_pci_in_cells_list"
                    ] += 1

                # -------------------------------------------------------------
                # 5.2 波束级列表
                # -------------------------------------------------------------

                if WRITE_BEAM_LEVEL_OUTPUT:
                    beam_lists = {
                        output_name:
                        split_semicolon_list(
                            row.get(source_name)
                        )
                        for (
                            output_name,
                            source_name,
                        ) in BEAM_LIST_COLUMNS.items()
                    }

                    beam_pcis = beam_lists[
                        "beam_pci"
                    ]
                    beam_rsrps = beam_lists[
                        "beam_rsrp_dbm"
                    ]

                    if len(beam_pcis) != len(
                        beam_rsrps
                    ):
                        audit = make_audit_row(
                            source_file,
                            source_row,
                            physical_row,
                            time_value,
                            "beam_pci_rsrp_length_mismatch",
                            (
                                f"PCI数量={len(beam_pcis)}, "
                                f"RSRP数量={len(beam_rsrps)}"
                            ),
                            cell_lists=cell_lists,
                            beam_lists=beam_lists,
                        )
                        audit_writer.writerow(
                            audit
                        )
                        audit_type_counts[
                            "beam_pci_rsrp_length_mismatch"
                        ] += 1
                        beam_pair_count = min(
                            len(beam_pcis),
                            len(beam_rsrps),
                        )
                    else:
                        beam_pair_count = len(
                            beam_pcis
                        )

                    best_beam_by_pci: Dict[
                        int,
                        Dict[str, object],
                    ] = {}

                    for beam_list_index in range(
                        beam_pair_count
                    ):
                        pci = safe_int(
                            list_value(
                                beam_lists[
                                    "beam_pci"
                                ],
                                beam_list_index,
                            )
                        )
                        beam_rsrp = safe_float(
                            list_value(
                                beam_lists[
                                    "beam_rsrp_dbm"
                                ],
                                beam_list_index,
                            )
                        )

                        if (
                            pci is None
                            or beam_rsrp is None
                        ):
                            continue

                        mapping_info = mapping.get(
                            pci
                        )

                        beam_record: Dict[
                            str,
                            object,
                        ] = {
                            "beam_measurement_id": stable_id(
                                "beam",
                                (
                                    f"{source_file}|"
                                    f"{source_row}|"
                                    f"{beam_list_index}|"
                                    f"{pci}"
                                ),
                            ),
                            "rx_point_id": rx_point_id,
                            "source_file": source_file,
                            "source_row": source_row,
                            "csv_physical_row": physical_row,
                            "time": common.get(
                                "time"
                            ),
                            "latitude": common.get(
                                "latitude"
                            ),
                            "longitude": common.get(
                                "longitude"
                            ),
                            "blender_x": common.get(
                                "blender_x"
                            ),
                            "blender_y": common.get(
                                "blender_y"
                            ),
                            "ground_z_m": common.get(
                                "ground_z_m"
                            ),
                            "receiver_z_m": common.get(
                                "receiver_z_m"
                            ),
                            "beam_list_index": beam_list_index,
                            "pci": pci,
                            "beam_index": safe_int(
                                list_value(
                                    beam_lists[
                                        "beam_index"
                                    ],
                                    beam_list_index,
                                )
                            ),
                            "beam_rsrp_dbm": beam_rsrp,
                            "rsrp_unit": "dBm",
                            "beam_rsrq_db": safe_float(
                                list_value(
                                    beam_lists[
                                        "beam_rsrq_db"
                                    ],
                                    beam_list_index,
                                )
                            ),
                            "rsrq_unit": "dB",
                            "beam_sinr_db": safe_float(
                                list_value(
                                    beam_lists[
                                        "beam_sinr_db"
                                    ],
                                    beam_list_index,
                                )
                            ),
                            "sinr_unit": "dB",
                            "is_target_27station_pci": int(
                                mapping_info
                                is not None
                            ),
                            "station_id": (
                                mapping_info[
                                    "station_id"
                                ]
                                if mapping_info
                                else ""
                            ),
                            "sector_index": (
                                mapping_info[
                                    "sector_index"
                                ]
                                if mapping_info
                                else ""
                            ),
                            "antenna_type": (
                                mapping_info[
                                    "antenna_type"
                                ]
                                if mapping_info
                                else ""
                            ),
                        }

                        beam_writer.writerow(
                            beam_record
                        )
                        stats[
                            "beam_observations_all"
                        ] += 1

                        previous = (
                            best_beam_by_pci.get(
                                pci
                            )
                        )

                        if (
                            previous is None
                            or beam_rsrp
                            > float(
                                previous[
                                    "beam_rsrp_dbm"
                                ]
                            )
                        ):
                            best_beam_by_pci[
                                pci
                            ] = beam_record

                    for best_record in (
                        best_beam_by_pci.values()
                    ):
                        beam_best_writer.writerow(
                            best_record
                        )
                        stats[
                            "beam_best_per_pci_observations"
                        ] += 1

            per_file_stats.append({
                "source_file": path.name,
                **{
                    key: int(value)
                    for key, value
                    in file_stats.items()
                },
            })

    missing_target_pcis = sorted(
        set(mapping)
        - set(target_pci_counts)
    )

    report = {
        "input_file_count": len(
            input_paths
        ),
        "input_files": [
            str(path)
            for path in input_paths
        ],
        "mapping_pci_count": len(mapping),
        "mapping_station_count": len({
            int(info["station_id"])
            for info in mapping.values()
        }),
        "units": {
            "RSRP": "dBm",
            "RSRQ": "dB",
            "SINR": "dB",
            "difference_between_two_RSRP_values": "dB",
        },
        "main_extraction_rule": {
            "pci_column": (
                "NR5G Cells PCI List"
            ),
            "rsrp_column": (
                "NR5G Cells RSRP List"
            ),
            "delimiter": ";",
            "pairing": (
                "same list index in the same original row"
            ),
            "aggregation": False,
            "dataset_split": False,
            "rsrp_clipping": False,
        },
        "beam_extraction_rule": {
            "pci_column": (
                "NR5G SSB Beam PCI List"
            ),
            "rsrp_column": (
                "NR5G SSB Beam RSRP List"
            ),
            "same_pci_may_repeat": True,
            "used_as_main_cell_RMSE_data": False,
        },
        "statistics": {
            key: int(value)
            for key, value
            in stats.items()
        },
        "audit_type_counts": {
            key: int(value)
            for key, value
            in audit_type_counts.items()
        },
        "target_pci_observation_counts": {
            str(key): int(value)
            for key, value
            in sorted(
                target_pci_counts.items()
            )
        },
        "all_pci_observation_counts": {
            str(key): int(value)
            for key, value
            in sorted(
                all_pci_counts.items()
            )
        },
        "target_pcis_without_observations": (
            missing_target_pcis
        ),
        "per_file_statistics": per_file_stats,
        "outputs": {
            "cell_all": str(
                all_cell_path
            ),
            "cell_27stations": str(
                target_cell_path
            ),
            "beam_all": str(
                beam_path
            ),
            "beam_best_per_pci": str(
                beam_best_path
            ),
            "audit": str(
                audit_path
            ),
            "report": str(
                report_path
            ),
        },
    }

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=2,
        )

    return report


# =============================================================================
# 6. 命令行
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "展开完整Cellular-Pro CSV中同一行的"
            "多个PCI及对应RSRP。"
        )
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=(
            "包含*_with_blender_xyz.csv"
            "的目录"
        ),
    )

    parser.add_argument(
        "--mapping",
        type=Path,
        default=DEFAULT_MAPPING_CSV,
        help="27站PCI映射CSV",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="输出目录",
    )

    # 兼容Jupyter/Blender传入的-f等参数。
    args, unknown = parser.parse_known_args()

    if unknown:
        print(
            "忽略未知参数：",
            unknown,
        )

    return args


def main() -> None:
    args = parse_args()

    print("=" * 86)
    print(
        "Cellular-Pro 多PCI / RSRP 提取"
    )
    print("=" * 86)
    print(
        "输入目录：",
        args.input_dir,
    )
    print(
        "PCI映射：",
        args.mapping,
    )
    print(
        "输出目录：",
        args.output_dir,
    )
    print(
        "主字段：NR5G Cells PCI List "
        "+ NR5G Cells RSRP List"
    )
    print("RSRP单位：dBm")
    print("RSRQ/SINR单位：dB")
    print("不划分数据集，不聚合，不截断RSRP。")

    if not args.input_dir.exists():
        raise FileNotFoundError(
            f"输入目录不存在：{args.input_dir}"
        )

    input_paths = sorted(
        path
        for path
        in args.input_dir.glob(
            INPUT_GLOB
        )
        if path.is_file()
    )

    if not input_paths:
        raise FileNotFoundError(
            f"没有找到："
            f"{args.input_dir / INPUT_GLOB}\n"
            "请确认使用的是完整的"
            "*_with_blender_xyz.csv。"
        )

    mapping = load_mapping(
        args.mapping
    )

    report = process_files(
        input_paths=input_paths,
        mapping=mapping,
        output_dir=args.output_dir,
    )

    statistics = report[
        "statistics"
    ]

    print("\n" + "=" * 86)
    print("提取完成")
    print("=" * 86)
    print(
        "读取完整CSV数：",
        report["input_file_count"],
    )
    print(
        "原始CSV数据行数：",
        statistics.get(
            "raw_csv_rows",
            0,
        ),
    )
    print(
        "有效空间坐标行数：",
        statistics.get(
            "valid_spatial_rows",
            0,
        ),
    )
    print(
        "展开后全部小区PCI-RSRP观测：",
        statistics.get(
            "cell_observations_all",
            0,
        ),
    )
    print(
        "属于27站的PCI-RSRP观测：",
        statistics.get(
            "cell_observations_target",
            0,
        ),
    )
    print(
        "展开后波束观测：",
        statistics.get(
            "beam_observations_all",
            0,
        ),
    )
    print(
        "未出现的目标PCI：",
        report[
            "target_pcis_without_observations"
        ],
    )
    print(
        "主输出：",
        report["outputs"][
            "cell_27stations"
        ],
    )
    print(
        "审计报告：",
        report["outputs"]["audit"],
    )
    print(
        "JSON报告：",
        report["outputs"]["report"],
    )
    print("=" * 86)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\n" + "!" * 86)
        print("运行失败：", exc)
        print("!" * 86)
        traceback.print_exc()
        raise
