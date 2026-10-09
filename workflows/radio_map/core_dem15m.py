#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compatibility facade for the single reviewed DEM+1.5 m implementation.

Historically this repository contained duplicated ``core_dem15m.py`` copies. To avoid
method drift, this module now re-exports the authoritative implementation from
``workflows/parameter_calibration/run_27station_parameter_calibration.py``.
"""
from __future__ import annotations
import importlib.util
import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
_PARAM_DIR = _THIS.parents[1] / "parameter_calibration"
_TARGET = _PARAM_DIR / "run_27station_parameter_calibration.py"
if str(_PARAM_DIR) not in sys.path:
    sys.path.insert(0, str(_PARAM_DIR))
_spec = importlib.util.spec_from_file_location("_reviewed_dem15m_impl", _TARGET)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Cannot load reviewed calibration implementation: {_TARGET}")
_impl = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("_reviewed_dem15m_impl", _impl)
_spec.loader.exec_module(_impl)
for _name in dir(_impl):
    if _name.startswith("__"):
        continue
    globals()[_name] = getattr(_impl, _name)

if __name__ == "__main__":
    _impl.main()
