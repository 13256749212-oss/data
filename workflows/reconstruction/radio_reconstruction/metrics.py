from __future__ import annotations

import numpy as np


def regression_metrics(measured: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    measured = np.asarray(measured, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    valid = np.isfinite(measured) & np.isfinite(predicted)
    y = measured[valid]
    p = predicted[valid]
    if len(y) == 0:
        return {
            "validation_count": 0,
            "rmse_db": float("nan"),
            "mae_db": float("nan"),
            "bias_pred_minus_meas_db": float("nan"),
            "r2": float("nan"),
            "p90_absolute_error_db": float("nan"),
            "max_absolute_error_db": float("nan"),
        }
    error = p - y
    ss_res = float(np.sum(error**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return {
        "validation_count": int(len(y)),
        "rmse_db": float(np.sqrt(np.mean(error**2))),
        "mae_db": float(np.mean(np.abs(error))),
        "bias_pred_minus_meas_db": float(np.mean(error)),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "p90_absolute_error_db": float(np.quantile(np.abs(error), 0.90)),
        "max_absolute_error_db": float(np.max(np.abs(error))),
    }
