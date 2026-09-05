from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import warnings

import numpy as np
from scipy.optimize import OptimizeWarning, curve_fit
from scipy.spatial import cKDTree, distance
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold


@dataclass
class ReconstructionResult:
    method: str
    prediction: np.ndarray
    uncertainty: np.ndarray | None
    metadata: dict[str, Any]


class NearestNeighborReconstructor:
    name = "nearest_neighbor"

    def fit(self, xy: np.ndarray, values: np.ndarray) -> "NearestNeighborReconstructor":
        self.xy = np.asarray(xy, dtype=float)
        self.values = np.asarray(values, dtype=float)
        self.tree = cKDTree(self.xy)
        return self

    def predict(self, query_xy: np.ndarray, return_std: bool = False):
        distances, indices = self.tree.query(np.asarray(query_xy, dtype=float), k=1)
        prediction = self.values[indices]
        if return_std:
            # 最近邻本身不提供统计预测标准差；返回NaN，避免把空间距离误写成dB不确定度。
            return prediction, np.full(len(prediction), np.nan, dtype=float)
        return prediction


class OrdinaryKrigingReconstructor:
    """Self-contained ordinary kriging with a fitted spherical variogram."""

    name = "ordinary_kriging"

    @staticmethod
    def _spherical(h: np.ndarray, nugget: float, partial_sill: float, range_m: float) -> np.ndarray:
        h = np.asarray(h, dtype=float)
        r = max(float(range_m), 1e-9)
        ratio = h / r
        core = nugget + partial_sill * (1.5 * ratio - 0.5 * ratio**3)
        return np.where(h <= r, core, nugget + partial_sill)

    def _fit_variogram(self, xy: np.ndarray, values: np.ndarray) -> dict[str, Any]:
        pair_h = distance.pdist(xy, metric="euclidean")
        pair_gamma = 0.5 * distance.pdist(values[:, None], metric="sqeuclidean")
        valid = np.isfinite(pair_h) & np.isfinite(pair_gamma) & (pair_h > 0)
        pair_h = pair_h[valid]
        pair_gamma = pair_gamma[valid]
        if len(pair_h) < 6:
            variance = float(np.var(values, ddof=1)) if len(values) > 1 else 1.0
            return {
                "nugget": max(0.01, 0.05 * variance),
                "partial_sill": max(0.1, 0.95 * variance),
                "range_m": max(10.0, float(np.max(pair_h)) if len(pair_h) else 100.0),
                "fit_status": "fallback_insufficient_pairs",
            }

        max_h = float(np.max(pair_h))
        bins = np.linspace(0.0, max_h, min(13, max(7, len(pair_h) // 8)) + 1)
        bin_id = np.digitize(pair_h, bins) - 1
        centers, gammas, weights = [], [], []
        for i in range(len(bins) - 1):
            mask = bin_id == i
            if np.sum(mask) < 2:
                continue
            centers.append(float(np.mean(pair_h[mask])))
            gammas.append(float(np.median(pair_gamma[mask])))
            weights.append(float(np.sum(mask)))
        centers = np.asarray(centers, dtype=float)
        gammas = np.asarray(gammas, dtype=float)
        weights = np.asarray(weights, dtype=float)

        variance = max(float(np.var(values, ddof=1)), 1e-3)
        p0 = [max(0.01, 0.02 * variance), max(0.1, 0.98 * variance), max(20.0, 0.45 * max_h)]
        lower = [0.0, 1e-6, max(1.0, 0.03 * max_h)]
        upper = [max(variance * 2.0, 1.0), max(variance * 5.0, 1.0), max(10.0, max_h * 3.0)]
        try:
            sigma = 1.0 / np.sqrt(np.maximum(weights, 1.0))
            with warnings.catch_warnings():
                # With only 10--20 sparse samples scipy may be unable to estimate
                # the covariance of fitted variogram parameters even when the
                # fitted parameters themselves are valid.  This warning is not a
                # reconstruction failure, so silence this specific warning only.
                warnings.simplefilter("ignore", OptimizeWarning)
                params, _ = curve_fit(
                    self._spherical,
                    centers,
                    gammas,
                    p0=p0,
                    bounds=(lower, upper),
                    sigma=sigma,
                    absolute_sigma=False,
                    maxfev=20000,
                )
            nugget, partial_sill, range_m = map(float, params)
            status = "fitted_spherical_variogram"
        except Exception as exc:
            nugget, partial_sill, range_m = p0
            status = f"fallback_fit_failed:{type(exc).__name__}"

        return {
            "nugget": float(nugget),
            "partial_sill": float(partial_sill),
            "range_m": float(range_m),
            "fit_status": status,
            "bin_centers_m": centers.tolist(),
            "bin_semivariance_db2": gammas.tolist(),
        }

    def fit(self, xy: np.ndarray, values: np.ndarray) -> "OrdinaryKrigingReconstructor":
        self.xy = np.asarray(xy, dtype=float)
        self.values = np.asarray(values, dtype=float)
        self.variogram = self._fit_variogram(self.xy, self.values)
        n = len(self.values)
        distances_train = distance.cdist(self.xy, self.xy)
        gamma = self._spherical(
            distances_train,
            self.variogram["nugget"],
            self.variogram["partial_sill"],
            self.variogram["range_m"],
        )
        np.fill_diagonal(gamma, 0.0)
        matrix = np.empty((n + 1, n + 1), dtype=float)
        matrix[:n, :n] = gamma
        matrix[:n, n] = 1.0
        matrix[n, :n] = 1.0
        matrix[n, n] = 0.0
        # pinv比直接solve更适合仅20点且可能近共线的道路实测点。
        self.system_inverse = np.linalg.pinv(matrix, rcond=1e-10)
        return self

    def predict(self, query_xy: np.ndarray, return_std: bool = False, chunk_size: int = 50000):
        query_xy = np.asarray(query_xy, dtype=float)
        predictions = np.empty(len(query_xy), dtype=float)
        variances = np.empty(len(query_xy), dtype=float)
        n = len(self.values)
        sill = self.variogram["nugget"] + self.variogram["partial_sill"]

        for start in range(0, len(query_xy), int(chunk_size)):
            stop = min(start + int(chunk_size), len(query_xy))
            distances_query = distance.cdist(self.xy, query_xy[start:stop])
            rhs = np.vstack([
                self._spherical(
                    distances_query,
                    self.variogram["nugget"],
                    self.variogram["partial_sill"],
                    self.variogram["range_m"],
                ),
                np.ones((1, stop - start), dtype=float),
            ])
            solution = self.system_inverse @ rhs
            weights = solution[:n]
            lagrange = solution[n]
            predictions[start:stop] = self.values @ weights
            variance = np.sum(weights * rhs[:n], axis=0) + lagrange
            variances[start:stop] = np.maximum(variance, 0.0)

        if return_std:
            return predictions, np.sqrt(variances)
        return predictions


class AdaptiveMaternGPRReconstructor:
    """
    Modern probabilistic reconstruction baseline:
    anisotropic Matérn-3/2 Gaussian process with automatic hyperparameter fitting.
    """

    name = "adaptive_matern_gpr"

    def __init__(self, random_seed: int = 20260805, optimizer_restarts: int = 12):
        self.random_seed = int(random_seed)
        self.optimizer_restarts = int(optimizer_restarts)

    def fit(self, xy: np.ndarray, values: np.ndarray) -> "AdaptiveMaternGPRReconstructor":
        xy = np.asarray(xy, dtype=float)
        values = np.asarray(values, dtype=float)
        self.x_scaler = StandardScaler().fit(xy)
        x_scaled = self.x_scaler.transform(xy)

        variance = max(float(np.var(values, ddof=1)), 1.0)
        kernel = (
            ConstantKernel(
                constant_value=variance,
                constant_value_bounds=(1e-2, 1e5),
            )
            * Matern(
                length_scale=np.ones(2, dtype=float),
                length_scale_bounds=(0.05, 20.0),
                nu=1.5,
            )
            + WhiteKernel(
                noise_level=max(0.5, 0.03 * variance),
                noise_level_bounds=(1e-3, 1e3),
            )
        )
        self.model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=self.optimizer_restarts,
            random_state=self.random_seed,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(x_scaled, values)
        return self

    def predict(self, query_xy: np.ndarray, return_std: bool = False, chunk_size: int = 50000):
        query_xy = np.asarray(query_xy, dtype=float)
        prediction = np.empty(len(query_xy), dtype=float)
        uncertainty = np.empty(len(query_xy), dtype=float)
        for start in range(0, len(query_xy), int(chunk_size)):
            stop = min(start + int(chunk_size), len(query_xy))
            x_scaled = self.x_scaler.transform(query_xy[start:stop])
            mean, std = self.model.predict(x_scaled, return_std=True)
            prediction[start:stop] = mean
            uncertainty[start:stop] = std
        if return_std:
            return prediction, uncertainty
        return prediction

    @property
    def fitted_kernel(self) -> str:
        return str(self.model.kernel_)




class RTExternalDriftKrigingReconstructor:
    """Kriging with Sionna RT as an external drift (RT-KED).

    Sionna RT is treated as a large-scale spatial covariate rather than a final
    prediction map.  A robust affine trend is estimated from the selected
    training measurements, while the semivariogram is fitted to the remaining
    residual.  Universal-kriging constraints preserve the external drift at
    query locations.  Where the RT prior is unavailable, the method falls back
    to an ordinary-kriging model trained on all measurements.
    """

    name = "rt_external_drift_kriging"

    def __init__(self, simulation_prior, min_valid_prior_points: int = 4):
        self.prior = simulation_prior
        self.min_valid_prior_points = int(min_valid_prior_points)

    @staticmethod
    def _robust_affine(prior_values: np.ndarray, values: np.ndarray) -> tuple[float, float]:
        x = np.asarray(prior_values, dtype=float)
        y = np.asarray(values, dtype=float)
        valid = np.isfinite(x) & np.isfinite(y)
        x, y = x[valid], y[valid]
        n = len(x)
        if n == 0:
            return 1.0, 0.0
        if n < 4 or float(np.ptp(x)) < 1e-6:
            return 1.0, float(np.median(y - x))
        X = np.column_stack([x, np.ones(n, dtype=float)])
        beta = np.asarray([1.0, float(np.median(y - x))], dtype=float)
        for _ in range(20):
            residual = y - X @ beta
            med = float(np.median(residual))
            mad = float(np.median(np.abs(residual - med)))
            scale = max(1.4826 * mad, 0.75)
            cutoff = 1.345 * scale
            abs_r = np.abs(residual)
            w = np.ones(n, dtype=float)
            tail = abs_r > cutoff
            w[tail] = cutoff / np.maximum(abs_r[tail], 1e-9)
            # Weak small-sample shrinkage: RT is a trend covariate, not truth.
            lam = 6.0 / max(float(n), 1.0)
            WX = X * np.sqrt(w)[:, None]
            Wy = y * np.sqrt(w)
            aug_X = np.vstack([WX, [np.sqrt(lam), 0.0]])
            aug_y = np.concatenate([Wy, [np.sqrt(lam)]])
            new_beta, *_ = np.linalg.lstsq(aug_X, aug_y, rcond=None)
            new_beta[0] = float(np.clip(new_beta[0], 0.20, 1.80))
            new_beta[1] = float(np.median(y - new_beta[0] * x))
            if float(np.linalg.norm(new_beta - beta)) < 1e-6:
                beta = new_beta
                break
            beta = new_beta
        return float(beta[0]), float(beta[1])

    def fit(self, xy: np.ndarray, values: np.ndarray) -> "RTExternalDriftKrigingReconstructor":
        self.xy_all = np.asarray(xy, dtype=float)
        self.values_all = np.asarray(values, dtype=float)
        self.safety_model = OrdinaryKrigingReconstructor().fit(self.xy_all, self.values_all)
        prior_all = np.asarray(self.prior.sample(self.xy_all), dtype=float)
        valid = np.isfinite(prior_all) & np.isfinite(self.values_all)
        self.valid_prior_training_count = int(valid.sum())
        self.training_prior_rmse_db = (
            float(np.sqrt(np.mean((prior_all[valid] - self.values_all[valid]) ** 2)))
            if np.any(valid) else float("nan")
        )
        self.enabled = self.valid_prior_training_count >= self.min_valid_prior_points
        if not self.enabled:
            self.affine_a, self.affine_b = 1.0, 0.0
            self.variogram = dict(self.safety_model.variogram)
            return self

        self.xy = self.xy_all[valid]
        self.values = self.values_all[valid]
        self.prior_train = prior_all[valid]
        self.affine_a, self.affine_b = self._robust_affine(self.prior_train, self.values)
        trend = self.affine_a * self.prior_train + self.affine_b
        residual = self.values - trend

        variogram_helper = OrdinaryKrigingReconstructor()
        self.variogram = variogram_helper._fit_variogram(self.xy, residual)
        n = len(self.values)
        d_train = distance.cdist(self.xy, self.xy)
        gamma = variogram_helper._spherical(
            d_train,
            self.variogram["nugget"],
            self.variogram["partial_sill"],
            self.variogram["range_m"],
        )
        np.fill_diagonal(gamma, 0.0)
        self._spherical = variogram_helper._spherical

        prior_mean = float(np.mean(self.prior_train))
        prior_std = max(float(np.std(self.prior_train)), 1e-6)
        self.prior_mean = prior_mean
        self.prior_std = prior_std
        drift = (self.prior_train - prior_mean) / prior_std
        F = np.column_stack([np.ones(n, dtype=float), drift])
        system = np.zeros((n + 2, n + 2), dtype=float)
        system[:n, :n] = gamma
        system[:n, n:] = F
        system[n:, :n] = F.T
        self.system_inverse = np.linalg.pinv(system, rcond=1e-10)
        return self

    def predict(self, query_xy: np.ndarray, return_std: bool = False, chunk_size: int = 50000):
        query_xy = np.asarray(query_xy, dtype=float)
        safety_pred, safety_std = self.safety_model.predict(query_xy, return_std=True)
        if not self.enabled:
            return (safety_pred, safety_std) if return_std else safety_pred
        prior_q = np.asarray(self.prior.sample(query_xy), dtype=float)
        pred = np.asarray(safety_pred, dtype=float).copy()
        std = np.asarray(safety_std, dtype=float).copy()
        finite_idx = np.flatnonzero(np.isfinite(prior_q))
        n = len(self.values)
        for start in range(0, len(finite_idx), int(chunk_size)):
            idx = finite_idx[start:start + int(chunk_size)]
            q = query_xy[idx]
            d_q = distance.cdist(self.xy, q)
            gamma_q = self._spherical(
                d_q,
                self.variogram["nugget"],
                self.variogram["partial_sill"],
                self.variogram["range_m"],
            )
            drift_q = (prior_q[idx] - self.prior_mean) / self.prior_std
            rhs = np.vstack([gamma_q, np.ones((1, len(idx))), drift_q[None, :]])
            sol = self.system_inverse @ rhs
            weights = sol[:n]
            multipliers = sol[n:]
            pred[idx] = self.values @ weights
            variance = np.sum(weights * gamma_q, axis=0) + np.sum(multipliers * rhs[n:], axis=0)
            std[idx] = np.sqrt(np.maximum(variance, 0.0))
        return (pred, std) if return_std else pred

    def predict_components(self, query_xy: np.ndarray) -> dict[str, np.ndarray]:
        query_xy = np.asarray(query_xy, dtype=float)
        prior = np.asarray(self.prior.sample(query_xy), dtype=float)
        calibrated = self.affine_a * prior + self.affine_b
        final, final_std = self.predict(query_xy, return_std=True)
        return {
            "simulation_prior_dbm": prior,
            "calibrated_rt_trend_dbm": calibrated,
            "final_prediction_dbm": final,
            "final_std_db": final_std,
        }

    @property
    def diagnostics(self) -> dict[str, Any]:
        return {
            "algorithm_name": "RT-informed Kriging with External Drift (RT-KED)",
            "simulation_prior_path": str(self.prior.source_path),
            "simulation_prior_key": self.prior.selected_key,
            "physics_enabled": bool(self.enabled),
            "valid_prior_training_count": self.valid_prior_training_count,
            "training_prior_rmse_db": self.training_prior_rmse_db,
            "affine_a": float(self.affine_a),
            "affine_b_db": float(self.affine_b),
            "residual_variogram_nugget_db2": float(self.variogram["nugget"]),
            "residual_variogram_partial_sill_db2": float(self.variogram["partial_sill"]),
            "residual_variogram_range_m": float(self.variogram["range_m"]),
            "residual_variogram_fit_status": self.variogram["fit_status"],
            "missing_rt_policy": "fallback_to_data_only_ordinary_kriging",
        }


class RTMeanMaternGPRReconstructor:
    """Matérn GPR with a calibrated Sionna RT mean function (RT-Mean GPR).

    The model learns only the residual around ``a * RT(x) + b``.  There is no
    post-hoc map fusion weight.  A data-only Matérn GPR is retained solely as a
    safety prediction for cells where the RT prior is unavailable.
    """

    name = "rt_mean_matern_gpr"

    def __init__(self, simulation_prior, random_seed: int = 20260805,
                 optimizer_restarts: int = 6, min_valid_prior_points: int = 4):
        self.prior = simulation_prior
        self.random_seed = int(random_seed)
        self.optimizer_restarts = int(optimizer_restarts)
        self.min_valid_prior_points = int(min_valid_prior_points)

    def fit(self, xy: np.ndarray, values: np.ndarray) -> "RTMeanMaternGPRReconstructor":
        self.xy = np.asarray(xy, dtype=float)
        self.values = np.asarray(values, dtype=float)
        self.safety_model = AdaptiveMaternGPRReconstructor(
            random_seed=self.random_seed,
            optimizer_restarts=self.optimizer_restarts,
        ).fit(self.xy, self.values)
        prior_train = np.asarray(self.prior.sample(self.xy), dtype=float)
        valid = np.isfinite(prior_train) & np.isfinite(self.values)
        self.valid_prior_training_count = int(valid.sum())
        self.training_prior_rmse_db = (
            float(np.sqrt(np.mean((prior_train[valid] - self.values[valid]) ** 2)))
            if np.any(valid) else float("nan")
        )
        self.enabled = self.valid_prior_training_count >= self.min_valid_prior_points
        if not self.enabled:
            self.affine_a, self.affine_b = 1.0, 0.0
            self.residual_model = None
            return self
        self.affine_a, self.affine_b = RTExternalDriftKrigingReconstructor._robust_affine(
            prior_train[valid], self.values[valid]
        )
        residual = self.values[valid] - (
            self.affine_a * prior_train[valid] + self.affine_b
        )
        self.residual_model = AdaptiveMaternGPRReconstructor(
            random_seed=self.random_seed + 17011,
            optimizer_restarts=self.optimizer_restarts,
        ).fit(self.xy[valid], residual)
        return self

    def predict(self, query_xy: np.ndarray, return_std: bool = False):
        query_xy = np.asarray(query_xy, dtype=float)
        safety_pred, safety_std = self.safety_model.predict(query_xy, return_std=True)
        pred = np.asarray(safety_pred, dtype=float).copy()
        std = np.asarray(safety_std, dtype=float).copy()
        if self.enabled and self.residual_model is not None:
            prior_q = np.asarray(self.prior.sample(query_xy), dtype=float)
            residual_pred, residual_std = self.residual_model.predict(query_xy, return_std=True)
            valid = np.isfinite(prior_q) & np.isfinite(residual_pred)
            pred[valid] = self.affine_a * prior_q[valid] + self.affine_b + residual_pred[valid]
            std[valid] = residual_std[valid]
        return (pred, std) if return_std else pred

    def predict_components(self, query_xy: np.ndarray) -> dict[str, np.ndarray]:
        query_xy = np.asarray(query_xy, dtype=float)
        prior = np.asarray(self.prior.sample(query_xy), dtype=float)
        trend = self.affine_a * prior + self.affine_b
        residual = np.full(len(query_xy), np.nan, dtype=float)
        residual_std = np.full(len(query_xy), np.nan, dtype=float)
        if self.enabled and self.residual_model is not None:
            residual, residual_std = self.residual_model.predict(query_xy, return_std=True)
        final, final_std = self.predict(query_xy, return_std=True)
        return {
            "simulation_prior_dbm": prior,
            "calibrated_rt_mean_dbm": trend,
            "residual_gpr_db": residual,
            "residual_gpr_std_db": residual_std,
            "final_prediction_dbm": final,
            "final_std_db": final_std,
        }

    @property
    def diagnostics(self) -> dict[str, Any]:
        return {
            "algorithm_name": "Sionna RT Mean-Function Matérn GPR (RT-Mean GPR)",
            "simulation_prior_path": str(self.prior.source_path),
            "simulation_prior_key": self.prior.selected_key,
            "physics_enabled": bool(self.enabled),
            "valid_prior_training_count": self.valid_prior_training_count,
            "training_prior_rmse_db": self.training_prior_rmse_db,
            "affine_a": float(self.affine_a),
            "affine_b_db": float(self.affine_b),
            "residual_gpr_fitted_kernel": (
                self.residual_model.fitted_kernel if self.residual_model is not None else None
            ),
            "data_only_safety_kernel": self.safety_model.fitted_kernel,
            "missing_rt_policy": "fallback_to_data_only_matern_gpr",
        }


class PhysicsGuidedResidualGPRFusion:
    """
    Enhanced physics-guided reconstruction using a calibrated Sionna RT prior
    and an anisotropic Matérn Gaussian-process residual model.

    The method fits two training-only models:
      1. a data-only Adaptive Matérn GPR safety model; and
      2. a GPR correction field for measured-minus-Sionna residuals.

    Residual shrinkage and the fusion weight between the corrected physics prior
    and the data-only GPR are selected by K-fold cross-validation using only the
    selected training measurements.  Validation measurements never participate
    in model or hyperparameter selection.  When the Sionna prior is missing at a
    query point, prediction automatically falls back to the data-only GPR.
    """

    name = "physics_guided_residual_gpr_fusion"

    def __init__(
        self,
        simulation_prior,
        random_seed: int = 20260805,
        cv_folds: int = 5,
        final_optimizer_restarts: int = 6,
        min_valid_prior_points: int = 4,
    ):
        self.prior = simulation_prior
        self.random_seed = int(random_seed)
        self.cv_folds = int(cv_folds)
        self.final_optimizer_restarts = int(final_optimizer_restarts)
        self.min_valid_prior_points = int(min_valid_prior_points)

    def _fit_components(
        self,
        xy: np.ndarray,
        values: np.ndarray,
        *,
        optimizer_restarts: int,
        seed_offset: int = 0,
    ) -> dict[str, Any]:
        xy = np.asarray(xy, dtype=float)
        values = np.asarray(values, dtype=float)
        raw_gpr = AdaptiveMaternGPRReconstructor(
            random_seed=self.random_seed + int(seed_offset),
            optimizer_restarts=int(optimizer_restarts),
        ).fit(xy, values)

        prior_values = np.asarray(self.prior.sample(xy), dtype=float)
        valid = np.isfinite(prior_values) & np.isfinite(values)
        if int(valid.sum()) < self.min_valid_prior_points:
            return {
                "raw_gpr": raw_gpr,
                "residual_gpr": None,
                "bias": 0.0,
                "valid_prior_count": int(valid.sum()),
            }

        bias = float(np.median(values[valid] - prior_values[valid]))
        residual = values[valid] - prior_values[valid] - bias
        residual_gpr = AdaptiveMaternGPRReconstructor(
            random_seed=self.random_seed + 10007 + int(seed_offset),
            optimizer_restarts=int(optimizer_restarts),
        ).fit(xy[valid], residual)
        return {
            "raw_gpr": raw_gpr,
            "residual_gpr": residual_gpr,
            "bias": bias,
            "valid_prior_count": int(valid.sum()),
        }

    def _predict_components(
        self, components: dict[str, Any], query_xy: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        query_xy = np.asarray(query_xy, dtype=float)
        raw_pred, raw_std = components["raw_gpr"].predict(query_xy, return_std=True)
        prior_pred = np.asarray(self.prior.sample(query_xy), dtype=float)
        corrected = np.full(len(query_xy), np.nan, dtype=float)
        residual_std = np.full(len(query_xy), np.nan, dtype=float)
        residual_model = components.get("residual_gpr")
        if residual_model is not None:
            residual_pred, residual_std_model = residual_model.predict(query_xy, return_std=True)
            valid = np.isfinite(prior_pred) & np.isfinite(residual_pred)
            corrected[valid] = (
                prior_pred[valid]
                + float(components["bias"])
                + residual_pred[valid]
            )
            residual_std[:] = residual_std_model
        return raw_pred, raw_std, prior_pred, corrected, residual_std

    def _cross_validate_fusion(self, xy: np.ndarray, values: np.ndarray) -> dict[str, Any]:
        n = int(len(values))
        if n < 5:
            return {
                "correction_shrinkage": 0.0,
                "fusion_weight": 0.0,
                "cv_rmse_db": float("nan"),
                "raw_gpr_cv_rmse_db": float("nan"),
                "best_hybrid_cv_rmse_db": float("nan"),
                "required_cv_gain_db": 0.0,
                "actual_cv_gain_db": 0.0,
                "candidate_count": 1,
            }

        splits = min(max(2, self.cv_folds), n)
        splitter = KFold(
            n_splits=splits, shuffle=True, random_state=self.random_seed
        )
        fold_cache: list[dict[str, np.ndarray]] = []
        for fold_id, (train_idx, test_idx) in enumerate(splitter.split(xy)):
            components = self._fit_components(
                xy[train_idx], values[train_idx],
                optimizer_restarts=0, seed_offset=fold_id * 101,
            )
            raw, _, prior, corrected, _ = self._predict_components(
                components, xy[test_idx]
            )
            fold_cache.append({
                "truth": values[test_idx],
                "raw": raw,
                "prior": prior,
                "corrected": corrected,
                "bias": np.full(len(test_idx), float(components["bias"])),
            })

        truth = np.concatenate([row["truth"] for row in fold_cache])
        raw = np.concatenate([row["raw"] for row in fold_cache])
        prior = np.concatenate([row["prior"] for row in fold_cache])
        corrected = np.concatenate([row["corrected"] for row in fold_cache])
        biases = np.concatenate([row["bias"] for row in fold_cache])

        raw_rmse = float(np.sqrt(np.mean((raw - truth) ** 2)))
        records: list[dict[str, float]] = [{
            "correction_shrinkage": 0.0,
            "fusion_weight": 0.0,
            "cv_rmse_db": raw_rmse,
        }]
        finite_corrected = np.isfinite(corrected) & np.isfinite(prior)
        for shrinkage in (0.50, 0.75, 1.00):
            # Shrink only the learned local residual while preserving the robust
            # global measured-minus-simulation bias correction.
            bias_prior = prior + biases
            corrected_shrunk = bias_prior + float(shrinkage) * (corrected - bias_prior)
            for fusion_weight in (0.25, 0.50, 0.75, 1.00):
                prediction = raw.copy()
                prediction[finite_corrected] = (
                    float(fusion_weight) * corrected_shrunk[finite_corrected]
                    + (1.0 - float(fusion_weight)) * raw[finite_corrected]
                )
                rmse = float(np.sqrt(np.mean((prediction - truth) ** 2)))
                records.append({
                    "correction_shrinkage": float(shrinkage),
                    "fusion_weight": float(fusion_weight),
                    "cv_rmse_db": rmse,
                })

        best = min(records, key=lambda row: row["cv_rmse_db"])
        required_gain = max(0.20, 0.02 * raw_rmse)
        actual_gain = raw_rmse - float(best["cv_rmse_db"])
        if actual_gain < required_gain:
            selected = dict(records[0])
        else:
            selected = dict(best)
        selected.update({
            "raw_gpr_cv_rmse_db": raw_rmse,
            "best_hybrid_cv_rmse_db": float(best["cv_rmse_db"]),
            "required_cv_gain_db": float(required_gain),
            "actual_cv_gain_db": float(actual_gain),
            "candidate_count": int(len(records)),
        })
        self.cv_records = records
        return selected

    def fit(self, xy: np.ndarray, values: np.ndarray):
        self.xy = np.asarray(xy, dtype=float)
        self.values = np.asarray(values, dtype=float)
        selected = self._cross_validate_fusion(self.xy, self.values)
        self.correction_shrinkage = float(selected["correction_shrinkage"])
        self.fusion_weight = float(selected["fusion_weight"])
        self.cv_rmse_db = float(selected["cv_rmse_db"])
        self.raw_gpr_cv_rmse_db = float(selected["raw_gpr_cv_rmse_db"])
        self.best_hybrid_cv_rmse_db = float(selected["best_hybrid_cv_rmse_db"])
        self.required_cv_gain_db = float(selected["required_cv_gain_db"])
        self.actual_cv_gain_db = float(selected["actual_cv_gain_db"])
        self.candidate_count = int(selected["candidate_count"])
        self.components = self._fit_components(
            self.xy, self.values,
            optimizer_restarts=self.final_optimizer_restarts, seed_offset=99991,
        )
        prior_train = np.asarray(self.prior.sample(self.xy), dtype=float)
        valid = np.isfinite(prior_train)
        self.valid_prior_training_count = int(valid.sum())
        self.training_prior_rmse_db = (
            float(np.sqrt(np.mean((prior_train[valid] - self.values[valid]) ** 2)))
            if np.any(valid) else float("nan")
        )
        return self

    def predict(self, query_xy: np.ndarray, return_std: bool = False):
        query_xy = np.asarray(query_xy, dtype=float)
        raw_pred, raw_std, prior_pred, corrected, residual_std = self._predict_components(
            self.components, query_xy
        )
        final = raw_pred.copy()
        valid = np.isfinite(corrected) & np.isfinite(prior_pred)
        if np.any(valid) and self.fusion_weight > 0.0:
            bias_prior = prior_pred + float(self.components["bias"])
            corrected_shrunk = (
                bias_prior + self.correction_shrinkage * (corrected - bias_prior)
            )
            final[valid] = (
                self.fusion_weight * corrected_shrunk[valid]
                + (1.0 - self.fusion_weight) * raw_pred[valid]
            )

        if return_std:
            uncertainty = np.asarray(raw_std, dtype=float).copy()
            valid_std = valid & np.isfinite(residual_std)
            if np.any(valid_std) and self.fusion_weight > 0.0:
                uncertainty[valid_std] = np.sqrt(
                    ((1.0 - self.fusion_weight) * raw_std[valid_std]) ** 2
                    + (self.fusion_weight * self.correction_shrinkage
                       * residual_std[valid_std]) ** 2
                )
            return final, uncertainty
        return final

    def predict_components(self, query_xy: np.ndarray) -> dict[str, np.ndarray]:
        raw_pred, raw_std, prior_pred, corrected, residual_std = self._predict_components(
            self.components, np.asarray(query_xy, dtype=float)
        )
        final, final_std = self.predict(query_xy, return_std=True)
        return {
            "raw_gpr_dbm": raw_pred,
            "raw_gpr_std_db": raw_std,
            "simulation_prior_dbm": prior_pred,
            "corrected_prior_gpr_dbm": corrected,
            "residual_gpr_std_db": residual_std,
            "final_prediction_dbm": final,
            "final_std_db": final_std,
        }

    @property
    def diagnostics(self) -> dict[str, Any]:
        raw_kernel = getattr(self.components.get("raw_gpr"), "fitted_kernel", None)
        residual_model = self.components.get("residual_gpr")
        residual_kernel = getattr(residual_model, "fitted_kernel", None) if residual_model else None
        return {
            "algorithm_name": "Enhanced Physics-Guided Residual Gaussian Process Fusion (PG-RGPR)",
            "simulation_prior_path": str(self.prior.source_path),
            "simulation_prior_key": self.prior.selected_key,
            "training_bias_measured_minus_sim_db": float(self.components["bias"]),
            "valid_prior_training_count": self.valid_prior_training_count,
            "training_prior_rmse_db": self.training_prior_rmse_db,
            "correction_shrinkage": self.correction_shrinkage,
            "fusion_weight_corrected_prior": self.fusion_weight,
            "fusion_weight_data_only_gpr": 1.0 - self.fusion_weight,
            "training_only_cv_rmse_db": self.cv_rmse_db,
            "raw_gpr_cv_rmse_db": self.raw_gpr_cv_rmse_db,
            "best_hybrid_cv_rmse_db": self.best_hybrid_cv_rmse_db,
            "required_cv_gain_db": self.required_cv_gain_db,
            "actual_cv_gain_db": self.actual_cv_gain_db,
            "cv_candidate_count": self.candidate_count,
            "raw_gpr_fitted_kernel": raw_kernel,
            "residual_gpr_fitted_kernel": residual_kernel,
            "cv_candidates": getattr(self, "cv_records", []),
        }




class DensityAdaptiveGuardedEnsembleReconstructor:
    """Density-adaptive guarded ensemble (DA-GE).

    The dominant estimator changes only with the number of selected measured
    points: RT-KED for ultra-sparse support, ordinary kriging for intermediate
    support, and Matérn GPR for denser support.  A small uncertainty-weighted
    auxiliary contribution is allowed locally, but the dominant estimator is
    guaranteed at least 80% weight.  Model selection never uses validation or
    ground-truth labels.
    """
    name = "density_adaptive_guarded_ensemble"

    def __init__(self, simulation_prior, random_seed: int = 20260805, ultra_sparse_max: int = 15, kriging_max: int = 35):
        self.prior = simulation_prior
        self.random_seed = int(random_seed)
        self.ultra_sparse_max = int(ultra_sparse_max)
        self.kriging_max = int(kriging_max)

    def fit(self, xy: np.ndarray, values: np.ndarray):
        self.xy = np.asarray(xy, float); self.values = np.asarray(values, float)
        self.ok = OrdinaryKrigingReconstructor().fit(self.xy, self.values)
        self.gpr = AdaptiveMaternGPRReconstructor(random_seed=self.random_seed, optimizer_restarts=6).fit(self.xy, self.values)
        self.rtked = RTExternalDriftKrigingReconstructor(self.prior).fit(self.xy, self.values)
        n=len(self.values)
        if n <= self.ultra_sparse_max:
            self.selected_model_name='rt_external_drift_kriging'; self.base_weights={'rt':0.92,'ok':0.08,'gpr':0.0}
        elif n <= self.kriging_max:
            self.selected_model_name='ordinary_kriging'; self.base_weights={'rt':0.06,'ok':0.90,'gpr':0.04}
        else:
            self.selected_model_name='adaptive_matern_gpr'; self.base_weights={'rt':0.0,'ok':0.10,'gpr':0.90}
        return self

    def predict(self, query_xy: np.ndarray, return_std: bool=False, chunk_size: int=50000):
        q=np.asarray(query_xy,float)
        pok,sok=self.ok.predict(q,return_std=True,chunk_size=chunk_size)
        pg,sg=self.gpr.predict(q,return_std=True,chunk_size=chunk_size)
        pr,sr=self.rtked.predict(q,return_std=True,chunk_size=chunk_size)
        means={'ok':pok,'gpr':pg,'rt':pr}; stds={'ok':sok,'gpr':sg,'rt':sr}
        bw=self.base_weights
        rel={k: bw[k]/(np.maximum(stds[k],1.0)**2 + 9.0) for k in bw}
        denom=rel['ok']+rel['gpr']+rel['rt']+1e-12
        w={k:rel[k]/denom for k in rel}
        main={'ordinary_kriging':'ok','adaptive_matern_gpr':'gpr','rt_external_drift_kriging':'rt'}[self.selected_model_name]
        min_main=0.80
        need=np.maximum(0.0,min_main-w[main])
        aux_sum=np.maximum(1.0-w[main],1e-12)
        for k in w:
            if k!=main: w[k]=w[k]*(1.0-need/aux_sum)
        w[main]=w[main]+need
        total=w['ok']+w['gpr']+w['rt']
        for k in w: w[k]=w[k]/total
        pred=w['ok']*pok+w['gpr']*pg+w['rt']*pr
        second=w['ok']*(sok**2+pok**2)+w['gpr']*(sg**2+pg**2)+w['rt']*(sr**2+pr**2)
        std=np.sqrt(np.maximum(second-pred**2,0.0))
        return (pred,std) if return_std else pred

    @property
    def diagnostics(self):
        return {
            'algorithm_name':'Density-Adaptive Guarded Ensemble (DA-GE)',
            'selected_mode':self.selected_model_name,
            'density_ultra_sparse_max':self.ultra_sparse_max,
            'density_kriging_max':self.kriging_max,
            'base_weights':str(self.base_weights),
            'selection_uses_validation_labels':False,
            'rt_valid_prior_training_count':self.rtked.valid_prior_training_count,
        }


class PhysicsGuidedResidualSpatialKriging:
    """Physics-Guided Residual Spatial Kriging/GPR (PG-RSK).

    This method is designed for the very small-sample regime used by the dataset
    application (10--50 measured cells).  It deliberately treats Sionna RT as a
    *shape prior*, not as an already calibrated absolute-RSRP predictor.

    Main steps
    ----------
    1. Robust affine calibration of the RT prior, ``measurement ~= a * RT + b``.
       The slope is regularized toward one to avoid over-fitting at 10 points.
    2. Model only the remaining spatial residual with either ordinary kriging or
       anisotropic Matérn GPR.
    3. Select the residual learner and the data-only safety learner using
       deterministic spatial-block cross-validation, not random K-fold CV.
    4. Use residual predictive uncertainty as a local gate.  Where the residual
       model is uncertain or the RT prior is unavailable, prediction falls back
       smoothly to the selected data-only kriging/GPR safety model.
    5. If the spatial-block CV hybrid gain is not material, the method
       automatically returns the best data-only safety model.  This prevents the
       physics prior from degrading the final map merely because it is available.
    """

    name = "physics_guided_residual_spatial_kriging"

    def __init__(
        self,
        simulation_prior,
        random_seed: int = 20260805,
        cv_blocks: int = 5,
        final_optimizer_restarts: int = 6,
        min_valid_prior_points: int = 4,
    ):
        self.prior = simulation_prior
        self.random_seed = int(random_seed)
        self.cv_blocks = int(cv_blocks)
        self.final_optimizer_restarts = int(final_optimizer_restarts)
        self.min_valid_prior_points = int(min_valid_prior_points)

    @staticmethod
    def _spatial_block_labels(xy: np.ndarray, n_blocks: int) -> np.ndarray:
        """Balanced contiguous spatial blocks via recursive median bisection."""
        xy = np.asarray(xy, dtype=float)
        n = len(xy)
        n_blocks = int(np.clip(n_blocks, 2, max(2, n)))
        blocks: list[np.ndarray] = [np.arange(n, dtype=int)]
        while len(blocks) < n_blocks:
            splittable = [i for i, idx in enumerate(blocks) if len(idx) >= 2]
            if not splittable:
                break
            # Split the spatially widest block first.
            def score(block_i: int) -> tuple[float, int]:
                idx = blocks[block_i]
                span = np.ptp(xy[idx], axis=0)
                return float(np.max(span)), int(len(idx))
            block_i = max(splittable, key=score)
            idx = blocks.pop(block_i)
            span = np.ptp(xy[idx], axis=0)
            axis = int(np.argmax(span))
            order = idx[np.argsort(xy[idx, axis], kind="mergesort")]
            cut = max(1, len(order) // 2)
            left, right = order[:cut], order[cut:]
            if len(right) == 0:
                blocks.append(idx)
                break
            blocks.extend([left, right])
        labels = np.full(n, -1, dtype=int)
        # Deterministic block numbering by block centroid.
        blocks = sorted(blocks, key=lambda idx: (float(np.mean(xy[idx, 0])), float(np.mean(xy[idx, 1]))))
        for label, idx in enumerate(blocks):
            labels[idx] = int(label)
        return labels

    @staticmethod
    def _robust_affine_calibration(prior_values: np.ndarray, values: np.ndarray) -> tuple[float, float]:
        """Robust, small-sample affine RT calibration with slope shrinkage."""
        x = np.asarray(prior_values, dtype=float)
        y = np.asarray(values, dtype=float)
        valid = np.isfinite(x) & np.isfinite(y)
        x, y = x[valid], y[valid]
        n = len(x)
        if n == 0:
            return 1.0, 0.0
        if n < 4 or float(np.ptp(x)) < 1e-6:
            return 1.0, float(np.median(y - x))

        X = np.column_stack([x, np.ones(n, dtype=float)])
        beta = np.asarray([1.0, float(np.median(y - x))], dtype=float)
        for _ in range(20):
            residual = y - X @ beta
            med = float(np.median(residual))
            mad = float(np.median(np.abs(residual - med)))
            scale = max(1.4826 * mad, 0.75)
            cutoff = 1.345 * scale
            abs_r = np.abs(residual)
            weights = np.ones(n, dtype=float)
            tail = abs_r > cutoff
            weights[tail] = cutoff / np.maximum(abs_r[tail], 1e-9)

            # Ridge-like prior a~=1.  It is deliberately stronger when n is tiny.
            slope_lambda = 8.0 / max(float(n), 1.0)
            WX = X * np.sqrt(weights)[:, None]
            Wy = y * np.sqrt(weights)
            aug_X = np.vstack([WX, [np.sqrt(slope_lambda), 0.0]])
            aug_y = np.concatenate([Wy, [np.sqrt(slope_lambda)]])
            new_beta, *_ = np.linalg.lstsq(aug_X, aug_y, rcond=None)
            new_beta[0] = float(np.clip(new_beta[0], 0.35, 1.65))
            # Recompute a robust intercept after slope clipping.
            new_beta[1] = float(np.median(y - new_beta[0] * x))
            if float(np.linalg.norm(new_beta - beta)) < 1e-6:
                beta = new_beta
                break
            beta = new_beta
        return float(beta[0]), float(beta[1])

    def _fit_raw(self, name: str, xy: np.ndarray, values: np.ndarray, *, restarts: int, seed_offset: int):
        if name == "ordinary_kriging":
            return OrdinaryKrigingReconstructor().fit(xy, values)
        if name == "adaptive_matern_gpr":
            return AdaptiveMaternGPRReconstructor(
                random_seed=self.random_seed + int(seed_offset),
                optimizer_restarts=int(restarts),
            ).fit(xy, values)
        raise ValueError(f"Unknown safety model: {name}")

    def _fit_residual_components(
        self,
        xy: np.ndarray,
        values: np.ndarray,
        residual_name: str,
        *,
        restarts: int,
        seed_offset: int,
    ) -> dict[str, Any]:
        prior_values = np.asarray(self.prior.sample(xy), dtype=float)
        valid = np.isfinite(prior_values) & np.isfinite(values)
        if int(valid.sum()) < self.min_valid_prior_points:
            return {
                "residual_model": None,
                "affine_a": 1.0,
                "affine_b": 0.0,
                "valid_prior_count": int(valid.sum()),
            }
        a, b = self._robust_affine_calibration(prior_values[valid], values[valid])
        calibrated = a * prior_values[valid] + b
        residual = values[valid] - calibrated
        if residual_name == "residual_kriging":
            residual_model = OrdinaryKrigingReconstructor().fit(xy[valid], residual)
        elif residual_name == "residual_gpr":
            residual_model = AdaptiveMaternGPRReconstructor(
                random_seed=self.random_seed + 17011 + int(seed_offset),
                optimizer_restarts=int(restarts),
            ).fit(xy[valid], residual)
        else:
            raise ValueError(f"Unknown residual model: {residual_name}")
        return {
            "residual_model": residual_model,
            "affine_a": float(a),
            "affine_b": float(b),
            "valid_prior_count": int(valid.sum()),
        }

    def _predict_physics(
        self, components: dict[str, Any], query_xy: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        query_xy = np.asarray(query_xy, dtype=float)
        prior = np.asarray(self.prior.sample(query_xy), dtype=float)
        physics = np.full(len(query_xy), np.nan, dtype=float)
        std = np.full(len(query_xy), np.nan, dtype=float)
        residual_model = components.get("residual_model")
        if residual_model is None:
            return prior, physics, std
        residual_pred, residual_std = residual_model.predict(query_xy, return_std=True)
        calibrated_prior = float(components["affine_a"]) * prior + float(components["affine_b"])
        valid = np.isfinite(calibrated_prior) & np.isfinite(residual_pred)
        physics[valid] = calibrated_prior[valid] + residual_pred[valid]
        std[valid] = residual_std[valid]
        return calibrated_prior, physics, std

    @staticmethod
    def _gate_from_std(std: np.ndarray, tau_db: float) -> np.ndarray:
        std = np.asarray(std, dtype=float)
        if not np.isfinite(tau_db) or tau_db >= 1e8:
            gate = np.ones_like(std, dtype=float)
        else:
            gate = np.exp(-np.square(std / max(float(tau_db), 1e-6)))
        gate[~np.isfinite(std)] = 0.0
        return np.clip(gate, 0.0, 1.0)

    def _spatial_block_select(self, xy: np.ndarray, values: np.ndarray) -> dict[str, Any]:
        n = int(len(values))
        if n < 5:
            return {
                "mode": "safety",
                "safety_model": "ordinary_kriging",
                "residual_model": "none",
                "gate_tau_db": float("nan"),
                "cv_rmse_db": float("nan"),
                "best_safety_cv_rmse_db": float("nan"),
                "best_hybrid_cv_rmse_db": float("nan"),
                "actual_cv_gain_db": 0.0,
                "required_cv_gain_db": 0.0,
            }

        n_blocks = min(self.cv_blocks, max(2, n // 3))
        labels = self._spatial_block_labels(xy, n_blocks)
        unique_blocks = sorted(int(v) for v in np.unique(labels))
        fold_cache: list[dict[str, Any]] = []
        for fold_id, block in enumerate(unique_blocks):
            test = labels == block
            train = ~test
            if int(train.sum()) < 4 or int(test.sum()) == 0:
                continue
            fold: dict[str, Any] = {
                "truth": values[test],
                "test_indices": np.flatnonzero(test),
            }
            for safety_name in ("ordinary_kriging", "adaptive_matern_gpr"):
                try:
                    safety_model = self._fit_raw(
                        safety_name, xy[train], values[train], restarts=0,
                        seed_offset=fold_id * 101 + (0 if safety_name == "ordinary_kriging" else 17),
                    )
                    pred, std = safety_model.predict(xy[test], return_std=True)
                except Exception:
                    pred = np.full(int(test.sum()), np.nan, dtype=float)
                    std = np.full(int(test.sum()), np.nan, dtype=float)
                fold[f"{safety_name}_pred"] = np.asarray(pred, dtype=float)
                fold[f"{safety_name}_std"] = np.asarray(std, dtype=float)

            for residual_name in ("residual_kriging", "residual_gpr"):
                try:
                    comp = self._fit_residual_components(
                        xy[train], values[train], residual_name,
                        restarts=0, seed_offset=fold_id * 211,
                    )
                    calibrated, physics, physics_std = self._predict_physics(comp, xy[test])
                except Exception:
                    calibrated = np.full(int(test.sum()), np.nan, dtype=float)
                    physics = np.full(int(test.sum()), np.nan, dtype=float)
                    physics_std = np.full(int(test.sum()), np.nan, dtype=float)
                fold[f"{residual_name}_calibrated_prior"] = calibrated
                fold[f"{residual_name}_physics_pred"] = physics
                fold[f"{residual_name}_physics_std"] = physics_std
            fold_cache.append(fold)

        if not fold_cache:
            return {
                "mode": "safety", "safety_model": "ordinary_kriging",
                "residual_model": "none", "gate_tau_db": float("nan"),
                "cv_rmse_db": float("nan"), "best_safety_cv_rmse_db": float("nan"),
                "best_hybrid_cv_rmse_db": float("nan"), "actual_cv_gain_db": 0.0,
                "required_cv_gain_db": 0.0,
            }

        records: list[dict[str, Any]] = []
        safety_records: list[dict[str, Any]] = []
        for safety_name in ("ordinary_kriging", "adaptive_matern_gpr"):
            errors = []
            valid_count = 0
            for fold in fold_cache:
                truth = np.asarray(fold["truth"], dtype=float)
                pred = np.asarray(fold[f"{safety_name}_pred"], dtype=float)
                valid = np.isfinite(truth) & np.isfinite(pred)
                errors.extend((pred[valid] - truth[valid]).tolist())
                valid_count += int(valid.sum())
            rmse = float(np.sqrt(np.mean(np.square(errors)))) if errors else float("inf")
            row = {
                "mode": "safety", "safety_model": safety_name,
                "residual_model": "none", "gate_tau_db": float("nan"),
                "cv_rmse_db": rmse, "valid_cv_count": valid_count,
            }
            records.append(row)
            safety_records.append(row)
        best_safety = min(safety_records, key=lambda r: r["cv_rmse_db"])

        for residual_name in ("residual_kriging", "residual_gpr"):
            for safety_name in ("ordinary_kriging", "adaptive_matern_gpr"):
                for tau in (4.0, 8.0, 12.0, 20.0, float("inf")):
                    errors = []
                    physics_weights = []
                    valid_count = 0
                    for fold in fold_cache:
                        truth = np.asarray(fold["truth"], dtype=float)
                        safety = np.asarray(fold[f"{safety_name}_pred"], dtype=float)
                        physics = np.asarray(fold[f"{residual_name}_physics_pred"], dtype=float)
                        pstd = np.asarray(fold[f"{residual_name}_physics_std"], dtype=float)
                        pred = safety.copy()
                        valid = np.isfinite(truth) & np.isfinite(safety)
                        use_physics = valid & np.isfinite(physics)
                        gate = self._gate_from_std(pstd, tau)
                        pred[use_physics] = (
                            gate[use_physics] * physics[use_physics]
                            + (1.0 - gate[use_physics]) * safety[use_physics]
                        )
                        final_valid = valid & np.isfinite(pred)
                        errors.extend((pred[final_valid] - truth[final_valid]).tolist())
                        physics_weights.extend(gate[use_physics].tolist())
                        valid_count += int(final_valid.sum())
                    rmse = float(np.sqrt(np.mean(np.square(errors)))) if errors else float("inf")
                    records.append({
                        "mode": "hybrid",
                        "safety_model": safety_name,
                        "residual_model": residual_name,
                        "gate_tau_db": tau,
                        "cv_rmse_db": rmse,
                        "mean_physics_weight": float(np.mean(physics_weights)) if physics_weights else 0.0,
                        "valid_cv_count": valid_count,
                    })

        hybrids = [r for r in records if r["mode"] == "hybrid" and np.isfinite(r["cv_rmse_db"])]
        best_hybrid = min(hybrids, key=lambda r: r["cv_rmse_db"]) if hybrids else best_safety
        required_gain = max(0.25, 0.025 * float(best_safety["cv_rmse_db"]))
        actual_gain = float(best_safety["cv_rmse_db"]) - float(best_hybrid["cv_rmse_db"])
        selected = dict(best_hybrid if actual_gain >= required_gain else best_safety)
        selected.update({
            "best_safety_cv_rmse_db": float(best_safety["cv_rmse_db"]),
            "best_hybrid_cv_rmse_db": float(best_hybrid["cv_rmse_db"]),
            "actual_cv_gain_db": float(actual_gain),
            "required_cv_gain_db": float(required_gain),
            "spatial_block_count": int(len(unique_blocks)),
            "spatial_block_labels": labels.tolist(),
            "candidate_count": int(len(records)),
        })
        self.cv_records = records
        return selected

    def fit(self, xy: np.ndarray, values: np.ndarray):
        self.xy = np.asarray(xy, dtype=float)
        self.values = np.asarray(values, dtype=float)
        selected = self._spatial_block_select(self.xy, self.values)
        self.selected_mode = str(selected["mode"])
        self.selected_safety_model_name = str(selected["safety_model"])
        self.selected_residual_model_name = str(selected["residual_model"])
        self.gate_tau_db = float(selected["gate_tau_db"])
        self.cv_rmse_db = float(selected["cv_rmse_db"])
        self.best_safety_cv_rmse_db = float(selected["best_safety_cv_rmse_db"])
        self.best_hybrid_cv_rmse_db = float(selected["best_hybrid_cv_rmse_db"])
        self.actual_cv_gain_db = float(selected["actual_cv_gain_db"])
        self.required_cv_gain_db = float(selected["required_cv_gain_db"])
        self.spatial_block_count = int(selected.get("spatial_block_count", 0))
        self.spatial_block_labels = selected.get("spatial_block_labels", [])
        self.candidate_count = int(selected.get("candidate_count", 0))

        self.safety_model = self._fit_raw(
            self.selected_safety_model_name,
            self.xy, self.values,
            restarts=self.final_optimizer_restarts,
            seed_offset=90001,
        )
        self.physics_components = None
        if self.selected_mode == "hybrid":
            self.physics_components = self._fit_residual_components(
                self.xy, self.values, self.selected_residual_model_name,
                restarts=self.final_optimizer_restarts,
                seed_offset=91001,
            )

        prior_train = np.asarray(self.prior.sample(self.xy), dtype=float)
        valid = np.isfinite(prior_train) & np.isfinite(self.values)
        self.valid_prior_training_count = int(valid.sum())
        self.training_prior_rmse_db = (
            float(np.sqrt(np.mean((prior_train[valid] - self.values[valid]) ** 2)))
            if np.any(valid) else float("nan")
        )
        if int(valid.sum()) >= self.min_valid_prior_points:
            self.affine_a, self.affine_b = self._robust_affine_calibration(
                prior_train[valid], self.values[valid]
            )
            calibrated_train = self.affine_a * prior_train[valid] + self.affine_b
            self.training_calibrated_prior_rmse_db = float(
                np.sqrt(np.mean((calibrated_train - self.values[valid]) ** 2))
            )
        else:
            self.affine_a, self.affine_b = 1.0, 0.0
            self.training_calibrated_prior_rmse_db = float("nan")
        return self

    def predict(self, query_xy: np.ndarray, return_std: bool = False):
        query_xy = np.asarray(query_xy, dtype=float)
        safety_pred, safety_std = self.safety_model.predict(query_xy, return_std=True)
        final = np.asarray(safety_pred, dtype=float).copy()
        final_std = np.asarray(safety_std, dtype=float).copy()
        if self.selected_mode == "hybrid" and self.physics_components is not None:
            _, physics, physics_std = self._predict_physics(self.physics_components, query_xy)
            valid = np.isfinite(physics) & np.isfinite(final)
            gate = self._gate_from_std(physics_std, self.gate_tau_db)
            final[valid] = gate[valid] * physics[valid] + (1.0 - gate[valid]) * final[valid]
            valid_std = valid & np.isfinite(physics_std) & np.isfinite(final_std)
            final_std[valid_std] = np.sqrt(
                np.square(gate[valid_std] * physics_std[valid_std])
                + np.square((1.0 - gate[valid_std]) * final_std[valid_std])
            )
        if return_std:
            return final, final_std
        return final

    def predict_components(self, query_xy: np.ndarray) -> dict[str, np.ndarray]:
        query_xy = np.asarray(query_xy, dtype=float)
        safety_pred, safety_std = self.safety_model.predict(query_xy, return_std=True)
        prior = np.asarray(self.prior.sample(query_xy), dtype=float)
        calibrated = self.affine_a * prior + self.affine_b
        physics = np.full(len(query_xy), np.nan, dtype=float)
        physics_std = np.full(len(query_xy), np.nan, dtype=float)
        gate = np.zeros(len(query_xy), dtype=float)
        if self.selected_mode == "hybrid" and self.physics_components is not None:
            calibrated, physics, physics_std = self._predict_physics(self.physics_components, query_xy)
            gate = self._gate_from_std(physics_std, self.gate_tau_db)
            gate[~np.isfinite(physics)] = 0.0
        final, final_std = self.predict(query_xy, return_std=True)
        return {
            "simulation_prior_dbm": prior,
            "affine_calibrated_prior_dbm": calibrated,
            "physics_residual_prediction_dbm": physics,
            "physics_residual_std_db": physics_std,
            "local_physics_weight": gate,
            "safety_prediction_dbm": np.asarray(safety_pred, dtype=float),
            "safety_std_db": np.asarray(safety_std, dtype=float),
            "final_prediction_dbm": final,
            "final_std_db": final_std,
        }

    @property
    def diagnostics(self) -> dict[str, Any]:
        residual_kernel = None
        residual_variogram = None
        if self.physics_components is not None:
            residual_model = self.physics_components.get("residual_model")
            residual_kernel = getattr(residual_model, "fitted_kernel", None)
            residual_variogram = getattr(residual_model, "variogram", None)
        safety_kernel = getattr(self.safety_model, "fitted_kernel", None)
        safety_variogram = getattr(self.safety_model, "variogram", None)
        return {
            "algorithm_name": "Physics-Guided Residual Spatial Kriging/GPR (PG-RSK)",
            "simulation_prior_path": str(self.prior.source_path),
            "simulation_prior_key": self.prior.selected_key,
            "selected_mode": self.selected_mode,
            "selected_safety_model": self.selected_safety_model_name,
            "selected_residual_model": self.selected_residual_model_name,
            "uncertainty_gate_tau_db": self.gate_tau_db,
            "affine_prior_slope_a": float(self.affine_a),
            "affine_prior_intercept_b_db": float(self.affine_b),
            "valid_prior_training_count": self.valid_prior_training_count,
            "training_raw_prior_rmse_db": self.training_prior_rmse_db,
            "training_affine_calibrated_prior_rmse_db": self.training_calibrated_prior_rmse_db,
            "training_only_spatial_block_cv_rmse_db": self.cv_rmse_db,
            "best_safety_spatial_cv_rmse_db": self.best_safety_cv_rmse_db,
            "best_hybrid_spatial_cv_rmse_db": self.best_hybrid_cv_rmse_db,
            "required_spatial_cv_gain_db": self.required_cv_gain_db,
            "actual_spatial_cv_gain_db": self.actual_cv_gain_db,
            "spatial_block_count": self.spatial_block_count,
            "spatial_block_labels": self.spatial_block_labels,
            "cv_candidate_count": self.candidate_count,
            "safety_fitted_kernel": safety_kernel,
            "safety_variogram": safety_variogram,
            "residual_fitted_kernel": residual_kernel,
            "residual_variogram": residual_variogram,
            "cv_candidates": getattr(self, "cv_records", []),
        }


class _IDWResidualModel:
    def __init__(self, power: float = 2.0, distance_floor_m: float = 1.0):
        self.power = float(power)
        self.distance_floor_m = float(distance_floor_m)

    def fit(self, xy: np.ndarray, values: np.ndarray):
        self.xy = np.asarray(xy, dtype=float)
        self.values = np.asarray(values, dtype=float)
        self.tree = cKDTree(self.xy)
        return self

    def predict(self, query_xy: np.ndarray, return_std: bool = False):
        query_xy = np.asarray(query_xy, dtype=float)
        k = min(max(2, len(self.values)), 8)
        distances, indices = self.tree.query(query_xy, k=k)
        if k == 1:
            distances = distances[:, None]
            indices = indices[:, None]
        weights = 1.0 / np.maximum(distances, self.distance_floor_m) ** self.power
        pred = np.sum(weights * self.values[indices], axis=1) / np.maximum(
            np.sum(weights, axis=1), 1e-30
        )
        if return_std:
            centered = self.values[indices] - pred[:, None]
            var = np.sum(weights * centered**2, axis=1) / np.maximum(
                np.sum(weights, axis=1), 1e-30
            )
            return pred, np.sqrt(np.maximum(var, 0.0))
        return pred


class _LinearRBFResidualModel:
    def __init__(self, smoothing: float = 10.0):
        self.smoothing = float(smoothing)

    def fit(self, xy: np.ndarray, values: np.ndarray):
        from scipy.interpolate import RBFInterpolator

        self.xy = np.asarray(xy, dtype=float)
        self.values = np.asarray(values, dtype=float)
        self.model = RBFInterpolator(
            self.xy,
            self.values,
            kernel="linear",
            smoothing=self.smoothing,
        )
        self.tree = cKDTree(self.xy)
        return self

    def predict(self, query_xy: np.ndarray, return_std: bool = False):
        query_xy = np.asarray(query_xy, dtype=float)
        pred = np.asarray(self.model(query_xy), dtype=float).reshape(-1)
        if return_std:
            distance_to_train, _ = self.tree.query(query_xy, k=1)
            scale = max(float(np.std(self.values, ddof=1)) if len(self.values) > 1 else 1.0, 0.5)
            std = scale * np.minimum(1.0 + distance_to_train / 150.0, 3.0)
            return pred, std
        return pred


class PhysicsGuidedSionnaResidualFusion:
    """
    Proposed method: Physics-Guided Sionna Residual Fusion (PG-SRF).

    The calibrated Sionna RT sector map is used as a physics prior. Ten measured
    points are used only to estimate a robust bias and a spatial residual field.
    A raw ordinary-kriging prediction and the corrected Sionna prior are fused.
    Residual model type, correction shrinkage and fusion weight are selected by
    leave-one-out cross-validation on the training points only.
    """

    name = "physics_guided_sionna_residual_fusion"

    def __init__(
        self,
        simulation_prior,
        random_seed: int = 20260805,
        min_valid_prior_points: int = 4,
    ):
        self.prior = simulation_prior
        self.random_seed = int(random_seed)
        self.min_valid_prior_points = int(min_valid_prior_points)

    @staticmethod
    def _residual_candidates():
        return [
            ("idw_p1", lambda: _IDWResidualModel(power=1.0)),
            ("idw_p2", lambda: _IDWResidualModel(power=2.0)),
            ("idw_p3", lambda: _IDWResidualModel(power=3.0)),
            ("rbf_linear_s1", lambda: _LinearRBFResidualModel(smoothing=1.0)),
            ("rbf_linear_s10", lambda: _LinearRBFResidualModel(smoothing=10.0)),
            ("rbf_linear_s100", lambda: _LinearRBFResidualModel(smoothing=100.0)),
            ("residual_kriging", lambda: OrdinaryKrigingReconstructor()),
        ]

    def _fit_components(self, xy: np.ndarray, values: np.ndarray, residual_factory):
        raw_kriging = OrdinaryKrigingReconstructor().fit(xy, values)
        prior_values = self.prior.sample(xy)
        valid = np.isfinite(prior_values) & np.isfinite(values)
        if int(valid.sum()) < self.min_valid_prior_points:
            return {
                "raw_kriging": raw_kriging,
                "bias": 0.0,
                "residual_model": None,
                "valid_prior_count": int(valid.sum()),
            }
        bias = float(np.median(values[valid] - prior_values[valid]))
        centered_residual = values[valid] - prior_values[valid] - bias
        residual_model = residual_factory().fit(xy[valid], centered_residual)
        return {
            "raw_kriging": raw_kriging,
            "bias": bias,
            "residual_model": residual_model,
            "valid_prior_count": int(valid.sum()),
        }

    def _predict_components(self, components, query_xy: np.ndarray):
        raw_pred, raw_std = components["raw_kriging"].predict(query_xy, return_std=True)
        prior_pred = self.prior.sample(query_xy)
        if components["residual_model"] is None:
            corrected = np.full(len(query_xy), np.nan, dtype=float)
            residual_std = np.full(len(query_xy), np.nan, dtype=float)
        else:
            residual_pred, residual_std = components["residual_model"].predict(
                query_xy, return_std=True
            )
            corrected = prior_pred + float(components["bias"]) + residual_pred
        return raw_pred, raw_std, prior_pred, corrected, residual_std

    def _loocv_select(self, xy: np.ndarray, values: np.ndarray) -> dict[str, Any]:
        n = len(values)
        if n < 4:
            return {
                "residual_model_name": "none",
                "residual_factory": lambda: _IDWResidualModel(power=2.0),
                "correction_shrinkage": 0.0,
                "fusion_weight": 0.0,
                "loocv_rmse_db": float("nan"),
                "candidate_count": 0,
            }

        shrinkages = [0.25, 0.5, 0.75, 1.0]
        fusion_weights = [0.25, 0.5, 0.75, 1.0]
        records: list[dict[str, Any]] = []

        # Include pure raw kriging as a safety candidate.
        raw_errors = []
        for holdout in range(n):
            keep = np.arange(n) != holdout
            raw = OrdinaryKrigingReconstructor().fit(xy[keep], values[keep])
            pred = float(raw.predict(xy[holdout:holdout + 1])[0])
            raw_errors.append(pred - float(values[holdout]))
        records.append({
            "residual_model_name": "none",
            "residual_factory": lambda: _IDWResidualModel(power=2.0),
            "correction_shrinkage": 0.0,
            "fusion_weight": 0.0,
            "loocv_rmse_db": float(np.sqrt(np.mean(np.asarray(raw_errors) ** 2))),
        })

        for residual_name, residual_factory in self._residual_candidates():
            fold_cache = []
            valid_candidate = True
            for holdout in range(n):
                keep = np.arange(n) != holdout
                try:
                    components = self._fit_components(
                        xy[keep], values[keep], residual_factory
                    )
                    raw_pred, _, prior_pred, corrected_pred, _ = self._predict_components(
                        components, xy[holdout:holdout + 1]
                    )
                    fold_cache.append((
                        float(raw_pred[0]),
                        float(prior_pred[0]),
                        float(corrected_pred[0]),
                        float(values[holdout]),
                        float(components["bias"]),
                    ))
                except Exception:
                    valid_candidate = False
                    break
            if not valid_candidate or len(fold_cache) != n:
                continue

            for shrinkage in shrinkages:
                for fusion_weight in fusion_weights:
                    errors = []
                    finite_count = 0
                    for raw_pred, prior_pred, corrected_pred, truth, bias in fold_cache:
                        if np.isfinite(corrected_pred):
                            # Shrink local residual toward bias-corrected prior.
                            bias_prior = prior_pred + bias
                            corrected_shrunk = (
                                bias_prior
                                + float(shrinkage) * (corrected_pred - bias_prior)
                            )
                            pred = (
                                float(fusion_weight) * corrected_shrunk
                                + (1.0 - float(fusion_weight)) * raw_pred
                            )
                            finite_count += 1
                        else:
                            pred = raw_pred
                        errors.append(pred - truth)
                    if finite_count < max(self.min_valid_prior_points, n // 2):
                        continue
                    rmse = float(np.sqrt(np.mean(np.asarray(errors) ** 2)))
                    records.append({
                        "residual_model_name": residual_name,
                        "residual_factory": residual_factory,
                        "correction_shrinkage": float(shrinkage),
                        "fusion_weight": float(fusion_weight),
                        "loocv_rmse_db": rmse,
                    })

        raw_record = records[0]
        best_record = min(records, key=lambda row: row["loocv_rmse_db"])
        # Ten-point LOOCV has high variance. Use the physics prior only when it
        # produces a material training-only gain; otherwise retain raw kriging.
        required_gain_db = max(0.50, 0.05 * float(raw_record["loocv_rmse_db"]))
        actual_gain_db = float(raw_record["loocv_rmse_db"]) - float(best_record["loocv_rmse_db"])
        selected = dict(best_record if actual_gain_db >= required_gain_db else raw_record)
        selected["candidate_count"] = int(len(records))
        selected["raw_kriging_loocv_rmse_db"] = float(raw_record["loocv_rmse_db"])
        selected["best_hybrid_loocv_rmse_db"] = float(best_record["loocv_rmse_db"])
        selected["required_loocv_gain_db"] = float(required_gain_db)
        selected["actual_loocv_gain_db"] = float(actual_gain_db)
        self.cv_records = [
            {k: v for k, v in row.items() if k != "residual_factory"}
            for row in records
        ]
        return selected

    def fit(self, xy: np.ndarray, values: np.ndarray):
        self.xy = np.asarray(xy, dtype=float)
        self.values = np.asarray(values, dtype=float)
        selected = self._loocv_select(self.xy, self.values)
        self.selected_residual_model_name = str(selected["residual_model_name"])
        self.correction_shrinkage = float(selected["correction_shrinkage"])
        self.fusion_weight = float(selected["fusion_weight"])
        self.loocv_rmse_db = float(selected["loocv_rmse_db"])
        self.candidate_count = int(selected["candidate_count"])
        self.raw_kriging_loocv_rmse_db = float(selected.get("raw_kriging_loocv_rmse_db", np.nan))
        self.best_hybrid_loocv_rmse_db = float(selected.get("best_hybrid_loocv_rmse_db", np.nan))
        self.required_loocv_gain_db = float(selected.get("required_loocv_gain_db", np.nan))
        self.actual_loocv_gain_db = float(selected.get("actual_loocv_gain_db", np.nan))
        self.components = self._fit_components(
            self.xy, self.values, selected["residual_factory"]
        )
        self.prior_training_values = self.prior.sample(self.xy)
        self.valid_prior_training_count = int(np.isfinite(self.prior_training_values).sum())
        self.training_prior_rmse_db = (
            float(np.sqrt(np.mean((self.prior_training_values[np.isfinite(self.prior_training_values)]
                                   - self.values[np.isfinite(self.prior_training_values)]) ** 2)))
            if self.valid_prior_training_count else float("nan")
        )
        return self

    def predict(self, query_xy: np.ndarray, return_std: bool = False):
        query_xy = np.asarray(query_xy, dtype=float)
        raw_pred, raw_std, prior_pred, corrected_pred, residual_std = self._predict_components(
            self.components, query_xy
        )
        final = raw_pred.copy()
        valid_corrected = np.isfinite(corrected_pred)
        if np.any(valid_corrected):
            bias_prior = prior_pred + float(self.components["bias"])
            corrected_shrunk = (
                bias_prior
                + self.correction_shrinkage * (corrected_pred - bias_prior)
            )
            final[valid_corrected] = (
                self.fusion_weight * corrected_shrunk[valid_corrected]
                + (1.0 - self.fusion_weight) * raw_pred[valid_corrected]
            )

        if return_std:
            uncertainty = np.asarray(raw_std, dtype=float)
            if residual_std is not None:
                residual_std = np.asarray(residual_std, dtype=float)
                valid_std = valid_corrected & np.isfinite(residual_std)
                uncertainty[valid_std] = np.sqrt(
                    ((1.0 - self.fusion_weight) * raw_std[valid_std]) ** 2
                    + (self.fusion_weight * self.correction_shrinkage
                       * residual_std[valid_std]) ** 2
                )
            return final, uncertainty
        return final

    def predict_components(self, query_xy: np.ndarray) -> dict[str, np.ndarray]:
        raw_pred, raw_std, prior_pred, corrected_pred, residual_std = self._predict_components(
            self.components, np.asarray(query_xy, dtype=float)
        )
        final, final_std = self.predict(query_xy, return_std=True)
        return {
            "raw_kriging_dbm": raw_pred,
            "raw_kriging_std_db": raw_std,
            "simulation_prior_dbm": prior_pred,
            "corrected_prior_dbm": corrected_pred,
            "residual_std_db": residual_std,
            "final_prediction_dbm": final,
            "final_std_db": final_std,
        }

    @property
    def diagnostics(self) -> dict[str, Any]:
        return {
            "algorithm_name": "Physics-Guided Sionna Residual Fusion (PG-SRF)",
            "simulation_prior_path": str(self.prior.source_path),
            "simulation_prior_key": self.prior.selected_key,
            "simulation_prior_finite_fraction": self.prior.metadata.get("finite_fraction"),
            "selected_residual_model": self.selected_residual_model_name,
            "correction_shrinkage": self.correction_shrinkage,
            "fusion_weight_corrected_prior": self.fusion_weight,
            "fusion_weight_raw_kriging": 1.0 - self.fusion_weight,
            "training_bias_measured_minus_sim_db": float(self.components["bias"]),
            "valid_prior_training_count": self.valid_prior_training_count,
            "training_prior_rmse_db": self.training_prior_rmse_db,
            "loocv_rmse_db": self.loocv_rmse_db,
            "raw_kriging_loocv_rmse_db": self.raw_kriging_loocv_rmse_db,
            "best_hybrid_loocv_rmse_db": self.best_hybrid_loocv_rmse_db,
            "required_loocv_gain_db": self.required_loocv_gain_db,
            "actual_loocv_gain_db": self.actual_loocv_gain_db,
            "loocv_candidate_count": self.candidate_count,
            "loocv_candidates": getattr(self, "cv_records", []),
        }
