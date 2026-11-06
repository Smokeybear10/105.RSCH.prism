from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


CANDIDATES = (
    "zero",
    "historical_median",
    "price_ridge_0.1n",
    "price_ridge_1n",
    "price_ridge_10n",
    "event_ridge_0.1n",
    "event_ridge_1n",
    "event_ridge_10n",
    "event_huber_0.1n",
)


@dataclass(frozen=True)
class FittedModel:
    name: str
    features: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray
    intercept: float

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if not self.features:
            return np.full(len(frame), self.intercept)
        values = frame[list(self.features)].to_numpy(dtype=float)
        return ((values - self.mean) / self.scale) @ self.coefficients + self.intercept


def _fit_candidate(name: str, frame: pd.DataFrame, price_features: tuple[str, ...] | None = None,
                   event_features: tuple[str, ...] | None = None) -> FittedModel:
    from relay.model import EVENT_FEATURES, PRICE_FEATURES

    target = frame["realized_excess_return"].to_numpy(dtype=float)
    if name in ("zero", "historical_median"):
        intercept = 0.0 if name == "zero" else float(np.median(target))
        return FittedModel(name, (), np.array([]), np.array([]), np.array([]), intercept)

    family, method, strength = name.split("_")
    features = (price_features or PRICE_FEATURES) if family == "price" else (event_features or EVENT_FEATURES)
    values = frame[list(features)].to_numpy(dtype=float)
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    normalized = (values - mean) / scale
    penalty = float(strength[:-1]) * len(frame)
    if method == "ridge":
        intercept = float(target.mean())
        coefficients = np.linalg.solve(
            normalized.T @ normalized + penalty * np.eye(len(features)),
            normalized.T @ (target - intercept),
        )
    else:
        intercept = float(np.median(target))
        target_scale = max(float(np.median(np.abs(target - intercept))) * 1.4826, 1e-6)
        standardized_target = (target - intercept) / target_scale
        design = np.column_stack([np.ones(len(frame)), normalized])
        penalize = np.diag([0.0, *([penalty] * len(features))])
        parameters = np.zeros(design.shape[1])
        for _ in range(30):
            residual = standardized_target - design @ parameters
            weights = np.minimum(1.0, 1.345 / np.maximum(np.abs(residual), 1e-12))
            updated = np.linalg.solve(
                design.T @ (weights[:, None] * design) + penalize,
                design.T @ (weights * standardized_target),
            )
            converged = np.max(np.abs(updated - parameters)) < 1e-8
            parameters = updated
            if converged:
                break
        intercept += float(parameters[0] * target_scale)
        coefficients = parameters[1:] * target_scale
    return FittedModel(name, features, mean, scale, coefficients, intercept)


def _sample(frame: pd.DataFrame) -> dict:
    return {
        "n_rows": len(frame),
        "n_events": int(frame["event_id"].nunique()),
        "n_decision_dates": int(frame["decision_date"].nunique()),
        "decision_start": frame["decision_date"].min().date().isoformat() if len(frame) else None,
        "decision_end": frame["decision_date"].max().date().isoformat() if len(frame) else None,
        "latest_exit": frame["exit_date"].max().date().isoformat() if len(frame) else None,
    }


def fit_selected(fit: pd.DataFrame, price_features: tuple[str, ...] | None = None,
                 event_features: tuple[str, ...] | None = None) -> tuple[FittedModel, dict]:
    dates = np.sort(fit["decision_date"].unique())
    audit = {
        "selected": "zero",
        "metric": "mean_absolute_error",
        "validation_fraction": 0.2,
        "validation_start": None,
        "inner_fit": _sample(fit.iloc[:0]),
        "inner_validation": _sample(fit.iloc[:0]),
        "purged_inner_fit_rows": 0,
        "candidate_scores": [],
        "fallback_reason": None,
        "refit": _sample(fit),
    }
    if len(dates) < 2:
        audit["fallback_reason"] = "Too few decision dates for chronological model selection."
        return _fit_candidate("zero", fit), audit

    cutoff = pd.Timestamp(dates[min(len(dates) - 1, int(0.8 * len(dates)))])
    before = fit.loc[fit["decision_date"].lt(cutoff)]
    inner_fit = before.loc[before["exit_date"].lt(cutoff)]
    validation = fit.loc[fit["decision_date"].ge(cutoff)]
    audit.update(
        validation_start=cutoff.date().isoformat(),
        inner_fit=_sample(inner_fit),
        inner_validation=_sample(validation),
        purged_inner_fit_rows=len(before) - len(inner_fit),
    )
    if len(inner_fit) < 100 or inner_fit["event_id"].nunique() < 20 or validation["event_id"].nunique() < 5:
        audit["fallback_reason"] = "Need 100 inner-fit rows, 20 inner-fit events and 5 validation events; use zero excess return."
        return _fit_candidate("zero", fit), audit

    actual = validation["realized_excess_return"].to_numpy(dtype=float)
    for name in CANDIDATES:
        forecast = _fit_candidate(name, inner_fit, price_features, event_features).predict(validation)
        audit["candidate_scores"].append({"candidate": name, "mae": float(np.abs(forecast - actual).mean())})
    audit["selected"] = min(audit["candidate_scores"], key=lambda row: row["mae"])["candidate"]
    return _fit_candidate(audit["selected"], fit, price_features, event_features), audit


@dataclass(frozen=True)
class SelectedModel:
    model: FittedModel
    audit: dict

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return self.model.predict(frame)


def select_model(fit: pd.DataFrame, price_features: tuple[str, ...],
                 event_features: tuple[str, ...]) -> SelectedModel:
    model, audit = fit_selected(fit, price_features, event_features)
    return SelectedModel(model, audit)


def selection_protocol() -> dict:
    return {
        "status": "Exploratory historical comparison; the existing sample has already been inspected.",
        "candidate_order": list(CANDIDATES),
        "selection_metric": "mean_absolute_error",
        "inner_validation_fraction_of_fit_dates": 0.2,
        "inner_min_fit_rows": 100,
        "inner_min_fit_events": 20,
        "inner_min_validation_events": 5,
        "penalty_basis": "Ridge/Huber coefficient penalty equals the candidate multiplier times fitting row count.",
        "ridge_penalty_multipliers": [0.1, 1.0, 10.0],
        "huber_penalty_multiplier": 0.1,
        "huber_delta": 1.345,
        "huber_max_iterations": 30,
        "huber_target_scale": "1.4826 times fitting-target median absolute deviation, floored at 1e-6.",
        "purging": "Inner fitting labels must exit strictly before the first inner-validation decision date.",
        "scaling": "Each candidate learns feature normalization on its inner fitting rows only.",
        "selection": "Minimum inner-validation MAE; exact ties favor the earlier, simpler candidate.",
        "fallback": "Zero excess return when inner sample minimums are not met.",
        "refit": "The selected candidate is refit on the whole purged outer fit. Outer calibration and test rows are never selection inputs.",
    }
