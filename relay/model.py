from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


PRICE_FEATURES = (
    "target_return_1d", "target_momentum_20d", "target_vol_20d",
    "market_return_1d", "market_momentum_20d", "industry_return_1d",
)
NETWORK_FEATURES = PRICE_FEATURES + (
    "source_return_1d", "source_event_return", "source_momentum_20d",
    "correlation_60d", "correlation_x_source_return", "correlation_x_event_return",
)
TEXT_FEATURES = (
    "demand_signal", "pricing_signal", "competitive_signal", "guidance_signal",
)
EVENT_FEATURES = NETWORK_FEATURES + TEXT_FEATURES + ("text_available",) + tuple(
    f"source_event_x_{name}" for name in TEXT_FEATURES
)
MODEL_SPECS = (
    ("zero_excess", "Zero excess return", ()),
    ("price_only", "Price controls", PRICE_FEATURES),
    ("network", "Network spillover", NETWORK_FEATURES),
    ("event_conditioned", "Event-conditioned spillover", EVENT_FEATURES),
)
INPUT_FEATURES = PRICE_FEATURES + (
    "source_return_1d", "source_event_return", "source_momentum_20d",
    "correlation_60d",
) + TEXT_FEATURES + ("text_available",)
DATE_COLUMNS = ("decision_date", "entry_date", "exit_date")
IDENTIFIER_COLUMNS = ("event_id", "source_ticker", "target_ticker", "event_type")
RETURN_COLUMNS = ("realized_return", "benchmark_return", "realized_excess_return")
RIDGE_PENALTY = 10.0
MIN_FIT_EVENTS = 20
MIN_CALIBRATION_ROWS = 20
MIN_CALIBRATION_EVENTS = 5
MIN_CALIBRATION_DATES = 5


@dataclass(frozen=True)
class RidgeFit:
    mean: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray
    intercept: float

    def predict(self, features: np.ndarray) -> np.ndarray:
        return ((features - self.mean) / self.scale) @ self.coefficients + self.intercept


def _fit_ridge(features: np.ndarray, target: np.ndarray) -> RidgeFit:
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    normalized = (features - mean) / scale
    intercept = float(target.mean())
    coefficients = np.linalg.solve(
        normalized.T @ normalized + RIDGE_PENALTY * np.eye(features.shape[1]),
        normalized.T @ (target - intercept),
    )
    return RidgeFit(mean, scale, coefficients, intercept)


def _prepare_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    required = IDENTIFIER_COLUMNS + DATE_COLUMNS + RETURN_COLUMNS + INPUT_FEATURES
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        return pd.DataFrame(), [f"Missing required columns: {', '.join(missing)}."]

    data = frame.loc[:, list(required)].copy()
    for name in DATE_COLUMNS:
        data[name] = pd.to_datetime(data[name], errors="coerce", utc=True).dt.tz_convert(None).dt.normalize()
    for name in RETURN_COLUMNS + INPUT_FEATURES:
        data[name] = pd.to_numeric(data[name], errors="coerce")

    # Missing text is an observed absence, not an imputed financial feature.
    no_text = data["text_available"].eq(0)
    data.loc[no_text, list(TEXT_FEATURES)] = 0.0
    valid = data[list(IDENTIFIER_COLUMNS) + list(DATE_COLUMNS)].notna().all(axis=1)
    valid &= np.isfinite(data[list(RETURN_COLUMNS + INPUT_FEATURES)].to_numpy(dtype=float)).all(axis=1)
    valid &= data["text_available"].isin([0, 1])
    valid &= data["decision_date"].lt(data["entry_date"])
    valid &= data["entry_date"].le(data["exit_date"])
    dropped = int((~valid).sum())
    warnings = [f"Excluded {dropped} rows with missing, nonfinite, or invalid dated inputs."] if dropped else []
    data = data.loc[valid].copy()
    for name in IDENTIFIER_COLUMNS:
        data[name] = data[name].astype(str)
    data["correlation_x_source_return"] = data["correlation_60d"] * data["source_return_1d"]
    data["correlation_x_event_return"] = data["correlation_60d"] * data["source_event_return"]
    for name in TEXT_FEATURES:
        data[f"source_event_x_{name}"] = data["source_event_return"] * data[name]
    return data.sort_values(["decision_date", "event_id", "target_ticker"], kind="stable").reset_index(drop=True), warnings


def _date(value: pd.Timestamp) -> str:
    return value.date().isoformat()


def _sample_audit(data: pd.DataFrame) -> dict:
    if data.empty:
        return {"n_rows": 0, "n_events": 0, "n_decision_dates": 0,
                "decision_start": None, "decision_end": None, "latest_exit": None}
    return {
        "n_rows": len(data),
        "n_events": int(data["event_id"].nunique()),
        "n_decision_dates": int(data["decision_date"].nunique()),
        "decision_start": _date(data["decision_date"].min()),
        "decision_end": _date(data["decision_date"].max()),
        "latest_exit": _date(data["exit_date"].max()),
    }


def _calibrate(predictions: np.ndarray, residuals: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lower, upper = np.quantile(residuals, [0.1, 0.9])
    ordered = np.sort(residuals)
    positive_counts = len(residuals) - np.searchsorted(ordered, -predictions, side="right")
    probabilities = (positive_counts + 0.5) / (len(residuals) + 1)
    return predictions + lower, predictions + upper, probabilities


def _metrics(predictions: list[dict]) -> list[dict]:
    actual = np.array([row["realized_excess_return"] for row in predictions])
    metrics = []
    for model_id, name, _ in MODEL_SPECS:
        model_rows = [row["models"][model_id] for row in predictions]
        forecast = np.array([row["prediction"] for row in model_rows])
        lower = np.array([row["lower_80"] for row in model_rows])
        upper = np.array([row["upper_80"] for row in model_rows])
        probabilities = np.array([row["p_positive"] for row in model_rows])
        directions = np.array([row["trade_direction"] for row in model_rows])
        active = directions != 0
        directional = np.abs(forecast) > 1e-12
        gross = np.array([row["gross_trade_return"] for row in model_rows])
        net = np.array([row["net_trade_return"] for row in model_rows])
        metrics.append({
            "model_id": model_id,
            "name": name,
            "n": len(actual),
            "mae": float(np.abs(forecast - actual).mean()),
            "rmse": float(np.sqrt(np.square(forecast - actual).mean())),
            "directional_accuracy": float((np.sign(forecast[directional]) == np.sign(actual[directional])).mean()) if directional.any() else None,
            "interval_coverage": float(((actual >= lower) & (actual <= upper)).mean()),
            "brier": float(np.square(probabilities - (actual > 0)).mean()),
            "mean_gross_trade_return": float(gross[active].mean()) if active.any() else None,
            "mean_net_trade_return": float(net[active].mean()) if active.any() else None,
            "n_trades": int(active.sum()),
            "fold_count": len({row["fold_id"] for row in predictions}),
        })
    return metrics


def _comparisons(predictions: list[dict], draws: int = 1000) -> list[dict]:
    actual = np.array([row["realized_excess_return"] for row in predictions])
    baseline = np.array([row["models"]["price_only"]["prediction"] for row in predictions])
    dates = np.array([row["decision_date"] for row in predictions])
    unique_dates, cluster_ids = np.unique(dates, return_inverse=True)
    counts = np.bincount(cluster_ids)
    rng = np.random.default_rng(42)
    samples = rng.integers(0, len(unique_dates), size=(draws, len(unique_dates)))
    results = []
    for model_id in ("network", "event_conditioned"):
        forecast = np.array([row["models"][model_id]["prediction"] for row in predictions])
        improvement = np.abs(baseline - actual) - np.abs(forecast - actual)
        sums = np.bincount(cluster_ids, weights=improvement)
        bootstrap = sums[samples].sum(axis=1) / counts[samples].sum(axis=1)
        lower, upper = np.quantile(bootstrap, [0.025, 0.975])
        results.append({
            "model_id": model_id,
            "baseline_model_id": "price_only",
            "mae_improvement": float(improvement.mean()),
            "ci_lower": float(lower),
            "ci_upper": float(upper),
            "confidence_level": 0.95,
            "bootstrap_unit": "decision_date",
            "n_clusters": len(unique_dates),
            "bootstrap_draws": draws,
            "note": "Exploratory date-cluster bootstrap; overlapping horizons across dates and model selection are not fully accounted for.",
        })
    return results


def run_study(
    frame: pd.DataFrame,
    min_train_rows: int = 100,
    cost_bps: float = 20,
    horizon_days: int = 5,
) -> dict:
    if not np.isfinite(cost_bps) or cost_bps < 0:
        raise ValueError("cost_bps must be finite and nonnegative")
    if not isinstance(min_train_rows, int) or min_train_rows < 1:
        raise ValueError("min_train_rows must be a positive integer")
    if not isinstance(horizon_days, int) or horizon_days < 1:
        raise ValueError("horizon_days must be a positive integer")

    minimum = max(100, min_train_rows)
    cost = float(cost_bps / 10_000)
    result = {
        "status": "insufficient_data",
        "warnings": [],
        "model_definitions": [{"model_id": model_id, "name": name, "features": list(features)} for model_id, name, features in MODEL_SPECS],
        "predictions": [],
        "metrics": [],
        "folds": [],
        "comparisons": [],
        "config": {
            "fold_frequency": "quarterly",
            "min_train_rows": minimum,
            "min_train_events": MIN_FIT_EVENTS,
            "calibration_fraction_of_prior_dates": 0.2,
            "min_calibration_rows": MIN_CALIBRATION_ROWS,
            "ridge_penalty": RIDGE_PENALTY,
            "nominal_interval_coverage": 0.8,
            "cost_bps": float(cost_bps),
            "horizon_days": horizon_days,
            "return_unit": "decimal",
            "trade_return_basis": "signed realized excess return; roundtrip cost charged to each active event trade",
            "interval_method": "Empirical calibration residual quantiles; coverage is not guaranteed.",
            "probability_method": "Smoothed empirical fraction of calibration residuals producing a positive excess return.",
            "portfolio_note": "Event trade averages are not a capital-constrained portfolio backtest; no annualized Sharpe is estimated.",
        },
        "sample": {"input_rows": len(frame), "valid_rows": 0, "predicted_rows": 0},
    }
    if frame.empty:
        result["warnings"].append("No completed event outcomes are available for walk-forward evaluation.")
        return result

    data, warnings = _prepare_frame(frame)
    result["warnings"].extend(warnings)
    result["sample"]["valid_rows"] = len(data)
    if data.empty:
        result["warnings"].append("No valid completed event rows remain.")
        return result
    if not data["text_available"].any():
        result["warnings"].append("No text features are available; the event-conditioned model is equivalent to the network model.")

    periods = data["decision_date"].dt.to_period("Q")
    for period in sorted(periods.unique()):
        fold_start = period.start_time.normalize()
        fold_end = (period + 1).start_time.normalize()
        test = data.loc[periods.eq(period)]
        preceding = data.loc[data["decision_date"].lt(fold_start)]
        prior = preceding.loc[preceding["exit_date"].lt(fold_start)]
        dates = np.sort(prior["decision_date"].unique())
        audit = {
            "fold_id": str(period),
            "status": "skipped",
            "fold_start": _date(fold_start),
            "fold_end_exclusive": _date(fold_end),
            "prior": _sample_audit(prior),
            "test": _sample_audit(test),
            "excluded_unmatured_prior_rows": len(preceding) - len(prior),
            "calibration_start": None,
            "fit": _sample_audit(pd.DataFrame()),
            "calibration": _sample_audit(pd.DataFrame()),
            "purged_fit_rows": 0,
        }
        result["folds"].append(audit)
        if len(dates) < MIN_CALIBRATION_DATES + 1:
            audit["reason"] = "Too few prior decision dates for a separate calibration period."
            continue
        split = min(len(dates) - 1, int(np.floor(0.8 * len(dates))))
        calibration_start = pd.Timestamp(dates[split])
        fit_candidates = prior.loc[prior["decision_date"].lt(calibration_start)]
        fit = fit_candidates.loc[fit_candidates["exit_date"].lt(calibration_start)]
        calibration = prior.loc[prior["decision_date"].ge(calibration_start)]
        audit.update({
            "calibration_start": _date(calibration_start),
            "fit": _sample_audit(fit),
            "calibration": _sample_audit(calibration),
            "purged_fit_rows": len(fit_candidates) - len(fit),
        })
        if len(fit) < minimum or fit["event_id"].nunique() < MIN_FIT_EVENTS:
            audit["reason"] = f"Need at least {minimum} purged fit rows and {MIN_FIT_EVENTS} distinct fit events."
            continue
        if (len(calibration) < MIN_CALIBRATION_ROWS
                or calibration["event_id"].nunique() < MIN_CALIBRATION_EVENTS
                or calibration["decision_date"].nunique() < MIN_CALIBRATION_DATES):
            audit["reason"] = "Need at least 20 calibration rows, 5 distinct events, and 5 decision dates."
            continue

        audit["status"] = "evaluated"
        fit_y = fit["realized_excess_return"].to_numpy(dtype=float)
        calibration_y = calibration["realized_excess_return"].to_numpy(dtype=float)
        actual = test["realized_excess_return"].to_numpy(dtype=float)
        fold_predictions = []
        for row in test.itertuples(index=False):
            fold_predictions.append({
                **{name: getattr(row, name) for name in IDENTIFIER_COLUMNS},
                **{name: _date(getattr(row, name)) for name in DATE_COLUMNS},
                **{name: float(getattr(row, name)) for name in RETURN_COLUMNS},
                "text_available": bool(row.text_available),
                "fold_id": str(period),
                "models": {},
            })

        for model_id, _, features in MODEL_SPECS:
            if features:
                model = _fit_ridge(fit[list(features)].to_numpy(dtype=float), fit_y)
                forecast = model.predict(test[list(features)].to_numpy(dtype=float))
                residuals = calibration_y - model.predict(calibration[list(features)].to_numpy(dtype=float))
            else:
                forecast = np.zeros(len(test))
                residuals = calibration_y.copy()
            lower, upper, probabilities = _calibrate(forecast, residuals)
            directions = np.where(np.abs(forecast) > cost, np.sign(forecast), 0).astype(int)
            gross = directions * actual
            net = gross - np.abs(directions) * cost
            for i, row in enumerate(fold_predictions):
                row["models"][model_id] = {
                    "prediction": float(forecast[i]),
                    "lower_80": float(lower[i]),
                    "upper_80": float(upper[i]),
                    "p_positive": float(probabilities[i]),
                    "trade_direction": int(directions[i]),
                    "gross_trade_return": float(gross[i]),
                    "net_trade_return": float(net[i]),
                }
        result["predictions"].extend(fold_predictions)

    if not result["predictions"]:
        result["warnings"].append("No quarter has enough matured training and calibration data; no model results were generated.")
        return result
    result["status"] = "ok"
    result["sample"]["predicted_rows"] = len(result["predictions"])
    result["metrics"] = _metrics(result["predictions"])
    result["comparisons"] = _comparisons(result["predictions"])
    return result
