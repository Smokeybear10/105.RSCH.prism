from __future__ import annotations

from datetime import date
import math

import numpy as np

from relay.model import MODEL_SPECS, _holding_window_clusters


MODEL_NAMES = {model_id: name for model_id, name, _ in MODEL_SPECS} | {
    "adaptive": "Training-selected model",
}
FIXED_MODELS = tuple(model_id for model_id, _, _ in MODEL_SPECS)
FORECAST_FIELDS = ("prediction", "lower_80", "upper_80", "p_positive")
ALPHA = 0.2


def _number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{field} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be a finite number")
    return number


def _validate(predictions: list[dict]) -> tuple[list[dict], tuple[str, ...]]:
    if not isinstance(predictions, list) or not predictions:
        raise ValueError("Diagnostics require a nonempty list of predictions")
    rows = []
    identities = set()
    events = {}
    expected_models = None
    for row in predictions:
        if not isinstance(row, dict):
            raise ValueError("Each prediction must be an object")
        for name in ("event_id", "source_ticker", "target_ticker", "fold_id"):
            if not isinstance(row.get(name), str) or not row[name].strip():
                raise ValueError(f"Prediction requires a nonempty {name}")
        identity = (row["event_id"], row["target_ticker"])
        if identity in identities:
            raise ValueError("Duplicate event/peer predictions")
        identities.add(identity)
        dates = []
        for name in ("decision_date", "entry_date", "exit_date"):
            value = row.get(name)
            try:
                parsed = date.fromisoformat(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Prediction requires an ISO {name}") from exc
            if parsed.isoformat() != value:
                raise ValueError(f"Prediction requires an ISO {name}")
            dates.append(parsed)
        if not dates[0] < dates[1] <= dates[2]:
            raise ValueError("Prediction dates must satisfy decision < entry <= exit")
        event_metadata = (row["source_ticker"], row["fold_id"], *dates)
        if row["event_id"] in events and events[row["event_id"]] != event_metadata:
            raise ValueError("Conflicting dates, source or fold within an event")
        events[row["event_id"]] = event_metadata
        actual = _number(row.get("realized_excess_return"), "realized_excess_return")
        models = row.get("models")
        if not isinstance(models, dict) or set(models) not in (set(FIXED_MODELS), set(MODEL_NAMES)):
            raise ValueError("Predictions require all four fixed models and optionally adaptive")
        model_ids = tuple(model_id for model_id in MODEL_NAMES if model_id in models)
        if expected_models is not None and model_ids != expected_models:
            raise ValueError("Every prediction must contain the same models")
        expected_models = model_ids
        forecasts = {}
        for model_id in model_ids:
            forecast = models[model_id]
            if not isinstance(forecast, dict):
                raise ValueError(f"{model_id} forecast must be an object")
            values = {name: _number(forecast.get(name), f"{model_id}.{name}") for name in FORECAST_FIELDS}
            if values["lower_80"] > values["upper_80"]:
                raise ValueError(f"{model_id} interval lower bound exceeds upper bound")
            if not 0 <= values["p_positive"] <= 1:
                raise ValueError(f"{model_id} probability must be between zero and one")
            forecasts[model_id] = values
        rows.append({**row, "realized_excess_return": actual, "models": forecasts})
    return sorted(rows, key=lambda row: (row["entry_date"], row["exit_date"], row["event_id"], row["target_ticker"])), expected_models


def _counts(rows: list[dict]) -> dict:
    return {"n_predictions": len(rows), "n_events": len({row["event_id"] for row in rows})}


def _metrics(rows: list[dict], model_ids: tuple[str, ...]) -> list[dict]:
    actual = np.array([row["realized_excess_return"] for row in rows])
    results = []
    for model_id in model_ids:
        forecasts = [row["models"][model_id] for row in rows]
        point, lower, upper, probability = (
            np.array([forecast[name] for forecast in forecasts]) for name in FORECAST_FIELDS
        )
        error = point - actual
        width = upper - lower
        interval_score = width + (2 / ALPHA) * (np.maximum(lower - actual, 0) + np.maximum(actual - upper, 0))
        values = {
            "mae": float(np.abs(error).mean()),
            "rmse": float(np.sqrt(np.square(error).mean())),
            "brier": float(np.square(probability - (actual > 0)).mean()),
            "interval_coverage": float(((actual >= lower) & (actual <= upper)).mean()),
            "mean_interval_width": float(width.mean()),
            "interval_score": float(interval_score.mean()),
        }
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError("Forecast magnitudes produced nonfinite diagnostic metrics")
        results.append({"model_id": model_id, "name": MODEL_NAMES[model_id], **_counts(rows), **values})
    return results


def _comparisons(rows: list[dict], model_ids: tuple[str, ...], clusters: np.ndarray, draws: int) -> list[dict]:
    pairs = [("price_only", "zero_excess"), ("network", "price_only"),
             ("event_conditioned", "network"), ("event_conditioned", "zero_excess")]
    if "adaptive" in model_ids:
        pairs.append(("adaptive", "zero_excess"))
    n_clusters = len(np.unique(clusters))
    counts = np.bincount(clusters)
    samples = np.random.default_rng(42).integers(0, n_clusters, size=(draws, n_clusters)) if n_clusters > 1 else None
    actual = np.array([row["realized_excess_return"] for row in rows])
    results = []
    for model_id, baseline_id in pairs:
        forecast = np.array([row["models"][model_id]["prediction"] for row in rows])
        baseline = np.array([row["models"][baseline_id]["prediction"] for row in rows])
        improvement = np.abs(baseline - actual) - np.abs(forecast - actual)
        lower, upper = None, None
        if samples is not None:
            sums = np.bincount(clusters, weights=improvement)
            bootstrap = sums[samples].sum(axis=1) / counts[samples].sum(axis=1)
            lower, upper = (float(value) for value in np.quantile(bootstrap, [0.025, 0.975]))
        results.append({
            "model_id": model_id, "baseline_model_id": baseline_id,
            "mae_improvement": float(improvement.mean()), "ci_lower": lower, "ci_upper": upper,
            "confidence_level": 0.95, "n_clusters": n_clusters, "bootstrap_draws": draws,
            "bootstrap_unit": "connected_holding_windows",
            "status": "estimated" if n_clusters > 1 else "insufficient_clusters",
        })
    return results


def diagnose(predictions: list[dict], draws: int = 2000) -> dict:
    if isinstance(draws, bool) or not isinstance(draws, int) or draws < 2:
        raise ValueError("Bootstrap draws must be an integer of at least two")
    rows, model_ids = _validate(predictions)
    clusters = _holding_window_clusters(rows)
    slices = []
    for dimension in ("year", "source_ticker", "target_ticker"):
        grouped = {}
        for row in rows:
            value = int(row["decision_date"][:4]) if dimension == "year" else row[dimension]
            grouped.setdefault(value, []).append(row)
        slices.extend([
            {"dimension": dimension, "value": value, **_counts(grouped[value]), "metrics": _metrics(grouped[value], model_ids)}
            for value in sorted(grouped)
        ])
    return {
        "schema_version": 1,
        "scope": "descriptive_research_diagnostics",
        "sample": {
            **_counts(rows), "n_holding_window_clusters": int(len(np.unique(clusters))),
            "decision_start": min(row["decision_date"] for row in rows),
            "decision_end": max(row["decision_date"] for row in rows),
        },
        "protocol": {
            "return_unit": "decimal", "nominal_interval_coverage": 1 - ALPHA,
            "interval_score_alpha": ALPHA, "bootstrap_draws": draws, "bootstrap_seed": 42,
            "comparison_sign": "baseline MAE minus model MAE; positive favors the model",
            "bootstrap_note": "Paired forecast errors share the same resampled holding-window blocks. Dependence between blocks and model development on this inspected sample are not accounted for.",
            "slice_note": "Every observed year, source and target is shown. Slices describe the same predictions without refitting; they are not independent tests or evidence of subgroup significance.",
        },
        "metrics": _metrics(rows, model_ids),
        "ablations": _comparisons(rows, model_ids, clusters, draws),
        "slices": slices,
    }
