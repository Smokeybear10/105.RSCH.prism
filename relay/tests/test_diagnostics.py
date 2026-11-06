import copy
import json

import numpy as np
import pytest

from relay.diagnostics import FIXED_MODELS, diagnose


def prediction(event_id, entry, exit_date, actual=0.0, target="BBB", source="AAA", adaptive=False):
    models = (*FIXED_MODELS, "adaptive") if adaptive else FIXED_MODELS
    return {
        "event_id": event_id, "source_ticker": source, "target_ticker": target,
        "fold_id": "2020Q1", "decision_date": "2020-01-01", "entry_date": entry, "exit_date": exit_date,
        "realized_excess_return": actual,
        "models": {name: {"prediction": 0.0, "lower_80": -0.01, "upper_80": 0.01, "p_positive": 0.5} for name in models},
    }


def test_interval_score_penalizes_misses_and_includes_boundaries():
    rows = [prediction(str(i), "2020-01-02", "2020-01-06", actual)
            for i, actual in enumerate((-0.02, -0.01, 0.0, 0.01, 0.03))]
    report = diagnose(rows)
    metric = report["metrics"][0]
    assert metric["mae"] == pytest.approx(0.014)
    assert metric["rmse"] == pytest.approx(np.sqrt(0.0015 / 5))
    assert metric["brier"] == pytest.approx(0.25)
    assert metric["interval_coverage"] == pytest.approx(0.6)
    assert metric["mean_interval_width"] == pytest.approx(0.02)
    assert metric["interval_score"] == pytest.approx((0.12 + 0.02 + 0.02 + 0.02 + 0.22) / 5)
    assert report["ablations"][0]["status"] == "insufficient_clusters"
    assert report["ablations"][0]["ci_lower"] is None
    json.dumps(report, allow_nan=False)


def test_joint_paired_bootstrap_preserves_chains_and_row_weights():
    rows = [
        prediction("a", "2020-01-02", "2020-01-06", 0.10),
        prediction("a", "2020-01-02", "2020-01-06", 0.20, target="CCC"),
        prediction("b", "2020-01-06", "2020-01-10", 0.30),
        prediction("c", "2020-01-10", "2020-01-14", 0.40),
        prediction("d", "2020-01-20", "2020-01-24", -0.50),
        prediction("e", "2020-02-03", "2020-02-07", 0.60),
    ]
    for row in rows:
        row["models"]["price_only"]["prediction"] = 0.01
        row["models"]["network"]["prediction"] = 0.02
        row["models"]["event_conditioned"]["prediction"] = 0.03
    report = diagnose(rows, draws=200)
    assert report["sample"]["n_holding_window_clusters"] == 3
    draws = np.random.default_rng(42).integers(0, 3, size=(200, 3))
    improvement = np.array([0.04, -0.01, 0.01])
    counts = np.array([4, 1, 1])
    expected = np.quantile(improvement[draws].sum(axis=1) / counts[draws].sum(axis=1), [0.025, 0.975])
    for comparison in report["ablations"][:3]:
        assert comparison["mae_improvement"] == pytest.approx(0.04 / 6)
        assert comparison["ci_lower"] == pytest.approx(expected[0])
        assert comparison["ci_upper"] == pytest.approx(expected[1])
    assert report == diagnose(list(reversed(rows)), draws=200)


def test_all_slices_reconcile_to_global_metrics_without_mutating_rows():
    rows = [prediction("a", "2020-01-02", "2020-01-06", 0.02, adaptive=True),
            prediction("b", "2021-01-04", "2021-01-08", -0.03, source="CCC", target="DDD", adaptive=True)]
    rows[1]["decision_date"] = "2021-01-01"
    rows[1]["fold_id"] = "2021Q1"
    before = copy.deepcopy(rows)
    report = diagnose(rows)
    assert rows == before
    assert len(report["metrics"]) == 5
    assert report["ablations"][-1]["model_id"] == "adaptive"
    assert [item["value"] for item in report["slices"] if item["dimension"] == "year"] == [2020, 2021]
    for dimension in ("year", "source_ticker", "target_ticker"):
        groups = [group for group in report["slices"] if group["dimension"] == dimension]
        assert sum(group["n_predictions"] for group in groups) == 2
        for metric in report["metrics"]:
            per_group = [next(m for m in group["metrics"] if m["model_id"] == metric["model_id"]) for group in groups]
            for key in ("mae", "brier", "interval_coverage", "mean_interval_width", "interval_score"):
                assert sum(m[key] * m["n_predictions"] for m in per_group) / 2 == pytest.approx(metric[key])


@pytest.mark.parametrize("field,value", [("prediction", float("nan")), ("lower_80", float("inf")),
                                        ("upper_80", "0.1"), ("p_positive", -0.1), ("p_positive", 1.1)])
def test_invalid_forecasts_are_rejected(field, value):
    row = prediction("a", "2020-01-02", "2020-01-06")
    row["models"]["network"][field] = value
    with pytest.raises(ValueError):
        diagnose([row])


def test_reversed_bounds_and_missing_forecast_fields_are_rejected():
    row = prediction("a", "2020-01-02", "2020-01-06")
    row["models"]["network"]["lower_80"] = 1
    with pytest.raises(ValueError, match="lower bound"):
        diagnose([row])
    del row["models"]["network"]["lower_80"]
    with pytest.raises(ValueError, match="lower_80"):
        diagnose([row])


def test_incomplete_and_mismatched_models_are_rejected():
    row = prediction("a", "2020-01-02", "2020-01-06")
    other = prediction("b", "2020-01-20", "2020-01-24", adaptive=True)
    with pytest.raises(ValueError, match="same models"):
        diagnose([row, other])
    del row["models"]["network"]
    with pytest.raises(ValueError, match="all four"):
        diagnose([row])


def test_duplicate_and_inconsistent_event_metadata_are_rejected():
    row = prediction("a", "2020-01-02", "2020-01-06")
    with pytest.raises(ValueError, match="Duplicate"):
        diagnose([row, row])
    other = prediction("a", "2020-01-03", "2020-01-06", target="CCC")
    with pytest.raises(ValueError, match="Conflicting"):
        diagnose([row, other])


@pytest.mark.parametrize("rows", [[], None, [None]])
def test_empty_and_invalid_input_is_rejected(rows):
    with pytest.raises(ValueError):
        diagnose(rows)


@pytest.mark.parametrize("field,value", [("event_id", ""), ("decision_date", "2020-01-02"),
                                        ("entry_date", "invalid"), ("exit_date", "2020-01-01"),
                                        ("realized_excess_return", float("nan"))])
def test_invalid_identity_dates_and_outcomes_are_rejected(field, value):
    row = prediction("a", "2020-01-02", "2020-01-06")
    row[field] = value
    with pytest.raises(ValueError):
        diagnose([row])


@pytest.mark.parametrize("draws", [0, 1, -1, 2.5, True])
def test_invalid_bootstrap_draws_are_rejected(draws):
    with pytest.raises(ValueError, match="Bootstrap draws"):
        diagnose([prediction("a", "2020-01-02", "2020-01-06")], draws=draws)
