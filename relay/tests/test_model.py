import json

import numpy as np
import pandas as pd
import pytest

from relay.model import INPUT_FEATURES, run_study


@pytest.fixture
def event_frame() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2021-01-04", periods=620)
    rows = []
    for i, day in enumerate(dates):
        source_return = rng.normal(0, 0.025)
        for target in ("BBB", "CCC"):
            features = dict(zip(INPUT_FEATURES, rng.normal(0, 0.02, len(INPUT_FEATURES))))
            features.update({
                "source_return_1d": source_return,
                "source_event_return": source_return,
                "correlation_60d": 0.6,
                "text_available": int(i % 3 != 0),
                "target_vol_20d": 0.02,
            })
            excess = 0.6 * source_return + rng.normal(0, 0.004)
            benchmark = rng.normal(0, 0.005)
            rows.append({
                "event_id": f"AAA_{day.date()}",
                "source_ticker": "AAA",
                "target_ticker": target,
                "event_type": "earnings",
                "decision_date": day.date().isoformat(),
                "entry_date": (day + pd.offsets.BDay(1)).date().isoformat(),
                "exit_date": (day + pd.offsets.BDay(6)).date().isoformat(),
                "realized_return": excess + benchmark,
                "benchmark_return": benchmark,
                "realized_excess_return": excess,
                **features,
            })
    return pd.DataFrame(rows)


def test_empty_and_insufficient_data_produce_no_forecasts(event_frame: pd.DataFrame) -> None:
    for data in (pd.DataFrame(), event_frame.head(80), event_frame.drop(columns="source_return_1d")):
        report = run_study(data)
        assert report["status"] == "insufficient_data"
        assert report["predictions"] == []
        assert report["metrics"] == []
        assert report["warnings"]
        json.dumps(report, allow_nan=False)


def test_folds_purge_labels_and_use_separate_calibration(event_frame: pd.DataFrame) -> None:
    report = run_study(event_frame)
    assert report["status"] == "ok"
    evaluated = [fold for fold in report["folds"] if fold["status"] == "evaluated"]
    assert len(evaluated) >= 5
    for fold in evaluated:
        assert fold["fit"]["n_rows"] >= 100
        assert fold["fit"]["n_events"] >= 20
        assert fold["fit"]["latest_exit"] < fold["calibration_start"]
        assert fold["fit"]["decision_end"] < fold["calibration"]["decision_start"]
        assert fold["calibration"]["latest_exit"] < fold["fold_start"]
        assert fold["calibration"]["decision_end"] < fold["test"]["decision_start"]
        assert fold["purged_fit_rows"] > 0
        assert fold["excluded_unmatured_prior_rows"] > 0


def test_future_labels_and_features_cannot_change_earlier_forecasts(event_frame: pd.DataFrame) -> None:
    original = run_study(event_frame)
    changed = event_frame.copy()
    future = changed["decision_date"] >= "2022-07-01"
    changed.loc[future, "realized_excess_return"] += 5
    changed.loc[future, "realized_return"] += 5
    changed.loc[future, "source_return_1d"] *= 100
    altered = run_study(changed)
    before = [row for row in original["predictions"] if row["decision_date"] < "2022-07-01"]
    after = [row for row in altered["predictions"] if row["decision_date"] < "2022-07-01"]
    assert before
    assert before == after


def test_test_labels_do_not_affect_forecasts_or_calibration(event_frame: pd.DataFrame) -> None:
    original = run_study(event_frame)
    first_fold = next(fold for fold in original["folds"] if fold["status"] == "evaluated")
    changed = event_frame.copy()
    mask = changed["decision_date"].ge(first_fold["fold_start"]) & changed["decision_date"].lt(first_fold["fold_end_exclusive"])
    changed.loc[mask, "realized_excess_return"] = 7
    changed.loc[mask, "realized_return"] = 7
    altered = run_study(changed)
    left = [row for row in original["predictions"] if row["fold_id"] == first_fold["fold_id"]]
    right = [row for row in altered["predictions"] if row["fold_id"] == first_fold["fold_id"]]
    fields = ("prediction", "lower_80", "upper_80", "p_positive", "trade_direction")
    for a, b in zip(left, right, strict=True):
        for model_id in a["models"]:
            assert {key: a["models"][model_id][key] for key in fields} == {
                key: b["models"][model_id][key] for key in fields
            }


def test_intervals_probabilities_costs_and_metrics(event_frame: pd.DataFrame) -> None:
    report = run_study(event_frame, cost_bps=20)
    json.dumps(report, allow_nan=False)
    for row in report["predictions"]:
        for model in row["models"].values():
            assert model["lower_80"] <= model["upper_80"]
            assert 0 < model["p_positive"] < 1
            active = abs(model["prediction"]) > 0.002
            assert (model["trade_direction"] != 0) == active
            assert model["gross_trade_return"] == pytest.approx(model["trade_direction"] * row["realized_excess_return"])
            assert model["net_trade_return"] == pytest.approx(model["gross_trade_return"] - (0.002 if active else 0))
        assert row["models"]["zero_excess"]["trade_direction"] == 0
    metrics = {metric["model_id"]: metric for metric in report["metrics"]}
    assert metrics["zero_excess"]["n_trades"] == 0
    assert metrics["zero_excess"]["mean_net_trade_return"] is None
    assert metrics["zero_excess"]["directional_accuracy"] is None
    assert metrics["network"]["mae"] < metrics["price_only"]["mae"]
    for metric in metrics.values():
        assert 0 <= metric["interval_coverage"] <= 1
        assert 0 <= metric["brier"] <= 1
        assert "sharpe" not in metric
    comparison = next(row for row in report["comparisons"] if row["model_id"] == "network")
    assert comparison["mae_improvement"] > 0
    assert comparison["ci_lower"] > 0
    assert comparison["n_clusters"] == len({row["decision_date"] for row in report["predictions"]})


def test_missing_text_does_not_create_an_artificial_event_lift(event_frame: pd.DataFrame) -> None:
    data = event_frame.copy()
    data["text_available"] = 0
    data["demand_signal"] = np.nan
    report = run_study(data)
    assert report["status"] == "ok"
    assert report["sample"]["valid_rows"] == len(data)
    for row in report["predictions"]:
        assert row["models"]["event_conditioned"]["prediction"] == pytest.approx(row["models"]["network"]["prediction"], abs=1e-12)


def test_invalid_rows_are_excluded_and_training_minimum_cannot_be_lowered(event_frame: pd.DataFrame) -> None:
    data = event_frame.copy()
    data.loc[0, "target_return_1d"] = np.inf
    data.loc[1, "exit_date"] = "1900-01-01"
    report = run_study(data, min_train_rows=1)
    assert report["sample"]["valid_rows"] == len(data) - 2
    assert report["config"]["min_train_rows"] == 100
    assert any("Excluded 2 rows" in warning for warning in report["warnings"])


@pytest.mark.parametrize("arguments", [{"cost_bps": -1}, {"cost_bps": float("nan")}, {"horizon_days": 0}, {"min_train_rows": 0}])
def test_invalid_configuration_is_rejected(arguments: dict) -> None:
    with pytest.raises(ValueError):
        run_study(pd.DataFrame(), **arguments)
