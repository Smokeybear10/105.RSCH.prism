import json

import numpy as np
import pandas as pd
import pytest

from relay.model import INPUT_FEATURES, MODEL_SPECS, _comparisons, _holding_window_clusters, _metrics, _period_diagnostics, run_study


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


@pytest.mark.parametrize("include_adaptive", [False, True])
def test_future_labels_and_features_cannot_change_earlier_forecasts(event_frame: pd.DataFrame, include_adaptive: bool) -> None:
    original = run_study(event_frame, include_adaptive=include_adaptive)
    changed = event_frame.copy()
    future = changed["decision_date"] >= "2022-07-01"
    changed.loc[future, "realized_excess_return"] += 5
    changed.loc[future, "realized_return"] += 5
    changed.loc[future, "source_return_1d"] *= 100
    altered = run_study(changed, include_adaptive=include_adaptive)
    before = [row for row in original["predictions"] if row["decision_date"] < "2022-07-01"]
    after = [row for row in altered["predictions"] if row["decision_date"] < "2022-07-01"]
    assert before
    assert before == after


@pytest.mark.parametrize("include_adaptive", [False, True])
def test_test_labels_do_not_affect_forecasts_or_calibration(event_frame: pd.DataFrame, include_adaptive: bool) -> None:
    original = run_study(event_frame, include_adaptive=include_adaptive)
    first_fold = next(fold for fold in original["folds"] if fold["status"] == "evaluated"
                      and (not include_adaptive or fold["selection"]["candidate_scores"]))
    changed = event_frame.copy()
    mask = changed["decision_date"].ge(first_fold["fold_start"]) & changed["decision_date"].lt(first_fold["fold_end_exclusive"])
    changed.loc[mask, "realized_excess_return"] = 7
    changed.loc[mask, "realized_return"] = 7
    altered = run_study(changed, include_adaptive=include_adaptive)
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


def test_calibration_outcomes_cannot_choose_adaptive_model(event_frame: pd.DataFrame) -> None:
    original = run_study(event_frame, include_adaptive=True)
    fold = next(row for row in original["folds"] if row.get("selection", {}).get("candidate_scores"))
    changed = event_frame.copy()
    calibration = changed["decision_date"].ge(fold["calibration_start"]) & changed["exit_date"].lt(fold["fold_start"])
    changed.loc[calibration, "realized_excess_return"] += 1
    changed.loc[calibration, "realized_return"] += 1
    altered = run_study(changed, include_adaptive=True)
    after_fold = next(row for row in altered["folds"] if row["fold_id"] == fold["fold_id"])
    assert fold["selection"] == after_fold["selection"]
    before = [row["models"]["adaptive"] for row in original["predictions"] if row["fold_id"] == fold["fold_id"]]
    after = [row["models"]["adaptive"] for row in altered["predictions"] if row["fold_id"] == fold["fold_id"]]
    assert [row["prediction"] for row in before] == [row["prediction"] for row in after]
    assert [row["lower_80"] for row in before] != [row["lower_80"] for row in after]


def test_missing_text_does_not_create_an_artificial_event_lift(event_frame: pd.DataFrame) -> None:
    data = event_frame.copy()
    data["text_available"] = 0
    data["demand_signal"] = np.nan
    report = run_study(data)
    assert report["status"] == "ok"
    assert report["sample"]["valid_rows"] == len(data)
    for row in report["predictions"]:
        assert row["models"]["event_conditioned"]["prediction"] == pytest.approx(row["models"]["network"]["prediction"], abs=1e-12)


def test_annual_diagnostics_use_identical_rows_for_every_baseline(event_frame: pd.DataFrame) -> None:
    report = run_study(event_frame)
    annual = report["period_diagnostics"]
    assert sum(period["n_predictions"] for period in annual) == len(report["predictions"])
    for period in annual:
        rows = [row for row in report["predictions"] if row["decision_date"].startswith(str(period["year"]))]
        assert period["n_predictions"] == len(rows)
        assert period["n_events"] == len({row["event_id"] for row in rows})
        assert period["n_decision_dates"] == len({row["decision_date"] for row in rows})
        assert sum(source["n_predictions"] for source in period["sources"]) == len(rows)
        actual = np.array([row["realized_excess_return"] for row in rows])
        metrics = {metric["model_id"]: metric for metric in period["models"]}
        for model_id, metric in metrics.items():
            forecast = np.array([row["models"][model_id]["prediction"] for row in rows])
            coverage = [row["models"][model_id]["lower_80"] <= row["realized_excess_return"] <= row["models"][model_id]["upper_80"]
                        for row in rows]
            assert metric["mae"] == pytest.approx(np.abs(forecast - actual).mean())
            assert metric["interval_coverage"] == pytest.approx(np.mean(coverage))
        assert {comparison["baseline_model_id"] for comparison in period["comparisons"]} == {"price_only", "zero_excess"}
        for comparison in period["comparisons"]:
            assert comparison["mae_improvement"] == pytest.approx(
                metrics[comparison["baseline_model_id"]]["mae"] - metrics["event_conditioned"]["mae"])
    zero = next(row for row in report["comparisons"] if row["baseline_model_id"] == "zero_excess")
    metrics = {metric["model_id"]: metric for metric in report["metrics"]}
    assert zero["model_id"] == "event_conditioned"
    assert zero["mae_improvement"] == pytest.approx(metrics["zero_excess"]["mae"] - metrics["event_conditioned"]["mae"])
    assert zero["n_clusters"] == len({row["decision_date"] for row in report["predictions"]})


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


def test_duplicate_event_peer_observations_are_rejected(event_frame: pd.DataFrame) -> None:
    duplicated = pd.concat([event_frame, event_frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate event/peer"):
        run_study(duplicated)


@pytest.mark.parametrize("column,offset", [("decision_date", -1), ("entry_date", 1), ("exit_date", 1)])
def test_conflicting_event_dates_are_rejected(event_frame: pd.DataFrame, column: str, offset: int) -> None:
    changed = event_frame.copy()
    changed.loc[1, column] = (pd.Timestamp(changed.loc[1, column]) + pd.Timedelta(days=offset)).date().isoformat()
    with pytest.raises(ValueError, match="Conflicting decision, entry or exit"):
        run_study(changed)


def test_holding_window_clusters_include_shared_endpoints_and_transitive_overlap() -> None:
    windows = [("2025-01-11", "2025-01-12"), ("2025-01-08", "2025-01-10"),
               ("2025-01-03", "2025-01-05"), ("2025-01-13", "2025-01-13"),
               ("2025-01-05", "2025-01-08")]
    rows = [{"entry_date": entry, "exit_date": exit_date} for entry, exit_date in windows]
    assert _holding_window_clusters(rows).tolist() == [1, 0, 0, 2, 0]


def test_overlap_comparison_keeps_entire_holding_blocks_together() -> None:
    windows = [("2025-01-03", "2025-01-04", "2025-01-08"),
               ("2025-01-05", "2025-01-06", "2025-01-10"),
               ("2025-01-09", "2025-01-10", "2025-01-12"),
               ("2025-01-20", "2025-01-21", "2025-01-25")]
    rows = [{"decision_date": decision, "entry_date": entry, "exit_date": exit_date,
             "realized_excess_return": 0,
             "models": {"zero_excess": {"prediction": 0}, "price_only": {"prediction": value + 0.1},
                        "network": {"prediction": value}, "event_conditioned": {"prediction": value}}}
            for (decision, entry, exit_date), value in zip(windows, [0.1, 0.2, 0.3, 0.4], strict=True)]
    comparison = _comparisons(rows, overlap_blocks=True)
    assert comparison == _comparisons(rows, overlap_blocks=True)
    zero = next(row for row in comparison if row["baseline_model_id"] == "zero_excess")
    assert zero["bootstrap_unit"] == "connected_holding_windows"
    assert zero["status"] == "estimated"
    assert zero["n_clusters"] == 2
    assert zero["mae_improvement"] == pytest.approx(-0.25)
    assert zero["ci_lower"] == pytest.approx(-0.4)
    assert zero["ci_upper"] == pytest.approx(-0.2)
    assert _comparisons(rows)[0]["n_clusters"] == 4

    for row in rows:
        row["entry_date"] = "2025-01-01"
        row["exit_date"] = "2025-01-31"
    for result in _comparisons(rows, overlap_blocks=True):
        assert result["status"] == "insufficient_clusters"
        assert result["n_clusters"] == 1
        assert result["ci_lower"] is None
        assert result["ci_upper"] is None


def test_comparisons_include_optional_model_on_identical_rows(event_frame: pd.DataFrame) -> None:
    rows = run_study(event_frame)["predictions"]
    for row in rows:
        row["models"]["adaptive"] = dict(row["models"]["zero_excess"])
    specs = MODEL_SPECS + (("adaptive", "Training-selected model", ()),)
    metrics = {row["model_id"]: row for row in _metrics(rows, specs)}
    assert metrics["adaptive"]["mae"] == metrics["zero_excess"]["mae"]
    comparisons = _comparisons(rows, model_specs=specs, overlap_blocks=True)
    adaptive = [row for row in comparisons if row["model_id"] == "adaptive"]
    assert {row["baseline_model_id"] for row in adaptive} == {"price_only", "event_conditioned", "zero_excess"}
    assert next(row for row in adaptive if row["baseline_model_id"] == "zero_excess")["mae_improvement"] == 0
    for year in _period_diagnostics(rows, specs):
        assert any(row["model_id"] == "adaptive" for row in year["models"])
        assert len([row for row in year["comparisons"] if row["model_id"] == "adaptive"]) == 3
