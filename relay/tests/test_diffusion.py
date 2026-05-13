import copy
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from relay.diffusion import GUIDANCE_FIELDS, DiffusionLedger, audit_ledger, guidance_signal, prepare_events, run_diffusion_study, select_candidates


@pytest.fixture
def source_ledger() -> dict:
    guide = {
        "metric": "net_sales_growth", "forecast_period": "FY2026", "unit": "percent_yoy",
        "basis": "constant_currency", "business_scope": "consolidated_including_VIZIO",
        "low": 3, "high": 4, "previous_event_id": None, "change_status": "initialization",
    }
    previous = {
        "event_id": "WMT_prior", "source_ticker": "WMT", "available_date": "2025-01-15",
        "source_url": "https://example.com/prior", "event_kind": "baseline_prior_year_results",
        "guidance": [guide],
    }
    event = {
        "event_id": "WMT_update", "source_ticker": "WMT", "available_date": "2025-02-10",
        "source_url": "https://example.com/update", "event_kind": "scheduled_results",
        "guidance": [{**guide, "low": 4, "high": 5, "previous_event_id": "WMT_prior", "change_status": "numeric_revision"}],
    }
    return {
        "metadata": {"study_period": ["2025-02-03", "2025-02-28"]},
        "events": [event], "baseline_events": [previous], "interim_events": [],
        "relationships": [{"source_ticker": "WMT", "target_ticker": "PG", "published_date": "2024-08-05",
                           "evidence_url": "https://example.com/filing", "publication_evidence_url": "https://example.com/index"}],
    }


@pytest.fixture
def daily_prices() -> pd.DataFrame:
    return pd.DataFrame([
        {"ticker": ticker, "date": day.date(), "adj_open": 100., "adj_close": 100.}
        for ticker in ("WMT", "PG", "SPY") for day in pd.bdate_range("2025-02-03", "2025-02-28")
        if day.date().isoformat() != "2025-02-17"
    ])


def test_guidance_requires_same_period_and_prior_public_history(source_ledger: dict) -> None:
    ledger = DiffusionLedger.model_validate(source_ledger)
    assert guidance_signal(ledger.events[0], ledger.all_events)["sign"] == 1
    changed = copy.deepcopy(source_ledger)
    changed["baseline_events"][0]["guidance"][0]["forecast_period"] = "FY2025"
    ledger = DiffusionLedger.model_validate(changed)
    with pytest.raises(ValueError, match="Incomparable"):
        guidance_signal(ledger.events[0], ledger.all_events)
    changed = copy.deepcopy(source_ledger)
    changed["baseline_events"][0]["available_date"] = "2025-02-10"
    ledger = DiffusionLedger.model_validate(changed)
    with pytest.raises(ValueError, match="not known before"):
        guidance_signal(ledger.events[0], ledger.all_events)


def test_known_interim_prevents_reusing_stale_guidance(source_ledger: dict) -> None:
    interim = copy.deepcopy(source_ledger["baseline_events"][0])
    interim.update(event_id="WMT_interim", available_date="2025-02-05", event_kind="interim_investor_meeting")
    source_ledger["interim_events"] = [interim]
    ledger = DiffusionLedger.model_validate(source_ledger)
    with pytest.raises(ValueError, match="Stale"):
        guidance_signal(ledger.events[0], ledger.all_events)


def test_qualitative_and_scope_changed_guidance_abstain(source_ledger: dict) -> None:
    for status in ("qualitative_revision", "scope_changed_revision", "withdrawn"):
        source_ledger["events"][0]["guidance"][0]["change_status"] = status
        ledger = DiffusionLedger.model_validate(source_ledger)
        assert guidance_signal(ledger.events[0], ledger.all_events)["sign"] is None
    source_ledger["events"][0]["guidance"][0].update(low=None, high=None, qualitative_value="slightly_positive")
    ledger = DiffusionLedger.model_validate(source_ledger)
    assert guidance_signal(ledger.events[0], ledger.all_events)["sign"] is None


def test_acquisition_sales_do_not_override_comparable_sales_cut(source_ledger: dict) -> None:
    for event in source_ledger["events"] + source_ledger["baseline_events"]:
        event["source_ticker"] = "LOW"
        event["guidance"][0].update(metric="comparable_sales_growth", basis="comparable_sales",
                                     business_scope="company_comparable_sales")
    source_ledger["baseline_events"][0]["guidance"][0].update(low=0, high=1)
    source_ledger["events"][0]["guidance"][0].update(low=0, high=0)
    sales = {**source_ledger["events"][0]["guidance"][0], "metric": "total_sales", "unit": "USD_bn",
             "basis": "reported", "business_scope": "consolidated", "low": 86, "high": 86,
             "change_status": "scope_changed_revision"}
    source_ledger["events"][0]["guidance"].append(sales)
    ledger = DiffusionLedger.model_validate(source_ledger)
    signal = guidance_signal(ledger.events[0], ledger.all_events)
    assert signal["sign"] == -1
    assert len(signal["components"]) == 1


def test_source_rows_are_not_counted_as_independent_events(source_ledger: dict) -> None:
    source_ledger["relationships"].append({**source_ledger["relationships"][0], "target_ticker": "CLX"})
    audit = audit_ledger(DiffusionLedger.model_validate(source_ledger))
    assert audit["scheduled_events"] == 1
    assert audit["linked_event_rows"] == 2
    assert audit["positive_guidance_events_by_month"] == {"2025-02": 1}


def test_post_event_links_are_not_backfilled(source_ledger: dict) -> None:
    source_ledger["relationships"][0]["published_date"] = "2025-02-11"
    with pytest.raises(ValueError, match="No prior-public relationship"):
        audit_ledger(DiffusionLedger.model_validate(source_ledger))


def test_dates_and_price_features_use_only_pre_entry_information(source_ledger: dict, daily_prices: pd.DataFrame) -> None:
    ledger = DiffusionLedger.model_validate(source_ledger)
    rows = prepare_events(ledger, daily_prices)
    assert rows[0]["reference_date"] == "2025-02-07"
    assert rows[0]["feature_date"] == "2025-02-10"
    assert rows[0]["entry_date"] == "2025-02-11"
    assert rows[0]["exit_date"] == "2025-02-18"
    delayed = prepare_events(ledger, daily_prices, entry_delay_sessions=1)
    assert delayed[0]["feature_date"] == "2025-02-11"
    assert delayed[0]["entry_date"] == "2025-02-12"
    changed = daily_prices.copy()
    future = changed["date"] >= pd.Timestamp("2025-02-11").date()
    changed.loc[future, ["adj_open", "adj_close"]] *= 100
    assert prepare_events(ledger, changed) == rows
    assert select_candidates(rows, "positive_source_price") == []
    assert len(select_candidates(rows, "positive_guidance_revision")) == 1


def test_weekend_and_holiday_use_observed_market_sessions(source_ledger: dict, daily_prices: pd.DataFrame) -> None:
    source_ledger["events"][0]["available_date"] = "2025-02-14"
    daily_prices = daily_prices[daily_prices["date"] != pd.Timestamp("2025-02-17").date()]
    row = prepare_events(DiffusionLedger.model_validate(source_ledger), daily_prices)[0]
    assert row["entry_date"] == "2025-02-18"
    assert row["exit_date"] == "2025-02-24"


def test_missing_price_is_not_silently_dropped_or_filled(source_ledger: dict, daily_prices: pd.DataFrame) -> None:
    missing = (daily_prices["ticker"] == "PG") & (daily_prices["date"] == pd.Timestamp("2025-02-13").date())
    with pytest.raises(ValueError, match="Missing PG bars"):
        prepare_events(DiffusionLedger.model_validate(source_ledger), daily_prices[~missing])


def test_session_missing_from_every_ticker_is_not_treated_as_holiday(source_ledger: dict, daily_prices: pd.DataFrame) -> None:
    missing = daily_prices["date"] == pd.Timestamp("2025-02-13").date()
    with pytest.raises(ValueError, match="SPY calendar mismatch"):
        prepare_events(DiffusionLedger.model_validate(source_ledger), daily_prices[~missing])


def test_duplicate_events_and_prices_fail(source_ledger: dict, daily_prices: pd.DataFrame) -> None:
    source_ledger["events"].append(copy.deepcopy(source_ledger["events"][0]))
    with pytest.raises(ValueError, match="Duplicate event"):
        DiffusionLedger.model_validate(source_ledger)
    source_ledger["events"].pop()
    with pytest.raises(ValueError, match="Duplicate ticker"):
        prepare_events(DiffusionLedger.model_validate(source_ledger), pd.concat([daily_prices, daily_prices.iloc[:1]]))


@pytest.mark.parametrize("ledger_hash_key", ["ledger_sha256", "frozen_ledger_sha256"])
def test_offline_replay_is_complete_reproducible_and_rejects_changed_inputs(
    source_ledger: dict, daily_prices: pd.DataFrame, tmp_path: Path, ledger_hash_key: str,
) -> None:
    ledger_path = tmp_path / "ledger.json"
    ledger_path.write_text(json.dumps(source_ledger))
    prices_path = tmp_path / "prices.parquet"
    daily_prices.to_parquet(prices_path, index=False)
    protocol = {
        "complete_protocol_frozen_before_price_access": True,
        "source_ledger_frozen_sha256": hashlib.sha256(ledger_path.read_bytes()).hexdigest(),
        "execution_settings": {
            "rules": ["all_linked_events", "positive_source_price", "positive_guidance_revision"],
            "entry_delays": [0, 1], "holding_sessions": 5, "cost_grid_bps": [0, 10, 20, 50],
            "initial_cash": 10000, "max_open_positions": 4, "max_new_position_budget_fraction": 0.25,
            "guidance_fields": {key: [list(item) for item in values] for key, values in GUIDANCE_FIELDS.items()},
        },
    }
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol))
    manifest_path = tmp_path / "price-manifest.json"
    manifest = {
        "prices_sha256": hashlib.sha256(prices_path.read_bytes()).hexdigest(),
        ledger_hash_key: protocol["source_ledger_frozen_sha256"],
    }
    manifest_path.write_text(json.dumps(manifest))
    output = tmp_path / "results"
    first = run_diffusion_study(ledger_path, prices_path, protocol_path, output)
    second = run_diffusion_study(ledger_path, prices_path, protocol_path, output)
    assert first["runs"] == second["runs"]
    assert len(first["runs"]) == 24
    assert len(pd.read_csv(output / "summary.csv")) == 24
    assert len(pd.read_csv(output / "event_windows.csv")) == 2
    for run in first["runs"]:
        assert run["result"]["n_trades"] == (0 if run["rule"] == "positive_source_price" else 1)
        assert run["result"]["period_start"] == "2025-02-03"
        assert run["result"]["period_end"] == "2025-02-28"
        assert run["result"]["portfolio_pnl"] <= 0
    assert first["inputs"]["ledger"]["sha256"] == protocol["source_ledger_frozen_sha256"]
    assert first["inputs"]["price_manifest"] == {
        "path": str(manifest_path.resolve()), "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    }
    assert first["implementation"]["study_session_count"] == 19

    ledger_path.write_text(json.dumps({**source_ledger, "changed_after_freeze": True}))
    with pytest.raises(ValueError, match="differs from the frozen protocol"):
        run_diffusion_study(ledger_path, prices_path, protocol_path, output)
    ledger_path.write_text(json.dumps(source_ledger))

    original_prices = prices_path.read_bytes()
    prices_path.write_bytes(original_prices + b"changed after collection")
    with pytest.raises(ValueError, match="Prices differ from the price manifest"):
        run_diffusion_study(ledger_path, prices_path, protocol_path, output)
    prices_path.write_bytes(original_prices)

    manifest_path.write_text(json.dumps({**manifest, ledger_hash_key: "wrong-ledger-hash"}))
    with pytest.raises(ValueError, match="Source ledger differs from the price manifest"):
        run_diffusion_study(ledger_path, prices_path, protocol_path, output)
    manifest_path.unlink()
    with pytest.raises(ValueError, match="price-manifest.json beside the prices file is required"):
        run_diffusion_study(ledger_path, prices_path, protocol_path, output)
    manifest_path.write_text(json.dumps(manifest))

    protocol["execution_settings"]["cost_grid_bps"] = [0]
    protocol_path.write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match="settings differ"):
        run_diffusion_study(ledger_path, prices_path, protocol_path, output)
