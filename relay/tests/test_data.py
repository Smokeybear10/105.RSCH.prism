from datetime import date

import numpy as np
import pandas as pd
import pytest

from relay.config import COMPANIES, StudyConfig
from relay.features import build_panel
from relay.ingest import extract_release, text_features


def inputs():
    dates = pd.bdate_range("2020-01-01", periods=200)
    prices = pd.concat([
        pd.DataFrame({"ticker": ticker, "date": dates, "adj_close": 100 * np.exp(np.cumsum(0.001 + np.sin(np.arange(200) + n) * 0.003)),
                      "adj_open": 100 * np.exp(np.cumsum(0.001 + np.sin(np.arange(200) + n) * 0.003)) / 1.001})
        for n, ticker in enumerate([*COMPANIES, "SPY", "SOXX"])
    ], ignore_index=True)
    event = {"event_id": "AMD_original", "source_ticker": "AMD", "filing_date": dates[100].date().isoformat(),
             **text_features("Revenue increased on strong demand. We raised our guidance.")}
    return prices, [event], dates


def test_timing_and_five_session_return():
    prices, events, dates = inputs()
    frame, _ = build_panel(prices, events, StudyConfig(start="2020-01-01", end="2020-12-31"))
    row = frame[frame.target_ticker == "NVDA"].iloc[0]
    assert row.decision_date == dates[101].date().isoformat()
    assert row.entry_date == dates[102].date().isoformat()
    assert row.exit_date == dates[106].date().isoformat()
    target = prices[prices.ticker == "NVDA"].reset_index(drop=True)
    assert row.realized_return == pytest.approx(target.loc[106, "adj_close"] / target.loc[102, "adj_open"] - 1)


def test_cutoff_never_allows_future_outcomes():
    prices, events, dates = inputs()
    frame, _ = build_panel(prices, events, StudyConfig(start="2020-01-01", end=dates[103].date()))
    assert frame.empty


def test_future_price_changes_do_not_change_features():
    prices, events, dates = inputs()
    config = StudyConfig(start="2020-01-01", end="2020-12-31")
    before, _ = build_panel(prices, events, config)
    prices.loc[prices.date > dates[101], ["adj_open", "adj_close"]] *= 2
    after, _ = build_panel(prices, events, config)
    features = ["source_event_return", "target_return_1d", "target_momentum_20d", "target_vol_20d", "correlation_60d"]
    pd.testing.assert_frame_equal(before[features], after[features])


@pytest.mark.parametrize("hour", ["08:00:00", "16:30:00"])
def test_filing_day_timestamps_wait_for_a_complete_later_session(hour):
    prices, events, dates = inputs()
    events[0]["event_at"] = f"{events[0]['filing_date']}T{hour}-04:00"
    frame, _ = build_panel(prices, events, StudyConfig(start="2020-01-01", end="2020-12-31"))
    assert frame["decision_date"].unique().tolist() == [dates[101].date().isoformat()]
    assert frame["entry_date"].unique().tolist() == [dates[102].date().isoformat()]


def test_later_acceptance_timestamp_cannot_use_an_earlier_decision_close():
    prices, events, dates = inputs()
    events[0]["event_at"] = f"{dates[102].date().isoformat()}T16:30:00-04:00"
    frame, audit = build_panel(prices, events, StudyConfig(start="2020-01-01", end="2020-12-31"))
    assert frame["decision_date"].unique().tolist() == [dates[103].date().isoformat()]
    assert frame["entry_date"].unique().tolist() == [dates[104].date().isoformat()]
    assert frame["information_date"].unique().tolist() == [dates[102].date().isoformat()]
    assert audit["source_timing"]["acceptance_date_after_filing_date"] == 1


def test_event_payload_keeps_filing_date_and_separate_information_clock():
    from relay.study import make_payload

    event = {"event_id": "AMD_release", "filing_date": "2020-05-20", "information_date": "2020-05-22",
             "event_at": "2020-05-22T16:30:00-04:00", "text_available": 1, "headline": "Original filing",
             "evidence_url": "https://www.sec.gov/original", "excerpt": "Source text"}
    row = {"event_id": event["event_id"], "source_ticker": "AMD", "target_ticker": "NVDA", "event_type": "earnings",
           "decision_date": "2020-05-26", "entry_date": "2020-05-27", "exit_date": "2020-06-02",
           "realized_excess_return": 0.02, "fold_id": "2020Q2", "models": {
               "event_conditioned": {"prediction": 0.01, "lower_80": -0.02, "upper_80": 0.04, "p_positive": 0.6},
               "price_only": {"prediction": 0.005}}}
    payload = make_payload({"predictions": [row], "metrics": [], "folds": [], "status": "ok"},
                           [event], {}, StudyConfig(start="2020-01-01", end="2020-12-31"), [])
    observed = payload["events"][0]
    assert observed["date"] == "2020-05-20"
    assert observed["information_date"] == "2020-05-22"
    assert observed["acceptance_timestamp"] == event["event_at"]


@pytest.mark.parametrize("timestamp", ["not a timestamp", "2020-05-20T16:30:00"])
def test_unusable_acceptance_timestamp_is_not_silently_ignored(timestamp):
    prices, events, _ = inputs()
    events[0]["event_at"] = timestamp
    with pytest.raises(ValueError, match="acceptance timestamp"):
        build_panel(prices, events, StudyConfig(start="2020-01-01", end="2020-12-31"))


def test_duplicate_prices_are_rejected():
    prices, events, _ = inputs()
    with pytest.raises(ValueError, match="Duplicate"):
        build_panel(pd.concat([prices, prices.iloc[:1]]), events, StudyConfig(start="2020-01-01", end="2020-12-31"))


def test_text_features_do_not_invent_missing_evidence():
    missing = text_features("")
    assert missing["text_available"] == 0
    assert missing["event_type"] == "earnings"
    assert all(missing[f"{name}_signal"] == 0 for name in ["demand", "pricing", "competitive", "guidance"])
    features = text_features("Revenue increased with strong demand. We lowered our guidance.")
    assert features["demand_signal"] > 0
    assert features["guidance_signal"] < 0


def test_text_features_exclude_disclaimer_sections_and_resume_at_outlook():
    features = text_features("Revenue increased on strong demand.\nForward-Looking Statements\n"
                             "We may lose a competitive advantage and face pricing pressure.\n"
                             "Business Outlook\nWe raised our guidance.")
    assert features["demand_signal"] > 0
    assert features["guidance_signal"] > 0
    assert features["competitive_signal"] == 0
    assert features["pricing_signal"] == 0
    assert features["text_audit"]["boilerplate_sections"] == 1
    assert features["text_audit"]["boilerplate_characters_removed"] > 0


def test_text_features_keep_outlook_references_but_reject_negated_positive_phrases():
    features = text_features('See "Forward-Looking Statements" below. We raised our guidance. '
                             'We may lose a competitive advantage. We do not expect stronger demand.')
    assert features["guidance_signal"] > 0
    assert features["competitive_signal"] == 0
    assert features["demand_signal"] == 0
    assert features["text_audit"]["boilerplate_sections"] == 0


def test_unheaded_disclaimer_removes_only_its_own_paragraph():
    features = text_features("Revenue increased.\nThis press release contains forward-looking statements "
                             "and risks including pricing pressure.\nWe raised our guidance.")
    assert features["demand_signal"] > 0
    assert features["guidance_signal"] > 0
    assert features["pricing_signal"] == 0
    assert features["text_audit"]["boilerplate_sections"] == 1


def test_text_features_do_not_treat_negated_risks_as_negative_news():
    negated = text_features("There is no pricing pressure. We are not losing market share. Demand did not decrease.")
    assert negated["pricing_signal"] == negated["competitive_signal"] == negated["demand_signal"] == 0
    actual = text_features("We face pricing pressure. We are losing market share. Demand decreased.")
    assert actual["pricing_signal"] < 0
    assert actual["competitive_signal"] < 0
    assert actual["demand_signal"] < 0


def test_cached_preparation_reprocesses_sources_without_network_or_mutating_inputs(monkeypatch, tmp_path):
    import hashlib
    import json
    from relay.ingest import PublicData, prepare_cached_events

    base = "https://www.sec.gov/Archives/edgar/data/1/abc/"
    pages = {
        base + "8k.htm": '<table><tr><td>99.1</td><td><a href="ex99-1.htm">Press release</a></td></tr></table>',
        base + "ex99-1.htm": '<p>Revenue increased.</p><p><b>Forward-Looking Statements</b></p>'
            '<p>We may lose a competitive advantage and face pricing pressure.</p>'
            '<p><b>Business Outlook</b></p><p>We raised our guidance.</p>',
    }
    raw = tmp_path / "raw"
    raw.mkdir()
    for url, body in pages.items():
        (raw / f"{hashlib.sha256(url.encode()).hexdigest()}.json").write_text(json.dumps({"url": url, "body": body}))
    before = {path: path.read_bytes() for path in raw.iterdir()}
    event = {"event_id": "AMD_1", "source_url": base + "8k.htm", "document_kind": "primary_filing",
             **text_features("competitive advantage and pricing pressure")}
    original = json.dumps(event, sort_keys=True)
    monkeypatch.setattr(PublicData, "get", lambda *args: pytest.fail("A rebuild must not fetch source documents"))
    prepared, audit, warnings = prepare_cached_events([event], tmp_path)
    assert prepared[0]["demand_signal"] > 0
    assert prepared[0]["guidance_signal"] > 0
    assert prepared[0]["competitive_signal"] == prepared[0]["pricing_signal"] == 0
    assert prepared[0]["evidence_url"] == base + "ex99-1.htm"
    assert prepared[0]["document_kind"] == "earnings_exhibit"
    assert audit["network_requests"] == 0
    assert audit["reprocessed_exhibits"] == audit["changed_events"] == audit["events_with_boilerplate_removed"] == 1
    assert len(audit["raw_responses"]) == 2
    assert warnings == []
    assert json.dumps(event, sort_keys=True) == original
    assert {path: path.read_bytes() for path in raw.iterdir()} == before
    assert {row["sha256"] for row in audit["raw_responses"]} == {hashlib.sha256(content).hexdigest() for content in before.values()}


@pytest.mark.parametrize("cached_payload", [None, {"url": "https://wrong.example/", "body": "Revenue increased."}])
def test_cached_preparation_never_reuses_unverified_old_features(tmp_path, cached_payload):
    import hashlib
    import json
    from relay.ingest import prepare_cached_events

    url = "https://www.sec.gov/Archives/edgar/data/1/abc/8k.htm"
    if cached_payload is not None:
        raw = tmp_path / "raw"
        raw.mkdir()
        (raw / f"{hashlib.sha256(url.encode()).hexdigest()}.json").write_text(json.dumps(cached_payload))
    event = {"event_id": "AMD_1", "source_url": url, "document_kind": "earnings_exhibit", **text_features("Revenue increased.")}
    prepared, audit, warnings = prepare_cached_events([event], tmp_path)
    assert prepared[0]["text_available"] == prepared[0]["demand_signal"] == 0
    assert prepared[0]["text_source_status"] == "cached_source_unavailable"
    assert prepared[0]["excerpt"] == ""
    assert audit["unavailable_cached_sources"] == 1
    assert audit["reprocessed_exhibits"] == 0
    assert warnings and "not reused" in warnings[0]


def test_release_only_follows_original_filing_links():
    base = "https://www.sec.gov/Archives/edgar/data/1/abc/"
    pages = {base + "8k.htm": '<a href="https://issuer.com/press-release.htm">press release</a><a href="ex99-1.htm">99.1</a>',
             base + "ex99-1.htm": "<html><body>Revenue increased.</body></html>"}

    class Client:
        def get(self, url):
            return pages[url]

    text, url = extract_release(Client(), {"source_url": base + "8k.htm"})
    assert url == base + "ex99-1.htm"
    assert text == "Revenue increased."


def test_release_resolves_unlinked_exhibit_from_original_filing_index():
    accession = "0000002488-17-000017"
    base = "https://www.sec.gov/Archives/edgar/data/2488/000000248817000017/"
    pages = {
        base + "8k.htm": "<table><tr><td>99.1</td><td>Press release dated January 31, 2017</td></tr>"
                          "<tr><td>99.2</td><td>CFO commentary</td></tr></table>",
        base + accession + "-index.html": "<table><tr><th>Seq</th><th>Description</th><th>Document</th><th>Type</th></tr>"
            '<tr><td>3</td><td>CFO commentary</td><td><a href="q416992.htm">q416992.htm</a></td><td>EX-99.2</td></tr>'
            '<tr><td>2</td><td>EXHIBIT 99.1</td><td><a href="/Archives/edgar/data/2488/000000248817000017/q416991.htm">q416991.htm</a></td><td>EX-99.1</td></tr></table>',
        base + "q416991.htm": "<html><head><title>ignored</title></head><body>Revenue increased.</body></html>",
    }
    requested = []

    class Client:
        def get(self, url):
            requested.append(url)
            return pages[url]

    record = {"source_url": base + "8k.htm", "accession": accession}
    assert extract_release(Client(), record) == ("Revenue increased.", base + "q416991.htm")
    assert record["source_url"] == base + "8k.htm"
    assert requested == [base + "8k.htm", base + accession + "-index.html", base + "q416991.htm"]


def test_release_keeps_bare_exhibit_99_news_release_links():
    base = "https://www.sec.gov/Archives/edgar/data/97476/000156459017019876/"
    pages = {
        base + "8k.htm": '<table><tr><td>99</td><td><a href="txn-ex99_6.htm">Registrant’s News Release</a></td></tr></table>',
        base + "txn-ex99_6.htm": "<html><body>Quarterly financial results.</body></html>",
    }

    class Client:
        def get(self, url):
            return pages[url]

    assert extract_release(Client(), {"source_url": base + "8k.htm"}) == ("Quarterly financial results.", base + "txn-ex99_6.htm")


@pytest.mark.parametrize("exhibit,description,href", [
    ("EX-99.2", "Press release", "release.htm"),
    ("EX-99.1", "CFO commentary", "commentary.htm"),
    ("EX-99.1", "Investor presentation", "presentation.htm"),
    ("EX-99.1", "Press release", "https://issuer.com/current-earnings.htm"),
    ("EX-99.1", "Press release", "../other-filing/release.htm"),
])
def test_release_index_rejects_wrong_exhibit_nonrelease_and_external_documents(exhibit, description, href):
    accession = "0000002488-17-000017"
    base = "https://www.sec.gov/Archives/edgar/data/2488/000000248817000017/"
    pages = {
        base + "8k.htm": "<table><tr><td>99.1</td><td>Press release</td></tr></table>",
        base + accession + "-index.html": f'<table><tr><td>2</td><td>{description}</td><td><a href="{href}">document</a></td><td>{exhibit}</td></tr></table>',
    }

    class Client:
        def get(self, url):
            return pages[url]

    text, url = extract_release(Client(), {"source_url": base + "8k.htm", "accession": accession})
    assert text == ""
    assert url == base + "8k.htm"
    assert text_features(text)["text_available"] == 0


def test_collect_preserves_event_when_release_is_unresolved(monkeypatch, tmp_path):
    import json
    from relay import ingest

    record = {"event_id": "AMD_original", "source_ticker": "AMD", "source_url": "https://www.sec.gov/original/8k.htm"}
    prices, _, _ = inputs()
    monkeypatch.setattr(ingest, "COMPANIES", {"AMD": (2488, "Advanced Micro Devices")})
    monkeypatch.setattr(ingest, "fetch_prices", lambda client, ticker, config: prices[prices.ticker == ticker])
    monkeypatch.setattr(ingest, "filing_records", lambda client, ticker, config: [record.copy()])
    monkeypatch.setattr(ingest, "extract_release", lambda client, event: ("", event["source_url"]))
    _, events, warnings = ingest.collect_data(StudyConfig(start="2020-01-01", end="2020-12-31"), directory=tmp_path, progress=lambda message: None)
    assert len(events) == 1
    assert events[0]["document_kind"] == "release_unavailable"
    assert events[0]["text_available"] == 0
    assert events[0]["evidence_url"] == record["source_url"]
    assert events[0]["source_url"] == record["source_url"]
    assert warnings and "cover filing excluded" in warnings[0]
    assert json.loads((tmp_path / "events.json").read_text()) == events


def test_current_day_is_rejected_as_incomplete():
    from datetime import datetime
    from zoneinfo import ZoneInfo
    with pytest.raises(ValueError, match="current-session"):
        StudyConfig(start=date(2020, 1, 1), end=datetime.now(ZoneInfo("America/New_York")).date())


def test_cached_study_preserves_costs_and_limits_metadata(tmp_path):
    import json
    from relay.study import run_cached_study

    prices, events, dates = inputs()
    later = {**events[0], "event_id": "later", "filing_date": dates[150].date().isoformat()}
    prices.to_parquet(tmp_path / "prices.parquet", index=False)
    (tmp_path / "events.json").write_text(json.dumps([*events, later]))
    config = StudyConfig(start="2020-01-01", end=dates[130].date(), cost_bps=50)
    (tmp_path / "config.json").write_text(config.model_dump_json())
    first = run_cached_study(tmp_path, config=config)
    assert first["summary"]["source_events"] == 1
    assert first["study"]["period_end"] == events[0]["filing_date"]
    assert first["study"]["cost_bps"] == 50
    original = config.model_copy(update={"cost_bps": 20})
    (tmp_path / "config.json").write_text(original.model_dump_json())
    second = run_cached_study(tmp_path)
    assert second["study"]["cost_bps"] == 50


def test_cached_study_fingerprints_the_inputs_it_read(monkeypatch, tmp_path):
    import hashlib
    import json
    from relay import study

    prices, events, _ = inputs()
    prices_path, events_path = tmp_path / "prices.parquet", tmp_path / "events.json"
    prices.to_parquet(prices_path, index=False)
    events_path.write_text(json.dumps(events))
    expected = {"prices": hashlib.sha256(prices_path.read_bytes()).hexdigest(),
                "events": hashlib.sha256(events_path.read_bytes()).hexdigest()}
    original_build = study.build_panel

    def replace_inputs_after_read(prices, events, config):
        prices_path.write_bytes(b"a later collection replaced this file")
        events_path.write_text("[]")
        return original_build(prices, events, config)

    monkeypatch.setattr(study, "build_panel", replace_inputs_after_read)
    result = study.run_cached_study(tmp_path, StudyConfig(start="2020-01-01", end="2020-12-31"))
    assert {key: result["audit"]["fingerprints"][key] for key in expected} == expected


def test_cached_study_writes_new_runs_without_replacing_originals(tmp_path):
    import hashlib
    import json
    from relay.study import run_cached_study

    prices, events, _ = inputs()
    prices.to_parquet(tmp_path / "prices.parquet", index=False)
    (tmp_path / "events.json").write_text(json.dumps(events))
    config = StudyConfig(start="2020-01-01", end="2020-12-31", cost_bps=50)
    (tmp_path / "config.json").write_text(config.model_dump_json())
    run_cached_study(tmp_path)
    originals = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in tmp_path.iterdir() if path.is_file()}
    output = tmp_path / "runs" / "latest"
    result = run_cached_study(tmp_path, config.model_copy(update={"cost_bps": 20}), output_directory=output)
    assert result["study"]["cost_bps"] == 20
    assert {path.name for path in output.iterdir()} == {"study.json", "predictions.csv", "observations.parquet", "study_config.json", "prepared_events.json"}
    assert hashlib.sha256((output / "prepared_events.json").read_bytes()).hexdigest() == result["audit"]["text_extraction"]["prepared_events_sha256"]
    assert {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in originals} == originals
    assert run_cached_study(tmp_path, output_directory=output)["study"]["cost_bps"] == 20
