from datetime import date

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

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


def test_empty_api_does_not_fabricate_results(monkeypatch, tmp_path):
    from relay import server
    monkeypatch.setattr(server, "DATA_DIR", tmp_path)
    monkeypatch.setattr(server, "PILOT_PATH", tmp_path / "absent.json")
    client = TestClient(server.app)
    payload = client.get("/api/research").json()
    assert payload["study"]["status"] == "insufficient_data"
    assert payload["events"] == []
    assert payload["benchmarks"] == []
    assert client.post("/api/research/run").status_code == 422
    assert client.get("/api/research/predictions.csv").status_code == 404
    assert client.post("/api/research/run", headers={"Origin": "https://unrelated.test"}).status_code == 403


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
