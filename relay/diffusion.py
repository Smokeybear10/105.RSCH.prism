from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path
from importlib.metadata import version

import exchange_calendars as xcals
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, model_validator


GUIDANCE_FIELDS = {
    "WMT": [("net_sales_growth", "constant_currency", "consolidated_including_VIZIO")],
    "HD": [("comparable_sales_growth", "comparable_52_week_period", "company_comparable_sales")],
    "LOW": [("comparable_sales_growth", "comparable_sales", "company_comparable_sales")],
    "DE": [("net_sales_growth", "reported_segment_sales", scope) for scope in (
        "production_precision_agriculture", "small_agriculture_turf", "construction_forestry")],
}
RULES = ("all_linked_events", "positive_source_price", "positive_guidance_revision")
ENTRY_DELAYS = (0, 1)
COST_GRID = (0, 10, 20, 50)


def market_sessions(start: date, end: date) -> list[date]:
    return list(xcals.get_calendar("XNYS", start=start, end=end).sessions.date)


class Guidance(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

    metric: str
    forecast_period: str
    unit: str
    basis: str
    business_scope: str
    low: float | None = None
    high: float | None = None
    qualitative_value: str | None = None
    approximate: bool = False
    previous_event_id: str | None = None
    change_status: str

    @model_validator(mode="after")
    def ordered_bounds(self):
        if (self.low is None) != (self.high is None):
            raise ValueError("Guidance must have both bounds or neither")
        if self.low is not None and self.low > self.high:
            raise ValueError("Guidance lower bound exceeds upper bound")
        return self

    @property
    def key(self) -> tuple[str, ...]:
        return self.metric, self.forecast_period, self.unit, self.basis, self.business_scope


class Disclosure(BaseModel):
    event_id: str
    source_ticker: str
    available_date: date
    source_url: str
    event_kind: str
    guidance: list[Guidance]


class Relationship(BaseModel):
    source_ticker: str
    target_ticker: str
    published_date: date
    evidence_url: str
    publication_evidence_url: str


class DiffusionLedger(BaseModel):
    events: list[Disclosure]
    interim_events: list[Disclosure] = []
    baseline_events: list[Disclosure] = []
    relationships: list[Relationship]

    @model_validator(mode="after")
    def unique_records(self):
        ids = [event.event_id for event in self.all_events]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate event IDs")
        edges = [(edge.source_ticker, edge.target_ticker) for edge in self.relationships]
        if len(edges) != len(set(edges)):
            raise ValueError("Duplicate directed relationships")
        if any(source == target for source, target in edges):
            raise ValueError("A customer-supplier edge cannot be a self-link")
        if any(event.event_kind != "scheduled_results" for event in self.events):
            raise ValueError("The trading denominator contains only scheduled result releases")
        return self

    @property
    def all_events(self) -> list[Disclosure]:
        return self.baseline_events + self.interim_events + self.events


def guidance_signal(event: Disclosure, history: list[Disclosure]) -> dict:
    fields = GUIDANCE_FIELDS.get(event.source_ticker)
    if fields is None:
        raise ValueError(f"No frozen guidance mapping for {event.source_ticker}")
    event_by_id = {item.event_id: item for item in history}
    components = []
    for metric, basis, scope in fields:
        matches = [guide for guide in event.guidance
                   if (guide.metric, guide.basis, guide.business_scope) == (metric, basis, scope)]
        if len(matches) != 1:
            components.append({"metric": metric, "scope": scope, "delta": None, "status": "missing_or_ambiguous_field"})
            continue
        guide = matches[0]
        component = {
            "metric": metric, "scope": scope, "forecast_period": guide.forecast_period,
            "low": guide.low, "high": guide.high, "approximate": guide.approximate,
            "previous_event_id": guide.previous_event_id, "delta": None, "status": guide.change_status,
        }
        components.append(component)
        if guide.previous_event_id is None:
            if guide.change_status != "initialization":
                component["status"] = "missing_predecessor"
            continue
        previous = event_by_id.get(guide.previous_event_id)
        if previous is None or previous.source_ticker != event.source_ticker:
            raise ValueError(f"Missing or wrong-source predecessor for {event.event_id}")
        if previous.available_date >= event.available_date:
            raise ValueError(f"Predecessor not known before {event.event_id}")
        prior_guides = [item for item in previous.guidance if item.key == guide.key]
        if len(prior_guides) != 1:
            raise ValueError(f"Incomparable guidance predecessor for {event.event_id}: {guide.key}")
        newer_known = [item for item in history
                       if item.source_ticker == event.source_ticker
                       and previous.available_date < item.available_date < event.available_date
                       and any(value.key == guide.key for value in item.guidance)]
        if newer_known:
            raise ValueError(f"Stale guidance predecessor for {event.event_id}")
        prior = prior_guides[0]
        component.update(previous_low=prior.low, previous_high=prior.high,
                         previous_approximate=prior.approximate)
        if guide.change_status not in {"numeric_revision", "reaffirmed"}:
            continue
        if guide.unit != "percent_yoy" or guide.low is None or prior.low is None:
            component["status"] = "nonnumeric_or_wrong_unit"
            continue
        if guide.qualitative_value is not None or prior.qualitative_value is not None:
            component["status"] = "qualitative"
            continue
        if guide.change_status == "reaffirmed" and (guide.low, guide.high) != (prior.low, prior.high):
            raise ValueError(f"Changed bounds labeled reaffirmed for {event.event_id}")
        component["delta"] = (guide.low + guide.high - prior.low - prior.high) / 2

    deltas = [component["delta"] for component in components]
    if any(delta is None for delta in deltas):
        sign, status = None, "unknown"
    elif any(delta > 0 for delta in deltas) and any(delta < 0 for delta in deltas):
        sign, status = None, "mixed"
    else:
        sign = 1 if any(delta > 0 for delta in deltas) else -1 if any(delta < 0 for delta in deltas) else 0
        status = {1: "positive", -1: "negative", 0: "unchanged"}[sign]
    return {"sign": sign, "status": status, "components": components}


def audit_ledger(ledger: DiffusionLedger) -> dict:
    counts = Counter()
    monthly = Counter()
    positive_monthly = Counter()
    rows = []
    eligible_edges = 0
    for event in sorted(ledger.events, key=lambda item: (item.available_date, item.event_id)):
        signal = guidance_signal(event, ledger.all_events)
        links = [edge for edge in ledger.relationships
                 if edge.source_ticker == event.source_ticker and edge.published_date < event.available_date]
        if not links:
            raise ValueError(f"No prior-public relationship for {event.event_id}")
        counts[signal["status"]] += 1
        month = event.available_date.strftime("%Y-%m")
        monthly[month] += 1
        if signal["sign"] == 1:
            positive_monthly[month] += 1
        eligible_edges += len(links)
        rows.append({"event_id": event.event_id, "source_ticker": event.source_ticker,
                     "available_date": event.available_date.isoformat(), "source_url": event.source_url,
                     "targets": sorted(edge.target_ticker for edge in links), "guidance": signal})
    return {
        "scheduled_events": len(ledger.events), "unique_announcement_dates": len({event.available_date for event in ledger.events}),
        "directed_relationships": len(ledger.relationships),
        "unique_suppliers": len({edge.target_ticker for edge in ledger.relationships}),
        "linked_event_rows": eligible_edges, "guidance_status_counts": dict(counts),
        "announcements_by_month": dict(sorted(monthly.items())),
        "positive_guidance_events_by_month": dict(sorted(positive_monthly.items())), "events": rows,
    }


def validated_prices(prices: pd.DataFrame) -> pd.DataFrame:
    required = {"ticker", "date", "adj_open", "adj_close"}
    if not required <= set(prices.columns):
        raise ValueError(f"Missing price columns: {sorted(required - set(prices.columns))}")
    frame = prices.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.date
    if frame.duplicated(["ticker", "date"]).any():
        raise ValueError("Duplicate ticker/date price rows")
    values = frame[["adj_open", "adj_close"]].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Price bars must be finite and positive")
    observed = set(frame.loc[frame["ticker"] == "SPY", "date"])
    if not observed:
        raise ValueError("SPY session calendar is required")
    expected = set(market_sessions(min(observed), max(observed)))
    if observed != expected:
        missing = sorted(day.isoformat() for day in expected - observed)
        extra = sorted(day.isoformat() for day in observed - expected)
        raise ValueError(f"SPY calendar mismatch; missing sessions {missing}, non-market dates {extra}")
    return frame.sort_values(["date", "ticker"]).reset_index(drop=True)


def prepare_events(ledger: DiffusionLedger, prices: pd.DataFrame, entry_delay_sessions: int = 0,
                   holding_sessions: int = 5) -> list[dict]:
    if entry_delay_sessions not in ENTRY_DELAYS or holding_sessions != 5:
        raise ValueError("This study fixes entry delays to 0/1 and holding period to five sessions")
    frame = validated_prices(prices)
    sessions = sorted(frame.loc[frame["ticker"] == "SPY", "date"].unique())
    if not sessions:
        raise ValueError("SPY session calendar is required")
    lookup = frame.set_index(["ticker", "date"])
    rows = []
    for event in ledger.events:
        first_after = next((i for i, session in enumerate(sessions) if session > event.available_date), None)
        if first_after is None or first_after == 0:
            raise ValueError(f"Missing prior/next session for {event.event_id}")
        entry_index = first_after + entry_delay_sessions
        exit_index = entry_index + holding_sessions - 1
        if exit_index >= len(sessions):
            raise ValueError(f"Incomplete forward window for {event.event_id}")
        before_release = [session for session in sessions if session < event.available_date]
        if not before_release:
            raise ValueError(f"Missing pre-release close for {event.event_id}")
        reference_date = before_release[-1]
        feature_date = sessions[entry_index - 1]
        entry_date, exit_date = sessions[entry_index], sessions[exit_index]
        signal = guidance_signal(event, ledger.all_events)
        links = [edge for edge in ledger.relationships
                 if edge.source_ticker == event.source_ticker and edge.published_date < event.available_date]
        if not links:
            raise ValueError(f"No prior-public relationship for {event.event_id}")
        for ticker in {"SPY", event.source_ticker, *(edge.target_ticker for edge in links)}:
            needed = {reference_date, feature_date, *sessions[entry_index:exit_index + 1]}
            missing = [day.isoformat() for day in sorted(needed) if (ticker, day) not in lookup.index]
            if missing:
                raise ValueError(f"Missing {ticker} bars for {event.event_id}: {missing}")
        source_return = float(lookup.loc[(event.source_ticker, feature_date), "adj_close"] /
                              lookup.loc[(event.source_ticker, reference_date), "adj_close"] - 1)
        market_return = float(lookup.loc[("SPY", feature_date), "adj_close"] /
                              lookup.loc[("SPY", reference_date), "adj_close"] - 1)
        for edge in links:
            target_return = float(lookup.loc[(edge.target_ticker, feature_date), "adj_close"] /
                                  lookup.loc[(edge.target_ticker, reference_date), "adj_close"] - 1)
            rows.append({
                "event_id": event.event_id, "event_ids": [event.event_id], "source_ticker": event.source_ticker,
                "target_ticker": edge.target_ticker, "available_date": event.available_date.isoformat(),
                "relationship_published_date": edge.published_date.isoformat(),
                "source_url": event.source_url, "relationship_url": edge.evidence_url,
                "reference_date": reference_date.isoformat(), "feature_date": feature_date.isoformat(),
                "entry_date": entry_date.isoformat(), "exit_date": exit_date.isoformat(),
                "entry_delay_sessions": entry_delay_sessions, "guidance_sign": signal["sign"],
                "guidance_status": signal["status"], "source_window_return": source_return,
                "source_window_excess_return": source_return - market_return,
                "target_window_excess_return": target_return - market_return,
            })
    return sorted(rows, key=lambda row: (row["entry_date"], row["target_ticker"], row["event_id"]))


def select_candidates(rows: list[dict], rule: str) -> list[dict]:
    if rule not in RULES:
        raise ValueError(f"Unknown fixed rule: {rule}")
    return [row for row in rows if rule == "all_linked_events"
            or (rule == "positive_source_price" and row["source_window_excess_return"] > 0)
            or (rule == "positive_guidance_revision" and row["guidance_sign"] == 1)]


def collect_diffusion_prices(ledger_path: Path, output: Path, refresh: bool = False) -> dict:
    from relay.config import StudyConfig
    from relay.ingest import PublicData, fetch_prices

    source = json.loads(ledger_path.read_text())
    ledger = DiffusionLedger.model_validate(source)
    audit_ledger(ledger)
    start, end = source["metadata"]["study_period"]
    config = StudyConfig(start=start, end=end)
    tickers = sorted({"SPY", *(event.source_ticker for event in ledger.events),
                      *(edge.target_ticker for edge in ledger.relationships)})
    client = PublicData(output / "raw", refresh=refresh)
    prices = validated_prices(pd.concat([fetch_prices(client, ticker, config) for ticker in tickers], ignore_index=True))
    expected = set(market_sessions(config.start, config.end))
    for ticker in tickers:
        observed = set(prices.loc[(prices["ticker"] == ticker) & prices["date"].between(config.start, config.end), "date"])
        if observed != expected:
            raise ValueError(f"Incomplete study-period session coverage for {ticker}")
    output.mkdir(parents=True, exist_ok=True)
    path = output / "prices.parquet"
    prices.to_parquet(path, index=False)
    raw = []
    for cached in sorted((output / "raw").glob("*.json")):
        record = json.loads(cached.read_text())
        raw.append({"path": str(cached.resolve()), "url": record["url"], "retrieved_at": record["retrieved_at"],
                    "sha256": hashlib.sha256(cached.read_bytes()).hexdigest()})
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(), "tickers": tickers,
        "study_period": [start, end], "study_sessions": len(expected), "rows": len(prices),
        "ledger_sha256": hashlib.sha256(ledger_path.read_bytes()).hexdigest(),
        "prices_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "raw_responses": raw,
        "calendar": "XNYS", "calendar_version": version("exchange-calendars"),
        "pricing": "Current Yahoo adjusted daily snapshot; adj_open=open*adj_close/close. No executable-fill or historical-vintage claim.",
    }
    (output / "price-manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return manifest


def run_diffusion_study(ledger_path: Path, prices_path: Path, protocol_path: Path, output: Path) -> dict:
    from relay.diffusion_portfolio import MAX_POSITION_FRACTION, MAX_POSITIONS, replay_portfolio

    raw_ledger = json.loads(ledger_path.read_text())
    ledger = DiffusionLedger.model_validate(raw_ledger)
    protocol = json.loads(protocol_path.read_text())
    if not protocol.get("complete_protocol_frozen_before_price_access"):
        raise ValueError("A complete frozen protocol is required")
    ledger_sha256 = hashlib.sha256(ledger_path.read_bytes()).hexdigest()
    if ledger_sha256 != protocol.get("source_ledger_frozen_sha256"):
        raise ValueError("Source ledger differs from the frozen protocol")
    manifest_path = prices_path.with_name("price-manifest.json")
    if not manifest_path.is_file():
        raise ValueError("A price-manifest.json beside the prices file is required")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("prices_sha256") != hashlib.sha256(prices_path.read_bytes()).hexdigest():
        raise ValueError("Prices differ from the price manifest")
    ledger_hashes = [manifest[key] for key in ("ledger_sha256", "frozen_ledger_sha256") if key in manifest]
    if not ledger_hashes or any(value != ledger_sha256 for value in ledger_hashes):
        raise ValueError("Source ledger differs from the price manifest")
    settings = {
        "rules": list(RULES), "entry_delays": list(ENTRY_DELAYS), "holding_sessions": 5,
        "cost_grid_bps": list(COST_GRID), "initial_cash": 10000, "max_open_positions": MAX_POSITIONS,
        "max_new_position_budget_fraction": MAX_POSITION_FRACTION,
        "guidance_fields": {key: [list(value) for value in values] for key, values in GUIDANCE_FIELDS.items()},
    }
    if protocol.get("execution_settings") != settings:
        raise ValueError("Frozen execution settings differ from this implementation")
    prices = validated_prices(pd.read_parquet(prices_path))
    audit = audit_ledger(ledger)
    study_start, study_end = map(date.fromisoformat, raw_ledger["metadata"]["study_period"])
    account_prices = prices[prices["date"].between(study_start, study_end)]
    account_sessions = set(account_prices.loc[account_prices["ticker"] == "SPY", "date"])
    if account_sessions != set(market_sessions(study_start, study_end)):
        raise ValueError("Price data does not cover the full study-period calendar")
    runs = []
    prepared = []
    for delay in ENTRY_DELAYS:
        rows = prepare_events(ledger, prices, entry_delay_sessions=delay)
        if any(not study_start <= date.fromisoformat(row["entry_date"]) <= date.fromisoformat(row["exit_date"]) <= study_end
               for row in rows):
            raise ValueError("All trading windows must fit inside the declared study period")
        prepared.extend(rows)
        for rule in RULES:
            candidates = select_candidates(rows, rule)
            for cost in COST_GRID:
                result = replay_portfolio(candidates, account_prices, cost_bps=cost)
                runs.append({"rule": rule, "entry_delay_sessions": delay, "cost_bps": cost,
                             "candidate_source_events": len({row["event_id"] for row in candidates}),
                             "candidate_linked_rows": len(candidates), "result": result})
    report = {
        "status": "exploratory_feasibility_only", "created_at": datetime.now(timezone.utc).isoformat(),
        "study_period": [study_start.isoformat(), study_end.isoformat()], "audit": audit,
        "inputs": {label: {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                   for label, path in (("ledger", ledger_path), ("prices", prices_path),
                                       ("price_manifest", manifest_path), ("protocol", protocol_path))},
        "protocol": protocol, "runs": runs,
        "implementation": {
            "files": {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                      for name in ("diffusion.py", "diffusion_portfolio.py", "ingest.py")},
            "packages": {name: version(name) for name in ("pandas", "numpy", "pydantic", "exchange-calendars")},
            "calendar": "XNYS", "study_session_count": len(account_sessions),
        },
        "limitations": [
            "Purposively chosen small cohort; no alpha certification, statistical significance or future-profit probability.",
            "Known-document history is incomplete for interim updates, calls and supplier preannouncements.",
            "Annual management-guidance change is not an analyst-consensus surprise or clean remaining-year demand shock.",
            "For after-close releases, the primary price feature predates the news; date-only inputs alone cannot certify a full post-news session.",
            "The delayed price rule updates both its decision signal and entry; it is not a same-signal execution-delay stress.",
            "Daily adjusted-price proxies and assumed costs are not executable quotes; fractional shares and no cash interest or taxes.",
            "Opening-equity sizing uses the same prices as assumed opening fills; exact orders cannot be specified before those auction prices exist.",
            "Adjusted dividend gains become synthetic cash at exit; dividend receivables and pay dates require separate broker cash accounting.",
            "Matched-schedule SPY is cash constrained; differences are not factor-adjusted alpha or proof of diffusion.",
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    pd.DataFrame(prepared).to_csv(output / "event_windows.csv", index=False)
    summaries = []
    trades = []
    for run in runs:
        identifiers = {name: run[name] for name in ("rule", "entry_delay_sessions", "cost_bps")}
        result = run["result"]
        summaries.append({**identifiers, "source_events": run["candidate_source_events"],
                          **{name: result[name] for name in ("n_trades", "portfolio_pnl", "portfolio_return",
                                                            "benchmark_pnl", "benchmark_return", "total_costs", "max_concurrency")}})
        trades.extend({**identifiers, **trade} for trade in result["trades"])
    pd.DataFrame(summaries).to_csv(output / "summary.csv", index=False)
    pd.DataFrame(trades).to_csv(output / "trades.csv", index=False)
    return report
