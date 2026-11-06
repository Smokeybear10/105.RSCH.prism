from __future__ import annotations

import numpy as np
import pandas as pd

from relay.config import COMPANIES, StudyConfig


def build_panel(prices: pd.DataFrame, events: list[dict], config: StudyConfig) -> tuple[pd.DataFrame, dict]:
    required = {"ticker", "date", "adj_close", "adj_open"}
    if not required.issubset(prices.columns):
        raise ValueError(f"Prices require {sorted(required)}")
    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"]).dt.normalize()
    prices = prices[prices["date"] <= pd.Timestamp(config.end)]
    if prices.duplicated(["ticker", "date"]).any():
        raise ValueError("Duplicate ticker/date prices")
    closes = prices.pivot(index="date", columns="ticker", values="adj_close").sort_index()
    opens = prices.pivot(index="date", columns="ticker", values="adj_open").reindex(closes.index)
    if not {"SPY", "SOXX", *COMPANIES}.issubset(closes.columns):
        raise ValueError("Prices must include all six study companies, SPY and SOXX")
    calendar = closes.index[closes["SPY"].notna()]
    closes = closes.reindex(calendar)
    opens = opens.reindex(calendar)
    returns = closes.pct_change(fill_method=None)
    momentum = closes.pct_change(20, fill_method=None)
    volatility = returns.rolling(20, min_periods=20).std()
    correlations = {
        (source, target): returns[source].rolling(60, min_periods=50).corr(returns[target])
        for source in COMPANIES for target in COMPANIES if source != target
    }
    filing_dates = {
        ticker: sorted(pd.Timestamp(e["filing_date"]) for e in events if e["source_ticker"] == ticker)
        for ticker in COMPANIES
    }
    rows = []
    skipped = {"outside_price_history": 0, "missing_features_or_prices": 0}
    source_timing = {"acceptance_timestamp_available": 0, "filing_date_only": 0,
                     "acceptance_date_after_filing_date": 0}
    for event in sorted(events, key=lambda e: (e["filing_date"], e["event_id"])):
        source = event["source_ticker"]
        filed = pd.Timestamp(event["filing_date"])
        if source not in COMPANIES or not pd.Timestamp(config.start) <= filed <= pd.Timestamp(config.end):
            continue
        available = filed
        if event.get("event_at"):
            try:
                accepted = pd.Timestamp(event["event_at"])
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Invalid acceptance timestamp for {event['event_id']}") from exc
            if pd.isna(accepted) or accepted.tzinfo is None:
                raise ValueError(f"A timezone-aware acceptance timestamp is required for {event['event_id']}")
            acceptance_date = accepted.tz_convert("America/New_York").normalize().tz_localize(None)
            available = max(filed, acceptance_date)
            source_timing["acceptance_timestamp_available"] += 1
            source_timing["acceptance_date_after_filing_date"] += int(acceptance_date > filed)
        else:
            source_timing["filing_date_only"] += 1
        decision_idx = int(calendar.searchsorted(available, side="right"))
        entry_idx = decision_idx + 1
        exit_idx = entry_idx + config.horizon_days - 1
        prior_idx = int(calendar.searchsorted(available, side="left")) - 1
        if decision_idx < 60 or prior_idx < 0 or exit_idx >= len(calendar):
            skipped["outside_price_history"] += 1
            continue
        decision, entry, exit_date = calendar[decision_idx], calendar[entry_idx], calendar[exit_idx]
        benchmark_return = closes.at[exit_date, "SPY"] / opens.at[entry, "SPY"] - 1
        source_reaction = closes.at[decision, source] / closes.iloc[prior_idx][source] - 1
        for target in COMPANIES:
            if target == source:
                continue
            realized = closes.at[exit_date, target] / opens.at[entry, target] - 1
            row = {
                "event_id": event["event_id"], "source_ticker": source, "target_ticker": target,
                "event_type": event["event_type"], "filing_date": event["filing_date"],
                "information_date": available.date().isoformat(),
                "decision_date": decision.date().isoformat(), "entry_date": entry.date().isoformat(),
                "exit_date": exit_date.date().isoformat(),
                "realized_return": realized, "benchmark_return": benchmark_return,
                "realized_excess_return": realized - benchmark_return,
                "target_return_1d": returns.at[decision, target],
                "target_momentum_20d": momentum.at[decision, target],
                "target_vol_20d": volatility.at[decision, target],
                "market_return_1d": returns.at[decision, "SPY"],
                "market_momentum_20d": momentum.at[decision, "SPY"],
                "industry_return_1d": returns.at[decision, "SOXX"],
                "source_return_1d": returns.at[decision, source],
                "source_event_return": source_reaction,
                "source_momentum_20d": momentum.at[decision, source],
                "correlation_60d": correlations[source, target].at[decision],
                **{key: float(event[key]) for key in ["demand_signal", "pricing_signal", "competitive_signal", "guidance_signal", "text_available"]},
                "overlapping_peer_disclosure": any(filed <= d <= exit_date for d in filing_dates[target]),
            }
            numeric = [v for v in row.values() if isinstance(v, (int, float, np.number))]
            if not np.isfinite(numeric).all():
                skipped["missing_features_or_prices"] += 1
                continue
            rows.append(row)
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(["decision_date", "event_id", "target_ticker"]).reset_index(drop=True)
    return frame, {
        "source_events": len(events), "usable_events": int(frame["event_id"].nunique()) if len(frame) else 0,
        "peer_observations": len(frame), "skipped": skipped,
        "source_timing": source_timing,
        "overlapping_peer_disclosures": int(frame["overlapping_peer_disclosure"].sum()) if len(frame) else 0,
    }
