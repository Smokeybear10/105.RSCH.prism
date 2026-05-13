from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from relay.config import COMPANIES, DATA_DIR, PILOT_PATH, StudyConfig
from relay.features import build_panel


def methodology(config: StudyConfig) -> list[dict[str, str]]:
    return [
        {"title": "Question", "body": "Do source-company price reactions and disclosed language improve forecasts of peer returns beyond a price-only model? This measures predictive association, not a causal transmission mechanism."},
        {"title": "Information clock", "body": "SEC Item 2.02 disclosures define events. Features are observed at the close of the first trading session strictly after the filing date. Entry is the following session's adjusted open; exit is the close of the fifth holding session for the default horizon. The deliberate delay avoids trading on ambiguous intraday filing timestamps."},
        {"title": "Prices and target", "body": f"The target is the peer's {config.horizon_days}-session adjusted open-to-close return minus SPY's return over the same interval. This is benchmark-relative, not beta-neutral. Yahoo adjusted history is a current vendor snapshot; revision and survivorship limitations remain."},
        {"title": "Research universe", "body": "AMD, NVIDIA, Intel, Qualcomm, Texas Instruments and Micron form a fixed, retrospectively selected semiconductor cohort. Every other cohort member is a peer. These edges are study assumptions, not verified supplier relationships or a point-in-time industry universe."},
        {"title": "Text features", "body": "Transparent, fixed phrase rules measure demand, pricing, competitive and guidance language in original SEC filing exhibits. Event labels describe matched language, not verified economic causes. No LLM, later news article or observed future return is used to produce these features. Missing text stays missing and is flagged."},
        {"title": "Walk-forward evaluation", "body": "Models are fit before each quarterly test block. Only outcomes completed before the cutoff are eligible. A chronological calibration segment is separated from fitting, and overlapping labels are purged. Scaling and ridge coefficients use fitting data only. Hyperparameters are fixed; all displayed forecasts are out of sample."},
        {"title": "Uncertainty", "body": "Intervals and positive-return probabilities use prior calibration residuals. An 80% interval is nominal; dependent and shifting markets can invalidate coverage. Observed coverage and probability error are reported. MAE comparison intervals resample decision-date clusters; multi-day serial dependence can remain."},
        {"title": "Costs and interpretation", "body": f"A hypothetical SPY-relative position is taken only when the forecast exceeds {config.cost_bps:g} bps in magnitude. {config.cost_bps:g} bps is deducted per active trade as an assumed total roundtrip cost. These are overlapping event returns, not portfolio P&L; financing, borrow, capacity and intraday execution are not modeled. No portfolio Sharpe is reported."},
        {"title": "What would invalidate the result", "body": "A benefit must survive price-only benchmarks, event clustering, alternative costs and future collection. Concurrent peer announcements remain in the sample and can confound interpretation. This pilot is exploratory: repeated model changes after inspecting test results would require a new untouched evaluation period."},
    ]


def make_payload(result: dict, events: list[dict], audit: dict, config: StudyConfig, warnings: list[str]) -> dict:
    lookup = {event["event_id"]: event for event in events}
    grouped = {}
    for row in result["predictions"]:
        event = lookup[row["event_id"]]
        if row["event_id"] not in grouped:
            grouped[row["event_id"]] = {
                "id": row["event_id"], "date": event["filing_date"],
                "source_ticker": row["source_ticker"], "event_type": row["event_type"],
                "decision_date": row["decision_date"], "entry_date": row["entry_date"], "exit_date": row["exit_date"],
                "source_return": event.get("source_return"), "headline": event["headline"],
                "text_available": bool(event["text_available"]),
                "evidence": [{"title": "Original SEC disclosure", "url": event["evidence_url"], "excerpt": event["excerpt"]}],
                "predictions": [],
            }
        forecast = row["models"]["event_conditioned"]
        baseline = row["models"]["price_only"]
        grouped[row["event_id"]]["predictions"].append({
            "peer_ticker": row["target_ticker"], "relationship": "Semiconductor cohort peer",
            "predicted_return": forecast["prediction"], "actual_return": row["realized_excess_return"],
            "lower": forecast["lower_80"], "upper": forecast["upper_80"],
            "p_positive": forecast["p_positive"], "baseline_return": baseline["prediction"],
            "split": "out_of_sample", "fold_id": row["fold_id"],
        })
    dates = sorted(event["filing_date"] for event in events)
    all_warnings = [*warnings, *result.get("warnings", [])]
    text_count = sum(bool(e["text_available"]) for e in events)
    return {
        "study": {
            "title": "Information diffusion across semiconductor equities", "status": "ready" if result["status"] == "ok" else "insufficient_data",
            "built_at": datetime.now(timezone.utc).isoformat(), "period_start": dates[0] if dates else None,
            "period_end": dates[-1] if dates else None, "data_as_of": config.end.isoformat(),
            "horizon_days": config.horizon_days, "universe": list(COMPANIES), "notes": all_warnings,
            "cost_bps": config.cost_bps, "model": "event_conditioned",
        },
        "summary": {"events": len(grouped), "source_events": len(events), "predictions": len(result["predictions"]),
                    "test_predictions": len(result["predictions"]), "source_tickers": list(COMPANIES),
                    "text_coverage": text_count / len(events) if events else 0,
                    "exhibit_coverage": sum(e.get("document_kind") == "earnings_exhibit" for e in events) / len(events) if events else 0},
        "events": sorted(grouped.values(), key=lambda e: e["date"], reverse=True),
        "benchmarks": [{"id": metric["model_id"], **metric} for metric in result["metrics"]],
        "methodology": methodology(config), "folds": result["folds"], "audit": audit,
        "comparisons": result.get("comparisons", []),
        "model_definitions": result.get("model_definitions", []),
    }


def run_cached_study(directory: Path = DATA_DIR, config: StudyConfig | None = None, publish_pilot: bool = False) -> dict:
    from relay.model import run_study

    if config is None:
        config_path = directory / "study_config.json"
        if not config_path.exists():
            config_path = directory / "config.json"
        config = StudyConfig.model_validate_json(config_path.read_text())
    prices = pd.read_parquet(directory / "prices.parquet")
    events = json.loads((directory / "events.json").read_text())
    events = [event for event in events if config.start.isoformat() <= event["filing_date"] <= config.end.isoformat()]
    frame, audit = build_panel(prices, events, config)
    audit["fingerprints"] = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in {
            "prices": directory / "prices.parquet", "events": directory / "events.json",
            "model": Path(__file__).with_name("model.py"), "features": Path(__file__).with_name("features.py"),
            "ingest": Path(__file__).with_name("ingest.py"),
        }.items()
    }
    audit["config"] = config.model_dump(mode="json")
    if frame.empty:
        raise ValueError("No complete event/peer observations. Check price coverage and event dates.")
    frame.to_parquet(directory / "observations.parquet", index=False)
    reaction = frame.groupby("event_id")["source_event_return"].first().to_dict()
    for event in events:
        event["source_return"] = reaction.get(event["event_id"])
    result = run_study(frame, min_train_rows=config.min_train_rows, cost_bps=config.cost_bps, horizon_days=config.horizon_days)
    warnings_path = directory / "ingestion_warnings.json"
    warnings = json.loads(warnings_path.read_text()) if warnings_path.exists() else []
    payload = make_payload(result, events, audit, config, warnings)
    serialized = json.dumps(payload, allow_nan=False)
    path = directory / "study.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(serialized)
    temporary.replace(path)
    flattened = []
    for row in result["predictions"]:
        for model_id, forecast in row["models"].items():
            flattened.append({**{k: v for k, v in row.items() if k != "models"}, "model_id": model_id, **forecast})
    csv_path = directory / "predictions.csv"
    pd.DataFrame(flattened).to_csv(csv_path.with_suffix(".tmp"), index=False)
    csv_path.with_suffix(".tmp").replace(csv_path)
    (directory / "study_config.json").write_text(config.model_dump_json(indent=2))
    if publish_pilot:
        PILOT_PATH.parent.mkdir(parents=True, exist_ok=True)
        PILOT_PATH.write_text(serialized)
    return payload
