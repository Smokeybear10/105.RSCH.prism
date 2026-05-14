from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from importlib.metadata import version
from io import BytesIO
from pathlib import Path
import platform

import pandas as pd

from relay.config import COMPANIES, DATA_DIR, StudyConfig
from relay.features import build_panel


def methodology(config: StudyConfig) -> list[dict[str, str]]:
    return [
        {"title": "Question", "body": "Do source-company price reactions and disclosed language improve forecasts of peer returns beyond a price-only model? This measures predictive association, not a causal transmission mechanism."},
        {"title": "Information clock", "body": "SEC Item 2.02 disclosures define events. Features use the close of the first trading session strictly after the later of the filing date and the known acceptance timestamp's New York date. Without a timestamp, the filing date is the conservative cutoff. Entry is the following session's adjusted open; exit is the close of the fifth holding session for the default horizon. This is a delayed filing-based window, not a claim to identify the original earnings-release time."},
        {"title": "Prices and target", "body": f"The target is the peer's {config.horizon_days}-session adjusted open-to-close return minus SPY's return over the same interval. This is benchmark-relative, not beta-neutral. Yahoo adjusted history is a current vendor snapshot; revision and survivorship limitations remain."},
        {"title": "Research universe", "body": "AMD, NVIDIA, Intel, Qualcomm, Texas Instruments and Micron form a fixed, retrospectively selected semiconductor cohort. Every other cohort member is a peer. These edges are study assumptions, not verified supplier relationships or a point-in-time industry universe."},
        {"title": "Text features", "body": "Fixed phrase rules score original SEC earnings exhibits after excluding explicit forward-looking and risk disclaimers. Cached releases are re-extracted for each rebuild; unavailable releases stay missing. These are simple lexical measures, not verified economic causes or analyst-consensus surprises. Constant features are reported in the source audit."},
        {"title": "Walk-forward evaluation", "body": "Models are fit before each quarterly test block using only completed outcomes. The training-selected model compares nine fixed candidates using an earlier/later split entirely inside the fitting sample, purges overlapping labels, then refits the chosen candidate. A separate later calibration segment supplies intervals and probabilities. Calibration and test outcomes never choose the model. The four original fixed models remain as comparisons."},
        {"title": "Uncertainty", "body": "Intervals and positive-return probabilities use prior calibration residuals. An 80% interval is nominal; dependent and shifting markets can invalidate coverage. Observed coverage and probability error are reported. Comparison intervals group transitively overlapping holding windows before resampling; remaining serial dependence and model development on an inspected sample still limit interpretation."},
        {"title": "Costs and interpretation", "body": f"A hypothetical SPY-relative position is taken only when the forecast exceeds {config.cost_bps:g} bps in magnitude. {config.cost_bps:g} bps is deducted per active trade as an assumed total roundtrip cost. These are overlapping event returns, not portfolio P&L; financing, borrow, capacity and intraday execution are not modeled. No portfolio Sharpe is reported."},
        {"title": "What would invalidate the result", "body": "A benefit must survive price-only benchmarks, event clustering, alternative costs and future collection. Concurrent peer announcements remain in the sample and can confound interpretation. This pilot is exploratory: repeated model changes after inspecting test results would require a new untouched evaluation period."},
    ]


def make_payload(result: dict, events: list[dict], audit: dict, config: StudyConfig, warnings: list[str]) -> dict:
    lookup = {event["event_id"]: event for event in events}
    active_model = "adaptive" if any(metric["model_id"] == "adaptive" for metric in result["metrics"]) else "event_conditioned"
    grouped = {}
    for row in result["predictions"]:
        event = lookup[row["event_id"]]
        if row["event_id"] not in grouped:
            grouped[row["event_id"]] = {
                "id": row["event_id"], "date": event["filing_date"],
                "information_date": event.get("information_date", event["filing_date"]),
                "acceptance_timestamp": event.get("event_at"),
                "source_ticker": row["source_ticker"], "event_type": row["event_type"],
                "decision_date": row["decision_date"], "entry_date": row["entry_date"], "exit_date": row["exit_date"],
                "source_return": event.get("source_return"), "headline": event["headline"],
                "text_available": bool(event["text_available"]),
                "evidence": [{"title": "Original SEC disclosure", "url": event["evidence_url"], "excerpt": event["excerpt"]}],
                "predictions": [],
            }
        forecast = row["models"][active_model]
        baseline = row["models"]["price_only"]
        grouped[row["event_id"]]["predictions"].append({
            "peer_ticker": row["target_ticker"], "relationship": "Semiconductor cohort peer",
            "predicted_return": forecast["prediction"], "actual_return": row["realized_excess_return"],
            "lower": forecast["lower_80"], "upper": forecast["upper_80"],
            "p_positive": forecast["p_positive"], "baseline_return": baseline["prediction"],
            "event_return": row["models"]["event_conditioned"]["prediction"],
            "model_id": active_model,
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
            "cost_bps": config.cost_bps, "model": active_model,
            "evaluation_scope": "Exploratory historical development; no untouched holdout or trading edge is established.",
        },
        "summary": {"events": len(grouped), "source_events": len(events), "predictions": len(result["predictions"]),
                    "test_predictions": len(result["predictions"]), "source_tickers": list(COMPANIES),
                    "text_coverage": text_count / len(events) if events else 0,
                    "exhibit_coverage": sum(e.get("document_kind") == "earnings_exhibit" for e in events) / len(events) if events else 0},
        "events": sorted(grouped.values(), key=lambda e: e["date"], reverse=True),
        "benchmarks": [{"id": metric["model_id"], **metric} for metric in result["metrics"]],
        "methodology": methodology(config), "folds": result["folds"], "audit": audit,
        "comparisons": result.get("comparisons", []),
        "overlap_comparisons": result.get("overlap_comparisons", []),
        "selection_protocol": result.get("config", {}).get("selection"),
        "period_diagnostics": result.get("period_diagnostics", []),
        "model_definitions": result.get("model_definitions", []),
    }


def load_study_config(directory: Path, output: Path) -> StudyConfig:
    candidates = [output / "study_config.json", directory / "study_config.json", directory / "config.json"]
    path = next((path for path in candidates if path.is_file()), candidates[-1])
    config = StudyConfig.model_validate_json(path.read_bytes())
    collection_path = directory / "config.json"
    if collection_path.is_file():
        collection = StudyConfig.model_validate_json(collection_path.read_bytes())
        config = config.model_copy(update={"start": collection.start, "end": collection.end})
    return config


def run_cached_study(directory: Path = DATA_DIR, config: StudyConfig | None = None,
                     output_directory: Path | None = None) -> dict:
    from relay.model import run_study

    output = output_directory if output_directory is not None else directory
    if config is None:
        config = load_study_config(directory, output)
    snapshots = {name: (directory / filename).read_bytes()
                 for name, filename in (("prices", "prices.parquet"), ("events", "events.json"))}
    prices = pd.read_parquet(BytesIO(snapshots["prices"]))
    events = json.loads(snapshots["events"])
    events = [event for event in events if config.start.isoformat() <= event["filing_date"] <= config.end.isoformat()]
    from relay.ingest import prepare_cached_events

    events, extraction_audit, extraction_warnings = prepare_cached_events(events, directory)
    prepared_snapshot = json.dumps(events, sort_keys=True, allow_nan=False)
    frame, audit = build_panel(prices, events, config)
    audit["text_extraction"] = extraction_audit
    audit["constant_features"] = [name for name in ("demand_signal", "pricing_signal", "competitive_signal", "guidance_signal", "text_available")
                                  if not frame.empty and frame[name].nunique() <= 1]
    audit["fingerprints"] = {
        **{name: hashlib.sha256(content).hexdigest() for name, content in snapshots.items()},
        **{name: hashlib.sha256(Path(__file__).with_name(f"{name}.py").read_bytes()).hexdigest()
           for name in ("model", "selection", "features", "ingest", "study", "config")},
    }
    audit["fingerprint_scope"] = {
        "inputs": "SHA-256 of the exact byte snapshots parsed for this run.",
        "source_files": "Source files on disk when the run was built; these are not an attestation of already-imported Python code.",
    }
    audit["environment"] = {
        "python": platform.python_version(),
        "packages": {name: version(name) for name in ("numpy", "pandas", "pyarrow", "pydantic")},
    }
    audit["config"] = config.model_dump(mode="json")
    if frame.empty:
        raise ValueError("No complete event/peer observations. Check price coverage and event dates.")
    output.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output / "observations.parquet", index=False)
    reaction = frame.groupby("event_id")["source_event_return"].first().to_dict()
    information_dates = frame.groupby("event_id")["information_date"].first().to_dict()
    for event in events:
        event["source_return"] = reaction.get(event["event_id"])
        event["information_date"] = information_dates.get(event["event_id"], event["filing_date"])
    result = run_study(frame, min_train_rows=config.min_train_rows, cost_bps=config.cost_bps,
                       horizon_days=config.horizon_days, include_adaptive=True)
    warnings_path = directory / "ingestion_warnings.json"
    warnings = [*extraction_warnings, *(json.loads(warnings_path.read_text()) if warnings_path.exists() else [])]
    payload = make_payload(result, events, audit, config, warnings)
    serialized = json.dumps(payload, allow_nan=False)
    path = output / "study.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(serialized)
    temporary.replace(path)
    (output / "prepared_events.json").write_text(prepared_snapshot)
    flattened = []
    for row in result["predictions"]:
        for model_id, forecast in row["models"].items():
            flattened.append({**{k: v for k, v in row.items() if k != "models"}, "model_id": model_id, **forecast})
    csv_path = output / "predictions.csv"
    pd.DataFrame(flattened).to_csv(csv_path.with_suffix(".tmp"), index=False)
    csv_path.with_suffix(".tmp").replace(csv_path)
    (output / "study_config.json").write_text(config.model_dump_json(indent=2))
    return payload
