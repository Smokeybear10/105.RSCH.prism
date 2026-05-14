from __future__ import annotations

import hashlib
import io
import json
import platform
from importlib.metadata import version
from pathlib import Path

import pandas as pd

from relay.diagnostics import diagnose
from relay.model import INPUT_FEATURES, RETURN_COLUMNS, run_study
from relay.reproduce import REFERENCE_DIR, _verify_metrics

FORMAT = "prism-research-analysis-v1"
OUTPUT_FILES = {
    "summary.json", "protocol.json", "metrics.csv", "ablations.csv", "slices.csv",
    "confounds.csv", "folds.json", "predictions.csv", "diagnostics.png", "diagnostics.pdf",
}


def _inputs(directory: Path) -> tuple[dict, pd.DataFrame]:
    manifest = json.loads((directory / "research_manifest.json").read_bytes())
    if manifest["schema_version"] != 1 or manifest["scope"] != "corrected_model_stage_replay":
        raise ValueError("Unsupported research manifest")
    snapshot = (directory / "research_observations.csv.gz").read_bytes()
    expected = manifest["observations"]
    if hashlib.sha256(snapshot).hexdigest() != expected["sha256"]:
        raise ValueError("Research fixture hash mismatch; restore the included numerical inputs")
    frame = pd.read_csv(io.BytesIO(snapshot), compression="gzip",
                        converters={name: float for name in RETURN_COLUMNS + INPUT_FEATURES})
    if list(frame.columns) != expected["columns"] or len(frame) != expected["rows"]:
        raise ValueError("Research fixture schema or row count differs from its manifest")
    if frame["event_id"].nunique() != expected["events"] or frame.duplicated(["event_id", "target_ticker"]).any():
        raise ValueError("Research event count or event/peer identity is invalid")
    if not frame["overlapping_peer_disclosure"].isin([True, False]).all():
        raise ValueError("Peer disclosure overlap flags must be complete booleans")
    return manifest, frame


def _check_output(output: Path) -> None:
    if output.is_symlink() or (output.exists() and not output.is_dir()):
        raise ValueError("Use a separate research output directory, not a file or symbolic link")
    if not output.exists() or not any(output.iterdir()):
        return
    if any(p.name not in OUTPUT_FILES or not p.is_file() or p.is_symlink() for p in output.iterdir()):
        raise ValueError("Output contains unrelated files; choose an empty or prior research output directory")
    try:
        previous = json.loads((output / "summary.json").read_text())
    except (FileNotFoundError, ValueError) as exc:
        raise ValueError("Output is not a prior PRISM research analysis") from exc
    if previous.get("format") != FORMAT:
        raise ValueError("Output is not a prior PRISM research analysis")


def _source_hashes() -> dict[str, str]:
    return {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("model.py", "selection.py", "diagnostics.py", "research.py")}


def analyze(output: Path = Path("out/prism-analysis"), reference_dir: Path = REFERENCE_DIR) -> dict:
    _check_output(output)
    manifest, frame = _inputs(reference_dir)
    code = _source_hashes()
    result = run_study(frame, **manifest["config"])
    expected = manifest["expected"]
    if result["status"] != "ok" or len(result["predictions"]) != expected["predictions"]:
        raise ValueError("Analysis did not reproduce the saved forecast sample")
    folds = [fold for fold in result["folds"] if fold["status"] == "evaluated"]
    if len(folds) != expected["evaluated_folds"]:
        raise ValueError("Analysis changed the expected number of evaluated folds")
    for fold in folds:
        if not (fold["fit"]["latest_exit"] < fold["calibration_start"]
                and fold["calibration"]["latest_exit"] < fold["fold_start"]):
            raise ValueError(f"Chronological separation failed in {fold['fold_id']}")
    error = _verify_metrics(result["metrics"], expected["metrics"], expected["metric_absolute_tolerance"])
    diagnostics = diagnose(result["predictions"])
    overlap = frame.set_index(["event_id", "target_ticker"])["overlapping_peer_disclosure"]
    confounds = []
    for label, value in (("no_overlap", False), ("overlap", True)):
        rows = [row for row in result["predictions"] if bool(overlap.loc[(row["event_id"], row["target_ticker"])]) == value]
        if rows:
            confounds.extend({"group": "peer_disclosure_overlap", "value": label, **metric}
                             for metric in diagnose(rows)["metrics"])
    if code != _source_hashes():
        raise ValueError("Research implementation changed during analysis; restart and rerun")
    protocol = {
        "scope": "Exploratory analysis of the same historical development sample, not a new holdout.",
        "model_config": result["config"],
        "diagnostics": diagnostics["protocol"],
        "fixture": {
            "manifest": "relay/data/research_manifest.json",
            "manifest_sha256": hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
            "manifest_hash_encoding": "UTF-8 json.dumps with sort_keys=True; semantic JSON identity",
            "observations": manifest["observations"],
            "source_built_at": manifest["source"]["study_built_at"],
            "data_as_of": manifest["source"]["data_as_of"],
        },
        "implementation_sha256": code,
        "implementation_scope": "Source files on disk before and after the run; not an attestation of previously imported code.",
        "environment": {"python": platform.python_version(), **{name: version(name) for name in ("numpy", "pandas")}},
        "confound_sensitivity": "Same fitted forecasts, split by observed peer-disclosure overlap; no retraining. Missing events outside the six-company cache are not ruled out.",
        "command": "uv run --locked --offline prism analyze --output out/prism-analysis",
        "return_unit": "decimal; multiply by 100 for percentage points",
    }
    report = {
        **diagnostics, "format": FORMAT, "scope": manifest["scope"],
        "fixture": {
            **{key: manifest["observations"][key] for key in ("rows", "events", "sha256")},
            "predictions": len(result["predictions"]), "data_as_of": manifest["source"]["data_as_of"],
            "source_built_at": manifest["source"]["study_built_at"],
        },
        "verification": {"status": "verified", "max_metric_error": error,
                         "tolerance": expected["metric_absolute_tolerance"], "timing_boundaries_checked": len(folds)},
        "protocol": protocol, "confounds": confounds, "limits": manifest["limits"],
    }
    output.mkdir(parents=True, exist_ok=True)
    for name, content in (("summary.json", report), ("protocol.json", protocol), ("folds.json", result["folds"])):
        (output / name).write_text(json.dumps(content, indent=2, allow_nan=False) + "\n")
    for name in ("metrics", "ablations", "confounds"):
        pd.DataFrame(report[name]).to_csv(output / f"{name}.csv", index=False)
    slices = [{"dimension": group["dimension"], "value": group["value"], **metric}
              for group in report["slices"] for metric in group["metrics"]]
    pd.DataFrame(slices).to_csv(output / "slices.csv", index=False)
    predictions = [{**{k: v for k, v in row.items() if k != "models"}, "model_id": model_id, **forecast}
                   for row in result["predictions"] for model_id, forecast in row["models"].items()]
    pd.DataFrame(predictions).to_csv(output / "predictions.csv", index=False)
    for filename in ("diagnostics.png", "diagnostics.pdf"):
        (output / filename).unlink(missing_ok=True)
    return report
