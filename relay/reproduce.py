from __future__ import annotations

import hashlib
import io
import json
import math
import platform
from importlib.metadata import version
from pathlib import Path

import pandas as pd

from relay.model import INPUT_FEATURES, RETURN_COLUMNS, run_study

REFERENCE_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_OUTPUT = Path("out/relay-reference")
COMMAND = "uv run --locked earnings-study reproduce"
FORMAT = "relay-reference-v1"


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


_IMPORTED_CODE = {name: _hash(Path(__file__).with_name(name)) for name in ("model.py", "reproduce.py")}


def _reference(reference_dir: Path) -> tuple[dict, dict, bytes]:
    manifest = json.loads((reference_dir / "reference_manifest.json").read_bytes())
    if manifest["schema_version"] != 1:
        raise ValueError("Unsupported reference manifest version")
    snapshots = {
        "observations": (reference_dir / "reference_observations.csv.gz").read_bytes(),
        "pilot": (reference_dir / "pilot.json").read_bytes(),
    }
    expected = {
        "observations": manifest["observations"]["sha256"],
        "pilot": manifest["source"]["pilot_sha256"],
    }
    for name, data in snapshots.items():
        if hashlib.sha256(data).hexdigest() != expected[name]:
            raise ValueError(f"Reference {name} hash mismatch; restore the checked-in reference files")
    return manifest, json.loads(snapshots["pilot"]), snapshots["observations"]


def _identity(manifest: dict) -> dict:
    code = {name: _hash(Path(__file__).with_name(name)) for name in ("model.py", "reproduce.py")}
    if code != _IMPORTED_CODE:
        raise ValueError("Reference source changed after import; restart the process before replaying")
    signature = json.dumps({"manifest": manifest, "code": code}, sort_keys=True).encode()
    return {"reference_id": hashlib.sha256(signature).hexdigest(), "code_sha256": code}


def reference_status(reference_dir: Path = REFERENCE_DIR) -> dict:
    try:
        manifest, _, _ = _reference(reference_dir)
        identity = _identity(manifest)
    except FileNotFoundError:
        return {"status": "missing", "message": "Reference files are missing. Restore the checked-in fixture to replay it.", "command": COMMAND}
    except (ValueError, KeyError, TypeError):
        return {"status": "mismatch", "message": "Reference files or loaded source do not match. Restore altered inputs, or restart the process after a code change. A replay has not been verified.", "command": COMMAND}
    return {
        "status": "available", "message": "Reference file hashes match. Run the replay to verify the model results.",
        "command": COMMAND, "scope": manifest["scope"],
        "rows": manifest["observations"]["rows"], "events": manifest["observations"]["events"],
        "predictions": manifest["expected"]["predictions"], "data_as_of": manifest["source"]["data_as_of"],
        **identity,
    }


def _verify_metrics(actual: list[dict], expected: list[dict], tolerance: float) -> float:
    by_id = {row["model_id"]: row for row in actual}
    if set(by_id) != {row["model_id"] for row in expected}:
        raise ValueError("Replayed model set differs from the saved pilot")
    maximum = 0.0
    for original in expected:
        replayed = by_id[original["model_id"]]
        for name, value in original.items():
            if name == "id":
                continue
            candidate = replayed.get(name)
            if isinstance(value, float):
                if not isinstance(candidate, (int, float)) or not math.isfinite(candidate):
                    raise ValueError(f"Invalid replayed metric {original['model_id']}.{name}")
                error = abs(candidate - value)
                if error > tolerance:
                    raise ValueError(f"Replayed metric {original['model_id']}.{name} differs from the saved pilot by {error:g}")
                maximum = max(maximum, error)
            elif candidate != value:
                raise ValueError(f"Replayed metric {original['model_id']}.{name} differs from the saved pilot")
    return maximum


def _prepare_output(output: Path) -> None:
    if output.is_symlink():
        raise ValueError("Use a separate output directory, not a symbolic link")
    if output.exists() and not output.is_dir():
        raise ValueError("Replay output must be a directory")
    if output.exists() and any(output.iterdir()):
        allowed = {"report.json", "benchmarks.csv", "predictions.csv"}
        if any(path.name not in allowed or not path.is_file() or path.is_symlink() for path in output.iterdir()):
            raise ValueError("Output contains unrelated files; choose an empty directory or a prior replay output")
        report = output / "report.json"
        try:
            previous = json.loads(report.read_text())
        except (FileNotFoundError, ValueError) as exc:
            raise ValueError("Output is not a prior reference replay; choose a separate directory") from exc
        if previous.get("format") != FORMAT:
            raise ValueError("Output is not a prior reference replay; choose a separate directory")
    output.mkdir(parents=True, exist_ok=True)


def reproduce(output: Path = DEFAULT_OUTPUT, reference_dir: Path = REFERENCE_DIR) -> dict:
    manifest, pilot, observations = _reference(reference_dir)
    identity = _identity(manifest)
    frame = pd.read_csv(io.BytesIO(observations), compression="gzip", converters={name: float for name in RETURN_COLUMNS + INPUT_FEATURES})
    expected = manifest["observations"]
    if list(frame.columns) != expected["columns"] or len(frame) != expected["rows"]:
        raise ValueError("Reference observation schema or row count differs from its manifest")
    if frame["event_id"].nunique() != expected["events"] or frame.duplicated(["event_id", "target_ticker"]).any():
        raise ValueError("Reference event count or event/peer identity is invalid")
    result = run_study(frame, **manifest["config"])
    if result["status"] != "ok" or len(result["predictions"]) != manifest["expected"]["predictions"]:
        raise ValueError("Replay did not produce the saved forecast sample")
    if result["folds"] != pilot["folds"]:
        raise ValueError("Replay fold boundaries or samples differ from the saved pilot")
    tolerance = manifest["expected"]["metric_absolute_tolerance"]
    maximum = _verify_metrics(result["metrics"], pilot["benchmarks"], tolerance)
    if _identity(manifest) != identity:
        raise ValueError("Reference implementation changed during the replay; restart and rerun")
    report = {
        "format": FORMAT, "status": "verified", "scope": manifest["scope"],
        "reference": {
            "rows": len(frame), "events": int(frame["event_id"].nunique()),
            "predictions": len(result["predictions"]),
            "sha256": expected["sha256"], "pilot_sha256": manifest["source"]["pilot_sha256"],
            "data_as_of": manifest["source"]["data_as_of"], "pilot_built_at": manifest["source"]["pilot_built_at"],
            **identity,
        },
        "verification": {"max_metric_error": maximum, "tolerance": tolerance, "folds_match": True},
        "environment": {"python": platform.python_version(), **{name: version(name) for name in ("numpy", "pandas")}},
        "implementation": {"model_sha256": identity["code_sha256"]["model.py"], "reproduce_sha256": identity["code_sha256"]["reproduce.py"]},
        "metrics": result["metrics"], "folds": result["folds"], "comparisons": result["comparisons"],
        "period_diagnostics": result.get("period_diagnostics", []),
        "sample": result["sample"], "config": result["config"], "limits": manifest["limits"],
        "output": str(output),
    }
    _prepare_output(output)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    pd.DataFrame(result["metrics"]).to_csv(output / "benchmarks.csv", index=False)
    rows = [
        {**{key: value for key, value in row.items() if key != "models"}, "model_id": model_id, **prediction}
        for row in result["predictions"] for model_id, prediction in row["models"].items()
    ]
    pd.DataFrame(rows).to_csv(output / "predictions.csv", index=False)
    return report
