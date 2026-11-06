import hashlib
import json
import shutil
import socket
from pathlib import Path

import pandas as pd
import pytest

from relay import reproduce as replay


@pytest.fixture
def reference_copy(tmp_path):
    target = tmp_path / "reference"
    target.mkdir()
    for name in ("reference_manifest.json", "reference_observations.csv.gz", "pilot.json"):
        shutil.copyfile(replay.REFERENCE_DIR / name, target / name)
    return target


def test_reference_replays_offline_without_changing_saved_inputs(monkeypatch, tmp_path):
    def no_network(*args, **kwargs):
        raise AssertionError("The reference replay must not access the network")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in replay.REFERENCE_DIR.glob("*") if path.is_file()}
    output = tmp_path / "results"
    report = replay.reproduce(output)
    assert report["status"] == "verified"
    assert report["scope"] == "model_stage_replay"
    assert report["reference"]["rows"] == 1205
    assert report["reference"]["predictions"] == 1050
    assert report["verification"]["max_metric_error"] <= report["verification"]["tolerance"]
    folds = [fold for fold in report["folds"] if fold["status"] == "evaluated"]
    assert len(folds) == 35
    for fold in folds:
        assert fold["fit"]["latest_exit"] < fold["calibration_start"]
        assert fold["calibration"]["latest_exit"] < fold["fold_start"]
    assert len(pd.read_csv(output / "benchmarks.csv")) == 4
    assert len(pd.read_csv(output / "predictions.csv")) == 4200
    assert json.loads((output / "report.json").read_text()) == report
    assert replay.reference_status()["reference_id"] == report["reference"]["reference_id"]
    first_bytes = {path.name: path.read_bytes() for path in output.iterdir()}
    replay.reproduce(output)
    assert {path.name: path.read_bytes() for path in output.iterdir()} == first_bytes
    assert all(hashlib.sha256(path.read_bytes()).hexdigest() == value for path, value in before.items())


@pytest.mark.parametrize("name", ["reference_observations.csv.gz", "pilot.json"])
def test_tampered_reference_fails_before_writing(reference_copy, tmp_path, name):
    path = reference_copy / name
    path.write_bytes(path.read_bytes() + b"\n")
    assert replay.reference_status(reference_copy)["status"] == "mismatch"
    output = tmp_path / "results"
    with pytest.raises(ValueError, match="hash mismatch"):
        replay.reproduce(output, reference_copy)
    assert not output.exists()


def test_missing_reference_is_not_reported_as_available(tmp_path):
    assert replay.reference_status(tmp_path)["status"] == "missing"


def test_model_regression_fails_before_writing(monkeypatch, tmp_path):
    original = replay.run_study

    def altered(*args, **kwargs):
        result = original(*args, **kwargs)
        result["metrics"][0]["mae"] += 0.01
        return result

    monkeypatch.setattr(replay, "run_study", altered)
    output = tmp_path / "results"
    with pytest.raises(ValueError, match="metric zero_excess.mae"):
        replay.reproduce(output)
    assert not output.exists()


def test_existing_nonreplay_outputs_are_preserved(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    saved = output / "study.json"
    saved.write_text('{"original": true}')
    with pytest.raises(ValueError, match="unrelated files"):
        replay.reproduce(output)
    assert saved.read_text() == '{"original": true}'
    assert sorted(path.name for path in output.iterdir()) == ["study.json"]


def test_status_rejects_source_changed_since_import(monkeypatch):
    initial = replay.reference_status()
    original = replay._hash
    monkeypatch.setattr(replay, "_hash", lambda path: "changed" if Path(path).name == "model.py" else original(path))
    changed = replay.reference_status()
    assert initial["status"] == "available"
    assert changed["status"] == "mismatch"
    assert "restart PRISM" in changed["message"]


def test_manifest_change_invalidates_previous_reference_id(reference_copy):
    initial = replay.reference_status(reference_copy)
    path = reference_copy / "reference_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["description"] += " Revised description."
    path.write_text(json.dumps(manifest))
    changed = replay.reference_status(reference_copy)
    assert initial["status"] == changed["status"] == "available"
    assert initial["reference_id"] != changed["reference_id"]


def test_replay_parses_the_verified_byte_snapshot(monkeypatch, reference_copy, tmp_path):
    original = replay._reference

    def replace_after_read(directory):
        snapshot = original(directory)
        (directory / "reference_observations.csv.gz").write_bytes(b"replaced after verified read")
        (directory / "pilot.json").write_bytes(b"replaced after verified read")
        return snapshot

    monkeypatch.setattr(replay, "_reference", replace_after_read)
    report = replay.reproduce(tmp_path / "results", reference_copy)
    assert report["status"] == "verified"
    assert report["verification"]["max_metric_error"] <= report["verification"]["tolerance"]
