import hashlib
import json
import shutil
import socket

import pandas as pd
import pytest

from relay import research


@pytest.fixture
def research_copy(tmp_path):
    directory = tmp_path / "fixture"
    directory.mkdir()
    for name in ("research_manifest.json", "research_observations.csv.gz"):
        shutil.copyfile(research.REFERENCE_DIR / name, directory / name)
    return directory


def test_analysis_replays_offline_and_partitions_confounds(monkeypatch, tmp_path):
    def no_network(*args, **kwargs):
        raise AssertionError("Research analysis must not access the network")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in research.REFERENCE_DIR.iterdir() if p.is_file()}
    output = tmp_path / "analysis"
    report = research.analyze(output)
    assert report["verification"]["max_metric_error"] == 0
    assert report["verification"]["timing_boundaries_checked"] == 35
    assert report["sample"]["n_predictions"] == 1050
    assert report["sample"]["n_events"] == 210
    assert report["sample"]["n_holding_window_clusters"] == 115
    assert len(report["metrics"]) == len(report["ablations"]) == 5
    for metric in report["metrics"]:
        subsets = [row for row in report["confounds"] if row["model_id"] == metric["model_id"]]
        assert {row["value"]: row["n_predictions"] for row in subsets} == {"no_overlap": 858, "overlap": 192}
        for name in ("mae", "brier", "interval_coverage", "mean_interval_width", "interval_score"):
            weighted = sum(row[name] * row["n_predictions"] for row in subsets) / metric["n_predictions"]
            assert weighted == pytest.approx(metric[name], abs=1e-12)
    assert len(pd.read_csv(output / "predictions.csv")) == 5250
    assert len(pd.read_csv(output / "slices.csv")) == 105
    assert json.loads((output / "summary.json").read_text()) == report
    original_output = {p.name: p.read_bytes() for p in output.iterdir()}
    (output / "diagnostics.png").write_bytes(b"old plot")
    (output / "diagnostics.pdf").write_bytes(b"old plot")
    assert research.analyze(output) == report
    assert {p.name: p.read_bytes() for p in output.iterdir()} == original_output
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == digest for p, digest in before.items())


def test_changed_fixture_fails_before_writing(research_copy, tmp_path):
    path = research_copy / "research_observations.csv.gz"
    path.write_bytes(path.read_bytes() + b"modified")
    output = tmp_path / "result"
    with pytest.raises(ValueError, match="hash mismatch"):
        research.analyze(output, research_copy)
    assert not output.exists()


def test_model_regression_is_not_published_as_verified(monkeypatch, tmp_path):
    original = research.run_study

    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        result["metrics"][0]["mae"] += 0.01
        return result

    monkeypatch.setattr(research, "run_study", changed)
    output = tmp_path / "analysis"
    with pytest.raises(ValueError, match="metric zero_excess.mae"):
        research.analyze(output)
    assert not output.exists()


@pytest.mark.parametrize("kind", ["unrelated", "symlink", "invalid_report"])
def test_analysis_does_not_overwrite_unrelated_outputs(tmp_path, kind):
    output = tmp_path / "output"
    output.mkdir()
    if kind == "unrelated":
        (output / "notes.txt").write_text("preserve these notes")
    elif kind == "symlink":
        target = tmp_path / "notes.txt"
        target.write_text("preserve these notes")
        (output / "summary.json").symlink_to(target)
    else:
        (output / "summary.json").write_text('{"format":"other-study"}')
    before = {p.name: p.read_bytes() for p in output.iterdir()}
    with pytest.raises(ValueError, match="unrelated files|not a prior"):
        research.analyze(output)
    assert {p.name: p.read_bytes() for p in output.iterdir()} == before
