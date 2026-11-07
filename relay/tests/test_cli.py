import hashlib
import json
import socket
import sys

import numpy as np
import pandas as pd
import pytest

from relay import cli
from relay.config import COMPANIES, PILOT_PATH, StudyConfig
from relay.ingest import text_features


def invoke(monkeypatch, *arguments):
    monkeypatch.setattr(sys, "argv", ["prism", *arguments])
    cli.main()


@pytest.fixture
def cached_inputs(monkeypatch, tmp_path):
    directory = tmp_path / "inputs"
    directory.mkdir()
    dates = pd.bdate_range("2020-01-01", periods=200)
    prices = pd.concat([
        pd.DataFrame({"ticker": ticker, "date": dates, "adj_open": 100 + np.arange(200) * 0.1,
                      "adj_close": 101 + np.arange(200) * 0.1})
        for ticker in [*COMPANIES, "SPY", "SOXX"]
    ], ignore_index=True)
    prices.to_parquet(directory / "prices.parquet", index=False)
    event = {"event_id": "AMD_original", "source_ticker": "AMD", "filing_date": dates[100].date().isoformat(),
             **text_features("Revenue increased on strong demand.")}
    (directory / "events.json").write_text(json.dumps([event]))
    (directory / "study_config.json").write_text(StudyConfig(start="2020-01-01", end="2020-12-31").model_dump_json())
    (directory / "study.json").write_text('{"original": true}')
    (directory / "predictions.csv").write_text("original,results\n")
    monkeypatch.setattr(cli, "DATA_DIR", directory)
    return directory


@pytest.mark.parametrize("custom_output", [False, True])
def test_study_preserves_original_inputs_and_reference(monkeypatch, tmp_path, cached_inputs, custom_output):
    original = {path: path.read_bytes() for path in cached_inputs.iterdir()}
    reference = hashlib.sha256(PILOT_PATH.read_bytes()).hexdigest()
    output = tmp_path / "experiment" if custom_output else cached_inputs / "runs" / "latest"
    arguments = ["study", "--cost-bps", "50"]
    if custom_output:
        arguments.extend(["--output", str(output)])
    invoke(monkeypatch, *arguments)
    report = json.loads((output / "study.json").read_text())
    assert report["study"]["cost_bps"] == 50
    assert report["summary"]["source_events"] == 1
    assert (output / "prepared_events.json").is_file()
    assert {path: path.read_bytes() for path in original} == original
    assert hashlib.sha256(PILOT_PATH.read_bytes()).hexdigest() == reference


def test_missing_inputs_fail_without_creating_outputs(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "DATA_DIR", tmp_path)
    with pytest.raises(SystemExit) as error:
        invoke(monkeypatch, "study")
    assert error.value.code == 1
    assert "earnings-study reproduce" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("cost_override", [None, "75"])
def test_new_collection_refreshes_saved_run_date_bounds(monkeypatch, cached_inputs, cost_override):
    early = StudyConfig(start="2020-01-01", end="2020-05-31", cost_bps=50)
    (cached_inputs / "study_config.json").write_text(early.model_dump_json())
    (cached_inputs / "config.json").write_text(early.model_dump_json())
    invoke(monkeypatch, "study")
    output = cached_inputs / "runs" / "latest"
    first = json.loads((output / "study.json").read_text())
    assert first["summary"]["source_events"] == 1
    events = json.loads((cached_inputs / "events.json").read_text())
    events.append({**events[0], "event_id": "AMD_later", "filing_date": "2020-07-01"})
    (cached_inputs / "events.json").write_text(json.dumps(events))
    collected = StudyConfig(start="2020-01-01", end="2020-12-31", cost_bps=20)
    for name in ("config.json", "study_config.json"):
        (cached_inputs / name).write_text(collected.model_dump_json())
    arguments = ("--cost-bps", cost_override) if cost_override else ()
    invoke(monkeypatch, "study", *arguments)
    latest = json.loads((output / "study.json").read_text())
    assert latest["summary"]["source_events"] == 2
    assert latest["study"]["period_end"] == "2020-07-01"
    assert latest["study"]["data_as_of"] == "2020-12-31"
    assert latest["study"]["cost_bps"] == (75 if cost_override else 50)


def test_reproduce_command_works_without_network(monkeypatch, tmp_path, capsys):
    def no_network(*args, **kwargs):
        raise AssertionError("Reference replay must be offline")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    invoke(monkeypatch, "reproduce", "--output", str(tmp_path / "replay"))
    report = json.loads((tmp_path / "replay" / "report.json").read_text())
    assert report["verification"]["max_metric_error"] == 0
    assert "1,050 forecasts" in capsys.readouterr().out


def test_web_command_is_not_available(monkeypatch, capsys):
    with pytest.raises(SystemExit) as help_result:
        invoke(monkeypatch, "--help")
    assert help_result.value.code == 0
    assert "serve" not in capsys.readouterr().out
    with pytest.raises(SystemExit) as removed_command:
        invoke(monkeypatch, "serve")
    assert removed_command.value.code == 2


def test_analyze_command_exports_verified_research(monkeypatch, tmp_path, capsys):
    def no_network(*args, **kwargs):
        raise AssertionError("Research analysis must be offline")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    output = tmp_path / "research"
    invoke(monkeypatch, "analyze", "--output", str(output))
    report = json.loads((output / "summary.json").read_text())
    assert report["verification"]["status"] == "verified"
    assert "1,050 forecasts" in capsys.readouterr().out
