import numpy as np
import pandas as pd
import pytest

from relay import selection
from relay.model import EVENT_FEATURES


def selection_frame(events: int = 50) -> pd.DataFrame:
    rng = np.random.default_rng(2026)
    rows = []
    for i, date in enumerate(pd.bdate_range("2018-01-01", periods=events, freq="5B")):
        for peer in range(5):
            rows.append({
                "event_id": f"event-{i}", "decision_date": date,
                "exit_date": date + pd.offsets.BDay(5),
                "realized_excess_return": rng.normal(0, 0.04),
                **{name: rng.normal(0, 1) for name in EVENT_FEATURES},
            })
    return pd.DataFrame(rows)


def test_inner_selection_purges_labels_and_refits_winner(monkeypatch):
    frame = selection_frame()
    fitted = []
    original = selection._fit_candidate

    def record(name, sample, *features):
        fitted.append((name, sample.copy()))
        return original(name, sample, *features)

    monkeypatch.setattr(selection, "_fit_candidate", record)
    model, audit = selection.fit_selected(frame)
    cutoff = pd.Timestamp(audit["validation_start"])
    assert audit["purged_inner_fit_rows"] == 5
    assert audit["inner_fit"]["latest_exit"] < audit["inner_validation"]["decision_start"]
    assert [name for name, _ in fitted[:-1]] == list(selection.CANDIDATES)
    for _, sample in fitted[:-1]:
        assert sample["exit_date"].max() < cutoff
        assert sample["decision_date"].max() < cutoff
    assert fitted[-1][0] == audit["selected"] == model.name
    pd.testing.assert_frame_equal(fitted[-1][1], frame)
    assert len(audit["candidate_scores"]) == len(selection.CANDIDATES)


def test_small_inner_fit_falls_back_without_fitting_complex_models():
    model, audit = selection.fit_selected(selection_frame(24))
    assert model.name == "zero"
    assert audit["fallback_reason"]
    assert audit["candidate_scores"] == []
    np.testing.assert_array_equal(model.predict(selection_frame(3)), np.zeros(15))


def test_ties_choose_simplest_candidate():
    frame = selection_frame()
    frame["realized_excess_return"] = 0.0
    model, audit = selection.fit_selected(frame)
    assert model.name == "zero"
    assert audit["fallback_reason"] is None
    assert all(row["mae"] == 0 for row in audit["candidate_scores"])


@pytest.mark.parametrize("name", selection.CANDIDATES)
def test_candidate_handles_constant_features_and_is_prediction_only(name):
    frame = selection_frame()
    frame["guidance_signal"] = 0.0
    frame["text_available"] = 1.0
    model = selection._fit_candidate(name, frame)
    original = model.predict(frame)
    changed_outcomes = frame.copy()
    changed_outcomes["realized_excess_return"] = 1_000_000
    assert np.isfinite(original).all()
    np.testing.assert_array_equal(original, model.predict(changed_outcomes))


def test_deterministic_with_unsorted_rows():
    frame = selection_frame()
    left, audit_left = selection.fit_selected(frame)
    right, audit_right = selection.fit_selected(frame.sample(frac=1, random_state=42))
    assert left.name == right.name
    assert audit_left["selected"] == audit_right["selected"]
    np.testing.assert_allclose(left.predict(frame), right.predict(frame), atol=1e-12)
