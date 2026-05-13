from __future__ import annotations

import csv
import io
import json
import logging
import threading
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from relay.config import DATA_DIR, PILOT_PATH
from relay.study import run_cached_study

STATIC = Path(__file__).resolve().parent / "static"
app = FastAPI(title="Relay · Event-driven equity research", version="0.1.0")
_lock = threading.Lock()
_run = {"status": "idle", "message": ""}
log = logging.getLogger(__name__)


def read_study() -> dict:
    path = DATA_DIR / "study.json"
    if not path.exists():
        path = PILOT_PATH
    if not path.exists():
        return {"study": {"status": "insufficient_data", "title": "No study built yet", "notes": ["Run relay collect, then relay study."]},
                "summary": {"events": 0, "predictions": 0, "test_predictions": 0, "source_tickers": [], "text_coverage": 0},
                "events": [], "benchmarks": [], "methodology": [], "folds": [], "comparisons": []}
    return json.loads(path.read_text())


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/research")
def research() -> JSONResponse:
    return JSONResponse(read_study(), headers={"Cache-Control": "no-store"})


@app.get("/api/research/status")
def research_status() -> dict:
    with _lock:
        return dict(_run)


def rebuild() -> None:
    try:
        run_cached_study()
    except Exception:
        log.exception("Study rebuild failed")
        with _lock:
            _run.update(status="failed", message="Study rebuild failed. Check the server log and local input data.")
    else:
        with _lock:
            _run.update(status="idle", message="Study rebuilt from cached public data.")


@app.post("/api/research/run", status_code=202)
def rerun(request: Request, background: BackgroundTasks) -> dict:
    origin = request.headers.get("origin")
    if origin and origin.rstrip("/") != str(request.base_url).rstrip("/"):
        raise HTTPException(403, "Rebuild requests must originate from this app.")
    if not all((DATA_DIR / name).exists() for name in ["prices.parquet", "events.json", "config.json"]):
        raise HTTPException(422, "Local research inputs are missing. Run `relay collect` and `relay study` first. The bundled study remains available to inspect.")
    with _lock:
        if _run["status"] == "running":
            return dict(_run)
        _run.update(status="running", message="Fitting chronological models and evaluating held-out forecasts…")
    background.add_task(rebuild)
    return {"status": "running", "message": _run["message"]}


@app.get("/api/research/predictions.csv")
def predictions_csv() -> Response:
    path = DATA_DIR / "predictions.csv"
    if path.exists():
        return FileResponse(path, media_type="text/csv", filename="relay-predictions.csv")
    rows = []
    for event in read_study()["events"]:
        for prediction in event["predictions"]:
            rows.append({"event_id": event["id"], "source_ticker": event["source_ticker"], "filing_date": event["date"],
                         "decision_date": event["decision_date"], "entry_date": event["entry_date"], "exit_date": event["exit_date"],
                         **prediction})
    if not rows:
        raise HTTPException(404, "No forecasts available to export.")
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return Response(output.getvalue(), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=relay-predictions.csv"})


app.mount("/static", StaticFiles(directory=STATIC), name="static")
