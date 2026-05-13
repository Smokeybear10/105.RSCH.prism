from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

from relay.config import COMPANIES, DATA_DIR, StudyConfig

log = logging.getLogger(__name__)


class PublicData:
    def __init__(self, directory: Path, refresh: bool = False):
        self.directory = directory
        self.refresh = refresh
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "Relay academic equity research; https://thomasou.com"
        self.last_sec_request = 0.0

    def get(self, url: str) -> str:
        path = self.directory / f"{hashlib.sha256(url.encode()).hexdigest()}.json"
        if path.exists() and not self.refresh:
            return json.loads(path.read_text())["body"]
        if "sec.gov" in url:
            time.sleep(max(0.0, 0.22 - (time.monotonic() - self.last_sec_request)))
        response = self.session.get(url, timeout=30)
        self.last_sec_request = time.monotonic()
        response.raise_for_status()
        self.directory.mkdir(parents=True, exist_ok=True)
        record = {"url": url, "retrieved_at": datetime.now(timezone.utc).isoformat(), "body": response.text}
        path.write_text(json.dumps(record))
        return response.text

    def json(self, url: str) -> dict:
        return json.loads(self.get(url))


def fetch_prices(client: PublicData, ticker: str, config: StudyConfig) -> pd.DataFrame:
    start = int(pd.Timestamp(config.start, tz="UTC").timestamp()) - 180 * 86400
    end = int(pd.Timestamp(config.end, tz="UTC").timestamp()) + 86400
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?period1={start}&period2={end}&interval=1d"
    payload = client.json(url)["chart"]
    if payload.get("error") or not payload.get("result"):
        raise ValueError(f"No price history for {ticker}: {payload.get('error')}")
    result = payload["result"][0]
    quotes = result["indicators"]["quote"][0]
    adjusted = result["indicators"].get("adjclose", [{}])[0].get("adjclose")
    if adjusted is None:
        raise ValueError(f"Adjusted close unavailable for {ticker}")
    frame = pd.DataFrame({
        "ticker": ticker,
        "date": pd.to_datetime(result["timestamp"], unit="s", utc=True).tz_convert("America/New_York").date,
        "open": quotes["open"], "close": quotes["close"], "adj_close": adjusted,
        "volume": quotes["volume"],
    }).dropna(subset=["open", "close", "adj_close"])
    frame = frame[(frame["open"] > 0) & (frame["close"] > 0) & (frame["adj_close"] > 0)]
    frame = frame[frame["date"] <= config.end].copy()
    frame["adj_open"] = frame["open"] * frame["adj_close"] / frame["close"]
    return frame.drop_duplicates("date").sort_values("date").reset_index(drop=True)


def filing_records(client: PublicData, ticker: str, config: StudyConfig) -> list[dict]:
    cik, _ = COMPANIES[ticker]
    payload = client.json(f"https://data.sec.gov/submissions/CIK{cik:010d}.json")
    tables = [payload["filings"]["recent"]]
    for item in payload["filings"].get("files", []):
        if item["filingTo"] >= config.start.isoformat() and item["filingFrom"] <= config.end.isoformat():
            tables.append(client.json(f"https://data.sec.gov/submissions/{item['name']}"))
    records = []
    for table in tables:
        for i, form in enumerate(table["form"]):
            filing_date = table["filingDate"][i]
            if form != "8-K" or "2.02" not in table.get("items", [""] * len(table["form"]))[i]:
                continue
            if not config.start.isoformat() <= filing_date <= config.end.isoformat():
                continue
            accession = table["accessionNumber"][i]
            base = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession.replace('-', '')}/"
            records.append({
                "event_id": f"{ticker}_{accession}", "source_ticker": ticker,
                "filing_date": filing_date, "event_at": table["acceptanceDateTime"][i],
                "accession": accession, "source_url": base + table["primaryDocument"][i],
                "headline": f"{COMPANIES[ticker][1]} · results of operations",
            })
    # Multiple releases on the same day are one information set for daily research.
    return list({r["filing_date"]: r for r in sorted(records, key=lambda x: x["event_at"])}.values())


def _original_document_url(source_url: str, href: str) -> str | None:
    if not re.search(r"\.html?(?:$|[?#])", href, re.I):
        return None
    candidate = urljoin(source_url, href)
    if candidate == source_url or not candidate.startswith(source_url.rsplit("/", 1)[0] + "/"):
        return None
    return candidate


def _is_release_description(text: str) -> bool:
    return bool(re.search(r"(?:press|news|earnings)\s+release", text, re.I))


def _exhibit_numbers(text: str) -> set[str]:
    return {number.replace("_", ".").replace("-", ".")
            for number in re.findall(r"(?<!\d)99(?:[._-]\d+)?(?![\d.])", text)}


def extract_release(client: PublicData, record: dict) -> tuple[str, str]:
    source_url = record["source_url"]
    soup = BeautifulSoup(client.get(source_url), "html.parser")
    release_exhibits = set()
    for node in soup.find_all(["tr", "p"]):
        description = node.get_text(" ", strip=True)
        exhibits = _exhibit_numbers(description)
        if len(exhibits) == 1 and _is_release_description(description):
            release_exhibits.update(exhibits)
    candidates = []
    for link in soup.find_all("a", href=True):
        context = (link.find_parent("tr") or link.parent).get_text(" ", strip=True)
        label = link.get_text(" ", strip=True)
        context_exhibits = _exhibit_numbers(context)
        exhibits = context_exhibits if len(context_exhibits) == 1 else _exhibit_numbers(f"{label} {link['href']}")
        if len(exhibits) != 1 or not exhibits <= {"99", "99.1"}:
            continue
        if not (_is_release_description(context) or exhibits <= release_exhibits):
            continue
        candidate = _original_document_url(source_url, link["href"])
        if candidate:
            candidates.append(candidate)

    if not candidates:
        accession = record.get("accession", "")
        if not re.fullmatch(r"\d{10}-\d{2}-\d{6}", accession):
            return "", source_url
        index_url = urljoin(source_url, f"{accession}-index.html")
        index = BeautifulSoup(client.get(index_url), "html.parser")
        for row in index.find_all("tr"):
            cells = row.find_all("td", recursive=False)
            if len(cells) < 4 or cells[3].get_text(strip=True).upper() not in {"EX-99", "EX-99.1"}:
                continue
            exhibit = cells[3].get_text(strip=True).upper().removeprefix("EX-")
            description = cells[1].get_text(" ", strip=True)
            generic_description = not description or re.fullmatch(r"(?:EXHIBIT\s*|EX-)?99(?:\.1)?", description, re.I)
            if not (_is_release_description(description) or (generic_description and exhibit in release_exhibits)):
                continue
            for link in cells[2].find_all("a", href=True):
                candidate = _original_document_url(source_url, link["href"])
                if candidate:
                    candidates.append(candidate)

    candidates = list(dict.fromkeys(candidates))
    if len(candidates) != 1:
        return "", source_url
    url = candidates[0]
    soup = BeautifulSoup(client.get(url), "html.parser")
    for node in soup(["script", "style", "ix:header", "head"]):
        node.decompose()
    text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True))
    return text[:120000], url


PATTERNS = {
    "demand": (r"strong(?:er)? demand|demand (?:growth|increas|improv)|record (?:revenue|sales)|revenue (?:grew|increased)|sales (?:grew|increased)",
               r"weak(?:er)? demand|demand (?:declin|decreas|soft)|revenue (?:declined|decreased)|sales (?:declined|decreased)"),
    "pricing": (r"higher (?:selling )?prices|pricing (?:power|improv)|margin(?:s)? (?:expanded|increased)|higher (?:gross )?margin",
                r"lower (?:selling )?prices|pricing pressure|margin(?:s)? (?:contracted|decreased)|lower (?:gross )?margin"),
    "competitive": (r"market share (?:gain|increas)|gain(?:ed|ing)? (?:market )?share|competitive (?:advantage|strength)|design wins",
                    r"market share (?:loss|declin)|los(?:t|ing) (?:market )?share|competitive pressure"),
    "guidance": (r"rais(?:e|ed|ing) (?:our |its |the )?(?:guidance|outlook)|increas(?:e|ed|ing) (?:our |its |the )?(?:guidance|outlook)",
                 r"lower(?:ed|ing)? (?:our |its |the )?(?:guidance|outlook)|reduc(?:e|ed|ing) (?:our |its |the )?(?:guidance|outlook)"),
}


def text_features(text: str) -> dict:
    # Explicit lexical features, not causal labels or an LLM return forecast.
    normalized = text.lower()
    scores = {}
    counts = {}
    matches = []
    for name, (positive, negative) in PATTERNS.items():
        pos = list(re.finditer(positive, normalized))
        neg = list(re.finditer(negative, normalized))
        counts[name] = len(pos) + len(neg)
        scores[f"{name}_signal"] = (len(pos) - len(neg)) / (len(pos) + len(neg) + 1)
        matches.extend(pos + neg)
    dominant = max(counts, key=counts.get)
    scores["event_type"] = dominant if counts[dominant] else "earnings"
    scores["text_available"] = float(bool(text))
    scores["excerpt"] = text[max(0, matches[0].start() - 100):matches[0].end() + 230] if matches else text[:330]
    return scores


def collect_data(config: StudyConfig, directory: Path = DATA_DIR, refresh: bool = False,
                 progress: Callable[[str], None] = print) -> tuple[pd.DataFrame, list[dict], list[str]]:
    client = PublicData(directory / "raw", refresh=refresh)
    prices = []
    for ticker in [*COMPANIES, "SPY", "SOXX"]:
        progress(f"Loading adjusted daily prices · {ticker}")
        prices.append(fetch_prices(client, ticker, config))
    records = []
    warnings = []
    for ticker in COMPANIES:
        progress(f"Loading SEC earnings disclosures · {ticker}")
        events = filing_records(client, ticker, config)
        for i, record in enumerate(events):
            if i % 10 == 0:
                progress(f"Reading original exhibits · {ticker} · {i + 1}/{len(events)}")
            try:
                text, url = extract_release(client, record)
                record.update(text_features(text), evidence_url=url,
                              document_kind="earnings_exhibit" if text else "release_unavailable")
                if not text:
                    warnings.append(f"Original earnings release unresolved for {record['event_id']}; cover filing excluded from text features.")
            except requests.RequestException as exc:
                # Preserve the event for price-only benchmarks; missing text is explicit.
                record.update(text_features(""), evidence_url=record["source_url"], document_kind="unavailable")
                warnings.append(f"Text unavailable for {record['event_id']}: {type(exc).__name__}")
            records.append(record)
    directory.mkdir(parents=True, exist_ok=True)
    panel = pd.concat(prices, ignore_index=True)
    panel.to_parquet(directory / "prices.parquet", index=False)
    (directory / "events.json").write_text(json.dumps(records, indent=2))
    (directory / "config.json").write_text(config.model_dump_json(indent=2))
    (directory / "study_config.json").write_text(config.model_dump_json(indent=2))
    (directory / "ingestion_warnings.json").write_text(json.dumps(warnings))
    return panel, records, warnings
