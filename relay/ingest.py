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
from bs4 import BeautifulSoup, NavigableString

from relay.config import COMPANIES, DATA_DIR, StudyConfig

log = logging.getLogger(__name__)


class PublicData:
    def __init__(self, directory: Path, refresh: bool = False):
        self.directory = directory
        self.refresh = refresh
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "Earnings Disclosure Spillovers research; https://thomasou.com"
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
    for node in list(soup.find_all(["p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "tr"])):
        node.insert_before(NavigableString("\n"))
        node.insert_after(NavigableString("\n"))
    text = "\n".join(re.sub(r"\s+", " ", line).strip() for line in soup.get_text(" ", strip=False).splitlines() if line.strip())
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


DISCLAIMER_HEADING = re.compile(
    r"(?:forward[-\s]looking statements?|safe[-\s]harbor(?: statements?)?|"
    r"cautionary statements?(?: (?:regarding|concerning|about) forward[-\s]looking statements?)?|"
    r"cautionary note (?:regarding|concerning|about) forward[-\s]looking statements?|risk factors)\s*[:.]?", re.I,
)
DISCLAIMER_PARAGRAPH = re.compile(
    r"^(?:this|the) (?:(?:press|news|earnings) )?(?:release|document|report|communication) "
    r"(?:contains|includes)\b.{0,100}\bforward[-\s]looking", re.I,
)
RESEARCH_HEADING = re.compile(
    r"(?:(?:business|financial|company) )?(?:outlook|guidance|results|highlights)"
    r"(?: for [\w ,'-]{1,60})?|(?:condensed )?(?:consolidated )?financial statements|"
    r"(?:condensed )?consolidated statements of .{1,60}|about .{1,60}", re.I,
)
NEGATED_PREFIX = re.compile(
    r"\b(?:no|not|never|without|lose|lost|losing|lack(?:s|ed|ing)?|fail(?:s|ed|ing)? to)"
    r"\s+(?:[\w’'-]+\s+){0,3}$", re.I,
)


def research_text(text: str) -> tuple[str, dict]:
    kept = []
    excluding = False
    removed = sections = 0
    for line in text.splitlines():
        line = re.sub(r"\s+", " ", line).strip()
        if not line:
            continue
        if DISCLAIMER_HEADING.fullmatch(line):
            sections += int(not excluding)
            excluding = True
        elif excluding and RESEARCH_HEADING.fullmatch(line.rstrip(":")):
            excluding = False
        if excluding:
            removed += len(line)
        elif DISCLAIMER_PARAGRAPH.match(line):
            sections += 1
            removed += len(line)
        else:
            kept.append(line)
    return " ".join(kept), {"boilerplate_sections": sections, "boilerplate_characters_removed": removed}


def text_features(text: str) -> dict:
    # Explicit lexical features, not causal labels or an LLM return forecast.
    scoring_text, audit = research_text(text)
    normalized = scoring_text.lower()
    scores = {}
    counts = {}
    matches = []
    for name, (positive, negative) in PATTERNS.items():
        pos = [match for match in re.finditer(positive, normalized)
               if not NEGATED_PREFIX.search(normalized[max(0, match.start() - 80):match.start()])]
        neg = [match for match in re.finditer(negative, normalized)
               if not NEGATED_PREFIX.search(normalized[max(0, match.start() - 80):match.start()])]
        counts[name] = len(pos) + len(neg)
        scores[f"{name}_signal"] = (len(pos) - len(neg)) / (len(pos) + len(neg) + 1)
        matches.extend(pos + neg)
    dominant = max(counts, key=counts.get)
    scores["event_type"] = dominant if counts[dominant] else "earnings"
    scores["text_available"] = float(bool(scoring_text))
    scores["excerpt"] = scoring_text[max(0, matches[0].start() - 100):matches[0].end() + 230] if matches else scoring_text[:330]
    scores["text_audit"] = audit
    return scores


class CachedPublicData:
    def __init__(self, directory: Path):
        self.directory = directory
        self.responses: dict[str, str] = {}

    def get(self, url: str) -> str:
        path = self.directory / f"{hashlib.sha256(url.encode()).hexdigest()}.json"
        content = path.read_bytes()
        payload = json.loads(content)
        if not isinstance(payload, dict) or payload.get("url") != url or not isinstance(payload.get("body"), str):
            raise ValueError("Cached source does not match the requested document")
        self.responses[url] = hashlib.sha256(content).hexdigest()
        return payload["body"]


def prepare_cached_events(events: list[dict], directory: Path) -> tuple[list[dict], dict, list[str]]:
    client = CachedPublicData(directory / "raw")
    prepared = []
    unavailable_cache = unresolved = changed = 0
    feature_keys = [f"{name}_signal" for name in PATTERNS] + ["text_available", "document_kind"]
    for event in events:
        source_url = event.get("source_url", "")
        try:
            if not isinstance(source_url, str) or not source_url:
                raise ValueError("Original filing URL is missing")
            body, url = extract_release(client, event)
            status = "reprocessed" if body else "release_unresolved"
            unresolved += int(not body)
        except (OSError, ValueError):
            body, url, status = "", source_url, "cached_source_unavailable"
            unavailable_cache += 1
        row = {**event, **text_features(body), "evidence_url": url,
               "document_kind": "earnings_exhibit" if body else "release_unavailable", "text_source_status": status}
        changed += int(any(row[key] != event.get(key) for key in feature_keys))
        prepared.append(row)
    responses = [{"url": url, "sha256": digest} for url, digest in sorted(client.responses.items())]
    audit = {
        "mode": "cached_source_reprocessing", "network_requests": 0,
        "extractor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "raw_responses": responses,
        "raw_responses_sha256": hashlib.sha256(json.dumps(responses, sort_keys=True).encode()).hexdigest(),
        "prepared_events_sha256": hashlib.sha256(json.dumps(prepared, sort_keys=True, allow_nan=False).encode()).hexdigest(),
        "events": len(prepared), "reprocessed_exhibits": sum(row["document_kind"] == "earnings_exhibit" for row in prepared),
        "changed_events": changed, "unresolved_releases": unresolved, "unavailable_cached_sources": unavailable_cache,
        "events_with_boilerplate_removed": sum(row["text_audit"]["boilerplate_sections"] > 0 for row in prepared),
        "boilerplate_characters_removed": sum(row["text_audit"]["boilerplate_characters_removed"] for row in prepared),
    }
    warnings = []
    if unavailable_cache:
        warnings.append(f"{unavailable_cache} events lack readable cached source documents; text features are unavailable, not reused from older extraction.")
    if unresolved:
        warnings.append(f"{unresolved} original releases remain unresolved; cover filings are excluded from text features.")
    return prepared, audit, warnings


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
