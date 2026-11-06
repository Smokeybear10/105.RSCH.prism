"""
Export per-ticker JSON fixtures for the historical attribution evaluation.

Run from the project root:
    python demo/build_static.py

Output:
    demo/static/data/{TICKER}.json    # prices + flagged moves + attributions + chunks
    demo/static/data/index.json       # tickers list + metadata
"""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import date, timedelta
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from backtest.signal import STRATEGY_REGISTRY
from demo.mock_data import FOCAL_TICKERS
from demo.pnl_summary import build_pnl_summary
from demo.real_chunks import (
    chunks_for_real,
    preload_earnings_transcripts,
    preload_finnhub_news,
    preload_news,
    preload_peer_and_sector_news,
    preload_thirteen_f,
)
from ingestion.prices import detect_significant_moves, load_prices
from model import attribute as model_attribute
from schema import AblationConfig, PriceMove, SourceType

# Full stack for the pre-baked "default" attribution on initial render.
FULL_STACK_SOURCES = [
    SourceType.NEWS,
    SourceType.SEC_10K,
    SourceType.SEC_8K,
    SourceType.EARNINGS_TRANSCRIPT,
    SourceType.PEER_NEWS,
    SourceType.MACRO,
    SourceType.THIRTEEN_F,
]

# Mirrors backtest.fixtures.ABLATION_BUNDLES but swaps RESEARCH_13F →
# THIRTEEN_F because chunks_for_real produces THIRTEEN_F-tagged chunks
# (demo/real_chunks.py:236). Without the swap, the +positioning filter
# would zero out and the chart-overlay's full-stack predicted line
# would equal the +macro one. Keep this dict in (count → ablation)
# order — must match COUNT_TO_BUNDLE in app.js.
DEMO_ABLATION_BUNDLES: dict[str, list[SourceType]] = {
    "base_news":    [SourceType.NEWS],
    "+sec":         [SourceType.NEWS, SourceType.SEC_10K, SourceType.SEC_8K],
    "+earnings":    [SourceType.NEWS, SourceType.SEC_10K, SourceType.SEC_8K,
                     SourceType.EARNINGS_TRANSCRIPT],
    "+peer_news":   [SourceType.NEWS, SourceType.SEC_10K, SourceType.SEC_8K,
                     SourceType.EARNINGS_TRANSCRIPT, SourceType.PEER_NEWS],
    "+sector_news": [SourceType.NEWS, SourceType.SEC_10K, SourceType.SEC_8K,
                     SourceType.EARNINGS_TRANSCRIPT, SourceType.PEER_NEWS,
                     SourceType.SECTOR_NEWS],
    "+macro":       [SourceType.NEWS, SourceType.SEC_10K, SourceType.SEC_8K,
                     SourceType.EARNINGS_TRANSCRIPT, SourceType.PEER_NEWS,
                     SourceType.SECTOR_NEWS, SourceType.MACRO],
    "+positioning": [SourceType.NEWS, SourceType.SEC_10K, SourceType.SEC_8K,
                     SourceType.EARNINGS_TRANSCRIPT, SourceType.PEER_NEWS,
                     SourceType.SECTOR_NEWS, SourceType.MACRO,
                     SourceType.THIRTEEN_F],
}

OUT_DIR = _ROOT / "demo" / "static" / "data"
WINDOW_YEARS = 5
CHUNK_TEXT_CAP = 600  # trim chunk body for payload size

# Cap concurrent live LLM calls per move at 4. Anthropic per-key tier
# allows much higher, but 4 keeps a single move's burst small enough that
# rate-limit retries don't dominate wall time.
ABLATION_CONCURRENCY = 4


async def _gather_predictions_by_ablation(
    m, chunks,
) -> "tuple[dict[str, float | None], dict[str, int]]":
    """Fan out the 7 canonical ablations for one move and gather their
    predicted_return_pct values into a {ablation_name: float|None} dict,
    plus a parallel {ablation_name: chunks_added_vs_immediate_prior} dict
    used by the UI to fade diamonds that contributed no new evidence.

    Dedupe: many pre-coverage moves have empty NEWS / PEER / SECTOR / 13F
    pools, so multiple cumulative ablations resolve to the byte-identical
    chunk-id set. Calling the model once per UNIQUE set and fanning the
    result out cuts cost by 5× on those moves without changing the result
    (same input → same prediction). Concurrency is preserved via
    asyncio.create_task; the Semaphore caps in-flight threads."""
    sem = asyncio.Semaphore(ABLATION_CONCURRENCY)

    # Pre-compute per-ablation filtered chunks + their chunk-id signature.
    items: list[tuple[str, list, list, frozenset]] = []
    for name, sources in DEMO_ABLATION_BUNDLES.items():
        src_set = set(sources)
        ab_chunks = [c for c in chunks if c.source_type in src_set]
        ab_ids = frozenset(c.chunk_id for c in ab_chunks)
        items.append((name, list(sources), ab_chunks, ab_ids))

    # Plan: for each ablation, decide compute vs. copy-from-earlier vs. none.
    # `seen` maps a chunk-id set → the index of the first ablation that
    # owned it, so later ablations with the identical set just copy.
    seen: dict[frozenset, int] = {}
    plan: list[tuple[str, str, object]] = []
    for i, (name, sources, ab_chunks, ab_ids) in enumerate(items):
        if not ab_chunks:
            plan.append((name, "none", None))
        elif ab_ids in seen:
            plan.append((name, "copy", seen[ab_ids]))
        else:
            seen[ab_ids] = i
            plan.append((name, "compute", (sources, ab_chunks)))

    async def _run(ab_chunks_local, cfg_local):
        async with sem:
            return await asyncio.to_thread(model_attribute, m, ab_chunks_local, cfg_local)

    tasks: dict[int, asyncio.Task] = {}
    for i, (name, action, payload) in enumerate(plan):
        if action == "compute":
            sources, ab_chunks = payload  # type: ignore[misc]
            cfg = AblationConfig(
                name=name, sources=sources,
                description="pre-baked ablation",
            )
            tasks[i] = asyncio.create_task(_run(ab_chunks, cfg))

    results_by_index: dict[int, "float | None"] = {}
    for i, task in tasks.items():
        attr = await task
        results_by_index[i] = (
            round(float(attr.predicted_return_pct), 6)
            if attr.predicted_return_pct is not None else None
        )

    predictions: dict[str, "float | None"] = {}
    for i, (name, action, payload) in enumerate(plan):
        if action == "compute":
            predictions[name] = results_by_index[i]
        elif action == "copy":
            predictions[name] = results_by_index.get(payload)  # type: ignore[arg-type]
        else:  # "none"
            predictions[name] = None

    # Per-ablation count of chunks newly added vs. the immediately prior
    # ablation. The UI uses this to fade diamonds that repeat the previous
    # prediction because no new evidence was actually added.
    new_counts: dict[str, int] = {}
    prev_ids: frozenset = frozenset()
    for name, _, _, ab_ids in items:
        new_counts[name] = len(ab_ids - prev_ids)
        prev_ids = ab_ids

    return predictions, new_counts


def _news_coverage_start(ticker: str) -> "date | None":
    """Earliest publication_date for `ticker` in the bundled news parquet,
    or None if no rows. Used to restrict the demo's flagged-moves list so
    every clickable dot has News + Peer-news data populated."""
    from demo.real_chunks import _NEWS_BY_TICKER, preload_news
    if ticker.upper() not in _NEWS_BY_TICKER:
        preload_news([ticker])
    df = _NEWS_BY_TICKER.get(ticker.upper())
    if df is None or len(df) == 0 or "_pub_date" not in df.columns:
        return None
    dates = [d for d in df["_pub_date"] if d is not None]
    return min(dates) if dates else None


def _dim_dict(score) -> dict:
    cited = []
    for ce in (score.cited_evidence or []):
        cited.append({
            "chunk_id": ce.chunk_id,
            "quote": ce.quote,
            "reasoning": ce.reasoning,
        })
    return {
        "cited_evidence": cited,
        "weight": round(float(score.weight), 4),
        "direction": score.direction,
        "rationale": score.rationale,
        "evidence_chunk_ids": list(score.evidence_chunk_ids),
    }


def _chunk_dict(c) -> dict:
    text = c.text if len(c.text) <= CHUNK_TEXT_CAP else c.text[:CHUNK_TEXT_CAP] + "…"
    return {
        "chunk_id": c.chunk_id,
        "source_type": c.source_type.value,
        "publication_date": c.publication_date.isoformat(),
        "section_name": c.section_name,
        "source_url": c.source_url,
        "text": text,
    }


def build_for_ticker(ticker: str, meta: dict, end_date: date) -> dict:
    start_date = end_date - timedelta(days=365 * WINDOW_YEARS)

    prices_df = load_prices([ticker], as_of=end_date)
    prices_df = prices_df[
        (prices_df["date"] >= start_date) & (prices_df["date"] <= end_date)
    ].reset_index(drop=True)

    prices = [
        {"date": d.isoformat() if hasattr(d, "isoformat") else str(d)[:10],
         "close": round(float(c), 4)}
        for d, c in zip(prices_df["date"], prices_df["close"])
    ]

    full_moves = detect_significant_moves(
        load_prices([ticker], as_of=end_date),
        lookback_vol=63,  # ~3 trading months
    )
    # Older flagged moves are kept even though Yahoo News only covers ~2025+;
    # the UI auto-disables source toggles whose chunk_available count is 0,
    # so users can still inspect SEC + Earnings + Macro + 13F evidence on
    # historical moves and the empty-data sources are visibly off.
    moves_in_window: list[PriceMove] = [
        m for m in full_moves
        if start_date <= m.move_date <= end_date
        and abs(m.vol_zscore) >= 3.0  # demo: keep only 3-sigma+ moves to declutter chart
    ]

    moves_sorted = sorted(moves_in_window, key=lambda x: x.move_date)
    moves_payload: list[dict] = []
    for idx, m in enumerate(moves_sorted, start=1):
        print(f"  [{idx}/{len(moves_sorted)}] {ticker} {m.move_date} "
              f"(return {m.return_pct:+.2%})…", flush=True)
        chunks = chunks_for_real(ticker, m.move_date)
        # Truthful counts over the FULL chunk pool (not just top-10) so the
        # UI toggle row can show accurate chunk counts on initial render.
        available_counts: dict[str, int] = {}
        for c in chunks:
            available_counts[c.source_type.value] = \
                available_counts.get(c.source_type.value, 0) + 1
        # "+macro" is the lowest-noise bundle in backtest.fixtures and matches
        # the 6-source full stack we pre-bake.
        config = AblationConfig(
            name="+macro",
            sources=list(FULL_STACK_SOURCES),
            description="pre-baked full-stack default",
        )
        attr = model_attribute(m, chunks, config)
        # Pre-compute strategy verdicts so the UI shows lean/fade/skip on
        # initial render (no API roundtrip needed).
        strategies: dict[str, str] = {}
        for sname, sfn in STRATEGY_REGISTRY.items():
            try:
                strategies[sname] = sfn(attr)
            except Exception:
                strategies[sname] = "neutral"

        # Per-ablation predicted_return_pct, baked once at build time so the
        # chart-overlay in app.js can switch source filters without round
        # trips. Each ablation re-filters the same `chunks` pool by source
        # type and re-attributes. Live LLM mode dominates wall-time here, so
        # we run the 7 ablation calls in parallel via asyncio + a semaphore
        # cap — synthetic mode is fast enough that the overhead is fine.
        predictions_by_ablation, new_chunks_by_ablation = asyncio.run(
            _gather_predictions_by_ablation(m, chunks)
        )

        # Bundle's `chunks` field must include every chunk_id the attribution
        # cites, otherwise the UI flags valid citations as "Missing chunk".
        # `chunks_for_real` returns chunks in stratified round-robin order;
        # `model.attribute` reorders by relevance and cites the top-5 of that
        # ranking, which often includes chunks ranked 11+ in the round-robin
        # order. Collect citations from each dim and union with the top-10.
        cited_ids: set[str] = set()
        for dim in (attr.demand, attr.pricing, attr.competitive,
                    attr.management_credibility, attr.macro):
            for cid in dim.evidence_chunk_ids:
                cited_ids.add(cid)
        chunk_by_id = {c.chunk_id: c for c in chunks}
        bundle_chunks: dict[str, "TextChunk"] = {}
        for c in chunks[:10]:
            bundle_chunks[c.chunk_id] = c
        for cid in cited_ids:
            if cid in chunk_by_id:
                bundle_chunks[cid] = chunk_by_id[cid]

        moves_payload.append({
            "move_date": m.move_date.isoformat(),
            "return_pct": round(float(m.return_pct), 6),
            "vol_zscore": round(float(m.vol_zscore), 3),
            "magnitude_rank": (round(float(m.magnitude_rank), 3)
                               if m.magnitude_rank is not None else None),
            "attribution": {
                "realized": round(float(attr.return_pct), 6),
                "predicted": (round(float(attr.predicted_return_pct), 6)
                              if attr.predicted_return_pct is not None else None),
                "character": attr.move_character,
                "confidence": round(float(attr.confidence), 3),
                "chunks_considered": attr.chunks_considered,
                "sources_used": [s.value for s in attr.sources_used],
                "dimensions": {
                    "demand": _dim_dict(attr.demand),
                    "pricing": _dim_dict(attr.pricing),
                    "competitive": _dim_dict(attr.competitive),
                    "management_credibility": _dim_dict(attr.management_credibility),
                    "macro": _dim_dict(attr.macro),
                },
                "model_notes": attr.model_notes,
            },
            # Bundle the union of (top-10 by round-robin order) + (every
            # chunk_id cited in the attribution). Keeps the bundle small but
            # guarantees the UI can resolve every citation.
            "chunks": [_chunk_dict(c) for c in bundle_chunks.values()],
            "chunks_available": available_counts,
            "chunks_total": len(chunks),
            "strategies": strategies,
            # Used by app.js renderChart to draw the predicted-price overlay.
            # Keyed by ablation name; values are the model's expected return
            # for this move under that source bundle.
            "predictions_by_ablation": predictions_by_ablation,
            # Parallel to predictions_by_ablation: how many chunks this
            # ablation added vs. the previous step. 0 = "same evidence
            # pool as prior ablation, prediction is a copy". The
            # prediction-trajectory chart uses this to fade those
            # diamonds instead of pretending the ablation did work.
            "new_chunks_by_ablation": new_chunks_by_ablation,
        })

    # Closes the loop: dollar P&L for the model + 4 baselines over a 5-day
    # window per flagged event. Failure here must not break the bundle build.
    try:
        pnl = build_pnl_summary(ticker, moves_payload, prices_df)
    except Exception as exc:
        print(f"  WARN: pnl summary failed for {ticker}: {exc}", flush=True)
        pnl = None

    return {
        "ticker": ticker,
        "name": meta["name"],
        "sector": meta["sector"],
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "prices": prices,
        "moves": moves_payload,
        "pnl": pnl,
    }


def _resolve_ticker_order() -> list[str]:
    """Honor BW_TICKER_ORDER=AAA,BBB,... to rebuild only a subset (or reorder).
    Tickers not in FOCAL_TICKERS are dropped with a warning."""
    import os
    raw = os.environ.get("BW_TICKER_ORDER", "").strip()
    if not raw:
        return list(FOCAL_TICKERS.keys())
    requested = [t.strip().upper() for t in raw.split(",") if t.strip()]
    valid = [t for t in requested if t in FOCAL_TICKERS]
    skipped = [t for t in requested if t not in FOCAL_TICKERS]
    if skipped:
        print(f"BW_TICKER_ORDER: skipping unknown tickers {skipped}", flush=True)
    print(f"BW_TICKER_ORDER set; building only: {valid}", flush=True)
    return valid


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    end_date = date.today()

    ticker_order = _resolve_ticker_order()

    print("Preloading news parquet (one-time ~30-60s)…", flush=True)
    preload_news(ticker_order)
    preload_peer_and_sector_news(ticker_order)
    preload_thirteen_f()
    preload_earnings_transcripts(ticker_order)
    # Optional Finnhub historical news (fills the pre-2025 gap in the
    # bundled Yahoo parquet). No-op without FINNHUB_API_KEY.
    preload_finnhub_news(ticker_order)

    # When BW_TICKER_ORDER is set, preserve other tickers' index entries by
    # seeding from the existing index.json. We only overwrite entries for
    # tickers we're rebuilding this run.
    existing_index: dict[str, dict] = {}
    index_path = OUT_DIR / "index.json"
    if len(ticker_order) < len(FOCAL_TICKERS) and index_path.exists():
        try:
            prior = json.loads(index_path.read_text())
            for entry in prior.get("tickers", []):
                existing_index[entry["ticker"]] = entry
        except (json.JSONDecodeError, KeyError):
            pass

    rebuilt: dict[str, dict] = {}
    for ticker in ticker_order:
        meta = FOCAL_TICKERS[ticker]
        print(f"Building {ticker} ({meta['name']}, {meta['sector']})…", flush=True)
        bundle = build_for_ticker(ticker, meta, end_date)
        path = OUT_DIR / f"{ticker}.json"
        path.write_text(json.dumps(bundle, separators=(",", ":")))
        size_kb = path.stat().st_size / 1024
        print(f"  → {path.name}  "
              f"({len(bundle['prices'])} bars, {len(bundle['moves'])} moves, "
              f"{size_kb:.1f} KB)")
        rebuilt[ticker] = {
            "ticker": ticker,
            "name": meta["name"],
            "sector": meta["sector"],
            "moves": len(bundle["moves"]),
        }

    # Merge: rebuilt entries win over existing; preserve untouched tickers.
    merged = {**existing_index, **rebuilt}
    # Keep FOCAL_TICKERS' canonical alphabetical order in the index.
    index_out = [merged[t] for t in FOCAL_TICKERS if t in merged]

    index_path.write_text(json.dumps({
        "generated_at": end_date.isoformat(),
        "window_years": WINDOW_YEARS,
        "tickers": index_out,
    }, indent=2))
    print(f"\nDone. {len(rebuilt)} ticker(s) rebuilt; index has {len(index_out)} total.")

    # Write the eval-harness report alongside the bundles so the static UI
    # has fresh accuracy numbers without a separate manual step. We score
    # the just-baked attributions (loader=bundled), so the harness panel
    # reflects what the page is actually showing.
    try:
        from eval.accuracy import (
            bundled_attribution_loader,
            run_accuracy,
            write_report,
        )
        from eval.cases import load_cases
        cases = load_cases()
        if cases:
            report = run_accuracy(
                cases,
                attribution_loader=bundled_attribution_loader(OUT_DIR),
                prompt_version="build_static",
                loader_name="bundled",
            )
            report_path = OUT_DIR / "eval_report.json"
            write_report(report, report_path)
            print(
                f"  → {report_path.name}  "
                f"({report.primary_n_correct}/{report.primary_n_scored} "
                f"primary cases correct)"
            )
        else:
            print("  no fixtures loaded — eval_report.json skipped")
    except Exception as e:
        print(f"  WARN: eval_report.json build failed: {e}")


if __name__ == "__main__":
    main()
