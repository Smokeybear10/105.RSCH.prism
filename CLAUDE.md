# Earnings Disclosure Spillovers and Peer-Return Prediction

The current project is research-only. Use `README.md` and the Python package in
`relay/`; the frontend, web servers and deployment configuration have been removed.
Run `uv run --locked --offline earnings-study reproduce` for the included reference and
`uv run --locked --extra test pytest relay/tests -q` for the current research suite.
Run `uv run --locked --offline earnings-study analyze` for the corrected numerical study,
paired comparisons and calibration diagnostics. Add `--figure` with the `research`
extra installed. Fresh analyses go to `out/`; `results/` contains saved evidence.
Use `earnings-study study --output PATH` for separate experiments. Preserve the original
reference inputs and distinguish inspected historical results from future evaluation.

The remaining sections record the earlier hackathon's ownership and architecture.
Its demo plans and status claims are historical, not current implementation tasks.

## Archived hackathon context: Equity Price Action Tagger — Lean or Fade?

## Project goal (read this first)
We're building a **financial historian**: given a significant stock price move, the system
ingests the text around that move (earnings transcripts, SEC filings, news, institutional
positioning, macro context) and produces a structured attribution across five dimensions
— demand, pricing, competitive dynamics, management credibility, macro exposure — with
evidence citations. Then we decide: does the market have this right (**lean**) or is it
overreacting to transient signals (**fade**)?

This is not sentiment analysis. We are decomposing qualitative language into measurable
dimensions that evolve over time, and testing whether that decomposition predicts
whether a move persists (structural) or reverts (transient).

## The deliverable
A **clickable demo of the pipeline's reasoning**, plus ablation results and a modest
backtest. Not a trading product. The demo sells the research process: "here's what the
model thinks with only news; now add 10-Ks; now add peer companies; now add macro —
watch the attribution and the predicted return converge to what actually happened."

## Current state (post-integration merge)

End-to-end pipeline is wired. Each step has a working implementation; remaining work
is tuning, ablations, and the clickable demo.

| Step | Module | Status |
|---|---|---|
| 1. Flag significant price moves | `ingestion/prices/` | **working** — 7-script pipeline builds `events_focal.parquet`; `detect_significant_moves` exposed |
| 2. Ingest text | `ingestion/sec/` (10-K + 8-K), `ingestion/earnings_news/`, `ingestion/idiosyncratic/` (13F, short-interest, index changes, short reports, FDA, analyst actions) | **working** — fixtures + live fetchers per source |
| 2b. Join per-move evidence | `ingestion/events/` | **working** — 8 adapters + aggregator + trading-day window joiner emitting `JoinedEvidence` |
| 3. Attribute moves to text | `model/attribution/` (`run`, `prompt`, `validate`) | **working** — Anthropic `tool_use`-structured call with fake-client test path |
| 4. Add macro / peer / sector drivers | ablation roadmap below | partial — macro / peer stubs present |
| 5. Coherence check | `model/attribution/coherence` | **working** — `CoherenceCheck` with fake-client test path |
| 6. Fade-or-follow + backtest | `backtest/` (`signal`, `pnl`, `runner`, `baselines`) | **working** — fade/lean signal + 4 baselines + ablation runner |
| Demo | `demo/` (`app`, `analyze_ticker`, `ablation_chart`, `mock_data`) | **in progress** — orchestrator CLI and mock chart in place |

## Three strategic messages from Mentor Meeting #1
1. **Sprint to a crappy end-to-end MVP before polishing anything.** Don't perfect Step 1
   for 5 hours. Get one ticker end-to-end first, then iterate.
2. **Demo the thought process and ablations, not just the final result.** Each data
   source we add is a talking point. Feature-importance across sources is the money shot.
3. **Put energy into data-source additive testing + interpretability, not a sophisticated
   trading strategy.** Lean/fade logic should be sensible, not clever. The tagging and
   the evaluation framework are the product.

---

## Team ownership & playbooks

Each person owns a data stream. Everyone must publish a typed artifact other modules
can hang off of, and everyone must write at least one fixture so downstream teammates
aren't blocked.

### Srilekha — Yahoo Finance (the backbone everyone joins onto)
**Dataset:** `defeatbeta/yahoo-finance-data`. Prices + earnings transcripts + fundamentals.

**Core deliverable:**
- `events.parquet` — one row per significant earnings event. Columns: `ticker`,
  `earnings_date`, `bmo_amc_flag`, `reaction_return`, `fwd_return_1d`, `fwd_return_5d`,
  `fwd_return_20d`, `market_neutralized_fwd_returns`, `pre_event_vol`, `is_significant`.
- `transcripts.parquet` — earnings transcripts keyed by `(ticker, earnings_date)`.

**Handoff:** these two tables are the anchor everyone else joins onto. Target: usable
event table by **hour 6**. Everyone else is blocked without it.

**Key risk:** BMO (before-market-open) vs. AMC (after-market-close) flag. This single
column decides whether the reaction window is day T or day T+1 for every event. Get it
wrong and every downstream result is silently corrupted. **Spot-check 5–10 famous events
(Meta Feb 2022, Nvidia May 2023, AAPL March 2020) against known outcomes** before calling
the table clean.

**Slack to pick up:** after hour 8 Srilekha is the natural backtest owner — the price
data is already in her head and the backtest is a price-math problem.

### Sophia — SEC filings (structural priors + change detection)
**Dataset:** `JanosAudran/financial-reports-sec` (pre-parsed, preferred) with `PatronusAI/financebench` as eval sanity check. Raw EDGAR via `edgartools` as fallback.

**Core deliverable:** a per-ticker **structural profile** extracted from the most recent
10-K before each earnings event, covering: stated competitive dynamics, risk factors,
guidance language, segment breakdown, macro exposures (FX, commodities, rates). Output:
`filing_profiles.parquet`, joined to events by `(ticker, most_recent_10k_before_event)`.

**Sections to extract, ignore the rest:** Item 1 (Business), Item 1A (Risk Factors),
Item 7 (MD&A). Skip financial statements — those are numeric and Srilekha has them.

**Stretch (the differentiator):** change detection. Compare current earnings-call language
to the most recent 10-K profile. Has the competitive narrative shifted? Has risk language
intensified? The *delta* is the signal no one else in the hackathon will have.

**Key risk:** getting lost in 10-K parsing. A single 10-K is 100+ pages of boilerplate.
**Timebox section extraction to 2 hours**; move on with whatever you have. Use header
regex, not LLM parsing, to find section boundaries.

**Slack to pick up:** Sophia is the natural owner of the **attribution schema** — her
filing profiles and the earnings-call tags must share vocabulary for comparison to work.

**Fallback if underwater:** drop to 8-Ks only, or skip filings entirely. A version of
the project with transcripts + news and no 10-Ks is still credible.

### Henry — 13F statements + open research (smart-money positioning)
**Datasets:** SEC 13F directly, WhaleWisdom if accessible, or pre-aggregated HF dataset
if available. Plus a curated bibliography of prior work on PEAD / event studies.

**Core deliverable:** a per-event **institutional positioning snapshot** — total
institutional ownership %, change in ownership over the last quarter, concentration
(Herfindahl index or top-5-holder share), and holder-type mix (long-only mutual funds
vs. hedge funds vs. passives). Output: `positioning.parquet`, joined to events by
`(ticker, last_13f_date_before_event)`.

**Framing (non-negotiable):** 13Fs are a **pre-event positioning signal**, not live
attribution. High institutional ownership + beat → squeeze potential (lean harder).
Concentrated institutional ownership + miss → forced-selling risk (fade harder). This
gives 13F a real role in the lean/fade logic rather than "context only."

**Research brief deliverable:** 3–5 papers/posts on PEAD, narrative-driven attribution,
and event-study methodology. Shared in team doc. Grounds our framing so we don't
reinvent wheels.

**Key risk:** 13F ingestion is the most failure-prone workstream. 13Fs are publicly filed
but messy to aggregate. **If hour 4 and no clean 13F data, pivot to analyst consensus**
(EPS estimates, price targets, ratings distribution from Yahoo). Better one working
positioning signal than two half-broken ones.

**Slack to pick up:** integration + backtest partner with Srilekha after hour 8. Also
natural **demo owner** — research bibliography primes him to own the narrative framing.

### Thomas — News (immediate narrative) [me]
**Core deliverable:** per-event bundle of news articles with source, timestamp, and text,
joined to events via ticker + timestamp match. For each event, the LLM produces a
news-based attribution tag bundle. Output: `news_chunks.parquet` and news-attribution runs.

**The data-access problem (active blocker):** WSJ / CNBC / Bloomberg are paywalled and
not scrape-friendly for this hackathon. Prioritized options:
1. **Yahoo's news feed via yfinance / `defeatbeta/yahoo-finance-data`** — coordinate
   with Srilekha, she may already have this. Zero cost.
2. **Finnhub free tier** — company news endpoint, major outlets, rate-limited but
   usable for a hackathon.
3. **Polygon.io Stocks Starter at $29/mo** — WSJ/CNBC/Bloomberg headlines + snippets,
   ticker-tagged, 5y history. Best paid option.
4. **SEC 8-K filings as near-news** — ticker-tagged, timestamped to the minute, legally
   required to be material, free via `edgartools`. Great proxy for non-earnings
   material events. Use 8-K Item codes (2.02 earnings, 5.02 exec changes, 7.01 Reg FD,
   8.01 other material) to filter.
5. **Abandon news as a standalone workstream** — pivot to reinforce 8-Ks (crosses over
   with Sophia) or the attribution LLM layer.

**If hour 2 and no workable source is chosen, pivot. Don't burn a day scraping.**

**Foreknowledge discipline (strictest for news):** for an AMC earnings release on day T,
only articles published T-1 through T at 4:00 PM ET are fair game. No "why XYZ fell 10%
after hours" articles from T+1. No articles whose text references the move itself.
Filter aggressively at ingestion.

**Slack to pick up:** Thomas is the natural owner of the **attribution LLM layer** —
news is its primary input, and pairing news with Sophia's structural profile is the
cross-source reconciliation the demo needs.

**Fallback if entirely stuck on news:** go all-in on 8-Ks via `edgartools`. They give
you attributable, timestamped material-event text without any ToS or scraping issues.

---

## Cross-cutting ownership gaps — RESOLVED
What was open at Mentor Meeting #1 is now filled in:

1. **Attribution LLM layer + schema design.** Landed in `model/attribution/`
   (`run.py`, `prompt.py`, `validate.py`, `coherence.py`). Anthropic `tool_use`-
   structured call with a fake-client test path so the suite runs without an API key.
2. **Backtest + signal construction.** Landed in `backtest/` with `signal`,
   `pnl`, `runner`, and four `baselines` (naive-lean, always-fade, random,
   sentiment-only) per mentor ask.
3. **Event-joining + evidence bundling.** Landed in `ingestion/events/` — 8
   per-source adapters feed a single unified events parquet; `join_evidence`
   pulls the trading-day window around a `PriceMove` into a `JoinedEvidence`
   the attribution runner consumes.
4. **Demo dashboard + orchestrator.** In progress in `demo/` — `analyze_ticker`
   CLI wires prices → moves → events → attribution → JSON payload; `app.py`
   and `ablation_chart.py` render the clickable demo.

---

## Repo layout & module boundaries

```
ingestion/prices/          Srilekha: 7-script pipeline → events_focal.parquet
                           (build_price_panel, build_earnings_events,
                           build_earnings_reactions, build_events_table,
                           build_focal_universe, sanity_checks, audit_pitfalls)
ingestion/sec/             Sophia: 10-K (tenk.py) + 8-K (eightk.py) ingestion
ingestion/earnings_news/   Thomas: news scraping + transcripts
ingestion/idiosyncratic/   Henry: 13F, FINRA short interest, index changes,
                           short-seller reports, FDA events, analyst actions
ingestion/events/          Henry: record → Event adapters (8 sources),
                           aggregator that unions into data/cache/events.parquet,
                           trading-day-window joiner emitting JoinedEvidence
ingestion/macro/           macro stub (FOMC, VIX, commodities) — not live yet
model/                     Shared
model/attribution/         LLM attribution runner + validator + coherence check
                           (run.py, prompt.py, validate.py, coherence.py)
backtest/                  Shared: signal, pnl, runner, baselines, basket, fixtures
demo/                      Shared: app, analyze_ticker CLI, ablation_chart,
                           mock_data
scripts/                   smoke_ingestion.py — end-to-end integration script
schema.py                  Shared contracts — DO NOT modify without team sign-off
tests/fixtures/            Sample data every module tests against
tests/                     pytest suite (~380+ tests, green)
```

**Claude Code: when editing inside one module, do NOT touch other modules.** Use
`/freeze` to lock scope to the current directory when debugging.

---

## Attribution schema

Five fixed dimensions (scored per move, evidence-cited):
- `demand` — unit volume, customer count, market share shifts
- `pricing` — price changes, mix, discounting
- `competitive` — new entrants, competitor moves, moats
- `management_credibility` — guidance changes, execution, leadership comments
- `macro` — rates, FX, commodities, geopolitics

Every `DimensionScore` MUST cite at least one `evidence_chunk_id`. Drop any score without
a valid citation — this is the anti-hallucination guardrail.

`move_character` is the single bit that drives the trade: `structural` → lean,
`transient` → fade, `mixed`/`unclear` → neutral.

See `schema.py` for the full Pydantic contracts. **Don't modify `schema.py` without
posting in team chat first.**

---

## MVP scope (hour 0 → next mentor check-in)
- **Universe:** ONE ticker first. Pick one where the team has personal intuition about
  at least one past move — mentor's point is that spot-checking requires knowing what
  the right attribution "feels like." Expand to ~5 after MVP works.
- **History:** 2–5 years. Not 20. Long history burns compute without adding MVP value.
- **Sources for MVP:** earnings transcripts + 10-Ks + (whatever news Thomas can land).
  Add 13F, macro, peer news, sector as ablations **after** MVP is end-to-end.
- **Cache everything to disk** (`*/.cache/`, gitignored). Mentor was explicit:
  **"make loops as fast as possible, store locally so we all can access it easily,
  make sure it's fast to run."** Re-running the pipeline must not re-hit APIs.

---

## Ablation roadmap (the demo goldmine)
Each run is a bar on the demo chart showing hit-rate / plausibility / Sharpe as we add
sources. This is the additive-testing story mentor called "demo gold."

1. `base_news` — company-specific news only.
2. `+sec` — add 10-K / 10-Q language.
3. `+earnings` — add earnings-call transcripts.
4. `+peer_news` — add news about peer / "family" tickers (Apple → Samsung, TSMC, Qualcomm).
5. `+sector_news` — sector-wide stories.
6. `+macro` — Fed decisions, VIX, commodities, geopolitics.
7. `+positioning` — Henry's 13F / consensus features.

The demo one-liner we're engineering toward: *"10-K language is the biggest attribution
driver; peer news adds 15% more signal; WSJ sentiment was basically noise; adding
institutional positioning flips the sign on crowded misses."*

---

## Evaluation framework (mentor emphasis)

**Core question per event:** what *actually* happened vs. what we *expected* to happen?

The `Attribution` object carries both `return_pct` (realized) and `predicted_return_pct`
(what the model expected given the evidence). The gap between them is the signal:
- Small gap, coherent evidence → model understands this move → trust it.
- Big gap, coherent evidence → market mispriced the structural content → potentially
  trade-worthy.
- Big gap, incoherent evidence → model is confused → reject via coherence check.

**Build a fast evaluation harness.** Mentor: *"have a tool that helps you evaluate
different frameworks fast — would help us stand out."* Concretely:
- One command runs attribution + signal + backtest on the full event table.
- One command regenerates the ablation chart.
- Each module re-runs in seconds against cached data (never re-fetch from disk-cached runs).
- Every attribution row is clickable in the demo to show which chunks it cited.

**Benchmarks / baselines we MUST report** (mentor: "find proxy for benchmark"):
1. **Naive "change in price = news" baseline** — always lean with the move.
2. **Always fade big moves** — pure mean reversion.
3. **Random attribution** — same signal logic on shuffled attribution vectors.
4. **Sentiment-only** — single scalar (positive/negative) classifier on the same text.

Our structured-attribution signal must beat all four to claim it's doing real work.

**Feature importance across data sources.** Within the news bracket, test additivity of
each source: what if we drop WSJ-equivalent headlines? What if we only use press releases?
This is the feature-importance story inside the ablation story.

---

## Frozen test case (pick once, don't swap)
Once the team picks the MVP ticker, pick ONE (ticker, date) pair where the team knows
what caused the move — e.g. a COVID-era drop for a consumer name, a rate-hike day for
a bank. That pair becomes the prompt-iteration target.

Rules (mentor, explicit):
- **Freeze the inputs.** Don't swap the test case mid-iteration — you lose the ability
  to isolate what a prompt change caused.
- **Write the expected output BEFORE running the model** (dominant dimension, direction,
  plausible `move_character`). A regression is only obvious if you can name it ahead
  of time.
- Store the expected-output contract as `tests/fixtures/<ticker>_<event>_expected.json`.
  A test should assert it parses and contains the expected-attribution keys.

---

## Definitions to lock at hour 1 (pick ONE and stick with it)
- **Significant price move**: `|1-day return| > 2x trailing 30-day realized vol`,
  OR top 5% absolute return in trailing 60 days. Pick one.
- **Fade window**: move reversed by >50% within 5 trading days.
- **Persist**: no reversal, or extension of the move, over 5 trading days.
- **As-of date**: `publication_date <= as_of` for every retrieval function.

---

## Non-negotiable rules

1. **No foreknowledge leak.** Every retrieval function takes an `as_of` parameter and
   MUST filter by `publication_date <= as_of`. This is THE most common way hackathon
   projects silently corrupt their backtest. Mentor also flagged **model foreknowledge**
   (LLM trained on post-event journalism about Meta Feb 2022, NVDA May 2023). We can't
   fully eliminate it; we can state it as a limitation and not rabbit-hole on it.
2. **Filing date != period end.** A 10-K for fiscal year ending Dec 31 might be filed
   in March. The market reacts on filing date. Store both; as-of queries use
   `publication_date`.
3. **Stable chunk IDs.** Format locked: `{source_type}_{ticker}_{YYYY-MM-DD}_{section}_{NNN}`.
   Citations reference these.
4. **Fixtures before code.** Write a fixture matching the schema before implementing a
   fetcher. Downstream modules build against fixtures — nobody blocks on live data.
5. **One schema, enforced.** Use the Pydantic models in `schema.py`. No loose dicts
   crossing module boundaries. **Post in team chat BEFORE editing `schema.py`.**
6. **Every DimensionScore cites at least one real chunk_id.** No citation → drop the score.
7. **Cache everything to disk.** No re-fetching across runs.

---

## Known traps
- **Yahoo Finance survivorship bias** — delisted companies disappear. Note in demo.
- **SEC filings are huge** — chunk aggressively, keep MD&A + Risk Factors only.
- **LLM will hallucinate attributions** — enforce evidence_chunk_ids, drop uncited scores.
- **Paywalled news** (WSJ, CNBC, Bloomberg) — see Thomas's playbook. Don't scrape.
- **BMO vs. AMC flag errors** — see Srilekha's playbook. Single wrong flag corrupts
  everything downstream.
- **13F lag + quarterly cadence** — can't drive real-time attribution. Use as pre-event
  positioning signal only.
- **Re-fetching APIs on every run** — will eat the day. Cache.
- **Schema drift between modules** — people editing `schema.py` without sign-off.
- **Demo ready only at hour 23** — start building the dashboard at hour 8, not hour 20.

---

## Workflow
- Each person works on their own git branch:
  - `person1-sec` / `sophia-eval-harness` — Sophia
  - `srilekha-yahoofinance` — Srilekha
  - `henry-idiosyncratic` — Henry
  - `thomas-test` — Thomas
  - `tagging` — shared integration branch for the event-joining layer + the
    Steps-3-through-6 wiring. Per-person work that's review-ready merges into
    `tagging` first so the integration surface has somewhere to live before
    hitting `main`.
- Build against `tests/fixtures/` while your real data source is in progress.
- `pytest tests/` must pass before any merge.
- Merge to `main` only after a teammate reviews the PR.
- **If you touch `schema.py`, post in team chat BEFORE merging.**

---

## Claude Code house rules
- When editing inside one module, **do NOT touch other modules**. If cross-module
  changes are needed, flag them and let the owning person do it.
- Use `/freeze` to lock scope to the current directory when debugging.
- Respect the "one company, short history" MVP scope — don't quietly broaden it.
- **Don't refetch cached data.** If the cache dir has it, read from there.
- **Every attribution must cite a real chunk_id.** Reject uncited output silently.

---

## Mentor Meeting #1 — key takeaways (reference)

**How we selected the project:** independent ranking → top 3 → discussed MVP feasibility,
iteration speed, ease of parallel work, post-MVP expansion. Winner: Track 1.

**Our framing:** "financial historian" — ingest equities + macro, output structured
reasoning + evidence for why a move happened. Decompose language into measurable
dimensions (not sentiment).

**Mentor's concrete advice:**
- Find a proxy benchmark; report baselines (naive, random, sentiment-only).
- Tech sector is a reasonable starting universe; consider niche if data coverage is thin.
- Make iteration loops as fast as possible; cache locally.
- Build a fast evaluation tool — "would help us stand out lowkey."
- Frame each run as a story: "here's news only → here's +SEC → here's +peer" — show
  additivity explicitly. Each added factor is a demo bar.
- Feature importance matters *within* sources too (which news outlets matter, which
  filing sections matter).
- Broaden scope past the target company: "Apple's family companies" (suppliers,
  partners, competitors). This is the `+peer_news` ablation.
- Midpoint evaluation: is the model making sense? Is it realistic? Evaluate on the frozen test case.
- Walk through: input (ticker) → Step 1 (significant moves) → Step 2 (news tagging) →
  connect tags to volatility → demo the matched intuition.
- Strategy must stand up to defense. Address what works AND what doesn't — include a
  failure case in the demo.

**Our open questions to revisit next check-in:**
- How structured should the attribution output be (fixed schema vs. flexible)? — decided:
  fixed 5-dim schema.
- Is the contribution the tagging system or the trading signal? — both, but tagging
  is the research result; signal is the stress test.
- Subtle foreknowledge leaks? — model pretraining on financial journalism. Acknowledge,
  don't rabbit-hole.
- How many tickers × how much history? — MVP: 1 ticker × 2–5 years. Post-MVP: 5 tickers.
- What to cut if time-constrained? — Sophia's 10-Ks are most compressible; Thomas's
  news pivots to 8-Ks; Henry's 13Fs pivot to analyst consensus.

---

## Data access (private HF dataset)

In addition to `defeatbeta/yahoo-finance-data` (public) and `yfinance`, we have
access to a private HF dataset with curated pre-packaged parquets:

- **Repo:** `BridgewaterAIHackathon/BW-AI-Hackathon` (private)
- **Auth:** `huggingface-cli login` once, then pass `token=True` to `load_dataset`
  (or use `HfFileSystem`, which picks up the cached token automatically)
- **Layout:** files sit directly under each source folder — e.g.
  `Structured_Data/SNE/yahoo-finance-data/*.parquet`. **No `data/` subfolder**
  (unlike the public `defeatbeta` mirror).
- **Full schema reference:** see [`docs/hf_schemas.md`](docs/hf_schemas.md).
  Includes column types, row counts, and global gotchas (`symbol` vs `ticker`,
  `decimal128` casting, `report_date` parsing, no `adj_close`, etc.).
- **Efficient schema probe** (reads only the parquet footer, not the data):
  ```python
  from huggingface_hub import HfFileSystem
  import pyarrow.parquet as pq
  with HfFileSystem().open("datasets/<repo>/<path>.parquet", "rb") as f:
      print(pq.read_metadata(f).schema.to_arrow_schema())
  ```
  Do not use `load_dataset` or `df.head()` just to inspect columns — those
  download the full file (435 MB for `stock_prices`).
