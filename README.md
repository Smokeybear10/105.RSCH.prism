# Relay | Corporate Disclosures and Peer Returns

**Original PRISM demo: [prism-004.vercel.app](https://prism-004.vercel.app/)**

I tested whether corporate announcements contain useful information about later returns in related stocks. One study forecasts semiconductor peer returns; another replays customer disclosures against previously documented supplier relationships.

Neither study establishes a trading edge. The semiconductor event model has higher forecast error than the simpler benchmarks. The customer–supplier guidance rule underperforms its matched SPY comparison in a small feasibility sample.

## Start Here

Read [the saved semiconductor study](relay/data/pilot.json) for benchmark scores, individual forecasts, source excerpts, fold audits and input/code hashes. The [local interface](#run-it) displays this file without API keys or data downloads; the hosted PRISM link above is the earlier hackathon demo, not this study.

For the customer–supplier study, start with the [source ledger](relay/data/diffusion_2025_sources.json) and [evaluation protocol](relay/data/diffusion_2025_protocol.json). They retain original-document links, disclosure dates, guidance comparisons and the rules used to select trades.

## Semiconductor Result

The saved run contains 1,050 held-out peer forecasts from 210 disclosures across 35 quarterly test blocks, from 2018 Q2 through the available portion of 2026 Q4. Collection covers 242 source disclosures and prices through October 30, 2026; early quarters without enough fitting and calibration data are skipped.

| Model | Mean absolute error | Root mean squared error | 80% interval coverage |
| --- | ---: | ---: | ---: |
| Zero excess return | 3.81 pp | 5.37 pp | 74.9% |
| Price controls | 3.90 pp | 5.49 pp | 74.2% |
| Network spillover | 3.95 pp | 5.56 pp | 73.9% |
| Event-conditioned spillover | 4.19 pp | 5.88 pp | 75.1% |

Errors are in percentage points of five-session SPY-relative return; lower is better. All four rows use the same observations. Adding source-price and disclosure-language features did not improve forecast error in this saved run, and every model's interval coverage was below the nominal 80% target.

### What It Compares

The cohort is AMD, NVDA, INTC, QCOM, TXN and MU. Every other company in the cohort is treated as a peer; SPY and SOXX supply market and industry controls. These peer links are study assumptions, not verified supplier relationships.

- **Zero excess return:** predict that the peer matches SPY.
- **Price controls:** the peer's recent returns and volatility, plus market and industry returns.
- **Network spillover:** add the source company's price reaction, momentum and historical peer correlation.
- **Event-conditioned spillover:** add fixed phrase rules for demand, pricing, competitive and guidance language in original SEC documents.

The fitted models use ridge regression. Features are observed at the close of the first trading session strictly after the SEC filing date. Entry is the following session's adjusted open; exit is the fifth holding session's close. Quarterly evaluation separates fitting, calibration and testing, purges overlapping labels at the boundaries, and fits normalization on training data only. The [model](relay/model.py) and [feature builder](relay/features.py) implement these rules.

## Customer–Supplier Check

This selected cohort contains 16 scheduled 2025 releases from Walmart, Home Depot, Lowe's and Deere, with ten previously disclosed customer links to seven suppliers. It compares buying all eligible linked suppliers, buying after a positive source-price change, and buying after a comparable positive guidance revision. No model is fitted to these 16 releases.

The local run tests all 24 combinations of three rules, two entry delays and four cost assumptions. At 20 basis points assumed roundtrip cost, the saved results are:

| Rule | Entry | Positions | Account return | Matched SPY return |
| --- | --- | ---: | ---: | ---: |
| All linked events | Next session | 19 | +4.59% | −3.10% |
| Positive source price | Next session | 14 | +4.72% | −0.06% |
| Positive guidance revision | Next session | 7 | −1.33% | +3.81% |
| All linked events | One session later | 19 | +9.07% | −1.62% |
| Positive source price | One session later | 14 | +6.98% | +1.05% |
| Positive guidance revision | One session later | 7 | +0.20% | +1.85% |

These are descriptive whole-2025 account returns from a $10,000 simulation, with at most four positions and a 25% allocation cap per new position. The SPY account follows the same entry/exit schedule under its own cash constraints. Only three releases triggered the guidance rule, yielding seven positions; the sample is too small to establish predictive power.

The table comes from the November 4, 2026 local `data/relay_diffusion/results/summary.csv`. That report and its price snapshot are ignored local outputs, not included reference results. The ledger and protocol are included; the commands below generate the full report. This feasibility study is exploratory, not an untouched preregistered test.

## Run It

Use Python 3.12. Dependencies are declared in [pyproject.toml](pyproject.toml).

```bash
git clone https://github.com/Smokeybear10/105.RSCH.relay.git
cd 105.RSCH.relay
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e '.[test]'
python -m pytest relay/tests -q
relay serve
```

Open http://localhost:2004. Use `relay serve --port 2005` if that port is occupied.

The tests use fixtures and synthetic data to check information timing, future-data isolation, forecast metrics, source-ledger validation, portfolio cash limits and API behavior. Passing them does not validate predictive power. The interface reads a locally rebuilt study when present, otherwise the included pilot.

## Reproduce the Studies

Run from the repository root after setup. For the semiconductor study:

```bash
relay collect --start 2017-01-01 --end 2026-10-30
relay study
```

This collects SEC submissions and filing exhibits plus Yahoo adjusted daily prices, then writes `study.json`, `predictions.csv`, `observations.parquet` and `study_config.json` under `data/relay/`. Compare the report's `benchmarks` and `audit` with [pilot.json](relay/data/pilot.json). `relay study --cost-bps 50` changes the cost assumption; `--publish-pilot` explicitly replaces the included pilot and is not needed to run the study.

For the customer–supplier study:

```bash
relay diffusion-prices --ledger relay/data/diffusion_2025_sources.json --output data/relay_diffusion
relay diffusion --ledger relay/data/diffusion_2025_sources.json --prices data/relay_diffusion/prices.parquet --protocol relay/data/diffusion_2025_protocol.json --output data/relay_diffusion/results
```

Outputs are `report.json`, `summary.csv`, `event_windows.csv` and `trades.csv`; the report includes daily account paths, every rule/cost combination, source audits and hashes. Reusing the same output path replaces its prior results.

Both collectors cache source responses and accept `--refresh`. Exact replay requires retaining the raw response cache, price files and manifests: a hash identifies an input but cannot restore it. Fresh downloads use current adjusted Yahoo history and may differ from the saved runs. The included semiconductor pilot records a different collector hash from the current `ingest.py`. Dependencies have version ranges rather than a locked reproduction environment. These are saved-run results, not a fresh full-data reproduction.

## Experimental Limits

The semiconductor universe was selected retrospectively. Concurrent peer announcements remain in the sample, and the fixed text rules describe matched language rather than verified causes. Residual-based intervals have no coverage guarantee under market shifts; date-cluster bootstrap comparisons retain some serial dependence. Its hypothetical trade returns overlap and do not form a capital-constrained portfolio.

The customer–supplier study uses management-guidance revisions, not analyst-consensus surprises. Missing or incomparable guidance causes the rule to abstain. A date-only next-open source-price signal can precede an after-close release; the delayed rule changes both its information and entry time, so it is not a pure execution-delay test.

Both studies use adjusted daily price proxies. Opening-price sizing and fills, spreads, market impact and dividend payment timing need separate execution validation. Neither result establishes causal information diffusion or a deployable strategy.

## Code and Data

The current studies live in [`relay/`](relay/). The source ledger links to the original disclosures; the saved pilot retains SEC evidence links. Raw caches and complete local outputs live under ignored `data/relay/` and `data/relay_diffusion/` directories.

The original PRISM hackathon code remains in `demo/`, `ingestion/`, `model/`, `backtest/` and `eval/`. Its synthetic fixtures and attribution backtest are separate from these studies. Earlier ingestion and attribution work includes contributions from Srilekha, Sophia and Henry.

Built by Thomas Ou
