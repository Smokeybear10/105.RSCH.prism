# PRISM | Earnings and Peer Returns

Tests whether earnings disclosures and the reporting company's price reaction help predict returns of other semiconductor stocks.

## Quick Start

Use Python 3.12 and [uv](https://docs.astral.sh/uv/). The [notebook](relay/notebooks/reference_replay.ipynb) has saved outputs you can read without installing anything.

```bash
git clone https://github.com/Smokeybear10/105.RSCH.prism.git
cd 105.RSCH.prism
uv sync --locked --extra research --extra test
uv run --locked --offline prism analyze --output out/prism-analysis --figure
```

This reruns the corrected analysis from included numerical inputs, without downloads. Use the installed environment as the notebook's Python kernel.

## What It Does

Compares five forecasting approaches on AMD, NVDA, INTC, QCOM, TXN and MU: zero excess return, price controls, source-price/correlation features, disclosure-language features, and a model selected using earlier fitting data. The target is a peer's five-session return minus SPY.

Events come from SEC filings.[^1] Features are measured at the first session's close after the later of the filing date and known acceptance date, using New York time. Entry is the next adjusted open. The study measures returns after filing availability.

Quarterly evaluation separates fitting, residual calibration and testing, excluding outcomes that cross either boundary. The selected model compares nine fixed candidates inside the fitting sample; calibration and test outcomes do not choose the candidate.

Results include feature comparisons, breakdowns by year and company, and checks for concurrent peer announcements. Interval scores[^2] penalize wide intervals and missed outcomes. Comparing the disclosure and network models measures what the text features add.

## Results

The study evaluates 1,050 peer forecasts from 210 announcements across 35 quarterly test blocks. Prices extend through October 30, 2026.

| Model | Original MAE | Corrected MAE |
| --- | ---: | ---: |
| Zero excess return | 3.81 pp | 3.81 pp |
| Price controls | 3.90 pp | 3.90 pp |
| Network spillover | 3.95 pp | 3.95 pp |
| Event-conditioned spillover | 4.19 pp | 3.96 pp |
| Training-selected model | — | 3.84 pp |

MAE is mean absolute error in percentage points; lower is better. Correcting disclosure extraction reduces the event model's error, but neither it nor model selection beats predicting zero excess return. The selected model's nominal 80% intervals cover 74.4% of outcomes.

![Paired forecast-error comparisons and interval coverage versus width](results/diagnostics.png)

Both columns use the same historical sample, which was also used during development. Saved [metrics](results/metrics.csv), [feature comparisons](results/ablations.csv) and [protocol](results/protocol.json) accompany the figure.

## Reproduce the Results

The quick-start command writes new outputs to `out/prism-analysis/`, preserving the saved `results/`. Tables and diagnostics run with core dependencies alone; omit `--figure` to skip plotting. Input hashes are recorded in the [corrected manifest](relay/data/research_manifest.json).

Replay the original four-model reference separately:

```bash
uv run --locked --offline prism reproduce --output out/prism-reference
```

Both offline commands cover the model stage. To download source data and rebuild disclosure features:

```bash
uv run --locked prism collect --start 2017-01-01 --end 2026-10-30
uv run --locked prism study --output data/relay/runs/new-study
```

Collection caches SEC disclosures and Yahoo adjusted prices locally. Fresh vendor downloads can differ from the included inputs. Each study records input hashes, extracted features, configuration and environment; use a new output path to preserve previous runs.

Checks run without source-data downloads:

```bash
uv run --locked --extra test pytest relay/tests -q
```

## Limits

The six-company cohort was chosen retrospectively. Peer links are assumptions, and a peer's own announcement can confound the measured response. Current adjusted-price snapshots are not historical data vintages. Phrase rules approximate financial meaning; they do not measure analyst-consensus surprises.

Residual-calibrated intervals have no coverage guarantee under market shifts. Bootstrap comparisons group overlapping holding windows but cannot remove all time dependence or account fully for decisions made after inspecting results. The model has not been tested prospectively.

A separate [customer–supplier study](relay/data/diffusion_2025_protocol.json), supported by a [source ledger](relay/data/diffusion_2025_sources.json), examines 16 scheduled releases and ten disclosed supplier links. It uses different inputs and is not a replication of the semiconductor study.

Current research lives in `relay/`. Earlier ingestion, attribution and backtest experiments remain separate; their synthetic fixtures are not the numerical reference. Earlier work includes contributions from Srilekha, Sophia and Henry.

[^1]: [SEC EDGAR APIs and filing histories](https://www.sec.gov/search-filings/edgar-application-programming-interfaces).
[^2]: Gneiting and Raftery (2007), [Strictly Proper Scoring Rules, Prediction, and Estimation](https://doi.org/10.1198/016214506000001437).

Built by Thomas Ou
