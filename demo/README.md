# Historical Research Helpers

Python utilities and frozen fixtures from the original attribution study. The web interface and server have been removed; the `demo` module name remains so existing research imports and tests keep working.

- `analyze_ticker.py`, `prep_ticker.py`, and `real_chunks.py` assemble attribution inputs.
- `ablation_chart.py`, `pnl_summary.py`, and `strategy_report.py` produce research summaries.
- `build_static.py` exports the historical per-ticker JSON format used by the legacy evaluation code.
- `mock_data.py` and `static/data/` include synthetic demonstration fixtures. They are separate from the current semiconductor study and do not establish forecasting performance.

The current reproducible research package is [`relay/`](../relay/). See the [project README](../README.md) for setup and evaluation commands.
