from __future__ import annotations

import argparse
from pathlib import Path

from relay.config import DATA_DIR, StudyConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Earnings Disclosure Spillovers and Peer-Return Prediction: collect disclosures, compare forecasts, and review research results.")
    sub = parser.add_subparsers(dest="command", required=True)
    collect = sub.add_parser("collect", help="Download public SEC disclosures and adjusted daily prices")
    collect.add_argument("--start", default="2017-01-01")
    collect.add_argument("--end", default=None, help="Last date included (YYYY-MM-DD); use a completed trading day")
    collect.add_argument("--refresh", action="store_true", help="Refresh cached responses")
    study = sub.add_parser("study", help="Fit chronological models using the local data cache")
    study.add_argument("--cost-bps", type=float, default=None, help="Assumed total roundtrip cost")
    study.add_argument("--output", type=Path, help="Run directory (default: data/relay/runs/latest); use a new path to retain each experiment")
    reproduce = sub.add_parser("reproduce", help="Verify and replay the bundled numerical reference without downloads")
    reproduce.add_argument("--output", type=Path, default=Path("out/relay-reference"), help="Separate output directory (default: out/relay-reference)")
    analyze = sub.add_parser("analyze", help="Replay the corrected study and export research diagnostics without downloads")
    analyze.add_argument("--output", type=Path, default=Path("out/earnings-analysis"))
    analyze.add_argument("--figure", action="store_true", help="Save scientific figures (requires the research extra)")
    diffusion = sub.add_parser("diffusion", help="Replay a source-audited customer-supplier event study")
    diffusion.add_argument("--ledger", type=Path, required=True)
    diffusion.add_argument("--prices", type=Path, required=True, help="Cached adjusted daily bars in Parquet format")
    diffusion.add_argument("--protocol", type=Path, required=True, help="Frozen research protocol JSON")
    diffusion.add_argument("--output", type=Path, required=True)
    diffusion_prices = sub.add_parser("diffusion-prices", help="Cache daily prices for a customer-supplier source ledger")
    diffusion_prices.add_argument("--ledger", type=Path, required=True)
    diffusion_prices.add_argument("--output", type=Path, required=True)
    diffusion_prices.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    if args.command == "collect":
        from relay.ingest import collect_data
        options = {"start": args.start}
        if args.end:
            options["end"] = args.end
        config = StudyConfig.model_validate(options)
        prices, events, warnings = collect_data(config, refresh=args.refresh)
        print(f"Saved {len(prices):,} daily prices and {len(events):,} disclosures. {len(warnings)} text-fetch warnings.")
    elif args.command == "study":
        from relay.study import load_study_config, run_cached_study
        config = None
        output = args.output if args.output is not None else DATA_DIR / "runs" / "latest"
        try:
            if args.cost_bps is not None:
                options = load_study_config(DATA_DIR, output).model_dump()
                options["cost_bps"] = args.cost_bps
                config = StudyConfig.model_validate(options)
            report = run_cached_study(directory=DATA_DIR, config=config, output_directory=output)
        except (FileNotFoundError, ValueError) as exc:
            parser.exit(1, f"Study failed: {exc}\nCollect the source inputs with `earnings-study collect` or replay the included reference with `earnings-study reproduce`.\n")
        print(f"{report['study']['status']}: {report['summary']['events']} events, {report['summary']['predictions']} held-out peer forecasts")
        for metric in report["benchmarks"]:
            print(f"  {metric['name']}: MAE {metric['mae']:.4%} | RMSE {metric['rmse']:.4%}")
        print(f"Saved research outputs to {output.resolve()}")
    elif args.command == "reproduce":
        from relay.reproduce import reproduce
        try:
            report = reproduce(output=args.output)
        except (ValueError, FileNotFoundError) as exc:
            parser.exit(1, f"Reference replay failed: {exc}\n")
        print(f"Verified model-stage replay: {report['reference']['predictions']:,} forecasts; no downloads.")
        for metric in report["metrics"]:
            print(f"  {metric['name']}: MAE {metric['mae']:.4%} | RMSE {metric['rmse']:.4%}")
        print(f"Saved separate results to {args.output.resolve()}")
    elif args.command == "analyze":
        from relay.research import analyze
        if args.figure:
            from importlib.util import find_spec
            if find_spec("matplotlib") is None:
                parser.exit(1, "Figures require the research extra: uv sync --locked --extra research\n")
        try:
            report = analyze(output=args.output)
        except (ValueError, FileNotFoundError) as exc:
            parser.exit(1, f"Research analysis failed: {exc}\n")
        if args.figure:
            from relay.figures import plot_diagnostics
            plot_diagnostics(report, args.output)
        print(f"Verified corrected model-stage replay: {report['fixture']['predictions']:,} forecasts; no downloads.")
        for metric in report["metrics"]:
            print(f"  {metric['name']}: MAE {100 * metric['mae']:.4f} pp | 80% coverage {metric['interval_coverage']:.1%}")
        print(f"Saved diagnostics and protocol to {args.output.resolve()}")
    elif args.command == "diffusion-prices":
        from relay.diffusion import collect_diffusion_prices
        manifest = collect_diffusion_prices(args.ledger, args.output, refresh=args.refresh)
        print(f"Cached {manifest['rows']} daily bars for {len(manifest['tickers'])} tickers; "
              f"verified {manifest['study_sessions']} study sessions per ticker")
    elif args.command == "diffusion":
        from relay.diffusion import run_diffusion_study
        report = run_diffusion_study(args.ledger, args.prices, args.protocol, args.output)
        print(f"{report['status']}: {report['audit']['scheduled_events']} source events, "
              f"{report['audit']['linked_event_rows']} linked rows, {len(report['runs'])} fixed configurations")
        print(f"Saved report and event windows to {args.output.resolve()}")


if __name__ == "__main__":
    main()
