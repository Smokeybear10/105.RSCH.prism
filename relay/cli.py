from __future__ import annotations

import argparse
import json
from pathlib import Path

from relay.config import DATA_DIR, StudyConfig


def main() -> None:
    parser = argparse.ArgumentParser(prog="relay", description="Collect disclosures, build forecasts, and inspect honest out-of-sample results.")
    sub = parser.add_subparsers(dest="command", required=True)
    collect = sub.add_parser("collect", help="Download public SEC disclosures and adjusted daily prices")
    collect.add_argument("--start", default="2017-01-01")
    collect.add_argument("--end", default=None, help="Last date included (YYYY-MM-DD); use a completed trading day")
    collect.add_argument("--refresh", action="store_true", help="Refresh cached responses")
    study = sub.add_parser("study", help="Fit chronological models using the local data cache")
    study.add_argument("--cost-bps", type=float, default=None, help="Assumed total roundtrip cost")
    study.add_argument("--publish-pilot", action="store_true", help="Include the derived study in the app package")
    diffusion = sub.add_parser("diffusion", help="Replay a source-audited customer-supplier event study")
    diffusion.add_argument("--ledger", type=Path, required=True)
    diffusion.add_argument("--prices", type=Path, required=True, help="Cached adjusted daily bars in Parquet format")
    diffusion.add_argument("--protocol", type=Path, required=True, help="Frozen research protocol JSON")
    diffusion.add_argument("--output", type=Path, required=True)
    diffusion_prices = sub.add_parser("diffusion-prices", help="Cache daily prices for a customer-supplier source ledger")
    diffusion_prices.add_argument("--ledger", type=Path, required=True)
    diffusion_prices.add_argument("--output", type=Path, required=True)
    diffusion_prices.add_argument("--refresh", action="store_true")
    serve = sub.add_parser("serve", help="Open the local research interface")
    serve.add_argument("--port", type=int, default=2004)
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
        from relay.study import run_cached_study
        config = None
        if args.cost_bps is not None:
            config_path = DATA_DIR / "study_config.json"
            if not config_path.exists():
                config_path = DATA_DIR / "config.json"
            options = json.loads(config_path.read_text())
            options["cost_bps"] = args.cost_bps
            config = StudyConfig.model_validate(options)
        report = run_cached_study(config=config, publish_pilot=args.publish_pilot)
        print(f"{report['study']['status']}: {report['summary']['events']} events, {report['summary']['predictions']} held-out peer forecasts")
        for metric in report["benchmarks"]:
            print(f"  {metric['name']}: MAE {metric['mae']:.4%} | RMSE {metric['rmse']:.4%}")
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
    elif args.command == "serve":
        import uvicorn
        uvicorn.run("relay.server:app", host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
