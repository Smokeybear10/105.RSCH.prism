from __future__ import annotations

from math import isfinite

import pandas as pd


MAX_POSITIONS = 4
MAX_POSITION_FRACTION = 0.25


def _candidate_date(value: str) -> pd.Timestamp:
    try:
        day = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid candidate date: {value}") from exc
    if pd.isna(day) or not isinstance(value, str) or day.date().isoformat() != value:
        raise ValueError("Candidate dates must be ISO calendar dates")
    return day


def _merge_candidates(candidates: list[dict]) -> list[dict]:
    merged = {}
    for candidate in candidates:
        if not {"target_ticker", "entry_date", "exit_date", "event_ids"}.issubset(candidate):
            raise ValueError("Candidates require target_ticker, entry_date, exit_date and event_ids")
        ticker = candidate["target_ticker"]
        event_ids = candidate["event_ids"]
        if not isinstance(ticker, str) or not ticker.strip():
            raise ValueError("Candidate target_ticker must be nonempty")
        if not isinstance(event_ids, list) or not event_ids or any(not isinstance(item, str) or not item for item in event_ids):
            raise ValueError("Candidate event_ids must be a nonempty list of strings")
        entry = _candidate_date(candidate["entry_date"])
        exit_date = _candidate_date(candidate["exit_date"])
        if exit_date < entry:
            raise ValueError("Candidate exit_date must not precede entry_date")
        key = (entry, ticker)
        if key in merged and merged[key]["exit_date"] != exit_date:
            raise ValueError("Same-date candidates for one target must have the same exit_date")
        if key not in merged:
            merged[key] = {"target_ticker": ticker, "entry_date": entry, "exit_date": exit_date, "event_ids": set()}
        merged[key]["event_ids"].update(event_ids)
    return [{**candidate, "event_ids": sorted(candidate["event_ids"])}
            for _, candidate in sorted(merged.items())]


def replay_portfolio(
    candidates: list[dict],
    prices: pd.DataFrame,
    initial_cash: float = 10000,
    cost_bps: float = 20,
    benchmark_ticker: str = "SPY",
) -> dict:
    if isinstance(initial_cash, bool) or not isfinite(initial_cash) or initial_cash <= 0:
        raise ValueError("initial_cash must be finite and positive")
    if isinstance(cost_bps, bool) or not isfinite(cost_bps) or not 0 <= cost_bps < 10000:
        raise ValueError("cost_bps must be finite, nonnegative and below 10000")
    required = {"ticker", "date", "adj_open", "adj_close"}
    if not required.issubset(prices.columns):
        raise ValueError(f"Prices require {sorted(required)}")
    orders = _merge_candidates(candidates)
    data = prices.loc[:, sorted(required)].copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce", utc=True).dt.tz_convert(None).dt.normalize()
    if data["date"].isna().any() or data["ticker"].isna().any():
        raise ValueError("Prices contain invalid dates or tickers")
    if data.duplicated(["ticker", "date"]).any():
        raise ValueError("Duplicate ticker/date prices")
    for name in ("adj_open", "adj_close"):
        data[name] = pd.to_numeric(data[name], errors="coerce")
    calendar = pd.DatetimeIndex(sorted(data.loc[data["ticker"].eq(benchmark_ticker), "date"].unique()))
    if len(data) and calendar.empty:
        raise ValueError(f"Prices require benchmark sessions for {benchmark_ticker}")
    if orders and calendar.empty:
        raise ValueError("Candidates require price history")
    if len(calendar):
        observed_dates = set(data.loc[data["date"].between(calendar[0], calendar[-1]), "date"])
        if observed_dates - set(calendar):
            raise ValueError("Benchmark has missing sessions present in other price histories")
    by_date = {}
    for order in orders:
        if order["entry_date"] not in calendar or order["exit_date"] not in calendar:
            raise ValueError("Candidate entry and exit must be observed benchmark sessions")
        by_date.setdefault(order["entry_date"], []).append(order)
    lookup = {(row.ticker, row.date): (row.adj_open, row.adj_close) for row in data.itertuples(index=False)}

    def bar(ticker: str, day: pd.Timestamp) -> tuple[float, float]:
        values = lookup.get((ticker, day))
        if values is None or any(not isfinite(value) or value <= 0 for value in values):
            raise ValueError(f"Missing or invalid price bar for {ticker} on {day.date()}")
        return float(values[0]), float(values[1])

    cash = float(initial_cash)
    benchmark_cash = float(initial_cash)
    side_fee = float(cost_bps) / 20000
    held = {}
    trades = []
    skips = []
    daily = []
    scaled_days = []
    max_concurrency = 0
    previous_equity = cash
    previous_benchmark_equity = benchmark_cash

    def skip(order: dict, reason: str) -> None:
        skips.append({**order, "entry_date": order["entry_date"].date().isoformat(),
                      "exit_date": order["exit_date"].date().isoformat(), "reason": reason})

    for day in calendar:
        date = day.date().isoformat()
        benchmark_open, benchmark_close = bar(benchmark_ticker, day)
        opening_cash = cash
        opening_benchmark_cash = benchmark_cash
        opening_equity = cash + sum(trade["shares"] * bar(ticker, day)[0] for ticker, trade in held.items())
        opening_benchmark_equity = benchmark_cash + sum(trade["benchmark_shares"] * benchmark_open for trade in held.values())
        eligible = []
        for order in by_date.get(day, []):
            if order["target_ticker"] in held:
                skip(order, "already_held")
            else:
                eligible.append(order)
        accepted = []
        if cash <= initial_cash * 1e-12:
            for order in eligible:
                skip(order, "no_available_cash")
        else:
            slots = MAX_POSITIONS - len(held)
            accepted = eligible[:slots]
            for order in eligible[slots:]:
                skip(order, "capacity")

        entry_spend = 0.0
        benchmark_entry_spend = 0.0
        if accepted:
            budget = min(MAX_POSITION_FRACTION * opening_equity, cash / len(accepted))
            fraction = budget / opening_equity
            benchmark_requested = fraction * opening_benchmark_equity
            benchmark_budget = min(benchmark_requested, benchmark_cash / len(accepted))
            if benchmark_budget < benchmark_requested - initial_cash * 1e-12:
                scaled_days.append(date)
            for order in accepted:
                ticker = order["target_ticker"]
                entry_price, _ = bar(ticker, day)
                notional = budget / (1 + side_fee)
                benchmark_notional = benchmark_budget / (1 + side_fee)
                trade = {
                    "trade_id": f"{date}:{ticker}", "target_ticker": ticker, "event_ids": order["event_ids"],
                    "entry_date": date, "exit_date": order["exit_date"].date().isoformat(),
                    "entry_price": entry_price, "shares": notional / entry_price,
                    "entry_notional": notional, "entry_fee": notional * side_fee,
                    "allocated_cash": budget, "allocation_fraction": fraction,
                    "benchmark_entry_price": benchmark_open,
                    "benchmark_shares": benchmark_notional / benchmark_open,
                    "benchmark_entry_notional": benchmark_notional,
                    "benchmark_entry_fee": benchmark_notional * side_fee,
                    "benchmark_allocated_cash": benchmark_budget,
                    "benchmark_requested_fraction": fraction,
                    "benchmark_actual_fraction": benchmark_budget / opening_benchmark_equity,
                    "counterfactual_benchmark_shares": notional / benchmark_open,
                }
                held[ticker] = trade
                trades.append(trade)
                entry_spend += budget
                benchmark_entry_spend += benchmark_budget
            cash = max(0.0, cash - entry_spend)
            benchmark_cash = max(0.0, benchmark_cash - benchmark_entry_spend)

        concurrency = len(held)
        max_concurrency = max(max_concurrency, concurrency)
        exit_proceeds = 0.0
        benchmark_exit_proceeds = 0.0
        exited = []
        for ticker, trade in held.items():
            _, close = bar(ticker, day)
            if trade["exit_date"] != date:
                continue
            proceeds = trade["shares"] * close
            exit_fee = proceeds * side_fee
            benchmark_proceeds = trade["benchmark_shares"] * benchmark_close
            benchmark_exit_fee = benchmark_proceeds * side_fee
            counterfactual_proceeds = trade["counterfactual_benchmark_shares"] * benchmark_close
            counterfactual_exit_fee = counterfactual_proceeds * side_fee
            pnl = proceeds - exit_fee - trade["allocated_cash"]
            benchmark_pnl = benchmark_proceeds - benchmark_exit_fee - trade["benchmark_allocated_cash"]
            counterfactual_pnl = counterfactual_proceeds - counterfactual_exit_fee - trade["allocated_cash"]
            trade.update({
                "exit_price": close, "exit_notional": proceeds, "exit_fee": exit_fee,
                "total_fees": trade["entry_fee"] + exit_fee, "pnl": pnl,
                "return_on_allocated_cash": pnl / trade["allocated_cash"],
                "benchmark_exit_price": benchmark_close, "benchmark_exit_notional": benchmark_proceeds,
                "benchmark_exit_fee": benchmark_exit_fee,
                "benchmark_total_fees": trade["benchmark_entry_fee"] + benchmark_exit_fee,
                "benchmark_pnl": benchmark_pnl,
                "counterfactual_benchmark_exit_notional": counterfactual_proceeds,
                "counterfactual_benchmark_total_fees": trade["entry_fee"] + counterfactual_exit_fee,
                "counterfactual_benchmark_pnl": counterfactual_pnl,
                "pnl_minus_same_dollar_benchmark": pnl - counterfactual_pnl,
            })
            exit_proceeds += proceeds - exit_fee
            benchmark_exit_proceeds += benchmark_proceeds - benchmark_exit_fee
            exited.append(ticker)
        for ticker in exited:
            del held[ticker]
        cash += exit_proceeds
        benchmark_cash += benchmark_exit_proceeds
        holdings = [{"target_ticker": ticker, "shares": trade["shares"],
                     "market_value": trade["shares"] * bar(ticker, day)[1]}
                    for ticker, trade in sorted(held.items())]
        equity = cash + sum(position["market_value"] for position in holdings)
        benchmark_equity = benchmark_cash + sum(trade["benchmark_shares"] * benchmark_close for trade in held.values())
        daily.append({
            "date": date, "opening_cash": opening_cash, "opening_equity": opening_equity,
            "entry_spend": entry_spend, "exit_proceeds": exit_proceeds,
            "cash": cash, "equity": equity, "daily_return": equity / previous_equity - 1,
            "holdings": holdings, "concurrent_positions": concurrency,
            "entries": len(accepted), "exits": len(exited),
            "benchmark_opening_cash": opening_benchmark_cash,
            "benchmark_opening_equity": opening_benchmark_equity,
            "benchmark_entry_spend": benchmark_entry_spend, "benchmark_exit_proceeds": benchmark_exit_proceeds,
            "benchmark_cash": benchmark_cash, "benchmark_equity": benchmark_equity,
            "benchmark_daily_return": benchmark_equity / previous_benchmark_equity - 1,
        })
        previous_equity = equity
        previous_benchmark_equity = benchmark_equity

    return {
        "accounting_basis": "synthetic_adjusted_total_return_units",
        "initial_cash": float(initial_cash), "cost_bps": float(cost_bps),
        "period_start": calendar[0].date().isoformat() if len(calendar) else None,
        "period_end": calendar[-1].date().isoformat() if len(calendar) else None,
        "portfolio_ending_equity": previous_equity, "portfolio_pnl": previous_equity - initial_cash,
        "portfolio_return": previous_equity / initial_cash - 1,
        "total_costs": sum(trade["total_fees"] for trade in trades),
        "benchmark_ticker": benchmark_ticker,
        "benchmark_ending_equity": previous_benchmark_equity,
        "benchmark_pnl": previous_benchmark_equity - initial_cash,
        "benchmark_return": previous_benchmark_equity / initial_cash - 1,
        "benchmark_total_costs": sum(trade["benchmark_total_fees"] for trade in trades),
        "benchmark_cash_scaled_days": scaled_days,
        "n_trades": len(trades), "max_concurrency": max_concurrency,
        "max_positions": MAX_POSITIONS, "max_position_fraction": MAX_POSITION_FRACTION,
        "trades": trades, "daily": daily, "skips": skips,
        "assumptions": [
            "Long-only, cash-constrained allocation simulation using fractional adjusted total-return units; not literal broker cash accounting.",
            "Each side pays half the configured roundtrip basis-point rate on its actual notional; entry budgets include fees.",
            "Entry budgets use opening marked equity. Exits use supplied closing dates; sale proceeds first become available at the following open.",
            "Sizing from the same opening prices used for fills is a research proxy, not a feasible pre-auction order specification.",
            "Adjusted-price dividend gains enter simulated cash at exit; actual dividend receivables and payment dates are not modeled.",
            "The 25% limit caps new position budgets; existing holding weights can drift above 25% without rebalancing.",
            "The supplied benchmark dates define the session calendar and full evaluation period; missing held-window bars fail validation.",
            f"Benchmark: separate cash-constrained {benchmark_ticker} lots on the stock entry/exit schedule, with requested stock allocation fractions and explicit cash scaling.",
            "Same-dollar benchmark trade P&L is attribution only, not a self-financing account or another portfolio return.",
        ],
    }
