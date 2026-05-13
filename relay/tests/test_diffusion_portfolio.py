import json

import pandas as pd
import pytest

from relay.diffusion_portfolio import replay_portfolio


DATES = pd.bdate_range("2025-01-06", periods=10)


def prices_for(tickers, paths=None):
    paths = paths or {}
    return pd.DataFrame([
        {"ticker": ticker, "date": day, "adj_open": paths.get(ticker, [100] * len(DATES))[i],
         "adj_close": paths.get(ticker, [100] * len(DATES))[i]}
        for ticker in ["SPY", *tickers] for i, day in enumerate(DATES)
    ])


def candidate(ticker, entry=0, exit_date=4, event_id=None):
    return {"target_ticker": ticker, "entry_date": DATES[entry].date().isoformat(),
            "exit_date": DATES[exit_date].date().isoformat(), "event_ids": [event_id or f"event-{ticker}-{entry}"]}


def test_flat_stock_and_benchmark_pay_both_fees_with_entry_fee_inside_budget():
    result = replay_portfolio([candidate("A")], prices_for(["A"]), cost_bps=20)
    trade = result["trades"][0]
    expected_notional = 2500 / 1.001
    expected_fees = 2 * expected_notional * 0.001
    assert trade["allocated_cash"] == 2500
    assert trade["entry_notional"] == pytest.approx(expected_notional)
    assert trade["entry_fee"] + trade["entry_notional"] == pytest.approx(2500)
    assert trade["total_fees"] == pytest.approx(expected_fees)
    assert result["portfolio_pnl"] == pytest.approx(-expected_fees)
    assert result["benchmark_pnl"] == pytest.approx(result["portfolio_pnl"])
    assert trade["counterfactual_benchmark_pnl"] == pytest.approx(trade["pnl"])
    assert result["daily"][0]["cash"] == 7500
    assert result["daily"][4]["holdings"] == []
    json.dumps(result, allow_nan=False)


def test_duplicate_targets_merge_event_provenance_without_duplicate_capital():
    orders = [candidate("A", event_id="z"), candidate("A", event_id="a"), candidate("A", event_id="z")]
    result = replay_portfolio(orders, prices_for(["A"]), cost_bps=0)
    assert result["n_trades"] == 1
    assert result["trades"][0]["event_ids"] == ["a", "z"]
    assert result["daily"][0]["entry_spend"] == 2500
    assert result["portfolio_return"] == 0


def test_capacity_and_alphabetical_tie_break_are_separate_from_allocation_cap():
    orders = [candidate(ticker, exit_date=3 if ticker == "A" else 7) for ticker in ["D", "C", "B", "A"]]
    orders += [candidate(ticker, entry=4, exit_date=8) for ticker in ["G", "F", "E"]]
    result = replay_portfolio(orders, prices_for(list("ABCDEFG")), cost_bps=0)
    assert [trade["target_ticker"] for trade in result["trades"]] == list("ABCDE")
    assert result["max_concurrency"] == 4
    assert [(skip["target_ticker"], skip["reason"]) for skip in result["skips"]] == [("F", "capacity"), ("G", "capacity")]
    assert all(trade["allocation_fraction"] <= 0.25 for trade in result["trades"])
    assert all(day["concurrent_positions"] <= 4 for day in result["daily"])


def test_closing_proceeds_cannot_fund_that_mornings_entry_and_held_targets_are_skipped():
    orders = [candidate(ticker, exit_date=4) for ticker in "ABCD"]
    orders += [candidate("A", entry=4, exit_date=8), candidate("E", entry=4, exit_date=8), candidate("E", entry=5, exit_date=9)]
    result = replay_portfolio(orders, prices_for(list("ABCDE")), cost_bps=0)
    assert result["daily"][4]["opening_cash"] == 0
    assert result["daily"][4]["entry_spend"] == 0
    assert result["daily"][4]["exit_proceeds"] == 10000
    assert result["daily"][5]["entry_spend"] == 2500
    reasons = {(skip["target_ticker"], skip["entry_date"]): skip["reason"] for skip in result["skips"]}
    assert reasons[("A", DATES[4].date().isoformat())] == "already_held"
    assert reasons[("E", DATES[4].date().isoformat())] == "no_available_cash"


def test_new_orders_share_available_cash_equally_after_marking_existing_holdings():
    orders = [candidate("A", exit_date=8)] + [candidate(ticker, entry=1, exit_date=5) for ticker in "BCD"]
    prices = prices_for(list("ABCD"), {"A": [100] + [200] * 9})
    result = replay_portfolio(orders, prices, cost_bps=0)
    assert result["daily"][1]["opening_equity"] == 12500
    assert [trade["allocated_cash"] for trade in result["trades"][1:]] == [2500, 2500, 2500]
    assert [trade["allocation_fraction"] for trade in result["trades"][1:]] == [0.2, 0.2, 0.2]
    assert result["daily"][1]["cash"] == 0


def test_matching_stock_paths_match_separate_benchmark_account_even_with_overlap():
    path = [100, 101, 98, 102, 104, 103, 106, 107, 105, 108]
    prices = prices_for(list("ABCD"), {ticker: path for ticker in ["SPY", *"ABCD"]})
    orders = [candidate("A", exit_date=4), candidate("B", entry=1, exit_date=5),
              candidate("C", entry=2, exit_date=6), candidate("D", entry=3, exit_date=7)]
    result = replay_portfolio(orders, prices, cost_bps=50)
    assert result["benchmark_cash_scaled_days"] == []
    for day in result["daily"]:
        assert day["equity"] == pytest.approx(day["benchmark_equity"])
        assert day["cash"] == pytest.approx(day["benchmark_cash"])
    assert result["portfolio_pnl"] == pytest.approx(result["benchmark_pnl"])
    assert result["total_costs"] == pytest.approx(result["benchmark_total_costs"])


def test_benchmark_cash_scaling_is_explicit_and_same_dollar_attribution_stays_separate():
    prices = prices_for(list("ABCD"), {"SPY": [100] + [200] * 9,
                                      **{ticker: [100] + [50] * 9 for ticker in "ABC"}})
    orders = [candidate(ticker, exit_date=6) for ticker in "ABC"] + [candidate("D", entry=1, exit_date=7)]
    result = replay_portfolio(orders, prices, cost_bps=0)
    last = result["trades"][-1]
    assert result["benchmark_cash_scaled_days"] == [DATES[1].date().isoformat()]
    assert last["allocated_cash"] == 1562.5
    assert last["benchmark_requested_fraction"] == 0.25
    assert last["benchmark_allocated_cash"] == 2500
    assert last["benchmark_actual_fraction"] == pytest.approx(2500 / 17500)
    assert last["counterfactual_benchmark_shares"] == pytest.approx(1562.5 / 200)
    assert last["benchmark_shares"] == pytest.approx(2500 / 200)
    assert all(day["cash"] >= 0 and day["benchmark_cash"] >= 0 for day in result["daily"])
    assert result["portfolio_pnl"] == pytest.approx(sum(trade["pnl"] for trade in result["trades"]))
    assert result["benchmark_pnl"] == pytest.approx(sum(trade["benchmark_pnl"] for trade in result["trades"]))


def test_future_exit_prices_do_not_change_earlier_entry_decisions():
    prices = prices_for(["A", "B"])
    orders = [candidate("A", exit_date=6), candidate("B", entry=1, exit_date=7)]
    original = replay_portfolio(orders, prices)
    changed = prices.copy()
    changed.loc[(changed.ticker == "A") & (changed.date >= DATES[6]), ["adj_open", "adj_close"]] *= 3
    altered = replay_portfolio(orders, changed)
    assert original["daily"][:6] == altered["daily"][:6]
    fields = ("target_ticker", "entry_date", "shares", "entry_notional", "allocated_cash", "allocation_fraction")
    assert [{key: trade[key] for key in fields} for trade in original["trades"]] == [
        {key: trade[key] for key in fields} for trade in altered["trades"]]
    assert original["portfolio_pnl"] != altered["portfolio_pnl"]


@pytest.mark.parametrize("problem", ["missing", "nonfinite", "zero"])
def test_missing_or_invalid_bars_inside_held_window_fail(problem):
    prices = prices_for(["A"])
    mask = (prices.ticker == "A") & (prices.date == DATES[2])
    if problem == "missing":
        prices = prices.loc[~mask]
    else:
        prices.loc[mask, "adj_close"] = float("nan") if problem == "nonfinite" else 0
    with pytest.raises(ValueError, match="Missing or invalid price bar"):
        replay_portfolio([candidate("A")], prices)


def test_missing_benchmark_session_future_exit_and_conflicting_duplicate_fail():
    prices = prices_for(["A"])
    missing_benchmark = prices.loc[~((prices.ticker == "SPY") & (prices.date == DATES[2]))]
    with pytest.raises(ValueError, match="Benchmark has missing sessions"):
        replay_portfolio([candidate("A")], missing_benchmark)
    future = {**candidate("A"), "exit_date": "2026-01-01"}
    with pytest.raises(ValueError, match="observed benchmark sessions"):
        replay_portfolio([future], prices)
    with pytest.raises(ValueError, match="same exit_date"):
        replay_portfolio([candidate("A"), candidate("A", exit_date=5)], prices)


def test_full_benchmark_period_and_empty_rules_preserve_cash_without_annualization():
    prices = prices_for(["A"])
    empty = replay_portfolio([], prices)
    active = replay_portfolio([candidate("A", entry=2, exit_date=6)], prices)
    assert len(empty["daily"]) == len(active["daily"]) == len(DATES)
    assert empty["period_start"] == active["period_start"] == DATES[0].date().isoformat()
    assert empty["period_end"] == active["period_end"] == DATES[-1].date().isoformat()
    assert empty["portfolio_return"] == empty["benchmark_return"] == 0
    assert empty["n_trades"] == 0 and empty["trades"] == []
    assert all(day["cash"] == day["benchmark_cash"] == 10000 for day in empty["daily"])
    assert "annualized_return" not in empty and "sharpe" not in empty
    no_prices = replay_portfolio([], prices.iloc[:0])
    assert no_prices["daily"] == [] and no_prices["portfolio_return"] == 0


@pytest.mark.parametrize("arguments", [{"initial_cash": 0}, {"initial_cash": float("inf")},
                                      {"cost_bps": -1}, {"cost_bps": float("nan")}, {"cost_bps": 10000}])
def test_invalid_cash_and_fee_inputs_fail(arguments):
    with pytest.raises(ValueError):
        replay_portfolio([], prices_for([]), **arguments)
