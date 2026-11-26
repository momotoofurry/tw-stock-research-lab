"""選股排名策略：基本面時間點、因子無 look-ahead、多檔引擎與單檔引擎一致。不連網。"""
import numpy as np
import pandas as pd
import pytest

from src import oracles
from src.backtest import Backtester, Costs, buy_and_hold_signals
from src.fundamentals import as_of
from src.rank_strategy import StockData, Universe, factor_table, run_portfolio, select


def _rev_table(periods, yoy3):
    t = pd.DataFrame({"period": pd.to_datetime(periods), "yoy3": yoy3, "revenue": 1.0})
    t["available"] = t["period"] + pd.offsets.MonthBegin(1) + pd.Timedelta(days=9)
    return t


def test_revenue_available_only_after_10th_of_next_month():
    t = _rev_table(["2024-08-01", "2024-09-01"], [0.1, -0.2])
    assert as_of(t, pd.Timestamp("2024-10-09"), "yoy3") == 0.1     # 9 月營收還沒公布
    assert as_of(t, pd.Timestamp("2024-10-10"), "yoy3") == -0.2    # 10/10 收盤後可用
    assert np.isnan(as_of(t, pd.Timestamp("2024-09-01"), "yoy3"))  # 8 月營收 9/10 才公布


def _universe(n_stocks=5, n_days=400, seed=0):
    stocks = {}
    for k in range(n_stocks):
        p = oracles.synthetic_prices(n_days, seed=seed + k, start="2020-01-01")
        p["stock_id"] = f"S{k}"
        px = p.set_index("date")
        stocks[f"S{k}"] = StockData(
            prices=px, adj_close=px["close"], dividends=pd.Series(dtype=float), share_mult=pd.Series(dtype=float),
            revenue=_rev_table(pd.date_range("2019-01-01", periods=40, freq="MS"), np.linspace(-0.2, 0.3, 40) + k / 10),
            eps=pd.DataFrame({"quarter_end": [pd.Timestamp("2019-12-31")], "eps": [1.0], "eps_ttm": [4.0],
                              "available": [pd.Timestamp("2020-03-30")]}))
    cal = pd.DatetimeIndex(sorted(set().union(*[set(s.prices.index) for s in stocks.values()])))
    return Universe(stocks, cal)


def test_factor_table_has_no_lookahead():
    u = _universe()
    day = u.calendar[300]
    full = factor_table(u, day)
    cut = Universe({s: StockData(x.prices[x.prices.index <= day], x.adj_close[x.adj_close.index <= day],
                                 x.dividends, x.share_mult, x.revenue[x.revenue["available"] <= day],
                                 x.eps[x.eps["available"] <= day]) for s, x in u.stocks.items()},
                   u.calendar[u.calendar <= day])
    pd.testing.assert_frame_equal(full.reset_index(drop=True), factor_table(cut, day).reset_index(drop=True))


def test_select_filters_and_ranks():
    f = pd.DataFrame([dict(stock_id="A", close=50, mom63=0.1, mom=0.3, yoy3=0.1, eps_ttm=1, turnover=1e8),
                      dict(stock_id="B", close=50, mom63=0.1, mom=0.5, yoy3=-0.1, eps_ttm=1, turnover=1e8),
                      dict(stock_id="C", close=50, mom63=0.1, mom=0.4, yoy3=0.2, eps_ttm=1, turnover=1e8),
                      dict(stock_id="D", close=5, mom63=0.1, mom=0.9, yoy3=0.2, eps_ttm=1, turnover=1e8)])
    assert select(f, top_n=2) == ["C", "A"]                        # B 營收衰退、D 股價太低
    assert select(f, top_n=2, use_fundamentals=False) == ["B", "C"]


def test_portfolio_engine_matches_single_stock_engine():
    u = _universe(n_stocks=1)
    p = u.stocks["S0"].prices.reset_index()
    costs = Costs()
    start = u.calendar[0]
    single = Backtester(costs, 1_000_000, "odd").run(p, buy_and_hold_signals(p, start))
    multi = run_portfolio(u, {start: ["S0"]}, costs, 1_000_000, cash_buffer=0.0)
    # 多檔引擎用前一日收盤估股數、單檔引擎用開盤價估，股數略有不同；報酬應非常接近
    r_single = single.equity.iloc[-1] / single.equity.iloc[0] - 1
    r_multi = multi.equity.iloc[-1] / multi.equity.iloc[0] - 1
    assert r_multi == pytest.approx(r_single, abs=0.01)
    assert (multi.trades["date"] > start).all()                     # 訊號日當天不成交


def test_portfolio_engine_split_keeps_value():
    u = _universe(n_stocks=1)
    x = u.stocks["S0"]
    d = u.calendar[200]
    px = x.prices.copy()
    px.loc[px.index >= d, ["open", "high", "low", "close"]] /= 4
    u.stocks["S0"] = StockData(px, x.adj_close, x.dividends, pd.Series({d: 4.0}), x.revenue, x.eps)
    r = run_portfolio(u, {u.calendar[0]: ["S0"]}, Costs(), 1_000_000, cash_buffer=0.0)
    jump = r.equity.loc[d] / r.equity.iloc[199] - 1
    assert abs(jump) < 0.1                                          # 沒處理會是 −75%


def test_delisted_holding_is_cashed_out():
    u = _universe(n_stocks=2)
    x = u.stocks["S1"]
    cut = u.calendar[200]
    u.stocks["S1"] = StockData(x.prices[x.prices.index <= cut], x.adj_close[x.adj_close.index <= cut], x.dividends,
                               x.share_mult, x.revenue, x.eps)
    r = run_portfolio(u, {u.calendar[0]: ["S0", "S1"]}, Costs(), 1_000_000, cash_buffer=0.0)
    sold = r.trades[(r.trades["stock_id"] == "S1") & (r.trades["side"] == "sell")]
    assert len(sold) == 1 and sold["date"].iloc[0] == cut
    assert r.skipped["reason"].str.contains("下市").any()
    assert not np.isnan(r.equity.iloc[-1])


def test_volume_cap_uses_only_prior_days_and_spreads_buys():
    u = _universe(n_stocks=1)
    x = u.stocks["S0"]
    px = x.prices.copy()
    px["volume"] = 10_000.0
    fill = u.calendar[1]
    px.loc[fill, "volume"] = 1e9                                    # 成交當天的大量不能拿來放寬上限
    u.stocks["S0"] = StockData(px, x.adj_close, x.dividends, x.share_mult, x.revenue, x.eps)
    r = run_portfolio(u, {u.calendar[0]: ["S0"]}, Costs(), 1_000_000, cash_buffer=0.0, max_participation=0.05)
    buys = r.trades[r.trades["side"] == "buy"]
    assert buys["shares"].iloc[0] == 500 and buys["capped"].iloc[0]  # 前 20 日均量 10,000 × 5%
    assert buys["date"].nunique() > 1                                # 沒買足的部分隔天繼續買


def test_eligible_pools_apply_filters_on_that_day_only():
    from src.rank_strategy import eligible_pools
    u = _universe(n_stocks=3)
    d = u.calendar[300]
    x = u.stocks["S2"]
    px = x.prices.copy()
    px.loc[px.index > d, "volume"] *= 1000                           # 之後才變熱門：當天不該入池
    px.loc[px.index <= d, "volume"] = 1.0
    u.stocks["S2"] = StockData(px, x.adj_close, x.dividends, x.share_mult, x.revenue, x.eps)
    pools = eligible_pools(u, [d], lookback=126, min_price=0, min_turnover=1e4)
    assert "S2" not in pools[d] and {"S0", "S1"} <= set(pools[d])


def test_matched_random_targets_match_count_and_holdover():
    from src.rank_strategy import matched_random_targets
    days = pd.bdate_range("2021-01-01", periods=4)
    pool = [f"P{k}" for k in range(30)]
    targets = {days[0]: ["P0", "P1", "P2"], days[1]: ["P0", "P1", "P3"], days[2]: [], days[3]: ["P5"]}
    pools = {d: pool for d in days}
    r = matched_random_targets(targets, pools, np.random.default_rng(1))
    assert [len(r[d]) for d in days] == [3, 3, 0, 1]
    assert len(set(r[days[0]]) & set(r[days[1]])) == 2               # 策略續抱 2 檔 → 隨機也續抱 2 檔
    assert all(set(v) <= set(pool) for v in r.values())


def test_liquid_ever_counts_any_day_after_start():
    from src.bulk import liquid_ever
    d = pd.bdate_range("2020-01-01", periods=200)
    px = pd.DataFrame({"date": d, "close": 20.0, "volume": 1.0})
    px.loc[150:175, "volume"] = 1e7                                 # 只有中間一小段達標，不在每 63 天的抽樣日上也要列入
    assert liquid_ever(px, "2020-01-01", min_price=10, min_turnover=5e7)
    assert not liquid_ever(px.iloc[:100], "2020-01-01", min_price=10, min_turnover=5e7)


def test_delist_recovery_zero_loses_the_position():
    u = _universe(n_stocks=2)
    x = u.stocks["S1"]
    cut = u.calendar[200]
    u.stocks["S1"] = StockData(x.prices[x.prices.index <= cut], x.adj_close[x.adj_close.index <= cut], x.dividends,
                               x.share_mult, x.revenue, x.eps)
    tg = {u.calendar[0]: ["S0", "S1"]}
    base = run_portfolio(u, tg, Costs(), 1_000_000, cash_buffer=0.0)
    zero = run_portfolio(u, tg, Costs(), 1_000_000, cash_buffer=0.0, delist_recovery=0.0)
    i = u.calendar.get_loc(cut)
    lost = base.equity.iloc[i] - zero.equity.iloc[i]
    held_value = base.equity.iloc[i - 1] - zero.equity.iloc[i - 1]
    assert held_value == pytest.approx(0) and lost > 0.3 * base.equity.iloc[i]   # 前一天相同；下市當天少掉約一半


def test_suspension_gaps():
    from src.integrity import suspension_gaps
    cal = pd.bdate_range("2024-01-01", periods=30)
    dates = pd.Series(cal.delete(range(10, 17)))                    # 缺 7 天
    assert suspension_gaps(dates, cal) == [(cal[10], cal[16], 7)]
    assert suspension_gaps(pd.Series(cal.delete([3, 4])), cal) == []
