"""第 1 步：股利、還原價、保留期、顯著性檢定。"""
import copy

import numpy as np
import pandas as pd
import pytest

from src import evaluate, oracles
from src.adjust import adjust_for_dividends
from src.backtest import Backtester, Costs, buy_and_hold_signals
from src.config import load_config
from src.features import build_features
from src.metrics import compute_metrics
from src.significance import matched_random_signals, random_benchmark, sign_test_p, transition_probs


def _flat(n=6, price=100.0):
    dates = pd.bdate_range("2024-01-01", periods=n)
    return pd.DataFrame({"date": dates, "open": price, "high": price + 1, "low": price - 1, "close": price,
                         "volume": 1_000_000, "spread": np.nan})


def _div(date, before, after):
    return pd.DataFrame({"date": [pd.Timestamp(date)], "stock_id": ["X"], "before_price": [before],
                         "after_price": [after], "stock_and_cache_dividend": [before - after]})


# ---------- 股利 ----------
def test_dividend_hand_calc():
    p = _flat()
    p.loc[3:, ["open", "high", "low", "close"]] = [97.0, 98.0, 96.0, 97.0]  # 第 3 天除息 3 元
    divs = _div(p["date"][3], 100.0, 97.0)
    sig = pd.Series([1.0, np.nan, np.nan, np.nan, np.nan, np.nan], index=p["date"])
    res = Backtester(Costs(), 200_000, "board").run(p, sig, divs)
    # 第 1 天開盤 100 買 1 張（手續費 142）；第 3 天前一日收盤持有 1000 股 → 入帳 3000
    assert res.total_dividends == 3_000
    assert res.equity.iloc[-1] == 200_000 - 100_142 + 3_000 + 97_000
    assert list(res.dividends["shares"]) == [1000]


def test_dividend_tax_and_no_dividend_when_flat():
    p = _flat()
    divs = _div(p["date"][3], 100.0, 97.0)
    taxed = Backtester(Costs(dividend_tax_rate=0.0211), 200_000).run(
        p, pd.Series([1.0] + [np.nan] * 5, index=p["date"]), divs)
    assert taxed.total_dividends == pytest.approx(3000 * (1 - 0.0211))
    # 除息前一天收盤就已賣出（第 2 天開盤賣）→ 沒有股利
    sold = Backtester(Costs(), 200_000).run(p, pd.Series([1.0, 0.0] + [np.nan] * 4, index=p["date"]), divs)
    assert sold.total_dividends == 0


def test_sell_on_ex_date_still_gets_dividend():
    p = _flat()
    divs = _div(p["date"][3], 100.0, 97.0)
    sig = pd.Series([1.0, np.nan, 0.0, np.nan, np.nan, np.nan], index=p["date"])  # 第 3 天開盤賣
    res = Backtester(Costs(), 200_000).run(p, sig, divs)
    assert res.total_dividends == 3000 and res.round_trips.iloc[0].dividends == 3000


# ---------- 還原價 ----------
def test_forward_adjust_scales_from_ex_date_on():
    p = _flat()
    p.loc[3:, "close"] = 97.0                     # 除息 3 元，價格沒漲沒跌
    adj = adjust_for_dividends(p, _div(p["date"][3], 100.0, 97.0))
    assert (adj["close"][:3] == 100.0).all()      # 除息日之前完全不動
    assert np.allclose(adj["close"][3:], 100.0)   # 除息後還原，總報酬為 0
    assert (p["close"][:3] == 100.0).all()        # 原資料不被改動


def test_future_dividend_never_changes_past_adjusted_values():
    p = oracles.synthetic_prices(300, seed=3)
    divs = _div(p["date"][200], p["close"][199], p["close"][199] * 0.9)
    full = adjust_for_dividends(p, divs)
    part = adjust_for_dividends(p.iloc[:150], divs)
    pd.testing.assert_frame_equal(full.iloc[:150].drop(columns="spread"), part.drop(columns="spread"))


def test_adjust_ignores_dividends_after_last_price():
    p = _flat()
    adj = adjust_for_dividends(p, _div("2030-01-01", 100.0, 50.0))
    pd.testing.assert_series_equal(adj["close"], p["close"])


def test_adjusted_features_have_no_lookahead():
    prices = oracles.synthetic_prices(400, seed=7)
    d = prices["date"]
    divs = pd.concat([_div(d[150], prices["close"][149], prices["close"][149] * 0.95),
                      _div(d[300], prices["close"][299], prices["close"][299] * 0.93)])
    r = oracles.check_feature_no_lookahead(lambda p: build_features(adjust_for_dividends(p, divs)), prices)
    assert r.passed, r.detail


# ---------- 期間績效 ----------
def test_window_metrics_only_count_costs_inside_window():
    p = oracles.synthetic_prices(300, seed=1)
    sig = oracles.random_signals(p, 0)
    res = Backtester(Costs(), 1_000_000).run(p, sig)
    cut = p["date"][150]
    a = compute_metrics(res, None, cut - pd.Timedelta(days=1))
    b = compute_metrics(res, cut, None)
    assert a["total_commission"] + b["total_commission"] == pytest.approx(res.total_commission)
    assert a["fills"] + b["fills"] == len(res.trades)
    total = (1 + a["total_return"]) * (res.equity[res.equity.index >= cut].iloc[0]
                                      / res.equity[res.equity.index < cut].iloc[-1]) * (1 + b["total_return"]) - 1
    assert total == pytest.approx(res.equity.iloc[-1] / res.equity.iloc[0] - 1)


# ---------- 保留期 ----------
@pytest.fixture
def fake_data(monkeypatch):
    prices = oracles.synthetic_prices(2300, seed=4, start="2018-01-01")  # 到 2026 年中，跨過保留期起日
    monkeypatch.setattr(evaluate, "fetch_prices",
                        lambda sid, s, e, cfg: prices[(prices["date"] >= s) & (prices["date"] <= e)].reset_index(drop=True))
    monkeypatch.setattr(evaluate, "fetch_events",
                        lambda sid, s, e, cfg, prices=None: pd.DataFrame(columns=["date", "stock_id", "before_price",
                                                                                  "after_price", "kind"]))
    return prices


def test_default_run_never_sees_holdout(fake_data):
    cfg = load_config()
    run = evaluate.run_stock("X", "ma_cross", "2018-01-01", "2024-12-31", cfg)
    hs = pd.Timestamp(cfg["evaluation"]["holdout_start"])
    run_late = evaluate.run_stock("X", "ma_cross", "2018-01-01", "2026-12-31", cfg)
    assert run.prices["date"].max() < hs and run_late.prices["date"].max() < hs
    assert run.signal.index.max() < hs and run.res.equity.index.max() < hs
    assert list(run_late.windows) == ["開發期"]


def test_holdout_run_has_both_windows_and_dev_unchanged(fake_data, tmp_path):
    cfg = copy.deepcopy(load_config())
    cfg["reports_dir"] = str(tmp_path)
    dev = evaluate.run_stock("X", "ma_cross", "2018-01-01", "2026-12-31", cfg)
    full = evaluate.run_stock("X", "ma_cross", "2018-01-01", "2026-12-31", cfg, holdout=True)
    assert list(full.windows) == ["開發期", "保留期"]
    m_dev = evaluate.window_metrics(dev, cfg)["開發期"][0]
    m_full = evaluate.window_metrics(full, cfg)["開發期"][0]
    assert m_dev["cagr"] == pytest.approx(m_full["cagr"])  # 加入保留期不改變開發期結果（沒有偷看）
    assert evaluate.log_holdout_view(cfg, "test") == 1
    assert evaluate.log_holdout_view(cfg, "test") == 2


# ---------- 顯著性 ----------
def test_matched_random_signals_match_turnover_and_exposure():
    rng = np.random.default_rng(0)
    base = pd.Series(np.repeat([0.0, 1.0] * 50, 10), index=pd.bdate_range("2020-01-01", periods=1000))
    pe, px, _ = transition_probs(base)
    sims = [matched_random_signals(base, rng) for _ in range(50)]
    assert np.mean([s.mean() for s in sims]) == pytest.approx(base.mean(), abs=0.05)
    assert np.mean([transition_probs(s)[0] for s in sims]) == pytest.approx(pe, rel=0.2)
    assert np.mean([transition_probs(s)[1] for s in sims]) == pytest.approx(px, rel=0.2)


def test_random_benchmark_flags_cheater_not_random():
    prices = oracles.synthetic_prices(600, seed=2)
    eng = Backtester(Costs(), 1_000_000)
    cheat = oracles.cheat_signals(prices, Costs())
    c = compute_metrics(eng.run(prices, cheat))["cagr"]
    assert random_benchmark(eng, prices, cheat, c, n=50)["p_value"] < 0.05
    rnd = oracles.random_signals(prices, 99)
    r = compute_metrics(eng.run(prices, rnd))["cagr"]
    assert random_benchmark(eng, prices, rnd, r, n=50)["p_value"] > 0.05


def test_sign_test():
    assert sign_test_p(0, 10) == 1.0
    assert sign_test_p(10, 10) == pytest.approx(1 / 1024)
    assert sign_test_p(5, 10) == pytest.approx(638 / 1024)


def test_buy_and_hold_with_dividends_beats_without():
    p = _flat(10)
    divs = _div(p["date"][5], 100.0, 97.0)
    p.loc[5:, ["open", "high", "low", "close"]] = [97.0, 98.0, 96.0, 97.0]
    eng = Backtester(Costs(), 1_000_000)
    with_d = eng.run(p, buy_and_hold_signals(p), divs).equity.iloc[-1]
    without = eng.run(p, buy_and_hold_signals(p)).equity.iloc[-1]
    assert with_d - without == 9 * 1000 * 3  # 9 張 × 每股 3 元


# ---------- 公司行動（減資 / 分割）----------
def test_split_event_keeps_value():
    dates = pd.bdate_range("2025-06-12", periods=6)
    close = [188.0, 188.65, 47.5, 47.6, 48.0, 48.2]
    p = pd.DataFrame({"date": dates, "stock_id": "0050", "open": close, "high": close, "low": close, "close": close,
                      "volume": 1_000_000, "spread": np.nan})
    ev = pd.DataFrame([dict(date=dates[2], stock_id="0050", before_price=188.65, after_price=47.16,
                            stock_and_cache_dividend=np.nan, kind="corporate")])
    sig = pd.Series([1.0] + [np.nan] * 5, index=dates)
    res = Backtester(Costs(), 1_000_000, "odd").run(p, sig, ev)
    assert res.equity.iloc[2] / res.equity.iloc[1] - 1 == pytest.approx(47.5 / 47.16 - 1, abs=0.002)
    no_adj = Backtester(Costs(), 1_000_000, "odd").run(p, sig)
    assert no_adj.equity.iloc[2] / no_adj.equity.iloc[1] - 1 < -0.7   # 沒處理就會是假的 −75%


def test_cash_then_stock_dividend_same_day():
    """同一天除息 2 元 + 配股每股 1 元（股數 ×1.1）：現金以原股數計，股數再調整。"""
    dates = pd.bdate_range("2024-01-01", periods=4)
    close = [100.0, 100.0, (100 - 2) / 1.1, (100 - 2) / 1.1]
    p = pd.DataFrame({"date": dates, "open": close, "high": close, "low": close, "close": close,
                      "volume": 1_000_000, "spread": np.nan})
    ev = pd.DataFrame([dict(date=dates[2], before_price=100.0, after_price=98.0, kind="dividend"),
                       dict(date=dates[2], before_price=1.1, after_price=1.0, kind="corporate")])
    res = Backtester(Costs(), 200_000, "board").run(p, pd.Series([1.0, np.nan, np.nan, np.nan], index=dates), ev)
    assert res.total_dividends == 2000                       # 1000 股 × 2 元（不是 1100 股）
    assert res.equity.iloc[2] == pytest.approx(res.equity.iloc[1])   # 價值不變


def test_official_events_shift_holiday_and_skip_known(monkeypatch, tmp_path):
    from src import fetch
    rows = {
        "TaiwanStockCapitalReductionReferencePrice": [],
        "TaiwanStockSplitPrice": [dict(date="2023-08-07", stock_id="X", before_price=100.0, after_price=25.0)],
        "TaiwanStockParValueChange": [dict(date="2023-08-07", stock_id="X", before_close=100.0, after_ref_close=25.0)],
        "TaiwanStockDividend": [dict(CashExDividendTradingDate="2023-08-03", StockExDividendTradingDate="2023-08-03",
                                     CashEarningsDistribution=7.7, CashStatutorySurplus=0, StockEarningsDistribution=1.2,
                                     StockStatutorySurplus=0)],
    }
    monkeypatch.setattr(fetch, "_cached_rows", lambda ds, key, params, cfg, ttl, sleep=None: rows[ds])
    dates = pd.DatetimeIndex(["2023-08-01", "2023-08-02", "2023-08-04", "2023-08-07"])   # 8/3 颱風休市
    px = pd.DataFrame({"date": dates, "close": [212.0, 212.0, 182.5, 46.0]})
    ev = fetch.official_corporate_events("X", px, pd.DataFrame(columns=["date"]), "2023-01-01", {})
    assert set(zip(ev["date"].dt.strftime("%m-%d"), ev["kind"])) == {("08-04", "dividend"), ("08-04", "corporate"),
                                                                      ("08-07", "corporate")}
    assert len(ev[ev["date"] == "2023-08-07"]) == 1          # 分割與面額變更重複只留一筆
    d = ev[(ev["kind"] == "dividend")].iloc[0]
    assert d["before_price"] - d["after_price"] == pytest.approx(7.7)
    known = pd.DataFrame({"date": [pd.Timestamp("2023-08-04")]})
    ev2 = fetch.official_corporate_events("X", px, known, "2023-01-01", {})
    assert not (ev2["date"] == "2023-08-04").any()           # 除權息結果表已有 → 不重複


def test_unexplained_gaps_only_reports():
    from src.fetch import unexplained_gaps
    p = _flat(6).assign(stock_id="X", spread=0.0)
    p.loc[3:, ["open", "high", "low", "close"]] = [97.0, 97.0, 97.0, 97.0]
    assert len(unexplained_gaps(p, _div(p["date"][3], 100.0, 97.0))) == 0
    assert len(unexplained_gaps(p, pd.DataFrame(columns=["date"]))) == 1


def test_report_available_dates():
    from src.fundamentals import report_available
    assert report_available(pd.Timestamp("2024-06-30"), False) == pd.Timestamp("2024-08-15")
    assert report_available(pd.Timestamp("2024-06-30"), True) == pd.Timestamp("2024-08-31")
    assert report_available(pd.Timestamp("2019-12-31"), False) == pd.Timestamp("2020-04-01")   # 閏年仍是 3/31 + 1
