"""策略實驗室：策略檔驗證、規則雜湊、select_spec、試驗帳本、截斷不變性。不連網。"""
import copy

import numpy as np
import pandas as pd
import pytest

from src import lab
from src.rank_strategy import StockData, Universe, factor_table, select, select_spec
from tests.test_rank import _universe

SPEC = dict(name="t", registered="2026-11-17", hypothesis="h", rationale="r", fails_if="f", universe="full",
            rebalance_days=63, top_n=2, lookback=126,
            filters=[{"factor": "mom63", "op": ">", "value": 0}, {"factor": "mom", "op": ">", "value": 0},
                     {"factor": "yoy3", "op": ">", "value": 0}, {"factor": "eps_ttm", "op": ">", "value": 0}],
            rank_by={"factor": "mom", "order": "desc"})


def _bad(**kw):
    s = copy.deepcopy(SPEC)
    for k, v in kw.items():
        if v is None:
            s.pop(k)
        else:
            s[k] = v
    return s


@pytest.mark.parametrize("kw,msg", [
    (dict(hypothesis=None), "缺少必填欄位：hypothesis"),
    (dict(filters=[{"factor": "roe", "op": ">", "value": 1}]), "因子 'roe' 不存在"),
    (dict(filters=[{"factor": "yoy3", "op": "==", "value": 1}]), "運算子 '==' 不合法"),
    (dict(filters=[{"factor": "yoy3", "op": ">"}]), "必須有 factor、op、value"),
    (dict(filters=[{"factor": "yoy3", "op": ">", "value": "高"}]), "value 必須是數字"),
    (dict(rank_by={"factor": "mom", "order": "up"}), "order 只能是 asc 或 desc"),
    (dict(rank_by={"factor": "pb", "order": "asc"}), "rank_by 的因子 'pb' 不存在"),
    (dict(top_n=0), "top_n 必須是正整數"),
    (dict(universe="tw"), "universe 只能是"),
])
def test_validate_spec_errors(kw, msg):
    with pytest.raises(lab.SpecError, match=msg):
        lab.validate_spec(_bad(**kw))


def test_validate_spec_ok_and_load(tmp_path):
    assert lab.validate_spec(copy.deepcopy(SPEC))["name"] == "t"
    p = tmp_path / "s.yaml"
    p.write_text("name: x\nregistered: 2026-11-17\n", encoding="utf-8")
    with pytest.raises(lab.SpecError, match="缺少必填欄位"):
        lab.load_spec(p)


def test_spec_hash_only_depends_on_rule_fields():
    h = lab.spec_hash(SPEC)
    assert len(h) == 8
    assert lab.spec_hash(_bad(name="另一個名字", hypothesis="別的假設", rationale="x", fails_if="y",
                              registered="2030-01-01")) == h
    assert lab.spec_hash(_bad(top_n=3)) != h
    assert lab.spec_hash(_bad(rank_by={"factor": "mom", "order": "asc"})) != h
    f2 = copy.deepcopy(SPEC["filters"])
    f2[0]["value"] = 0.1
    assert lab.spec_hash(_bad(filters=f2)) != h
    assert lab.spec_hash(_bad(top_n=2.0 * 1)) == h          # 2 與 2.0 正規化後相同


def test_neighbors_one_at_a_time_and_rounding():
    s = _bad(top_n=1, filters=[{"factor": "yoy3", "op": ">", "value": 0.1}, {"factor": "eps_ttm", "op": ">", "value": 0}])
    nb = dict(lab.neighbors(s))
    assert all(v["top_n"] >= 1 for v in nb.values())
    # top_n=1：×0.8 → 1（與原規則相同，略過）、×1.2 → 1（略過）；數值 0 的門檻不動
    assert not any(k.startswith("top_n") for k in nb)
    assert sum(k.startswith("yoy3") for k in nb) == 2
    assert not any(k.startswith("eps_ttm") for k in nb)
    for v in nb.values():
        diff = [k for k in ("rebalance_days", "top_n", "lookback", "filters") if v[k] != s[k]]
        assert len(diff) == 1                                # 一次只動一個


def _eq_spec(top_n):
    return _bad(top_n=top_n)


def test_select_spec_matches_select():
    rng = np.random.default_rng(1)
    f = pd.DataFrame({"stock_id": [f"S{i}" for i in range(60)], "close": rng.uniform(5, 100, 60),
                      "mom63": rng.normal(0, .2, 60), "mom": rng.normal(0, .3, 60), "yoy3": rng.normal(0, .3, 60),
                      "eps_ttm": rng.normal(1, 2, 60), "turnover": rng.uniform(1e6, 1e8, 60)})
    f.loc[3, "yoy3"] = np.nan
    for n in (1, 5, 10, 100):
        assert select_spec(f, _eq_spec(n), min_price=10, min_turnover=2e7) == select(f, n, 10, 2e7)
    assert select_spec(f.iloc[0:0], SPEC) == []
    # 升序與 >= / < 運算
    s = _bad(filters=[{"factor": "close", "op": "<=", "value": 50}], rank_by={"factor": "close", "order": "asc"}, top_n=3)
    got = select_spec(f, s, min_price=10, min_turnover=2e7)
    ok = f[(f["close"] > 10) & (f["turnover"] > 2e7) & (f["close"] <= 50)].sort_values("close")
    assert got == list(ok["stock_id"].head(3))


def test_select_spec_truncation_invariance():
    u = _universe(n_stocks=6, n_days=500)
    day = u.calendar[350]
    s = _bad(top_n=3, filters=[{"factor": "mom", "op": ">", "value": -1}])
    full = select_spec(factor_table(u, day, 126), s, min_price=0, min_turnover=0)
    cut = Universe({k: StockData(x.prices[x.prices.index <= day], x.adj_close[x.adj_close.index <= day], x.dividends,
                                 x.share_mult, x.revenue[x.revenue["available"] <= day],
                                 x.eps[x.eps["available"] <= day]) for k, x in u.stocks.items()},
                   u.calendar[u.calendar <= day])
    assert full and full == select_spec(factor_table(cut, day, 126), s, min_price=0, min_turnover=0)


def _row(name, h, sharpe="0.5"):
    return dict(time="2026-11-17T10:00:00", name=name, spec_hash=h, cagr="0.1", sharpe=sharpe, p_random_sharpe="0.01",
                p_random_cagr="0.01", verdict="不通過", git_commit="abc", data_hash="d")


def test_ledger_append_and_rule_count(tmp_path):
    path = tmp_path / "lab" / "trials.csv"
    led = lab.read_ledger(path)
    assert lab.count_rules(led, "aaaa1111") == 1             # 空帳本：只有本次
    lab.append_trial(path, _row("a", "aaaa1111"))
    lab.append_trial(path, _row("b", "bbbb2222"))
    lab.append_trial(path, _row("a-重跑", "aaaa1111", "0.6"))   # 同規則重跑
    led = lab.read_ledger(path)
    assert len(led) == 3 and list(led.columns) == lab.TRIAL_COLS
    assert lab.count_rules(led, "aaaa1111") == 2             # 同規則不增加
    assert lab.count_rules(led, "cccc3333") == 3             # 新規則 +1
    first_two = path.read_text(encoding="utf-8-sig").splitlines()[:3]
    lab.append_trial(path, _row("d", "dddd4444"))            # 只追加：舊列不變
    assert path.read_text(encoding="utf-8-sig").splitlines()[:3] == first_two


def test_bonferroni():
    assert lab.bonferroni(0.01, 3) == pytest.approx(0.03)
    assert lab.bonferroni(0.04, 5) == pytest.approx(0.2)
    assert lab.bonferroni(0.5, 10) == 1.0                    # 上限 1


def test_stock_contribution_by_stock():
    u = _universe(n_stocks=2, n_days=50)
    end = u.calendar[-1]
    last = {k: float(u.stocks[k].prices["close"].iloc[-1]) for k in u.stocks}
    t = pd.DataFrame([
        dict(side="buy", value=1000.0, shares=10, commission=5.0, tax=0.0, stock_id="S0"),
        dict(side="sell", value=1500.0, shares=10, commission=5.0, tax=4.0, stock_id="S0"),
        dict(side="buy", value=2000.0, shares=20, commission=5.0, tax=0.0, stock_id="S1")])
    class R: trades = t
    c = lab.stock_contribution(R, u, end)
    assert c["S0"] == pytest.approx(500 - 5 - 5 - 4)
    assert c["S1"] == pytest.approx(-2005 + 20 * last["S1"])
    assert list(c.index) == sorted(c.index, key=lambda k: -c[k])


_CFG = dict(rank=dict(min_price=10, min_turnover=1, full_min_turnover=2, start="2019-01-02", full_start="2017-07-03",
                      benchmark="0050"),
            costs=dict(commission_rate=0.001425, commission_discount=1.0, min_commission=20, sell_tax_rate=0.003,
                       price_limit=0.1, dividend_tax_rate=0.0),
            backtest=dict(initial_cash=5000000, max_volume_participation=0.05, volume_window=20,
                          trading_days_per_year=252),
            evaluation=dict(holdout_start="2025-01-01", universe=["2330"]))


def test_spec_hash_changes_with_config():
    base = lab.spec_hash(SPEC, _CFG)
    assert base == lab.spec_hash(SPEC, copy.deepcopy(_CFG))
    for path, v in ((("rank", "min_price"), 15), (("costs", "commission_discount"), 0.6),
                    (("backtest", "volume_window"), 30), (("backtest", "initial_cash"), 1),
                    (("evaluation", "holdout_start"), "2024-01-01")):
        c = copy.deepcopy(_CFG)
        c[path[0]][path[1]] = v
        assert lab.spec_hash(SPEC, c) != base, path


def test_ledger_old_file_without_n_random(tmp_path):
    p = tmp_path / "trials.csv"
    p.write_text("time,name,spec_hash,cagr,sharpe,p_random_sharpe,p_random_cagr,verdict,git_commit,data_hash\n"
                 "t0,old,aaaa1111,0.1,0.5,0.1,0.1,不通過,c,d\n", encoding="utf-8-sig")
    assert list(lab.read_ledger(p)["n_random"]) == [""]
    row = dict(time="t1", name="new", spec_hash="bbbb2222", cagr="0.2", sharpe="1", p_random_sharpe="0.01",
               p_random_cagr="0.01", verdict="通過", git_commit="c", data_hash="d", n_random="500")
    lab.append_trial(p, row)
    df = lab.read_ledger(p)
    assert list(df["spec_hash"]) == ["aaaa1111", "bbbb2222"]
    assert list(df["n_random"]) == ["", "500"]
    assert df.loc[0, "name"] == "old"


def test_factor_table_short_history_small_lookback():
    u = _universe(n_stocks=2, n_days=50)
    f = factor_table(u, u.calendar[-1], lookback=40)
    assert len(f) == 0


def test_pe_and_dy_factors(tmp_path):
    u = _universe(n_stocks=3, n_days=400)
    day = u.calendar[300]
    close = {k: float(s.prices.loc[day, "close"]) for k, s in u.stocks.items()}
    eps = lambda v: pd.DataFrame({"quarter_end": [pd.Timestamp("2019-12-31")], "eps": [1.0], "eps_ttm": [v],
                                  "available": [pd.Timestamp("2020-03-30")]})
    u.stocks["S0"].eps = eps(4.0)
    u.stocks["S1"].eps = eps(-1.0)
    u.stocks["S2"].eps = eps(0.0)
    u.stocks["S0"].dividends = pd.Series({day + pd.Timedelta(days=1): 9.0,      # day 之後：不計
                                          day - pd.Timedelta(days=365): 7.0,    # 恰 365 天前：不計
                                          day - pd.Timedelta(days=364): 1.0,
                                          day: 2.0})                            # 當天：計入
    f = factor_table(u, day).set_index("stock_id")
    assert f.loc["S0", "pe"] == pytest.approx(close["S0"] / 4.0)
    assert np.isnan(f.loc["S1", "pe"]) and np.isnan(f.loc["S2", "pe"])
    assert f.loc["S0", "dy"] == pytest.approx(3.0 / close["S0"])
    assert f.loc["S1", "dy"] == 0.0

    spec = copy.deepcopy(SPEC)
    spec["filters"] = [{"factor": "pe", "op": "<", "value": 1e9}, {"factor": "dy", "op": ">=", "value": 0}]
    spec["rank_by"] = {"factor": "dy", "order": "desc"}
    assert lab.validate_spec(spec)["name"] == "t"
    assert select_spec(f.reset_index(), spec, min_price=0, min_turnover=0) == ["S0"]


def test_dy_adjusts_for_share_changes():
    u = _universe(n_stocks=1, n_days=400)
    s = u.stocks["S0"]
    day = u.calendar[300]
    close = float(s.prices.loc[day, "close"])
    d = day - pd.Timedelta(days=30)
    s.dividends = pd.Series({d: 4.0})

    def dy(mult):
        s.share_mult = pd.Series(mult, dtype=float)
        return factor_table(u, day).set_index("stock_id").loc["S0", "dy"]

    assert dy({}) == pytest.approx(4.0 / close)
    assert dy({d + pd.Timedelta(days=5): 4.0}) == pytest.approx(1.0 / close)      # 除息後 1 拆 4
    assert dy({day + pd.Timedelta(days=1): 4.0}) == pytest.approx(4.0 / close)    # day 之後不影響
    assert dy({d: 4.0}) == pytest.approx(1.0 / close)                             # 同日除息與拆分
    assert dy({d - pd.Timedelta(days=1): 4.0}) == pytest.approx(4.0 / close)      # 除息前的拆分不影響


def test_pe_adjusts_for_share_changes():
    u = _universe(n_stocks=1, n_days=400)
    s = u.stocks["S0"]
    day = u.calendar[300]
    close = float(s.prices.loc[day, "close"])   # eps 季末 2019-12-31、eps_ttm 4.0

    def pe(mult):
        s.share_mult = pd.Series(mult, dtype=float)
        return factor_table(u, day).set_index("stock_id").loc["S0", "pe"]

    assert pe({}) == pytest.approx(close / 4.0)
    assert pe({pd.Timestamp("2020-02-01"): 0.5}) == pytest.approx(close / 8.0)    # 季末後減資，EPS 變兩倍
    assert pe({day: 0.5}) == pytest.approx(close / 8.0)                           # day 當天含
    assert pe({day + pd.Timedelta(days=1): 0.5}) == pytest.approx(close / 4.0)    # day 之後不影響
    assert pe({pd.Timestamp("2019-12-31"): 0.5}) == pytest.approx(close / 4.0)    # 季末當天不含
    assert factor_table(u, day).set_index("stock_id").loc["S0", "eps_ttm"] == 4.0


def test_pe_dy_nan_when_close_not_positive():
    u = _universe(n_stocks=1, n_days=400)
    s = u.stocks["S0"]
    day = u.calendar[300]
    s.prices = s.prices.copy()
    s.prices.loc[day, "close"] = 0.0
    s.dividends = pd.Series({day: 1.0})
    f = factor_table(u, day).set_index("stock_id")
    assert np.isnan(f.loc["S0", "pe"]) and np.isnan(f.loc["S0", "dy"])


# ---------------- AI 提出的策略檔 ----------------
def _ai(**kw):
    return _bad(author="ai", who_pays="過度悲觀的散戶", source="某文獻", family="value", **kw)


def test_author_validation():
    assert lab.validate_spec(copy.deepcopy(SPEC))["name"] == "t"        # 預設 user
    with pytest.raises(lab.SpecError, match="author 只能是"):
        lab.validate_spec(_bad(author="bot"))
    lab.validate_spec(_ai())


@pytest.mark.parametrize("k", ["who_pays", "source", "family"])
def test_ai_requires_fields(k):
    s = _ai()
    s.pop(k)
    with pytest.raises(lab.SpecError, match=k):
        lab.validate_spec(s)


def test_existing_spec_hashes_pinned():
    from src.config import load_config
    cfg = load_config()
    for n, h in (("retired_rank", "b4237c2e"), ("global_trend_momentum", "cbf75b0d"), ("cheap_growth", "5199915f"),
                  ("ai_illiquidity_quality", "26892b6e")):
        spec = lab.load_spec(lab.ROOT / "strategies" / "lab" / f"{n}.yaml")
        assert lab.spec_hash(spec, cfg)[:8] == h
    assert lab.spec_hash(_ai(), cfg) == lab.spec_hash(SPEC, cfg)       # 中繼資料不進雜湊


def _setup(tmp_path, rows):
    """rows: (name, author, family, verdict, time, spec_hash)；寫假帳本與假策略檔。"""
    sd = tmp_path / "specs"
    sd.mkdir()
    led = tmp_path / "lab" / "trials.csv"
    for name, author, fam, verdict, t, h in rows:
        (sd / f"{name}.yaml").write_text(f"author: {author}\nfamily: {fam}\n", encoding="utf-8")
        r = _row(name, h)
        r.update(time=t, verdict=verdict)
        lab.append_trial(led, r)
    return led, sd


TODAY = __import__("datetime").date(2026, 11, 26)


def test_ai_budget_exhausted_rejected(tmp_path):
    led, sd = _setup(tmp_path, [("a1", "ai", "f1", "通過", "2026-11-02T10:00:00", "h1"),
                                ("a2", "ai", "f2", "通過", "2026-11-03T10:00:00", "h2")])
    before = led.read_text(encoding="utf-8-sig")
    with pytest.raises(lab.LabRejected, match="額度已用完"):
        lab.check_ai_registration(_ai(), {**_CFG, "lab": {"ai_monthly_budget": 2}}, led, sd, TODAY)
    assert led.read_text(encoding="utf-8-sig") == before
    lab.check_ai_registration(_ai(), {**_CFG, "lab": {"ai_monthly_budget": 3}}, led, sd, TODAY)
    lab.check_ai_registration(_ai(), _CFG, led, sd, __import__("datetime").date(2026, 12, 1))   # 下個月重算


def test_failed_family_rejected(tmp_path):
    led, sd = _setup(tmp_path, [("a1", "ai", "value", "不通過", "2026-10-02T10:00:00", "h1")])
    with pytest.raises(lab.LabRejected, match="已被淘汰"):
        lab.check_ai_registration(_ai(), _CFG, led, sd, TODAY)
    s = _ai()
    s["family"] = "momentum"
    lab.check_ai_registration(s, _CFG, led, sd, TODAY)


def test_ai_rerun_same_rule_unrestricted(tmp_path):
    h = lab.spec_hash(_ai(), _CFG)
    led, sd = _setup(tmp_path, [("a1", "ai", "value", "不通過", "2026-11-02T10:00:00", h),
                                ("a2", "ai", "f2", "通過", "2026-11-03T10:00:00", "h2")])
    lab.check_ai_registration(_ai(), {**_CFG, "lab": {"ai_monthly_budget": 1}}, led, sd, TODAY)


def test_user_spec_not_limited(tmp_path):
    led, sd = _setup(tmp_path, [("a1", "ai", "value", "不通過", "2026-11-02T10:00:00", "h1"),
                                ("a2", "ai", "f2", "通過", "2026-11-03T10:00:00", "h2")])
    s = copy.deepcopy(SPEC)
    s["family"] = "value"
    lab.check_ai_registration(s, {**_CFG, "lab": {"ai_monthly_budget": 0}}, led, sd, TODAY)
    assert lab.ledger_authors(lab.read_ledger(led), sd) == (0, 2, 0)


def test_run_lab_name_must_match_filename(tmp_path):
    led = tmp_path / "trials.csv"
    with pytest.raises(lab.SpecError, match="一致"):
        lab.run_lab(_ai(), _CFG, spec_path=tmp_path / "ai_t.yaml")      # name 是 t
    assert not led.exists() and not list(tmp_path.iterdir())


def test_unreadable_spec_counts_as_ai(tmp_path):
    led, sd = _setup(tmp_path, [("a1", "ai", "f1", "通過", "2026-11-02T10:00:00", "h1")])
    lab.append_trial(led, {**_row("ghost", "h2"), "time": "2026-11-03T10:00:00"})   # 沒有 ghost.yaml
    assert lab.ledger_authors(lab.read_ledger(led), sd) == (0, 1, 1)
    with pytest.raises(lab.LabRejected, match="額度已用完"):
        lab.check_ai_registration(_ai(), {**_CFG, "lab": {"ai_monthly_budget": 2}}, led, sd, TODAY)


# ---- 小額帳戶情境 ----
def _sa_cfg(cash=20000, cmin=20):
    return dict(lab=dict(small_account=dict(initial_cash=cash, commission_min=cmin)))


def _sa_run(u, targets, cfg, cash0=5_000_000):
    from functools import partial
    from src.backtest import Costs
    from src.rank_strategy import run_portfolio
    from src.rank_study import _metrics
    costs = Costs()
    run = partial(run_portfolio, max_participation=None, volume_window=20)
    ws, we = u.calendar[0], u.calendar[-1]
    main = run(u, targets, costs, cash0, None, None)
    m = _metrics(main, ws, we, 252)
    L = lab.small_account_section(cfg, u, targets, run, costs, None, None, ws, we, 252, m, main)
    return L, m


def test_small_account_skipped_without_section():
    u = _universe(n_stocks=3, n_days=120)
    tg = {u.calendar[10]: list(u.stocks)[:2]}
    assert _sa_run(u, tg, dict(lab={}))[0] == []
    assert _sa_run(u, tg, {})[0] == []


def test_small_account_cost_ratio_higher_and_counts_unbuyable():
    u = _universe(n_stocks=3, n_days=120)
    ids = list(u.stocks)
    tg = {u.calendar[10]: ids[:2], u.calendar[60]: ids[1:]}
    L, m = _sa_run(u, tg, _sa_cfg())
    txt = "\n".join(L)
    assert "小額帳戶情境（本金 20,000 元、最低手續費 20 元；不影響判定）" in txt
    row = [x for x in L if x.startswith("| 成本佔平均淨值")][0]
    small = float(row.split("|")[2].split("%")[0])
    main = float(row.split("主回測")[1].split("%")[0])
    assert small > main
    # 本金極小：每檔分到的金額買不到 1 股 -> 全部 4 次都買不到
    L2, _ = _sa_run(u, tg, _sa_cfg(cash=10))
    assert any("| 4 / 4 次 |" in x for x in L2)
    L3, _ = _sa_run(u, tg, _sa_cfg(cash=10_000_000))
    assert any("| 0 / 4 次 |" in x for x in L3)
