"""前瞻模擬追蹤：登記 / 停止、規則被改自動停、訊號只追加與凍結、涵蓋率順延、截斷不變性、摘要警語。不連網。"""
import copy
import re

import numpy as np
import pandas as pd
import pytest
import yaml

from src import lab, track
from src.config import load_config
from src.rank_strategy import StockData, Universe
from tests.test_rank import _universe

SPEC = dict(name="trk", registered="2026-11-26", hypothesis="h", rationale="r", fails_if="f", universe="default",
            rebalance_days=60, top_n=2, lookback=126,
            filters=[{"factor": "mom", "op": ">", "value": -1}], rank_by={"factor": "mom", "order": "desc"})
FORBIDDEN = re.compile("買進|賣出|建議|目標價|下一步|推薦|應該")
START = 200       # 開始日在合成日曆的位置（要有足夠歷史算 lookback）


class Env:
    def __init__(self, tmp_path, monkeypatch):
        cfg = copy.deepcopy(load_config())
        cfg["data_dir"], cfg["reports_dir"] = str(tmp_path / "data"), str(tmp_path / "reports")
        cfg["rank"].update(min_price=0, min_turnover=0, full_min_turnover=0, benchmark="S0")
        cfg["evaluation"]["universe"] = {f"S{k}": "x" for k in range(5)}
        cfg["track"] = dict(min_coverage=0.95, max_defer_days=5, prefetch_days=5, prefetch_minutes=1,
                            random_groups=3, seed=1)
        self.cfg, self.tmp = cfg, tmp_path
        self.u = _universe(n_stocks=5, n_days=400)
        self.cal = self.u.calendar
        self.missing = {}                      # 模擬某檔某天沒有價格
        self.loads = []
        self.truncate = True
        self.hole = set()                      # 模擬 0050 快取缺的日子
        monkeypatch.setattr(track, "_calendar", lambda cfg, s, e: self.cal[(self.cal >= pd.Timestamp(s)) &
                                                                           (self.cal <= pd.Timestamp(e)) &
                                                                           ~self.cal.isin(sorted(self.hole))])
        monkeypatch.setattr(track, "_load", self._load)
        monkeypatch.setattr(track, "_is_trading_day", lambda cfg, d: pd.Timestamp(d) in set(self.cal))
        monkeypatch.setattr(track, "_price_dates", self._dates)
        monkeypatch.setattr(track, "_fetch_stock", lambda sid, end, cfg: None)
        self.path = tmp_path / "trk.yaml"
        self.write_spec(SPEC)

    def write_spec(self, spec):
        self.path.write_text(yaml.safe_dump(spec, allow_unicode=True), encoding="utf-8")

    def _dates(self, cfg, sid):
        d = self.u.stocks[sid].prices.index
        return d.difference(pd.DatetimeIndex(sorted(self.missing.get(sid, set()))))

    def _load(self, ids, start, end, cfg, ends=None, fundamentals=True):
        self.loads.append((tuple(ids), start, end, dict(ends or {})))
        e0 = pd.Timestamp(end)
        st = {}
        for s in ids:
            x = self.u.stocks[s]
            e = min(e0, pd.Timestamp(ends[s])) if ends and s in ends else e0     # 和真的 load_universe 一樣遵守 ends
            if self.truncate or e < e0:
                x = StockData(x.prices[x.prices.index <= e], x.adj_close[x.adj_close.index <= e], x.dividends,
                              x.share_mult, x.revenue[x.revenue["available"] <= e], x.eps[x.eps["available"] <= e])
            st[s] = x
        cal = pd.DatetimeIndex(sorted(set().union(*[set(x.prices.index) for x in st.values()])))
        if self.truncate:
            cal = cal[cal <= e0]
        return Universe(st, cal)

    def day(self, i):
        return self.cal[i]

    def start(self):
        return track.add(self.cfg, self.path, today=self.day(START))

    def sig_path(self):
        return track.track_dir(self.cfg) / "trk_signals.csv"

    def sig_text(self):
        return self.sig_path().read_text(encoding="utf-8-sig")

    def sigs(self):
        return pd.read_csv(self.sig_path(), dtype=str).fillna("")


@pytest.fixture
def env(tmp_path, monkeypatch):
    return Env(tmp_path, monkeypatch)


def test_add_duplicate_stop(env):
    row = env.start()
    assert row["spec_hash"] == lab.spec_hash(SPEC, env.cfg) and row["name"] == "trk"
    assert list(track.active(env.cfg)) == ["trk"]
    with pytest.raises(ValueError, match="已在追蹤中"):
        track.add(env.cfg, env.path)
    track.stop(env.cfg, "trk", today=env.day(250))
    assert track.active(env.cfg) == {}
    with pytest.raises(ValueError, match="不在追蹤中"):
        track.stop(env.cfg, "trk")
    with pytest.raises(ValueError, match="另存新檔"):
        track.add(env.cfg, env.path)                  # 已停止的名稱也不能覆蓋
    assert list(track.read_registry(env.cfg)["action"]) == ["add", "stop"]     # 只追加
    env.write_spec(dict(SPEC, filters="oops"))
    with pytest.raises(lab.SpecError):
        track.add(env.cfg, env.path)


def test_spec_modified_auto_stops(env):
    env.start()
    env.write_spec(dict(SPEC, top_n=3))
    r = track.update(env.cfg, env.day(205))
    assert r["stopped"] == ["trk"] and r["signals"] == 0
    assert track.active(env.cfg) == {}
    assert "策略檔被修改" in track.read_registry(env.cfg).iloc[-1]["note"]
    assert not env.sig_path().exists()
    assert "已自動停止" in track.summary_text(env.cfg, env.day(205))
    # 文字欄位（假設等）改動不影響規則雜湊
    spec2 = dict(SPEC, name="trk2", hypothesis="換一句話")
    env.write_spec(spec2)
    track.add(env.cfg, env.path, today=env.day(206))
    env.write_spec(dict(spec2, hypothesis="又換一句"))
    assert track.update(env.cfg, env.day(207))["stopped"] == []


def test_signals_append_only_and_idempotent(env):
    env.start()
    r = track.update(env.cfg, env.day(205))
    assert r["signals"] == 1 + 3 and r["errors"] == []
    first = env.sig_text()
    df = env.sigs()
    assert list(df["kind"]) == ["strategy", "random_1", "random_2", "random_3"]
    assert set(df["signal_date"]) == {f"{env.day(START):%Y-%m-%d}"}
    assert all(len(s.split(";")) == 2 for s in df["stocks"])           # 策略與隨機組同檔數
    assert track.update(env.cfg, env.day(205))["signals"] == 0          # 重跑同一天
    assert env.sig_text() == first
    track.update(env.cfg, env.day(262))                                 # 第二個訊號日（START + 60）
    second = env.sig_text()
    assert second.startswith(first) and len(second) > len(first)         # 只在後面追加
    df = env.sigs()
    assert list(df["signal_date"].unique()) == [f"{env.day(START):%Y-%m-%d}", f"{env.day(START + 60):%Y-%m-%d}"]
    # 隨機組固定 seed：同樣資料重新來過得到同樣的隨機選股
    a = df["stocks"].tolist()
    env.sig_path().unlink()
    track.update(env.cfg, env.day(262))
    assert env.sigs()["stocks"].tolist() == a


def test_coverage_defers_then_records(env):
    env.start()
    env.missing["S4"] = {env.day(i) for i in range(START, START + 4)}     # 4 天沒有 S4 的價格 → 涵蓋率 0.8
    assert track.update(env.cfg, env.day(START + 1))["signals"] == 0      # 資料還沒齊：先等
    assert not env.sig_path().exists()
    track.update(env.cfg, env.day(START + 6))
    df = env.sigs()
    assert set(df["signal_date"]) == {f"{env.day(START + 4):%Y-%m-%d}"}
    assert set(df["nominal_date"]) == {f"{env.day(START):%Y-%m-%d}"}
    assert "順延" in df.loc[0, "note"] and float(df.loc[0, "coverage"]) == 1.0


def test_coverage_too_low_records_skipped(env):
    env.start()
    env.missing["S4"] = {env.day(i) for i in range(START, START + 7)}     # 超過 max_defer_days = 5
    track.update(env.cfg, env.day(START + 8))
    df = env.sigs()
    assert list(df["kind"]) == ["skipped"] and "不記訊號" in df.loc[0, "note"]
    assert float(df.loc[0, "coverage"]) == pytest.approx(0.8)
    assert track.update(env.cfg, env.day(START + 8))["signals"] == 0       # 不會重複記
    assert track.update(env.cfg, env.day(START + 8))["nav"] == 0           # 沒有策略訊號就沒有淨值


def test_nav_uses_frozen_signals(env):
    env.start()
    track.update(env.cfg, env.day(205))
    track.update(env.cfg, env.day(215))
    sig0 = env.sig_text()
    navp = track.track_dir(env.cfg) / "trk.csv"
    nav0 = navp.read_text(encoding="utf-8-sig")
    nav = pd.read_csv(navp, dtype=str).fillna("")
    assert nav.loc[0, "date"] == f"{env.day(205):%Y-%m-%d}" and float(nav.loc[0, "nav_strategy"]) == 1.0
    assert len(nav) == 11 and nav["holdings"].iloc[-1]
    assert set(nav["restated_diff"].iloc[1:].astype(float).abs()) <= {0.0}     # 資料沒變：重算和當初記錄一致
    held = set(nav["holdings"].iloc[-1].split(";"))
    picked = set(env.sigs().loc[0, "stocks"].split(";"))
    assert held == picked
    # 之後資料（含基本面、價格）全變，已記的選股與淨值列都不會被改，只追加新的日期；並標示重算差
    env.u = _universe(n_stocks=5, n_days=400, seed=77)
    r = track.update(env.cfg, env.day(216))
    assert r["signals"] == 0 and r["nav"] == 1
    assert env.sig_text() == sig0
    assert navp.read_text(encoding="utf-8-sig").startswith(nav0)
    assert track.update(env.cfg, env.day(216))["nav"] == 0                  # 同一天不重複追加
    last = pd.read_csv(navp, dtype=str).iloc[-1]
    assert abs(float(last["restated_diff"])) > track.RESTATE_WARN
    assert "淨值曾因資料更正而重算，與當時紀錄差" in track.summary_text(env.cfg, env.day(216))


def test_late_recording_trades_after_record_day(env):
    env.start()
    track.update(env.cfg, env.day(START + 2))               # 訊號日 START，晚 2 天才記
    df = env.sigs()
    assert set(df["signal_date"]) == {f"{env.day(START):%Y-%m-%d}"}
    assert set(df["trade_key"]) == {f"{env.day(START + 2):%Y-%m-%d}"}
    assert all("FinMind" in n for n in df["note"])
    track.update(env.cfg, env.day(START + 10))
    nav = pd.read_csv(track.track_dir(env.cfg) / "trk.csv", dtype=str).fillna("")
    assert nav.loc[0, "date"] == f"{env.day(START + 2):%Y-%m-%d}"    # 淨值從記錄日開始，不回溯 START+1 的開盤
    assert nav.loc[0, "holdings"] == "" and nav.loc[1, "holdings"] != ""   # 記錄日之後第一個交易日才成交


def test_record_day_missing_from_calendar_still_trades_after_it(env):
    env.start()
    env.hole = {env.day(START + 5)}                         # 0050 快取還沒有記錄日當天的資料
    track.update(env.cfg, env.day(START + 5))                # 訊號日 START，記錄日 START + 5
    assert set(env.sigs()["trade_key"]) == {f"{env.day(START + 5):%Y-%m-%d}"}
    env.hole = set()
    track.update(env.cfg, env.day(START + 9))
    nav = pd.read_csv(track.track_dir(env.cfg) / "trk.csv", dtype=str).fillna("")
    assert nav.loc[0, "date"] == f"{env.day(START + 5):%Y-%m-%d}" and nav.loc[0, "holdings"] == ""
    assert nav.loc[1, "date"] == f"{env.day(START + 6):%Y-%m-%d}" and nav.loc[1, "holdings"] != ""


def test_append_refuses_mismatched_header(env, tmp_path):
    p = tmp_path / "x.csv"
    p.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="表頭"):
        track._append(p, ["a", "b", "c"], [dict(a=1, b=2, c=3)])
    assert p.read_text(encoding="utf-8") == "a,b\n1,2\n"
    track._append(p, ["a", "b"], [dict(a=3, b=4)])
    assert p.read_text(encoding="utf-8").splitlines()[-1] == "3,4"


def test_random_group_matches_overlap(env):
    prev_s = ["A", "B", "C", "D"]
    prev_r = ["R1", "R2", "R3", "R4"]
    pool = prev_r + ["R5", "R6", "R7", "R8", "R9", "A", "B"]
    for seed in range(20):
        rng = np.random.default_rng(seed)
        # 策略 4 檔全續抱：隨機組也得全部續抱上一期（重疊 4 檔）
        out = track._random_pick(env.day(200), env.day(260), prev_s, prev_r, list(prev_s), pool, rng)
        assert set(out) == set(prev_r)
        # 策略續抱 2 檔：隨機組至少續抱 2 檔、總檔數 4
        out = track._random_pick(env.day(200), env.day(260), prev_s, prev_r, ["A", "B", "E", "F"], pool + ["E", "F"], rng)
        assert len(out) == 4 and len(set(out) & set(prev_r)) >= 2
    # 沒有上一期：只看本期檔數
    assert len(track._random_pick(None, env.day(260), [], [], ["A", "B"], pool, np.random.default_rng(0))) == 2


def test_grid_not_shifted_by_late_calendar_fill(env):
    env.start()
    env.hole = {env.day(210)}                                # 0050 快取缺一天
    track.update(env.cfg, env.day(262))
    noms = list(env.sigs()["nominal_date"].unique())
    assert len(noms) == 2
    text = env.sig_text()
    env.hole = set()                                         # 之後補進缺的日子
    r = track.update(env.cfg, env.day(262))
    assert r["signals"] == 0 and env.sig_text() == text
    assert list(env.sigs()["nominal_date"].unique()) == noms


def test_full_universe_ends_only_for_delisted(env):
    spec = dict(SPEC, name="trkfull", universe="full")
    env.write_spec(spec)
    data = env.tmp / "data"
    data.mkdir(exist_ok=True)
    stale = f"{env.day(100):%Y-%m-%d}"                       # 名單是舊的：上市中股票的 end 都停在名單建立日
    pd.DataFrame({"stock_id": [f"S{k}" for k in range(5)] + ["X9"],
                  "end": [stale] * 5 + [f"{env.day(40):%Y-%m-%d}"]}).to_csv(data / "full_candidates.csv", index=False)
    ids, ends = track.candidates(env.cfg, spec, env.day(205))
    assert ends == {} and "X9" not in ids                    # 下市很久的不算候選；上市中的不放 end
    ids2, ends2 = track.candidates(env.cfg, spec, env.day(60))
    assert ends2 == {"X9": f"{env.day(40):%Y-%m-%d}"} or "X9" not in ids2
    track.add(env.cfg, env.path, today=env.day(START))
    track.update(env.cfg, env.day(205))
    sg = pd.read_csv(track.track_dir(env.cfg) / "trkfull_signals.csv", dtype=str).fillna("")
    assert sg.loc[0, "kind"] == "strategy" and sg.loc[0, "stocks"] != ""    # 選得到股
    assert env.loads[0][3] == {}
    assert track.delisted_ids(env.cfg) == {"X9"}


def test_data_gap_does_not_liquidate_but_official_delisting_does(env):
    env.start()
    track.update(env.cfg, env.day(205))
    pick = env.sigs().loc[0, "stocks"].split(";")[0]
    x = env.u.stocks[pick]
    cut = env.u.stocks[pick].prices[lambda p: p.index <= env.day(212)]
    st = dict(env.u.stocks)
    st[pick] = StockData(cut, x.adj_close[x.adj_close.index <= env.day(212)], x.dividends, x.share_mult, x.revenue, x.eps)
    env.u = Universe(st, env.u.calendar)
    navp = track.track_dir(env.cfg) / "trk.csv"
    track.update(env.cfg, env.day(220))
    assert pick in pd.read_csv(navp, dtype=str).fillna("").iloc[-1]["holdings"]     # 沒有官方下市：沿用最後收盤、仍持有
    navp.unlink()
    data = env.tmp / "data"
    ends = [f"{env.day(399):%Y-%m-%d}"] * 5
    ends[int(pick[1:])] = f"{env.day(212):%Y-%m-%d}"
    pd.DataFrame({"stock_id": [f"S{k}" for k in range(5)], "end": ends}).to_csv(data / "full_candidates.csv", index=False)
    track.update(env.cfg, env.day(221))
    assert pick not in pd.read_csv(navp, dtype=str).fillna("").iloc[-1]["holdings"]   # 官方下市：出清


def test_signal_truncation_invariance(env):
    spec = dict(SPEC, top_n=3)
    day = env.day(300)
    ids = [f"S{k}" for k in range(5)]
    env.truncate = True
    a = track.compute_signal(env.cfg, spec, day, ids, None)
    assert env.loads[-1][2] == f"{day:%Y-%m-%d}"       # 載入資料只到訊號日
    env.truncate = False                               # 載入器把訊號日之後的資料也給進來
    b = track.compute_signal(env.cfg, spec, day, ids, None)
    assert a == b and a[0]
    # 把訊號日之後的價格改掉，選股也不變
    st = {}
    for k, x in env.u.stocks.items():
        p = x.prices.copy()
        p.loc[p.index > day, ["open", "high", "low", "close"]] *= 3.0
        st[k] = StockData(p, p["close"], x.dividends, x.share_mult, x.revenue, x.eps)
    env.u = Universe(st, env.u.calendar)
    assert track.compute_signal(env.cfg, spec, day, ids, None) == a


def _manual_nav(env, n, bench_dip):
    rng = np.random.default_rng(0)
    nav = 1 + np.cumsum(rng.normal(0.001, 0.01, n))
    b = np.linspace(1, 1.2, n)
    if bench_dip:
        b[100:140] *= 0.8
    rows = [dict(date=f"{d:%Y-%m-%d}", nav_strategy=f"{x:.6f}", nav_0050=f"{y:.6f}", nav_random_median=f"{x * 0.99:.6f}",
                 holdings="S1;S2", recorded_at="t") for d, x, y in zip(pd.bdate_range("2027-01-04", periods=n), nav, b)]
    track._append(track.track_dir(env.cfg) / "trk.csv", track.NAV_COLS, rows)


def test_summary_insufficient_sample_and_wording(env):
    row = env.start()
    _manual_nav(env, 30, bench_dip=False)
    text = track.summary_text(env.cfg, env.day(260))
    assert "樣本不足，不能下結論" in text and "未經實驗室" in text
    assert "Sharpe" not in text and "S1、S2" in text and "累積報酬" in text
    assert not FORBIDDEN.search(text)
    assert "資料中斷超過 5 個交易日" in text and "bulk-fetch" in text and "S1、S2" in text.split("bulk-fetch")[1]
    lab.append_trial(track.ROOT / env.cfg["reports_dir"] / "lab" / "trials.csv",
                     dict(name="trk", spec_hash=row["spec_hash"], verdict="不通過"))
    assert "實驗室判定：不通過" in track.summary_text(env.cfg, env.day(260))
    assert "trk" in track.list_text(env.cfg, env.day(260))


def test_summary_no_gap_warning_when_data_current(env, monkeypatch):
    env.start()
    _manual_nav(env, 30, bench_dip=False)
    monkeypatch.setattr(track, "_price_dates", lambda cfg, sid: pd.bdate_range("2027-01-04", periods=40))
    assert "資料中斷" not in track.summary_text(env.cfg, env.day(260))
    monkeypatch.setattr(track, "_price_dates", lambda cfg, sid: pd.bdate_range("2027-01-04", periods=22)
                        if sid == "S2" else pd.bdate_range("2027-01-04", periods=40))
    text = track.summary_text(env.cfg, env.day(260))
    assert "資料中斷" in text and text.split("bulk-fetch")[1].count("S2") == 1 and "S1" not in text.split("bulk-fetch")[1]


def test_summary_enough_days_and_drawdown(env):
    env.start()
    _manual_nav(env, 300, bench_dip=True)
    text = track.summary_text(env.cfg, env.day(260))
    assert "樣本不足" not in text and "Sharpe" in text and not FORBIDDEN.search(text)
    # 滿 252 日但 0050 沒回檔：仍然不能下結論
    (track.track_dir(env.cfg) / "trk.csv").unlink()
    _manual_nav(env, 300, bench_dip=False)
    t2 = track.summary_text(env.cfg, env.day(260))
    assert "樣本不足，不能下結論" in t2 and "未曾從高點回檔" in t2 and "追蹤未滿" not in t2


def test_summary_empty_without_tracking(env):
    assert track.summary_text(env.cfg, env.day(260)) == ""
    r = track.update(env.cfg, env.day(260))
    assert r["updated"] == [] and r["errors"] == []


def test_update_error_isolated(env, monkeypatch):
    env.start()
    monkeypatch.setattr(track, "compute_signal", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    r = track.update(env.cfg, env.day(205))
    assert r["errors"] and "trk" in r["errors"][0] and r["signals"] == 0
