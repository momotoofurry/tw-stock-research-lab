"""自選股每週變化提醒：合成資料、不連網、不碰真實狀態檔。"""
import json

import numpy as np
import pandas as pd
import pytest

from src import watch

BANNED = ["買進", "賣出", "加碼", "減碼", "建議", "目標價", "應該"]


@pytest.fixture
def cfg(tmp_path):
    return dict(data_dir=str(tmp_path / "data"), reports_dir=str(tmp_path / "reports"),
                dashboard=dict(watchlist={"1111": "甲公司", "2222": "乙公司"}),
                watch=dict(high_low_window=250, chips_days=5, recent_days=5))


def make(rev_period="2026-10", yoy=0.1, eps_q="2026Q2", eps=1.0, phase="盤整 / 方向不明", support=(90.0,),
         resistance=(120.0,), closes=None, foreign5=1234.0):
    if closes is None:
        closes = np.concatenate([np.linspace(100, 110, 245), [108, 107, 109, 108, 108]])
    adj = pd.DataFrame({"date": pd.bdate_range("2025-01-01", periods=len(closes)), "close": closes})
    return dict(
        revenue=pd.DataFrame({"期間": [rev_period], "年增": [yoy]}),
        quarterly=pd.DataFrame({"EPS": [0.8, eps]}, index=[f"{int(eps_q[:4]) - 1}{eps_q[4:]}", eps_q]),
        adj=adj,
        trend=dict(phase=phase, close=float(closes[-1]), levels=dict(support=list(support),
                                                                     resistance=list(resistance))),
        chips=dict(inst={"外資": {1: 1.0, 5: foreign5, 20: 9.0}}))


def run(cfg, datas, write=True, stocks=("1111", "2222")):
    def g(sid, cfg):
        d = datas[sid]
        if isinstance(d, Exception):
            raise d
        return d
    return watch.check(cfg, list(stocks), write_state=write, gather_fn=g)


def body(text):
    return text.replace(watch.FOOTER, "")


def test_first_run_builds_baseline(cfg):
    t = run(cfg, {"1111": make(), "2222": make()})
    assert "已建立基準，下週開始比較" in t and watch.FOOTER in t
    st = json.loads(watch.state_path(cfg).read_text(encoding="utf-8"))
    assert set(st["stocks"]) == {"1111", "2222"} and st["stocks"]["1111"]["rev_period"] == "2026-10"


def test_dry_run_writes_nothing(cfg):
    run(cfg, {"1111": make(), "2222": make()}, write=False)
    assert not watch.state_path(cfg).exists()


def test_no_change_one_line(cfg):
    run(cfg, {"1111": make(), "2222": make()})
    t = run(cfg, {"1111": make(), "2222": make()})
    assert "本週追蹤的 2 檔都沒有新的變化" in t
    assert "外資" not in t and "•" not in t          # 沒有變化就不列任何項目


def test_detects_revenue_sign_flip_and_eps(cfg):
    run(cfg, {"1111": make(yoy=0.05), "2222": make()})
    t = run(cfg, {"1111": make(rev_period="2026-11", yoy=-0.08, eps_q="2026Q3", eps=1.2), "2222": make()})
    assert "2026-11" in t and "-8.0%" in t and "年增率轉負" in t
    assert "2026Q3 EPS 1.20" in t and "去年同季 0.80" in t and "較高" in t
    assert "1111 甲公司" in t and "2222" not in t.split("1111")[1]
    assert "外資近 5 日累計買賣超 +1,234 張" in t       # 有變化的股票附上外資數字
    run(cfg, {"1111": make(rev_period="2026-12", yoy=0.03), "2222": make()})   # 負 → 正
    st = json.loads(watch.state_path(cfg).read_text(encoding="utf-8"))
    assert st["stocks"]["1111"]["rev_period"] == "2026-12"


def test_revenue_flip_to_positive(cfg):
    run(cfg, {"1111": make(yoy=-0.05), "2222": make()})
    t = run(cfg, {"1111": make(rev_period="2026-11", yoy=0.02), "2222": make()})
    assert "年增率轉正" in t


def test_new_high_and_new_low(cfg):
    run(cfg, {"1111": make(), "2222": make()})
    up = np.concatenate([np.linspace(100, 110, 245), [111, 112, 113, 114, 115]])
    down = np.concatenate([np.linspace(100, 110, 245), [95, 94, 93, 92, 91]])
    t = run(cfg, {"1111": make(closes=up), "2222": make(closes=down)})
    a, b = t.split("2222")
    assert "新高" in a and "新低" not in a and "新低" in b


def test_phase_and_levels(cfg):
    run(cfg, {"1111": make(), "2222": make()})
    closes = np.concatenate([np.linspace(100, 110, 245), [108, 107, 106, 125, 125]])
    t = run(cfg, {"1111": make(phase="上升趨勢轉弱（跌破季線）", closes=np.r_[closes[:-2], 85, 85], support=(80.0,)),
                  "2222": make(closes=closes, resistance=(130.0,))})
    a, b = t.split("2222")
    assert "趨勢階段：盤整 / 方向不明 → 上升趨勢轉弱（跌破季線）" in a
    assert "跌破上次記錄的支撐 90.00" in a
    run(cfg, {"1111": make(), "2222": make()})                      # 重設狀態
    t = run(cfg, {"1111": make(), "2222": make(closes=np.r_[np.linspace(100, 110, 245), [108, 107, 106, 125, 125]])})
    assert "突破上次記錄的壓力 120.00" in t


def test_one_stock_failure_does_not_affect_others(cfg):
    run(cfg, {"1111": make(), "2222": make()})
    t = run(cfg, {"1111": RuntimeError("boom"), "2222": make(rev_period="2026-11")})
    assert "1111 甲公司：資料取得失敗" in t and "2026-11" in t
    st = json.loads(watch.state_path(cfg).read_text(encoding="utf-8"))
    assert st["stocks"]["1111"]["rev_period"] == "2026-10"          # 失敗那檔保留舊基準
    assert "boom" not in t


def test_new_stock_gets_baseline(cfg):
    run(cfg, {"1111": make()}, stocks=("1111",))
    t = run(cfg, {"1111": make(), "2222": make()})
    assert "新加入追蹤、已建立基準：2222 乙公司" in t


def test_no_banned_words(cfg):
    run(cfg, {"1111": make(), "2222": make()})
    up = np.concatenate([np.linspace(100, 110, 245), [111, 112, 113, 114, 125]])
    t = run(cfg, {"1111": make(rev_period="2026-11", yoy=-0.1, eps_q="2026Q3", eps=0.5, closes=up,
                               phase="下降趨勢中（價格 < 月線 < 季線 < 半年線，季線下彎）", support=(80.0,)),
                  "2222": RuntimeError("x")})
    assert "年增率轉負" in t and "新高" in t
    for w in BANNED:
        assert w not in body(t), w
    assert t.rstrip().endswith("只描述資料變化，不是投資建議。")
