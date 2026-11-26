"""LINE 推播、前瞻紀錄評分、每日警示去重。全部不連網。"""
import numpy as np
import pandas as pd

from src import daily, journal, notify, oracles


class _Resp:
    def __init__(self, code, body=None):
        self.status_code, self._b = code, body or {}

    def json(self):
        return self._b


def test_send_line_dry_run_without_credentials(monkeypatch, capsys):
    monkeypatch.delenv("LINE_CHANNEL_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("LINE_USER_ID", raising=False)
    monkeypatch.setattr(notify, "_env", lambda name: "")       # 不讀到本機真正設定的值
    assert notify.send_line("hi") is False
    assert "只印出不送出" in capsys.readouterr().out


def test_send_line_posts_and_hides_token_in_errors(monkeypatch):
    monkeypatch.setenv("LINE_CHANNEL_ACCESS_TOKEN", "secret-token")
    monkeypatch.setenv("LINE_USER_ID", "U" + "0123456789abcdef" * 2)
    seen = {}

    def post(url, json, headers, timeout):
        seen.update(url=url, json=json, headers=headers)
        return _Resp(200)

    assert notify.send_line("hello", post=post) is True
    assert seen["json"]["to"] == "U" + "0123456789abcdef" * 2 and seen["json"]["messages"][0]["text"] == "hello"
    assert seen["headers"]["Authorization"] == "Bearer secret-token"
    try:
        notify.send_line("x", post=lambda *a, **k: _Resp(401, {"message": "Authentication failed"}))
    except notify.NotifyError as e:
        assert "401" in str(e) and "secret-token" not in str(e)
    else:
        raise AssertionError("expected NotifyError")


def test_split_message_limits():
    parts = notify.split_message("\n".join(["x" * 100] * 120), limit=1000)
    assert all(len(p) <= 1000 for p in parts) and len(parts) <= 5


def test_new_triggers_dedup():
    alerts = pd.DataFrame([dict(stock_id="A", 規則="停損價", 狀態="觸發", 說明=""),
                           dict(stock_id="B", 規則="跌破 MA60", 狀態="觸發", 說明=""),
                           dict(stock_id="C", 規則="停利價", 狀態="接近", 說明="")])
    new = daily.new_triggers(alerts, {"triggered": ["A|停損價"]})
    assert list(new["stock_id"]) == ["B"]                 # A 昨天就觸發、C 只是接近
    assert len(daily.new_triggers(alerts, {})) == 2


def test_journal_scoring_against_known_future():
    prices = oracles.synthetic_prices(200, seed=9)
    d = prices["date"]
    c = prices["close"]
    log = pd.DataFrame([dict(date=d[100], stock_id="X", close=c[100], lgbm_p=0.7, lgbm_horizon=20, lgbm_hurdle=0.0,
                             lgbm_threshold=0.5, sig_ma_cross=1, sig_candle_volume=0, sig_lgbm=1,
                             r5_lo68=c[100] * 0.5, r5_hi68=c[100] * 1.5, r20_lo68=c[100] * 0.99,
                             r20_hi68=c[100] * 1.01, recorded_at=""),
                        dict(date=d[190], stock_id="X", close=c[190], lgbm_p=0.4, lgbm_horizon=20, lgbm_hurdle=0.0,
                             lgbm_threshold=0.5, sig_ma_cross=0, sig_candle_volume=0, sig_lgbm=0,
                             r5_lo68=0, r5_hi68=1e9, r20_lo68=0, r20_hi68=1e9, recorded_at="")])
    s = journal.score({}, log=log, loader=lambda sid: (prices, None, prices))
    f20 = c[120] / c[100] - 1
    assert s["fwd20"].iloc[0] == f20 and s["lgbm_outcome"].iloc[0] == float(f20 > 0)
    assert s["in_r5"].iloc[0] == 1.0
    assert s["in_r20"].iloc[0] == float(abs(f20) <= 0.01)
    assert np.isnan(s["lgbm_outcome"].iloc[1]) and np.isnan(s["fwd20"].iloc[1])   # 還沒到期
    assert s["in_r5"].iloc[1] == 1.0                                              # 第 195 天已揭曉
    summ = journal.summary(s)
    assert summ["lgbm_n"] == 1 and summ["n_records"] == 2


def test_summary_text_empty():
    assert "尚無資料" in journal.summary_text(journal.summary(pd.DataFrame()))
