import pandas as pd
import pytest

from src import fetch


class _Resp:
    def __init__(self, status_code, body):
        self.status_code, self._body, self.text = status_code, body, str(body)

    def json(self):
        return self._body


def _cfg(tmp_path):
    return {"data_dir": str(tmp_path), "finmind": {"url": "http://x", "timeout_seconds": 1,
                                                   "max_retries": 2, "retry_wait_seconds": 1}}


def _row(d, c=100.0):
    return {"date": d, "stock_id": "2330", "open": c, "max": c, "min": c, "close": c,
            "Trading_Volume": 1000, "spread": 0.0}


def test_cache_only_fetches_missing_range(tmp_path, monkeypatch):
    calls = []

    def fake_get(url, params, headers, timeout):
        calls.append((params["start_date"], params["end_date"]))
        days = pd.bdate_range(params["start_date"], params["end_date"])
        return _Resp(200, {"status": 200, "msg": "success", "data": [_row(d.strftime("%Y-%m-%d")) for d in days]})

    monkeypatch.setattr(fetch.requests, "get", fake_get)
    cfg = _cfg(tmp_path)
    df1 = fetch.fetch_prices("2330", "2024-01-08", "2024-01-12", cfg)
    assert len(df1) == 5 and calls == [("2024-01-08", "2024-01-12")]
    df2 = fetch.fetch_prices("2330", "2024-01-01", "2024-01-19", cfg)
    assert calls[1:] == [("2024-01-01", "2024-01-07"), ("2024-01-13", "2024-01-19")]
    assert len(df2) == 15
    fetch.fetch_prices("2330", "2024-01-03", "2024-01-17", cfg)
    assert len(calls) == 3  # 完全在快取內，不再請求


def test_holiday_at_start_is_not_refetched(tmp_path, monkeypatch):
    calls = []

    def fake_get(url, params, headers, timeout):
        calls.append(params["start_date"])
        days = pd.bdate_range("2024-01-02", params["end_date"])  # 2024-01-01 假日，無資料
        days = days[days >= params["start_date"]]
        return _Resp(200, {"status": 200, "data": [_row(d.strftime("%Y-%m-%d")) for d in days]})

    monkeypatch.setattr(fetch.requests, "get", fake_get)
    cfg = _cfg(tmp_path)
    fetch.fetch_prices("2330", "2024-01-01", "2024-01-05", cfg)
    fetch.fetch_prices("2330", "2024-01-01", "2024-01-05", cfg)
    assert calls == ["2024-01-01"]


def test_rate_limit_waits_and_retries(tmp_path, monkeypatch):
    seq = [_Resp(402, {"status": 402, "msg": "Requests reach the upper limit"}),
           _Resp(200, {"status": 200, "data": [_row("2024-01-02")]})]
    monkeypatch.setattr(fetch.requests, "get", lambda *a, **k: seq.pop(0))
    waits = []
    df = fetch.fetch_prices("2330", "2024-01-02", "2024-01-02", _cfg(tmp_path), sleep=waits.append)
    assert len(df) == 1 and waits == [1]


def test_rate_limit_exhausted_raises_clear_error(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch.requests, "get",
                        lambda *a, **k: _Resp(402, {"status": 402, "msg": "Requests reach the upper limit"}))
    monkeypatch.delenv("FINMIND_TOKEN", raising=False)
    import src.notify
    monkeypatch.setattr(src.notify, "_env", lambda n: "")   # 不讀本機真正設定的 token
    waits = []
    with pytest.raises(fetch.FinMindRateLimitError, match="已重試 2 次.*FINMIND_TOKEN"):
        fetch.fetch_prices("2330", "2024-01-02", "2024-01-02", _cfg(tmp_path), sleep=waits.append)
    assert waits == [1, 2]


def test_token_sent_only_as_header(tmp_path, monkeypatch):
    seen = {}

    def fake_get(url, params, headers, timeout):
        seen.update(params=params, headers=headers)
        return _Resp(200, {"status": 200, "data": [_row("2024-01-02")]})

    monkeypatch.setattr(fetch.requests, "get", fake_get)
    monkeypatch.setenv("FINMIND_TOKEN", "dummy-test-token")
    fetch.fetch_prices("2330", "2024-01-02", "2024-01-02", _cfg(tmp_path))
    assert seen["headers"]["Authorization"] == "Bearer dummy-test-token"
    assert "token" not in seen["params"]
    assert "dummy-test-token" not in "".join(p.read_text() for p in tmp_path.glob("*.csv"))


def test_missing_dates_against_calendar():
    df = pd.DataFrame({"date": pd.to_datetime(["2024-01-02", "2024-01-04"])})
    cal = pd.DatetimeIndex(pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]))
    miss, basis = fetch.missing_dates(df, "2024-01-01", "2024-01-05", cal)
    assert miss == [pd.Timestamp("2024-01-03")] and "交易日曆" in basis
