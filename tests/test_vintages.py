"""基本面版本快照：只追加、同值不重複、數值改變保留舊列；游標輪替（不連網）。"""
import json

import pandas as pd

from src import vintages

DS = "TaiwanStockMonthRevenue"


def _cfg(tmp_path):
    return {"data_dir": str(tmp_path), "dashboard": {"watchlist": []}}


def _row(y, m, rev):
    return {"date": "2026-10-01", "stock_id": "2330", "revenue_year": y, "revenue_month": m, "revenue": rev}


def _read(tmp_path):
    return pd.read_csv(tmp_path / "vintages" / DS / "2330.csv", dtype=str)


def test_record_append_dedupe_and_change(tmp_path):
    cfg = _cfg(tmp_path)
    assert vintages.record(DS, "2330", [_row(2026, 9, 100), _row(2026, 8, 90)], cfg, today="2026-11-01") == 2
    assert vintages.record(DS, "2330", [_row(2026, 9, 100.0), _row(2026, 8, 90)], cfg, today="2026-11-02") == 0
    assert vintages.record(DS, "2330", [_row(2026, 9, 105)], cfg, today="2026-11-03") == 1
    df = _read(tmp_path)
    assert len(df) == 3
    assert list(df["first_seen"]) == ["2026-11-01", "2026-11-01", "2026-11-03"]
    assert vintages.record(DS, "2317", [_row(2026, 9, 100)], cfg, today="2026-11-03") == 1   # 別檔股票獨立


def test_record_missing_columns_or_unknown_dataset(tmp_path):
    cfg = _cfg(tmp_path)
    assert vintages.record(DS, "2330", [{"date": "2026-10-01"}], cfg) == 0
    assert vintages.record("Other", "2330", [_row(2026, 9, 1)], cfg) == 0
    assert vintages.record(DS, "2330", [], cfg) == 0


def test_refresh_batch_cursor_wraps(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    pd.DataFrame({"stock_id": ["1101", "1102", "1103"], "end": ["x"] * 3}).to_csv(tmp_path / "full_candidates.csv", index=False)
    import src.daily as daily
    import src.research as research
    monkeypatch.setattr(daily, "tracked_stocks", lambda pf, cfg: [])
    monkeypatch.setattr("src.portfolio.load_portfolio", lambda p: {"holdings": []})
    monkeypatch.setattr("src.users.owner_profile", lambda cfg: type("P", (), {"portfolio_path": None})())
    seen = []

    def fake(ds, sid, start, cfg, ttl_hours=12):
        seen.append((ds, sid))
        vintages.record(ds, sid, [_row(2026, 9, 1)] if ds == DS else [], cfg)
        return pd.DataFrame()

    monkeypatch.setattr(research, "fetch_dataset", fake)
    r = vintages.refresh_batch(cfg, 2)
    assert r == {"stocks": 2, "added": 2, "errors": 0}
    assert json.loads((tmp_path / "vintages" / "cursor.json").read_text())["cursor"] == 2
    seen.clear()
    vintages.refresh_batch(cfg, 2)
    assert sorted({s for _, s in seen}) == ["1101", "1103"]   # 繞回開頭
    assert json.loads((tmp_path / "vintages" / "cursor.json").read_text())["cursor"] == 1


def _setup_refresh(tmp_path, monkeypatch, fake):
    pd.DataFrame({"stock_id": ["1101", "1102", "1103", "1104"], "end": ["x"] * 4}).to_csv(tmp_path / "full_candidates.csv", index=False)
    import src.daily as daily
    import src.research as research
    monkeypatch.setattr(daily, "tracked_stocks", lambda pf, cfg: [])
    monkeypatch.setattr("src.portfolio.load_portfolio", lambda p: {"holdings": []})
    monkeypatch.setattr("src.users.owner_profile", lambda cfg: type("P", (), {"portfolio_path": None})())
    monkeypatch.setattr(research, "fetch_dataset", fake)


def test_refresh_batch_stops_on_finmind_error(tmp_path, monkeypatch):
    seen = []

    def fake(ds, sid, start, cfg, ttl_hours=12):
        seen.append(sid)
        if sid == "1102":
            raise vintages.FinMindError("rate limit")
        return pd.DataFrame()

    _setup_refresh(tmp_path, monkeypatch, fake)
    r = vintages.refresh_batch(_cfg(tmp_path), 3)
    assert r["stopped"] == "rate limit" and r["errors"] == 1
    assert set(seen) == {"1101", "1102"}   # 1103 沒有被抓
    assert json.loads((tmp_path / "vintages" / "cursor.json").read_text())["cursor"] == 1


def test_refresh_batch_time_limit(tmp_path, monkeypatch):
    seen = []
    clock = iter([0, 0, 10_000])   # 起算、第一檔前、第二檔前（已超過上限）

    def fake(ds, sid, start, cfg, ttl_hours=12):
        seen.append(sid)
        return pd.DataFrame()

    _setup_refresh(tmp_path, monkeypatch, fake)
    monkeypatch.setattr(vintages.time, "monotonic", lambda: next(clock))
    r = vintages.refresh_batch(_cfg(tmp_path), 3)
    assert r["stopped"] == "time"
    assert set(seen) == {"1101"}
    assert json.loads((tmp_path / "vintages" / "cursor.json").read_text())["cursor"] == 1
