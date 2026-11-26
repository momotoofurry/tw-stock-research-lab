"""每個交易日收盤後的例行工作（排程執行）：

1. 確認今天是交易日、資料已更新（沒更新就等一下再試）
2. 記錄今天的預測快照（前瞻紀錄），並對到期的舊預測評分
3. 產生持股摘要 reports/brief_YYYYMMDD.md
4. 檢查你設定的規則，只把「新觸發」（昨天沒觸發、今天觸發）推播到 LINE；可選每天一則摘要

只做監控與通知，不下單、不給買賣建議。
"""
from __future__ import annotations

import json
import time
from datetime import date, datetime

import pandas as pd

from . import journal
from . import portfolio as pfm
from .config import ROOT
from .fetch import FinMindError, fetch_prices, fetch_trading_calendar
from . import users
from .notify import NotifyError, line_configured, send_line, send_line_to


def _state_path(cfg):
    return ROOT / cfg["data_dir"] / "alert_state.json"


def _log(cfg, msg: str) -> None:
    p = ROOT / cfg["data_dir"] / "daily.log"
    p.parent.mkdir(parents=True, exist_ok=True)
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
    print(line)
    with open(p, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def tracked_stocks(pf: dict, cfg: dict) -> list[str]:
    held = [h["stock_id"] for h in pf["holdings"] if float(h.get("shares") or 0) > 0]
    return list(dict.fromkeys(held + [str(s) for s in cfg["dashboard"]["watchlist"]]))


def is_trading_day(day: date, cfg: dict) -> bool | None:
    cal = fetch_trading_calendar(day.isoformat(), day.isoformat(), cfg)
    if cal is None:
        return None if day.weekday() < 5 else False
    return pd.Timestamp(day) in set(cal)


def wait_for_data(probe: str, day: date, cfg: dict, tries: int = 4, wait_s: int = 900, sleep=time.sleep) -> bool:
    for k in range(tries):
        px = fetch_prices(probe, (pd.Timestamp(day) - pd.Timedelta(days=10)).strftime("%Y-%m-%d"), day.isoformat(), cfg,
                          refresh=True)
        if len(px) and px["date"].max().date() >= day:
            return True
        if k < tries - 1:
            _log(cfg, f"{day} 的收盤資料還沒出來，{wait_s // 60} 分鐘後再試（{k + 1}/{tries}）")
            sleep(wait_s)
    return False


def new_triggers(alerts: pd.DataFrame, prev: dict) -> pd.DataFrame:
    """只保留昨天沒有觸發、今天觸發的。"""
    if alerts.empty:
        return alerts
    trig = alerts[alerts["狀態"] == "觸發"]
    prev_keys = set(prev.get("triggered", []))
    return trig[[f"{r.stock_id}|{r.規則}" not in prev_keys for r in trig.itertuples()]]


def alert_message(day, alerts: pd.DataFrame, book) -> str:
    names = book.positions.set_index("stock_id")["name"].to_dict() if book is not None and len(book.positions) else {}
    lines = [f"【持股規則觸發】{day}", ""]
    for a in alerts.itertuples():
        lines.append(f"• {a.stock_id} {names.get(a.stock_id, '')}｜{a.規則}\n  {a.說明}")
    lines += ["", "※ 這是你自己設定的規則被觸發，不是買賣建議。"]
    return "\n".join(lines)


def summary_message(day, book, risk, alerts: pd.DataFrame, journal_text: str, market_lines=None) -> str:
    lines = [f"【收盤摘要】{day}"] + list(market_lines or [])
    if book is not None and len(book.positions):
        pos = book.positions
        day_pnl = float((pos["market_value"] * pos["day_chg"] / (1 + pos["day_chg"])).sum())
        lines.append(f"總資產 {book.total:,.0f}｜今日 {day_pnl:+,.0f}｜未實現 {pos['unrealized'].sum():+,.0f}")
        for p in pos.itertuples():
            lines.append(f"{p.stock_id} {p.name} {p.close:,.2f}（{p.day_chg:+.1%}）損益 {p.unrealized_pct:+.1%}")
        lines.append(f"組合距近一年高點 {risk['dd_now']:+.1%}，年化波動 {risk['vol']:.0%}")
        trig, near = (alerts["狀態"] == "觸發").sum(), (alerts["狀態"] == "接近").sum()
        lines.append(f"規則：觸發 {trig}、接近 {near}")
    else:
        lines.append("尚未輸入持股（可以傳「買 代號 股數 價格」記錄）。")
    lines += (["", journal_text] if journal_text else []) + ["", "※ 監控工具輸出，不是投資建議。"]
    return "\n".join(lines)


def _user_messages(day, profile, pf, cfg, jtext: str, mode: str, daily_summary: bool,
                   market_lines=None) -> tuple[list[str], list[str], int]:
    """回傳（要推給這個人的訊息、今天觸發的規則 key、新觸發數），並寫出他的摘要檔。"""
    book = risk = None
    alerts = pd.DataFrame(columns=["stock_id", "規則", "狀態", "說明"])
    if any(float(h.get("shares") or 0) > 0 for h in pf["holdings"]):
        book = pfm.build_book(pf, cfg)
        risk = pfm.risk_summary(book, cfg["backtest"]["trading_days_per_year"])
        alerts = pfm.evaluate_rules(pf, book, risk)
    brief = pfm.daily_brief(pf, cfg) + "\n\n## 前瞻紀錄\n\n" + jtext + "\n"
    out = ROOT / cfg["reports_dir"] / f"brief_{day:%Y%m%d}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(brief, encoding="utf-8")

    sp = profile.alert_state_path
    prev = json.loads(sp.read_text(encoding="utf-8")) if sp.exists() else {}
    to_send = new_triggers(alerts, prev) if mode == "new" else alerts[alerts["狀態"] == "觸發"]
    msgs = []
    if len(to_send):
        msgs.append(alert_message(day, to_send, book))
    if daily_summary:
        msgs.append(summary_message(day, book, risk, alerts, jtext, market_lines))
    if cfg.get("notify", {}).get("holdings_trend", True) and book is not None and len(book.positions):
        from .push import analysis_messages
        msgs += analysis_messages(list(book.positions["stock_id"]), cfg, title="持股走向")
    triggered = [f"{r.stock_id}|{r.規則}" for r in alerts.itertuples() if r.狀態 == "觸發"]
    sp.parent.mkdir(parents=True, exist_ok=True)
    sp.write_text(json.dumps({"day": str(day), "triggered": triggered}, ensure_ascii=False), encoding="utf-8")
    return msgs, triggered, len(new_triggers(alerts, prev))


def run_daily(cfg: dict, day: date | None = None, force: bool = False, notify: bool = True,
              sleep=time.sleep) -> dict:
    """收盤後例行工作：前瞻紀錄、檢查你的規則、寫摘要、推播。"""
    day = day or date.today()
    ncfg = cfg.get("notify", {})
    owner = users.owner_profile(cfg)
    pfs = {"owner": (owner, pfm.load_portfolio(owner.portfolio_path))}
    held = [h["stock_id"] for _, pf in pfs.values() for h in pf["holdings"] if float(h.get("shares") or 0) > 0]
    stocks = list(dict.fromkeys(held + [str(s) for s in cfg["dashboard"]["watchlist"]]))
    result = {"day": day, "sent": [], "messages": {}}
    try:
        trading = is_trading_day(day, cfg)
        if trading is False and not force:
            _log(cfg, f"{day} 不是交易日，略過")
            return result | {"status": "holiday"}
        if not force and stocks and not wait_for_data(stocks[0], day, cfg, sleep=sleep):
            _log(cfg, f"{day} 收盤資料一直沒更新，今天略過（明天會補記）")
            return result | {"status": "no_data"}

        added = journal.record(stocks, cfg)
        jtext = journal.summary_text(journal.summary(journal.score(cfg)))
        _log(cfg, f"前瞻紀錄新增 {len(added)} 筆；{jtext.splitlines()[0]}")
    except FinMindError as e:
        _log(cfg, f"失敗：{e}")
        if notify:
            try:
                send_line(f"【例行工作失敗】{day}\n{e}")
            except NotifyError:
                pass
        return result | {"status": "error", "error": str(e)}

    errors = []
    market_lines = []
    try:
        from . import market
        market_lines = market.short_lines(market.overview(cfg)) + [""]
    except Exception as e:  # 大盤資料抓不到不影響個人摘要
        errors.append(f"大盤概況：{type(e).__name__}")
    for profile, pf in pfs.values():
        try:
            msgs, triggered, n_new = _user_messages(day, profile, pf, cfg, jtext, ncfg.get("alert_mode", "new"),
                                                    ncfg.get("daily_summary", True), market_lines)
        except FinMindError as e:
            errors.append(f"{profile.name}：{e}")
            continue
        if cfg.get("llm", {}).get("enabled", False) and msgs:
            try:   # 本機 LLM 白話摘要：加在最前面，原訊息照常送出；不通過或出錯就只送原訊息
                from .llm_summary import HEADER, summarize
                s, why = summarize(cfg, "\n".join(msgs))
                if s:
                    msgs = [f"{HEADER}\n{s}"] + msgs
                else:
                    _log(cfg, f"[{profile.name}] 白話摘要略過：{why.split('：')[0]}")
            except Exception as e:
                _log(cfg, f"白話摘要失敗：{type(e).__name__}")
        result["messages"][profile.name] = msgs
        if notify and profile.user_id:
            for m in msgs:
                try:
                    result["sent"].append(send_line_to(profile.user_id, m))
                except NotifyError as e:
                    errors.append(f"{profile.name} 推播失敗：{e}")
                    break
        _log(cfg, f"[{profile.name}] 觸發 {len(triggered)}（新觸發 {n_new}），訊息 {len(msgs)} 則")
    # 每週檢討：在 notify.weekly_review_weekday（預設週五 = 4）收盤後多推一則預測檢討
    if day.weekday() == ncfg.get("weekly_review_weekday", 4):
        try:
            from .review import build_review, review_text, save_review
            rv = build_review(cfg)
            save_review(rv, cfg)
            text = review_text(rv, short=True)
            result["messages"]["weekly_review"] = [text]
            if notify:
                result["sent"].append(send_line(text))
            _log(cfg, "已推每週預測檢討")
        except (FinMindError, NotifyError) as e:
            errors.append(f"每週檢討：{e}")
    try:   # 基本面版本快照：每天輪流更新一批候選股，累積「當時看到的版本」
        from .vintages import refresh_batch
        vr = refresh_batch(cfg, cfg.get("vintages", {}).get("per_run", 100))
        _log(cfg, f"基本面快照：更新 {vr['stocks']} 檔、新增 {vr['added']} 筆"
                  + (f"，提前停止：{vr['stopped']}" if vr.get("stopped") else ""))
    except Exception as e:   # 快照失敗只記 log，不讓每日工作變 partial
        _log(cfg, f"基本面快照失敗：{type(e).__name__}")
    try:   # 前瞻模擬追蹤：補資料、記訊號、更新淨值；失敗只記 log，不影響 status / exit code
        from .track import update as track_update
        tr = track_update(cfg, day)
        if tr["updated"] or tr["stopped"] or tr["errors"]:
            _log(cfg, f"前瞻追蹤：更新 {len(tr['updated'])} 個策略、新增訊號 {tr['signals']} 列、淨值 {tr['nav']} 列"
                      + (f"；自動停止 {tr['stopped']}" if tr["stopped"] else "")
                      + (f"；錯誤 {tr['errors']}" if tr["errors"] else "")
                      + (f"；補資料 {tr['prefetch']}" if tr.get("prefetch") else ""))
    except Exception as e:
        _log(cfg, f"前瞻追蹤失敗：{type(e).__name__}")
    if day.weekday() == ncfg.get("weekly_review_weekday", 4):
        try:   # 每週檢討那天附上追蹤中策略的淨值摘要（在更新之後，才是最新的）；失敗只記 log
            from .track import summary_text as track_summary
            ttext = track_summary(cfg, day)
            if ttext:
                result["messages"]["track"] = [ttext]
                if notify:
                    result["sent"].append(send_line(ttext))
        except Exception as e:
            _log(cfg, f"前瞻追蹤摘要失敗：{type(e).__name__}")
        try:   # 自選股每週變化提醒（只描述資料變化）；失敗只記 log
            from .watch import check as watch_check
            wtext = watch_check(cfg, tracked_stocks(pfs["owner"][1], cfg))
            result["messages"]["watch"] = [wtext]
            if notify:
                result["sent"].append(send_line(wtext))
            _log(cfg, "已推自選股每週變化")
        except Exception as e:
            _log(cfg, f"自選股變化提醒失敗：{type(e).__name__}")
    if cfg.get("decisions", {}).get("reminders", True):
        try:   # 決策日誌到期提醒：每天檢查，有到期才推；失敗只記 log
            from .decisions import due_reminders
            dtext = due_reminders(cfg, day.isoformat())
            if dtext:
                result["messages"]["decision_due"] = [dtext]
                if notify:
                    result["sent"].append(send_line(dtext))
                _log(cfg, "已推決策日誌到期提醒")
        except Exception as e:
            _log(cfg, f"決策日誌到期提醒失敗：{type(e).__name__}")
    if notify and not line_configured():
        _log(cfg, "未設定 LINE 憑證：訊息只印出、沒有送出")
    for e in errors:
        _log(cfg, f"失敗：{e}")
    if errors and notify:
        try:
            send_line(f"【例行工作部分失敗】{day}\n" + "\n".join(errors))
        except NotifyError:
            pass
    return result | {"status": "ok" if not errors else "partial", "errors": errors}
