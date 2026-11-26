"""LINE 文字指令：設定提醒規則、查詢持股、記錄交易。只改 portfolio.yaml 與交易紀錄，沒有任何下單功能。

每次修改前把舊的 portfolio.yaml 存到 data/portfolio_undo.yaml，傳「復原」可以回到上一步。
"""
from __future__ import annotations

import re
import shutil
from dataclasses import dataclass

from . import portfolio as pfm
from . import users

HELP = """可用指令（數字都是你自己決定）：
【全體規則】
停損 10%｜停利 30%｜移動停損 15%
均線 60（收盤跌破 60 日線）｜權重上限 40%
組合回撤 15%｜接近 3%（「接近」提醒範圍）
【個股規則】（代號或名稱開頭）
聯電 停損價 65｜2303 停利價 90
鴻海 移動停損 20%｜2317 均線 20｜統一 權重上限 50%
【取消】取消 停損｜鴻海 取消 移動停損
【查詢】規則｜持股｜趨勢 2317｜體檢 2317（策略歷史表現）
【記帳】買 2317 1000 450｜賣 2303 1000 80（股數、價格）
【其他】復原（回到上一次修改前）｜說明"""

# 指令關鍵字 → (portfolio 欄位, 是否為百分比, 個股欄位)
RULES = {
    "移動停損": ("trailing_stop_pct", True, "trailing_stop_pct"),
    "停損價": (None, False, "stop_loss"),
    "停利價": (None, False, "take_profit"),
    "停損": ("stop_loss_pct", True, None),
    "停利": ("take_profit_pct", True, None),
    "權重上限": ("max_weight", True, "max_weight"),
    "組合回撤": ("portfolio_drawdown_pct", True, None),
    "均線": ("ma_break", False, "ma_break"),
    "接近": ("near_pct", True, None),
}
NEGATIVE = {"stop_loss_pct", "trailing_stop_pct", "portfolio_drawdown_pct"}
RULE_LABEL = {"stop_loss_pct": "停損（低於成本）", "take_profit_pct": "停利（高於成本）",
              "trailing_stop_pct": "移動停損（從高點回落）", "ma_break": "跌破均線", "max_weight": "單檔權重上限",
              "portfolio_drawdown_pct": "組合回撤", "near_pct": "接近提醒範圍", "stop_loss": "停損價",
              "take_profit": "停利價"}


@dataclass
class Result:
    reply: str
    changed: bool = False


def _backup(profile: users.Profile) -> None:
    if profile.portfolio_path.exists():
        profile.undo_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(profile.portfolio_path, profile.undo_path)


def _num(s: str) -> float:
    return float(s.replace(",", "").replace("％", "%").rstrip("%"))


def resolve_stock(token: str, pf: dict, cfg: dict, lookup=None) -> str | None:
    """代號直接用；名稱先比對持股，再查全市場名稱（完全相符才算）。"""
    token = token.strip()
    if re.fullmatch(r"\d{4,6}[A-Z]?", token):
        return token
    lookup = lookup or (lambda sid: pfm.stock_name(sid, cfg)[0])
    for h in pf["holdings"]:
        if lookup(h["stock_id"]) == token:
            return h["stock_id"]
    try:
        from .fetch import fetch_stock_info
        info = fetch_stock_info(cfg)
        hit = info[info["stock_name"] == token]
        return str(hit["stock_id"].iloc[0]) if len(hit) == 1 else None
    except Exception:
        return None


def _fmt_rule(key: str, v) -> str:
    if v is None:
        return "未設定"
    if key in ("stop_loss", "take_profit"):
        return f"{v:,.2f} 元"
    if key == "ma_break":
        return f"{int(v)} 日"
    return f"{abs(v):.0%}" if key != "near_pct" else f"{v:.1%}"


def rules_text(pf: dict, names: dict) -> str:
    d = pf["defaults"]
    lines = ["【全體規則】"]
    for k in ("stop_loss_pct", "take_profit_pct", "trailing_stop_pct", "ma_break", "max_weight",
              "portfolio_drawdown_pct", "near_pct"):
        lines.append(f"{RULE_LABEL[k]}：{_fmt_rule(k, d.get(k))}")
    per = []
    for h in pf["holdings"]:
        items = [f"{RULE_LABEL[k]} {_fmt_rule(k, h.get(k))}" for k in
                 ("stop_loss", "take_profit", "trailing_stop_pct", "ma_break", "max_weight") if h.get(k) is not None]
        if items:
            per.append(f"{h['stock_id']} {names.get(h['stock_id'], '')}：" + "、".join(items))
    lines += ["", "【個股規則】"] + (per or ["無"])
    return "\n".join(lines)


def holdings_text(pf: dict, cfg: dict) -> str:
    book = pfm.build_book(pf, cfg)
    if book.positions.empty:
        return "尚未輸入持股。"
    risk = pfm.risk_summary(book, cfg["backtest"]["trading_days_per_year"])
    alerts = pfm.evaluate_rules(pf, book, risk)
    pos = book.positions
    lines = [f"持股（{pos['date'].max():%Y-%m-%d} 收盤）", f"總資產 {book.total:,.0f}｜未實現 {pos['unrealized'].sum():+,.0f}"]
    for p in pos.itertuples():
        lines.append(f"{p.stock_id} {p.name} {p.close:,.2f}（{p.unrealized_pct:+.1%}）權重 {p.weight:.0%}")
    if len(alerts):
        lines += ["", "規則狀態："] + [f"{'⚠️觸發' if a.狀態 == '觸發' else '接近'} {a.stock_id} {a.規則}" for a in alerts.itertuples()]
    else:
        lines.append("沒有規則被觸發或接近觸發。")
    lines.append("※ 監控工具輸出，不是投資建議。")
    return "\n".join(lines)


def stock_analysis_text(kind: str, sid: str, cfg: dict) -> str:
    """「趨勢 代號」：趨勢分析；「體檢 代號」：策略體檢。只描述現況與歷史，不是建議。"""
    from .dashboard_data import load
    name = pfm.stock_name(sid, cfg)[0]
    if kind == "趨勢":
        from .trend import analyze_trend, trend_lines
        prices, _, adj = load(sid, "2018-01-01", cfg)
        if prices.empty:
            return f"找不到 {sid} 的資料。"
        t = analyze_trend(adj, prices, cfg["evaluation"]["holdout_start"])
        lines = [f"{sid} {name} 趨勢（{t['date']:%Y-%m-%d} 收盤 {t['close']:,.2f}）"] + trend_lines(t)
        return "\n".join(lines + ["※ 只描述現況，不是買賣建議。"])
    from .checkup import checkup_lines, run_checkup
    try:
        r = run_checkup(sid, cfg, n_random=cfg["dashboard"]["n_random"])
    except ValueError as e:
        return str(e)
    return f"{sid} {name}\n" + "\n".join(checkup_lines(r))


def handle(text: str, cfg: dict, profile: users.Profile, lookup=None, holdings_fn=holdings_text,
           analysis_fn=None) -> Result:
    """解析並執行一則指令。只有「設定 / 取消 / 記帳 / 復原」會改檔案。"""
    t = re.sub(r"\s+", " ", text.replace("　", " ")).strip()
    path = profile.portfolio_path
    pf = pfm.load_portfolio(path)
    names_fn = lookup or (lambda sid: pfm.stock_name(sid, cfg)[0])

    analysis_fn = analysis_fn or stock_analysis_text

    def save(pf_):
        pfm.save_portfolio(pf_, path)

    if t in ("說明", "help", "?", "？", "指令"):
        return Result(HELP)
    if t in ("規則", "rules"):
        return Result(rules_text(pf, {h["stock_id"]: names_fn(h["stock_id"]) for h in pf["holdings"]}))
    m = re.fullmatch(r"(趨勢|體檢) (\S+)", t)
    if m:
        sid = resolve_stock(m.group(2), pf, cfg, lookup)
        if sid is None:
            return Result(f"找不到股票「{m.group(2)}」，請改用代號。")
        return Result(analysis_fn(m.group(1), sid, cfg))
    if t in ("持股", "損益", "摘要"):
        return Result(holdings_fn(pf, cfg))
    if t == "復原":
        src = profile.undo_path
        if not src.exists():
            return Result("沒有可以復原的修改。")
        shutil.copyfile(src, path)
        src.unlink()
        return Result("已復原到上一次修改前。傳「規則」或「持股」確認。", changed=True)

    m = re.fullmatch(r"(買|賣|buy|sell) (\S+) ([\d,]+) ([\d.,]+)", t, re.I)
    if m:
        side = "buy" if m.group(1).lower() in ("買", "buy") else "sell"
        sid = resolve_stock(m.group(2), pf, cfg, lookup)
        if sid is None:
            return Result(f"找不到股票「{m.group(2)}」，請改用代號。")
        _backup(profile)
        try:
            r = pfm.record_trade(pf, cfg, side, sid, _num(m.group(3)), _num(m.group(4)),
                                 trades_file=profile.trades_path)
        except ValueError as e:
            return Result(f"沒有記錄：{e}")
        save(pf)
        tr, h = r["trade"], r["holding"]
        extra = f"，已實現損益 {tr['realized_pnl']:+,.0f}" if tr["realized_pnl"] is not None else ""
        return Result(f"已記錄{'買進' if side == 'buy' else '賣出'} {sid} {tr['shares']:,.0f} 股 @ {tr['price']:,.2f}"
                      f"（手續費 {tr['commission']}、稅 {tr['tax']}{extra}）\n目前 {h['shares']:,.0f} 股，平均成本 "
                      f"{h['avg_cost']:,.2f}\n記錯了傳「復原」", changed=True)

    # 規則：[股票] [取消] 關鍵字 [數值]
    keys = "|".join(sorted(RULES, key=len, reverse=True))
    m = re.fullmatch(rf"(?:(\S+) )?(取消 ?)?({keys})(?: ?([\d.,]+ ?[%％]?))?(?: ?日線| ?日)?", t)
    if not m:
        return Result("看不懂這個指令。傳「說明」看可用指令。")
    stock_tok, cancel, key, val = m.groups()
    if stock_tok == "取消":
        stock_tok, cancel = None, "取消"
    gkey, is_pct, hkey = RULES[key]
    if cancel is None and val is None:
        return Result(f"請加上數值，例如「{key} {'10%' if is_pct else '60'}」。")
    value = None
    if not cancel:
        value = _num(val)
        if is_pct:
            value = value / 100 if (val.strip().endswith(("%", "％")) or value > 1) else value
            if not 0 < value < 1:
                return Result("百分比要介於 0% 和 100% 之間。")
        if key == "均線" and not (2 <= value <= 480):
            return Result("均線天數要介於 2 和 480 之間。")
        if key in ("停損價", "停利價") and value <= 0:
            return Result("價格要大於 0。")

    if stock_tok:
        sid = resolve_stock(stock_tok, pf, cfg, lookup)
        h = next((x for x in pf["holdings"] if x["stock_id"] == sid), None) if sid else None
        if h is None:
            return Result(f"持股裡找不到「{stock_tok}」。個股規則只能設在持有的股票上。")
        if hkey is None:
            return Result(f"「{key}」只能設成全體規則，不能針對個股。")
        if hkey == "ma_break" and value is not None:
            value = int(value)
        _backup(profile)
        h[hkey] = -value if (value is not None and hkey == "trailing_stop_pct") else value
        save(pf)
        target = f"{sid} {names_fn(sid)}"
        label, shown = RULE_LABEL[hkey], _fmt_rule(hkey, h[hkey])
    else:
        if gkey is None:
            return Result(f"「{key}」要加上股票，例如「聯電 {key} 65」。")
        if gkey == "ma_break" and value is not None:
            value = int(value)
        _backup(profile)
        pf["defaults"][gkey] = -value if (value is not None and gkey in NEGATIVE) else value
        save(pf)
        target = "全體持股"
        label, shown = RULE_LABEL[gkey], _fmt_rule(gkey, pf["defaults"][gkey])
    verb = "已取消" if cancel else "已設定"
    return Result(f"{verb}：{target}的{label}" + ("" if cancel else f" = {shown}") +
                  "\n傳「規則」看全部，設錯了傳「復原」。", changed=True)
