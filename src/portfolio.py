"""持股監控：損益、風險、使用者自訂規則的警示、情境試算、每日摘要。

只做監控與試算，不產生買賣建議，也沒有任何下單功能。所有警示都是「你設定的規則被觸發 / 接近觸發」。
持股資料存在專案根目錄的 portfolio.yaml（已 gitignore，不上傳）。
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .backtest import Costs
from .config import ROOT
from .dashboard_data import analyze, load, stock_name
from .fetch import FinMindError, _cache_path, _request

PORTFOLIO_PATH = ROOT / "portfolio.yaml"

RULE_KEYS = ["stop_loss_pct", "take_profit_pct", "trailing_stop_pct", "ma_break", "max_weight"]
DEFAULT_PORTFOLIO = {
    "cash": 0,
    "defaults": {
        "stop_loss_pct": None,          # 例如 -0.10：收盤低於成本 10% 觸發
        "take_profit_pct": None,        # 例如 0.30：收盤高於成本 30% 觸發
        "trailing_stop_pct": None,      # 例如 -0.15：從買進後最高收盤（還原權息）回落 15% 觸發
        "ma_break": None,               # 例如 60：收盤跌破 60 日均線觸發
        "max_weight": None,             # 例如 0.40：單檔佔總資產超過 40% 觸發
        "portfolio_drawdown_pct": None,  # 例如 -0.15：整體組合從近一年高點回落 15% 觸發
        "near_pct": 0.03,               # 距離觸發 3% 以內標為「接近」
    },
    "holdings": [],
}


# ---------------- 讀寫 ----------------
def load_portfolio(path: Path = PORTFOLIO_PATH) -> dict:
    if not path.exists():
        return json.loads(json.dumps(DEFAULT_PORTFOLIO))
    with open(path, encoding="utf-8") as f:
        pf = yaml.safe_load(f) or {}
    out = json.loads(json.dumps(DEFAULT_PORTFOLIO))
    out["cash"] = pf.get("cash", 0) or 0
    out["defaults"].update(pf.get("defaults") or {})
    out["holdings"] = [dict(h, stock_id=str(h["stock_id"])) for h in pf.get("holdings") or []]
    return out


def save_portfolio(pf: dict, path: Path = PORTFOLIO_PATH) -> None:
    header = ("# 我的持股與自訂規則（個人資料，已 gitignore，不會上傳）\n"
              "# 規則留空 = 不啟用。個股欄位（stop_loss / take_profit 為價格，其餘同 defaults）會覆寫 defaults。\n")
    with open(path, "w", encoding="utf-8") as f:
        f.write(header)
        yaml.safe_dump(pf, f, allow_unicode=True, sort_keys=False)


# ---------------- 部位與損益 ----------------
@dataclass
class Book:
    positions: pd.DataFrame
    cash: float
    total: float
    returns: pd.DataFrame       # 各持股日報酬（還原權息）
    prices: dict                # stock_id -> (prices, adj)


def build_book(pf: dict, cfg: dict, costs: Costs | None = None) -> Book:
    costs = costs or Costs.from_config(cfg)
    rows, rets, data = [], {}, {}
    start = (pd.Timestamp.today() - pd.Timedelta(days=365 * 9)).strftime("%Y-%m-%d")
    for h in pf["holdings"]:
        sid, shares = h["stock_id"], float(h.get("shares") or 0)
        if shares <= 0:
            continue
        prices, _, adj = load(sid, start, cfg)
        if prices.empty:
            raise FinMindError(f"{sid} 沒有價格資料")
        data[sid] = (prices, adj)
        name, industry = stock_name(sid, cfg)
        px = prices.set_index("date")["close"]
        ac = adj.set_index("date")["close"]
        close = float(px.iloc[-1])
        avg_cost = float(h.get("avg_cost") or 0)
        mv = shares * close
        sell_cost = costs.commission(mv) + costs.sell_tax(mv)
        cost_value = shares * avg_cost
        buy_date = pd.Timestamp(h["buy_date"]) if h.get("buy_date") else None
        since = ac[ac.index >= buy_date] if buy_date is not None else ac.iloc[-252:]
        rows.append(dict(
            stock_id=sid, name=name, industry=industry, shares=shares, avg_cost=avg_cost, close=close,
            date=px.index[-1], day_chg=float(ac.iloc[-1] / ac.iloc[-2] - 1) if len(ac) > 1 else np.nan,
            market_value=mv, cost_value=cost_value, unrealized=mv - cost_value,
            unrealized_pct=(close / avg_cost - 1) if avg_cost else np.nan,
            net_if_sold=mv - sell_cost - cost_value, est_sell_cost=sell_cost,
            high_since=float(since.max()), dd_from_high=float(ac.iloc[-1] / since.max() - 1),
            high_basis="買進日起" if buy_date is not None else "近一年",
            **{f"ma{n}": float(px.rolling(n).mean().iloc[-1]) if len(px) >= n else np.nan for n in (20, 60, 120, 240)},
        ))
        rets[sid] = ac.pct_change()
    pos = pd.DataFrame(rows)
    cash = float(pf.get("cash") or 0)
    total = float(pos["market_value"].sum() if len(pos) else 0) + cash
    if len(pos):
        pos["weight"] = pos["market_value"] / total
    return Book(pos, cash, total, pd.DataFrame(rets).sort_index(), data)


# ---------------- 風險 ----------------
def risk_summary(book: Book, trading_days: int = 252) -> dict:
    """以「目前持股權重固定」回推的組合風險（描述性，用全部可得歷史）。"""
    pos = book.positions
    if pos.empty:
        return {}
    w = pos.set_index("stock_id")["market_value"] / book.total   # 現金報酬視為 0
    r = book.returns[w.index].dropna()
    port = (r * w).sum(axis=1)
    last_year = port.iloc[-trading_days:]
    eq = (1 + last_year).cumprod()
    roll5 = (1 + port).rolling(5).apply(np.prod, raw=True) - 1
    roll20 = (1 + port).rolling(20).apply(np.prod, raw=True) - 1
    worst5_date, worst20_date = roll5.idxmin(), roll20.idxmin()
    stock_w = w / w.sum()
    return dict(
        vol=float(last_year.std() * np.sqrt(trading_days)),
        var95_1d=float(-np.quantile(last_year, 0.05) * book.total),
        mdd_1y=float((eq / eq.cummax() - 1).min()),
        dd_now=float(eq.iloc[-1] / eq.max() - 1),
        worst5=float(roll5.min()), worst5_date=worst5_date,
        worst20=float(roll20.min()), worst20_date=worst20_date,
        history_start=r.index[0],
        max_weight=float(w.max()), max_weight_id=str(w.idxmax()),
        effective_n=float(1 / (stock_w ** 2).sum()),
        corr=r.iloc[-trading_days:].corr(),
        industry=pos.groupby("industry")["market_value"].sum() / book.total,
        cash_weight=book.cash / book.total if book.total else 0.0,
    )


def scenario(book: Book, shocks: dict[str, float]) -> tuple[pd.DataFrame, float, float]:
    """情境試算：各持股價格變動 shocks[stock_id]（例如 -0.2），回傳每檔與總資產的變化。"""
    pos = book.positions.copy()
    pos["shock"] = pos["stock_id"].map(shocks).fillna(0.0)
    pos["change"] = pos["market_value"] * pos["shock"]
    pos["new_value"] = pos["market_value"] + pos["change"]
    total_change = float(pos["change"].sum())
    out = pos[["stock_id", "name", "market_value", "shock", "change", "new_value"]]
    return out, total_change, total_change / book.total if book.total else 0.0


# ---------------- 自訂規則 ----------------
def _rule(h: dict, defaults: dict, key: str):
    v = h.get(key)
    return defaults.get(key) if v is None else v


def evaluate_rules(pf: dict, book: Book, risk: dict | None = None) -> pd.DataFrame:
    """回傳 [stock_id, 規則, 狀態(觸發/接近), 說明]。只描述你設定的規則，不是建議。"""
    d = pf["defaults"]
    near = float(d.get("near_pct") or 0.03)
    holdings = {h["stock_id"]: h for h in pf["holdings"]}
    out = []

    def add(sid, rule, gap, msg):
        # gap：距離觸發的比例，<= 0 已觸發，0 < gap <= near 接近
        if gap <= 0:
            out.append(dict(stock_id=sid, 規則=rule, 狀態="觸發", 說明=msg))
        elif gap <= near:
            out.append(dict(stock_id=sid, 規則=rule, 狀態="接近", 說明=msg + f"（距離觸發 {gap:.1%}）"))

    for p in book.positions.itertuples():
        h = holdings[p.stock_id]
        c = p.close
        stop = h.get("stop_loss")
        if stop is None and d.get("stop_loss_pct") is not None and p.avg_cost:
            stop = p.avg_cost * (1 - abs(d["stop_loss_pct"]))
        if stop:
            add(p.stock_id, "停損價", c / stop - 1, f"收盤 {c:,.2f}，你設定的停損價 {stop:,.2f}")
        tp = h.get("take_profit")
        if tp is None and d.get("take_profit_pct") is not None and p.avg_cost:
            tp = p.avg_cost * (1 + abs(d["take_profit_pct"]))
        if tp:
            add(p.stock_id, "停利價", 1 - c / tp, f"收盤 {c:,.2f}，你設定的停利價 {tp:,.2f}")
        ts = _rule(h, d, "trailing_stop_pct")
        if ts is not None:
            ts = -abs(ts)
            add(p.stock_id, "移動停損", p.dd_from_high - ts,
                f"從{p.high_basis}最高收盤（還原權息）回落 {-p.dd_from_high:.1%}，你設定回落 {-ts:.0%} 觸發")
        ma = _rule(h, d, "ma_break")
        if ma:
            mv = getattr(p, f"ma{int(ma)}", np.nan) if int(ma) in (20, 60, 120, 240) else np.nan
            if np.isnan(mv):
                prices, _ = book.prices[p.stock_id]
                mv = float(prices["close"].rolling(int(ma)).mean().iloc[-1])
            add(p.stock_id, f"跌破 MA{int(ma)}", c / mv - 1, f"收盤 {c:,.2f}，{int(ma)} 日均線 {mv:,.2f}")
        mw = _rule(h, d, "max_weight")
        if mw:
            add(p.stock_id, "單檔權重上限", mw - p.weight, f"佔總資產 {p.weight:.1%}，你設定上限 {mw:.0%}")
    pdd = d.get("portfolio_drawdown_pct")
    if pdd is not None and risk:
        pdd = -abs(pdd)
        add("組合", "組合回撤", risk["dd_now"] - pdd,
            f"組合從近一年高點回落 {-risk['dd_now']:.1%}，你設定回落 {-pdd:.0%} 觸發")
    return pd.DataFrame(out, columns=["stock_id", "規則", "狀態", "說明"])


# ---------------- 除權息行事曆 ----------------
def upcoming_dividends(stock_ids: list[str], cfg: dict, days: int = 90) -> pd.DataFrame:
    """已公告、尚未發生的除權息日（TaiwanStockDividend），快取一天。"""
    today = pd.Timestamp(date.today())
    rows = []
    for sid in stock_ids:
        path = _cache_path(sid, cfg["data_dir"], "TaiwanStockDividend")
        if path.exists() and time.time() - path.stat().st_mtime < 86400:
            data = json.loads(path.read_text(encoding="utf-8"))
        else:
            try:
                data = _request("TaiwanStockDividend", {"data_id": sid,
                                                        "start_date": (today - pd.Timedelta(days=400)).strftime("%Y-%m-%d")}, cfg)
            except FinMindError:
                continue
            path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        for r in data:
            for kind, dkey, amt in (("除息", "CashExDividendTradingDate", r.get("CashEarningsDistribution", 0)),
                                    ("除權", "StockExDividendTradingDate", r.get("StockEarningsDistribution", 0))):
                dt = r.get(dkey)
                if dt and today <= pd.Timestamp(dt) <= today + pd.Timedelta(days=days):
                    rows.append(dict(stock_id=sid, 類型=kind, 日期=pd.Timestamp(dt).date(),
                                     每股=float(amt or 0), 發放日=r.get("CashDividendPaymentDate", "")))
    return pd.DataFrame(rows, columns=["stock_id", "類型", "日期", "每股", "發放日"]).drop_duplicates()


# ---------------- 每日摘要 ----------------
def daily_brief(pf: dict, cfg: dict) -> str:
    if not any(float(h.get("shares") or 0) > 0 for h in pf["holdings"]):
        return ("# 每日持股摘要\n\n尚未輸入持股。請在看盤介面「我的持股」分頁，或直接編輯 portfolio.yaml，"
                "填入股數與平均成本。\n")
    book = build_book(pf, cfg)
    risk = risk_summary(book, cfg["backtest"]["trading_days_per_year"])
    alerts = evaluate_rules(pf, book, risk)
    pos = book.positions
    hs = cfg["evaluation"]["holdout_start"]
    last = pos["date"].max().date()
    day_pnl = float((pos["market_value"] * pos["day_chg"] / (1 + pos["day_chg"])).sum())
    lines = [
        f"# 每日持股摘要（{last}）",
        "",
        "> 監控與試算工具的輸出，不是投資建議。警示只代表「你自己設定的規則」被觸發或接近觸發。",
        "",
        f"- 總資產 **{book.total:,.0f}** 元（股票 {book.total - book.cash:,.0f}、現金 {book.cash:,.0f}）",
        f"- 今日損益約 {day_pnl:+,.0f} 元；未實現損益 {pos['unrealized'].sum():+,.0f} 元"
        f"（若全部賣出，扣手續費與證交稅約 {pos['est_sell_cost'].sum():,.0f} 元後為 {pos['net_if_sold'].sum():+,.0f} 元）",
        "",
        "## 警示（你設定的規則）",
        "",
    ]
    if alerts.empty:
        lines.append("沒有規則被觸發或接近觸發。" if _has_rules(pf) else "尚未設定任何規則。")
    else:
        lines += ["| 股票 | 規則 | 狀態 | 說明 |", "|---|---|---|---|"]
        lines += [f"| {a.stock_id} | {a.規則} | {'**觸發**' if a.狀態 == '觸發' else '接近'} | {a.說明} |"
                  for a in alerts.itertuples()]
    lines += ["", "## 持股", "",
              "| 股票 | 股數 | 成本 | 收盤 | 今日 | 損益% | 市值 | 權重 | 距高點 |",
              "|---|---|---|---|---|---|---|---|---|"]
    for p in pos.itertuples():
        lines.append(f"| {p.stock_id} {p.name} | {p.shares:,.0f} | {p.avg_cost:,.2f} | {p.close:,.2f} | "
                     f"{p.day_chg:+.1%} | {p.unrealized_pct:+.1%} | {p.market_value:,.0f} | {p.weight:.1%} | "
                     f"{p.dd_from_high:+.1%} |")
    lines += ["", "## 風險（以目前權重回推）", "",
              f"- 年化波動率 {risk['vol']:.0%}；單日 95% VaR 約 {risk['var95_1d']:,.0f} 元（100 天約有 5 天虧損超過此數）",
              f"- 近一年最大回撤 {risk['mdd_1y']:.1%}，目前距近一年高點 {risk['dd_now']:+.1%}",
              f"- {risk['history_start']:%Y} 年以來最差 5 日 {risk['worst5']:.1%}（{risk['worst5_date']:%Y-%m-%d}，"
              f"約 {risk['worst5'] * (book.total - book.cash):,.0f} 元）、最差 20 日 {risk['worst20']:.1%}"
              f"（{risk['worst20_date']:%Y-%m-%d}）",
              f"- 集中度：最大單檔 {risk['max_weight_id']} 佔 {risk['max_weight']:.0%}；"
              f"等效持股檔數 {risk['effective_n']:.1f}（越接近 1 越集中）",
              "- 產業：" + "、".join(f"{k} {v:.0%}" for k, v in risk["industry"].items())]
    div = upcoming_dividends(list(pos["stock_id"]), cfg)
    lines += ["", "## 近 90 天除權息", ""]
    if div.empty:
        lines.append("無已公告的除權息。")
    else:
        sh = pos.set_index("stock_id")["shares"]
        for r in div.itertuples():
            est = f"，以目前股數估計約 {r.每股 * sh[r.stock_id]:,.0f} 元" if r.類型 == "除息" else ""
            lines.append(f"- {r.stock_id} {r.日期} {r.類型}，每股 {r.每股:.4g}{est}")
    lines += ["", "## 各持股技術面", ""]
    for sid, (prices, adj) in book.prices.items():
        a = analyze(prices, adj, hs, cfg["backtest"]["trading_days_per_year"])
        nm = pos.set_index("stock_id").loc[sid, "name"]
        lines.append(f"**{sid} {nm}**")
        lines += [f"- {n}" for n in a["notes"]]
        r5 = a["ranges"].iloc[0]
        lines.append(f"- 未來 5 日 68% 波動範圍：{r5['下緣_68']:,.1f} ~ {r5['上緣_68']:,.1f}（統計區間，不是方向預測）")
        lines.append("")
    return "\n".join(lines)


def _has_rules(pf: dict) -> bool:
    d = pf["defaults"]
    if any(d.get(k) is not None for k in RULE_KEYS + ["portfolio_drawdown_pct"]):
        return True
    return any(h.get(k) is not None for h in pf["holdings"] for k in RULE_KEYS + ["stop_loss", "take_profit"])


# ---------------- 交易紀錄（對話式記帳用） ----------------
TRADE_COLUMNS = ["date", "stock_id", "side", "shares", "price", "commission", "tax", "realized_pnl", "note"]


def trades_path(cfg: dict) -> Path:
    d = Path(cfg["data_dir"])
    return (d if d.is_absolute() else ROOT / d) / "trades.csv"


def record_trade(pf: dict, cfg: dict, side: str, stock_id: str, shares: float, price: float,
                 day: str | None = None, note: str = "", costs: Costs | None = None,
                 trades_file: Path | None = None) -> dict:
    """依實際成交更新持股（加權平均成本不含手續費），並把這筆交易附加到 data/trades.csv。

    賣出：已實現損益 = (賣價 − 平均成本) × 股數 − 手續費 − 證交稅（買進手續費不計入成本，另記在買進那筆）。
    回傳 {"holding": 更新後的持股, "trade": 交易紀錄}。只更新 pf，存檔由呼叫端決定。
    """
    if side not in ("buy", "sell"):
        raise ValueError("side 必須是 buy 或 sell")
    if shares <= 0 or price <= 0:
        raise ValueError("股數與價格必須 > 0")
    costs = costs or Costs.from_config(cfg)
    day = day or date.today().isoformat()
    stock_id = str(stock_id)
    h = next((x for x in pf["holdings"] if x["stock_id"] == stock_id), None)
    if h is None:
        h = dict(stock_id=stock_id, shares=0.0, avg_cost=0.0, buy_date=None, stop_loss=None, take_profit=None,
                 trailing_stop_pct=None, ma_break=None, max_weight=None, note="")
        pf["holdings"].append(h)
    held, avg = float(h.get("shares") or 0), float(h.get("avg_cost") or 0)
    value = shares * price
    fee = costs.commission(value)
    tax = costs.sell_tax(value) if side == "sell" else 0
    realized = None
    if side == "buy":
        h["avg_cost"] = round((held * avg + value) / (held + shares), 4)
        h["shares"] = held + shares
        if held == 0:
            h["buy_date"] = day
    else:
        if shares > held:
            raise ValueError(f"{stock_id} 只持有 {held:,.0f} 股，不能賣 {shares:,.0f} 股")
        realized = (price - avg) * shares - fee - tax
        h["shares"] = held - shares
        if h["shares"] == 0:
            h["avg_cost"], h["buy_date"] = 0.0, None
    trade = dict(date=day, stock_id=stock_id, side=side, shares=shares, price=price, commission=fee, tax=tax,
                 realized_pnl=realized, note=note)
    p = trades_file or trades_path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    new = not p.exists()
    pd.DataFrame([trade], columns=TRADE_COLUMNS).to_csv(p, mode="a", header=new, index=False, encoding="utf-8")
    return {"holding": h, "trade": trade}
