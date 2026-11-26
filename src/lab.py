"""策略實驗室：讀使用者的策略檔（事前登記），依 docs/策略實驗室.md 與 docs/研究流程.md 做完整檢驗並記錄試驗帳本。

只用開發期（evaluation.holdout_start 之前）；沒有保留期選項。只回答「證據過不過門檻」，不給買賣建議。
流程與對照組沿用 rank_study（0050 含息、eligible_pools、matched_random_targets、成交量上限、delist_recovery）。
"""
from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
import math
from datetime import date, datetime
from functools import partial
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from .backtest import Costs  # noqa: E402
from .config import ROOT  # noqa: E402
from .evaluate import effective_end, holdout_start  # noqa: E402
from .rank_strategy import (eligible_pools, factor_table, load_universe, matched_random_targets,  # noqa: E402
                            rebalance_days, run_portfolio, select_spec, Universe)
from .rank_study import _metrics, _scaled_costs, study_setup  # noqa: E402

FACTORS = ("mom63", "mom", "yoy3", "eps_ttm", "turnover", "close", "pe", "dy")
OPS = (">", ">=", "<", "<=")
REQUIRED = ("name", "registered", "hypothesis", "rationale", "fails_if", "universe", "rebalance_days", "top_n",
            "lookback", "filters", "rank_by")
AI_REQUIRED = ("who_pays", "source", "family")    # 文字 / 中繼資料，不進 spec_hash
RULE_FIELDS = ("universe", "rebalance_days", "top_n", "lookback", "filters", "rank_by")
TRIAL_COLS = ["time", "name", "spec_hash", "cagr", "sharpe", "p_random_sharpe", "p_random_cagr", "verdict",
              "git_commit", "data_hash", "n_random"]
DOWNSIDE_MDD = 0.10     # 最大回撤最多比 0050 深 10 個百分點
ALPHA = 0.05


# ---------------- 策略檔 ----------------
class SpecError(ValueError):
    pass


def validate_spec(spec) -> dict:
    if not isinstance(spec, dict):
        raise SpecError("策略檔內容必須是 YAML 對照表（key: value）")
    missing = [k for k in REQUIRED if k not in spec or spec[k] in (None, "", [])]
    if missing:
        raise SpecError("策略檔缺少必填欄位：" + "、".join(missing))
    author = spec.get("author", "user")
    if author not in ("user", "ai"):
        raise SpecError(f"author 只能是 user 或 ai，收到 {author!r}")
    if author == "ai":
        miss = [k for k in AI_REQUIRED if spec.get(k) in (None, "", [])]
        if miss:
            raise SpecError("author 為 ai 時必填：" + "、".join(miss))
    if spec["universe"] not in ("full", "default"):
        raise SpecError(f"universe 只能是 full 或 default，收到 {spec['universe']!r}")
    for k in ("rebalance_days", "top_n", "lookback"):
        v = spec[k]
        if isinstance(v, bool) or not isinstance(v, int) or v < 1:
            raise SpecError(f"{k} 必須是正整數，收到 {v!r}")
    if not isinstance(spec["filters"], list):
        raise SpecError("filters 必須是清單，每項 {factor, op, value}")
    for i, flt in enumerate(spec["filters"]):
        if not isinstance(flt, dict) or not {"factor", "op", "value"} <= set(flt):
            raise SpecError(f"filters[{i}] 必須有 factor、op、value：{flt!r}")
        if flt["factor"] not in FACTORS:
            raise SpecError(f"filters[{i}] 的因子 {flt['factor']!r} 不存在；可用：{'、'.join(FACTORS)}")
        if flt["op"] not in OPS:
            raise SpecError(f"filters[{i}] 的運算子 {flt['op']!r} 不合法；可用：{' '.join(OPS)}")
        if isinstance(flt["value"], bool) or not isinstance(flt["value"], (int, float)):
            raise SpecError(f"filters[{i}] 的 value 必須是數字，收到 {flt['value']!r}")
    rb = spec["rank_by"]
    if not isinstance(rb, dict) or "factor" not in rb or "order" not in rb:
        raise SpecError("rank_by 必須有 factor 與 order（asc / desc）")
    if rb["factor"] not in FACTORS:
        raise SpecError(f"rank_by 的因子 {rb['factor']!r} 不存在；可用：{'、'.join(FACTORS)}")
    if rb["order"] not in ("asc", "desc"):
        raise SpecError(f"rank_by 的 order 只能是 asc 或 desc，收到 {rb['order']!r}")
    return spec


def load_spec(path) -> dict:
    with open(path, encoding="utf-8") as f:
        try:
            spec = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise SpecError(f"策略檔不是合法的 YAML：{e}") from e
    return validate_spec(spec)


def _cfg_part(cfg: dict) -> dict:
    """run_lab 實際讀到、會影響結果的 config 參數；任何一個改變都算新規則。"""
    rc, bt = cfg["rank"], cfg["backtest"]
    return {"rank": {k: rc.get(k) for k in ("min_price", "min_turnover", "full_min_turnover", "start", "full_start",
                                             "benchmark")},
            "costs": dict(cfg["costs"]),
            "backtest": {k: bt.get(k) for k in ("max_volume_participation", "volume_window", "initial_cash",
                                                 "trading_days_per_year")},
            "holdout_start": str(cfg["evaluation"]["holdout_start"]),
            "default_universe": list(cfg["evaluation"]["universe"])}


def spec_hash(spec: dict, cfg: dict | None = None) -> str:
    """規則欄位（加上 cfg 時含影響結果的 config 參數）做正規化 JSON 的 8 碼 sha1；名稱、假設等文字欄位不影響。"""
    body = {
        "universe": spec["universe"], "rebalance_days": int(spec["rebalance_days"]), "top_n": int(spec["top_n"]),
        "lookback": int(spec["lookback"]),
        "filters": [{"factor": f["factor"], "op": f["op"], "value": float(f["value"])} for f in spec["filters"]],
        "rank_by": {"factor": spec["rank_by"]["factor"], "order": spec["rank_by"]["order"]}}
    if cfg is not None:
        body["config"] = _cfg_part(cfg)
    return hashlib.sha1(json.dumps(body, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:8]


def rule_text(spec: dict, rc: dict) -> str:
    fl = "、".join(f"{f['factor']} {f['op']} {f['value']:g}" for f in spec["filters"]) or "（無額外條件）"
    rb = spec["rank_by"]
    return (f"每 {spec['rebalance_days']} 個交易日，篩選 股價 > {rc['min_price']}、20 日均成交金額 > "
            f"{rc['min_turnover'] / 1e6:.0f} 百萬、{fl}（mom = 近 {spec['lookback']} 日漲幅）；"
            f"依 {rb['factor']} {'由大到小' if rb['order'] == 'desc' else '由小到大'}取前 {spec['top_n']} 檔等權；"
            "t 收盤決定、t+1 開盤成交；零股；含成本、股利、公司行動")


def neighbors(spec: dict, cfg: dict | None = None) -> list[tuple[str, dict]]:
    """鄰近參數：top_n、lookback、rebalance_days、每個非零數值門檻，一次只動一個，×0.8 與 ×1.2。
    整數參數四捨五入且至少 1；和原規則相同或彼此重複的略過。"""
    out, seen = [], {spec_hash(spec, cfg)}

    def add(label, s2):
        h = spec_hash(s2, cfg)
        if h not in seen:
            seen.add(h)
            out.append((label, s2))
    for key in ("top_n", "lookback", "rebalance_days"):
        for k in (0.8, 1.2):
            add(f"{key} ×{k:g}", {**spec, key: max(1, round(spec[key] * k))})
    for i, flt in enumerate(spec["filters"]):
        if flt["value"] == 0:
            continue
        for k in (0.8, 1.2):
            fl = [dict(x) for x in spec["filters"]]
            fl[i]["value"] = flt["value"] * k
            add(f"{flt['factor']} {flt['op']} {flt['value'] * k:g}（×{k:g}）", {**spec, "filters": fl})
    return out


# ---------------- 試驗帳本 ----------------
def read_ledger(path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=TRIAL_COLS)
    df = pd.read_csv(path, dtype=str, encoding="utf-8-sig").fillna("")
    for c in TRIAL_COLS:             # 舊帳本缺的欄位（例如 n_random）補空值
        if c not in df.columns:
            df[c] = ""
    return df


class LabRejected(Exception):
    pass


def _author_of(name: str, spec_dir) -> tuple[str, str]:
    """帳本的 name 對應 strategies/lab/<name>.yaml 的 (author, family)；讀不到回 ("unknown", "")。"""
    try:
        with open(Path(spec_dir) / f"{name}.yaml", encoding="utf-8") as f:
            d = yaml.safe_load(f)
        return str(d.get("author", "user")), str(d.get("family", ""))
    except Exception:
        return "unknown", ""


def ledger_authors(ledger: pd.DataFrame, spec_dir) -> tuple[int, int, int]:
    """帳本中（依 spec_hash 去重）使用者、AI、未知（讀不到策略檔）的不同規則數。"""
    first = ledger.drop_duplicates("spec_hash")
    au = [_author_of(n, spec_dir)[0] for n in first["name"]]
    return au.count("user"), au.count("ai"), au.count("unknown")


def check_ai_registration(spec: dict, cfg: dict, ledger_path, spec_dir, today: date | None = None) -> None:
    """AI 提出的新規則（spec_hash 從未出現在帳本）：檢查當月額度與已淘汰家族；違反就丟 LabRejected。"""
    if spec.get("author", "user") != "ai":
        return
    led = read_ledger(ledger_path)
    if spec_hash(spec, cfg) in set(led["spec_hash"]):
        return
    today = today or date.today()
    budget = int((cfg.get("lab") or {}).get("ai_monthly_budget", 2))
    first = led.drop_duplicates("spec_hash")
    month = f"{today:%Y-%m}"
    used = sum(_author_of(r["name"], spec_dir)[0] in ("ai", "unknown") and      # 讀不到策略檔保守視為 AI
                str(r["time"]).startswith(month)
               for _, r in first.iterrows())
    if used >= budget:
        raise LabRejected(f"本月 AI 新登記額度已用完（{used}/{budget}，lab.ai_monthly_budget）；同一規則重跑不受限")
    for _, r in led.iterrows():
        a, fam = _author_of(r["name"], spec_dir)
        if a in ("ai", "unknown") and r["verdict"] == "不通過" and fam == spec["family"]:
            raise LabRejected(f"這個家族（{fam}）的想法已被淘汰，不能換參數再試；要試請換一個有不同依據的想法")


def count_rules(ledger: pd.DataFrame, this_hash: str) -> int:
    """至今不同規則數（依 spec_hash 去重，含本次）。同一規則重跑不增加。"""
    return len(set(ledger["spec_hash"]) | {this_hash})


def bonferroni(p: float, n: int) -> float:
    return min(1.0, p * n)


def append_trial(path, row: dict) -> None:
    """只追加、不改舊列。"""
    path = Path(path)
    new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not new:
        with open(path, encoding="utf-8-sig", newline="") as f:
            old = list(csv.reader(f))
        if old and "n_random" not in old[0]:     # 舊帳本沒有 n_random 欄：舊列內容不變，只補表頭與空欄
            with open(path, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f)
                w.writerow(old[0] + ["n_random"])
                w.writerows(r + [""] for r in old[1:])
    with open(path, "a", newline="", encoding="utf-8-sig" if new else "utf-8") as f:
        w = csv.DictWriter(f, fieldnames=TRIAL_COLS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in TRIAL_COLS})


# ---------------- 檢驗 ----------------
def _sharpe(daily: pd.Series, tdays: int) -> float:
    return float(daily.mean() / daily.std() * np.sqrt(tdays)) if daily.std() > 0 else 0.0


def stock_contribution(res, u: Universe, end) -> pd.Series:
    """各股損益貢獻（元）：賣出淨額 − 買進成本 + 期末持股市值；不含股利，公司行動造成的股數變化也不計，是近似排序。"""
    t = res.trades
    if t.empty:
        return pd.Series(dtype=float)
    sid = t["stock_id"].to_numpy()
    flow = np.where(t["side"] == "sell", 1.0, -1.0) * t["value"].to_numpy() - t["commission"].to_numpy() - t["tax"].to_numpy()
    cash = pd.Series(flow, index=sid).groupby(level=0).sum()
    net = pd.Series(np.where(t["side"] == "buy", 1.0, -1.0) * t["shares"].to_numpy(), index=sid).groupby(level=0).sum()
    last = pd.Series({s: float(u.stocks[s].prices["close"][u.stocks[s].prices.index <= end].iloc[-1]) for s in net.index})
    return (cash + net.clip(lower=0) * last).sort_values(ascending=False)


def small_account_section(cfg: dict, u: Universe, targets: dict, run, costs: Costs, start, end, ws, we, tdays,
                          m: dict, main_res) -> list[str]:
    """小額帳戶情境（只是報告多一段資訊，不影響判定）：同一份凍結的目標股，本金與最低手續費換成
    lab.small_account 的值。沒有該區段就回傳空串列。"""
    sa = (cfg.get("lab") or {}).get("small_account")
    if not sa:
        return []
    c2 = dataclasses.replace(costs, min_commission=sa["commission_min"])
    res = run(u, targets, c2, sa["initial_cash"], start, end)
    ms = _metrics(res, ws, we, tdays)
    years = max((we - ws).days / 365.25, 1e-9)
    def mean_eq(r):
        e = r.equity[(r.equity.index >= ws) & (r.equity.index <= we)]
        return float(e.mean()) if len(e) else float("nan")

    # 近似：調整日收盤的淨值 × 0.99（run_portfolio 預設現金緩衝）/ 檔數，除以當天收盤價，不足 1 股就算買不到
    tot = miss = 0
    for d, picks in targets.items():
        d = pd.Timestamp(d)
        if not len(picks) or d < ws or d > we or d not in res.equity.index:
            continue
        per = float(res.equity.loc[d]) * 0.99 / len(picks)
        for s in picks:
            px = u.stocks[s].prices["close"]
            px = px[px.index <= d]
            tot += 1
            if len(px) and px.iloc[-1] > 0 and math.floor(per / px.iloc[-1]) < 1:
                miss += 1
    def ratio(x, mean):
        return x["costs"] / mean / years if mean and mean == mean else float("nan")
    return ["", f"## 小額帳戶情境（本金 {sa['initial_cash']:,.0f} 元、最低手續費 {sa['commission_min']:g} 元；不影響判定）", "",
            "同一份凍結的目標股，只換本金與最低手續費，其餘成本沿用 costs 區段；整張以下以零股（1 股）為單位成交。", "",
            "| 項目 | 數字 |", "|---|---|",
            f"| 年化 | {ms['cagr']:.1%} |", f"| 最大回撤 | {ms['mdd']:.1%} |", f"| Sharpe | {ms['sharpe']:.2f} |",
            f"| 總成本（元） | {ms['costs']:,.0f} |",
            f"| 成本佔平均淨值的年化比例 | {ratio(ms, mean_eq(res)):.2%}（主回測 {ratio(m, mean_eq(main_res)):.2%}） |",
            f"| 與主回測年化的差距 | {ms['cagr'] - m['cagr']:+.1%}（主回測 {m['cagr']:.1%}） |",
            f"| 因每檔分到的金額買不到 1 股而沒買到的目標 | {miss} / {tot} 次 |", "",
            "買不到的次數是近似：以各調整日收盤淨值 × 0.99（現金緩衝）÷ 目標檔數，除以當天收盤價，不足 1 股就計一次；"
            "沒考慮成交量上限、漲停或現金不足等其他原因。"]


def run_lab(spec: dict, cfg: dict, n_random: int = 100, spec_path=None) -> Path:
    spec = validate_spec(spec)
    if spec_path is not None and Path(spec_path).stem != spec["name"]:
        raise SpecError(f"策略檔檔名主幹（{Path(spec_path).stem}）必須和 name（{spec['name']}）一致，"
                        "否則帳本無法對應回策略檔")
    check_ai_registration(spec, cfg, ROOT / cfg["reports_dir"] / "lab" / "trials.csv", ROOT / "strategies" / "lab")
    rc, ids, ends, data_start, names, pool_desc = study_setup(cfg, spec["universe"])
    tdays = cfg["backtest"]["trading_days_per_year"]
    end = effective_end(cfg, None, False)            # 一律截在保留期之前
    U = load_universe(ids, data_start, end, cfg, ends=ends)
    B = load_universe([rc["benchmark"]], data_start, end, cfg, fundamentals=False)
    costs, bt = Costs.from_config(cfg), cfg["backtest"]
    cash0, start = bt["initial_cash"], rc["start"]
    ws, we = pd.Timestamp(start), min(pd.Timestamp(end), holdout_start(cfg) - pd.Timedelta(days=1))
    kw = dict(min_price=rc["min_price"], min_turnover=rc["min_turnover"])
    run = partial(run_portfolio, max_participation=bt.get("max_volume_participation"),
                  volume_window=bt.get("volume_window", 20))

    cache: dict = {}

    def targets_for(s, exclude=()):
        out = {}
        for d in rebalance_days(U, start, end, s["rebalance_days"]):
            if (d, s["lookback"]) not in cache:
                cache[(d, s["lookback"])] = factor_table(U, d, s["lookback"])
            f = cache[(d, s["lookback"])]
            out[d] = select_spec(f[~f["stock_id"].isin(exclude)] if len(exclude) else f, s, **kw)
        return out

    def sim(s, c=costs, exclude=(), **k):
        u = U if not len(exclude) else Universe({a: b for a, b in U.stocks.items() if a not in exclude}, U.calendar)
        tg = targets_for(s, exclude)
        return run(u, tg, c, cash0, start, end, **k), tg

    main, targets = sim(spec)
    first = U.calendar[U.calendar >= ws][0]
    bench = run(B, {first: [rc["benchmark"]]}, costs, cash0, start, end)
    pools = eligible_pools(U, rebalance_days(U, start, end, spec["rebalance_days"]), spec["lookback"], **kw)
    ew = run(U, pools, costs, cash0, start, end)
    m, bm, em = (_metrics(x, ws, we, tdays) for x in (main, bench, ew))

    checks = []   # (項目, 數字說明, 過?)
    ok = m["sharpe"] > bm["sharpe"] and m["mdd"] >= bm["mdd"] - DOWNSIDE_MDD
    checks.append(("對 0050 含息", f"Sharpe {m['sharpe']:.2f} vs {bm['sharpe']:.2f}；最大回撤 {m['mdd']:.1%} vs "
                   f"{bm['mdd']:.1%}（容許深 {DOWNSIDE_MDD * 100:.0f} 個百分點）", ok))

    rng = np.random.default_rng(0)
    rm = [_metrics(run(U, matched_random_targets(targets, pools, rng), costs, cash0, start, end), ws, we, tdays)
          for _ in range(n_random)]
    p = {}
    rows = []
    for key, label, fmt in (("cagr", "年化", "{:.1%}"), ("sharpe", "Sharpe", "{:.2f}")):
        rnd = np.array([x[key] for x in rm])
        p[key] = (1 + (rnd >= m[key]).sum()) / (n_random + 1)
        rows.append(f"| {label} | {fmt.format(m[key])} | {fmt.format(np.median(rnd))} | "
                    f"{fmt.format(np.quantile(rnd, 0.95))} | {(rnd < m[key]).mean():.0%} | {p[key]:.3f} |")
    checks.append(("配對隨機選股（前 5%）", f"年化 p = {p['cagr']:.3f}、Sharpe p = {p['sharpe']:.3f}（{n_random} 組）",
                   p["cagr"] <= ALPHA and p["sharpe"] <= ALPHA))

    nb = []
    for label, s2 in neighbors(spec, cfg):
        mm = _metrics(sim(s2)[0], ws, we, tdays)
        nb.append((label, mm))
    wins = sum(x["sharpe"] > bm["sharpe"] for _, x in nb)
    need = int(np.ceil(len(nb) * 0.75))
    checks.append(("鄰近參數", f"Sharpe 勝過 0050：{wins} / {len(nb)} 組（至少 {need} 組）", bool(nb) and wins >= need))

    mid = ws + (we - ws) / 2
    halves = [(a, b, _metrics(main, a, b, tdays), _metrics(bench, a, b, tdays))
              for a, b in ((ws, mid), (mid + pd.Timedelta(days=1), we))]
    checks.append(("前後兩段", "；".join(f"{a.date()}~{b.date()} Sharpe {x['sharpe']:.2f} vs {y['sharpe']:.2f}"
                                       for a, b, x, y in halves), all(x["sharpe"] > y["sharpe"] for *_, x, y in halves)))

    m2 = _metrics(sim(spec, c=_scaled_costs(costs, 2.0))[0], ws, we, tdays)
    checks.append(("兩倍成本", f"Sharpe {m2['sharpe']:.2f}（年化 {m2['cagr']:.1%}）vs 0050 {bm['sharpe']:.2f}",
                   m2["sharpe"] > bm["sharpe"]))

    stress = run(U, targets, costs, cash0, start, end, delist_recovery=0.0)
    ms = _metrics(stress, ws, we, tdays)
    n_delist = int(main.skipped["reason"].str.contains("下市").sum())
    checks.append(("下市持股歸零", f"Sharpe {ms['sharpe']:.2f}（年化 {ms['cagr']:.1%}）vs 0050 {bm['sharpe']:.2f}；"
                   f"遇到下市 {n_delist} 次", ms["sharpe"] > bm["sharpe"]))

    top3 = list(stock_contribution(main, U, we).head(3).index)
    m3 = _metrics(sim(spec, exclude=top3)[0], ws, we, tdays)
    top3_txt = "、".join(f"{s} {names.get(s, '')}".strip() for s in top3)
    checks.append(("拿掉貢獻最大 3 檔", f"{top3_txt} 從股票池移除後重跑："
                   f"Sharpe {m3['sharpe']:.2f}（年化 {m3['cagr']:.1%}）vs 0050 {bm['sharpe']:.2f}",
                   m3["sharpe"] > bm["sharpe"]))

    def daily(res):
        e = res.equity[(res.equity.index >= ws) & (res.equity.index <= we)]
        return e.pct_change().dropna()
    dm, db = daily(main), daily(bench)
    by_year = (1 + dm).groupby(dm.index.year).prod()
    best = int(by_year.idxmax())
    sm, sb = _sharpe(dm[dm.index.year != best], tdays), _sharpe(db[db.index.year != best], tdays)
    checks.append(("拿掉最好的一年", f"策略最好的年度 {best}（{by_year[best] - 1:+.1%}）；兩邊都移除該年日報酬後 Sharpe "
                   f"{sm:.2f} vs 0050 {sb:.2f}", sm > sb))

    # ---- 試驗帳本與多重比較 ----
    out_dir = ROOT / cfg["reports_dir"] / "lab"
    out_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = out_dir / "trials.csv"
    h = spec_hash(spec, cfg)
    n_rules = count_rules(read_ledger(ledger_path), h)
    n_user, n_ai, n_unk = ledger_authors(read_ledger(ledger_path), ROOT / "strategies" / "lab")
    if h not in set(read_ledger(ledger_path)["spec_hash"]):
        if spec.get("author", "user") == "ai":
            n_ai += 1
        else:
            n_user += 1
    p_worst = max(p["cagr"], p["sharpe"])
    p_adj = bonferroni(p_worst, n_rules)
    checks.append(("多重比較（Bonferroni）", f"隨機選股較大的 p {p_worst:.3f} × 規則數 {n_rules} = {p_adj:.3f}", p_adj <= ALPHA))
    min_p1 = 1 / (n_random + 1)
    min_pb = min(1.0, n_rules / (n_random + 1))
    power_ok = min_pb <= ALPHA   # 和 Bonferroni 判定同一個比較方向；K = 20N−1 時剛好 0.05
    need_k = int(np.ceil(20 * n_rules)) - 1
    if not power_ok:
        checks.append(("隨機次數是否足以判定", f"n_random = {n_random}，Bonferroni 後最小可達 p = {min_pb:.3f} ≥ {ALPHA}；"
                       f"目前的隨機次數不足以在 {n_rules} 個規則下判定顯著，請加大 --n-random 至少到 {need_k}", False))
    passed = all(c[2] for c in checks)

    from .provenance import data_manifest, git_commit, run_info
    stem = f"{spec['name']}_{date.today():%Y%m%d}"
    dh = data_manifest(cfg, ids + [rc["benchmark"]], out_dir / f"{stem}.data.json")
    append_trial(ledger_path, dict(time=datetime.now().isoformat(timespec="seconds"), name=spec["name"], spec_hash=h,
                                   cagr=f"{m['cagr']:.4f}", sharpe=f"{m['sharpe']:.3f}",
                                   p_random_sharpe=f"{p['sharpe']:.3f}", p_random_cagr=f"{p['cagr']:.3f}",
                                   verdict="通過" if passed else "不通過", git_commit=git_commit(), data_hash=dh,
                                   n_random=str(n_random)))

    # ---- 報告 ----
    flt = "、".join(f"{f['factor']} {f['op']} {f['value']:g}" for f in spec["filters"]) or "無"
    L = [f"# 策略實驗室：{spec['name']}", "",
         "> 僅供研究與模擬，不構成投資建議；本工具不含任何下單功能。只用開發期（保留期之前），實驗室沒有保留期選項。", "",
         "## 策略檔（事前登記）", "",
         f"- 登記日：{spec['registered']}　規則雜湊：`{h}`",
         f"- 提出者：{'AI' if spec.get('author', 'user') == 'ai' else '使用者'}",
         *[f"- {lbl}：{spec[k]}" for k, lbl in (("who_pays", "誰在賠錢"), ("source", "依據"), ("family", "家族"))
           if spec.get(k)],
         f"- 假設：{spec['hypothesis']}", f"- 理由：{spec['rationale']}", f"- 失效條件：{spec['fails_if']}",
         f"- 規則：{rule_text(spec, rc)}", f"- 參數：universe={spec['universe']}、rebalance_days={spec['rebalance_days']}、"
         f"top_n={spec['top_n']}、lookback={spec['lookback']}、filters=[{flt}]、"
         f"rank_by={spec['rank_by']['factor']} {spec['rank_by']['order']}",
         f"- 股票池：{pool_desc}；實際載入 {len(U.stocks)} 檔",
         "- 基本面只用當時已公布的資料（src/fundamentals.py）；FinMind 只有最新版本，沒有當時公告的原始數字",
         f"- 成交量上限：每檔每天買進 ≤ 前 {bt.get('volume_window', 20)} 日均量 × {bt.get('max_volume_participation')}；"
         f"對照組股票池為各調整日當天通過門檻的股票（平均 {np.mean([len(v) for v in pools.values()]):.0f} 檔）",
         "- 重現資訊：" + run_info(cfg, U.calendar.max(), dh) + f"（逐檔雜湊：{stem}.data.json）", "",
         f"![equity]({stem}.png)", "", f"## 總判定：{'通過' if passed else '不通過'}", "",
         "全部檢驗都過才算通過；「通過」只代表開發期證據過門檻，不是投資建議，也不是樣本外證據。", "",
         "| 檢驗 | 數字 | 結果 |", "|---|---|---|"]
    L += [f"| {a} | {b} | {'過' if c else '不過'} |" for a, b, c in checks]
    L += ["", f"## 績效（{ws.date()} ~ {we.date()}）", "",
          "| 組合 | 年化 | 最大回撤 | Sharpe | 年週轉率 | 成本 | 股利 |", "|---|---|---|---|---|---|---|"]
    for k, x in (("策略", m), ("0050 含息", bm), ("股票池等權", em), ("策略（壓力：下市持股歸零）", ms)):
        L.append(f"| {k} | {x['cagr']:.1%} | {x['mdd']:.1%} | {x['sharpe']:.2f} | {x['turnover']:.1f}x | "
                 f"{x['costs']:,.0f} | {x['dividends']:,.0f} |")
    L += small_account_section(cfg, U, targets, run, costs, start, end, ws, we, tdays, m, main)
    L += ["", f"**配對隨機選股**（{n_random} 組，同池、同檔數、同換手）", "",
          "| 指標 | 策略 | 隨機中位數 | 隨機前 5% 門檻 | 策略勝過 | 單尾 p |", "|---|---|---|---|---|---|", *rows, "",
          "**鄰近參數**", "", "| 變動 | Sharpe | 年化 | 勝過 0050 |", "|---|---|---|---|"]
    L += [f"| {a} | {x['sharpe']:.2f} | {x['cagr']:.1%} | {'是' if x['sharpe'] > bm['sharpe'] else '否'} |" for a, x in nb]
    L += ["", "**稽核方法備註**",
          "- 拿掉貢獻最大 3 檔：各股損益 = 賣出淨額 − 買進成本 + 期末持股市值（不含股利、不處理公司行動造成的股數變化，"
          "是近似排序）；找出前 3 檔後從股票池移除、重新選股重跑",
          "- 拿掉最好的一年：以策略各年度複利報酬最高的年度為準，策略與 0050 都移除該年度的日報酬後重算 Sharpe",
          "- 前後兩段：開發期依日期對半切；兩倍成本：手續費與證交稅都乘 2", "",
          "## 試驗帳本", "",
          f"至今不同規則數 N = {n_rules}（依規則雜湊去重，含本次；同一規則重跑不增加）。"
          f"隨機選股較大的 p = {p_worst:.3f}，Bonferroni 後 min(1, p × N) = {p_adj:.3f}；"
          f"{'仍顯著' if p_adj <= ALPHA else '不顯著，不通過'}（門檻 {ALPHA}）。", "",
          f"以 n_random = {n_random}，單次最小可達 p = 1/(K+1) = {min_p1:.4f}，"
          f"Bonferroni 後最小可達 p = min(1, N/(K+1)) = {min_pb:.4f}。",
          *([f"目前的隨機次數不足以在 {n_rules} 個規則下判定顯著，請加大 --n-random 至少到 ⌈20N⌉−1 = {need_k}。"
             "總判定因此不通過。"] if not power_ok else []), "",
          f"其中使用者 {n_user} 個、AI {n_ai} 個" + (f"、未知 {n_unk} 個" if n_unk else "") + "。",
          f"帳本：reports/lab/trials.csv（只追加不改）。"]

    png = out_dir / f"{stem}.png"
    fig, ax = plt.subplots(figsize=(10, 5))
    for lab, res, col, eng in (("策略", main, "#2a78d6", "Strategy"), ("0050 含息", bench, "#eb6834", "0050 (incl. dividends)"),
                               ("股票池等權", ew, "#1baf7a", "Universe equal-weight")):
        e = res.equity[(res.equity.index >= ws) & (res.equity.index <= we)]
        ax.plot(e.index, e / e.iloc[0], label=eng, color=col, lw=2 if lab == "策略" else 1.5)
    ax.set_yscale("log")
    ax.set_ylabel("Equity (start = 1, log scale)")
    ax.legend(frameon=False)
    ax.grid(True, color="#e6e6e6", lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(png, dpi=120)
    plt.close(fig)
    md = out_dir / f"{stem}.md"
    md.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"[lab] {md}（{'通過' if passed else '不通過'}）")
    return md
