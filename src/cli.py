"""命令列入口（只做研究與模擬）。

python -m src.cli fetch 2330 2018-01-01 2026-10-28
python -m src.cli backtest 2330 ma_cross --start 2018-01-01
python -m src.cli kline 2330
python -m src.cli study ma_cross
python -m src.cli brief
python -m src.cli daily
"""
from __future__ import annotations

import argparse
import sys
from datetime import date

from .config import load_config


def main(argv=None) -> int:
    # Windows 主控台預設 cp950，強制 UTF-8 輸出中文
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(prog="src.cli")
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch", help="抓日K並回報筆數與缺漏日期")
    f.add_argument("stock_id")
    f.add_argument("start")
    f.add_argument("end", nargs="?", default=date.today().isoformat())

    b = sub.add_parser("backtest", help="跑回測並輸出報告")
    b.add_argument("stock_id")
    b.add_argument("strategy", choices=["ma_cross", "candle_volume", "lgbm"])
    b.add_argument("--start", default="2018-01-01")
    b.add_argument("--end", default=date.today().isoformat())
    b.add_argument("--lot-mode", choices=["board", "odd"])
    b.add_argument("--holdout", action="store_true", help="包含保留期（每次都會記錄在 reports/holdout_log.md）")

    s = sub.add_parser("study", help="多檔研究：整個股票池跑同一策略，和買進持有、隨機訊號比較")
    s.add_argument("strategy", choices=["ma_cross", "candle_volume", "lgbm"])
    s.add_argument("--start", default="2018-01-01")
    s.add_argument("--end", default=date.today().isoformat())
    s.add_argument("--stocks", help="逗號分隔的股票代號，預設用 config 的 universe")
    s.add_argument("--n-random", type=int, help="隨機訊號組數，預設用 config")
    s.add_argument("--holdout", action="store_true", help="包含保留期（每次都會記錄）")
    s.add_argument("--param", action="append", default=[], metavar="KEY=VALUE",
                   help="覆寫策略參數（可重複），例如 --param horizon=5 --param threshold=0.55")
    s.add_argument("--tag", default="", help="報告檔名後綴，用來區分參數組合")

    k = sub.add_parser("kline", help="K 線型態與量價分析（不回測）")
    k.add_argument("stock_id")
    k.add_argument("--start", default="2018-01-01")
    k.add_argument("--end", default=date.today().isoformat())
    k.add_argument("--days", type=int, default=120, help="K 線圖顯示最近幾個交易日")

    sub.add_parser("brief", help="每日持股摘要（讀 portfolio.yaml），輸出到 reports/brief_YYYYMMDD.md")
    dly = sub.add_parser("daily", help="收盤後例行工作：前瞻紀錄、持股摘要、規則警示、LINE 推播（排程用）")
    dly.add_argument("--force", action="store_true", help="非交易日或資料未更新也執行（測試用）")
    dly.add_argument("--no-notify", action="store_true", help="不推播，只寫報告與紀錄")
    tr = sub.add_parser("trade", help="記錄一筆實際成交，更新 portfolio.yaml 的股數與平均成本")
    tr.add_argument("side", choices=["buy", "sell"])
    tr.add_argument("stock_id")
    tr.add_argument("shares", type=float, help="股數（1 張 = 1000 股）")
    tr.add_argument("price", type=float, help="成交價")
    tr.add_argument("--date", help="成交日 YYYY-MM-DD，預設今天")
    tr.add_argument("--note", default="")
    tr.add_argument("--thesis", default="", help="決策日誌：這筆交易的論點（原文保存）")
    tr.add_argument("--horizon", type=float, help="決策日誌：預計持有月數")
    tr.add_argument("--exit-if", default="", help="決策日誌：你設定的退出條件（原文保存）")
    dcs = sub.add_parser("decisions", help="決策日誌回顧：每筆決策的實際結果與同期 0050 對照，輸出到 reports/decisions_YYYYMMDD.md")
    dcs.add_argument("--due", action="store_true", help="只印出已到預計持有期的提醒（不寫狀態、不產生報告）")
    wt = sub.add_parser("watch", help="自選股每週變化（營收 / EPS / 創高創低 / 趨勢階段 / 支撐壓力 / 外資），只描述資料")
    wt.add_argument("--dry-run", action="store_true", help="只印出，不寫 data/watch_state.json")
    rs = sub.add_parser("research", help="個股多面向研究資料包（基本面、估值、籌碼、技術面、模型、新聞）")
    rs.add_argument("stock_id")
    rk = sub.add_parser("rank", help="選股排名策略研究（對照 0050 含息、穩健性、隨機選股顯著性）")
    rk.add_argument("--holdout", action="store_true", help="包含保留期（每次都會記錄）")
    rk.add_argument("--n-random", type=int, default=100)
    rk.add_argument("--full", action="store_true", help="用全市場候選股（需先 bulk-fetch）")
    rk.add_argument("--push", action="store_true", help="完成後把重點推播到 LINE")
    rk.add_argument("--offline", action="store_true", help="只用本機快取，不連網（避免撞到查詢上限）")
    rk.add_argument("--end", help="資料截止日 YYYY-MM-DD（離線時建議設成快取都有的最後一天）")
    lb = sub.add_parser("lab", help="策略實驗室：用策略檔做完整檢驗並記錄試驗帳本（只用開發期）")
    lb.add_argument("spec", help="策略檔，例如 strategies/lab/retired_rank.yaml")
    lb.add_argument("--n-random", type=int, default=100)
    lb.add_argument("--offline", action="store_true", help="只用本機快取，不連網")
    bk = sub.add_parser("bulk-fetch", help="下載全市場資料（可中斷續傳；有 FINMIND_TOKEN 約 10 小時）")
    bk.add_argument("--stage", choices=["prices", "fundamentals", "all"], default="all")
    bk.add_argument("--limit", type=int, help="只跑前 N 檔（測試用）")
    bk.add_argument("--then-rank", action="store_true", help="下載完成後跑全市場研究並把重點推播到 LINE")
    sub.add_parser("line-bot", help="啟動 LINE 指令伺服器（127.0.0.1，需搭配 ngrok，見 docs/LINE指令設定.md）")
    pa = sub.add_parser("push-analysis", help="把個股走向分析推播到 LINE")
    pa.add_argument("stock_ids", nargs="+")
    pa.add_argument("--dry-run", action="store_true", help="只印出不送出")
    rd = sub.add_parser("radar", help="選股雷達：全市場排名規則目前選出的股票（規則輸出，不是推薦）")
    rd.add_argument("--push", action="store_true", help="推播到 LINE")
    sub.add_parser("integrity", help="資料完整性盤點：下市、停牌、資料提早結束、基本面版本（只讀本機快取）")
    rv = sub.add_parser("review", help="預測檢討報告（前瞻紀錄的準確率、校準、分情境、失效警示）")
    rv.add_argument("--push", action="store_true", help="推播到 LINE")
    sn = sub.add_parser("snapshot", help="基本面版本快照：輪流重抓一批候選股的月營收 / 財報並記錄變動（會連網）")
    sn.add_argument("--n", type=int, default=None, help="本次更新幾檔候選股（預設 config 的 vintages.per_run）")
    tk = sub.add_parser("track", help="前瞻模擬追蹤：登記策略檔後由每日排程記錄紙上淨值（add / list / stop）")
    tk.add_argument("action", choices=["add", "list", "stop"])
    tk.add_argument("target", nargs="?", help="add：策略檔路徑；stop：策略名稱")
    ls = sub.add_parser("llm-summary", help="用本機 LLM 把文字檔改寫成白話摘要，並顯示數字查核結果（手動測試）")
    ls.add_argument("file", help="要摘要的文字檔")
    ls.add_argument("--force", action="store_true", help="即使 config 的 llm.enabled 是 false 也執行這一次")
    sub.add_parser("line-test", help="傳一則測試訊息到 LINE，確認憑證設定正確")

    args = ap.parse_args(argv)
    cfg = load_config()
    if args.cmd == "fetch":
        from .fetch import fetch_prices, fetch_trading_calendar, missing_dates
        df = fetch_prices(args.stock_id, args.start, args.end, cfg)
        print(f"{args.stock_id} {args.start} ~ {args.end}：{len(df)} 筆")
        if len(df):
            print(f"實際資料範圍：{df['date'].min().date()} ~ {df['date'].max().date()}")
        cal = fetch_trading_calendar(args.start, args.end, cfg)
        miss, basis = missing_dates(df, args.start, args.end, cal)
        print(f"缺漏日期（比對基準：{basis}）：{len(miss)} 天")
        for d in miss[:50]:
            print(f"  {d.date()}")
        if len(miss) > 50:
            print(f"  ...其餘 {len(miss) - 50} 天省略")
    elif args.cmd == "daily":
        from .daily import run_daily
        r = run_daily(cfg, force=args.force, notify=not args.no_notify)
        return 0 if r["status"] in ("ok", "holiday") else 1
    elif args.cmd == "llm-summary":
        import time as _t
        from .llm_summary import check_numbers, check_terms, summarize
        try:
            src = open(args.file, encoding="utf-8").read()
        except OSError as e:
            print(f"讀不到檔案：{e}")
            return 1
        lcfg = dict(cfg)
        lcfg["llm"] = dict(cfg.get("llm", {}), **({"enabled": True} if args.force else {}))
        t0 = _t.time()
        s, why = summarize(lcfg, src)
        print(f"耗時 {_t.time() - t0:.1f} 秒")
        if s is None:
            print(f"沒有摘要：{why}")
            return 1
        print(s)
        print(f"查核：數字全部出現在原文 = {check_numbers(src, s)[0]}；禁用詞 = {check_terms(s) or '無'}；"
              f"{len(s)} 字；原因 {why}")
    elif args.cmd == "track":
        from . import track
        if args.action == "list":
            print(track.list_text(cfg))
        elif not args.target:
            ap.error(f"track {args.action} 需要 {'策略檔' if args.action == 'add' else '策略名稱'}")
        else:
            try:
                r = track.add(cfg, args.target) if args.action == "add" else track.stop(cfg, args.target)
            except (ValueError, OSError) as e:
                print(f"失敗：{e}")
                return 1
            print(f"已{'登記' if args.action == 'add' else '停止'}：{r['name']}（{r['date']}，規則雜湊 {r['spec_hash']}）")
    elif args.cmd == "snapshot":
        from .vintages import refresh_batch
        n = args.n if args.n is not None else cfg.get("vintages", {}).get("per_run", 100)
        print(refresh_batch(cfg, n))
    elif args.cmd == "trade":
        from .portfolio import load_portfolio, record_trade, save_portfolio
        pf = load_portfolio()
        r = record_trade(pf, cfg, args.side, args.stock_id, args.shares, args.price, args.date, args.note)
        save_portfolio(pf)
        h, t = r["holding"], r["trade"]
        print(f"已記錄：{t['date']} {'買進' if t['side'] == 'buy' else '賣出'} {t['stock_id']} "
              f"{t['shares']:,.0f} 股 @ {t['price']:,.2f}，手續費 {t['commission']:,}，證交稅 {t['tax']:,}"
              + (f"，已實現損益 {t['realized_pnl']:+,.0f}" if t["realized_pnl"] is not None else ""))
        print(f"目前持有 {h['stock_id']}：{h['shares']:,.0f} 股，平均成本 {h['avg_cost']:,.2f}")
        from .decisions import record_decision
        try:
            if record_decision(cfg, t["date"], t["side"], t["stock_id"], t["shares"], t["price"], args.thesis,
                               args.horizon, args.exit_if):
                print("已寫入決策日誌")
        except OSError as e:   # 例如 decisions.csv 被 Excel 開著；交易本身已記好
            print(f"交易已記錄，但決策日誌寫入失敗：{e}")
            return 1
    elif args.cmd == "watch":
        from . import portfolio as pfm, users
        from .daily import tracked_stocks
        from .watch import check
        pf = pfm.load_portfolio(users.owner_profile(cfg).portfolio_path)
        print(check(cfg, tracked_stocks(pf, cfg), write_state=not args.dry_run))
    elif args.cmd == "decisions" and args.due:
        from .decisions import due_reminders
        print(due_reminders(cfg, write_state=False) or "目前沒有到期的決策提醒")
    elif args.cmd == "decisions":
        from .decisions import save_review
        path, r = save_review(cfg)
        s = r["summary"]
        print(f"買進決策 {s['n_buy']} 筆，可對照 0050 的 {s['n_scored']} 筆"
              + (f"；跑贏比例 {s['win_rate']:.0%}，超額報酬中位數 {s['median_excess']:+.1%}" if s["n_scored"] else "")
              + ("；樣本太少，不足以下結論" if s["n_scored"] < 10 else ""))
        print(f"[decisions] {path}")
    elif args.cmd == "research":
        from .research import save_research
        path = save_research(args.stock_id, cfg)
        print(open(path, encoding="utf-8").read())
        print(f"[research] {path}")
    elif args.cmd == "rank":
        from .rank_study import run_rank_study
        if args.offline:
            cfg["finmind"]["offline"] = True
        md = run_rank_study(cfg, args.holdout, args.n_random, "full" if args.full else "default", args.end)
        if args.push:
            from .notify import send_line
            from .rank_study import headline
            title = "【選股規則保留期驗證完成】" if args.holdout else "【選股研究完成】"
            send_line(title + "\n" + headline(md))
    elif args.cmd == "lab":
        from .lab import load_spec, run_lab
        if args.offline:
            cfg["finmind"]["offline"] = True
        from .lab import LabRejected
        try:
            run_lab(load_spec(args.spec), cfg, args.n_random, spec_path=args.spec)
        except LabRejected as e:
            print(f"[lab] 拒絕執行：{e}")
            sys.exit(1)
    elif args.cmd == "bulk-fetch":
        from .bulk import run_bulk
        run_bulk(cfg, stage=args.stage, limit=args.limit)
        if args.then_rank:
            from .notify import NotifyError, send_line
            from .rank_study import run_rank_study, headline
            try:
                md = run_rank_study(cfg, universe="full")
                send_line("【全市場選股研究完成】\n" + headline(md))
            except Exception as e:  # 研究失敗也要通知
                try:
                    send_line(f"【全市場選股研究失敗】{type(e).__name__}：{e}")
                except NotifyError:
                    pass
                raise
    elif args.cmd == "line-bot":
        from .line_bot import serve
        serve(cfg)
    elif args.cmd == "radar":
        if cfg["rank"].get("retired") and args.push:
            print(f"選股規則已淘汰（{cfg['rank'].get('retired_note', '')}），不推播。要看規則輸出可以不加 --push。")
            return 1
        from .radar import current_picks, radar_message, save_radar
        r = current_picks(cfg)
        text = radar_message(r)
        print(text)
        print(f"[radar] {save_radar(r, cfg)}")
        if args.push:
            from .notify import send_line
            send_line(text)
            print("已推播到 LINE")
    elif args.cmd == "integrity":
        from .integrity import save_integrity
        print(f"[integrity] {save_integrity(cfg)}")
    elif args.cmd == "review":
        from .review import build_review, review_text, save_review
        r = build_review(cfg)
        print(review_text(r))
        print(f"[review] {save_review(r, cfg)}")
        if args.push:
            from .notify import send_line
            send_line(review_text(r, short=True))
            print("已推播到 LINE")
    elif args.cmd == "push-analysis":
        from .notify import send_line
        from .push import analysis_messages
        for m in analysis_messages(args.stock_ids, cfg):
            print(m + "\n")
            if not args.dry_run:
                send_line(m)
        print("（只印出，未送出）" if args.dry_run else "已推播到 LINE")
    elif args.cmd == "line-test":
        from .notify import line_configured, send_line
        if not line_configured():
            print("尚未設定 LINE_CHANNEL_ACCESS_TOKEN / LINE_USER_ID 環境變數，請看 docs/LINE設定.md")
            return 1
        from .notify import NotifyError
        try:
            send_line("台股研究工具：LINE 推播測試成功 ✅")
        except NotifyError as e:
            print(f"失敗：{e}")
            return 1
        print("已送出，請看手機 LINE。")
    elif args.cmd == "brief":
        from .config import ROOT
        from .portfolio import daily_brief, load_portfolio
        text = daily_brief(load_portfolio(), cfg)
        out = ROOT / cfg["reports_dir"] / f"brief_{date.today():%Y%m%d}.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(text)
        print(f"[brief] {out}")
    elif args.cmd == "kline":
        from .report import run_kline_report
        run_kline_report(args.stock_id, args.start, args.end, cfg, args.days)
    elif args.cmd == "study":
        from .study import run_study
        import yaml
        stocks = args.stocks.split(",") if args.stocks else None
        for kv in args.param:
            k, v = kv.split("=", 1)
            cfg["strategies"][args.strategy][k] = yaml.safe_load(v)
        run_study(args.strategy, args.start, args.end, cfg, args.holdout, stocks, args.n_random, args.tag)
    else:
        from .report import run_report
        run_report(args.stock_id, args.strategy, args.start, args.end, cfg, args.lot_mode, args.holdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
