---
name: tw-backtest
description: 對台股跑回測並輸出報告（含與買進持有比較、oracle 檢查清單）。輸入股票代號和策略（ma_cross、candle_volume 或 lgbm），例如「/tw-backtest 2330 lgbm」。只做模擬，不下單。
---

# tw-backtest

輸入：股票代號、策略（`ma_cross`、`candle_volume` 或 `lgbm`），可選 `--start`（預設 2018-01-01）、`--end`（預設今天）、`--lot-mode board|odd`、`--holdout`。

預設只看開發期（`config.yaml` 的 `evaluation.holdout_start` 之前）。**只有使用者明確要求才加 `--holdout`**，並提醒每次查看都會記錄在 `reports/holdout_log.md`。

步驟：
1. 先跑 oracle 測試，必須全部通過才可以相信回測結果：
   ```
   .venv\Scripts\python -m pytest -q tests/test_oracles.py
   ```
   有任何失敗 → 停下來回報，先修回測，不要輸出報告。
2. 跑回測：
   ```
   .venv\Scripts\python -m src.cli backtest <股票代號> <策略> [--start YYYY-MM-DD] [--end YYYY-MM-DD]
   ```
   產出 `reports/<代號>_<策略>_<起>_<迄>.md`、同名 `.png`（權益曲線）與 `_kline.png`（近 120 日 K 線 + 成交量 + 模擬買賣點）。報告含 K 線型態統計與近 10 日量價指標。
   只想看 K 線與量價、不回測：`.venv\Scripts\python -m src.cli kline <股票代號>`
   報告內已附「Oracle 檢查（在本次資料上執行）」表。
3. 在報告最後補上一節「pytest oracle 測試」，列出步驟 1 實際跑過的測試名稱與結果（通過數 / 失敗數）。
4. 回報顯著性那一行（策略勝過幾 % 的同換手率隨機訊號、p 值）。單檔 p ≥ 0.05 代表無法和運氣區分。
   要判斷策略是否普遍有效，建議跑多檔研究：`.venv\Scripts\python -m src.cli study <策略>`（約數分鐘）。
5. 回報給使用者：總報酬、年化、最大回撤、Sharpe、勝率、交易次數、手續費與證交稅總額，並與買進持有比較；列出跑過哪些 oracle 測試與結果；附上報告路徑。
6. 提醒：結果僅供研究，不構成投資建議；股利已計入（現金入帳不再投入），股利稅預設不計。
