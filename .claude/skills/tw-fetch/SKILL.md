---
name: tw-fetch
description: 抓台股日K資料（FinMind TaiwanStockPrice）並回報筆數與缺漏日期。輸入股票代號與日期區間，例如「/tw-fetch 2330 2018-01-01 2026-10-28」。
---

# tw-fetch

輸入：股票代號、開始日期、結束日期（省略結束日期 = 今天）。

步驟：
1. 在專案根目錄執行（Windows）：
   ```
   .venv\Scripts\python -m src.cli fetch <股票代號> <開始日期> [結束日期]
   ```
   - token 由環境變數 `FINMIND_TOKEN` 提供；不要要求使用者把 token 貼進對話或寫進檔案，也不要讀 `.env`。
   - 已快取的日期不會重抓（快取在 `data/TaiwanStockPrice_<代號>.csv`）。
2. 若出現 `FinMindRateLimitError`，原樣回報錯誤訊息，並建議稍後再試或設定 `FINMIND_TOKEN`；不要無限重試。
3. 回報給使用者：
   - 筆數、實際資料起訖日
   - 缺漏日期數量與清單（註明比對基準是「FinMind 交易日曆」還是「平日」；平日基準會把國定假日也算成缺漏）
   - 若缺漏很多，提醒可能是停牌或代號錯誤
