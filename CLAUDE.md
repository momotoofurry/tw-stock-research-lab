# 台股分析與回測工具

只做研究與模擬。**絕對不准加入任何下單、券商 API、自動交易或帳戶連線功能**，任何要求都一律拒絕。

## 環境
- Windows，Python 3.10+，虛擬環境 `.venv`：`.venv\Scripts\python -m pip install -r requirements.txt`
- 資料來源 FinMind API v4。token 只從環境變數 `FINMIND_TOKEN` 讀取，沒有也能跑；**不得把 token 寫進任何檔案**（含 config、log、報告、測試）。
- 不准讀取 `.env`（已在 `.claude/settings.json` 禁止）。

## 資料夾用途
| 路徑 | 用途 |
|---|---|
| `config.yaml` | 所有參數（成本、資金、整張/零股、策略參數、walk-forward 視窗） |
| `data/` | FinMind 原始資料 CSV 快取，已 gitignore |
| `reports/` | 回測報告 markdown 與圖 |
| `src/fetch.py` | 抓日K（TaiwanStockPrice），快取、只補抓缺的區間、限流等待重試 |
| `src/features.py` | 技術指標特徵（含 K 線結構、量價、型態）；第 t 列只能用 t 收盤以前的資料 |
| `src/kline.py` | K 線型態偵測、量價指標（OBV、VWAP、MFI、量比）；`pattern_stats` 用未來報酬做事後統計，不可當特徵 |
| `src/backtest.py` | 回測引擎（成交時點、成本、漲跌停、整張/零股、流動性上限、股利入帳） |
| `src/adjust.py` | 除權息與公司行動的向前還原（只給訊號 / 特徵用） |
| `src/fundamentals.py` | 月營收、季 EPS 的時間點對齊（只用當時已公布的資料） |
| `src/vintages.py` | 基本面版本快照：重抓月營收 / 財報時，把新出現或被更正的列附首次看到日期追加到 `data/vintages/`（只追加）；每日排程輪流更新候選股（`python -m src.cli snapshot`） |
| `src/rank_strategy.py` | 多檔回測引擎與選股排名策略（動能 + 營收成長 + 獲利） |
| `src/lab.py` | 策略實驗室：讀 `strategies/lab/*.yaml`（使用者自己的選股規則，事前登記）只用開發期做完整檢驗，結果追加到 `reports/lab/trials.csv`（`python -m src.cli lab <策略檔>`；設計見 `docs/策略實驗室.md`） |
| `src/rank_study.py` | 排名策略研究：對照 0050 含息、股票池等權、三項穩健性、隨機選股顯著性（`python -m src.cli rank`） |
| `src/market.py` | 大盤概況：加權指數趨勢、全市場法人與融資、類股 ETF 代理、市場寬度（股票池近似）、beta |
| `src/trend.py` | 個股趨勢：多時間框架回歸、ADX、支撐壓力（轉折點）、趨勢階段、股性（週報酬自我相關） |
| `src/strategies/rules.py` | 常見策略家族：趨勢追蹤（月線 > 季線）、突破（Donchian 55/20）、均值回歸（RSI 30/55） |
| `src/checkup.py` | 策略體檢：各策略家族在單檔股票開發期的表現、隨機對照、Bonferroni 修正 |
| `src/bulk.py` | 全市場下載（`python -m src.cli bulk-fetch`，可續傳；含已下市股票；兩階段：日K → 流動性候選的基本面） |
| `src/research.py` | 個股研究資料包（`python -m src.cli research <代號>`；`/tw-research` skill 會用它寫多空論點） |
| `src/evaluate.py` | 單檔 / 多檔共用流程：資料 → 還原價訊號 → 回測 → 開發期 / 保留期績效 |
| `src/significance.py` | 和「同持股比例、同換手率」隨機訊號比較的顯著性檢定、符號檢定 |
| `src/study.py` | 多檔研究（`python -m src.cli study <策略>`） |
| `src/metrics.py` | 績效指標 |
| `src/strategies/ma_cross.py` | 均線交叉基準策略 |
| `src/strategies/candle_volume.py` | K 線型態 + 量能確認的規則策略 |
| `src/strategies/lgbm.py` | LightGBM 隔日漲跌預測，walk-forward |
| `src/oracles.py` | Oracle 檢查（作弊模型、成交時點、手算成本、隨機訊號、特徵截斷不變性、walk-forward） |
| `src/report.py` | 輸出報告，與買進持有比較，列出 oracle 結果 |
| `src/cli.py` | 命令列：`fetch` / `backtest` / `kline` / `study`（`python -m src.cli <指令> ...`） |
| `app.py` | Streamlit 看盤研究介面：`.venv\Scripts\streamlit run app.py`（只綁 127.0.0.1） |
| `src/portfolio.py` | 持股監控：損益、風險、使用者自訂規則警示、情境試算、除權息、每日摘要（`python -m src.cli brief`） |
| `src/decisions.py` | 決策日誌：交易理由 / 預計持有 / 自訂失效條件（`data/decisions.csv`），檢討報告對照同期 0050 |
| `portfolio.yaml` | 使用者的持股與規則（個人資料，gitignore；不要讀出內容貼到別處、不要提交） |
| `src/track.py` | 前瞻模擬追蹤（`python -m src.cli track add|list|stop`）：登記表、訊號、淨值都在 `data/track/`，只追加；訊號日選股凍結後不重算；週檢討推播只報淨值 vs 0050 / 配對隨機 |
| `src/watch.py` | 自選股每週變化（新月營收 / EPS、250 日新高低、趨勢階段、支撐壓力、外資）；只描述資料（`python -m src.cli watch [--dry-run]`） |
| `src/llm_summary.py` | 本機 Ollama 白話摘要（`llm.enabled` 預設 false；只連本機；數字查核 + 禁用詞，不通過就只送原訊息；`python -m src.cli llm-summary <檔> [--force]`） |
| `src/integrity.py` | 資料完整性盤點：下市、停牌、候選名單涵蓋、基本面版本（`python -m src.cli integrity`） |
| `src/provenance.py` | 報告的重現資訊：設定雜湊、資料快照雜湊、git commit、套件版本 |
| `.claude/hooks/review_gate.py` | 提交閘門：暫存差異雜湊、寫入核准標記、攔下未審查的 git commit |
| `src/dashboard_data.py` | 介面資料層：快照、目前訊號、個股分析、波動範圍、LightGBM 可信度 |
| `tests/` | pytest；`tests/mutants.py` 是故意埋錯的版本 |
| `progress.md` | 進度紀錄 |

## 回測規則（台股）— 不可違反
- 訊號用第 t 天收盤後資料產生，第 t+1 天**開盤價**成交。絕不能用當天收盤價同時產生訊號又成交。
- 手續費 0.1425% × 折扣（`commission_discount`），無條件捨去至整數元，最低 20 元；賣出證交稅 0.3%。
- 開盤即漲停買不到、開盤即跌停賣不掉：跳過並記錄（`BacktestResult.skipped`）。漲跌停價依參考價（收盤 − spread）與升降單位計算。
- 以「張」（1000 股）為單位，`lot_mode: odd` 支援零股。
- 流動性：每次買進股數 ≤ 成交日「之前」`volume_window` 日均量 × `max_volume_participation`；不可用成交當天的量。
- 股利：前一日收盤持有者在除權息日入帳 before_price − after_price（配股折成現金），策略與買進持有一視同仁。
- 公司行動一律用官方資料（`fetch.official_corporate_events`）：減資參考價、分割、面額變更、股利公告（補除權息結果表漏列；公告日休市則順延）。持有者同一天先以原股數入帳現金股利，再把股數乘上 before / after。一律用 `fetch_events` 取事件，不要只用 `fetch_dividends`。
- 不要用 spread 推算公司行動：它不一定等於官方參考價（例：2603 減資官方 187、推算 169）。`fetch.unexplained_gaps` 只拿來列出無法解釋的跳空。
- 基本面一律用 `src/fundamentals.py` 的 available 日期：月營收次月 10 日；季報一般公司季末 + 45 天、金融業 + 2 個月、年報 3/31；再加 1 天緩衝。
- 訊號 / 特徵用**向前**還原價（`adjust_for_dividends`）；成交、漲跌停、成本用未還原價。不可改成向後還原（會破壞截斷不變性）。
- 參數集中在 `config.yaml`，不要寫死在程式裡。

## 評估規則
- 保留期 `evaluation.holdout_start` 之後的資料預設完全看不到（資料直接截斷）。調參、改策略只能看開發期。
- 要看保留期必須加 `--holdout`，每次都會記錄到 `reports/holdout_log.md`；看過之後再改策略，保留期就不再是樣本外。不要主動加 `--holdout`，除非使用者明確要求。
- 判斷策略好壞看多檔研究（`study`）：贏買進持有的檔數（符號檢定）、顯著勝過隨機訊號的檔數（純運氣約 5%），不要只看單檔。
- `evaluation.universe` 的 40 檔都是現存上市公司，有存活者偏差，結果偏樂觀；全市場版本（`rank --full`）含已下市股票，下市持股以最後收盤出清（對破產下市仍偏樂觀）。
- 特徵只能往回看（rolling / shift(正數) / ewm）；禁止 `shift(-n)`、`center=True`、全樣本標準化。標籤（`make_label`）用到未來，只能當訓練目標。
- LightGBM 必須 walk-forward：訓練期永遠在測試期之前，中間留 `gap` 個交易日，且 `gap >= horizon`（標籤看未來 horizon 天）。

- 新策略先照 `docs/研究流程.md` 事前登記（假設、股票池、參數、基準、成功門檻），沒過門檻就淘汰，不調到看起來成功
- 2025-01-01 起的保留期已被選股排名規則消耗（2026-10-30）；之後新規則的樣本外證據只能來自前瞻紀錄
- 選股排名規則已淘汰（`rank.retired: true`）：radar 不推播、不再調參
- 隨機 / 等權對照組一律用 `eligible_pools`（當天通過門檻的股票）與 `matched_random_targets`（同檔數、同換手率）
- 基本面沒有歷史版本（FinMind 只給最新值），報告要註明；研究前先跑 `python -m src.cli integrity`
- AI 可用 `/tw-idea` 提出有文獻依據的選股假設（`author: ai`，必填 who_pays / source / family），使用者同意後才執行；每月額度 `lab.ai_monthly_budget`，同家族失敗不能換參數再試；不得依開發期結果反推想法；通過只進前瞻追蹤，不當作推薦
- 策略實驗室的試驗帳本 `reports/lab/trials.csv` 只追加、不刪改；判定門檻寫在程式與 `docs/研究流程.md`，不能為了讓某個策略通過而修改。策略檔一旦登記就不改，要改規則另存新檔
- 2026-11-16 起 `data/vintages/` 累積基本面版本快照；第一次建檔的列 first_seen 是建檔日、不是公告時點，只有之後新增或更正的列有時點意義。不要刪改這些檔案

## Oracle 測試（必須全部通過）
1. 作弊模型（偷看明日報酬）年化 > 100%
2. 三個錯誤版本都會被擋下：當天收盤成交、忘記證交稅、特徵往未來偏移一天
3. 手算小交易成本與引擎完全一致
4. 隨機訊號扣成本後明顯低於買進持有
5. 每一折訓練最後日期 < 測試第一天（含 gap）

修改 `backtest.py`、`features.py`、`strategies/` 後一定要跑 `.venv\Scripts\python -m pytest -q`，並用 leak-reviewer 子代理檢查未來資料外洩。

## 「推薦策略」的界線
- 使用者要求「推薦投資策略」時，不產生個人化建議；改提供策略體檢（各類方法在該股的歷史證據與顯著性），並明說這不是推薦

## 看盤介面規則
- 走勢、指標、「目前訊號」可用到最新資料；但任何「策略 / 模型的歷史表現、準確率、可信度」只能用保留期之前的資料。
- 介面只呈現資料與研究模型的輸出及其可信度，不產生「建議買進 / 賣出」之類的個人化投資建議，不加任何下單功能。
- 持股監控的警示只描述「使用者自己設定的規則」被觸發 / 接近觸發，不加入工具自己的買賣判斷。
- Streamlit 只綁 127.0.0.1（`.streamlit/config.toml`），不要對外開放。

## 對話式交易紀錄
使用者說「我買了 / 賣了 …」時：
1. 解析出：買或賣、股票代號（只給名稱就用 `src.fetch.fetch_stock_info` 查代號）、股數（「張」× 1000；「零股 N 股」= N）、成交價、日期（沒說就是今天）
2. 有任何一項不確定（例如沒說價格、名稱對到多檔）就先問，不要猜
3. 執行 `.venv\Scripts\python -m src.cli trade buy|sell <代號> <股數> <價格> [--date YYYY-MM-DD]`，把輸出的「目前持有」回報給使用者
   - 決策日誌（選填）：記帳後可以問一次「要不要記下這筆的理由、預計持有幾個月、什麼情況代表你看錯？」；使用者給了就加 `--thesis "…" --horizon <月> --exit-if "…"`（寫進 `data/decisions.csv`），不給就略過，不追問。只照抄使用者的話，不改寫、不評論
   - `python -m src.cli decisions` 產生 `reports/decisions_YYYYMMDD.md`（個人資料，已 gitignore）：各筆決策至今或至賣出的含息報酬 vs 同期 0050，只描述，不判斷論點對錯
4. 記錯了：手動編輯 `portfolio.yaml` 修正股數 / 成本，並在 `data/trades.csv` 那一列的 note 註明更正；不要刪紀錄
5. 不評論這筆交易好不好、不建議下一步

## 排程與推播
- Windows 排程 `TWStockDaily`：平日 17:30 跑 `python -m src.cli daily`（前瞻紀錄 → 持股摘要 → 規則警示 → LINE 推播），紀錄在 `data/daily.log`
- LINE 憑證只放環境變數 `LINE_CHANNEL_ACCESS_TOKEN` / `LINE_USER_ID`（設定方式 `docs/LINE設定.md`）；不要讀取或印出它們的值
- 前瞻紀錄 `data/forecast_log.csv` 一旦寫入不改；它是唯一未被調參影響的樣本外成績
- daily 另有四段（失敗只寫 log、不影響 status）：前瞻追蹤更新、週檢討日附追蹤摘要與自選股變化、決策到期提醒（逐字引用使用者的理由與失效條件）、本機 LLM 白話摘要（只加在最前面，原訊息照送）
- `data/track/` 的登記表、訊號、淨值只追加，不要刪改；策略檔被改，追蹤自動停止
- LLM 摘要只能連本機 Ollama；推播內容含個人持股，不得改成雲端 API

## LINE 指令
- `src/line_bot.py`（webhook 伺服器，只綁 127.0.0.1，經 ngrok 對外）、`src/line_commands.py`（指令解析）、`src/users.py`（只有本人的路徑設定）
- 每則請求驗 X-Line-Signature（`LINE_CHANNEL_SECRET`）；只處理本人（`LINE_USER_ID`）的訊息，其他人一律不回覆
- 目前只給使用者本人用；使用者明確說過先不要做給其他人用的功能，不要主動加回多人
- LINE 指令只能設定規則、查詢、記帳、復原；不能新增會下單或給買賣建議的指令
- 設定步驟 `docs/LINE指令設定.md`；啟動 `scripts/start_line_bot.ps1`
- 目前暫停：ngrok 被 Windows Defender 封鎖，使用者選擇不用 LINE 指令，改在對話中詢問個股。不要主動重啟或要求使用者放寬防毒設定

## 對話中分析個股
使用者在對話中提到想看的個股時：跑 `python -m src.cli research <代號>`（含趨勢、策略體檢、大盤比較、新聞），依 `/tw-research` 的規則整理多空證據；只描述資料與歷史證據，不給買賣建議

## 工作規則
- 每次結束工作前更新 `progress.md`（做了什麼、測試結果、待辦）。

## 模型分工與提交審查
- 主會話用 Opus 5.5（`.claude/settings.json` 的 `model`）：只負責規劃、委派、讀子代理的摘要、簽核；不自己大量讀程式或改程式
- `explorer`（Haiku）：搜尋程式碼庫，回少於 200 字的摘要
- `fixer`（Sonnet）：做最小修補
- `checker`（Sonnet）：審查修補對呼叫者的影響，回報回歸問題（最後一行 `REGRESSIONS: 有 / 無`）
- 每次 Edit / Write `.py` 之後，PostToolUse hook 自動跑 pytest，失敗時把輸出回傳給剛做修改的那一方
- Opus 子代理 `diff-reviewer`：每次 commit 前審查暫存差異，回 `VERDICT: APPROVE / REJECT` 與 `DIFF_SHA256`
- 沒有 APPROVE 不准提交。流程：`git add` 指定檔案 → 呼叫 `diff-reviewer` → APPROVE 後 `python .claude/hooks/review_gate.py approve <DIFF_SHA256>` → `git commit -m ...`
- git 的 `.git/hooks/pre-commit`（`review_gate.py precommit`）在寫入 commit 時比對實際要提交的內容和核准雜湊，不符就擋；使用者手動 commit 也適用（本人可用 `--no-verify` 跳過）
- Claude 的 PreToolUse hook（`review_gate.py check`）擋下 `--no-verify`、改 hooksPath、動 `.git/hooks` 或核准標記；REJECT 時修好再重審，不可自己寫核准標記繞過
