# LINE 推播設定（約 10 分鐘）

LINE Notify 已在 2025-03-31 停止服務，所以改用 **LINE 官方帳號 + Messaging API**。免費方案每月 200 則推播，本工具每個交易日約 1–2 則，夠用。

## 1. 建立官方帳號與 Messaging API channel
1. 到 LINE Developers（developers.line.biz）用你的 LINE 帳號登入
2. 建立一個 Provider（名字隨意，例如「我的台股工具」）
3. 在 Provider 底下建立 **Messaging API** channel（會同時建立一個 LINE 官方帳號）
4. 進入 channel：
   - **Basic settings** 分頁最下方的 **Your user ID**（U 開頭的一串）→ 這是 `LINE_USER_ID`
   - **Messaging API** 分頁：用手機掃 QR code 把官方帳號**加為好友**（沒加好友收不到推播）
   - **Messaging API** 分頁最下方 **Channel access token (long-lived)** 按 Issue → 這是 `LINE_CHANNEL_ACCESS_TOKEN`
5. 建議到 LINE Official Account Manager 把「自動回應訊息」關掉，避免它回你罐頭訊息

## 2. 設定環境變數（token 不要寫進任何檔案）
在 PowerShell 執行（把 `...` 換成你的值）：

```powershell
setx LINE_CHANNEL_ACCESS_TOKEN "..."
setx LINE_USER_ID "U..."
```

`setx` 設定的是使用者環境變數，**之後新開的視窗與排程才讀得到**。順便建議也設定 FinMind token（免費註冊，額度較高）：

```powershell
setx FINMIND_TOKEN "..."
```

## 3. 測試
開一個**新的**終端機視窗：

```powershell
cd D:\stock
.venv\Scripts\python -m src.cli line-test
```

手機收到「LINE 推播測試成功」就完成了。

## 排程
已註冊 Windows 排程 `TWStockDaily`：每個平日 17:30 執行 `python -m src.cli daily`。錯過時間（關機、睡眠）會在下次開機後補跑。紀錄在 `data/daily.log`。

- 收到什麼：每天一則收盤摘要（`config.yaml` → `notify.daily_summary`）＋ 你設定的規則**新觸發**時另一則警示
- 移除排程：`powershell -ExecutionPolicy Bypass -File scripts\uninstall_daily_task.ps1`
- 手動跑一次：`.venv\Scripts\python -m src.cli daily`
