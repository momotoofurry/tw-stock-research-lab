# 用 LINE 設定規則、查詢持股、記帳

LINE 傳來的訊息要能進到你的電腦，所以需要：電腦上的小伺服器（已寫好）＋ ngrok 通道（把網路上的固定網址轉到你的電腦）。
**電腦要開著、伺服器要在跑，指令才會有回應**；每天 17:30 的推播不受影響。

## 一次性設定（約 15 分鐘）

### 1. 重新發行 Channel secret 並設定
之前截圖露出過，先作廢：LINE Developers → 你的 channel → **Basic settings** → Channel secret 旁的 **Issue**。
把新的值設成環境變數（不要截圖、不要貼到任何地方）：

```powershell
setx LINE_CHANNEL_SECRET "..."
```

### 2. 安裝 ngrok 並拿一個免費固定網域
1. 到 ngrok.com 註冊（免費），照網站說明下載 Windows 版 `ngrok.exe`，放到 PATH 裡的資料夾（或 `C:\Windows`）
2. 在 ngrok 後台複製 **Authtoken**，在 PowerShell 執行：`ngrok config add-authtoken <你的 authtoken>`
3. ngrok 後台 **Domains** → 領一個免費固定網域（像 `xxxx.ngrok-free.app`）
4. 打開 `config.yaml`，把最下面 `line_bot:` 裡的 `public_domain: ""` 填上網域（網域不是密碼，可以寫在設定檔）：
   ```yaml
     public_domain: "xxxx.ngrok-free.app"
   ```

### 3. 啟動
```powershell
powershell -ExecutionPolicy Bypass -File D:\stock\scripts\start_line_bot.ps1
```

### 4. 把網址填到 LINE
LINE Developers → 你的 channel → **Messaging API** 分頁：
- **Webhook URL**：`https://xxxx.ngrok-free.app/callback` → 按 **Verify**，顯示 Success
- **Use webhook**：打開
- 到 LINE Official Account Manager → 設定 → 回應設定：**關閉「自動回應訊息」與「加入好友的歡迎訊息」**（不然它會回罐頭訊息）

### 5. 測試
在 LINE 對官方帳號傳「說明」，應該會收到指令清單。

### 6. 開機自動啟動（建議）
```powershell
powershell -ExecutionPolicy Bypass -File D:\stock\scripts\install_line_bot_task.ps1
```

## 指令一覽（傳「說明」也看得到）
| 類別 | 範例 |
|---|---|
| 全體規則 | 停損 10%、停利 30%、移動停損 15%、均線 60、權重上限 40%、組合回撤 15%、接近 3% |
| 個股規則 | 聯電 停損價 65、2303 停利價 90、鴻海 移動停損 20%、2317 均線 20 |
| 取消 | 取消 停損、鴻海 取消 移動停損 |
| 查詢 | 規則、持股 |
| 記帳 | 買 2317 1000 450、賣 2303 1000 80（股數、價格） |
| 其他 | 復原（回到上一次修改前）、說明 |

## 安全
- 每則訊息都用 Channel secret 驗證簽章，偽造的請求會被拒絕（403）
- 只回應你本人（LINE_USER_ID）；其他人加了官方帳號、傳什麼都不會有回應
- 伺服器只綁 127.0.0.1，對外只透過 ngrok 這一個網址；只能設規則、查詢、記帳，**沒有任何下單功能**
- 紀錄在 `data/line_bot.log`（只記指令前 60 字，不記憑證）
