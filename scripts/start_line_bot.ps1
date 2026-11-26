# 啟動 LINE 指令伺服器（127.0.0.1:8787）與 ngrok 通道。設定方式見 docs\LINE指令設定.md
$root = Split-Path -Parent $PSScriptRoot
$cfg = Get-Content (Join-Path $root "config.yaml") -Encoding UTF8 -Raw
$m = [regex]::Match($cfg, '(?m)^\s*public_domain:\s*"?([^"\s#]+)')
if (-not $m.Success -or $m.Groups[1].Value -eq "") {
    Write-Host "config.yaml 的 line_bot.public_domain 還沒填（ngrok 免費固定網域，例如 xxx.ngrok-free.app）"
    exit 1
}
$domain = $m.Groups[1].Value
$port = 8787
# 已經在跑就不重複啟動
if (-not (Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue)) {
    Start-Process (Join-Path $root ".venv\Scripts\pythonw.exe") -ArgumentList "-m src.cli line-bot" -WorkingDirectory $root -WindowStyle Hidden
}
if (-not (Get-Process ngrok -ErrorAction SilentlyContinue)) {
    # winget 安裝的 ngrok 不一定在 PATH，依序找：PATH → WinGet Links → WinGet Packages
    $ngrok = (Get-Command ngrok -ErrorAction SilentlyContinue).Source
    if (-not $ngrok) { $ngrok = Join-Path $env:LOCALAPPDATA "Microsoft\WinGet\Links\ngrok.exe" }
    if (-not (Test-Path $ngrok)) {
        $ngrok = Get-ChildItem (Join-Path $env:LOCALAPPDATA "Microsoft\WinGet\Packages") -Recurse -Filter ngrok.exe -ErrorAction SilentlyContinue |
            Select-Object -First 1 -ExpandProperty FullName
    }
    if (-not $ngrok) { Write-Host "找不到 ngrok.exe，請先安裝（docs\LINE指令設定.md）"; exit 1 }
    # --domain 新舊版 ngrok 都支援（3.3 這類舊版沒有 --url）
    Start-Process $ngrok -ArgumentList "http --domain=$domain $port" -WindowStyle Hidden
}
"LINE 指令伺服器已啟動：https://$domain → 127.0.0.1:$port"
