# 登入 Windows 時自動啟動 LINE 指令伺服器與 ngrok（目前使用者、不需管理員權限）
# 移除：Unregister-ScheduledTask -TaskName "TWStockLineBot" -Confirm:$false
$root = Split-Path -Parent $PSScriptRoot
$script = Join-Path $PSScriptRoot "start_line_bot.ps1"
$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$script`"" -WorkingDirectory $root
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Seconds 0)
Register-ScheduledTask -TaskName "TWStockLineBot" -Description "台股研究工具：LINE 指令伺服器（設定規則、查詢、記帳；不下單）" -Action $action -Trigger $trigger -Settings $settings -Force | Out-Null
Get-ScheduledTask -TaskName "TWStockLineBot" | Select-Object TaskName, State
