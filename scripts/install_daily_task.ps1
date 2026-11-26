# 註冊 Windows 排程：每個平日 17:30 執行收盤後例行工作（python -m src.cli daily）
# 只用目前使用者身分、登入時才執行，不需要系統管理員權限。錯過時間（關機 / 睡眠）會在下次開機後補跑。
# 移除：powershell -ExecutionPolicy Bypass -File scripts\uninstall_daily_task.ps1
$root = Split-Path -Parent $PSScriptRoot
$py = Join-Path $root ".venv\Scripts\pythonw.exe"
$action = New-ScheduledTaskAction -Execute $py -Argument "-m src.cli daily" -WorkingDirectory $root
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At 17:30
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Hours 2) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
Register-ScheduledTask -TaskName "TWStockDaily" -Description "台股研究工具：收盤後前瞻紀錄、持股摘要、規則警示、LINE 推播（不下單）" -Action $action -Trigger $trigger -Settings $settings -Force | Out-Null
Get-ScheduledTask -TaskName "TWStockDaily" | Select-Object TaskName, State
