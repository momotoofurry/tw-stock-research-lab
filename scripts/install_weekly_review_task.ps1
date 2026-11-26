# 註冊 Windows 排程：每週六 10:00 推播預測檢討報告（python -m src.cli review --push）
# 只用目前使用者身分，不需要系統管理員權限；錯過時間會在下次開機後補跑。
# 移除：Unregister-ScheduledTask -TaskName "TWStockWeeklyReview" -Confirm:$false
$root = Split-Path -Parent $PSScriptRoot
$py = Join-Path $root ".venv\Scripts\pythonw.exe"
$action = New-ScheduledTaskAction -Execute $py -Argument "-m src.cli review --push" -WorkingDirectory $root
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Saturday -At 10:00
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Hours 1) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
Register-ScheduledTask -TaskName "TWStockWeeklyReview" -Description "台股研究工具：每週預測檢討報告推播（不下單）" -Action $action -Trigger $trigger -Settings $settings -Force | Out-Null
Get-ScheduledTask -TaskName "TWStockWeeklyReview" | Select-Object TaskName, State
