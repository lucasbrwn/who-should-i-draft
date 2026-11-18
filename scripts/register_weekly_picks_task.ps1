# Register (or replace) the scheduled task for weekly predictions. Local (Pacific) times:
#   Thursday 2:00 PM  lock picks before Thursday night kickoff (5:15 PM PT)
#   Saturday 10:00 AM re-run with Friday's final injury report for Sunday/Monday games, + prop lines
# Missed runs (computer off/asleep) run when it's back on; check the timestamp in the run's meta.txt
# against kickoff before counting a late run as "locked".

$taskName = "WhoShouldIDraft - Weekly Picks"
$script = Join-Path $PSScriptRoot "weekly_picks.ps1"
$shell = (Get-Command pwsh -ErrorAction SilentlyContinue).Source
if (-not $shell) { $shell = (Get-Command powershell).Source }

$action = New-ScheduledTaskAction -Execute $shell `
    -Argument "-NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$script`""
$triggers = @(
    New-ScheduledTaskTrigger -Weekly -DaysOfWeek Thursday -At "2:00PM"
    New-ScheduledTaskTrigger -Weekly -DaysOfWeek Saturday -At "10:00AM"
)
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Minutes 45)

Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $triggers -Settings $settings `
    -Description "Weekly Who Should I Draft predictions, benchmark snapshots and prop lines" -Force | Out-Null

Get-ScheduledTask -TaskName $taskName | Get-ScheduledTaskInfo | Select-Object TaskName, NextRunTime
