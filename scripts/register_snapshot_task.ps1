# Register (or replace) the Windows scheduled task that snapshots Vegas lines.
# Times are local (Pacific): daily 9:00 AM, plus shortly before each primetime/late slot.
# Missed runs (computer off/asleep) run as soon as the computer is available again.

$taskName = "WhoShouldIDraft - Line Snapshot"
$script = Join-Path $PSScriptRoot "snapshot_lines.ps1"
$shell = (Get-Command pwsh -ErrorAction SilentlyContinue).Source
if (-not $shell) { $shell = (Get-Command powershell).Source }

$action = New-ScheduledTaskAction -Execute $shell `
    -Argument "-NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$script`""

$triggers = @(
    New-ScheduledTaskTrigger -Daily -At "9:00AM"
    New-ScheduledTaskTrigger -Weekly -DaysOfWeek Thursday -At "4:30PM"
    New-ScheduledTaskTrigger -Weekly -DaysOfWeek Sunday -At "12:30PM"
    New-ScheduledTaskTrigger -Weekly -DaysOfWeek Sunday -At "4:30PM"
    New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday -At "4:30PM"
)

$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Minutes 15)

Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $triggers -Settings $settings `
    -Description "Snapshot NFL Vegas lines for the Who Should I Draft project" -Force | Out-Null

Get-ScheduledTask -TaskName $taskName | Get-ScheduledTaskInfo |
    Select-Object TaskName, NextRunTime
