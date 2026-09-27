# Snapshot current Vegas lines for upcoming games. Run by the "WhoShouldIDraft - Line Snapshot"
# scheduled task (see scripts/register_snapshot_task.ps1); safe to run by hand too.

$repo = Split-Path -Parent $PSScriptRoot
$python = Join-Path $env:USERPROFILE ".venvs\whoshouldidraft\Scripts\python.exe"
$log = Join-Path $repo "logs\line_snapshots.log"

New-Item -ItemType Directory -Force (Split-Path $log) | Out-Null
Set-Location $repo

"=== $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ===" | Add-Content $log
& $python -m src.ingest.schedules --current 2>&1 | ForEach-Object { "$_" } | Add-Content $log
"exit code: $LASTEXITCODE" | Add-Content $log
exit $LASTEXITCODE
