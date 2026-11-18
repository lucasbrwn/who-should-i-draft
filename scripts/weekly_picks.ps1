# Weekly predictions, run by the "WhoShouldIDraft - Weekly Picks" scheduled task
# (scripts/register_weekly_picks_task.ps1). Safe to run by hand too.
#   1. refresh this season's data (stats, injury report, rosters, depth charts, Vegas lines)
#   2. predict the upcoming week + snapshot ESPN / FantasyPros at the same moment  -> predictions/
#   3. Saturdays (or -Props): save player prop lines (The Odds API, ~78 credits)    -> data/raw/props/
param([switch]$Props)

$repo = Split-Path -Parent $PSScriptRoot
$python = Join-Path $env:USERPROFILE ".venvs\whoshouldidraft\Scripts\python.exe"
$log = Join-Path $repo "logs\weekly_picks.log"
New-Item -ItemType Directory -Force (Split-Path $log) | Out-Null
Set-Location $repo
$env:PYTHONIOENCODING = "utf-8"

function Run-Step($name, [string[]]$pyArgs) {
    "--- $name ($(Get-Date -Format 'HH:mm:ss'))" | Add-Content $log
    & $python @pyArgs 2>&1 | ForEach-Object { "$_" } | Add-Content $log
    "exit code: $LASTEXITCODE" | Add-Content $log
    return $LASTEXITCODE
}

"=== $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ===" | Add-Content $log
$sw = (& $python -m src.predict.next_week).Trim().Split(" ")
$season, $week = $sw[0], $sw[1]
"upcoming: season $season week $week" | Add-Content $log

$code = Run-Step "refresh data" @("-m", "src.ingest", "--seasons", $season)
if ($code -ne 0) { "STOPPED: data refresh failed" | Add-Content $log; exit $code }
$code = Run-Step "predict" @("-m", "src.predict", "--season", $season, "--week", $week)
if ($Props -or (Get-Date).DayOfWeek -eq "Saturday") {
    $null = Run-Step "props" @("-m", "src.ingest.props", "--season", $season, "--week", $week)
}
exit $code
