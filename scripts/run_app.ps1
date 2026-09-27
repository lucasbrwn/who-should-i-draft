# Start the web app in debug mode (auto-restarts when you save a file). Works from any folder:
#   .\scripts\run_app.ps1          (from the project folder)
# Then open http://127.0.0.1:5000 and press Ctrl+C here to stop.

$repo = Split-Path -Parent $PSScriptRoot
$python = Join-Path $env:USERPROFILE ".venvs\whoshouldidraft\Scripts\python.exe"

if (-not (Test-Path $python)) {
    Write-Error "Project environment not found at $python. See README > Setup."
    exit 1
}

Set-Location $repo
& $python -m flask --app app run --debug
