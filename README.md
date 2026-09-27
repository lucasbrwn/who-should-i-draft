# Front Office

**Your league. Your data. Your advantage.**

*Course project: "Who Should I Draft?"*

An AI fantasy football draft assistant. It projects player performance from real stats, usage, and defensive matchups, then recommends the best pick for your league's settings — with the reasons behind every pick.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS/Linux
pip install -r requirements.txt
pytest
```

> If the repo lives in a synced folder (OneDrive, Dropbox), create the virtual environment outside it, e.g. `python -m venv %USERPROFILE%\.venvs\whoshouldidraft`, so thousands of package files aren't synced.

## Web app

```powershell
.\scripts\run_app.ps1                      # Windows, from the project folder
```

or, with the virtual environment activated (`...\Scripts\Activate.ps1` in PowerShell, `source .venv/bin/activate` on macOS/Linux), from the project folder:

```bash
flask --app app run --debug
```

Then open http://127.0.0.1:5000. `--debug` reloads automatically when you save a Python or template file (refresh the browser to see changes). League settings are saved as JSON in `data/leagues/`.

## Data

```bash
python -m src.ingest.player_stats     # weekly player stats, 2020+
python -m src.ingest.schedules        # schedules, Vegas lines, weather + line snapshot
python -m src.ingest.availability     # snap counts + injury reports (run after player_stats)
python -m src.ingest.crosscheck       # verify stats against play-by-play
```

**Line snapshots:** Vegas lines for upcoming games are appended to `data/raw/schedules/line_snapshots.parquet` with a timestamp (only when a line changes). On Windows, `scripts/register_snapshot_task.ps1` schedules this daily at 9 AM plus before the Thursday/Sunday/Monday game slots; output goes to `logs/line_snapshots.log`.

## Layout

| Path | Module |
|---|---|
| `src/ingest/` | M1 — data ingestion (nflverse, 2020+) |
| `src/scoring/` | M2 — league config + scoring engine |
| `src/features/` | M3 — feature engineering |
| `src/models/` | M4 — projection models |
| `src/backtest/` | M5 — backtest and validation |
| `src/draft/` | M6 — draft recommender |
| `app/` | M7 — Flask frontend |
| `notebooks/` | Research and exploration (M1.5) |
| `data/raw/`, `data/processed/` | Generated data (not committed) |
| `tests/` | pytest suite |

Planning and tasks are tracked in Jira (project SCRUM).
