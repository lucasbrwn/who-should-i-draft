# Who Should I Draft?

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
