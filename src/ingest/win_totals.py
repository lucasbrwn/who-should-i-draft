"""Vegas preseason win totals (SCRUM-88): turn Lucas's SportsOddsHistory paste into a clean CSV.

Input:  data/reference/win_totals_paste.txt  (copied by hand, one block per season)
            25: Tm,Super Bowl Odds,W/L O-U,Record
            Arizona Cardinals,+5000,8.5,3-14 (under)
            ...
Output: data/reference/win_totals.csv  (season, team, win_total; nflverse team codes)

Only the W/L O-U column is kept. Record is that season's actual result (mid-season for the current
year), so using it as a preseason feature would leak the answer. Super Bowl odds are dropped because
the page doesn't say whether they're preseason or current.

Both files live in data/reference/ (committed), not data/raw/ (git-ignored), because the paste
can't be re-downloaded by the pipeline.

Usage:
    python -m src.ingest.win_totals
"""

import re
from pathlib import Path

import polars as pl

PASTE_PATH = Path("data/reference/win_totals_paste.txt")
OUT_PATH = Path("data/reference/win_totals.csv")

# Full team name on SportsOddsHistory -> nflverse code. Washington changed names in 2020 and 2022.
TEAM_CODES = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF", "Carolina Panthers": "CAR", "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE", "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC", "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LA", "Miami Dolphins": "MIA", "Minnesota Vikings": "MIN",
    "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF", "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN", "Washington Football Team": "WAS", "Washington Commanders": "WAS",
}

SEASON_LINE = re.compile(r"^(\d{2}|\d{4}):")  # "25:" or "2025:", header text may follow
SUM_TOLERANCE = 8  # how far a season's totals may drift from the real number of wins available


def parse_paste(text: str) -> pl.DataFrame:
    """One row per (season, team). Fails loudly on an unknown team name or a row before any season."""
    rows, season = [], None
    for line_no, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if m := SEASON_LINE.match(line):
            year = int(m.group(1))
            season = year + 2000 if year < 100 else year
            continue
        if not line or line.startswith("Tm,"):
            continue
        if season is None:
            raise ValueError(f"line {line_no}: team row before any season header: {line!r}")
        name, _sb_odds, win_total, *_record = line.split(",")
        if name not in TEAM_CODES:
            raise ValueError(f"line {line_no}: unknown team name {name!r}")
        try:
            value = float(win_total)
        except ValueError:
            raise ValueError(f"line {line_no}: {name} has no usable win total ({win_total!r})") from None
        rows.append({"season": season, "team": TEAM_CODES[name], "win_total": value})
    return pl.DataFrame(rows, schema={"season": pl.Int32, "team": pl.String, "win_total": pl.Float64})


def validate(df: pl.DataFrame) -> None:
    """Raise if any season looks wrong: missing/duplicate teams, impossible totals, or an unbalanced market."""
    problems = []
    for (season,), g in df.group_by("season", maintain_order=True):
        if g["team"].n_unique() != 32 or g.height != 32:
            problems.append(f"{season}: {g.height} rows / {g['team'].n_unique()} teams, expected 32")
        games = 16 if season <= 2020 else 17
        if not g["win_total"].is_between(0, games).all():
            problems.append(f"{season}: a win total is outside 0-{games}")
        # Every game has one winner, so the league hands out 32 * games / 2 wins. A balanced market's
        # totals should add up to about that; a typo (95 instead of 9.5) breaks it.
        expected, total = 16 * games, g["win_total"].sum()
        if abs(total - expected) > SUM_TOLERANCE:
            problems.append(f"{season}: totals sum to {total}, expected about {expected}")
    if problems:
        raise ValueError("win totals failed validation:\n  " + "\n  ".join(problems))


def build(paste_path: Path = PASTE_PATH, out_path: Path = OUT_PATH) -> pl.DataFrame:
    df = parse_paste(paste_path.read_text(encoding="utf-8")).sort("season", "team")
    validate(df)
    df.write_csv(out_path)
    return df


if __name__ == "__main__":
    df = build()
    print(f"wrote {OUT_PATH}: {df.height} rows, seasons {df['season'].min()}-{df['season'].max()}")
    for season, total in df.group_by("season").agg(pl.col("win_total").sum()).sort("season").rows():
        print(f"  {season}: totals sum to {total:g} (real wins available: {16 * (16 if season <= 2020 else 17)})")
