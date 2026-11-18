"""Save this week's NFL player prop lines (The Odds API), building our own history from 2026 week 11.

Why: props (anytime-TD odds, yardage lines) are the sharpest public forecast of individual players,
and TDs are our biggest error. Historical props are paid-only, so we start our own archive now; once
there's a season or more of it, props can be TESTED as a model input (until then: display only).

Setup (once): get a free key at https://the-odds-api.com and put it in a .env file in the repo root
(already gitignored):  ODDS_API_KEY=your_key   (or set it as an environment variable)

Usage (weekly, e.g. Saturday after Friday's injury report, alongside python -m src.predict):
    python -m src.ingest.props --season 2026 --week 11
    python -m src.ingest.props --season 2026 --week 11 --dry-run     # list games, spend no credits

Cost: listing games is free; each game costs (markets returned x regions) credits, about
6 x 1 x 14 games = ~84 per weekly pull. Each run is saved separately (append-only, timestamped):
    data/raw/props/props_<season>_wk<week>_<UTC time>.parquet
"""

import argparse
import json
import os
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import nflreadpy
import polars as pl

from src.features.team import read_schedules

API = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl"
MARKETS = ["player_anytime_td", "player_rush_yds", "player_reception_yds", "player_receptions",
           "player_pass_yds", "player_pass_tds"]
REGION = "us"
RAW_DIR = Path("data/raw/props")
ROSTERS_DIR = Path("data/raw/rosters")


def api_key() -> str:
    key = os.environ.get("ODDS_API_KEY")
    env = Path(".env")
    if not key and env.exists():
        for line in env.read_text().splitlines():
            if line.strip().startswith("ODDS_API_KEY="):
                key = line.split("=", 1)[1].strip()
    if not key:
        raise SystemExit("No ODDS_API_KEY: add ODDS_API_KEY=... to .env (see this module's docstring)")
    return key


def get(url: str, params: dict) -> tuple[object, dict]:
    with urllib.request.urlopen(f"{url}?{urllib.parse.urlencode(params)}", timeout=60) as resp:
        return json.load(resp), dict(resp.headers)


def team_codes() -> dict[str, str]:
    """Full team name (as the odds API writes it) -> nflverse abbreviation."""
    teams = nflreadpy.load_teams()
    return dict(zip(teams["team_name"].to_list(), teams["team_abbr"].to_list()))


def week_events(events: list[dict], schedules: pl.DataFrame, season: int, week: int) -> list[dict]:
    """The odds API's events that are this week's games, tagged with our game_id."""
    codes = team_codes()
    games = schedules.filter((pl.col("season") == season) & (pl.col("week") == week))
    by_teams = {(r["away_team"], r["home_team"]): r["game_id"] for r in games.iter_rows(named=True)}
    out = []
    for e in events:
        key = (codes.get(e["away_team"]), codes.get(e["home_team"]))
        if key in by_teams:
            out.append({**e, "game_id": by_teams[key]})
    return out


def flatten(event_odds: dict, game_id: str, fetched: str) -> list[dict]:
    """One row per bookmaker x market x player x outcome (Over/Under/Yes) with line and price."""
    rows = []
    for book in event_odds.get("bookmakers", []):
        for market in book.get("markets", []):
            for o in market.get("outcomes", []):
                rows.append({"game_id": game_id, "bookmaker": book["key"], "market": market["key"],
                             "player_name": o.get("description"), "outcome": o.get("name"),
                             "line": o.get("point"), "price": o.get("price"),
                             "book_updated": market.get("last_update"), "fetched_utc": fetched})
    return rows


def name_key(name: pl.Expr) -> pl.Expr:
    return (name.str.to_lowercase().str.replace_all(r"[^a-z ]", "")
            .str.replace_all(r"\b(jr|sr|ii|iii|iv)\b", "").str.strip_chars().str.replace_all(r"\s+", " "))


def attach_ids(props: pl.DataFrame, season: int, week: int) -> pl.DataFrame:
    """Match prop player names to gsis ids using that week's rosters (names unique in the week only)."""
    rosters = pl.read_parquet(ROSTERS_DIR / f"rosters_{season}.parquet").filter(pl.col("week") == week)
    names = (rosters.select(k=name_key(pl.col("full_name")), player_id="gsis_id").drop_nulls().unique()
             .filter(pl.len().over("k") == 1))
    return props.with_columns(k=name_key(pl.col("player_name"))).join(names, on="k", how="left").drop("k")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--dry-run", action="store_true", help="list this week's games only (free)")
    parser.add_argument("--games", nargs="+", default=None, help="only these game_ids (e.g. to test cheaply)")
    args = parser.parse_args()
    key = api_key()
    events, _ = get(f"{API}/events", {"apiKey": key})
    week = week_events(events, read_schedules(), args.season, args.week)
    if args.games:
        week = [e for e in week if e["game_id"] in args.games]
    print(f"{len(week)} games this week found in the odds API")
    if args.dry_run:
        for e in week:
            print(f"  {e['game_id']}  {e['commence_time']}")
        return
    fetched = datetime.now(timezone.utc).isoformat()
    rows, remaining = [], None
    for e in week:
        data, headers = get(f"{API}/events/{e['id']}/odds",
                            {"apiKey": key, "regions": REGION, "markets": ",".join(MARKETS), "oddsFormat": "american"})
        rows += flatten(data, e["game_id"], fetched)
        remaining = headers.get("x-requests-remaining", remaining)
    if not rows:
        print("No prop lines returned (books may not have posted them yet). No credits spent on empty games.")
        return
    props = attach_ids(pl.DataFrame(rows), args.season, args.week).with_columns(
        season=pl.lit(args.season), week=pl.lit(args.week))
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"props_{args.season}_wk{args.week:02d}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.parquet"
    props.write_parquet(path)
    matched = props.filter(pl.col("player_id").is_not_null())["player_name"].n_unique()
    print(f"Saved {props.height:,} prop lines ({props['player_name'].n_unique()} players, {matched} matched to ids, "
          f"{props['bookmaker'].n_unique()} books) -> {path}. Credits remaining: {remaining}")


if __name__ == "__main__":
    main()
