"""Rebuild all data with one command: download every raw source, then build the player-week table.

Usage:
    python -m src.ingest                       # everything, 2020 through the current season
    python -m src.ingest --seasons 2026        # just this season (e.g. a weekly refresh)
    python -m src.ingest --no-download         # rebuild processed tables from the raw files on disk

Order matters: snap counts need player stats (to link IDs), and in-game injuries need play-by-play.
"""

import argparse
import time

from src.ingest import availability, in_game_injuries, pbp, player_stats, player_weeks, schedules

DOWNLOADS = [
    ("weekly player stats", player_stats.ingest),
    ("schedules + Vegas lines", schedules.ingest),
    ("snap counts + injury reports", availability.ingest),
    ("play-by-play", pbp.ingest),
    ("weekly rosters", in_game_injuries.download_rosters),
]


def run(seasons: list[int], download: bool = True) -> None:
    steps = DOWNLOADS if download else []
    steps = [*steps, ("in-game injuries", lambda s: in_game_injuries.ingest(s, download=False))]
    for name, step in steps:
        start = time.perf_counter()
        print(f"\n== {name} ==")
        step(seasons)
        print(f"   ({time.perf_counter() - start:.0f}s)")

    print("\n== player-week table ==")
    # the table always spans every season, so a one-season refresh still rebuilds it fully
    player_weeks.build(player_stats.default_seasons())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seasons", type=int, nargs="+", default=None)
    parser.add_argument("--no-download", action="store_true", help="skip downloads; rebuild from raw files")
    args = parser.parse_args()
    run(args.seasons or player_stats.default_seasons(), download=not args.no_download)


if __name__ == "__main__":
    main()
