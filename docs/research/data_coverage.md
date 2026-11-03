# Play-by-play data coverage (SCRUM-19, SCRUM-47)

Measured 2026-11-03 on the files from `python -m src.ingest.pbp`.

## What we have

| Source | Files | Seasons | Rows/season |
|---|---|---|---|
| nflverse play-by-play | `data/raw/pbp/pbp_<season>.parquet` | 2020–2026 (2026 in progress) | ~49k plays, 370 columns |
| NGS participation | `data/raw/participation/` | 2020–2025 | ~46–51k plays |
| FTN charting | `data/raw/ftn/` | 2022–2026 | ~41–48k plays |

**Participation is published after each season ends**, so 2026 has none yet. That's fine for drafting: preseason projections only use completed seasons.

## Defensive columns (play-by-play, every season)

`defteam`, `epa`, `pass_location` (left/middle/right), `pass_length` (short/deep), `air_yards`, `sack`, `qb_hit`, `cpoe`, `xyac_epa`. All ~100% filled on pass plays. These are enough for the SCRUM-36 defensive ratings.

## Coverage and pass rush (participation), share of dropbacks filled

| Season | Players on field | Man/zone | Pressure | # pass rushers |
|---|---|---|---|---|
| 2020 | 100% | 89% | 90% | 99% |
| 2021 | 100% | 89% | 89% | 99% |
| 2022 | 100% | 89% | 89% | 99% |
| 2023 | 100% | 100% | 100% | 100% |
| 2024 | 100% | 100% | 100% | 100% |
| 2025 | 100% | 100% | 100% | 100% |

`defense_coverage_type` (Cover 1/2/3...) is also present at the same rates. FTN adds blitzers, play action, screens, and drops from 2022 on.

## Routes run (SCRUM-47 decision)

**Decision: build route features ourselves. No paid source needed.**

- `route` in participation is only the *targeted* receiver's route, not everyone's.
- But `offense_players` lists all 11 offensive players (gsis IDs, `;`-separated) on 100% of dropbacks.
- So **routes run ≈ dropbacks a WR/TE/RB was on the field for.** This slightly overcounts (it includes blocking snaps, which matters most for TEs and RBs), but it's the standard free approximation.
- That gives us **route participation** (routes ÷ team dropbacks), **targets per route run (TPRR)**, and **yards per route run (YPRR)** for 2020–2025.

## Gotchas

- `play_id` is a float in participation and an integer in play-by-play. Cast both to `Int64` before joining.
- Participation's game key is `nflverse_game_id`; play-by-play calls it `game_id`.
- Empty strings (not nulls) mean "no value" in participation's text columns.
