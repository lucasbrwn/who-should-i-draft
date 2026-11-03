# Player-week table: data report

Generated 2026-11-03 by `python scripts/report_player_weeks.py` from `data/processed/player_weeks.parquet`.

**42,832 player-weeks**, 94 columns, seasons 2020–2026. Checks: all passed.

## Rows by season and position

| season | QB | RB | WR | TE | total | last week | snap-only % |
|---|---|---|---|---|---|---|---|
| 2020 | 631 | 1,575 | 2,559 | 1,531 | 6,296 | 17 | 12.0 |
| 2021 | 670 | 1,617 | 2,772 | 1,640 | 6,699 | 18 | 12.4 |
| 2022 | 647 | 1,668 | 2,680 | 1,641 | 6,636 | 18 | 12.5 |
| 2023 | 680 | 1,590 | 2,804 | 1,614 | 6,688 | 18 | 13.3 |
| 2024 | 682 | 1,624 | 2,735 | 1,662 | 6,703 | 18 | 12.5 |
| 2025 | 674 | 1,641 | 2,775 | 1,681 | 6,771 | 18 | 10.8 |
| 2026 | 284 | 725 | 1,229 | 801 | 3,039 | 8 | 11.1 |

*Snap-only* = played on offense but recorded no stat, so nflverse has no row for it. These are real zero games (busts), and they'd be missing without this table.

## Biggest bust games (WR/TE on 75%+ of snaps)

| season | week | player | position | team | snap % | targets | PPR |
|---|---|---|---|---|---|---|---|
| 2021 | 16 | Equanimeous St. Brown | WR | GB | 81% | 0 | -1.0 |
| 2024 | 2 | Cade Otton | TE | TB | 100% | 2 | -0.4 |
| 2020 | 13 | Marquez Valdes-Scantling | WR | GB | 75% | 2 | -0.4 |
| 2023 | 11 | Xavier Gipson | WR | NYJ | 79% | 2 | -0.3 |
| 2025 | 13 | Chimere Dike | WR | TEN | 76% | 4 | -0.1 |
| 2022 | 14 | Foster Moreau | TE | LV | 100% | 1 | 0.0 |
| 2023 | 9 | Alec Pierce | WR | IND | 100% | 2 | 0.0 |
| 2025 | 9 | Rome Odunze | WR | CHI | 100% | 3 | 0.0 |

## Injury report status of players who played

Players ruled Out mostly don't appear (they didn't play). Blank = not on the game-status report.

| report_status | player-weeks |
|---|---|
| (none) | 41,067 |
| Questionable | 1,759 |
| Doubtful | 3 |
| Note | 2 |
| Out | 1 |

## Does `left_injured` find real injuries?

If the flag is right, flagged players should miss their next game far more often than everyone else.

| group | games | missed next game % |
|---|---|---|
| flagged left_injured | 272 | 76.1 |
| injury in play text, normal snaps | 992 | 32.3 |
| starter, snaps < half usual, no injury text | 1,255 | 30.0 |
| everyone else | 37,543 | 10.2 |

Groups 2 and 3 are why the flag needs *both* signals: injury text alone includes players who returned to the game, and a snap drop alone is mostly benchings and QB changes.

| season | left injured | injury in play text |
|---|---|---|
| 2020 | 38 | 179 |
| 2021 | 49 | 220 |
| 2022 | 33 | 181 |
| 2023 | 43 | 237 |
| 2024 | 45 | 201 |
| 2025 | 48 | 227 |
| 2026 | 34 | 106 |
