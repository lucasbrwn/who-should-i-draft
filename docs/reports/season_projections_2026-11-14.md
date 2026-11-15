# Season projections backtest (2026-11-14)

Walk-forward: each season's boards use earlier seasons only. 2025 untouched. Standard PPR.
Spearman correlation with actual season points, within position, on the draft pool
(players any board ranks top 2x starters: QB 48, RB 72, WR 96, TE 36). Higher = better.

| position | naive | a_season | season_b | B - naive [95% CI over seasons x positions] |
| --- | --- | --- | --- | --- |
| QB | 0.495 | 0.170 | 0.712 | 0.217 [0.071, 0.373] |
| RB | 0.580 | 0.568 | 0.627 | 0.047 [-0.062, 0.151] |
| WR | 0.568 | 0.538 | 0.680 | 0.112 [0.046, 0.156] |
| TE | 0.479 | 0.461 | 0.487 | 0.007 [-0.223, 0.220] |
| ALL | 0.531 | 0.435 | 0.626 | 0.096 [0.015, 0.173] |

Per season:

| season | position | players | naive | a_season | season_b |
| --- | --- | --- | --- | --- | --- |
| 2022 | QB | 67 | 0.648 | -0.116 | 0.719 |
| 2022 | RB | 88 | 0.580 | 0.532 | 0.630 |
| 2022 | TE | 50 | 0.674 | 0.629 | 0.451 |
| 2022 | WR | 127 | 0.574 | 0.524 | 0.620 |
| 2023 | QB | 60 | 0.459 | 0.283 | 0.665 |
| 2023 | RB | 89 | 0.471 | 0.531 | 0.622 |
| 2023 | TE | 45 | 0.324 | 0.390 | 0.543 |
| 2023 | WR | 126 | 0.567 | 0.554 | 0.701 |
| 2024 | QB | 63 | 0.378 | 0.344 | 0.752 |
| 2024 | RB | 88 | 0.689 | 0.642 | 0.627 |
| 2024 | TE | 43 | 0.441 | 0.364 | 0.465 |
| 2024 | WR | 130 | 0.564 | 0.537 | 0.719 |
