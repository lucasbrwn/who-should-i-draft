# Season projections backtest (2026-11-14)

Walk-forward: each season's boards use earlier seasons only. 2025 untouched. Standard PPR.
Spearman correlation with actual season points, within position, on the draft pool
(players any board ranks top 2x starters: QB 48, RB 72, WR 96, TE 36). Higher = better.

| position | naive | a_season | season_b | B - naive [95% CI over seasons x positions] |
| --- | --- | --- | --- | --- |
| QB | 0.514 | 0.212 | 0.724 | 0.210 [0.033, 0.419] |
| RB | 0.535 | 0.511 | 0.688 | 0.152 [0.009, 0.237] |
| WR | 0.559 | 0.532 | 0.723 | 0.165 [0.141, 0.208] |
| TE | 0.484 | 0.457 | 0.634 | 0.151 [-0.039, 0.348] |
| ALL | 0.523 | 0.428 | 0.693 | 0.170 [0.097, 0.246] |

Per season:

| season | position | players | naive | a_season | season_b |
| --- | --- | --- | --- | --- | --- |
| 2022 | QB | 72 | 0.671 | -0.010 | 0.703 |
| 2022 | RB | 94 | 0.508 | 0.456 | 0.719 |
| 2022 | TE | 45 | 0.678 | 0.625 | 0.638 |
| 2022 | WR | 125 | 0.562 | 0.517 | 0.703 |
| 2023 | QB | 58 | 0.493 | 0.306 | 0.672 |
| 2023 | RB | 92 | 0.460 | 0.487 | 0.697 |
| 2023 | TE | 48 | 0.312 | 0.355 | 0.660 |
| 2023 | WR | 129 | 0.559 | 0.553 | 0.704 |
| 2024 | QB | 60 | 0.379 | 0.339 | 0.798 |
| 2024 | RB | 90 | 0.638 | 0.590 | 0.647 |
| 2024 | TE | 43 | 0.461 | 0.391 | 0.605 |
| 2024 | WR | 131 | 0.555 | 0.525 | 0.764 |
