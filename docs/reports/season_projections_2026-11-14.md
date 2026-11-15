# Season projections backtest (2026-11-14)

Walk-forward: each season's boards use earlier seasons only. 2025 untouched. Standard PPR.
Spearman correlation with actual season points, within position, on the draft pool
(players any board ranks top 2x starters: QB 48, RB 72, WR 96, TE 36). Higher = better.

| position | naive | a_season | season_b | B - naive [95% CI over seasons x positions] |
| --- | --- | --- | --- | --- |
| QB | 0.487 | 0.168 | 0.709 | 0.222 [0.079, 0.390] |
| RB | 0.568 | 0.554 | 0.624 | 0.057 [-0.050, 0.182] |
| WR | 0.550 | 0.527 | 0.655 | 0.105 [0.028, 0.146] |
| TE | 0.529 | 0.492 | 0.514 | -0.015 [-0.257, 0.215] |
| ALL | 0.533 | 0.435 | 0.626 | 0.092 [0.005, 0.174] |

Per season:

| season | position | players | naive | a_season | season_b |
| --- | --- | --- | --- | --- | --- |
| 2022 | QB | 67 | 0.640 | -0.126 | 0.720 |
| 2022 | RB | 92 | 0.559 | 0.526 | 0.598 |
| 2022 | TE | 54 | 0.677 | 0.593 | 0.420 |
| 2022 | WR | 128 | 0.546 | 0.516 | 0.574 |
| 2023 | QB | 60 | 0.459 | 0.287 | 0.656 |
| 2023 | RB | 87 | 0.451 | 0.491 | 0.633 |
| 2023 | TE | 46 | 0.398 | 0.420 | 0.613 |
| 2023 | WR | 127 | 0.537 | 0.547 | 0.678 |
| 2024 | QB | 62 | 0.362 | 0.344 | 0.752 |
| 2024 | RB | 88 | 0.692 | 0.645 | 0.642 |
| 2024 | TE | 46 | 0.511 | 0.462 | 0.509 |
| 2024 | WR | 130 | 0.566 | 0.518 | 0.712 |
