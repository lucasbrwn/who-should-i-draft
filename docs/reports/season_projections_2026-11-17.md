# Season projections backtest (2026-11-17)

Walk-forward: each season's boards use earlier seasons only. 2025 untouched. Standard PPR.
Spearman correlation with actual season points, within position, on the draft pool
(players any board ranks top 2x starters: QB 48, RB 72, WR 96, TE 36). Higher = better.

| position | naive | a_season | season_b | B - naive [95% CI over seasons x positions] |
| --- | --- | --- | --- | --- |
| QB | 0.514 | 0.206 | 0.732 | 0.218 [0.053, 0.421] |
| RB | 0.533 | 0.505 | 0.684 | 0.151 [0.010, 0.235] |
| WR | 0.556 | 0.526 | 0.724 | 0.168 [0.144, 0.203] |
| TE | 0.478 | 0.449 | 0.628 | 0.150 [-0.057, 0.348] |
| ALL | 0.520 | 0.421 | 0.692 | 0.172 [0.098, 0.248] |

Per season:

| season | position | players | naive | a_season | season_b |
| --- | --- | --- | --- | --- | --- |
| 2022 | QB | 71 | 0.670 | -0.028 | 0.722 |
| 2022 | RB | 94 | 0.509 | 0.454 | 0.717 |
| 2022 | TE | 44 | 0.674 | 0.622 | 0.617 |
| 2022 | WR | 123 | 0.555 | 0.499 | 0.711 |
| 2023 | QB | 58 | 0.493 | 0.306 | 0.674 |
| 2023 | RB | 92 | 0.460 | 0.487 | 0.696 |
| 2023 | TE | 48 | 0.312 | 0.355 | 0.661 |
| 2023 | WR | 129 | 0.559 | 0.553 | 0.702 |
| 2024 | QB | 60 | 0.379 | 0.339 | 0.800 |
| 2024 | RB | 91 | 0.628 | 0.575 | 0.638 |
| 2024 | TE | 42 | 0.448 | 0.370 | 0.607 |
| 2024 | WR | 131 | 0.555 | 0.525 | 0.758 |
