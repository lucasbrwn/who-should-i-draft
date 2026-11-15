# Season projections backtest (2026-11-14)

Walk-forward: each season's boards use earlier seasons only. 2025 untouched. Standard PPR.
Spearman correlation with actual season points, within position, on the draft pool
(players any board ranks top 2x starters: QB 48, RB 72, WR 96, TE 36). Higher = better.

| position | naive | a_season | season_b | B - naive [95% CI over seasons x positions] |
| --- | --- | --- | --- | --- |
| QB | 0.495 | 0.187 | 0.724 | 0.229 [0.077, 0.391] |
| RB | 0.575 | 0.553 | 0.636 | 0.061 [-0.047, 0.177] |
| WR | 0.548 | 0.523 | 0.645 | 0.097 [0.048, 0.139] |
| TE | 0.527 | 0.500 | 0.507 | -0.020 [-0.285, 0.228] |
| ALL | 0.536 | 0.441 | 0.628 | 0.092 [-0.001, 0.176] |

Per season:

| season | position | players | naive | a_season | season_b |
| --- | --- | --- | --- | --- | --- |
| 2022 | QB | 67 | 0.648 | -0.126 | 0.725 |
| 2022 | RB | 90 | 0.570 | 0.517 | 0.623 |
| 2022 | TE | 51 | 0.715 | 0.648 | 0.430 |
| 2022 | WR | 130 | 0.547 | 0.490 | 0.595 |
| 2023 | QB | 60 | 0.459 | 0.295 | 0.678 |
| 2023 | RB | 86 | 0.445 | 0.485 | 0.622 |
| 2023 | TE | 44 | 0.349 | 0.424 | 0.577 |
| 2023 | WR | 125 | 0.537 | 0.548 | 0.642 |
| 2024 | QB | 63 | 0.378 | 0.393 | 0.769 |
| 2024 | RB | 87 | 0.709 | 0.658 | 0.661 |
| 2024 | TE | 46 | 0.518 | 0.428 | 0.514 |
| 2024 | WR | 129 | 0.560 | 0.531 | 0.699 |
