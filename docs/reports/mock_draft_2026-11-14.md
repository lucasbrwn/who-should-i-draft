# Mock-draft test (2026-11-14)

Standard PPR, 12 teams, 2022-24 (2025 untouched). Opponents draft from preseason ECR + noise. Scored on each team's actual season (best lineup every week). Rules in src/backtest/mock_draft.py.

| drafter | drafts | avg points | edge vs league avg [95% CI] | finishes 1st | avg rank |
| --- | --- | --- | --- | --- | --- |
| sim | 360 | 1819 | -44.698 [-68.736, -21.770] | 3.1% | 6.76 |
| vor | 360 | 1732 | -133.665 [-161.150, -108.003] | 1.4% | 7.92 |
| adp | 360 | 1870 | 23.437 [1.339, 44.901] | 13.1% | 6.03 |

Paired differences (same drafts):

| comparison | edge difference [95% CI] |
| --- | --- |
| sim - vor | 88.967 [68.681, 109.115] |
| sim - adp | -68.135 [-96.875, -39.310] |
| vor - adp | -157.102 [-188.307, -125.509] |

By season (edge vs league average):

| season | sim | vor | adp |
| --- | --- | --- | --- |
| 2022 | 103 | -21 | 153 |
| 2023 | 43 | 83 | -26 |
| 2024 | -280 | -462 | -56 |
