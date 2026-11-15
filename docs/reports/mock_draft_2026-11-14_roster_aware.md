# Mock-draft test (2026-11-14)

Standard PPR, 12 teams, 2022-24 (2025 untouched). Opponents draft from preseason ECR + noise. Scored on each team's actual season (best lineup every week). Rules in src/backtest/mock_draft.py.

| drafter | drafts | avg points | edge vs league avg [95% CI] | finishes 1st | avg rank |
| --- | --- | --- | --- | --- | --- |
| sim | 360 | 1838 | -24.703 [-43.184, -5.759] | 2.8% | 6.63 |
| vor | 360 | 1780 | -88.028 [-112.931, -63.522] | 1.7% | 7.28 |
| adp | 360 | 1870 | 23.437 [1.339, 44.901] | 13.1% | 6.03 |

Paired differences (same drafts):

| comparison | edge difference [95% CI] |
| --- | --- |
| sim - vor | 63.325 [46.260, 81.531] |
| sim - adp | -48.140 [-72.664, -20.992] |
| vor - adp | -111.465 [-141.356, -82.726] |

By season (edge vs league average):

| season | sim | vor | adp |
| --- | --- | --- | --- |
| 2022 | 79 | 47 | 153 |
| 2023 | 54 | 89 | -26 |
| 2024 | -207 | -401 | -56 |
