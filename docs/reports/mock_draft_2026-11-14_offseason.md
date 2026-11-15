# Mock-draft test (2026-11-14)

Standard PPR, 12 teams, 2022-24 (2025 untouched). Opponents draft from preseason ECR + noise. Scored on each team's actual season (best lineup every week). Rules in src/backtest/mock_draft.py.

| drafter | drafts | avg points | edge vs league avg [95% CI] | finishes 1st | avg rank |
| --- | --- | --- | --- | --- | --- |
| sim | 360 | 1828 | -34.638 [-55.564, -13.828] | 4.7% | 6.83 |
| vor | 360 | 1723 | -144.202 [-170.168, -120.593] | 0.8% | 8.27 |
| adp | 360 | 1852 | 3.495 [-18.788, 26.150] | 13.9% | 6.38 |

Paired differences (same drafts):

| comparison | edge difference [95% CI] |
| --- | --- |
| sim - vor | 109.564 [94.950, 124.093] |
| sim - adp | -38.133 [-64.271, -10.708] |
| vor - adp | -147.697 [-175.725, -120.132] |

By season (edge vs league average):

| season | sim | vor | adp |
| --- | --- | --- | --- |
| 2022 | 56 | -8 | 173 |
| 2023 | 92 | -12 | -71 |
| 2024 | -252 | -412 | -91 |

Mistake audit (share of drafts with each error; target 0%):

| mistake | sim | vor | adp |
| --- | --- | --- | --- |
| QB in rounds 1-2 (1-QB league) | 8.6% | 2.8% | 17.8% |
| 2 QBs before round 6 | 0.0% | 0.0% | 32.5% |
| 2 TEs before round 6 | 0.6% | 1.4% | 6.9% |
| K/DST before the last 2 rounds | 0.0% | 0.0% | 0.0% |
| empty starting slot | 0.3% | 0.0% | 38.1% |
