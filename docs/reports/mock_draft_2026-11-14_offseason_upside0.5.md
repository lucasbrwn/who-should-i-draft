# Mock-draft test (2026-11-14)

Standard PPR, 12 teams, 2022-24 (2025 untouched). Opponents draft from preseason ECR + noise. Scored on each team's actual season (best lineup every week). Rules in src/backtest/mock_draft.py.

| drafter | drafts | avg points | edge vs league avg [95% CI] | finishes 1st | avg rank |
| --- | --- | --- | --- | --- | --- |
| sim | 360 | 1811 | -52.796 [-72.946, -34.849] | 1.9% | 7.24 |
| vor | 360 | 1706 | -161.453 [-184.123, -140.637] | 0.3% | 8.87 |
| adp | 360 | 1852 | 3.495 [-18.788, 26.150] | 13.9% | 6.38 |

Paired differences (same drafts):

| comparison | edge difference [95% CI] |
| --- | --- |
| sim - vor | 108.657 [92.529, 124.689] |
| sim - adp | -56.291 [-80.116, -31.679] |
| vor - adp | -164.948 [-191.114, -138.425] |

By season (edge vs league average):

| season | sim | vor | adp |
| --- | --- | --- | --- |
| 2022 | 69 | -47 | 173 |
| 2023 | -22 | -76 | -71 |
| 2024 | -205 | -362 | -91 |

Mistake audit (share of drafts with each error; target 0%):

| mistake | sim | vor | adp |
| --- | --- | --- | --- |
| QB in rounds 1-2 (1-QB league) | 8.6% | 2.8% | 17.8% |
| 2 QBs before round 6 | 3.6% | 9.7% | 32.5% |
| 2 TEs before round 6 | 0.3% | 0.8% | 6.9% |
| K/DST before the last 2 rounds | 0.0% | 0.0% | 0.0% |
| empty starting slot | 3.3% | 1.9% | 38.1% |
