# Mock-draft test (2026-11-14)

Standard PPR, 12 teams, 2022-24 (2025 untouched). Opponents draft from preseason ECR + noise. Scored on each team's actual season (best lineup every week). Rules in src/backtest/mock_draft.py.

| drafter | drafts | avg points | edge vs league avg [95% CI] | finishes 1st | avg rank |
| --- | --- | --- | --- | --- | --- |
| sim | 360 | 1918 | 61.687 [46.289, 77.602] | 7.2% | 5.33 |
| vor | 360 | 1938 | 83.375 [69.518, 97.395] | 10.0% | 5.02 |
| adp | 360 | 1853 | 3.758 [-20.383, 26.438] | 11.1% | 6.35 |

Paired differences (same drafts):

| comparison | edge difference [95% CI] |
| --- | --- |
| sim - vor | -21.688 [-37.882, -6.330] |
| sim - adp | 57.929 [35.479, 81.918] |
| vor - adp | 79.617 [54.565, 105.072] |

By season (edge vs league average):

| season | sim | vor | adp |
| --- | --- | --- | --- |
| 2022 | 131 | 47 | 158 |
| 2023 | 64 | 124 | -62 |
| 2024 | -10 | 79 | -84 |

Mistake audit (share of drafts with each error; target 0%):

| mistake | sim | vor | adp |
| --- | --- | --- | --- |
| QB in rounds 1-2 (1-QB league) | 5.3% | 33.3% | 15.8% |
| 2 QBs before round 6 | 0.0% | 0.0% | 32.8% |
| 2 TEs before round 6 | 0.3% | 2.8% | 4.7% |
| K/DST before the last 2 rounds | 0.0% | 0.0% | 0.0% |
| empty starting slot | 0.0% | 0.0% | 36.7% |
