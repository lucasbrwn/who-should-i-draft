# FINAL EXAM 2025, part B: draft side (run once, 2026-11-18T03:34:02.422004+00:00)

Frozen model code: tag `model-freeze` = 0c24b2b.

Rules (LOCKED 2026-11-18, before the 2025 board or any 2025 draft result was built or looked at):
    board     src/models/season.py exactly as frozen: option A (weekly model on Week 1 preseason rows,
              trained on 2020-24) + season model B (with ADP inputs, SCRUM-93) trained on 2021-24 seasons.
    tests     1. close pairs vs ECR (src/backtest/vs_consensus.py rules: market top-200, same position,
                 within 12 spots, total PPR season points): our board's win rate when it disagrees.
                 Reported; "CONFIRMED" only if CI > 50%. Dev: 47.1% [40.4, 53.7].
              2. HEADLINE mock draft (src/backtest/mock_draft.py rules, 12 teams, standard PPR, 10 reps x
                 12 slots, opponents draft from ECR + noise): paired edge of the sim drafter over the ADP
                 drafter. "CONFIRMED" if the 95% CI is entirely > 0. Dev: +58 [+35, +82].
                 Also reported: vor - adp, sim - vor, each drafter vs league average.
    safety    refuses to run if the results file exists, or if model code (src/models, src/features,
              src/draft, src/scoring, models/) differs from tag model-freeze. The frozen modules' 2025
              guards are lifted from HERE only, for this run. Results reported as-is.

## Close pairs: our board vs ECR (primary)

| group | pairs | we disagree | **we're right when we disagree** [95% CI] | our accuracy | market accuracy |
| --- | --- | --- | --- | --- | --- |
| ALL | 615 | 258 | **46.9%** [34.9%, 58.4%] | 46.7% | 49.3% |
| QB | 48 | 22 | **36.4%** [0.0%, 75.0%] | 35.4% | 47.9% |
| RB | 208 | 90 | **44.4%** [27.8%, 63.1%] | 44.2% | 49.0% |
| WR | 332 | 133 | **49.6%** [32.8%, 66.4%] | 49.4% | 49.7% |
| TE | 27 | 13 | **53.8%** [7.1%, 100.0%] | 51.9% | 48.1% |

Points per game view:

| group | pairs | we disagree | **we're right when we disagree** [95% CI] | our accuracy | market accuracy |
| --- | --- | --- | --- | --- | --- |
| ALL (PPG, 6+ games) | 526 | 223 | **48.0%** [35.5%, 60.4%] | 50.6% | 52.3% |

Pairs dropped (not on our board): {2025: 1}

## Close pairs: our board vs ADP

| group | pairs | we disagree | **we're right when we disagree** [95% CI] | our accuracy | market accuracy |
| --- | --- | --- | --- | --- | --- |
| ALL | 529 | 232 | **47.4%** [35.6%, 59.8%] | 49.3% | 51.6% |
| QB | 40 | 14 | **42.9%** [0.0%, 85.7%] | 47.5% | 52.5% |
| RB | 181 | 78 | **35.9%** [16.5%, 57.4%] | 43.6% | 55.8% |
| WR | 271 | 125 | **52.0%** [35.2%, 67.5%] | 50.9% | 49.1% |
| TE | 37 | 15 | **73.3%** [33.3%, 100.0%] | 67.6% | 48.6% |

Points per game view:

| group | pairs | we disagree | **we're right when we disagree** [95% CI] | our accuracy | market accuracy |
| --- | --- | --- | --- | --- | --- |
| ALL (PPG, 6+ games) | 463 | 200 | **48.0%** [34.5%, 62.0%] | 53.6% | 55.3% |

Pairs dropped (not on our board): {2025: 1}

## Mock drafts

Standard PPR, 12 teams, 2022-24 (2025 untouched). Opponents draft from preseason ECR + noise. Scored on each team's actual season (best lineup every week). Rules in src/backtest/mock_draft.py.

| drafter | drafts | avg points | edge vs league avg [95% CI] | finishes 1st | avg rank |
| --- | --- | --- | --- | --- | --- |
| sim | 120 | 1812 | -21.110 [-52.412, 9.433] | 5.0% | 6.98 |
| vor | 120 | 1704 | -137.235 [-168.221, -104.668] | 1.7% | 8.74 |
| adp | 120 | 1741 | -80.849 [-114.906, -48.372] | 1.7% | 7.64 |

Paired differences (same drafts):

| comparison | edge difference [95% CI] |
| --- | --- |
| sim - vor | 116.125 [85.727, 147.709] |
| sim - adp | 59.739 [17.019, 102.870] |
| vor - adp | -56.386 [-100.377, -13.965] |

By season (edge vs league average):

| season | sim | vor | adp |
| --- | --- | --- | --- |
| 2025 | -21 | -137 | -81 |

Mistake audit (share of drafts with each error; target 0%):

| mistake | sim | vor | adp |
| --- | --- | --- | --- |
| QB in rounds 1-2 (1-QB league) | 1.7% | 20.8% | 6.7% |
| 2 QBs before round 6 | 0.0% | 0.0% | 18.3% |
| 2 TEs before round 6 | 0.8% | 2.5% | 0.8% |
| K/DST before the last 2 rounds | 0.0% | 0.0% | 0.0% |
| empty starting slot | 0.0% | 0.0% | 33.3% |

## Verdict (rules locked before the run)

1. Board vs ECR close pairs: 46.9% [34.9%, 58.4%] -> NOT CONFIRMED  (dev 47.1% [40.4, 53.7])
2. **HEADLINE sim drafter minus ADP drafter: +60 pts/season [+17, +103] -> CONFIRMED**  (dev +58 [+35, +82])

Note (added after the run, 2026-11-18): the mock-draft table header reads "2022-24 (2025 untouched)" because the dev report template was reused; every number in this file is the 2025 run. Nothing else was edited.
