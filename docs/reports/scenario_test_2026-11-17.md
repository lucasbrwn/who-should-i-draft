# Scenario projection test (2026-11-17)

Live-style 2024 test, like play_chance_test.py: every pre-game active player each week, 0 if he didn't
play, chance-he-plays rates from 2021-23. Two versions of expected points:

Affected player-weeks: 4069 (of 8068)

AFFECTED MAE: current 3.473, scenario 3.481, diff +0.008 [-0.007, +0.021]

| position (all players) | MAE diff [95% CI] | Spearman diff [95% CI] |
| --- | --- | --- |
| QB | +0.003 [-0.005, +0.011] | +0.0017 [-0.0005, +0.0038] |
| RB | -0.003 [-0.022, +0.016] | +0.0072 [+0.0025, +0.0122] |
| WR | +0.013 [+0.000, +0.021] | +0.0056 [+0.0022, +0.0105] |
| TE | -0.006 [-0.017, +0.006] | +0.0044 [-0.0005, +0.0096] |

RULE: FAIL -> keep current
