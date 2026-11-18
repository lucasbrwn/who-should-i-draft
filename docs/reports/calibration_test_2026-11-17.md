# Tier calibration test (2026-11-17)

For each test season S, fit the isotonic curves (src/models/weekly_calibration.py) on the walk-forward
predictions of EARLIER seasons only (2021+, never 2025), apply them to season S, compare per week.

## Rule seasons 2022-2024

| position | MAE before | MAE after | diff [95% CI] | Spearman diff [95% CI] |
| --- | --- | --- | --- | --- |
| QB | 5.682 | 5.627 | -0.055 [-0.097, -0.011] | -0.0014 [-0.0053, +0.0023] |
| RB | 4.170 | 4.143 | -0.027 [-0.042, -0.012] | -0.0004 [-0.0019, +0.0011] |
| WR | 4.199 | 4.172 | -0.028 [-0.038, -0.017] | -0.0003 [-0.0011, +0.0005] |
| TE | 2.844 | 2.834 | -0.010 [-0.023, +0.004] | +0.0003 [-0.0009, +0.0016] |

| position | tier | n | bias before (actual - proj) | bias after |
| --- | --- | --- | --- | --- |
| QB | 50-80% | 602 | -0.07 | +0.84 |
| QB | 80-95% | 301 | -0.43 | -0.01 |
| QB | bottom 50% | 1005 | +0.10 | +0.15 |
| QB | top 5% | 101 | +0.05 | +0.54 |
| RB | 50-80% | 1464 | +0.17 | +0.19 |
| RB | 80-95% | 732 | +0.41 | +0.54 |
| RB | bottom 50% | 2441 | -0.18 | -0.01 |
| RB | top 5% | 245 | -0.84 | -0.14 |
| TE | 50-80% | 1474 | +0.28 | +0.31 |
| TE | 80-95% | 738 | +0.37 | +0.12 |
| TE | bottom 50% | 2459 | -0.02 | +0.07 |
| TE | top 5% | 246 | -0.03 | +0.48 |
| WR | 50-80% | 2465 | +0.13 | +0.19 |
| WR | 80-95% | 1233 | +0.35 | -0.09 |
| WR | bottom 50% | 4110 | -0.14 | +0.01 |
| WR | top 5% | 411 | -0.66 | -0.29 |

## Extra check: 2026 weeks 1-10 (curves fit on 2021-2024)

| position | MAE before | MAE after | diff [95% CI] | Spearman diff [95% CI] |
| --- | --- | --- | --- | --- |
| QB | 5.818 | 5.798 | -0.019 [-0.112, +0.052] | -0.0062 [-0.0130, +0.0002] |
| RB | 4.202 | 4.179 | -0.023 [-0.056, +0.013] | +0.0008 [-0.0009, +0.0025] |
| WR | 4.064 | 4.033 | -0.031 [-0.046, -0.017] | -0.0002 [-0.0015, +0.0013] |
| TE | 3.066 | 3.074 | +0.007 [-0.011, +0.026] | -0.0018 [-0.0053, +0.0012] |

RULE: PASS -> calibrate live projections
