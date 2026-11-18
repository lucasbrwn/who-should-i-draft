# Chance-he-plays test (2026-11-17)

The normal backtest only scores players who played, so it can't see this. Here every week of 2024 is
predicted the way live predictions are: every QB/RB/WR/TE on the pre-game active roster (ACT + INA)
gets a row, and anyone who didn't play scores 0.

| position | MAE raw | MAE expected | MAE diff [95% CI] | Spearman raw | Spearman expected | Spearman diff [95% CI] |
| --- | --- | --- | --- | --- | --- | --- |
| QB | 5.365 | 3.856 | -1.508 [-1.633, -1.385] | 0.718 | 0.749 | +0.030 [+0.013, +0.054] |
| RB | 3.905 | 3.393 | -0.512 [-0.607, -0.429] | 0.708 | 0.777 | +0.069 [+0.046, +0.098] |
| WR | 4.115 | 3.741 | -0.374 [-0.459, -0.284] | 0.661 | 0.724 | +0.063 [+0.048, +0.079] |
| TE | 2.793 | 2.544 | -0.249 [-0.291, -0.206] | 0.633 | 0.688 | +0.055 [+0.043, +0.067] |

Guard (QBs listed as backups inside the weekly top 12, 17 weeks): raw 2, expected 0

RULE: PASS -> use expected points
