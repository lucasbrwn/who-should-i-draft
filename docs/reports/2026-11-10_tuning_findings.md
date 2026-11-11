# Boosting tuning findings (SCRUM-50), 2026-11-10

**Bottom line:** tuning for **ranking** (option 3) is the best setting for a draft tool. It ranks players a little better at every position, significantly at TE and nearly at QB, and picks more of the real starters at QB, RB, and TE. The cost is about 0.02–0.05 fantasy points of accuracy per player. Tuning on per-stat error (the first try) didn't transfer to 2024.

## Setup

- **Tuning seasons:** train through 2021 → check on 2022, and train through 2022 → check on 2023. Never 2024 (kept for this test) or 2025 (final exam). A test proves scrambling 2024 doesn't change the chosen settings.
- **Search:** defaults + 25 random settings from the design doc ranges.
- **Fair test:** walk-forward over 2024 only (18 weeks), retraining every 4 weeks, paired 95% intervals over weeks.

## The four versions

| Version | What it is |
|---|---|
| `boosting` | default settings (learning rate 0.05, 300 trees, 31 leaves, ≥50 games per leaf) |
| `boosting_tuned` | try 1: best settings per position × stat by that stat's own error (`models/boosting_params_stat_error.json`) |
| `boosting_td` | defaults, but TD stats use try 1's settings |
| `boosting_rank` | try 2: one setting per position, chosen by weekly rank correlation of predicted fantasy points (`models/boosting_params.json`) |

## 2024 results

| Position | Rank correlation: default → **rank-tuned** | Starters picked: default → **rank-tuned** | MAE: default → rank-tuned |
|---|---|---|---|
| QB | 0.548 → **0.570** (+0.021, interval −0.002 to +0.043) | 51.4% → **55.1%** | 5.93 → 5.98 |
| RB | 0.741 → 0.744 | 65.5% → **66.7%** | 4.07 → 4.11 |
| WR | 0.695 → 0.696 | 47.7% → 46.3% | 4.18 → 4.22 (significantly worse) |
| TE | 0.672 → **0.680** (+0.008, interval +0.002 to +0.014: significant) | 47.2% → **52.8%** | 2.82 → 2.85 |

TDs-only (`boosting_td`) and try 1 (`boosting_tuned`) land between the defaults and rank-tuned on ranking, and are slightly worse than the defaults on MAE.

## What we learned

1. **The defaults train too long.** Every tuned setting used about a third of the default "training budget" (fewer trees or a slower learning rate, with bigger leaves). Shorter training holds up better on noisy stats like TDs.
2. **Tune for the goal.** Tuning each stat's own error (try 1) won easily on 2022–23 but tied on 2024. Tuning the thing the app is judged on, player rankings, transferred better.
3. **Shorter training trades a bit of accuracy for better ranking.** Predictions shrink slightly toward average, which costs a few hundredths of a point of MAE but keeps the order right.
4. **One season is a small test.** 18 weeks gives wide intervals; only TE's ranking gain is clearly significant. The 2025 final exam (SCRUM-66) will be the real check.

## Recommendation

Use the rank-tuned settings (`models/boosting_params.json`) as the boosting configuration going forward. Re-tune with folds through 2024 before the final exam.

Full tables: `backtest_2026-11-10_boosting-boosting_td-boosting_rank-boosting_tuned.md`, `backtest_2026-11-10_boosting-boosting_tuned.md`.
