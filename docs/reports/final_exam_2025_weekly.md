# FINAL EXAM 2025, part A: weekly product (run once, 2026-11-18T03:09:00.527447+00:00)

Frozen code: tag `model-freeze` = 0c24b2b. 2025 never used for training before this run.

Rules (LOCKED 2026-11-18, before any 2025 prediction or outcome was looked at):
    what      the weekly product exactly as it runs live (src/predict/week.py):
                proj_if_plays  boosting, each 2025 week trained only on EARLIER games (2020-24 + earlier
                               2025 weeks: the same walk-forward as live use within a season)
                calibrated     tier calibration fit on 2021-24 walk-forward predictions
                expected       calibrated x chance he plays (rates fit on 2021-24)
                blend          1/3 expected + 1/3 ESPN projection + 1/3 FantasyPros ECR -> points
                               (frozen table models/fp_rank_points.json, fit on 2022-24)
    data      ESPN 2025 weekly projections (cached by src/ingest/exam_benchmarks.py); FantasyPros 2025
              Friday ECR snapshots (weeks 2-17; Thursday games dropped). Same pools / pairs / player
              bootstrap as the development tests (src/backtest/expert_blend.py).
    graded    1. HEADLINE  blend vs FantasyPros, start/sit: "CONFIRMED" if the 95% CI is entirely > 50%
              2.           blend vs ESPN: "BEATS ESPN" if CI > 50%; "MATCHES" if CI includes 50%;
                           "LOSES" if CI < 50%
              3.           our model alone vs both (reported, no pass line)
              4.           MAE of blend / ours / ESPN / FantasyPros on player-weeks with every source
              5.           likely range (25-75) coverage, all players and projection >= 10
    reference 2022-24 development results (docs/reports/expert_blend_2026-11-17.md): blend vs FP 53.0%
              [50.9, 54.9], vs ESPN 52.9% [51.1, 54.6]; MAE blend 3.953, ours 4.099, ESPN 3.993.
              (Dev used uncalibrated "expected"; the exam grades the product, which is calibrated.
               Calibration never reorders players within a position.)
    safety    refuses to run if the results file exists, or if src/ / models/ differ from tag model-freeze.
              Results are reported as-is, good or bad, in the README and slides.

## Head-to-head: when the version disagrees with the benchmark, how often it's right

| version | vs FantasyPros: disagreements | right [95% CI] | vs ESPN: disagreements | right [95% CI] |
| --- | --- | --- | --- | --- |
| blend | 2933 | **51.9%** [48.2%, 55.5%] | 4191 | **50.5%** [47.6%, 53.5%] |
| ours alone | 3642 | **49.7%** [46.6%, 52.9%] | 5059 | **48.8%** [46.2%, 51.5%] |

## Points error (MAE), 5,053 player-weeks with every source

| version | 2025 MAE |
| --- | --- |
| blend | 3.980 |
| ours (expected) | 4.054 |
| ESPN | 4.005 |
| FantasyPros (rank -> pts) | 4.117 |

Blend minus ESPN MAE: -0.025 [-0.057, +0.008] (negative = blend better)

## Likely range (25-75) coverage (target about 50%)

- all players who played: 59.3% (6,771 player-weeks)
- projection >= 10: 48.7%

## Verdict (rules locked before the run)

1. **Blend vs FantasyPros: 51.9% [48.2%, 55.5%] -> NOT CONFIRMED (CI includes or is below 50%)**
2. **Blend vs ESPN: 50.5% [47.6%, 53.5%] -> MATCHES ESPN**

Dev reference (2022-24): blend vs FP 53.0% [50.9, 54.9], vs ESPN 52.9% [51.1, 54.6].
