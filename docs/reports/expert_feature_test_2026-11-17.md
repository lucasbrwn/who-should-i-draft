# Challenger (our model + FantasyPros rank as input) vs champion (blend), [2022, 2023, 2024] (2026-11-17)

Rules (LOCKED 2026-11-18 before the challenger's results were computed; Lucas: "Do it"):
    challenger  walk-forward boosting on our features + fp_ecr (src/features/expert.py), each week trained
                only on earlier weeks; shown as expected points = projection x chance he plays
    champion    the blend that passed on 2026-11-18 (src/backtest/expert_blend.py)
    seasons     2022-2024; 2025 untouched. Same pools, pairs, bootstrap as the blend test.

| version | vs FantasyPros: disagreements | right [95% CI] | vs ESPN: disagreements | right [95% CI] |
| --- | --- | --- | --- | --- |
| challenger | 12020 | **49.9%** [48.1%, 51.6%] | 17724 | **51.2%** [49.6%, 52.6%] |
| blend | 8823 | **53.0%** [51.0%, 54.9%] | 13516 | **52.9%** [51.2%, 54.5%] |
| ours alone | 12466 | **49.9%** [48.2%, 51.6%] | 17864 | **50.6%** [49.1%, 52.1%] |

MAE on 14,665 player-weeks with every source: challenger 4.065, blend 3.953, expected 4.099, espn_proj 3.993

Challenger minus blend MAE: +0.112 [+0.088, +0.136] (negative = challenger better)

**Verdict: challenger FAILS the bar -> blend stays**
