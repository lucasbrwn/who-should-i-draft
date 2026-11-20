# Ranking objective (LambdaRank + rank remap) vs current model, [2022, 2023, 2024] (2026-11-19)

Rules (LOCKED 2026-11-19 before any ranker result was computed; Lucas: no changes to the proposal):
    ranker    LightGBM lambdarank, one per position, same feature columns as the boosting model,
              groups = position-week, label = actual PPR points rounded and clipped to 0..40 with
              LINEAR gain (a 30-pt game matters 30x a 1-pt game, not 2^30x). PPR only (known
              mismatch for half/standard leagues - accepted to save 3x training).
              Settings fixed in advance, never tuned: n_estimators 300, learning_rate 0.05,
              num_leaves 31, min_child_samples 50, reg_lambda 0.1, seed 0.
    training  walk-forward like the boosting model: every week uses only earlier weeks, refit every
              4 weeks and at each season start; games a player left injured are not trained on.
    seasons   test 2022-2024 (same as the blend test). 2025 is "seen" (final exam) - not used.
    remap     expected_rank = (our point projection reassigned in the ranker's order) x chance he plays
    champion  the current product: 1/3 blend of ours + ESPN + FantasyPros
    KEEP the ranker only if BOTH:
      1. blend(ranker) vs FantasyPros "right when we disagree" is higher than blend(current) AND the
         two 95% CIs do not overlap
      2. blend(ranker) MAE is not worse than blend(current) by more than 0.02 points
    Otherwise (including any tie) the simpler current model stays.

| version | vs FantasyPros: disagreements | right [95% CI] | vs ESPN: disagreements | right [95% CI] |
| --- | --- | --- | --- | --- |
| blend_rank | 9429 | **51.3%** [49.4%, 53.4%] | 14431 | **52.0%** [50.4%, 53.5%] |
| blend | 8823 | **53.0%** [51.0%, 55.0%] | 13516 | **52.9%** [51.3%, 54.5%] |
| expected_rank | 13170 | **49.0%** [47.3%, 50.7%] | 18858 | **49.7%** [48.3%, 51.0%] |
| expected | 12466 | **49.9%** [48.2%, 51.6%] | 17864 | **50.6%** [49.1%, 52.0%] |

MAE on 14,665 player-weeks with every source: blend_rank 3.960, blend 3.953, expected_rank 4.172, expected 4.099

- Rule 1 (higher win rate, CIs don't overlap): FAIL
- Rule 2 (blend MAE not worse by > 0.02): PASS (+0.008)
- **Verdict: REJECT - current model stays**
