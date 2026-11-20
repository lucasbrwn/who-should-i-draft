# Season ranker (LambdaRank + rank remap) vs current draft board, [2022, 2023, 2024] (2026-11-19)

Rules (LOCKED 2026-11-19 before any season-ranker result was computed; Lucas: "Go."):
    ranker    LightGBM lambdarank on the season model's own inputs (b_inputs: last season's stats, age,
              team changes, vacated/arrived opportunity, win totals, ADP, option A's per-game number...),
              groups = season x position, label = actual season PPR points rounded, clipped 0..499,
              LINEAR gain. Settings fixed in advance, never tuned (mirror B_PARAMS): n_estimators 200,
              learning_rate 0.05, num_leaves 15, min_child_samples 20, reg_lambda 1.0, seed 0,
              lambdarank_truncation_level 100 (draft pools go 72 RBs / 96 WRs deep, default is 30).
    training  walk-forward: the 2022 board learns from 2021 only, 2023 from 2021-22, 2024 from 2021-23.
              2025 is "seen" (final exam) and not used.
    remap     inside each season-position, the season model's point projections (season_b) are handed
              out in the ranker's order, so VOR / the recommender / every scoring format still get points.
    champion  the current board (season model B), rebuilt here the same way into a scratch folder.
    KEEP the season ranker only if BOTH:
      1. close pairs vs ECR (same position, within 12 ECR spots, src/backtest/vs_consensus.py):
         ranker board's "right when we disagree" is higher than the current board's AND the two 95%
         CIs do not overlap
      2. mock draft (only run if 1 passes): sim drafter's edge with the ranker board minus with the
         current board, paired on the same drafts - REJECT if the 95% CI is entirely below 0
    Otherwise (including any tie) the current board stays.

| board | close pairs | we disagree with ECR | **right when we disagree** [95% CI] | by position |
| --- | --- | --- | --- | --- |
| current | 1877 | 813 | **48.1%** [41.6%, 54.5%] | QB 45%, RB 46%, WR 49%, TE 59% |
| ranker | 1877 | 754 | **46.7%** [39.9%, 53.4%] | QB 40%, RB 51%, WR 45%, TE 49% |

- Rule 1 (higher, CIs don't overlap): **FAIL -> REJECT, current board stays**
