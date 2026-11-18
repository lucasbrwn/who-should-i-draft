# Stacker (one final model on ours + ESPN + FantasyPros) vs the 1/3 blend, [2023, 2024] (2026-11-17)

Rules (LOCKED 2026-11-18 before any stacker result was computed; Lucas: "Yes"):
    inputs    our expected points, our raw projection, chance he plays, ESPN projection, FantasyPros
              ECR RANK (raw rank, not the rank->points map, which was fit on other seasons incl. later
              ones), position, injury report status, depth rank, years of experience. Missing = blank.
    model     sklearn HistGradientBoostingRegressor, settings fixed in advance (never tuned):
              max_depth 3, learning_rate 0.05, max_iter 300, min_samples_leaf 50, l2 1.0, seed 0
    training  week-by-week walk-forward: each test week is predicted by a stacker trained only on
              EARLIER weeks (from 2022 week 1). Test seasons 2023-2024 (2022 has too little history).
              2025 untouched.
    champion  the 1/3 blend on the same 2023-24 rows. (Its FP rank->points map was fit leave-one-season-
              out, so for 2023 it saw 2024 outcomes: a small edge FOR the champion, i.e. conservative.)
    judge     same pools / pairs / player bootstrap as expert_blend.py:
      1. stacker vs FantasyPros head-to-head CI entirely above 50%, AND vs ESPN not significantly worse
      2. stacker minus blend MAE (player-weeks with every source): if significantly WORSE (CI > 0) the
         blend stays; if tied or better, the STACKER becomes the product (Lucas prefers one model).
      If 1 fails: the blend stays.
Known bias: ESPN history is its FINAL number (may include post-Friday news), so the stacker may learn to
over-trust ESPN compared with our live Thursday snapshot.

| version | vs FantasyPros: disagreements | right [95% CI] | vs ESPN: disagreements | right [95% CI] |
| --- | --- | --- | --- | --- |
| stacker | 6605 | **52.6%** [50.1%, 55.0%] | 8137 | **51.4%** [49.3%, 53.6%] |
| blend | 5808 | **53.3%** [50.7%, 55.9%] | 9008 | **52.8%** [50.7%, 54.9%] |
| experts only | 5033 | **53.9%** [51.4%, 56.5%] | 5053 | **52.9%** [50.1%, 55.8%] |
| ours alone | 8139 | **50.1%** [48.0%, 52.2%] | 11899 | **50.6%** [48.9%, 52.3%] |

MAE on 9,518 player-weeks with every source: stacker 3.955, blend 3.919, experts only 3.939, expected 4.067, espn_proj 3.933

Stacker minus blend MAE: +0.036 [+0.019, +0.052] (negative = stacker better)

**Verdict: stacker passes but has significantly WORSE MAE than the blend -> blend stays**
