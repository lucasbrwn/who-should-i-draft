# Model + experts blend, seasons [2022, 2023, 2024] (2026-11-17)

Rules (LOCKED 2026-11-18 before any blend result was computed):
    ours      expected points = walk-forward boosting projection x chance he plays (what the product shows)
    ESPN      ESPN's weekly PPR projection (points)
    FP        FantasyPros weekly ECR (Friday snapshot) turned into points: per position, a decreasing
              isotonic fit of actual PPR points on ECR rank, fit on the OTHER test seasons only
              (leave-one-season-out, so a season's own outcomes never set its map). Players whose game
              was before the snapshot (Thursday) get no FP value - the experts may have seen the result.
    blend     plain average of whichever of the three exist. EQUAL WEIGHTS, fixed in advance - no
              fitting, so nothing here can overfit (simple averages are hard to beat in forecasting).
    seasons   2022-2024 (2025 is the final exam). Same pools / pairs / bootstrap as vs_espn.py.

## Head-to-head: when the version disagrees with the benchmark, how often it's right

| version | vs FantasyPros: disagreements | right [95% CI] | vs ESPN: disagreements | right [95% CI] |
| --- | --- | --- | --- | --- |
| blend | 8823 | **53.0%** [50.9%, 54.9%] | 13516 | **52.9%** [51.1%, 54.6%] |
| ours+ESPN | 10700 | **52.5%** [50.6%, 54.4%] | 13723 | **51.9%** [50.3%, 53.6%] |
| ours+FP | 9635 | **50.5%** [48.6%, 52.4%] | 16462 | **51.8%** [50.3%, 53.3%] |
| experts only | 7516 | **53.1%** [50.9%, 55.2%] | 7664 | **53.3%** [51.0%, 55.5%] |
| ours alone | 12466 | **49.9%** [48.2%, 51.6%] | 17864 | **50.6%** [49.1%, 52.0%] |

## Points error (MAE) on 14,665 player-weeks with all three numbers

| version | MAE |
| --- | --- |
| expected | 4.099 |
| espn_proj | 3.993 |
| fp_pts | 4.078 |
| blend | 3.953 |
| ours+ESPN | 3.946 |
| ours+FP | 4.009 |
| experts only | 3.979 |
| ours alone | 4.099 |

## Verdict (rules locked before the run)

- PRIMARY blend vs FantasyPros: 53.0% [50.9%, 54.9%] -> CI above 50%
- blend vs ESPN: 52.9% [51.1%, 54.6%] -> not significantly worse
- **PASS**
- CLAIM CHECK blend minus experts-only MAE: -0.027 [-0.039, -0.014] -> our model adds value on top of the experts
