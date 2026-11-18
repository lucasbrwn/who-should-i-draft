# Backtest report (2026-11-17)

Walk-forward over [2021, 2022, 2023, 2024] (2025 untouched). Standard PPR. Values: mean [95% CI over weeks].

| model | position | weeks | mae | spearman | top_n_hit |
| --- | --- | --- | --- | --- | --- |
| boosting | QB | 72 | 5.821 [5.615, 6.052] | 0.573 [0.539, 0.604] | 0.544 [0.515, 0.572] |
| boosting_no_qbredzone | QB | 72 | 5.834 [5.630, 6.059] | 0.569 [0.537, 0.601] | 0.539 [0.509, 0.571] |
| boosting | RB | 72 | 4.338 [4.219, 4.457] | 0.715 [0.701, 0.728] | 0.616 [0.598, 0.635] |
| boosting_no_qbredzone | RB | 72 | 4.338 [4.226, 4.450] | 0.715 [0.701, 0.728] | 0.616 [0.597, 0.635] |
| boosting | TE | 72 | 2.888 [2.816, 2.964] | 0.668 [0.652, 0.684] | 0.491 [0.462, 0.517] |
| boosting_no_qbredzone | TE | 72 | 2.888 [2.814, 2.965] | 0.668 [0.653, 0.684] | 0.491 [0.464, 0.517] |
| boosting | WR | 72 | 4.242 [4.157, 4.330] | 0.688 [0.675, 0.701] | 0.478 [0.461, 0.496] |
| boosting_no_qbredzone | WR | 72 | 4.242 [4.155, 4.327] | 0.688 [0.675, 0.702] | 0.478 [0.460, 0.495] |

## boosting_no_qbredzone minus boosting (paired; MAE: negative = better, Spearman: positive = better)

| position | mae_diff | spearman_diff |
| --- | --- | --- |
| QB | 0.013 [-0.006, 0.030] | -0.003 [-0.008, 0.001] |
| RB | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| WR | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| TE | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
