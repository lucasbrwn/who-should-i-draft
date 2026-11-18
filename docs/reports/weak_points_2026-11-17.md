# Weak points of the weekly model (2026-11-17)

Uses the walk-forward predictions from src/backtest/vs_espn.py (2022-2024, current model, each week
trained only on earlier games; players who played). No model changes here, only diagnosis.

Player-weeks: 20027. Overall MAE 4.01, bias -0.02.

## 1. Segments: where errors are bigger than usual (MAE relative to that position's average)

| segment | value | player-weeks | MAE vs position avg | bias (pred - actual) |
| --- | --- | --- | --- | --- |
| week_group | weeks 1-3 | 3514 | 1.03x | +0.17 |
| week_group | weeks 10-18 | 9991 | 0.99x | -0.11 |
| week_group | weeks 4-9 | 6522 | 1.01x | +0.02 |
| experience | 2-3rd year | 5675 | 0.97x | -0.05 |
| experience | rookie | 3067 | 0.89x | -0.13 |
| experience | veteran | 11285 | 1.04x | +0.03 |
| new_team | False | 15610 | 1.02x | -0.05 |
| new_team | True | 4417 | 0.92x | +0.09 |
| depth | backup | 8246 | 0.78x | +0.00 |
| depth | off chart | 1748 | 0.64x | +0.22 |
| depth | starter ** | 10033 | 1.24x | -0.08 |
| injury | Questionable | 862 | 1.15x | +0.63 |
| injury | none | 19162 | 0.99x | -0.05 |
| game_script | big favorite | 2422 | 1.04x | -0.18 |
| game_script | big underdog | 2340 | 0.94x | +0.04 |
| game_script | close | 15265 | 1.00x | -0.00 |
| sample | 0-2 games this season | 5245 | 0.92x | +0.24 |
| sample | 3+ games | 14782 | 1.03x | -0.11 |
| tier | 50-80% ** | 6005 | 1.20x | -0.15 |
| tier | 80-95% ** | 3004 | 1.55x | -0.29 |
| tier | bottom 50% | 10015 | 0.63x | +0.10 |
| tier | top 5% ** | 1003 | 1.83x | +0.48 |

** = bias of 0.75+ points or 15%+ larger errors than usual.

## 2. Which stat drives the points error (2024, one model fit on 2020-23)

Share of each position's points-error variance from each stat group (errors in points).

| position | TDs (6/td, 4/pass td) | yards | receptions (PPR) | interceptions/fumbles |
| --- | --- | --- | --- | --- |
| QB | 58% | 34% | 0% | 8% |
| RB | 46% | 44% | 9% | 1% |
| WR | 40% | 42% | 18% | 0% |
| TE | 40% | 34% | 26% | 1% |

(Shares of summed variance; cross-terms ignored, so read as rough proportions.)

## 3. Calibration by projection tier (mean projection vs mean actual)

| position | tier | n | mean projection | mean actual | gap (actual - proj) | ESPN mean (same players) |
| --- | --- | --- | --- | --- | --- | --- |
| QB | 50-80% | 602 | 16.05 | 15.98 | -0.07 | 16.11 |
| QB | 80-95% | 301 | 19.20 | 18.76 | -0.43 | 18.46 |
| QB | bottom 50% | 1005 | 8.83 | 8.93 | +0.10 | 8.62 |
| QB | top 5% | 101 | 22.63 | 22.68 | +0.05 | 21.68 |
| RB | 50-80% | 1464 | 9.17 | 9.33 | +0.17 | 9.51 |
| RB | 80-95% | 732 | 14.45 | 14.87 | +0.41 | 15.0 |
| RB | bottom 50% | 2441 | 2.90 | 2.71 | -0.18 | 2.53 |
| RB | top 5% | 245 | 18.60 | 17.76 | -0.84 | 18.75 |
| TE | 50-80% | 1474 | 4.32 | 4.59 | +0.28 | 4.47 |
| TE | 80-95% | 738 | 8.63 | 9.00 | +0.37 | 8.81 |
| TE | bottom 50% | 2459 | 1.24 | 1.22 | -0.02 | 1.09 |
| TE | top 5% | 246 | 13.11 | 13.08 | -0.03 | 12.67 |
| WR | 50-80% | 2465 | 8.06 | 8.19 | +0.13 | 8.29 |
| WR | 80-95% | 1233 | 13.32 | 13.67 | +0.35 | 13.49 |
| WR | bottom 50% | 4110 | 2.57 | 2.42 | -0.14 | 2.41 |
| WR | top 5% | 411 | 17.96 | 17.30 | -0.66 | 17.41 |

## 4. Where ESPN is more accurate than us (players both projected, MAE ours - ESPN; + = ESPN better)

| segment | value | player-weeks | ours | ESPN | gap |
| --- | --- | --- | --- | --- | --- |
| position | QB ** | 2007 | 5.71 | 5.39 | +0.32 |
| position | RB | 4857 | 4.18 | 4.04 | +0.14 |
| position | TE | 4893 | 2.85 | 2.77 | +0.08 |
| position | WR | 8173 | 4.21 | 4.10 | +0.11 |
| week_group | weeks 1-3 | 3502 | 4.13 | 4.14 | -0.00 |
| week_group | weeks 10-18 ** | 9941 | 3.98 | 3.81 | +0.17 |
| week_group | weeks 4-9 ** | 6487 | 4.03 | 3.88 | +0.15 |
| experience | 2-3rd year | 5634 | 3.90 | 3.79 | +0.11 |
| experience | rookie ** | 3017 | 3.60 | 3.43 | +0.17 |
| experience | veteran | 11279 | 4.20 | 4.06 | +0.13 |
| new_team | False | 15513 | 4.11 | 3.99 | +0.11 |
| new_team | True ** | 4417 | 3.73 | 3.53 | +0.20 |
| depth | backup ** | 8238 | 3.00 | 2.77 | +0.23 |
| depth | off chart ** | 1659 | 2.63 | 2.31 | +0.32 |
| depth | starter | 10033 | 5.09 | 5.07 | +0.02 |
| injury | Questionable | 861 | 4.60 | 4.72 | -0.13 |
| injury | none | 19066 | 4.00 | 3.85 | +0.14 |
| game_script | big favorite ** | 2409 | 4.18 | 3.90 | +0.28 |
| game_script | big underdog | 2328 | 3.78 | 3.77 | +0.02 |
| game_script | close | 15193 | 4.03 | 3.91 | +0.13 |
| sample | 0-2 games this season ** | 5157 | 3.79 | 3.62 | +0.16 |
| sample | 3+ games | 14773 | 4.10 | 3.98 | +0.12 |
| tier | 50-80% | 6005 | 4.79 | 4.73 | +0.05 |
| tier | 80-95% | 3004 | 6.10 | 6.14 | -0.04 |
| tier | bottom 50% ** | 9918 | 2.61 | 2.37 | +0.24 |
| tier | top 5% | 1003 | 7.17 | 7.10 | +0.07 |

** = ESPN at least 0.15 points/player-week more accurate.

