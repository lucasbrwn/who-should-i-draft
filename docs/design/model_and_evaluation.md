# Model and Evaluation Design

**Task:** SCRUM-6 · **Applies to:** M3 features, M4 projection model, M5 backtest, M6 draft recommender
**Inputs:** [literature review](../research/literature_review.md), planning decisions (Sep 2026)

## 1. What the model predicts

The model predicts **component stats per player per week**, never fantasy points directly. The scoring engine (M2) turns components into points for any league, so one set of models serves every scoring format.

| Position | Modeled components (one regressor each) | Loss |
|---|---|---|
| QB | passing yards, passing TDs, interceptions, rushing yards, rushing TDs | yards: squared error · counts: Poisson |
| RB | rushing yards, rushing TDs, receptions, receiving yards, receiving TDs | same |
| WR | receptions, receiving yards, receiving TDs | same |
| TE | receptions, receiving yards, receiving TDs | same |

**Rare components are rates, not models:** 2-point conversions, fumbles lost, special-teams TDs, and WR/TE rushing. Each uses the player's rate per opportunity, shrunk toward the position average (few events, too noisy to learn), times projected opportunities.

**Why Poisson for counts:** TDs, receptions, and interceptions are small non-negative counts. scikit-learn's `HistGradientBoostingRegressor(loss="poisson")` fits them directly and never predicts negatives.

## 2. Model progression

Each step must beat the previous one in the backtest (section 5) to be kept.

1. **Baseline:** recency-weighted average of the player's last games (prior season blended in early).
2. **Ridge** (scikit-learn pipeline: impute → scale → Ridge) on the M3 features.
3. **Gradient boosting:** `HistGradientBoostingRegressor` (LightGBM only if it's clearly better).

No neural networks in v1 (six seasons of tabular data; boosting is as good or better and easier to explain).

## 3. Floor, median, ceiling (change from the original plan)

**Original plan:** quantile regression for the 10th/50th/90th percentile.
**New plan:** mean models per component + **simulation from historical errors**.

Why the change:
- Quantiles don't add up. The 90th percentile of receptions + the 90th percentile of yards is not the 90th percentile of fantasy points.
- Floor and ceiling must be in **the user's league scoring** (a PPR ceiling differs from a standard ceiling), and must handle bonuses (e.g. +3 for 100+ yards), which depend on the whole distribution.

How it works:
1. For every past prediction, store the vector of errors across that player's components (actual − predicted). Keeping them together preserves correlations (a big yardage game usually comes with more catches and TDs).
2. To build a player's distribution, draw ~1,000 error vectors from similar past predictions (same position, similar projected volume), add them to the projection, clip at zero, and score each draw with the league's settings.
3. Floor = 10th percentile, median = 50th, ceiling = 90th of those simulated fantasy points.

Quantile regression stays as a comparison in the backtest. Target: **~80% of actual outcomes fall between floor and ceiling.**

## 4. Weekly vs. season projections

- **Weekly model** trains on player-weeks and projects one week ahead.
- **Season projection (for the draft)** = sum over the real schedule of weekly projections × probability the player is active that week.
- **Availability model:** games missed by position, age, and injury history (start simple: historical rates; logistic regression if it helps).
- **Preseason inputs:** before Week 1 there are no current-season games, so features come from the prior season plus situation changes (new team, vacated targets, depth chart).

## 5. Validation protocol (no leakage)

**Every split is by time.** No random `train_test_split`.

| Seasons | Role |
|---|---|
| 2020–2021 | Training history only |
| 2022, 2023, 2024 | Development: walk-forward tests, used to choose models and features |
| **2025** | **Final exam.** Untouched until the model is frozen (Sprint 4). Run once. |
| 2026 | Live season (no labels yet for most weeks) |

- **Weekly walk-forward:** for each test week, train only on earlier weeks. To keep runtime reasonable, retrain every 4 weeks within a season.
- **Season / draft backtest:** for each development season, build preseason projections using only data available before Week 1, then compare to the full season.
- **Leakage test (automated):** no feature for week N may use data from week N or later (SCRUM-39).

## 6. Hyperparameter tuning

- Tuning happens **inside the training seasons only**, never on the season being tested.
- Example (testing 2024): tune on folds "train ≤2021 → validate 2022" and "train ≤2022 → validate 2023"; pick the best settings; refit on ≤2023; test on 2024.
- Implemented with scikit-learn `RandomizedSearchCV` and explicit time-based folds (a list of train/validation index pairs), ~30–50 settings per model.
- Tuning metric: Poisson deviance for count stats, MAE for yards. Final selection also checks MAE on PPR fantasy points.
- Search space (boosting): `learning_rate` 0.02–0.2, `max_leaf_nodes` 15–63, `min_samples_leaf` 20–200, `l2_regularization` 0–1, `max_iter` with early stopping. Ridge: `alpha` 0.1–100.

## 7. Benchmark: market consensus

- **Source:** FantasyPros expert consensus rankings (ECR) via `nflreadpy.load_ff_rankings("all")`: the **preseason PPR cheat sheet** snapshot taken just before Week 1, available for **2021–2026** (none for 2020). Best-ball ECR is also available for 2021+.
- **ECR is not ADP.** ECR is what experts rank; ADP is where drafters actually pick. They're closely related and ECR is arguably the tougher benchmark. True ADP is still wanted for the draft simulator's opponent behavior (SCRUM-43).
- ECR is **not** used as a model feature in v1, so the edge we measure is our own signal. Blending it in later is an option.

## 8. Metrics and success criteria

**Primary: decision accuracy vs. consensus (season level).**
For every pair of players ranked within 12 spots of each other in preseason ECR (about one draft round), check whether our projections order them correctly by actual fantasy points per game. Report the percentage correct with a 95% confidence interval from a **player-level bootstrap** (pairs share players, so they aren't independent).

- Rough scale: ~2,400 pairs per season × 3 development seasons ≈ 7,000 pairs, giving a CI half-width of about ±1.2% before accounting for shared players. A season-level edge of ~51% may not be provable on its own; the **weekly** start/sit comparison (many more decisions) and mock drafts add evidence.

**Success (all on development seasons, then confirmed once on 2025):**
1. Decision accuracy vs. ECR above 50%, with the 95% CI excluding 50%.
2. Gradient boosting beats the baseline on MAE for every position.
3. Floor–ceiling calibration between 75% and 85%.
4. Mock drafts: a team drafting from our board outscores ECR-based drafters on average.

**Secondary metrics:** MAE (fantasy points per game), Spearman rank correlation within position, top-N hit rate (e.g. of our top 24 WRs, how many finished top 24).

## 9. Explainability

- SHAP values for every projection (tree SHAP for boosting, coefficients × feature values for Ridge).
- Top positive and negative contributions are turned into plain-English reasons with fixed templates (no free-form text generation), shown in the app.

## 10. Open items

- Routes-run data for YPRR / route participation (SCRUM-47).
- Expected fantasy points ingest (SCRUM-46).
- True historical ADP source for opponent modeling (SCRUM-43).
