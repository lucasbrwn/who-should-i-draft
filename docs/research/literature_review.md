# Literature Review: What Predicts Fantasy Points

**Task:** SCRUM-24 (M1.5 Position Research) · **Feeds:** SCRUM-25 (predictive power), SCRUM-26 (stability), SCRUM-27 (defense), M3 features

## How to read this

This review collects what published fantasy analytics research says predicts **future** fantasy points, by position. Correlations quoted here are **as reported by the sources** (different samples, seasons, and methods). They are hypotheses, not facts for our model: SCRUM-25/26 re-measure every candidate on our own 2020–2025 data before anything becomes a feature.

Two kinds of correlation show up in the literature and are easy to confuse:

- **Descriptive:** how well a stat explains fantasy points *in the same season* (e.g. touches vs. points). High numbers here are expected and say little about prediction.
- **Predictive / stable ("sticky"):** how well a stat in season N predicts season N+1 (or week N predicts week N+1). This is what a draft tool needs.

## Principles that hold across positions

1. **Opportunity beats efficiency.** Volume and role (targets, carries, snaps, routes, share of team opportunity) are far stickier than per-touch efficiency. Efficiency matters mostly as a tiebreaker and as a sign a role may grow.
2. **Touchdowns regress.** TD totals swing a lot year to year. Expected TDs from *where* opportunities happen (distance to the goal line) predict future TDs better than past TDs or broad "red zone" counts.
3. **Shares beat raw totals.** Target share, air-yards share, and carry share adjust for team volume and are stickier than raw counts.
4. **Team context scales everything.** A player's share multiplies by team volume and scoring; Vegas implied team totals are a compact measure of expected scoring.
5. **Age curves are real and position-specific.** RBs peak and decline earliest.

## Quarterbacks

| Candidate feature | Why it matters | Reported evidence | In our data? |
|---|---|---|---|
| Rushing attempts / yards per game | QB rushing is the biggest fantasy separator and is very stable | YoY correlation: rush yards 0.77, carries 0.79 (4for4) | ✅ player_stats |
| Passing yards / attempts per game | Volume, stable | Pass yards YoY 0.79 (4for4) | ✅ player_stats |
| Passing TDs | Big part of weekly scoring, but low year-to-year predictability | Low season-level predictability (4for4) | ✅; model via expected TDs instead |
| Expected pass TDs / fantasy points | Opportunity-based, regresses less than actual TDs | xTD concept (Fantasy Points, Mike Clay oTD) | ✅ `load_ff_opportunity` |
| Team implied total, spread | Expected team scoring | Strongest Vegas–fantasy link of any position (Fantasy Projection Lab; weak source, to verify) | ✅ schedules + snapshots |
| EPA per dropback | Passing efficiency | Commonly used; stability to test | ✅ play-by-play |

## Running backs

| Candidate feature | Why it matters | Reported evidence | In our data? |
|---|---|---|---|
| Weighted opportunity (carries × ~0.58 + targets × ~1.59, PPR) | Weights touches by fantasy value | One of the most stable, predictive RB stats (Barrett, Fantasy Points); 0.95 correlation with fantasy points is *same-season* (PFF) | ✅ derive from player_stats |
| Targets / target share | A target is worth ~2.55× a carry in PPR (1.53 pts), ~1.9× half-PPR, ~1.3× standard | PFF "value of a carry vs. a target" | ✅ |
| Carry share / snap share | Role and workload | Opportunity principle | ✅ player_stats + snap counts |
| Goal-line / inside-5 carries, expected rush TDs | TDs driven by field position; broad red-zone counts mislead (1-yd line ≈ 54% TD vs 17-yd ≈ 4%) | Fantasy Points XTD | ✅ play-by-play, `load_ff_opportunity` |
| Age / years in league | Peak ~24–26, decline from ~27; value peaks in year 3 | PFF, 4for4, ESPN aging curves | ✅ rosters/players |
| Team implied total | Scoring environment; noisier for RBs than QB/WR | Fantasy Projection Lab (to verify) | ✅ |

## Wide receivers

| Candidate feature | Why it matters | Reported evidence | In our data? |
|---|---|---|---|
| Target share | The stickiest WR stat | YoY ≈ 0.70 since 2021 (4for4 / Sharp Football) | ✅ player_stats |
| Air-yards share | Downfield role | Part of WOPR (Hermsmeyer) | ✅ player_stats |
| WOPR = 1.5 × target share + 0.7 × air-yards share | Combines both; weighted for predicting fantasy points; more predictive than target share alone | Hermsmeyer; Action Network, CBS | ✅ player_stats (`wopr`) |
| Yards per route run (YPRR) | Best efficiency predictor; flags talent on limited volume (≥ 2.0 YPRR notable) | 0.43 corr. with next-season receiving points vs 0.46 raw targets, 0.18 yds/target (PFF); >0.60 stability since 2021 (4for4) | ⚠️ needs routes run (see gaps) |
| Targets per route run (TPRR) | Earning targets when on the field | Sticky for WRs (4for4) | ⚠️ needs routes |
| Yards per team pass attempt | Sticky rate stat | 4for4 | ✅ derive |
| Age | Peak ~26–28; WR1/WR2 odds drop after 30 (some newer work says peak 29–31) | PFF, Fantasy Footballers, Fantasy Life | ✅ |

## Tight ends

| Candidate feature | Why it matters | Reported evidence | In our data? |
|---|---|---|---|
| Route participation (routes / team dropbacks) | TEs also block; routes, not snaps, drive opportunity | PlayerProfiler, 4for4 | ⚠️ needs routes |
| Yards / targets / receptions per game | The stickiest TE stats; yards per game most predictive of next season | Fantasy Life (2026), 4for4 | ✅ player_stats |
| Yards per team pass attempt | Even stickier for TEs than WRs | 4for4 | ✅ derive |
| YPRR (≥ 1.75 notable) | Efficiency on limited volume | Sharp Football | ⚠️ needs routes |
| Targets per route run | Only about half as stable for TEs as for WRs, so weight it less | 4for4 | ⚠️ needs routes |
| Red-zone / expected TDs | Big share of TE scoring | Fantasy Projection Lab; xTD | ✅ |

## Defensive matchups

- **Raw "fantasy points allowed" (defense vs. position) is noisy.** One backup's three-TD game can distort a defense's ranking for weeks, and defenses that face strong offenses look worse than they are.
- **WR/TE matchups are less stable than QB/RB matchups.** Past points allowed to WRs/TEs predict future points allowed less well.
- **Fixes used in practice:** adjust for opponent quality (schedule-adjusted points allowed) and shrink small samples toward the league average (Bayesian adjustment).
- **Implication for us (SCRUM-36/37):** build defensive ratings from play-level data (EPA allowed, pass-area splits, pressure), adjust for opponents faced, shrink early-season numbers toward last season, and test per position. Expect bigger matchup effects for QB/RB than WR/TE; if WR/TE matchup features don't beat noise, drop them rather than keep a flattering story.

## Vegas lines

- Implied team total = total ÷ 2 ± spread ÷ 2 (spread sign per nflverse: positive = home favored).
- Reported relationship to fantasy points: strongest for QB, moderate for WR/TE, noisiest for RB; most useful when implied totals differ by a lot (e.g. 28 vs 19), weak within ~3 points.
- Evidence here is mostly from practitioner sites; we test it directly (we have closing lines for every game since 2020).

## Injuries and availability (our own data)

From SCRUM-21 (skill positions, 2020–2025): Doubtful players played 0.4–2.5% of the time; Questionable + did-not-practice 41%; Questionable + limited 64%; Questionable + full 71%. Availability is a model input on its own, and injuries to teammates create **vacated opportunity** (SCRUM-38).

## Data gaps

| Need | Status | Options |
|---|---|---|
| Routes run (for YPRR, TPRR, route participation) | Not in nflverse weekly stats | Approximate from per-play participation data (players on field on dropbacks) if available for our seasons; else PFF/FTN (paid) |
| Expected TDs / expected fantasy points | ✅ `nflreadpy.load_ff_opportunity` (per player-week expected receptions, yards, TDs, fantasy points, by component) | Add to ingest (M1) |
| Opponent-adjusted defense | Build ourselves | SCRUM-36 |

## Candidate features for SCRUM-25/26 (ranked hypotheses)

1. **All positions:** snap share; team implied total; expected fantasy points and expected TDs (from `ff_opportunity`); age / years in league; availability (injury status, games missed).
2. **QB:** rush attempts & yards per game; pass attempts per game; EPA per dropback.
3. **RB:** weighted opportunity; target share; carry share; inside-5 carries.
4. **WR:** target share; WOPR; air-yards share; yards per team pass attempt; YPRR/TPRR if routes are obtainable.
5. **TE:** yards per game; targets per game; yards per team pass attempt; route participation if obtainable.
6. **Matchup:** opponent-adjusted defensive ratings by position (expect QB/RB > WR/TE).
7. **To down-weight:** raw TDs, red-zone counts, yards per target/carry, raw fantasy points allowed.

## Practitioner insights (Lucas's research)

Published stats say what's measurable; experienced managers know what isn't in the box score yet. Add findings here — each one becomes a testable hypothesis in SCRUM-25/26 or a rule for the draft recommender (M6).

Questions worth asking experienced managers:

1. What early signs of a **breakout** do you trust (camp reports, snap share jumps, a teammate leaving)? Which ones burned you?
2. How do you judge a **rookie** before he has NFL stats (draft capital, college production, landing spot)?
3. Which **coaching tendencies** matter most (pass rate, RB committees, how new coordinators change usage)?
4. When do you **trust or ignore a matchup**? Any positions where matchups never mattered for you?
5. How do you handle **injury-prone** players and **Questionable** tags on game day?
6. What **draft mistakes** do you see most often (reaching, position runs, ignoring bye weeks, handcuffs)?
7. What would make you **trust a tool's recommendation** over your own gut?

> _Findings:_
>
> -

## Sources

- 4for4 — [Most predictable WR stats](https://www.4for4.com/2024/preseason/most-predictable-wide-receiver-stats), [QB stats](https://www.4for4.com/2024/preseason/most-predictable-quarterback-stats), [TE stats](https://www.4for4.com/2024/preseason/most-predictable-tight-end-stats), [Production curves by age](https://www.4for4.com/2025/preseason/production-curves-positional-breakouts-prime-years-and-falloffs-age)
- PFF — [Yards per route run](https://www.pff.com/news/fantasy-football-metrics-that-matter-yards-per-route-run), [Weighted opportunity](https://www.pff.com/news/fantasy-football-weighted-opportunity-a-better-fantasy-football-predictor-than-raw-touches), [Value of a carry vs. a target](https://www.pff.com/news/fantasy-football-metrics-that-matter-carry-target-value), [Aging curves by position](https://www.pff.com/news/fantasy-football-metrics-that-matter-aging-curves-by-position), [Expected fantasy points & TD efficiency](https://www.pff.com/news/fantasy-football-expected-fantasy-points-the-most-and-least-efficient-touchdown-scorers)
- Fantasy Points — [Weighted opportunity for RBs](https://www.fantasypoints.com/nfl/articles/2023/weighted-opportunity-for-rbs), [XTD regression](https://www.fantasypoints.com/nfl/articles/2023/xtd-touchdown-regression-candidates), [Most important QB stats](https://www.fantasypoints.com/nfl/articles/2023/fantasy-points-data-most-important-qb-stats), [TE stats that matter (2026)](https://www.fantasypoints.com/nfl/articles/2026/fantasy-football-tight-ends-what-stats-matter)
- Sharp Football Analysis — [WR stats that matter](https://www.sharpfootballanalysis.com/fantasy/wide-receiver-stats-that-matter-fantasy-football-2024/), [QB stats](https://www.sharpfootballanalysis.com/fantasy/quarterback-stats-that-matter-fantasy-football-2025/), [TE stats](https://www.sharpfootballanalysis.com/fantasy/te-stats-that-matter-fantasy-football/), [RB red-zone vs. expectation](https://www.sharpfootballanalysis.com/fantasy/nfl-red-zone-stats-expectations-running-backs/)
- Action Network — [Weighted Opportunity Rating (WOPR)](https://www.actionnetwork.com/education/weighted-opportunity-rating-definition-how-find-boosted-receiver-production-with-this-stat); CBS Sports — [How to use air yards](https://www.cbssports.com/fantasy/football/news/2019-fantasy-football-draft-prep-how-should-i-use-air-yards-in-my-research-process/)
- Fantasy Life — [What matters for TEs (2026)](https://www.fantasylife.com/articles/fantasy/what-matters-for-tight-ends-in-fantasy-football-2026-ypg), [Age and WR performance](https://www.fantasylife.com/articles/redraft/how-does-age-impact-wr-performance-in-fantasy-football)
- PlayerProfiler — [Route participation (TE)](https://www.playerprofiler.com/article/mark-andrews-fantasy-football-ranking-stats-profile-week-3-metric-of-the-week-route-participation/)
- SumerSports — [Sticky football stats](https://sumersports.com/the-zone/sticky-football-stats-predictive-nfl-metrics/)
- Fantasy Footballers — [Mythbusters: matchups](https://www.thefantasyfootballers.com/articles/the-fantasy-football-mythbusters-making-the-most-of-matchups/), [Lifecycle of a dynasty WR](https://www.thefantasyfootballers.com/articles/the-lifecycle-of-a-dynasty-wide-receiver-fantasy-football/)
- ESPN — [When players peak and decline](https://www.espn.com/fantasy/football/story/_/id/37933720/2023-fantasy-football-players-peak-decline-quarterback-running-back-wide-receiver)
- Fantasy Projection Lab — [Vegas lines as projection inputs](https://fantasyprojectionlab.com/vegas-lines-and-fantasy-projections) (practitioner source; treat as unverified)
- League Station — [Defense vs. position (Bayesian-adjusted)](https://www.leaguestation.com/analytics/defense-vs-position)
- nflverse — `load_player_stats`, `load_pbp`, `load_snap_counts`, `load_ff_opportunity` via [nflreadpy](https://github.com/nflverse/nflreadpy)
