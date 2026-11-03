# Scoring engine validation (SCRUM-33)

Run: `python -m src.scoring.validate`. Checked 2026-11-03 by Lucas and Claude.

## Result: pass

| Check | Rows | Match | Notes |
|---|---|---|---|
| nflverse `fantasy_points` / `fantasy_points_ppr`, 2020–2026 | 114,744 player-weeks | 239 differ, **0 unexplained** | Every difference is a return fumble nflverse leaves out (below) |
| Sleeper PPR / half / standard, 2022 wk 4 – 2025 (7 weeks) | 1,898 offensive player-weeks | **99.8%** (4 differ) | Using Sleeper's defaults: −1 per INT |
| Sleeper, 2022 wks 1–3 | 868 | 88% | Sleeper's stored points are inconsistent these weeks (below) |

## What we found

**1. Return fumbles count as lost fumbles.** nflverse's sack/rushing/receiving fumble columns miss fumbles on punt and kick returns; only `fumbles_lost_total` has them. ESPN and Sleeper both penalize them. Example: Skyy Moore, 2022 wk 3, muffed punt recovered by IND. ESPN shows "1 FUM 1 LOST", nflverse's own `fantasy_points` gives 0.0, and we now give −2.0. This affects ~0.5% of offensive player-weeks, mostly returners. The engine now uses `fumbles_lost_total` / `fumbles_total` when present.

**2. Sleeper doesn't penalize recovered fumbles by default.** We first guessed −1 per fumble from one player (Josh Allen, 2022 wk 3) and it was wrong: 25 sampled player-weeks with fumbles matched without it. We kept the setting ("Fumble (any)", default 0) because some leagues do use it.

**3. Sleeper's early-2022 points are inconsistent.** In weeks 1–3, ~12% of rows differ, and no single rule explains them: a few TEs got +0.5 per catch but most didn't (Kelce, Waller, Ertz), and a few recovered fumbles were penalized. Hand check: Cole Kmet, 2022 wk 3, 2 rec / 40 yds / 0 TD. ESPN's box score gives 6.0 PPR, which is ours; Sleeper shows 7.0. From week 4 on the mismatches nearly disappear. We treat this as a Sleeper data issue and leave those weeks out of the default sample.

## Still open

Four rows where Sleeper gives exactly +1 more, mostly special-teams players. Lucas is researching special-teams scoring (also a 2025 Eagles returner case):

| Season | Week | Player | Ours | Sleeper |
|---|---|---|---|---|
| 2022 | 4 | Gary Brightwell (RB) | 0.0 | 1.0 |
| 2022 | 10 | Sean McKeon (TE) | 1.1 | 2.1 |
| 2023 | 12 | Ronnie Bell (WR) | 0.0 | 1.0 |
| 2024 | 17 | Jalen Nailor (WR) | 19.1 | 20.1 |

## Principle

Trust stats, not other sites' point totals. Every source agreed on the raw stats; the disagreements were all in how (or how carefully) points were computed. To match a platform, apply its settings to our stats; never copy its stored points.
