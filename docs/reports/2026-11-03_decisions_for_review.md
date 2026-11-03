# Decisions made while Lucas was away (2026-11-03): please review

All work is on branch `scrum-22-injuries`, not `main`. Nothing in Jira was changed. Tests: 163 passed.

## SCRUM-22: player-week master table (built, needs your OK)

Decided together before you left:
- Regular season only; Week 18 kept (the model decides whether to use it).
- Row = offensive snaps > 0 **or** a recorded stat. Zero = played and did nothing; no row = didn't play.
- Injured-mid-game games are **flagged, not deleted**.

Decided by Claude, with the recommended option. Reverse any of these:

| # | Decision | Alternative | Why |
|---|---|---|---|
| 1 | Injury report = option A: `on_injury_report`, `report_status`, `practice_code`, `injury` | One combined text column (B) | Separate columns are easier for a model to use; `on_injury_report = false` says "healthy" without making up a status |
| 2 | `left_injured` = injury in play text **and** snaps < ½ his usual share | Text only, or snap drop only | Measured both: text only includes players who came back (32% miss next game), snap drop only is mostly benchings (30%). Both together: **76%** miss next game vs 10% baseline |
| 3 | "Usual share" = median of his **previous 5 games** only | Median of his whole season | A season median would let later games change an earlier game's flag, which is future data leaking backwards. Earlier games only can't leak |
| 4 | No earlier games → flag if under 50% of snaps | Never flag first games | Rookies and new signings get hurt in Week 1 too |
| 5 | Shared jersey numbers resolved by the abbreviated name; if still ambiguous, leave unmatched | Guess | Only 10 of ~6,000 mentions end up unmatched |
| 6 | `python -m src.ingest` always rebuilds the table across **all** seasons, even with `--seasons 2026` | Rebuild only that season | The table is small (seconds to build) and the "previous 5 games" baseline crosses seasons |

## Known limitation

The flag misses injuries the play text doesn't mention, e.g. **Joe Burrow 2023 Wk 11** (44% of snaps, wrist, no "injured" in the text). "Snap drop + Out the next week + new injury on the report" could catch these, but snap drops alone are mostly benchings, so I didn't add it without asking. Your call.

## Please spot-check (you know the 2026 season better than I do)

2026 has 34 `left_injured` games in 8 weeks, a bit more than usual. Examples: Lamar Jackson Wk 8 (2% of snaps), Ja'Marr Chase Wk 4 (20%), Saquon Barkley Wk 2 and Wk 4, T.J. Hockenson Wk 5 (3%). Did these players really leave those games hurt? Full list:

```
python -c "import polars as pl; print(pl.read_parquet('data/processed/player_weeks.parquet').filter((pl.col('season')==2026) & pl.col('left_injured')).select('week','player_display_name','team','offense_pct','usual_offense_pct'))"
```

## Still open from before you left

- **Your question:** is using the *same week's* injury report as a model input data leakage? (Think about when it comes out compared to kickoff.)
- **Your research:** the 4 Sleeper rows where Sleeper gives +1 (Brightwell, McKeon, Bell, Nailor), see `docs/research/scoring_validation.md`.

## Reports

- `docs/reports/player_weeks_report.md`: rows per season, bust games, injury report mix, flag validation. Regenerate with `python scripts/report_player_weeks.py`.
