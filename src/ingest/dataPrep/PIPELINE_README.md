# NBA Prediction - Data Preparation Pipeline

One entry point, three modes:

```bash
python -m src.ingest.dataPrep.pipeline --mode ingest    # steps 1, 3, 3a  (downloads; needs nba_api)
python -m src.ingest.dataPrep.pipeline --mode features  # steps 2, 4, 5, 6, 6a
python -m src.ingest.dataPrep.pipeline --mode full      # all of the above
python tools/run_features.py                            # 3a + features, cache reads off (~40 s)
```

| Step | Module | Output (`data/processed/`) |
|---|---|---|
| 1 | `fetch_games` | `nba_games_with_game_id.csv` — schedule, scores, spread, NBA game id |
| 3 | `fetch_player_boxscores` | `all_boxscores.csv`, `team_boxscores_v3.csv` (per-game cache in `data/cache/boxscores/<family>/`) |
| 3a | `aggregate_player_to_team` | `team opp stats.csv` — team and opponent sums, V3 team metrics, usage concentration |
| 2 | `calculate_elo_and_rest` | `nba_games_with_game_id_processed.csv` — Elo, rest, form, streaks |
| 4 | `engineer_advanced_features` | `df_model.csv`, `team_game.csv` — pregame absences, top-8 form, bench, after-OT |
| 5 | `calculate_travel_fatigue` | `df_model_with_travel.csv` |
| 6 | `engineer_rolling_features` | `df_model_2.csv` — rolling / EWM / season-to-date team stats, style matchup |
| 6a | `finalize_feature_selection` | `df_model_3.csv` — the model table `experiment.py` reads |

`df_model_2.csv` still contains the raw same-game box score columns; only step 6a's allowlist removes them. Read
`df_model_3.csv`, never `df_model_2.csv`, for modeling. Every feature must use information available before tip-off:
after changing a step, run `python -m pytest tests -q`, `python tools/leak_screen.py` and
`RUN_PIT=1 python -m pytest tests -k point_in_time` (see `docs/feature-leakage-audit-and-fix-plan.md`).
