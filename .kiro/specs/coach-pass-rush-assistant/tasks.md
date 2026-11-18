# Implementation Plan: Coach Pass Rush Assistant

## Overview

The backend (Python 3.12, FastAPI, pandas 2.2.3, numpy 2.1.3) is built bottom-up: data loading, cleaning, play-level dataset, linear regressions, recommender, reports, optional Nova, then API wiring. The dashboard (TypeScript, React, Vite, Recharts, Tailwind 3) is built from pure helper libraries up to the views and the app shell. Property tests use `hypothesis` (backend) and `fast-check` (dashboard), only if the Stack_Contract allows them; otherwise use seeded random generators looped 100+ times. Test files live in `tests/` (backend) and next to the sources (dashboard). Existing `.hackathon/`, `.venv/`, `.kiro/` folders are not touched.

## Tasks

- [ ] 1. Backend data layer
  - [ ] 1.1 Implement config and CSV loader
    - Create `backend/config.py` with `resolve_data_dir` (env value relative to project root, default `Datos/data`)
    - Create `backend/loader.py` with `REQUIRED_FILES`, `StartupError` (missing, file, category), `check_files`, `read_all` (read-only, completely loaded, never opens tracking files, categories unreadable, empty, malformed)
    - _Requirements: 1.1, 1.2, 1.3, 1.5, 1.6, 1.7, 1.8_

  - [ ]* 1.2 Write property tests for config and loader
    - **Property 1: Data directory resolution**
    - **Property 2: Missing files are all reported**
    - **Validates: Requirements 1.2, 1.3, 1.4**

  - [ ] 1.3 Implement play cleaning
    - Create `backend/cleaning.py` with `clean_plays` and `train_median_defenders_in_box`
    - Keep down 0 and quarter 5 unchanged, add `is_two_point` and `is_overtime` labels
    - Null `dropBackType`/`offenseFormation` become `Unknown`; null `defendersInBox` becomes training-set median
    - _Requirements: 2.1, 2.2, 2.3, 2.4_

  - [ ] 1.4 Implement play-level dataset builder
    - Create `backend/dataset.py` with `DataError` and `build_play_dataset`
    - Join on (gameId, playId); one row per Pass_Play; `pressure` and `sack` labels; yardsGained numeric
    - Match pff_role case-sensitively; compute blocking only from `Pass Block` and rushing only from `Pass Rush`; ignore role-inapplicable nulls
    - Raise `DataError` naming the file or column and return no partial output
    - _Requirements: 2.5, 2.6, 2.7, 2.8, 2.9, 2.10, 8.10_

  - [ ]* 1.5 Write property test for cleaning
    - **Property 3: Cleaning preserves labels and removes nulls**
    - **Validates: Requirements 2.1, 2.2, 2.3, 2.4**

  - [ ]* 1.6 Write property tests for dataset builder
    - **Property 4: Role-specific metrics and composite-key join**
    - **Property 5: One labeled row per Pass_Play**
    - **Property 6: Missing columns fail without partial output**
    - **Validates: Requirements 2.5, 2.6, 2.7, 2.8, 2.9, 2.10, 8.10**

- [ ] 2. Linear regression models
  - [ ] 2.1 Implement OLS utilities
    - Create `backend/linreg.py` with `split_indices` (seeded `default_rng`, 80/20), one-hot encoding with stored vocabulary, `fit_ols` (`numpy.linalg.lstsq`), and `metrics` (R-squared, MAE)
    - Use only numpy and pandas
    - _Requirements: 3.5, 4.1, 10.6_

  - [ ] 2.2 Implement Pressure_Model and Yards_Model
    - Create `backend/models.py` with `Situation`, `validate_situation`, `PressureModel` (clipped to 0..1, metrics with n_train and n_heldout), `YardsModel` (`pressure_coef`, `predict_raw`, `predict` rounded to 2 dp, metrics)
    - Train Yards_Model on the clipped Pressure_Model prediction; reject invalid situations naming the field without changing the model
    - Record fit failure (missing/empty dataset or fewer than 100 labeled plays) as unavailable
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.6, 3.7, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7_

  - [ ]* 2.3 Write property test for the split
    - **Property 7: Deterministic, shared 80/20 split**
    - **Validates: Requirements 3.5, 4.1**

  - [ ]* 2.4 Write property tests for models
    - **Property 8: Model outputs are bounded and rounded**
    - **Property 9: Reported metrics match held-out recomputation**
    - **Property 10: Invalid situations are rejected and leave models unchanged**
    - **Property 11: Yards respond linearly to pressure**
    - **Validates: Requirements 3.3, 3.4, 3.6, 4.2, 4.3, 4.4, 4.5, 4.7, 5.2, 5.7, 6.4**

  - [ ]* 2.5 Write unit tests for model failure paths
    - Fewer than 100 labeled plays and injected fit failure give "model unavailable"
    - _Requirements: 3.7, 4.6_

- [ ] 3. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 4. Recommender
  - [ ] 4.1 Implement the Recommender
    - Create `backend/recommender.py` with `CandidateCall`, `candidate_calls`, `rank_calls` (sort by pressure descending, then yards ascending), and `recommend`
    - Return ranked calls with pressure (0..1) and yards (1 dp), Best_Call, typical call, yards delta (1 dp), `two_point_conversion`, `low_confidence` (fewer than 30 matches), `recommendation_id`
    - Reject invalid input and unavailable predictions without a Best_Call
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7_

  - [ ]* 4.2 Write property tests for the Recommender
    - **Property 12: Ranking order and Best_Call**
    - **Property 13: Typical call and yards delta**
    - **Property 14: Situation flags**
    - **Validates: Requirements 5.1, 5.3, 5.4, 5.5, 5.6**

- [ ] 5. Reports, findings, and replay data
  - [ ] 5.1 Implement the Opponent_Report
    - Create `backend/reports.py` with team report: offense/defense pass plays, pressure rate (3 dp), sack rate (3 dp), average yards (2 dp); breakdowns by coverage and formation with counts; league average
    - Top five sackers and pressure players (count descending, name ascending, only count of 1 or more); fall back to player id and position `unknown`; empty report with message for teams without Pass_Plays
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.8, 8.10_

  - [ ]* 5.2 Write property tests for the Opponent_Report
    - **Property 21: Opponent report aggregates**
    - **Property 22: Top player rankings**
    - **Validates: Requirements 8.1, 8.2, 8.3, 8.4, 8.5, 8.6**

  - [ ] 5.3 Implement findings computation
    - Extend `backend/reports.py` with the five summary values (null on zero denominator), group tables by coverage, formation, and defendersInBox, top 10 players by sacks, top 10 teams by pressure rate; all computed from loaded data
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5_

  - [ ]* 5.4 Write property test for findings
    - **Property 23: Findings computed from data**
    - **Validates: Requirements 9.2, 9.4, 9.5**

  - [ ] 5.5 Implement games list and replay plays
    - Extend `backend/reports.py` with `list_games` (date ascending, with home and visitor team) and `replay_plays` (quarter ascending, game clock descending) with Best_Call, actual call, actual pressure outcome, and actual yards
    - _Requirements: 7.1, 7.2, 7.3_

  - [ ]* 5.6 Write property tests for games and replay
    - **Property 16: Games sorted by date**
    - **Property 17: Replay play ordering**
    - **Validates: Requirements 7.1, 7.2**

- [ ] 6. Optional Amazon Nova extension
  - [ ] 6.1 Implement the Nova explainer
    - Create `backend/nova.py` using `boto3.client("bedrock-runtime", region_name=AWS_REGION)` only, model from `BEDROCK_MODEL_ID`, 10 second timeout, output truncated to 1,000 characters
    - Never read, verify, or log credentials; missing env or Bedrock failure returns an "explanation unavailable" result
    - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.7_

  - [ ]* 6.2 Write property and unit tests for Nova
    - **Property 25: Explanation length cap**
    - Mock Bedrock: failure, timeout, and missing env map to "explanation unavailable"; core recommendation code never calls Nova
    - **Validates: Requirements 11.1, 11.3, 11.5, 11.7**

- [ ] 7. API wiring
  - [ ] 7.1 Define API schemas
    - Create `backend/schemas.py` with Pydantic request and response models for recommend, metrics, games, replay, report, findings, vocab, and explain
    - _Requirements: 5.2, 6.4, 7.3, 8.1, 9.1_

  - [ ] 7.2 Implement startup state and fallback app in `backend/main.py`
    - Module-level `app`; startup builds `AppState` (load, clean, dataset, split, fit models, candidates, findings, team stats, in-memory recommendation store)
    - On `StartupError` exit non-zero with the file list and expose a fallback that returns 503 listing missing files for every `/api/` path
    - Model fit failures recorded as unavailable without stopping other endpoints
    - _Requirements: 1.1, 1.3, 1.4, 1.5, 1.8, 1.10, 3.1, 3.7, 4.6, 10.1_

  - [ ] 7.3 Implement routes and static serving in `backend/main.py`
    - `GET /health`; under `/api/`: `vocab`, `recommend` (422 naming invalid fields), `metrics`, `games`, `games/{gameId}/plays`, `teams`, `teams/{team}/report`, `findings`, `recommendations/{id}/explain` (404 for unknown id without calling Bedrock; 503 when unavailable)
    - Mount `dashboard/dist` with `StaticFiles(html=True)` last, same host and port
    - _Requirements: 1.9, 3.4, 4.5, 5.7, 6.4, 6.5, 7.1, 7.2, 8.8, 8.9, 10.1, 10.4, 11.1, 11.3, 11.6, 11.7_

  - [ ]* 7.4 Write API unit tests
    - `/health` body, 422 on invalid situations, 503 when the model is unavailable, 404 on unknown recommendation, fallback app listing missing files, empty team report, route enumeration shows only `/api/*`, `/health`, and static root
    - _Requirements: 1.3, 1.4, 1.9, 5.7, 6.4, 8.8, 10.1, 11.6_

- [ ] 8. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 9. Dashboard
  - [ ] 9.1 Scaffold the dashboard
    - Create `dashboard/` with Vite (`base: './'`), React, TypeScript, Recharts, Tailwind CSS 3, using only versions allowed by the Stack_Contract
    - _Requirements: 10.3, 10.5_

  - [ ] 9.2 Implement the API client
    - Create `dashboard/src/api.ts` with `getJson`/`postJson` using relative URLs (`api/...`, no leading slash) and a 10 second AbortController timeout
    - _Requirements: 10.2, 10.9_

  - [ ] 9.3 Implement validation and formatting libraries
    - Create `dashboard/src/lib/validate.ts` (`validateWhatIf`) and `dashboard/src/lib/format.ts` (`formatPercent`, `formatYards`, `formatOrNA`, `quarterLabel`, `downLabel`)
    - _Requirements: 6.3, 7.4, 7.5, 9.1, 9.3_

  - [ ]* 9.4 Write property tests for validation and formatting
    - **Property 15: What-If client validation**
    - **Property 18: Replay labels**
    - **Property 24: Safe numeric formatting**
    - **Validates: Requirements 6.3, 7.4, 7.5, 9.1, 9.3**

  - [ ] 9.5 Implement replay navigation library
    - Create `dashboard/src/lib/replay.ts` with `navReduce`, `canPrev`, `canNext`, `cumulative` (sums rounded to 1 dp)
    - _Requirements: 7.6, 7.7, 7.8, 7.9, 7.10_

  - [ ]* 9.6 Write property tests for replay library
    - **Property 19: Navigation bounds**
    - **Property 20: Cumulative comparison**
    - **Validates: Requirements 7.6, 7.7, 7.8, 7.9, 7.10**

  - [ ] 9.7 Implement WhatIfView
    - Selectable formation and coverage from `api/vocab` only; submit disabled until all four inputs have values; client validation with retained values; ordered results with percent and one-decimal yards; mark result outdated within 500 ms of any change; error state with retry; loading indicator
    - _Requirements: 6.1, 6.2, 6.3, 6.5, 6.6, 6.7, 10.8, 10.9_

  - [ ] 9.8 Implement ReplayView
    - Game selector, step forward/backward and jump controls with disabled states, play detail (Best_Call, actual call, outcome, yards), overtime and two-point labels, empty-game message, cumulative comparison, error and retry keeping the previous play
    - _Requirements: 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 7.10, 7.11, 10.8, 10.9_

  - [ ] 9.9 Implement OpponentReportView
    - Team selector; Recharts charts comparing team versus league average for pressure rate, sack rate, and average yards; top player lists; no-data message and error message with retained previous team
    - _Requirements: 8.7, 8.8, 8.9, 10.8, 10.9_

  - [ ] 9.10 Implement FindingsView
    - Five summary values with units or "N/A", three Recharts group charts (coverage, formation, defendersInBox), top 10 player and team rankings, error state with retry and no placeholder values
    - _Requirements: 9.1, 9.3, 9.4, 9.5, 9.6, 10.8, 10.9_

  - [ ] 9.11 Wire the app shell
    - Create `App.tsx` and `main.tsx` with navigation among the four views; each view keeps its own loading and error state so one failure leaves the others usable
    - _Requirements: 10.2, 10.8, 10.9_

  - [ ]* 9.12 Write dashboard UI tests
    - What-If: submit disabled until complete, outdated mark, error keeps values; Replay: empty game, boundary controls, retry; Findings: error state; Opponent report: no-data message
    - _Requirements: 6.1, 6.6, 6.7, 7.7, 7.8, 7.9, 7.11, 8.8, 9.6_

- [ ] 10. Compliance and integration
  - [ ]* 10.1 Write compliance smoke tests
    - Dashboard sources contain no request URL beginning with `/` or containing a scheme; `vite build` exits 0 into `dashboard/dist` with base `./`; dependency files contain only Stack_Contract packages and no ML library; root URL serves the index and an API route responds on the same port
    - _Requirements: 10.2, 10.3, 10.4, 10.5, 10.6, 10.7_

- [ ] 11. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Property tests validate the universal correctness properties in the design; unit tests cover specific examples and edge cases
- Confirm `hypothesis` and `fast-check` are allowed by the Stack_Contract before adding them

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "1.3", "2.1", "6.1", "9.1"] },
    { "id": 1, "tasks": ["1.2", "1.4", "1.5", "2.2", "2.3", "6.2", "9.2", "9.3", "9.5"] },
    { "id": 2, "tasks": ["1.6", "2.4", "2.5", "4.1", "5.1", "7.1", "9.4", "9.6", "9.7", "9.8", "9.9", "9.10"] },
    { "id": 3, "tasks": ["4.2", "5.2", "5.3", "9.11"] },
    { "id": 4, "tasks": ["5.4", "5.5", "9.12"] },
    { "id": 5, "tasks": ["5.6", "7.2"] },
    { "id": 6, "tasks": ["7.3"] },
    { "id": 7, "tasks": ["7.4"] },
    { "id": 8, "tasks": ["10.1"] }
  ]
}
```
