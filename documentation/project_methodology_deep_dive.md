# CMG Revenue Nowcasting Project: End-to-End Methodology Deep Dive

## 1. Purpose of This Document
This document explains, in practical and mathematical terms, exactly what the project does from raw inputs to final deliverables.

It is written to be:
- human-readable for business stakeholders,
- auditable for technical reviewers,
- complete enough for a new analyst to reproduce and defend the logic.

## 2. Business Problem and Design Goal
The project solves a finance reporting gap:
- revenue has missing (`is_na=1`) and zero (`is_zero=1`) values,
- streams are comparatively complete and timely,
- late months have concentrated incompleteness.

Goal:
- produce a single Finance-ready revenue field (`filled_revenue`) that keeps actual values when reliable and nowcasts only where needed,
- attach reasons, uncertainty bands, and risk audits.

What this means in day-to-day analyst language:
- if a row already has trustworthy revenue, we leave it untouched,
- if revenue is missing/zero or clearly suspicious versus local history, we estimate it,
- every estimate is labeled with a reason so finance users can filter, challenge, or approve quickly.

## 3. Project Structure (What Lives Where)
Repository top-level:
- `data/`: merged input and all final outputs.
- `notebook/`: analysis notebooks.
- `sql_pipeline/`: SQL harmonization specification.
- `sql_workflow/`: reproducible production-style script.
- `utils/`: reusable feature, model, and test utilities.
- `documentation/`: methodology and audit logs.

Operationally important files:
- `sql_workflow/build_finance_revenue_view.py`: production run entry point.
- `utils/forecast_utils.py`: feature engineering, baselines, metrics, stress-test, prediction intervals, quality flags.
- `utils/model_utils.py`: model training/prediction functions and lag-aware nowcast logic.
- `utils/test_utils.py`: baseline wrappers used in stress-test registry.
- `notebook/01_case_study_eda.ipynb`: exploratory data analysis and harmonization checks.
- `notebook/02_case_study_fct.ipynb`: model benchmarking, selection, final deliverable build, and visualization.

## 4. Data Model and Core Variables
Main row grain after harmonization:
- `(month, business_unit, territory_name, dsp)`

Core numeric fields:
- `total_streams`
- `total_gross_amount` (reported revenue, may be missing/zero)
- `filled_revenue` (final output after nowcasting when needed)

Row-status fields:
- `is_na`, `is_zero`
- `value_status` in `{actual, nowcasted}`
- `nowcast_reason` in `{missing_or_zero_revenue, quality_flag_low_revenue_high_streams, N/A}`

Derived business metric:
- RPS (revenue per million streams)

\[
\text{RPS} = \frac{\text{filled\_revenue}}{\text{total\_streams}/1{,}000{,}000}
\]

## 5. End-to-End Workflow Logic (High Level)
1. Harmonize Revenue + Streams with SQL.
2. Mark initial missing/zero revenue flags.
3. Add quality flags for suspiciously low revenue rows using fixed-effects residual logic.
4. Build lag/time features.
5. Train models on clean known rows.
6. Nowcast only selected target rows.
7. Build prediction intervals.
8. Build final deliverable and labels.
9. Run final audits and reliability risk flags.
10. Export CSV/XLSX + audit files + JSON run summary.

Decision gates in plain language:
- Gate A (Can we compare apples-to-apples?): harmonization enforces one shared grain.
- Gate B (Should this row be estimated?): target mask + quality mask decide eligibility.
- Gate C (Is the estimate stable enough?): bounded blending + reconciliation + DSP guardrail.
- Gate D (How much trust should we assign?): interval width + nowcast rate drive risk labels.

## 6. Detailed SQL Harmonization Logic
Implemented in `harmonize_with_sql(...)` inside `sql_workflow/build_finance_revenue_view.py`.

### 6.1 Input handling
Two supported modes:
- CSV mode: `read_csv_auto(...)` views for `revenue` and `streams`.
- Excel mode: load both sheets in pandas, then register as DuckDB views.

### 6.2 CTE-by-CTE explanation
The SQL query is composed of layered CTEs. Each step is intentional.

1. `revenue_standardized`
- Selects base revenue columns.
- Maps many raw DSP labels into canonical buckets using `CASE`:
  - YouTube family -> `YouTube`
  - `Tiktok` -> `TikTok`
  - Amazon variants -> `Amazon`
  - Apple variants -> `Apple`
  - others -> `Other`
- Why: avoids DSP granularity mismatch with streams.

2. `revenue_aggregated`
- Aggregates revenue to target grain:
  - group by `(month, business_unit, territory_name, dsp)`
  - sum `total_gross_amount`
- Why: removes duplicated fragmentation at sub-DSP level.

3. `known_territories`
- Collects distinct territories present in aggregated revenue.
- Why: defines valid territory universe for streams alignment.

4. `streams_standardized`
- Maps stream `country` to revenue territory logic:
  - if `country` exists in `known_territories`, keep it,
  - else map to `All Other Locations`.
- Why: ensures geographic dimensions are join-compatible.

5. `streams_aggregated`
- Aggregates streams at same target grain:
  - `(month, business_unit, territory_name, dsp)`
  - sum `total_streams`

6. `merged_data`
- Performs `FULL OUTER JOIN` on all grain keys.
- Uses `COALESCE` for keys to preserve rows existing on either side.
- Why:
  - keeps stream-only rows (potential missing revenue targets),
  - keeps revenue-only rows (data completeness diagnostics).

7. Final `SELECT`
- Outputs merged fields.
- Creates indicators:
  - `is_na = 1` when revenue is NULL,
  - `is_zero = 1` when revenue equals zero.

### 6.3 Why this SQL design is audit-friendly
- each transform is explicit and named,
- mappings are visible and reviewable,
- no hidden row-level mutation loops,
- grain alignment is deterministic.

## 7. Function-by-Function Technical Explanation

## 7.1 `utils/forecast_utils.py`

1. `add_time_columns(data)`
- Converts `month` to datetime.
- Adds:
  - `month_str` (YYYY-MM)
  - `month_num` (1-12)
  - `quarter` (1-4)
- Purpose: consistent calendar features used across models.

2. `add_quality_flags(data, stream_col, target_col, low_revenue_threshold, min_group_rows, fe_threshold_quantile)`
- Builds `quality_flag_low_revenue_high_streams` using fixed-effects residuals on log-RPS.

Detailed math:
- Known-row filter:
\[
\mathcal{K} = \{i: is\_na_i=0,\ is\_zero_i=0,\ streams_i>0,\ revenue_i>0\}
\]
- Row EPSR:
\[
\text{epsr}_i = \frac{revenue_i}{streams_i}
\]
- Log-RPS transform:
\[
z_i = \log(\max(\text{epsr}_i, 10^{-12}))
\]
- Fixed-effects structure:
\[
\hat z_i = \mu + \alpha_{territory(i)} + \beta_{dsp(i)}
\]
with effects shrunk to 0 if group count is below `min_group_rows`.
- Residual:
\[
r_i = z_i - \hat z_i
\]
- Learn lower-tail threshold:
\[
\tau = Q_{q}(\{r_i : i \in \mathcal{K}\}),\quad q=\text{fe\_threshold\_quantile}
\]
- Flag rule:
\[
flag_i = 1\{i\in \mathcal{K}\ \land\ r_i \le \tau\}
\]
plus optional absolute cap if `low_revenue_threshold` is set.

Why it is valid:
- compares rows against local territory and DSP behavior,
- avoids naive global thresholds,
- focuses on left-tail under-reporting patterns.

Important practical note:
- Active workflow and utility default are aligned at `fe_threshold_quantile=0.03` to avoid accidental parameter drift.

3. `add_lag_features(data, group_cols, target_col, stream_col, lag=1)`
- Sorts by group and month.
- Adds lag features:
  - `gross_lag_1`
  - `streams_lag_1`
  - `epsr_lag_1 = gross_lag_1 / streams_lag_1`
- Purpose: captures short-term temporal change per cohort.

4. `compute_wmape(actual, pred)`
\[
\text{WMAPE} = \frac{\sum_i |y_i-\hat y_i|}{\sum_i |y_i|}
\]
- Returns NaN if denominator is zero.

5. `evaluate_predictions(data, target_col, pred_col, label)`
- Returns compact metrics dict with MAE and WMAPE.

6. `prepare_eval_frames(data, stream_col, excluded_months)`
- Returns:
  - `work`: with time columns,
  - `known_all`: known rows,
  - `known_eval`: known rows excluding specified months,
  - `target_rows`: missing or zero rows.

7. `fit_group_average_lookup(...)` and `apply_group_average_lookup(...)`
- Cohort-mean baseline with global fallback.

8. `fit_epsr_lookup(...)` and `apply_epsr_lookup(...)`
- Learns EPSR rates by hierarchical fallback:
  - full cohort,
  - territory+dsp,
  - dsp,
  - global.
- Prediction:
\[
\hat y = streams \times epsr\_rate\_used
\]

9. `run_distribution_backtest(known_pool, target_pool, fit_predict_fn, ...)`
- Simulates production missingness by month.
- Steps:
  - estimate month distribution from `target_pool`,
  - sample holdout rows from known rows with matching month proportions,
  - train on remaining known,
  - evaluate on synthetic holdout,
  - repeat `n_repeats`.
- Returns:
  - overall repeat-level metrics,
  - monthly breakdown metrics.

Why it is valid:
- preserves non-random missingness pattern by month.
Guardrail:
- returns empty frames when target-month distribution has no overlap with known months.

10. `build_relative_prediction_intervals(...)`
- Uses calibration residuals to build multiplicative intervals.

Math:
- Relative error on calibration:
\[
e_i = \frac{|y_i-\hat y_i|}{\max(|\hat y_i|, \epsilon)}
\]
- Quantile at confidence level:
\[
q = Q_{1-\alpha}(e)
\]
- Prediction interval per row:
\[
L = \hat y(1-q),\quad U = \hat y(1+q)
\]
with `L` clipped at 0.
- Supports group-level quantiles with global fallback when group sample size is below `min_group_rows`.

## 7.2 `utils/model_utils.py`

1. `split_time_holdout(eval_df)`
- Uses latest month as validation.
- Falls back to random split only if fewer than 2 months exist.
- Purpose: preserves temporal realism in holdout.

2. `_non_negative(pred)`
- Clips predictions at 0.

3. `_apply_dsp_rps_guardrail(train_data, scored, pred_col, stream_col, target_col)`
- Learns DSP RPS caps from training data:
\[
cap_{dsp} = \max(3\cdot median_{dsp}(RPS),\ 1.25\cdot p95_{dsp}(RPS))
\]
- Caps predicted revenue implied by cap and streams.
- Prevents extreme spikes, especially in sparse/noisy cohorts.

4. `_build_feature_matrices(...)`
- Shared tree/KNN feature matrix builder.
- Adds log-stream feature:
\[
log\_streams = \log(1 + streams)
\]
- Includes lag and calendar features.
- One-hot encodes categoricals.
- Produces raw target and log-target for tree models.

5. `_build_linear_feature_matrices(...)`
- Similar to above but tailored for linear-family models.
- Uses median imputation on lag features and one-hot encoding with `drop_first=True`.

6. `fit_predict_linear_refined(...)`
- Fits `LinearRegression` on engineered matrix.
- Returns non-negative predictions.

7. `fit_predict_ridge_refined(..., alpha=10.0)`
- Fits `Ridge` with L2 regularization.
- Returns non-negative predictions.

8. `fit_predict_xgb_refined(...)`
- Fits `XGBRegressor` on log-target:
\[
z = \log(1+y),\quad \hat y = \exp(\hat z)-1
\]
- Uses conservative hyperparameters to reduce instability.

9. `fit_predict_lgbm_refined(...)`
- LightGBM counterpart with similar log-target treatment and conservative settings.

10. `fit_predict_knn_refined(...)`
- Scales features using `StandardScaler`.
- Fits distance-weighted KNN.

11. `fit_predict_lag_hier_nowcast(...)`
- Final selected strategy.

Detailed mechanics:
- Step A: EPSR anchor prediction.
- Step B: lag projection using stream growth:
\[
stream\_growth_t = \frac{streams_t}{streams_{t-1}}
\]
clipped to `[0.50, 1.80]`.
- Step C: anchor-first blend:
  - lag weight = 0.20 when lag exists, else 0.
- Step D: bound adjustment ratio around anchor:
\[
ratio = \frac{base\_pred}{anchor\_pred}\in[0.85,1.15]
\]
- Step E: reconcile at `(month, dsp)`:
\[
scale = \frac{\sum anchor}{\sum blended}\in[0.85,1.15]
\]
- Step F: apply DSP RPS guardrail.

Why this is strong:
- EPSR provides structural stability,
- lag adds responsiveness,
- bounded reconciliation and guardrail control tail risk.

## 7.3 `utils/test_utils.py`

1. `group_avg_fit_predict(...)`
- Wrapper for group-average lookup baseline.

2. `naive_fit_predict(...)`
- Lag-1 baseline within cohorts.
- Uses train median fallback when lag is missing.

3. `epsr_fit_predict(...)`
- Wrapper for EPSR lookup baseline.

## 8. Production Workflow Script Explained (`build_finance_revenue_view.py`)

Main orchestrator stages:

1. Parse CLI args.
2. Build or load merged data.
3. Run `run_nowcast_and_audit(...)`.
4. Save output artifacts.
5. Save run summary JSON.

Inside `run_nowcast_and_audit(...)`:
- first-pass flagging,
- target row selection,
- lag feature build and re-flagging,
- clean-train filtering,
- nowcast prediction,
- calibration and prediction intervals,
- final labels and RPS,
- hard-invalid + robust suspicious audits,
- DSP-month reliability audit,
- compact JSON summary.

Operational interpretation:
- the function is effectively a policy engine, not just a model call,
- it encodes what to estimate, how to estimate, and how to label confidence,
- this keeps business behavior reproducible even if model internals evolve later.

## 9. Model Evaluation Methodology

Holdout concept:
- time-aware split using latest month as validation.

Stress test concept:
- production-like missingness simulation by month distribution,
- repeated 30 times.

Decision criteria:
- low mean WMAPE,
- controlled p90 WMAPE,
- stable numerics,
- governance clarity.

## 10. Reliability and Audit Outputs

1. `output_audit_suspicious_rows.csv`
- rows where robust z-score on log-RPS exceeds threshold and row is otherwise valid.

Robust z-score:
\[
z_i = 0.6745\frac{x_i - median(x)}{MAD(x)}
\]
with \(x=\log(1+RPS)\).

2. `output_audit_dsp_month_reliability.csv`
- one row per `(month, dsp)` with:
  - nowcast share,
  - uncertainty width,
  - risk label.

Risk rule:
- `caution` if `nowcast_rate >= 0.80` OR `uncertainty_width_pct >= 35`.

## 11. EDA Process Summary
EDA notebook validates:
- missing and duplicate profiles,
- category levels (DSP, territory, BU),
- trend behavior across month by cohort,
- harmonization sanity after mapping and merge,
- expected concentration of missingness.

Why this matters:
- modeling assumptions (stream completeness, non-random missingness, grouping grain) are evidence-backed before forecasting.

## 12. Strengths of the Approach
- Strong auditability:
  - SQL CTE harmonization,
  - explicit row-status labels,
  - exported risk and run summaries.
- Methodological realism:
  - distribution-aware stress test mirrors actual missingness.
- Stability controls:
  - bounded adjustments,
  - group-level fallback design,
  - RPS guardrails.
- Business interpretability:
  - EPSR anchor remains explainable,
  - reason labels show why each row was nowcasted.

Implementation strength worth highlighting:
- notebook and script both rely on shared utilities, reducing logic drift between experimentation and production-style runs.

## 13. Limitations and Practical Caveats
- Fixed-effects flagging depends on sufficient group history; sparse groups are simplified by effect fallback.
- Month-pattern stress test captures month-level missingness, but not all possible operational shocks.
- Interval construction is residual-based (empirical), not a full probabilistic structural model.
- Late months with very high nowcast rates should be treated with caution even with guardrails.
- Some documentation values can become stale if outputs are regenerated; JSON summary files should be treated as source of truth for counts.

Implementation caveat:
- if someone bypasses utility functions and rewrites logic inline in notebooks, auditability will decline quickly; keeping notebooks thin is an ongoing governance requirement.

## 14. Assumptions to State Explicitly in Stakeholder Reviews
- Streams are treated as the most reliable near-real-time signal.
- Revenue-stream relationship is stable enough locally to support EPSR anchoring.
- Missingness pattern is materially month-dependent.
- Strict low-residual flagging is for targeted correction, not broad anomaly removal.
- Final outputs are decision-support estimates, not ledger replacements.

## 15. Reproducibility Checklist
- Run from project root.
- Use:
  - `python sql_workflow/build_finance_revenue_view.py --input-merged data/merged_data.csv`
  - or rebuild merged data from source Excel with `--refresh-merged`.
- Confirm generated artifacts:
  - forecast CSV/XLSX,
  - suspicious rows audit,
  - DSP-month reliability audit,
  - `pipeline_run_summary.json`.
- Verify no hard-invalid rows and review `caution` concentration before handoff.

## 17. Implementation Verification Snapshot
This section records whether the documented plan is actually reflected in the current codebase.

Verification items checked:
- SQL harmonization has explicit DSP mapping, territory alignment, and outer-join preservation.
- Nowcast target definition includes both missing/zero rows and strict quality-flag rows.
- Final selected production rule is lag-aware hierarchical nowcast.
- Prediction interval logic is relative-error quantile based with group fallback.
- DSP-month reliability risk rule is implemented with thresholds:
  - `nowcast_rate >= 0.80` OR
  - `uncertainty_width_pct >= 35`.
- Export set includes finance output files and dedicated audit artifacts.

Verification result:
- Core plan-to-code alignment is correct for harmonization, targeting, nowcasting, uncertainty, and audits.
- Minor consistency adjustment applied in utilities: default `fe_threshold_quantile` now matches active workflow convention (`0.03`) to reduce accidental drift in ad-hoc calls.

Practical recommendation:
- continue treating `sql_workflow/build_finance_revenue_view.py` as the canonical run path for handoff outputs, and use notebooks primarily for analysis/communication.

## 16. Recommended Reading Order for New Contributors
1. This file (`project_methodology_deep_dive.md`).
2. `sql_workflow/build_finance_revenue_view.py`.
3. `utils/forecast_utils.py`.
4. `utils/model_utils.py`.
5. `notebook/01_case_study_eda.ipynb` then `notebook/02_case_study_fct.ipynb`.
6. `documentation/forecasting_methodology_log.md` for concise operational notes.

## 18. Case Study Presentation Guideline
Use this sequence when presenting to stakeholders so the story is clear, credible, and decision-oriented.

1. Start with the business problem (1-2 slides)
- Revenue has missing/zero values while streams are more complete.
- Explain the impact in plain terms: reporting delays, decision uncertainty, and inconsistent comparability by month.

2. Explain data harmonization before modeling (1-2 slides)
- Show the target grain: `(month, business_unit, territory_name, dsp)`.
- Briefly explain DSP bucketing and territory alignment.
- Emphasize that this step makes revenue and streams comparable.

3. Show quality controls before forecasts (1 slide)
- Missing/zero flags and fixed-effects quality flag.
- State the empirical threshold choice explicitly: `fe_threshold_quantile = 0.03` (kept because it performs best in your observed runs).

4. Present model evaluation in two layers (2 slides)
- Layer A: quick time-aware holdout (latest-month validation).
- Layer B: 30-repeat distribution-aware stress test (production-like missingness).
- Explain selection logic: choose method by mean performance + stability spread, not by one lucky split.

5. Explain the selected method in business language (1 slide)
- Lag-Aware Hierarchical Nowcast = EPSR anchor + lag adjustment + bounded reconciliation + DSP guardrail.
- Highlight why this is trustworthy: responsive but bounded.

6. Present uncertainty and risk framing (1 slide)
- 90% prediction intervals from empirical relative errors.
- DSP-month risk label rule:
  - `caution` when `nowcast_rate >= 0.80` OR `uncertainty_width_pct >= 35`.

7. Close with deliverables and governance (1 slide)
- Final output includes `filled_revenue`, `value_status`, `nowcast_reason`, bounds, and RPS.
- Mention audit exports and run summary JSON for traceability.

8. Suggested speaking style
- Keep formulas minimal in the main narrative; move full math to appendix.
- Prefer "what decision this enables" over "which model is fancy".
- For every technical component, include one plain-language sentence and one business implication sentence.

9. Suggested Q&A readiness checklist
- Be ready to explain why `0.03` is used for the quality-flag quantile.
- Be ready to explain when a row is left as actual vs nowcasted.
- Be ready to explain what a `caution` DSP-month means operationally.
- Be ready to explain limitations (late-month concentration, empirical intervals, sparse groups).
