# Forecasting Methodology Log

## Scope
This log captures the current technical approach used to generate the Finance-ready revenue output.

Primary code assets:
- `notebook/02_case_study_fct.ipynb`
- `utils/forecast_utils.py`
- `utils/model_utils.py`
- `utils/test_utils.py`
- `sql_workflow/build_finance_revenue_view.py`

---

## Terminology Standard
- **Reported Revenue**: `total_gross_amount`
- **Final Revenue**: `filled_revenue`
- **Estimated Row**: `value_status = nowcasted`
- **Actual Row**: `value_status = actual`
- **Estimation Reason**: `nowcast_reason`
- **RPS**: revenue per 1M streams

This vocabulary is used consistently across workflow outputs and documentation.

---

## Current Assumptions
1. Revenue values are non-negative.
2. Streams are the primary complete signal.
3. Rows with `is_na=1` or `is_zero=1` are direct estimation targets.
4. Strict quality-correction targets are locally abnormal low-EPSR rows.
5. Missingness is not random across months, so random split alone is insufficient for selection.
6. Quality flagging is designed for targeted correction, not broad outlier deletion.

### Assumption rationale
- Late-month statement lag creates concentrated missingness.
- Stream volume is available earlier and is structurally tied to revenue.
- Local pricing behavior varies by `(territory_name, dsp)`, so anomaly rules must be local-aware.

---

## Pipeline Logic (Standardized)
Notebook 02 execution order is kept minimal and reproducible:
- Baselines and initial flag profile
- **Flagging Report (Current Rule)** near the beginning
- Model-based section including tree, KNN, and linear-family methods
- EPSR and Lag-Aware sections, then stress test and final deliverable

1. **SQL harmonization**
   - Standardize DSP labels.
   - Align stream geography to revenue territory logic.
   - Merge to one row grain with `is_na` / `is_zero` indicators.

2. **Quality targeting**
   - Flag strict local low-EPSR rows (`quality_flag_low_revenue_high_streams`) using fixed-effects de-mean residuals.
   - Fixed effects use `territory_name` and `dsp`; a single learned global residual threshold is applied after de-meaning.
   - Active parameters in notebook/workflow: `fe_threshold_quantile = 0.03`, `min_group_rows = 24`, `low_revenue_threshold = None`.

3. **Feature preparation**
   - Add time features and lag features (`gross_lag_1`, `streams_lag_1`, `epsr_lag_1`).

4. **Training filter**
   - Train on clean known rows only:
     - non-missing
     - non-zero
     - positive streams
     - not strict quality-flagged

5. **Estimation**
   - Use Lag-Aware Hierarchical Nowcast on target rows.

6. **Output + audit**
   - Produce final output fields.
   - Run hard-validity checks and robust RPS outlier checks.

### Lag-Aware Hierarchical Nowcast mechanics (technical)
The selected final rule combines three layers:
1. **EPSR anchor**
   - Generates structurally stable revenue from streams.
2. **Lag-aware adjustment**
   - Uses lag features (`gross_lag_1`, `streams_lag_1`, `epsr_lag_1`) to adapt to recent cohort shifts.
3. **Bounded reconciliation**
   - Applies constrained scaling at aggregation level to keep forecasts coherent while limiting extreme drift.

Design intent:
- anchor for robustness,
- lag signal for responsiveness,
- bounded reconciliation for consistency and governance.

---

## Candidate Models Evaluated
- Group Average baseline
- Naive lag-1 baseline
- EPSR benchmark
- XGBoost refined
- LightGBM refined
- KNN refined
- LinearRegression
- Ridge(alpha=10)
- Lag-Aware Hierarchical Nowcast

### Rule-by-rule pros and cons
1. **Group Average**
   - Pros: robust baseline; easy explainability.
   - Cons: no temporal adaptation.
2. **Naive lag-1**
   - Pros: captures immediate momentum.
   - Cons: sensitive to prior-month noise/gaps.
3. **EPSR**
   - Pros: transparent stream-rate mapping; strong governance benchmark.
   - Cons: less responsive to short-term local structural shifts.
4. **XGBoost / LightGBM**
   - Pros: expressive nonlinear modeling.
   - Cons: weaker stress stability and interpretability in this project context.
5. **KNN**
   - Pros: simple nonparametric fit.
   - Cons: poor high-dimensional stability and weakest aggregate stress performance.
6. **Lag-Aware Hierarchical Nowcast**
   - Pros: blends stream anchor + lag signal + bounded reconciliation; strongest fit-to-risk balance.
   - Cons: more moving parts than EPSR.

---

## Selection Criterion (Technical)
Final selection uses a multi-gate rule:
1. Competitive **mean WMAPE** under 30-repeat distribution-aware test.
2. Acceptable **p90 WMAPE** (tail-risk control).
3. Stable numeric behavior (no NaN/inf/negative collapse).
4. Governance fit (transparent output labels and reproducible run).

Selected model: **Lag-Aware Hierarchical Nowcast**.
EPSR is retained as benchmark/fallback.

---

## Test Process
### Distribution-aware backtest
- Build month-distribution profile from real target rows.
- Sample holdouts from known rows using the same month distribution.
- Train on remaining known rows.
- Score on sampled holdouts.
- Repeat 30 times and aggregate mean/std/p10/p50/p90.

### Why this process
It reflects production-like missingness better than IID random splits and gives a decision basis that includes both expected performance and downside behavior.

Validation assumptions:
- Holdout month mix must overlap with known months; otherwise stress-test output is treated as unavailable.
- Selection is based on both central tendency (`mean`) and tail risk (`p90`), not only one metric.

---

## Prediction Interval Method (Reusable Utility)
Utility function:
- `utils/forecast_utils.py::build_relative_prediction_intervals`

Construction steps:
1. Build calibration set on clean known rows with model predictions.
2. Compute absolute relative error:
   \[
   e_i = \frac{|y_i - \hat{y}_i|}{\max(|\hat{y}_i|, \epsilon)}
   \]
3. Learn uncertainty quantile at level \(1-\alpha\) (default 0.90), with group-level estimate (`territory_name`, `dsp`) and global fallback when sample size is small.
4. For each nowcasted row, produce interval:
   \[
   	ext{lower} = \hat{y}(1-q),\quad \text{upper} = \hat{y}(1+q)
   \]
   with lower bound clipped at 0.

Output columns (consistent in notebook + workflow):
- `nowcast_lower_90`
- `nowcast_upper_90`

Interpretation:
- Interval width is an uncertainty indicator.
- Intervals are populated only for estimated rows (`value_status = nowcasted`).

---

## Row Accounting (Current Verified Run)
- Row accounting is produced per run in `data/pipeline_run_summary.json`.
- Core monitored fields: `rows_total`, `rows_nowcasted`, `rows_nowcasted_missing_or_zero`, `rows_nowcasted_quality_flag`, `rows_actual`.

Interpretation: estimation is targeted, not blanket replacement.

---

## Output Standard (Canonical Files)
- `data/imputed_revenue_lag-aware_hierarchical_nowcast.csv`
- `data/imputed_revenue_lag-aware_hierarchical_nowcast.xlsx`
- `data/output_audit_suspicious_rows.csv`
- `data/output_audit_dsp_month_reliability.csv`
- `data/pipeline_run_summary.json`

Required output fields:
- `month`, `business_unit`, `territory_name`, `dsp`
- `total_streams`, `total_gross_amount`, `filled_revenue`
- `value_status`, `nowcast_reason`
- `quality_flag_low_revenue_high_streams`, `rps`

---

## Current Validation Snapshot
- Validation snapshot is tracked in `data/pipeline_run_summary.json` and `data/output_audit_dsp_month_reliability.csv`.
- Zero hard-invalid rows and low caution concentration are required before handoff.

Status: stable and handoff-ready.