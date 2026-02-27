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
4. Strict quality-correction targets are locally abnormal low-revenue rows.
5. Missingness is not random across months, so random split alone is insufficient for selection.

### Assumption rationale
- Late-month statement lag creates concentrated missingness.
- Stream volume is available earlier and is structurally tied to revenue.
- Local pricing behavior varies by `(territory_name, dsp)`, so anomaly rules must be local-aware.

---

## Pipeline Logic (Standardized)
1. **SQL harmonization**
   - Standardize DSP labels.
   - Align stream geography to revenue territory logic.
   - Merge to one row grain with `is_na` / `is_zero` indicators.

2. **Quality targeting**
   - Flag strict local low-EPSR rows (`quality_flag_low_revenue_high_streams`).

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

---

## Candidate Models Evaluated
- Group Average baseline
- Naive lag-1 baseline
- EPSR benchmark
- XGBoost refined
- LightGBM refined
- KNN refined
- Lag-Aware Hierarchical Nowcast

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

---

## Row Accounting (Current Verified Run)
- Harmonized rows: `2000`
- Missing/zero target rows: `286`
- Strict quality-correction rows: `36`
- Final estimated rows: `322` (`286 + 36`)
- Final actual rows: `1678`

Interpretation: estimation is targeted, not blanket replacement.

---

## Output Standard (Canonical Files)
- `data/imputed_revenue_lag-aware_hierarchical_nowcast.csv`
- `data/imputed_revenue_lag-aware_hierarchical_nowcast.xlsx`
- `data/output_audit_suspicious_rows.csv`
- `data/pipeline_run_summary.json`

Required output fields:
- `month`, `business_unit`, `territory_name`, `dsp`
- `total_streams`, `total_gross_amount`, `filled_revenue`
- `value_status`, `nowcast_reason`
- `quality_flag_low_revenue_high_streams`, `rps`

---

## Current Validation Snapshot
- `rows_total = 2000`
- `rows_nowcasted = 322`
- `rows_nowcasted_missing_or_zero = 286`
- `rows_nowcasted_quality_flag = 36`
- `hard_invalid_rows = 0`
- `severe_suspicious_rows = 0`

Status: stable and handoff-ready.