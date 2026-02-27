# Revenue Misreporting Audit Report

## 1) Scope and What Was Reviewed
This audit reviewed:
- Harmonized modeling dataset: `data/merged_data.csv`
- Final output dataset: `data/final_imputed_revenue_lag-aware_hierarchical_nowcast.csv`
- Processing logic in `notebook/01_case_study_eda.ipynb`
- Forecasting and nowcasting implementation in `notebook/02_case_study_fct.ipynb`, `notebook/model_utils.py`, `notebook/forecast_utils.py`, `notebook/test_utils.py`

A suspicious-row extract was generated at:
- `data/suspicious_revenue_rows.csv`

---

## 2) Data Integrity and Suspicious Findings

## 2.1 Basic integrity checks (good)
- No duplicate keys on `(month, business_unit, territory_name, dsp)`.
- No negative revenue rows in `merged_data.csv`.
- No non-positive streams in `merged_data.csv`.

## 2.2 Missingness concentration (high risk for bias)
- `is_na = 188`, `is_zero = 98` (total target rows = 286).
- Missing/zero concentration is highly skewed:
  - `2025-10`: 90% target share
  - `2025-09`: 41% target share

Interpretation: any model trained/evaluated without explicitly handling this skew can overstate reliability.

## 2.3 Revenue patterns that look suspicious
- 14 rows have **tiny positive revenue** (`total_gross_amount <= 1`) with **very large streams** (`> 1,000,000`).
- 29 rows show **sharp month-over-month drops** (`jump_ratio < 0.1`) within the same cohort.
- RPS distribution is very wide:
  - 1st percentile ≈ `0.116`
  - median ≈ `1272.4`
  - 99th percentile ≈ `18916.4`
- DSP-level robust outliers (MAD-based, `|z| > 4.5`) are concentrated in Amazon (`41` rows).

Interpretation: these patterns can be valid for some contracts, but they are strong candidates for misreporting or processing artifacts and should be quarantined from model training unless verified.

Update after stricter rule rollout:
- Flag logic was tightened to include single/two-digit revenues (`<= 99`) using local cohort comparison.
- Total quality-flag rows under current cohort-aware rule: `36`.

---

## 3) Processing Step Review (Potential Risk Points)

## 3.1 Harmonization mapping coverage
- Revenue DSP mapping coverage is complete for the current data snapshot (no unmapped values).
- Merge results are structurally consistent:
  - `both = 1812`
  - `left_only = 188` (streams-only keys)
  - `right_only = 0` (no revenue-only keys)

## 3.2 Suspicious processing risks to monitor
1. **Hard-coded mapping fragility**
   - New DSP labels in future files may silently fall into `Other` unless mapping is updated.
2. **Territory fallback compression**
   - Mapping unknown countries to `All Other Locations` is practical, but can mask regional anomalies.
3. **Aggregation behavior risk**
   - `groupby().sum()` can turn all-null groups into zero in some workflows; this was not observed in the current snapshot, but should be guarded explicitly.

---

## 4) Recommended Misreporting / Outlier Framework

Use a two-table pattern:
- **Raw table**: keep reported values unchanged.
- **Modeling table**: apply anomaly flags and robust adjustments for training only.

## 4.1 Detection rules (combined)
For known rows (`is_na=0`, `is_zero=0`, streams>0), create flags:

1. **Rule A: strict low revenue vs local cohort (implemented)**
   - `0 < total_gross_amount <= 99` and local EPSR ratio is abnormally low vs `territory + dsp` baseline
   - fallback hierarchy: `territory + dsp` -> `dsp` -> global
2. **Rule B: robust cross-sectional RPS outlier**
   - Compute `log1p(rps)` and robust z-score by DSP (or BU+DSP when enough data)
   - Flag when `|robust_z| > 4.5`
3. **Rule C: temporal jump anomaly**
   - Within cohort, flag `jump_ratio < 0.1` or `jump_ratio > 10`
4. **Rule D: global tail guard**
   - Flag RPS outside `[0.5%, 99.5%]` quantiles as soft anomalies

Recommended action by severity:
- `flag_count >= 2`: quarantine from training
- `flag_count = 1`: keep but down-weight or winsorize
- `flag_count = 0`: normal training

## 4.2 Pollution control for modeling
- Fit EPSR and lag-aware components on **cleaned known set** (excluding quarantined rows).
- Keep final reported actuals untouched in deliverable; only training views are adjusted.
- Track `quality_flag` and `quality_reason` columns for auditability.

Implemented behavior in notebook:
- Rows flagged by Rule A are nowcasted with the same final lag-aware model.
- Final output now records `nowcast_reason` and `quality_flag_low_revenue_high_streams`.
- Current composition: 322 nowcasted rows = 286 missing/zero + 36 strict quality-flag rows.

---

## 5) Nowcasting Implementation Review

## 5.1 What is correct
- Core predictor functions return valid outputs (lengths match score rows, no NaN/inf, no negatives).
- Lag-aware nowcast implementation has index-alignment safeguards and bounded adjustments.
- Stress-test method is distribution-aware and repeats 30 times with fixed seed.

## 5.2 Issues to watch (process, not core math)
1. **Comparability caveat in metric table**
   - Baselines/EPSR and model methods are evaluated on different row counts in some notebook sections.
   - Keep comparisons within the same evaluation pool when making ranking claims.
2. **Notebook run-order dependency**
   - Mitigation implemented: EPSR metric creation is now consolidated in one cell.
3. **Deliverable schema consistency**
   - Ensure documentation and exported columns stay synchronized (current simplified output is fine, but docs should match exactly).

---

## 6) Immediate Action Plan
1. Keep strict quality-flag pipeline active in final notebook workflow.
2. Monitor monthly flagged volume by BU/Territory/DSP for drift.
3. Re-tune thresholds if business context changes (contract events, catalog changes).
4. Complete utility migration task to package-based imports for reproducibility.

---

## 7) Final Output Audit Status (Post-Implementation)
Final output audit now runs in notebook and checks both `actual` and `nowcasted` values.

Latest audit result:
- Hard-invalid rows (`filled_revenue < 0`, non-positive streams, invalid RPS): `0`
- Severe suspicious rows exported (`|robust_z_all| > 5.5`): `0`

Conclusion: no evidence of significantly mismatched or abnormal records remains in the current final output under the configured strict quality-gate.

---

## 8) Artifacts Generated by This Audit
- `data/suspicious_revenue_rows.csv`
- `data/final_output_audit_suspicious_rows.csv`
- `misreporting_audit_report.md`
- `workflow_migration_tasks.md`
