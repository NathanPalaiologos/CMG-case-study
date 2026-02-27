# Forecasting Methodology Log

## Scope
This log tracks assumptions, implementation choices, and latest findings for:
- `notebook/02_case_study_fct.ipynb`
- `notebook/forecast_utils.py`
- `notebook/model_utils.py`
- `notebook/test_utils.py`

It should be updated after each meaningful notebook/model change.

---

## Current Assumptions
- Target: `total_gross_amount`.
- Revenue must be non-negative.
- Rows with `is_na=1` or `is_zero=1` are prediction targets.
- Rows with very low positive revenue under high streams are treated as quality-correction targets.
- Excluded scoring months (default): `2025-09`, `2025-10`.
- Stream volume is the dominant explanatory signal.

---

## Reusable Architecture (Current)

### Utility module
Reusable helper functions are now externalized in:
- `notebook/forecast_utils.py`
- `notebook/model_utils.py`
- `notebook/test_utils.py`

Key reusable blocks:
- time feature engineering
- lag feature engineering
- evaluation (`WMAPE`, `MAE`)
- baseline lookup/apply functions (Group Average, EPSR)
- distribution-aware backtesting
- model-specific fit/predict utilities
- stress-test wrappers with explicit arguments (no global variable dependency)

### Notebook flow
1. Data load and quick profiling.
2. Baselines:
   - Group Average baseline (pure per-group average)
   - Naive baseline (lag-1 within cohort)
3. EPSR benchmark setup (reference only).
4. Model comparison:
   - XGBoost / LightGBM / KNN
   - Lag-Aware Hierarchical Nowcast
5. Distribution-aware stress test (end of notebook) for all models.
6. Final-rule selection section (Lag-Aware vs EPSR benchmark).
7. Final case-study deliverable table (simplified imputed dataframe + RPS + quality flags).
8. Presentation-ready revenue and RPS trend charts across cohort levels.
9. Final output audit block for actual vs nowcasted abnormality checks.

### Why this architecture is stakeholder-safe
- Utility modules keep rules explicit and auditable.
- The same scoring protocol is reused for every method, so comparisons are fair.
- Stress testing is integrated into the workflow, so we evaluate risk, not only average accuracy.

---

## Mathematical Formulation

### Group Average baseline
\[
\hat{y}_{g,t}^{\text{avg}} = \frac{1}{|\mathcal{T}_g|} \sum_{\tau \in \mathcal{T}_g} y_{g,\tau}
\]

### EPSR main strategy
\[
\text{EPSR}_g = \frac{\sum_{\tau \in \mathcal{T}_g} y_{g,\tau}}{\sum_{\tau \in \mathcal{T}_g} s_{g,\tau}},
\quad
\hat{y}_{g,t}^{\text{epsr}} = s_{g,t}\cdot\text{EPSR}_g
\]

### Main metrics
\[
\text{WMAPE} = \frac{\sum_i |y_i-\hat{y}_i|}{\sum_i |y_i|},
\quad
\text{MAE} = \frac{1}{n}\sum_i |y_i-\hat{y}_i|
\]

### Refined tree modeling transform
\[
z = \log(1+y),
\quad
\hat{y} = \exp(\hat{z}) - 1,
\quad
\hat{y} \leftarrow \max(0,\hat{y})
\]

---

## Current Model Settings (Humanized + Controlled)

### Group Average Baseline
- Uses historical average at `(BU, Territory, DSP)` level.
- Used as simple reference baseline.

### Naive Baseline
- Uses previous observed revenue (lag-1) within `(BU, Territory, DSP)`.
- Falls back only where lag history is unavailable.

### EPSR
- Uses weighted revenue-per-stream estimate.
- Fallback chain: cohort -> territory+dsp -> dsp -> global.

### XGBoost
- Uses calendar + stream + lag-aware features.
- Conservative complexity and stronger regularization.
- Log-target training and non-negative clipping.

### LightGBM
- Same feature framework as XGBoost for fair comparison.
- Conservative leaf/sample settings.
- Log-target training and non-negative clipping.

### KNN
- Same encoded feature space.
- Standardized features before distance model.
- Distance weighting and non-negative clipping.

### Lag-Aware Hierarchical Nowcast
- Uses lag EPSR signal and EPSR anchor blend.
- Applies simple `(month, dsp)` reconciliation scale.
- Enforces non-negative, finite outputs (NaN-safe postprocessing).
- Is now applied to both missing/zero rows and strict quality-flag rows.

### Strict quality-flag rule (automatic)
Flag a row when all are true:
- `is_na == 0`
- `is_zero == 0`
- `0 < total_gross_amount <= 99`
- revenue-per-stream is materially below local cohort baseline (`territory + dsp`, fallback to `dsp/global`)

Purpose: catch suspicious single/two-digit revenues using local comparison, without treating high streams alone as an anomaly.

---

## Latest Findings Snapshot

### Included-month validation (excluding Sep/Oct)
- Group Average Baseline: MAE `6584.16`, WMAPE `0.1112`, Rows `1576`
- Naive Baseline Rule: MAE `5571.95`, WMAPE `0.0927`, Rows `1378`
- EPSR: MAE `4880.51`, WMAPE `0.0824`, Rows `1576`
- XGBoost: MAE `10055.48`, WMAPE `0.1558`, Rows `190`
- LightGBM: MAE `12255.83`, WMAPE `0.1899`, Rows `190`
- KNN: MAE `16248.80`, WMAPE `0.2518`, Rows `190`
- Lag-Aware Hierarchical Nowcast: MAE `5456.68`, WMAPE `0.0846`, Rows `190` (post-fix holdout)

### Distribution-aware stress test (30 repeats)
- EPSR mean WMAPE: `0.1020`
- Group Average Baseline mean WMAPE: `0.1299`
- LightGBM mean WMAPE: `0.1666`
- XGBoost mean WMAPE: `0.1761`
- Naive Baseline Rule mean WMAPE: `0.1871`
- KNN mean WMAPE: `0.2738`
- Lag-Aware Hierarchical Nowcast mean WMAPE: `0.1002` (notebook rerun, p50 `0.0946`, p90 `0.1450`)

Conclusion (updated): Lag-Aware Hierarchical Nowcast is selected as the final nowcasting rule. EPSR remains the simple reference and fallback.

### Distribution-aware test explanation (for non-technical readers)
This test answers one business question:
**"If future missing rows follow the same month pattern we see today, which method is most reliable?"**

Step-by-step process:
1. **Profile real missingness**
   - We measure the month-by-month share of rows that are missing or zero in the real target set.
2. **Create realistic holdouts from known rows**
   - We temporarily hide known rows using the same month distribution as the real missing set.
3. **Train and score like production**
   - Each model trains on the remaining known rows and predicts the hidden rows.
4. **Repeat 30 times**
   - We repeat this process to reduce random luck from any single split.
5. **Read both level and risk**
   - `mean` shows expected error level.
   - `p50` shows typical performance.
   - `p90` shows downside risk in harder repeats.

Why this is decision-grade for stakeholders:
- The test mirrors operational missingness rather than using unrealistic random splits.
- It separates "good on average" from "stable under pressure."
- It supports risk-aware planning decisions, not just technical model scoring.

---

## Final Deliverable Dataset

The notebook now produces and saves:
- `data/final_imputed_revenue_with_rps.csv`

Key output columns:
- `total_gross_amount` (reported raw value)
- `filled_revenue` (final value after imputation/nowcasting)
- `value_status` (`actual` or `nowcasted`)
- `nowcast_reason` (`missing_or_zero_revenue`, `quality_flag_low_revenue_high_streams`, `reported_actual`)
- `quality_flag_low_revenue_high_streams` (binary)
- `rps` (revenue per million streams)

Current row counts after strict quality correction:
- total rows: `2000`
- nowcasted rows: `322`
   - missing/zero nowcasted: `286`
   - quality-flag nowcasted: `36`
- actual rows: `1678`

---

## Next-Step Interpretation: Lag-Aware Hierarchical Nowcasting

The path forward is clear but staged:

1. **Lag-aware model first**
   - Build robust bottom-level forecasts with lag features.
2. **Hierarchy second**
   - Reconcile forecasts to enforce coherence across BU/Territory/DSP totals.
3. **Promotion gate**
   - Replace EPSR only if lag-aware hierarchical model beats EPSR on mean and p90 WMAPE under the same stress-test protocol.

This keeps the project practical: maintain a strong explainable strategy now, and promote complexity only when it is measurably superior.

---

## Visualization Add-on (Presentation Readiness)

The notebook now includes clear trend visuals for stakeholder communication:
- Revenue trend split by `value_status` to clearly expose imputed/nowcasted contribution.
- Revenue trend by business unit for management-level review.
- RPS trend by top DSP and top territory cohorts for operational insight.

These visual blocks are intentionally simple and human-readable, with explicit labels and monthly axes.

The NA/Zero distribution panel now includes strict quality-flag counts, so suspicious low-revenue/high-stream concentration is visible by BU, territory, DSP, and month.

---

## Final Output Audit (Status)

The notebook now runs a final audit on `final_deliverable_df` before handoff:
- hard validity checks (`filled_revenue >= 0`, streams > 0, valid RPS)
- robust outlier summary by `value_status` (`actual` vs `nowcasted`)
- suspicious row export to `data/final_output_audit_suspicious_rows.csv`

Latest run:
- hard-invalid rows: `0`
- severe suspicious rows: `0`

---

## Reproducibility Migration Task (Created)

Utility migration tasks for maintainable workflow:
Completed:
1. Created `utils/` package.
2. Migrated `forecast_utils.py`, `model_utils.py`, `test_utils.py` into `utils/`.
3. Updated notebook imports to load from `utils` package directly.
4. Re-ran notebook sections and validated outputs.

---

## Change Log (Latest Update)
- Migrated utility imports to `utils/` package and validated notebook execution.
- Replaced strict stream-threshold-only signal with cohort-aware low-revenue rule using local EPSR comparison.
- Updated NA/Zero plots to include cohort-aware quality-flag counts.
- Re-ran final deliverable and audit (`hard-invalid = 0`, `severe suspicious = 0`).
- Tightened quality-flag rule to include single/two-digit revenues under high-stream conditions.
- Automated quality-flag generation with shared notebook helper function.
- Included quality-flag counts in NA/Zero distribution visualizations.
- Extended final deliverable nowcasting to quality-flag rows.
- Added final output audit block and suspicious-row export for handoff assurance.
- Created utility-folder migration task list for reproducible workflow transition.
- Reorganized notebook narrative to separate final-rule decisioning after stress testing.
- Selected `Lag-Aware Hierarchical Nowcast` as final nowcasting rule and kept EPSR as benchmark.
- Standardized model names to simple labels (`EPSR`, `XGBoost`, `LightGBM`, `KNN`) for cleaner business communication.
- Simplified final deliverable schema to remove redundant columns while preserving auditability.
- Added presentation-ready revenue and RPS trend visualizations at multiple cohort levels.
- Added more humanized comments in notebook code blocks for readability.
- Fixed a root-cause index-alignment bug in `fit_predict_lag_hier_nowcast` that caused anchor/prediction misalignment and zeroed outputs.
- Refined lag-aware hierarchical nowcast with safer rate bounds, lag-aware blending, jump guards, and bounded reconciliation.
- Added function-level explanatory docstrings/comments across all helper functions.
- Fixed helper conflicts by removing hidden globals in `test_utils.py` and using explicit function arguments.
- Added final notebook section that builds full imputed dataframe, labels forecasted vs actual rows, computes RPS, and writes CSV output.
- Added import reloads in notebook so utility edits are applied without kernel restart.
- Added NaN-safe guards to `fit_predict_lag_hier_nowcast` in `notebook/model_utils.py`.
- Re-ran baseline/model/stress cells after patch verification.
- Extracted helper functions into `notebook/forecast_utils.py` and imported them in notebook.
- Restructured baselines to:
  - Group Average baseline (reference)
   - Naive baseline
   - EPSR main strategy
- Refined regressors with lag-aware features and controlled complexity.
- Added log-target modeling for tree models and non-negative clipping across all regressors.
- Updated stress test to compare all strategies consistently.
- Rewrote plan/log with explicit mathematical formulation and next-phase interpretation.
