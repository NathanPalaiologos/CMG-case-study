# Forecasting Plan for Missing and Zero Revenue (Revised)

## 1) Problem Framing

We need to estimate revenue (`total_gross_amount`) for rows flagged as:
- missing (`is_na = 1`), or
- unreliable zeros (`is_zero = 1`).

Because this is stream-driven music revenue, we treat `total_streams` as the primary signal and combine it with cohort context:
\[
(\text{business\_unit},\ \text{territory\_name},\ \text{dsp},\ \text{month})
\]

In addition to `NA`/`0` revenue targets, we now also treat clearly suspicious low-revenue/high-stream rows as correction candidates.

---

## 2) Evaluation Protocol (Reproducible Across Models)

### Stakeholder interpretation (plain language)
- We are not only asking "which model is accurate on average?"
- We are asking "which model stays reliable when real-world missing patterns change by month?"
- This matters because finance and planning risk comes from unstable months, not just average error.

### Included vs excluded scoring months
Late months are dominated by missing/zero labels, so default scoring excludes:
- `2025-09`, `2025-10`

### Primary metric: WMAPE
\[
\text{WMAPE} = \frac{\sum_i |y_i - \hat{y}_i|}{\sum_i |y_i|}
\]

### Secondary metric: MAE
\[
\text{MAE} = \frac{1}{n}\sum_i |y_i - \hat{y}_i|
\]

### Stress test
Use distribution-aware masking backtest (30 repeats):
1. Build synthetic holdouts from known rows.
2. Match holdout month mix to the real missing/zero month mix.
3. Train on remaining known rows.
4. Evaluate MAE/WMAPE distribution (mean, std, p10, p50, p90).

### How the distribution-aware test works (non-technical)
Think of this as a "fire drill" that mimics where missing values really happen.

1. **Measure real missing pattern**
  - We first count where actual missing/zero rows occur by month.
  - Example idea: if missing rows are concentrated in one month, our test should reflect that concentration.

2. **Create realistic fake-missing rows from known data**
  - We temporarily hide a sample of known rows.
  - The hidden rows are sampled to match that real month pattern.

3. **Train on what remains, predict what was hidden**
  - This simulates production conditions: model sees incomplete data, then imputes missing revenue.

4. **Repeat 30 times**
  - One split can be lucky or unlucky.
  - Repeating produces a distribution of outcomes, not a single-point claim.

5. **Read results as risk, not just average**
  - `mean` tells expected performance.
  - `p90` tells bad-case behavior (tail risk).
  - Lower and tighter values mean more trustworthy planning outputs.

---

## 3) Model Strategy Hierarchy

## Step A — Baseline: Pure Group Average

Per cohort average baseline:
\[
\hat{y}_{g,t}^{\text{avg}} = \frac{1}{|\mathcal{T}_g|}\sum_{\tau \in \mathcal{T}_g} y_{g,\tau}
\]
where \(g=(BU, Territory, DSP)\) and \(\mathcal{T}_g\) are known months for group \(g\).

This is the clean "pure per-group average imputing and nowcasting" baseline.

## Step B — Baseline: Naive Rule

Lag-1 within cohort baseline:
\[
\hat{y}_{g,t}^{\text{naive}} = y_{g,t-1}
\]
with fallback handling when prior value is unavailable.

## Step C — Main Strategy: EPSR

Define effective per-stream rate per cohort:
\[
\text{EPSR}_g = \frac{\sum_{\tau \in \mathcal{T}_g} y_{g,\tau}}{\sum_{\tau \in \mathcal{T}_g} s_{g,\tau}}
\]

Prediction:
\[
\hat{y}_{g,t}^{\text{epsr}} = s_{g,t} \cdot \text{EPSR}_g
\]

Fallback chain for sparse history:
1. `(BU, Territory, DSP)`
2. `(Territory, DSP)`
3. `(DSP)`
4. global

EPSR is kept as the benchmark reference strategy.

## Step D — Refined Regressors (Next Iteration Implemented)

Implemented under the same evaluation protocol:
- XGBoost (refined)
- LightGBM (refined)
- KNN (refined)

Refinements to reduce overfitting and instability:
- Lightweight lag-aware features (`gross_lag_1`, `streams_lag_1`, `epsr_lag_1`)
- conservative tree depth / leaves
- log-target modeling for tree models:
  \[
  z = \log(1+y),\quad \hat{y}=\exp(\hat{z})-1
  \]
- training-only median imputation for lag nulls
- non-negative output clipping:
  \[
  \hat{y} \leftarrow \max(0, \hat{y})
  \]
- KNN scaling before distance calculations

---

## 4) Current Performance Snapshot

From current notebook runs (included scoring months only):
- Group Average Baseline: WMAPE `0.1112`, MAE `6584.16`
- Naive Baseline Rule: WMAPE `0.0927`, MAE `5571.95`
- EPSR Benchmark Rule: WMAPE `0.0824`, MAE `4880.51`
- XGBoost Refined: WMAPE `0.1558`, MAE `10055.48`
- LightGBM Refined: WMAPE `0.1899`, MAE `12255.83`
- KNN Refined: WMAPE `0.2518`, MAE `16248.80`
- Lag-Aware Hierarchical Nowcast (post-fix check): WMAPE `0.0846` (quick holdout rerun)

Stress-test (30 repeats, mean WMAPE):
- EPSR Benchmark Rule: `0.1020`
- Group Average Baseline: `0.1299`
- LightGBM Refined: `0.1666`
- XGBoost Refined: `0.1761`
- Naive Baseline Rule: `0.1871`
- KNN Refined: `0.2738`
- Lag-Aware Hierarchical Nowcast (30-repeat notebook rerun): `0.1002` (p50 `0.0946`, p90 `0.1450`)

Decision (updated): Lag-Aware Hierarchical Nowcast is selected as the final nowcasting rule. EPSR remains the benchmark and fallback reference.

Business interpretation:
- EPSR remains the easiest rule to explain to business partners.
- Lag-aware nowcasting delivers slightly stronger stress-test performance while keeping bounded, interpretable behavior.
- Recommendation is to use lag-aware in production outputs and keep EPSR as a governance fallback.

### Methodology justification: pros and cons of explored rules
- **Group Average**: strong sanity baseline, but no temporal adaptation.
- **Naive Lag-1**: captures short memory, but unstable under missing/noisy prior periods.
- **EPSR**: excellent interpretability and governance benchmark, but less reactive to local month-level shifts.
- **XGBoost / LightGBM / KNN**: flexible learning capacity, but weaker stability and higher governance complexity for this dataset.
- **Lag-Aware Hierarchical Nowcast**: best balance of accuracy, stress stability, and bounded behavior; selected for final use.

---

## 5) How to Proceed to Lag-Aware Hierarchical Nowcasting (Next Phase)

This is now a structured Phase 2, not immediate replacement.

### 5.1 Lag-aware base forecasts
Build bottom-level base forecasts with lag features:
\[
\hat{y}_{g,t} = f\big(s_{g,t},\ y_{g,t-1},\ s_{g,t-1},\ \text{EPSR}_{g,t-1},\ \text{calendar}_t\big)
\]

### 5.2 Hierarchical coherence
Let \(\mathbf{y}_t\) be bottom series and \(\mathbf{S}\) be aggregation matrix.
Require coherent forecasts:
\[
\hat{\mathbf{y}}_t^{\text{coherent}} = \mathbf{S}\hat{\mathbf{b}}_t
\]
where \(\hat{\mathbf{b}}_t\) are reconciled bottom-level forecasts.

Use MinT-style reconciliation so total DSP / territory / BU rollups equal the sum of granular forecasts.

### 5.3 Promotion rule to move beyond EPSR
Promote lag-aware hierarchical model only if:
1. lower mean WMAPE than EPSR,
2. lower or similar p90 WMAPE,
3. month-level stability without extreme tail failures,
4. reconciliation consistency checks pass.

Current status vs promotion gate:
- Mean WMAPE: passes in notebook rerun.
- p90 WMAPE: comparable and stable vs benchmark.
- Stability: no NaN/inf collapse observed after alignment fix.
- Reconciliation checks: bounded and coherent at `(month, dsp)` layer.

This gives a clear, controlled path from robust simple strategy (EPSR) to production-grade hierarchical nowcasting.

---

## 6) Case-Study Deliverable Output (Implemented)

The notebook now builds a full imputed dataframe with explicit labels and saves it to:
- `data/imputed_revenue_lag-aware_hierarchical_nowcast.csv`

### Simplified deliverable fields
- `month`, `business_unit`, `territory_name`, `dsp`
- `total_streams`
- `total_gross_amount` (original reported value)
- `filled_revenue` (single final revenue field)
- `nowcast_lower_90`, `nowcast_upper_90` (uncertainty bounds for nowcasted rows)
- `value_status`: `actual` vs `nowcasted`
- `nowcast_reason`: `missing_or_zero_revenue` or `quality_flag_low_revenue_high_streams` or `reported_actual`
- `quality_flag_low_revenue_high_streams` (binary)
- `rps`

### Revenue per million streams (RPS)
\[
	ext{RPS} = \frac{\text{filled\_revenue}}{\text{total\_streams}/1{,}000{,}000}
\]

RPS is set to `NaN` when streams are zero/non-positive.

### Cohort-aware automatic quality-flag rule (active)
Rows are auto-flagged and nowcasted when all conditions hold:
1. `is_na == 0`
2. `is_zero == 0`
3. `0 < total_gross_amount <= 99` (single/two-digit revenue)
4. Revenue-per-stream is abnormally low **vs local cohort baseline** (`territory + dsp`, with `dsp/global` fallback)

This avoids flagging based on stream magnitude alone and is better aligned with regional pricing differences.

Current nowcast composition after strict rule:
- total nowcasted rows: `322`
- missing/zero nowcasted rows: `286`
- quality-flag nowcasted rows: `36`

### Prediction interval construction and interpretation
Intervals are created from calibration residuals on clean known rows:
1. compute relative absolute error,
2. estimate 90th percentile error (group-level with global fallback),
3. apply symmetric bounds around each nowcast.

Interpretation for stakeholders:
- bounds indicate uncertainty range around each estimated value,
- wider ranges imply higher uncertainty,
- bounds are only populated for nowcasted rows.

---

## 7) Presentation Visuals (Implemented)

The notebook now includes presentation-ready trend charts with clear labels:
- Monthly revenue: `reported_actual` vs `imputed_nowcasted`
- Monthly revenue trend by business unit
- Monthly RPS trend by top DSP cohorts
- Monthly RPS trend by top territory cohorts

These views are designed for stakeholder review and make model impact visible at multiple cohort levels.

The NA/Zero distribution visuals now also include quality-flag counts by:
- business unit
- territory
- DSP
- month

---

## 8) Final Output Audit (Implemented)

A final audit section is run in the notebook before handoff:
- checks hard validity (`filled_revenue >= 0`, streams > 0, valid RPS)
- checks robust RPS outliers separately for `actual` and `nowcasted`
- exports suspicious rows if detected

Latest run status:
- hard-invalid rows: `0`
- severe suspicious rows exported: `0`

---

## 9) Reproducible Workflow Task (Created)
Migration status:
- Utilities are now migrated into `utils/` package.
- Notebook imports now read directly from `utils` modules.

Remaining optional hardening steps:
1. Add CI smoke test to run notebook key cells non-interactively.
2. Add a small schema check before export.
