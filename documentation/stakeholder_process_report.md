# CMG Revenue Visibility Project — Stakeholder Summary

## Executive Summary
This project delivers a single monthly revenue view for Finance through Oct 2025 by combining:
- **Reported Revenue** (`total_gross_amount`) when statements are available, and
- **Estimated Revenue** (`filled_revenue`) when reported values are missing, zero, or clearly anomalous.

Final production method: **Lag-Aware Hierarchical Nowcast**.
Benchmark/fallback method: **EPSR**.

---

## Slide-Ready Model Summary
- **Baseline 1: Group Average**
  $$\hat{y}^{\text{avg}}_{g,t} = \frac{1}{|\mathcal{T}_g|}\sum_{\tau \in \mathcal{T}_g} y_{g,\tau}$$
- **Baseline 2: Naive Lag-1 Forecast**
  $$\hat{y}^{\text{naive}}_{g,t} = y_{g,t-1}$$
- **Strong Benchmark: EPSR (Expected Per Stream Revenue)**
  $$\text{EPSR}_g = \frac{\sum_{\tau \in \mathcal{T}_g} y_{g,\tau}}{\sum_{\tau \in \mathcal{T}_g} s_{g,\tau}},\quad \hat{y}^{\text{epsr}}_{g,t} = s_{g,t}\cdot\text{EPSR}_g$$
- **Model-Based Methods:** XGBoost, LightGBM, KNN
- **Production Rule:** Lag-Aware Hierarchical Nowcast
  $$\hat{y}^{\text{lag-hier}}_{g,t}=\max\Big(0,\;\rho\,\hat{y}^{\text{epsr}}_{g,t} + (1-\rho)\,y_{g,t-1}\cdot\frac{s_{g,t}}{\max(s_{g,t-1},\epsilon)}\Big),\quad 0\le\rho\le1$$

This format is intentionally compact for direct slide usage.

---

## Terminology (Standard)
- **Reported Revenue**: raw DSP statement value (`total_gross_amount`).
- **Estimated Revenue**: model-produced value (`filled_revenue`) for target rows.
- **Estimated Row**: `value_status = nowcasted`.
- **Actual Row**: `value_status = actual`.
- **Reason for Estimation** (`nowcast_reason`):
  - `missing_or_zero_revenue`
  - `quality_flag_low_revenue_high_streams`
  - `reported_actual`

Using these terms keeps business and technical discussions aligned.

---

## Problem We Solved
Finance needs a complete revenue picture by month, DSP, business unit, and territory, but DSP revenue statements arrive with delays.

Main data challenge: Revenue and Streams sources are not dimension-aligned out of the box.

---

## Process in Plain Language
1. **Harmonize dimensions**
   - Standardize DSP labels to major groups.
   - Align stream geography to revenue territory logic.
   - Merge both sources into one consistent table.

2. **Identify rows that need estimation**
   - Missing reported revenue.
   - Zero reported revenue likely caused by statement timing.
   - Strict quality-correction rows (very low revenue with locally abnormal revenue-per-stream behavior).

3. **Estimate only target rows**
   - Keep most reported values untouched.
   - Use lag-aware stream-informed nowcasting for target rows.
   - Write transparent status/reason labels in output.

4. **Run final quality audit**
   - Hard-validity checks.
   - Robust outlier checks on Revenue per Million Streams (RPS).

---

## Why This Final Model Was Chosen
### Stakeholder perspective
- It is **reliable under real missingness patterns**, not just on average.
- It uses streams (the strongest complete signal) while preserving business interpretability.
- It retains EPSR as a clear benchmark for governance and challenge.
- It passed final quality gates with no severe issues.

### Technical perspective (in plain language)
- Selection is based on repeated distribution-aware tests (30 repeats).
- Decision uses both average performance and downside risk (tail behavior).
- Final method remained stable and valid in full-run checks.

### How Lag-Aware Hierarchical Nowcasting Works
1. **Start with a stream-based anchor (EPSR)**
  - Build a robust expected-revenue anchor from stream behavior.
2. **Add lag information**
  - Use prior-period revenue/stream context to adjust toward recent cohort trend.
3. **Apply bounded reconciliation**
  - Keep cohort-level outputs coherent and avoid unrealistic jumps.
4. **Enforce value safety**
  - Clip to non-negative, finite values before final output.

Mathematically (business-friendly form):
$$
\hat{y}^{\text{anchor}}_{g,t}=s_{g,t}\cdot \text{EPSR}_g
$$
$$
\hat{y}^{\text{lag}}_{g,t}=y_{g,t-1}\cdot\frac{s_{g,t}}{\max(s_{g,t-1},\epsilon)}
$$
$$
\hat{y}^{\text{raw}}_{g,t}=\rho\,\hat{y}^{\text{anchor}}_{g,t}+(1-\rho)\,\hat{y}^{\text{lag}}_{g,t}
$$
$$
\hat{y}^{\text{final}}_{g,t}=\max(0,\;\lambda_{t,d}\cdot\hat{y}^{\text{raw}}_{g,t})
$$

Where:
- $g$ is cohort (BU, territory, DSP),
- $\rho$ controls anchor-vs-lag blending,
- $\lambda_{t,d}$ is bounded reconciliation scale at month–DSP level,
- $\epsilon$ avoids divide-by-zero on lag streams.

In simple terms:
- Anchor gives structural stability,
- lag gives local responsiveness,
- reconciliation gives roll-up consistency.

---

## Row Number Explanation
- **Total rows: 2000**
  - Complete harmonized key space for month x BU x territory x DSP.
- **Actual rows: 1678**
  - Rows where reported revenue was retained.
- **Estimated rows: 322**
  - Rows where `filled_revenue` replaced reported value.
  - Breakdown:
    - **286** missing/zero revenue rows
    - **36** strict quality-correction rows

This means most values remain reported actuals, and estimation is applied only where justified.

---

## Model Assumptions
- Streams are sufficiently complete to anchor monthly estimation.
- Revenue is non-negative.
- Missing/zero rows are valid estimation targets.
- Local cohort context (`territory + dsp`, fallback `dsp/global`) is required for anomaly detection.
- Stability under realistic missingness matters more than one split’s average score.

---

## Selection Criteria and Test Process
### Selection criteria
- Primary: stable WMAPE under repeated distribution-aware tests.
- Secondary: MAE and operational interpretability.
- Risk lens: tail behavior (p90), not only mean.
- Governance lens: clear labels + reproducible pipeline.

### Test process (non-technical)
1. Measure real month pattern of missing/zero rows.
2. Hide known rows using that same month pattern.
3. Train each method and predict hidden rows.
4. Repeat 30 times.
5. Compare expected performance and bad-case behavior.

---

## Prediction Intervals (How Uncertainty Is Shown)
To reflect remaining uncertainty in estimated rows, we now publish a 90% interval around each nowcast:
- `nowcast_lower_90`
- `nowcast_upper_90`

How it is built (plain language):
1. Use clean historical rows where both reported revenue and model predictions are available.
2. Measure relative error size for those rows.
3. Take the 90th-percentile error size (with local cohort learning and global fallback).
4. Apply that error band around each nowcasted value.

How to interpret for stakeholders:
- The interval is a **risk range**, not a guarantee.
- Wider intervals mean higher uncertainty for that cohort/context.
- Actual rows keep reported values; intervals are populated only for estimated rows.

---

## Why We Picked This Rule: Pros and Cons by Method
### Group Average Baseline
- **Pros:** very simple; easy to explain.
- **Cons:** ignores month dynamics and lag behavior.

### Naive Lag-1 Baseline
- **Pros:** captures short-term continuity.
- **Cons:** fragile when prior month is noisy/missing.

### EPSR Benchmark
- **Pros:** highly interpretable stream-to-revenue logic; strong benchmark.
- **Cons:** less adaptive to local month-to-month shifts.

### XGBoost / LightGBM / KNN
- **Pros:** flexible nonlinear fitting.
- **Cons:** weaker stability in this dataset under distribution-aware stress testing; lower governance clarity.

### Lag-Aware Hierarchical Nowcast (Selected)
- **Pros:** combines stream anchor + lag dynamics + bounded reconciliation; best stability profile for this use case.
- **Cons:** slightly more complex than EPSR.

Decision rationale:
- We selected Lag-Aware Hierarchical Nowcast for production estimates and retained EPSR as benchmark/fallback for governance.

---

## Final Output Delivered
Main output files:
- `data/imputed_revenue_lag-aware_hierarchical_nowcast.csv`
- `data/imputed_revenue_lag-aware_hierarchical_nowcast.xlsx`
- `data/output_audit_suspicious_rows.csv`
- `data/pipeline_run_summary.json`

Key business fields:
- `total_gross_amount` (Reported Revenue)
- `filled_revenue` (Estimated/Final Revenue)
- `value_status`
- `nowcast_reason`
- `rps`

---

## Latest Verified Snapshot
- Total rows: **2000**
- Actual rows: **1678**
- Estimated rows: **322**
  - Missing/zero: **286**
  - Quality-correction: **36**
- Hard-invalid rows: **0**
- Severe suspicious rows: **0**

---

## Reproducible Workflow
Canonical script:
- `sql_workflow/build_finance_revenue_view.py`

Workflow steps:
1) SQL harmonization
2) quality flagging
3) estimation (nowcasting)
4) output audit
5) export

---

## Business Outcome
Finance now has a complete, auditable monthly revenue view through Oct 2025 with a clear distinction between reported actuals and model-estimated values.