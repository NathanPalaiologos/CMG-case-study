# CMG Revenue Visibility Project — Stakeholder Summary

## Executive Summary
This project delivers a single monthly revenue view for Finance through Oct 2025 by combining:
- **Reported Revenue** (`total_gross_amount`) when statements are available, and
- **Estimated Revenue** (`filled_revenue`) when reported values are missing, zero, or clearly anomalous.

Final production method: **Lag-Aware Hierarchical Nowcast**.
Benchmark/fallback method: **EPSR**.

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