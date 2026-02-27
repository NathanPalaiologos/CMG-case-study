# CMG Revenue Visibility Project — Stakeholder Summary

## Executive Summary
To improve monthly accrual visibility for Finance, we built a unified revenue view that combines:
- **actual reported revenue** where statements are available, and
- **model-based nowcasts** where revenue is missing or clearly incomplete.

This produces a single month-level view through Oct 2025 with explicit labels for what is actual versus estimated.

## What Problem We Solved
Finance needed a complete monthly revenue picture by:
- DSP,
- business unit,
- territory,
while DSP payouts arrive with reporting delays.

The main challenge was that Revenue and Streams data were not aligned at the same dimension levels.

## Approach in Plain Language
1. **Standardize dimensions**
   - Mapped detailed DSP labels to major DSP groups.
   - Aligned stream territories to revenue territories.
   - Merged both sources into one consistent table.

2. **Identify rows needing estimation**
   - Missing revenue rows.
   - Zero-revenue rows likely incomplete for finance timing.
   - A small set of suspicious low-revenue rows flagged by local cohort behavior.

3. **Estimate only where needed**
   - Kept actual reported values untouched.
   - Applied a lag-aware, stream-informed nowcast for target rows.
   - Added reason labels for auditability.

4. **Quality-audit final output**
   - Checked for hard-invalid records.
   - Ran robust anomaly checks on final Revenue per Million Streams (RPS).

## Final Output Delivered
Primary table:
- `data/final_imputed_revenue_lag-aware_hierarchical_nowcast.csv`
- `data/final_imputed_revenue_lag-aware_hierarchical_nowcast.xlsx`

Key fields for Finance:
- `total_gross_amount` (reported value)
- `filled_revenue` (final value used for analysis)
- `value_status` (`actual` or `nowcasted`)
- `nowcast_reason`
- `rps`

## Latest Run Snapshot
- Total rows: **2000**
- Actual rows: **1678**
- Nowcasted rows: **322**
  - Missing/zero: **286**
  - Quality-flag corrections: **36**
- Final audit hard-invalid rows: **0**
- Final audit severe suspicious rows: **0**

## Controls and Governance
- Transparent rule-based harmonization (SQL + Python equivalents).
- Reusable utility modules and reproducible script workflow.
- Explicit actual-vs-nowcast indicators in deliverable output.
- Retained benchmark method (EPSR) for governance comparison.

## Reproducible Workflow
The full verified flow is packaged in:
- `sql_workflow/build_finance_revenue_view.py`

This script runs:
1) harmonization, 2) quality-flagging, 3) nowcasting, 4) output audit, 5) export.

## Business Outcome
Finance now has a complete and auditable monthly revenue dataset through Oct 2025 that is suitable for accrual visibility, with clear distinction between reported actuals and estimated values.
