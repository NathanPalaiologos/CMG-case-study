# Reproducible Workflow

## Purpose
This folder contains the production-style workflow that turns harmonized source data into a Finance-ready revenue view.

## Primary Script
- `build_finance_revenue_view.py`

## What It Does
1. Harmonizes Revenue and Streams with SQL (DuckDB CTE pipeline)
2. Applies missing/zero and cohort-aware quality flags
3. Runs lag-aware hierarchical nowcast for targeted rows
4. Runs final output audit checks
5. Exports `.csv` / `.xlsx` and run summary `.json`

Quality-flag rule:
- Fixed-effects residual flagging on log-RPS (`territory_name` + `dsp`)
- `fe_threshold_quantile = 0.05`, `min_group_rows = 24`, `low_revenue_threshold = None`

DSP-month reliability flagging:
- `risk_flag = caution` when `nowcast_rate >= 0.80` or `uncertainty_width_pct >= 35`

## Prerequisites
- Run from the **project root**: `CMG Case Study/`
- Python environment with:
	- `pandas`, `numpy`, `openpyxl`
	- `duckdb` (required for SQL-based harmonization)

Install example:

```bash
pip install duckdb
```

## Usage

### A) Fast run from existing merged data
```bash
python sql_workflow/build_finance_revenue_view.py --input-merged data/merged_data.csv
```

### B) Rebuild merged data from source Excel using SQL logic
```bash
python sql_workflow/build_finance_revenue_view.py --refresh-merged --input-xlsx "data/Case Study Dataset.xlsx"
```

### C) Large-data friendly path from CSV sources
```bash
python sql_workflow/build_finance_revenue_view.py --refresh-merged --revenue-csv data/revenue.csv --streams-csv data/streams.csv
```

## Outputs
- `data/merged_data.csv` (when `--refresh-merged` is used)
- `data/Nathan_Zhang_CMG_Revenue_Forecast_Oct2025.csv`
- `data/Nathan_Zhang_CMG_Revenue_Forecast_Oct2025.xlsx`
- `data/output_audit_suspicious_rows.csv`
- `data/output_audit_dsp_month_reliability.csv`
- `data/pipeline_run_summary.json`
