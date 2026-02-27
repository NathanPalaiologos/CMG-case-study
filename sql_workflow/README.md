# Reproducible Workflow

## Purpose
This folder contains the production-style workflow that turns harmonized source data into a Finance-ready revenue view.

## Primary Script
- `build_finance_revenue_view.py`

## What It Does
1. Harmonizes Revenue + Streams with SQL (DuckDB CTE pipeline)
2. Applies missing/zero and cohort-aware quality flags
3. Runs lag-aware hierarchical nowcast for targeted rows
4. Runs final output audit checks
5. Exports CSV/XLSX and run summary JSON

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

## Why SQL for Cleaning
- Uses SQL CTEs mirroring `sql_pipeline/01_data_harmonization.sql`.
- Easier to audit transformation rules.
- DuckDB executes vectorized SQL efficiently for larger tabular datasets.

## Outputs
- `data/merged_data.csv` (when `--refresh-merged` is used)
- `data/final_imputed_revenue_lag-aware_hierarchical_nowcast.csv`
- `data/final_imputed_revenue_lag-aware_hierarchical_nowcast.xlsx`
- `data/final_output_audit_suspicious_rows.csv`
- `data/pipeline_run_summary.json`
