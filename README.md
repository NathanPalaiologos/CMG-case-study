# CMG Case Study

Finance-focused revenue visibility project that harmonizes Revenue and Streams, nowcasts missing/incomplete revenue, and exports an auditable Finance-ready dataset.

## Core Assets

- Notebooks (analysis + modeling):
	- `notebook/01_case_study_eda.ipynb`
	- `notebook/02_case_study_fct.ipynb`
	- `notebook/03_linear_ridge_nowcast.ipynb`
- Reusable utilities:
	- `utils/forecast_utils.py`
	- `utils/model_utils.py`
	- `utils/test_utils.py`
- SQL harmonization spec:
	- `sql_pipeline/01_data_harmonization.sql`
- Reproducible production workflow:
	- `sql_workflow/build_finance_revenue_view.py`

## Run Workflow (from project root)

```bash
python sql_workflow/build_finance_revenue_view.py --input-merged data/merged_data.csv
```

Rebuild merged data from source Excel:
```bash
python sql_workflow/build_finance_revenue_view.py --refresh-merged --input-xlsx "data/Case Study Dataset.xlsx"
```

## Main Outputs

- `data/imputed_revenue_lag-aware_hierarchical_nowcast.csv`
- `data/imputed_revenue_lag-aware_hierarchical_nowcast.xlsx`
- `data/output_audit_suspicious_rows.csv`
- `data/output_audit_dsp_month_reliability.csv`
- `data/pipeline_run_summary.json`