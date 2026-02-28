from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import duckdb

# Allow direct execution from project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from utils.forecast_utils import add_lag_features, add_quality_flags, build_relative_prediction_intervals
from utils.model_utils import fit_predict_lag_hier_nowcast


TARGET_COL = "total_gross_amount"
STREAM_COL = "total_streams"
GROUP_COLS = ["business_unit", "territory_name", "dsp"]
MIN_FLAG_GROUP_ROWS = 24
FE_THRESHOLD_QUANTILE = 0.03


def harmonize_with_sql(
    revenue_csv: Path | None,
    streams_csv: Path | None,
    source_xlsx: Path | None,
) -> pd.DataFrame:
    """
    Harmonize source datasets with SQL CTEs.

    Why SQL here:
    - keeps transformation logic auditable and close to the provided SQL script
    - scales better than row-wise Python for large tabular processing
    """
    con = duckdb.connect(database=":memory:")

    # Input mode A: large-data friendly CSV path.
    if revenue_csv and streams_csv:
        con.execute(
            """
            CREATE OR REPLACE VIEW revenue AS
            SELECT * FROM read_csv_auto(?, header=true);
            """,
            [str(revenue_csv)],
        )
        con.execute(
            """
            CREATE OR REPLACE VIEW streams AS
            SELECT * FROM read_csv_auto(?, header=true);
            """,
            [str(streams_csv)],
        )
    # Input mode B: case-study Excel path (load once, register as SQL views).
    elif source_xlsx:
        revenue_df = pd.read_excel(source_xlsx, sheet_name="Revenue")
        streams_df = pd.read_excel(source_xlsx, sheet_name="Streams")
        con.register("revenue", revenue_df)
        con.register("streams", streams_df)
    else:
        raise ValueError("Provide either --input-xlsx or both --revenue-csv and --streams-csv")
    
    # Flow: standardize labels -> aggregate -> align territories -> full join -> add flags.
    query = """
    WITH revenue_standardized AS (
        SELECT
            month,
            business_unit,
            territory_name,
            CASE
                WHEN dsp IN (
                    'YouTube Premium Individual monthly', 'YouTube - Ads', 'YouTube - Premium',
                    'YouTube Ad Revenue', 'YouTube Premium Family monthly', 'YouTube - Audio Tier',
                    'YouTube', 'YouTube Premium Student monthly', 'YouTube Music Family monthly',
                    'YouTube Music Student monthly', 'YouTube Music', 'YouTube Music Individual monthly',
                    'YouTube Premium Lite Individual monthly', 'Youtube Licensing', 'YouTube Premium',
                    'YouTube - Other'
                ) THEN 'YouTube'
                WHEN dsp IN ('Tiktok') THEN 'TikTok'
                WHEN dsp IN ('Spotify') THEN 'Spotify'
                WHEN dsp IN ('Amazon', 'Amazon Prime', 'Amazon Unlimited', 'Amazon Ad Supported', 'Amazon Cloud') THEN 'Amazon'
                WHEN dsp IN ('Apple/iTunes', 'Apple Music Dj Mixes', 'Apple Music', 'Apple Inc.') THEN 'Apple'
                ELSE 'Other'
            END AS dsp,
            total_gross_amount
        FROM revenue
    ),
    revenue_aggregated AS (
        SELECT
            month,
            business_unit,
            territory_name,
            dsp,
            SUM(total_gross_amount) AS total_gross_amount
        FROM revenue_standardized
        GROUP BY month, business_unit, territory_name, dsp
    ),
    known_territories AS (
        SELECT DISTINCT territory_name FROM revenue_aggregated
    ),
    streams_standardized AS (
        SELECT
            month,
            business_unit,
            CASE
                WHEN country IN (SELECT territory_name FROM known_territories) THEN country
                ELSE 'All Other Locations'
            END AS territory_name,
            dsp,
            total_streams
        FROM streams
    ),
    streams_aggregated AS (
        SELECT
            month,
            business_unit,
            territory_name,
            dsp,
            SUM(total_streams) AS total_streams
        FROM streams_standardized
        GROUP BY month, business_unit, territory_name, dsp
    ),
    merged_data AS (
        SELECT
            COALESCE(s.month, r.month) AS month,
            COALESCE(s.business_unit, r.business_unit) AS business_unit,
            COALESCE(s.territory_name, r.territory_name) AS territory_name,
            COALESCE(s.dsp, r.dsp) AS dsp,
            s.total_streams,
            r.total_gross_amount
        FROM streams_aggregated s
        FULL OUTER JOIN revenue_aggregated r
            ON s.month = r.month
            AND s.business_unit = r.business_unit
            AND s.territory_name = r.territory_name
            AND s.dsp = r.dsp
    )
    SELECT
        month,
        business_unit,
        territory_name,
        dsp,
        total_streams,
        total_gross_amount,
        CASE WHEN total_gross_amount IS NULL THEN 1 ELSE 0 END AS is_na,
        CASE WHEN total_gross_amount = 0 THEN 1 ELSE 0 END AS is_zero
    FROM merged_data
    """

    out = con.execute(query).fetchdf()
    return out


def run_nowcast_and_audit(merged_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Run quality flags, nowcast selected rows, and return final table + audits + summary."""
    # Step 1) First-pass quality flags on harmonized data.
    # These flags identify suspicious low-revenue rows using local cohort comparison.
    work_df = add_quality_flags(
        merged_df,
        stream_col=STREAM_COL,
        target_col=TARGET_COL,
        low_revenue_threshold=None,
        min_group_rows=MIN_FLAG_GROUP_ROWS,
        fe_threshold_quantile=FE_THRESHOLD_QUANTILE,
    )

    # Step 2) Define rows that should be estimated.
    # - missing/zero revenue rows
    # - strict quality-correction rows
    target_mask = (work_df["is_na"] == 1) | (work_df["is_zero"] == 1)
    quality_mask = work_df["quality_flag_low_revenue_high_streams"] == 1
    rows_to_nowcast = target_mask | quality_mask

    # Step 3) Build lag features and re-flag on lag frame for consistent training filters.
    lag_df = add_lag_features(
        data=work_df,
        group_cols=GROUP_COLS,
        target_col=TARGET_COL,
        stream_col=STREAM_COL,
        lag=1,
    )
    lag_df = add_quality_flags(
        lag_df,
        stream_col=STREAM_COL,
        target_col=TARGET_COL,
        low_revenue_threshold=None,
        min_group_rows=MIN_FLAG_GROUP_ROWS,
        fe_threshold_quantile=FE_THRESHOLD_QUANTILE,
    )

    # Step 4) Train only on cleaner known rows.
    # We intentionally exclude missing/zero rows and strict quality-flag rows from training.
    train_mask = (
        (lag_df["is_na"] == 0)
        & (lag_df["is_zero"] == 0)
        & (lag_df[STREAM_COL] > 0)
        & (lag_df["quality_flag_low_revenue_high_streams"] == 0)
    )

    train_df = lag_df.loc[train_mask].copy()
    score_df = lag_df.loc[rows_to_nowcast].copy()

    # Step 5) Run final nowcast model on selected rows.
    score_df["nowcast_pred"] = fit_predict_lag_hier_nowcast(
        train_data=train_df,
        score_data=score_df,
        group_cols=GROUP_COLS,
        stream_col=STREAM_COL,
        target_col=TARGET_COL,
    )

    # Step 5b) Build prediction intervals from calibration residuals on known clean rows.
    calibration_df = train_df.copy()
    calibration_df['calibration_pred'] = fit_predict_lag_hier_nowcast(
        train_data=train_df,
        score_data=calibration_df.copy(),
        group_cols=GROUP_COLS,
        stream_col=STREAM_COL,
        target_col=TARGET_COL,
    )

    score_df = build_relative_prediction_intervals(
        calibration_df=calibration_df,
        prediction_df=score_df,
        pred_col='nowcast_pred',
        actual_col=TARGET_COL,
        calibration_pred_col='calibration_pred',
        group_cols=['territory_name', 'dsp'],
        alpha=0.10,
        min_group_rows=30,
        lower_col='nowcast_lower_90',
        upper_col='nowcast_upper_90',
    )

    # Step 6) Build final revenue field and transparent status labels.
    final_df = work_df.copy()
    final_df["filled_revenue"] = final_df[TARGET_COL]
    final_df.loc[score_df.index, "filled_revenue"] = score_df["nowcast_pred"].values
    final_df['nowcast_lower_90'] = np.nan
    final_df['nowcast_upper_90'] = np.nan
    final_df.loc[score_df.index, 'nowcast_lower_90'] = score_df['nowcast_lower_90'].values
    final_df.loc[score_df.index, 'nowcast_upper_90'] = score_df['nowcast_upper_90'].values

    final_df["value_status"] = np.where(rows_to_nowcast, "nowcasted", "actual")
    final_df["nowcast_reason"] = np.select(
        [target_mask, quality_mask],
        ["missing_or_zero_revenue", "quality_flag_low_revenue_high_streams"],
        default="N/A",
    )

    # Step 7) Create RPS in business-friendly units (revenue per 1M streams).
    final_df["rps"] = np.where(
        final_df[STREAM_COL] > 0,
        final_df["filled_revenue"] / (final_df[STREAM_COL] / 1_000_000),
        np.nan,
    )

    # Keep only delivery columns required by Finance and audit reviewers.
    deliverable_cols = [
        "month",
        "business_unit",
        "territory_name",
        "dsp",
        STREAM_COL,
        TARGET_COL,
        "filled_revenue",
        "nowcast_lower_90",
        "nowcast_upper_90",
        "value_status",
        "nowcast_reason",
        "rps",
    ]
    final_df = final_df[deliverable_cols].copy()

    # Step 8) Final audit:
    #   A) hard-invalid checks (domain validity)
    #   B) robust outlier checks using MAD on log-RPS
    audit_df = final_df.copy()
    log_rps = np.log1p(audit_df["rps"].replace([np.inf, -np.inf], np.nan))
    med = np.nanmedian(log_rps)
    mad = np.nanmedian(np.abs(log_rps - med))

    audit_df["robust_z_all"] = np.nan
    if mad > 0 and np.isfinite(mad):
        audit_df["robust_z_all"] = 0.6745 * (log_rps - med) / mad

    hard_invalid = (
        (audit_df["filled_revenue"] < 0)
        | (audit_df[STREAM_COL] <= 0)
        | (audit_df["rps"].isna())
        | (~np.isfinite(audit_df["rps"]))
    )
    suspicious = audit_df[(~hard_invalid) & (audit_df["robust_z_all"].abs() > 5.5)].copy()

    # Step 8b) DSP-month reliability audit for stakeholder monitoring.
    # Flags months with heavy nowcast dependency or wide uncertainty range.
    dsp_month_audit = (
        final_df.groupby(["month", "dsp"], dropna=False)
        .agg(
            total_streams=(STREAM_COL, "sum"),
            revenue_mid=("filled_revenue", "sum"),
            revenue_lower=("nowcast_lower_90", "sum"),
            revenue_upper=("nowcast_upper_90", "sum"),
            nowcast_rows=("value_status", lambda s: (s == "nowcasted").sum()),
            total_rows=("value_status", "size"),
        )
        .reset_index()
    )
    dsp_month_audit = dsp_month_audit[dsp_month_audit["total_streams"] > 0].copy()
    dsp_month_audit["rps_mid"] = dsp_month_audit["revenue_mid"] / (dsp_month_audit["total_streams"] / 1_000_000)
    dsp_month_audit["nowcast_rate"] = dsp_month_audit["nowcast_rows"] / dsp_month_audit["total_rows"]
    dsp_month_audit["uncertainty_width_pct"] = np.where(
        dsp_month_audit["revenue_mid"] > 0,
        (dsp_month_audit["revenue_upper"] - dsp_month_audit["revenue_lower"]) / dsp_month_audit["revenue_mid"] * 100,
        np.nan,
    )
    dsp_month_audit["risk_flag"] = np.where(
        (dsp_month_audit["nowcast_rate"] >= 0.80) | (dsp_month_audit["uncertainty_width_pct"] >= 35),
        "caution",
        "normal",
    )
    high_risk_count = int((dsp_month_audit["risk_flag"] == "caution").sum())

    # Step 9) Compact run summary for monitoring and handoff.
    summary = {
        "rows_total": int(len(final_df)),
        "rows_nowcasted": int((final_df["value_status"] == "nowcasted").sum()),
        "rows_nowcasted_missing_or_zero": int((final_df["nowcast_reason"] == "missing_or_zero_revenue").sum()),
        "rows_nowcasted_quality_flag": int((final_df["nowcast_reason"] == "quality_flag_low_revenue_high_streams").sum()),
        "rows_actual": int((final_df["value_status"] == "actual").sum()),
        "quality_flag_rows": int(quality_mask.sum()),
        "hard_invalid_rows": int(hard_invalid.sum()),
        "severe_suspicious_rows": int(len(suspicious)),
        "dsp_month_caution_rows": high_risk_count,
    }

    return final_df, suspicious, dsp_month_audit, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Finance-ready revenue view (harmonize + nowcast + audit)",
    )
    parser.add_argument("--input-merged", default="data/merged_data.csv")
    parser.add_argument("--input-xlsx", default="")
    parser.add_argument("--revenue-csv", default="")
    parser.add_argument("--streams-csv", default="")
    parser.add_argument("--refresh-merged", action="store_true")
    args = parser.parse_args()

    # Convention: run from repo root. Outputs always land in ./data.
    root = Path.cwd()
    data_dir = root / "data"

    # Data ingestion mode:
    # - refresh_merged: rebuild harmonized dataset with SQL from source files
    # - otherwise: use existing merged_data.csv for a fast run
    if args.refresh_merged:
        merged_df = harmonize_with_sql(
            revenue_csv=Path(args.revenue_csv) if args.revenue_csv else None,
            streams_csv=Path(args.streams_csv) if args.streams_csv else None,
            source_xlsx=Path(args.input_xlsx) if args.input_xlsx else None,
        )
        merged_df.to_csv(data_dir / "merged_data.csv", index=False)
    else:
        merged_df = pd.read_csv(root / args.input_merged)

    # Run modeling + audit pipeline.
    final_df, suspicious_df, dsp_month_audit_df, summary = run_nowcast_and_audit(merged_df)

    final_csv = data_dir / "Nathan_Zhang_CMG_Revenue_Forecast_Oct2025.csv"
    final_xlsx = data_dir / "Nathan_Zhang_CMG_Revenue_Forecast_Oct2025.xlsx"
    suspicious_csv = data_dir / "output_audit_suspicious_rows.csv"
    dsp_month_audit_csv = data_dir / "output_audit_dsp_month_reliability.csv"
    run_summary_json = data_dir / "pipeline_run_summary.json"

    # Persist outputs used by finance consumers and QA reviewers.
    final_df.to_csv(final_csv, index=False)
    final_df.to_excel(final_xlsx, index=False, engine="openpyxl")
    suspicious_df.to_csv(suspicious_csv, index=False)
    dsp_month_audit_df.to_csv(dsp_month_audit_csv, index=False)
    run_summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Build completed.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
