import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from typing import Optional


def add_time_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Add parsed month and simple calendar features used across models."""
    out = data.copy()
    out['month'] = pd.to_datetime(out['month'])
    out['month_str'] = out['month'].dt.to_period('M').astype(str)
    out['month_num'] = out['month'].dt.month
    out['quarter'] = out['month'].dt.quarter
    return out

def add_quality_flags(
    data: pd.DataFrame,
    stream_col: Optional[str] = None,
    target_col: Optional[str] = None,
    low_ratio_threshold: float = 0.20,
    low_revenue_threshold: float = 99
) -> pd.DataFrame:
    """
    Add a quality flag for rows that have low revenue but high streams, 
    which may indicate potential data issues or edge cases that require 
    special handling in the modeling process.
    """
    out = data.copy()
    out['quality_flag_low_revenue_high_streams'] = 0

    if stream_col is None:
        stream_candidates = ['total_streams', 'streams', 'stream_count']
        stream_col = next((col for col in stream_candidates if col in out.columns), None)

    if target_col is None:
        target_candidates = ['total_gross_amount', 'gross_revenue_usd', 'revenue']
        target_col = next((col for col in target_candidates if col in out.columns), None)

    if stream_col is None or target_col is None:
        raise ValueError(
            'add_quality_flags could not infer stream/target columns. '
            'Pass stream_col and target_col explicitly.'
        )

    known_mask = (
        (out['is_na'] == 0)
        & (out['is_zero'] == 0)
        & (out[stream_col] > 0)
        & (out[target_col] > 0)
    )

    if known_mask.sum() == 0:
        return out

    out.loc[known_mask, 'epsr_row'] = out.loc[known_mask, target_col] / out.loc[known_mask, stream_col]

    known = out.loc[known_mask].copy()
    cohort_stats = (
        known.groupby(['territory_name', 'dsp'])['epsr_row']
        .agg(cohort_epsr_median='median', cohort_epsr_p10=lambda s: s.quantile(0.10), cohort_n='size')
        .reset_index()
    )
    dsp_stats = (
        known.groupby(['dsp'])['epsr_row']
        .agg(dsp_epsr_median='median', dsp_epsr_p10=lambda s: s.quantile(0.10), dsp_n='size')
        .reset_index()
    )
    global_median = known['epsr_row'].median()
    global_p10 = known['epsr_row'].quantile(0.10)

    out = out.merge(cohort_stats, on=['territory_name', 'dsp'], how='left')
    out = out.merge(dsp_stats, on=['dsp'], how='left')

    out['epsr_ref_median'] = out['cohort_epsr_median'].fillna(out['dsp_epsr_median']).fillna(global_median)
    out['epsr_ref_p10'] = out['cohort_epsr_p10'].fillna(out['dsp_epsr_p10']).fillna(global_p10)
    out['epsr_ratio_vs_local'] = out['epsr_row'] / out['epsr_ref_median'].replace(0, np.nan)

    local_low_signal = (
        out['epsr_ratio_vs_local'] <= low_ratio_threshold
    ) | (
        out['epsr_row'] <= (0.5 * out['epsr_ref_p10'])
    )

    out['quality_flag_low_revenue_high_streams'] = (
        known_mask
        & (out[target_col] <= low_revenue_threshold)
        & local_low_signal.fillna(False)
    ).astype(int)

    drop_cols = [
        'epsr_row', 'cohort_epsr_median', 'cohort_epsr_p10', 'cohort_n',
        'dsp_epsr_median', 'dsp_epsr_p10', 'dsp_n', 'epsr_ref_median', 'epsr_ref_p10', 'epsr_ratio_vs_local'
    ]
    out = out.drop(columns=[c for c in drop_cols if c in out.columns])

    return out

def add_lag_features(
    data: pd.DataFrame,
    group_cols,
    target_col: str,
    stream_col: str,
    lag: int = 1
) -> pd.DataFrame:
    """Create lagged revenue, lagged streams, and lagged EPSR for each cohort."""
    out = add_time_columns(data).sort_values(group_cols + ['month']).copy()

    out[f'gross_lag_{lag}'] = out.groupby(group_cols)[target_col].shift(lag)
    out[f'streams_lag_{lag}'] = out.groupby(group_cols)[stream_col].shift(lag)

    lag_streams = out[f'streams_lag_{lag}'].replace(0, np.nan)
    out[f'epsr_lag_{lag}'] = out[f'gross_lag_{lag}'] / lag_streams

    return out


def compute_wmape(actual, pred):
    """Compute weighted MAPE with a zero-denominator guard."""
    actual = np.asarray(actual)
    pred = np.asarray(pred)
    denominator = np.abs(actual).sum()
    if denominator == 0:
        return np.nan
    return np.abs(actual - pred).sum() / denominator


def evaluate_predictions(data: pd.DataFrame, target_col: str, pred_col: str, label: str):
    """Return a compact MAE/WMAPE summary for one prediction column."""
    mae = mean_absolute_error(data[target_col], data[pred_col])
    wmape = compute_wmape(data[target_col].values, data[pred_col].values)
    return {
        'Method': label,
        'MAE': mae,
        'WMAPE': wmape,
        'Rows': len(data)
    }


def prepare_eval_frames(
    data: pd.DataFrame,
    stream_col: str,
    excluded_months=None
):
    """Split data into known rows for training/eval and target rows to impute."""
    if excluded_months is None:
        excluded_months = []

    work = add_time_columns(data)
    known_mask = (work['is_na'] == 0) & (work['is_zero'] == 0) & (work[stream_col] > 0)
    include_mask = ~work['month_str'].isin(excluded_months)

    known_all = work.loc[known_mask].copy()
    known_eval = work.loc[known_mask & include_mask].copy()
    target_rows = work.loc[(work['is_na'] == 1) | (work['is_zero'] == 1)].copy()

    return work, known_all, known_eval, target_rows


def fit_group_average_lookup(train_known: pd.DataFrame, group_cols, target_col: str):
    """Fit cohort-level and global averages for the group-average baseline."""
    group_avg = (
        train_known.groupby(group_cols)[target_col]
        .mean()
        .reset_index(name='group_avg')
    )
    global_avg = train_known[target_col].mean()

    return {
        'group_avg': group_avg,
        'global_avg': global_avg
    }


def apply_group_average_lookup(score_data: pd.DataFrame, lookup, group_cols):
    """Apply a pre-fitted group-average lookup with global fallback."""
    scored = score_data.copy()
    scored = scored.merge(lookup['group_avg'], on=group_cols, how='left')
    scored['pred'] = scored['group_avg'].fillna(lookup['global_avg'])
    return scored


def fit_epsr_lookup(train_known: pd.DataFrame, group_cols, target_col: str, stream_col: str):
    """Fit EPSR fallback tables from most-granular to global level."""
    cohort = (
        train_known.groupby(group_cols).apply(
            lambda x: x[target_col].sum() / x[stream_col].sum()
        )
        .reset_index(name='epsr_cohort')
    )

    territory_dsp = (
        train_known.groupby(['territory_name', 'dsp']).apply(
            lambda x: x[target_col].sum() / x[stream_col].sum()
        )
        .reset_index(name='epsr_territory_dsp')
    )

    dsp = (
        train_known.groupby(['dsp']).apply(
            lambda x: x[target_col].sum() / x[stream_col].sum()
        )
        .reset_index(name='epsr_dsp')
    )

    global_rate = train_known[target_col].sum() / train_known[stream_col].sum()

    return {
        'cohort': cohort,
        'territory_dsp': territory_dsp,
        'dsp': dsp,
        'global_rate': global_rate
    }


def apply_epsr_lookup(
    score_data: pd.DataFrame,
    lookup,
    group_cols,
    stream_col: str,
    target_col: str
):
    """Apply EPSR lookup and produce revenue prediction from streams."""
    scored = score_data.copy()
    base_cols = group_cols + ['month', stream_col, target_col, 'is_na', 'is_zero']
    base_cols = [col for col in base_cols if col in scored.columns]
    scored = scored[base_cols].copy()

    scored = scored.merge(lookup['cohort'], on=group_cols, how='left')
    scored = scored.merge(lookup['territory_dsp'], on=['territory_name', 'dsp'], how='left')
    scored = scored.merge(lookup['dsp'], on=['dsp'], how='left')

    scored['epsr_rate_used'] = (
        scored['epsr_cohort']
        .fillna(scored['epsr_territory_dsp'])
        .fillna(scored['epsr_dsp'])
        .fillna(lookup['global_rate'])
    )
    scored['pred'] = scored[stream_col] * scored['epsr_rate_used']

    return scored


def run_distribution_backtest(
    known_pool: pd.DataFrame,
    target_pool: pd.DataFrame,
    fit_predict_fn,
    target_col: str,
    n_repeats: int = 30,
    random_state: int = 42,
    model_name: str = 'Model'
):
    """Run repeated month-distribution-aware holdout tests for one model function."""
    rng = np.random.default_rng(random_state)

    month_dist = target_pool['month'].value_counts(normalize=True).sort_index()
    month_dist = month_dist[month_dist.index.isin(set(known_pool['month'].unique()))]

    if month_dist.empty:
        raise ValueError('No overlapping months between known pool and target pool.')

    month_dist = month_dist / month_dist.sum()

    holdout_ratio = len(target_pool) / (len(known_pool) + len(target_pool))
    holdout_size = max(1, int(len(known_pool) * holdout_ratio))

    overall_rows = []
    monthly_rows = []

    for repeat in range(1, n_repeats + 1):
        per_month_counts = np.floor(month_dist.values * holdout_size).astype(int)
        remainder = holdout_size - per_month_counts.sum()

        if remainder > 0:
            for idx in rng.choice(len(per_month_counts), size=remainder, replace=True, p=month_dist.values):
                per_month_counts[idx] += 1

        sampled_idx = []
        for month_value, n_rows in zip(month_dist.index, per_month_counts):
            candidates = known_pool[known_pool['month'] == month_value]
            n_rows = min(n_rows, len(candidates))
            if n_rows > 0:
                sampled_idx.extend(rng.choice(candidates.index.values, size=n_rows, replace=False).tolist())

        if len(sampled_idx) == 0:
            continue

        test_df = known_pool.loc[sampled_idx].copy()
        train_df = known_pool.drop(index=sampled_idx).copy()

        pred_values = fit_predict_fn(train_df, test_df)

        scored = test_df.copy()
        scored['pred'] = pred_values
        scored['abs_error'] = np.abs(scored[target_col] - scored['pred'])
        scored['abs_actual'] = np.abs(scored[target_col])

        overall_rows.append({
            'Model': model_name,
            'repeat': repeat,
            'rows': len(scored),
            'MAE': mean_absolute_error(scored[target_col], scored['pred']),
            'WMAPE': compute_wmape(scored[target_col].values, scored['pred'].values)
        })

        month_eval = (
            scored.groupby('month')
            .apply(lambda x: pd.Series({
                'rows': len(x),
                'mae': mean_absolute_error(x[target_col], x['pred']),
                'wmape': compute_wmape(x[target_col].values, x['pred'].values),
                'abs_error_sum': x['abs_error'].sum(),
                'abs_actual_sum': x['abs_actual'].sum()
            }))
            .reset_index()
        )
        month_eval['Model'] = model_name
        month_eval['repeat'] = repeat
        monthly_rows.append(month_eval)

    overall_df = pd.DataFrame(overall_rows)
    monthly_df = pd.concat(monthly_rows, ignore_index=True) if monthly_rows else pd.DataFrame()

    return overall_df, monthly_df
