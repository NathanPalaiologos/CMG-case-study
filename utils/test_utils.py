import pandas as pd

from utils.forecast_utils import (
    add_time_columns,
    apply_epsr_lookup,
    apply_group_average_lookup,
    fit_epsr_lookup,
    fit_group_average_lookup,
)


def group_avg_fit_predict(train_df, score_df, group_cols, target_col):
    """Fit the group-average baseline on train rows and predict on score rows."""
    lookup = fit_group_average_lookup(
        train_known=train_df,
        group_cols=group_cols,
        target_col=target_col,
    )
    scored = apply_group_average_lookup(
        score_data=score_df,
        lookup=lookup,
        group_cols=group_cols,
    )
    return scored['pred'].values


def naive_fit_predict(train_df, score_df, group_cols, target_col):
    """Use lag-1 revenue within each cohort, with median fallback when lag is missing."""
    base = pd.concat([train_df, score_df], axis=0).copy()
    base = add_time_columns(base).sort_values(group_cols + ['month'])
    base['naive_pred'] = base.groupby(group_cols)[target_col].shift(1)

    out = base.loc[score_df.index, 'naive_pred']
    fallback = train_df[target_col].median()
    return out.fillna(fallback).clip(lower=0).values


def epsr_fit_predict(train_df, score_df, group_cols, target_col, stream_col):
    """Fit EPSR rates on train rows and predict revenue from streams on score rows."""
    lookup = fit_epsr_lookup(
        train_known=train_df,
        group_cols=group_cols,
        target_col=target_col,
        stream_col=stream_col,
    )
    scored = apply_epsr_lookup(
        score_data=score_df,
        lookup=lookup,
        group_cols=group_cols,
        stream_col=stream_col,
        target_col=target_col,
    )
    return scored['pred'].values