import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from utils.forecast_utils import add_time_columns, fit_epsr_lookup, apply_epsr_lookup


def split_time_holdout(eval_df: pd.DataFrame):
    """Use the latest available month as validation for a simple time-aware split."""
    eval_months = sorted(eval_df['month'].unique())
    if len(eval_months) < 2:
        return train_test_split(eval_df, test_size=0.2, random_state=42)

    val_month = eval_months[-1]
    train_df = eval_df[eval_df['month'] < val_month].copy()
    val_df = eval_df[eval_df['month'] == val_month].copy()
    return train_df, val_df


def _non_negative(pred):
    """Clip predictions to valid non-negative revenue."""
    return np.clip(pred, a_min=0, a_max=None)


def _build_feature_matrices(
    train_data: pd.DataFrame,
    score_data: pd.DataFrame,
    categorical_cols,
    stream_col: str,
):
    """Build aligned train/score matrices with light feature engineering and lag support."""
    train_feat = add_time_columns(train_data).copy()
    score_feat = add_time_columns(score_data).copy()

    # Human-readable lightweight features
    train_feat['log_streams'] = np.log1p(train_feat[stream_col])
    score_feat['log_streams'] = np.log1p(score_feat[stream_col])

    feature_cols = categorical_cols + [
        'month_num', 'quarter', stream_col, 'log_streams',
        'gross_lag_1', 'streams_lag_1', 'epsr_lag_1'
    ]

    lag_cols = ['gross_lag_1', 'streams_lag_1', 'epsr_lag_1']
    lag_fill = {col: train_feat[col].median() for col in lag_cols}

    train_feat[lag_cols] = train_feat[lag_cols].fillna(lag_fill)
    score_feat[lag_cols] = score_feat[lag_cols].fillna(lag_fill)

    X_train = pd.get_dummies(train_feat[feature_cols], columns=categorical_cols, dummy_na=True)
    X_score = pd.get_dummies(score_feat[feature_cols], columns=categorical_cols, dummy_na=True)
    X_score = X_score.reindex(columns=X_train.columns, fill_value=0)

    clean_cols = [col.replace(' ', '_') for col in X_train.columns]
    X_train.columns = clean_cols
    X_score.columns = clean_cols

    y_train = train_feat['total_gross_amount'].values
    y_train_log = np.log1p(y_train)

    return X_train, X_score, y_train, y_train_log


def fit_predict_xgb_refined(
    train_data: pd.DataFrame,
    score_data: pd.DataFrame,
    categorical_cols,
    stream_col: str,
):
    """Train an XGBoost regressor on log-revenue and return non-negative predictions."""
    X_train, X_score, _, y_train_log = _build_feature_matrices(train_data, score_data, categorical_cols, stream_col)

    model = XGBRegressor(
        n_estimators=350,
        learning_rate=0.04,
        max_depth=4,
        min_child_weight=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=3.0,
        random_state=42,
        objective='reg:squarederror'
    )
    model.fit(X_train, y_train_log)

    pred_log = model.predict(X_score)
    return _non_negative(np.expm1(pred_log))


def fit_predict_lgbm_refined(
    train_data: pd.DataFrame,
    score_data: pd.DataFrame,
    categorical_cols,
    stream_col: str,
):
    """Train a LightGBM regressor on log-revenue and return non-negative predictions."""
    X_train, X_score, _, y_train_log = _build_feature_matrices(train_data, score_data, categorical_cols, stream_col)

    model = LGBMRegressor(
        n_estimators=350,
        learning_rate=0.04,
        num_leaves=24,
        min_child_samples=30,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=-1
    )
    model.fit(X_train, y_train_log)

    pred_log = model.predict(X_score)
    return _non_negative(np.expm1(pred_log))


def fit_predict_knn_refined(
    train_data: pd.DataFrame,
    score_data: pd.DataFrame,
    categorical_cols,
    stream_col: str,
):
    """Train a scaled KNN regressor and return non-negative predictions."""
    X_train, X_score, y_train, _ = _build_feature_matrices(train_data, score_data, categorical_cols, stream_col)

    scaler = StandardScaler(with_mean=True, with_std=True)
    X_train_scaled = scaler.fit_transform(X_train.values)
    X_score_scaled = scaler.transform(X_score.values)

    model = KNeighborsRegressor(
        n_neighbors=21,
        weights='distance',
        metric='minkowski',
        p=2
    )
    model.fit(X_train_scaled, y_train)

    pred = model.predict(X_score_scaled)
    return _non_negative(pred)


def fit_predict_lag_hier_nowcast(
    train_data: pd.DataFrame,
    score_data: pd.DataFrame,
    group_cols,
    stream_col: str,
    target_col: str,
):
    """Lag-aware nowcast: EPSR anchor + bounded lag trend adjustment + simple reconciliation."""
    scored = score_data.copy().reset_index(drop=True)
    train_epsr = fit_epsr_lookup(train_data, group_cols, target_col, stream_col)
    epsr_anchor = apply_epsr_lookup(scored, train_epsr, group_cols, stream_col, target_col)

    safe_streams = scored[stream_col].fillna(0).clip(lower=0)
    global_epsr = np.divide(
        train_data[target_col].sum(),
        max(train_data[stream_col].sum(), 1),
    )
    anchor_fallback = _non_negative(safe_streams * global_epsr)
    anchor_pred = epsr_anchor['pred'].fillna(anchor_fallback).fillna(train_data[target_col].median())
    anchor_pred = pd.Series(anchor_pred.values, index=scored.index)
    anchor_pred = _non_negative(anchor_pred)

    # Lag projection from prior revenue and stream growth.
    lag_streams = scored['streams_lag_1'].replace(0, np.nan)
    stream_growth = safe_streams / lag_streams
    stream_growth = stream_growth.replace([np.inf, -np.inf], np.nan).clip(lower=0.50, upper=1.80)

    lag_proj = scored['gross_lag_1'] * stream_growth
    lag_proj = _non_negative(lag_proj)

    scored['anchor_pred'] = anchor_pred
    scored['lag_proj'] = lag_proj

    # Anchor-first blend for stability.
    has_lag = scored['lag_proj'].notna()
    lag_weight = pd.Series(np.where(has_lag, 0.20, 0.00), index=scored.index)
    lag_value = scored['lag_proj'].fillna(scored['anchor_pred'])
    base_pred = lag_weight * lag_value + (1 - lag_weight) * scored['anchor_pred']

    # Keep lag adjustment within an intuitive and explainable range around EPSR anchor.
    adjust_ratio = base_pred.div(scored['anchor_pred'].replace(0, np.nan))
    adjust_ratio = adjust_ratio.replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(lower=0.85, upper=1.15)
    scored['blended_pred'] = scored['anchor_pred'] * adjust_ratio

    # Reconcile at (month, dsp) so totals remain close to anchor totals.
    grouped = scored.groupby(['month', 'dsp'], as_index=False).agg(
        raw_sum=('blended_pred', 'sum'),
        anchor_sum=('anchor_pred', 'sum')
    )
    grouped['scale'] = np.where(grouped['raw_sum'] > 0, grouped['anchor_sum'] / grouped['raw_sum'], 1.0)
    grouped['scale'] = np.clip(grouped['scale'], 0.85, 1.15)
    grouped['scale'] = np.where(np.isfinite(grouped['scale']), grouped['scale'], 1.0)

    scored = scored.merge(grouped[['month', 'dsp', 'scale']], on=['month', 'dsp'], how='left')
    scored['scale'] = scored['scale'].fillna(1.0)

    reconciled = _non_negative(scored['blended_pred'] * scored['scale'])
    reconciled = np.nan_to_num(reconciled, nan=0.0, posinf=0.0, neginf=0.0)
    return reconciled
