import numpy as np
import pandas as pd


def add_calendar_features(df, date_col="ds"):
    out = df.copy()
    out["month"] = out[date_col].dt.month.astype(int)
    out["quarter"] = out[date_col].dt.quarter.astype(int)
    return out


def add_fourier_features(df, period=12, order=2, date_col="ds", id_col="unique_id"):
    out = df.copy()
    out["_t"] = out.groupby(id_col).cumcount() + 1

    for k in range(1, order + 1):
        out[f"fourier_sin_{k}"] = np.sin(2 * np.pi * k * out["_t"] / period)
        out[f"fourier_cos_{k}"] = np.cos(2 * np.pi * k * out["_t"] / period)

    out = out.drop(columns=["_t"])
    return out


def add_lag_features(df, lags, id_col="unique_id", target_col="y"):
    out = df.copy()
    for lag in sorted(set(lags)):
        out[f"lag_{lag}"] = out.groupby(id_col)[target_col].shift(lag)
    return out


def add_seasonal_lag_features(df, seasonal_lags, id_col="unique_id", target_col="y"):
    out = df.copy()
    for lag in sorted(set(seasonal_lags)):
        out[f"slag_{lag}"] = out.groupby(id_col)[target_col].shift(lag)
    return out


def apply_feature_config(
    df,
    config,
    id_col="unique_id",
    date_col="ds",
    target_col="y",
):
    out = df.copy()

    if config.get("use_lags", False):
        out = add_lag_features(
            out,
            lags=config.get("lags", []),
            id_col=id_col,
            target_col=target_col,
        )

    if config.get("use_seasonal_lags", False):
        out = add_seasonal_lag_features(
            out,
            seasonal_lags=config.get("seasonal_lags", []),
            id_col=id_col,
            target_col=target_col,
        )

    if config.get("use_calendar", False):
        out = add_calendar_features(
            out,
            date_col=date_col,
        )

    if config.get("use_fourier", False):
        out = add_fourier_features(
            out,
            period=config.get("fourier_period", 12),
            order=config.get("fourier_order", 2),
            date_col=date_col,
            id_col=id_col,
        )

    return out


def get_feature_columns(config):
    cols = []

    if config.get("use_lags", False):
        cols += [f"lag_{lag}" for lag in config.get("lags", [])]

    if config.get("use_seasonal_lags", False):
        cols += [f"slag_{lag}" for lag in config.get("seasonal_lags", [])]

    if config.get("use_calendar", False):
        cols += config.get("calendar_features", [])

    if config.get("use_fourier", False):
        order = config.get("fourier_order", 2)
        for k in range(1, order + 1):
            cols += [f"fourier_sin_{k}", f"fourier_cos_{k}"]

    return cols


def get_categorical_columns(config):
    cols = []
    if config.get("use_calendar", False):
        for c in config.get("calendar_features", []):
            if c in ["month", "quarter"]:
                cols.append(c)
    return cols


def prepare_feature_frame(
    df,
    config,
    id_col="unique_id",
    date_col="ds",
    target_col="y",
):
    out = apply_feature_config(
        df=df,
        config=config,
        id_col=id_col,
        date_col=date_col,
        target_col=target_col,
    )

    feature_cols = get_feature_columns(config)
    use_cols = [id_col, date_col, target_col] + feature_cols
    out = out[use_cols].copy()
    return out


def drop_na_feature_rows(
    df,
    config,
    target_col="y",
):
    feature_cols = get_feature_columns(config)
    need_cols = feature_cols + [target_col]
    out = df.dropna(subset=need_cols).reset_index(drop=True)
    return out