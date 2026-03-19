import numpy as np
import pandas as pd


def make_future_dates(last_date, horizon, freq="M"):
    return pd.date_range(start=last_date, periods=horizon + 1, freq=freq)[1:]


def get_future_calendar_features(future_dates):
    out = {}
    for i, d in enumerate(future_dates, start=1):
        out[f"month_t{i}"] = int(d.month)
        out[f"quarter_t{i}"] = int(d.quarter)
    return out


def get_future_fourier_features(future_dates, period=12, order=2):
    out = {}
    for i, d in enumerate(future_dates, start=1):
        t = d.month
        for k in range(1, order + 1):
            out[f"fourier_sin_{k}_t{i}"] = np.sin(2 * np.pi * k * t / period)
            out[f"fourier_cos_{k}_t{i}"] = np.cos(2 * np.pi * k * t / period)
    return out


def extract_lag_features_from_history(history_values, lags):
    out = {}
    n = len(history_values)

    for lag in lags:
        if lag > n:
            out[f"lag_{lag}"] = np.nan
        else:
            out[f"lag_{lag}"] = float(history_values[-lag])

    return out


def extract_seasonal_lag_features_from_history(history_values, seasonal_lags):
    out = {}
    n = len(history_values)

    for lag in seasonal_lags:
        if lag > n:
            out[f"slag_{lag}"] = np.nan
        else:
            out[f"slag_{lag}"] = float(history_values[-lag])

    return out


def build_feature_row(
    history_values,
    future_dates,
    config,
):
    row = {}

    if config.get("use_lags", False):
        row.update(
            extract_lag_features_from_history(
                history_values=history_values,
                lags=config.get("lags", []),
            )
        )

    if config.get("use_seasonal_lags", False):
        row.update(
            extract_seasonal_lag_features_from_history(
                history_values=history_values,
                seasonal_lags=config.get("seasonal_lags", []),
            )
        )

    if config.get("use_calendar", False):
        row.update(get_future_calendar_features(future_dates))

    if config.get("use_fourier", False):
        row.update(
            get_future_fourier_features(
                future_dates=future_dates,
                period=config.get("fourier_period", 12),
                order=config.get("fourier_order", 2),
            )
        )

    return row


def build_target_block(y_values, start_idx, model_horizon):
    y_block = y_values[start_idx : start_idx + model_horizon]
    return np.asarray(y_block, dtype=float)


def make_train_windows_one_series(
    g,
    config,
    history,
    model_horizon,
    block_offset=0,
    id_col="unique_id",
    date_col="ds",
    target_col="y",
):
    g = g.sort_values(date_col).reset_index(drop=True).copy()

    y = g[target_col].values
    ds = pd.to_datetime(g[date_col].values)
    uid = g[id_col].iloc[0]

    rows = []
    targets = []

    n = len(g)
    last_start = n - history - block_offset - model_horizon

    if last_start < 0:
        return pd.DataFrame(), np.empty((0, model_horizon))

    for start in range(last_start + 1):
        hist_start = start
        hist_end = start + history

        target_start = hist_end + block_offset
        target_end = target_start + model_horizon

        history_values = y[hist_start:hist_end]
        future_dates = ds[target_start:target_end]

        row = build_feature_row(
            history_values=history_values,
            future_dates=future_dates,
            config=config,
        )
        row[id_col] = uid
        row["window_start"] = start
        row["window_end"] = hist_end - 1
        row["block_offset"] = block_offset

        y_block = build_target_block(
            y_values=y,
            start_idx=target_start,
            model_horizon=model_horizon,
        )

        rows.append(row)
        targets.append(y_block)

    X = pd.DataFrame(rows)
    Y = np.vstack(targets)

    return X, Y


def make_train_windows_all_series(
    train_df,
    config,
    history,
    model_horizon,
    block_offset=0,
    id_col="unique_id",
    date_col="ds",
    target_col="y",
):
    X_parts = []
    Y_parts = []

    for _, g in train_df.groupby(id_col):
        X_i, Y_i = make_train_windows_one_series(
            g=g,
            config=config,
            history=history,
            model_horizon=model_horizon,
            block_offset=block_offset,
            id_col=id_col,
            date_col=date_col,
            target_col=target_col,
        )

        if len(X_i) > 0:
            X_parts.append(X_i)
            Y_parts.append(Y_i)

    if len(X_parts) == 0:
        return pd.DataFrame(), np.empty((0, model_horizon))

    X = pd.concat(X_parts, axis=0).reset_index(drop=True)
    Y = np.vstack(Y_parts)

    return X, Y


def make_holdout_window_one_series(
    g,
    config,
    history,
    model_horizon,
    block_offset=0,
    id_col="unique_id",
    date_col="ds",
    target_col="y",
):
    g = g.sort_values(date_col).reset_index(drop=True).copy()

    y = g[target_col].values
    ds = pd.to_datetime(g[date_col].values)
    uid = g[id_col].iloc[0]

    need_len = history + block_offset + model_horizon
    if len(g) < need_len:
        raise ValueError(
            f"Ряд {uid} слишком короткий для history={history}, block_offset={block_offset}, model_horizon={model_horizon}"
        )

    history_values = y[:history]

    target_start = history + block_offset
    target_end = target_start + model_horizon
    future_dates = ds[target_start:target_end]

    row = build_feature_row(
        history_values=history_values,
        future_dates=future_dates,
        config=config,
    )
    row[id_col] = uid
    row["block_offset"] = block_offset

    X = pd.DataFrame([row])

    y_block = y[target_start:target_end]
    Y = np.asarray(y_block, dtype=float).reshape(1, -1)

    return X, Y


def make_holdout_windows_all_series(
    holdout_df,
    config,
    history,
    model_horizon,
    block_offset=0,
    id_col="unique_id",
    date_col="ds",
    target_col="y",
):
    X_parts = []
    Y_parts = []

    for _, g in holdout_df.groupby(id_col):
        X_i, Y_i = make_holdout_window_one_series(
            g=g,
            config=config,
            history=history,
            model_horizon=model_horizon,
            block_offset=block_offset,
            id_col=id_col,
            date_col=date_col,
            target_col=target_col,
        )
        X_parts.append(X_i)
        Y_parts.append(Y_i)

    X = pd.concat(X_parts, axis=0).reset_index(drop=True)
    Y = np.vstack(Y_parts)

    return X, Y


def get_categorical_feature_names_from_config(config, model_horizon):
    cat_cols = []

    if config.get("use_calendar", False):
        calendar_features = config.get("calendar_features", [])
        for step in range(1, model_horizon + 1):
            if "month" in calendar_features:
                cat_cols.append(f"month_t{step}")
            if "quarter" in calendar_features:
                cat_cols.append(f"quarter_t{step}")

    return cat_cols


def sanity_check_window_outputs(X, Y, model_horizon):
    if len(X) == 0:
        print("X is empty")
        return

    assert Y.ndim == 2
    assert Y.shape[1] == model_horizon
    assert len(X) == len(Y)

    print("window outputs look ok")
    print("X shape:", X.shape)
    print("Y shape:", Y.shape)