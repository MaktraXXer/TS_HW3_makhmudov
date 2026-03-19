import numpy as np
import pandas as pd


def smape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    denom = np.abs(y_true) + np.abs(y_pred)
    diff = np.abs(y_true - y_pred)

    out = np.where(denom == 0, 0.0, 2.0 * diff / denom)
    return float(np.mean(out) * 100)


def mase(y_train, y_true, y_pred, m=12):
    y_train = np.asarray(y_train, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if len(y_train) <= m:
        return np.nan

    naive_errors = np.abs(y_train[m:] - y_train[:-m])
    scale = np.mean(naive_errors)

    if scale == 0:
        return np.nan

    return float(np.mean(np.abs(y_true - y_pred)) / scale)


def _prepare_truth_pred_merge(
    truth_df,
    pred_df,
    id_col="unique_id",
    date_col="ds",
    target_col="y",
    pred_col="y_pred",
):
    truth = truth_df[[id_col, date_col, target_col]].copy()
    pred = pred_df[[id_col, date_col, pred_col]].copy()

    merged = truth.merge(
        pred,
        on=[id_col, date_col],
        how="inner",
        validate="one_to_one",
    ).sort_values([id_col, date_col]).reset_index(drop=True)

    if merged.empty:
        raise ValueError(
            "После merge truth_df и pred_df получился пустой датафрейм.ошибка работы с айди и ds"
        )

    return merged


def evaluate_one_model_per_series(
    train_df,
    truth_df,
    pred_df,
    m=12,
    id_col="unique_id",
    date_col="ds",
    target_col="y",
    pred_col="y_pred",
):
    merged = _prepare_truth_pred_merge(
        truth_df=truth_df,
        pred_df=pred_df,
        id_col=id_col,
        date_col=date_col,
        target_col=target_col,
        pred_col=pred_col,
    )

    rows = []

    train_map = {
        uid: g.sort_values(date_col)[target_col].values
        for uid, g in train_df.groupby(id_col)
    }

    for uid, g in merged.groupby(id_col):
        y_true = g[target_col].values
        y_pred = g[pred_col].values
        y_train = train_map.get(uid)

        rows.append(
            {
                id_col: uid,
                "smape": smape(y_true, y_pred),
                "mase": mase(y_train, y_true, y_pred, m=m),
                "n_obs": len(g),
            }
        )

    out = pd.DataFrame(rows).sort_values(id_col).reset_index(drop=True)
    return out


def summarize_metrics(
    per_series_metrics_df,
    model_name=None,
):
    out = {
        "mean_smape": per_series_metrics_df["smape"].mean(),
        "median_smape": per_series_metrics_df["smape"].median(),
        "mean_mase": per_series_metrics_df["mase"].mean(),
        "median_mase": per_series_metrics_df["mase"].median(),
        "n_series": per_series_metrics_df["unique_id"].nunique(),
    }

    if model_name is not None:
        out["model"] = model_name

    return pd.DataFrame([out])


def compare_models(metrics_dict):
    rows = []

    for model_name, df_metrics in metrics_dict.items():
        one = summarize_metrics(df_metrics, model_name=model_name)
        rows.append(one)

    out = pd.concat(rows, axis=0).reset_index(drop=True)
    cols = ["model", "mean_smape", "median_smape", "mean_mase", "median_mase", "n_series"]
    out = out[cols].sort_values(["mean_mase", "mean_smape"]).reset_index(drop=True)
    return out


def count_series_wins(
    metrics_dict,
    metric="mase",
    id_col="unique_id",
):
    if metric not in ["mase", "smape"]:
        raise ValueError("metric must be 'mase' or 'smape'")

    merged = None

    for model_name, df_metrics in metrics_dict.items():
        one = df_metrics[[id_col, metric]].copy()
        one = one.rename(columns={metric: model_name})

        if merged is None:
            merged = one
        else:
            merged = merged.merge(one, on=id_col, how="inner", validate="one_to_one")

    model_cols = [c for c in merged.columns if c != id_col]

    merged["best_model"] = merged[model_cols].idxmin(axis=1)

    wins = (
        merged["best_model"]
        .value_counts(dropna=False)
        .rename_axis("model")
        .reset_index(name="n_wins")
    )

    wins["share_wins"] = wins["n_wins"] / merged[id_col].nunique()
    wins = wins.sort_values(["n_wins", "model"], ascending=[False, True]).reset_index(drop=True)

    return wins


def add_model_name(per_series_metrics_df, model_name):
    df = per_series_metrics_df.copy()
    df["model"] = model_name
    return df


def stack_metrics(metrics_dict):
    parts = []

    for model_name, df_metrics in metrics_dict.items():
        one = add_model_name(df_metrics, model_name)
        parts.append(one)

    out = pd.concat(parts, axis=0).reset_index(drop=True)
    return out


def compare_two_models_per_series(
    metrics_a,
    metrics_b,
    model_a="model_a",
    model_b="model_b",
    metric="mase",
    id_col="unique_id",
):
    a = metrics_a[[id_col, metric]].rename(columns={metric: f"{model_a}_{metric}"})
    b = metrics_b[[id_col, metric]].rename(columns={metric: f"{model_b}_{metric}"})

    out = a.merge(b, on=id_col, how="inner", validate="one_to_one")
    out["diff"] = out[f"{model_a}_{metric}"] - out[f"{model_b}_{metric}"]
    out["better_model"] = np.where(out["diff"] < 0, model_a, model_b)
    out["abs_diff"] = np.abs(out["diff"])
    return out