import numpy as np
import pandas as pd


def check_series_length(df, h=12, L=36, id_col="unique_id"):
    lengths = df.groupby(id_col).size().rename("length").reset_index()
    lengths["enough_for_holdout"] = lengths["length"] >= h + 1
    lengths["enough_for_ml_tail"] = lengths["length"] >= L + h
    return lengths


def split_one_series(g, h=12, L=36, date_col="ds", target_col="y"):
    g = g.sort_values(date_col).reset_index(drop=True).copy()
    m = len(g)

    if m < L + h:
        raise ValueError(f"Ряд слишком короткий: len={m}, нужно минимум {L+h}")

    train_block = g.iloc[: m - h].copy()
    holdout_block = g.iloc[m - h :].copy()

    tail = g.iloc[m - (L + h) :].copy()
    holdout_fit = tail.copy()
    holdout_pred = tail.copy()
    holdout_pred[target_col] = holdout_pred[target_col].astype(float)
    holdout_pred.iloc[L:, holdout_pred.columns.get_loc(target_col)] = np.nan
    truth = g.iloc[m - h :].copy()

    return {
        "train_block": train_block,
        "holdout_block": holdout_block,
        "holdout_fit": holdout_fit,
        "holdout_pred": holdout_pred,
        "truth": truth,
    }


def make_splits_for_all_series(
    df,
    h=12,
    L=36,
    id_col="unique_id",
    date_col="ds",
    target_col="y",
):
    train_parts = []
    holdout_parts = []
    holdout_fit_parts = []
    holdout_pred_parts = []
    truth_parts = []

    for uid, g in df.groupby(id_col):
        parts = split_one_series(
            g,
            h=h,
            L=L,
            date_col=date_col,
            target_col=target_col,
        )
        train_parts.append(parts["train_block"])
        holdout_parts.append(parts["holdout_block"])
        holdout_fit_parts.append(parts["holdout_fit"])
        holdout_pred_parts.append(parts["holdout_pred"])
        truth_parts.append(parts["truth"])

    train_df = pd.concat(train_parts, axis=0).reset_index(drop=True)
    holdout_df = pd.concat(holdout_parts, axis=0).reset_index(drop=True)
    holdout_fit_df = pd.concat(holdout_fit_parts, axis=0).reset_index(drop=True)
    holdout_pred_df = pd.concat(holdout_pred_parts, axis=0).reset_index(drop=True)
    truth_df = pd.concat(truth_parts, axis=0).reset_index(drop=True)

    return {
        "train_df": train_df,
        "holdout_df": holdout_df,
        "holdout_fit_df": holdout_fit_df,
        "holdout_pred_df": holdout_pred_df,
        "truth_df": truth_df,
    }


def sanity_check_splits(
    splits,
    df,
    h=12,
    L=36,
    id_col="unique_id",
    target_col="y",
):
    train_df = splits["train_df"]
    holdout_df = splits["holdout_df"]
    holdout_fit_df = splits["holdout_fit_df"]
    holdout_pred_df = splits["holdout_pred_df"]
    truth_df = splits["truth_df"]

    n_ids = df[id_col].nunique()

    assert train_df[id_col].nunique() == n_ids
    assert holdout_df[id_col].nunique() == n_ids
    assert holdout_fit_df[id_col].nunique() == n_ids
    assert holdout_pred_df[id_col].nunique() == n_ids
    assert truth_df[id_col].nunique() == n_ids

    holdout_sizes = holdout_df.groupby(id_col).size()
    truth_sizes = truth_df.groupby(id_col).size()
    tail_sizes = holdout_fit_df.groupby(id_col).size()

    assert (holdout_sizes == h).all()
    assert (truth_sizes == h).all()
    assert (tail_sizes == L + h).all()

    for uid, g in holdout_pred_df.groupby(id_col):
        y = g[target_col].values
        assert np.isnan(y[-h:]).all()
        assert pd.notna(pd.Series(y[:L])).all()

    print("splits look ok")


def inspect_one_series_split(
    df,
    uid,
    h=12,
    L=36,
    id_col="unique_id",
    date_col="ds",
    target_col="y",
):
    g = df[df[id_col] == uid].sort_values(date_col).reset_index(drop=True).copy()
    parts = split_one_series(g, h=h, L=L, date_col=date_col, target_col=target_col)

    print("uid:", uid)
    print("full len:", len(g))
    print("train len:", len(parts["train_block"]))
    print("holdout len:", len(parts["holdout_block"]))
    print("holdout_fit len:", len(parts["holdout_fit"]))
    print("holdout_pred len:", len(parts["holdout_pred"]))
    print("truth len:", len(parts["truth"]))

    return parts