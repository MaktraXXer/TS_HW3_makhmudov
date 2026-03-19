import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

from statsmodels.tsa.stattools import acf
from datasetsforecast.m4 import M4


warnings.filterwarnings("ignore")


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_monthly_m4(data_dir: str = "data") -> pd.DataFrame:
    df, *_ = M4.load(directory=data_dir, group="Monthly")
    df = df.copy()
    df = df.sort_values(["unique_id", "ds"]).reset_index(drop=True)

    parts = []
    for uid, g in df.groupby("unique_id"):
        g = g.copy().reset_index(drop=True)
        g["ds"] = pd.date_range(start="2000-01-31", periods=len(g), freq="M")
        parts.append(g)

    df = pd.concat(parts, axis=0).reset_index(drop=True)
    return df


def get_series_lengths(df: pd.DataFrame) -> pd.DataFrame:
    lengths = (
        df.groupby("unique_id", as_index=False)
        .agg(length=("y", "size"))
        .sort_values("length")
        .reset_index(drop=True)
    )
    return lengths


def acf_lag_is_significant(
    y,
    lag: int = 12,
    alpha: float = 0.05,
) -> tuple[bool, float, float, float]:
    y = pd.Series(y).dropna().astype(float).values
    if len(y) <= lag + 1:
        return False, np.nan, np.nan, np.nan

    acf_vals, confint = acf(y, nlags=lag, alpha=alpha, fft=False)
    val = float(acf_vals[lag])
    low, high = confint[lag]
    is_sig = (low > 0) or (high < 0)
    return is_sig, val, float(low), float(high)


def build_acf_filter_table(
    df: pd.DataFrame,
    lag: int = 12,
    alpha: float = 0.05,
    min_len: int = 48,
) -> pd.DataFrame:
    rows = []

    for uid, g in df.groupby("unique_id"):
        is_sig, acf_val, low, high = acf_lag_is_significant(
            g["y"].values,
            lag=lag,
            alpha=alpha,
        )
        rows.append(
            {
                "unique_id": uid,
                "length": len(g),
                f"acf{lag}": acf_val,
                f"acf{lag}_low": low,
                f"acf{lag}_high": high,
                f"acf{lag}_sig": is_sig,
                "enough_len": len(g) >= min_len,
            }
        )

    return pd.DataFrame(rows)


def get_filtered_ids(
    acf_info: pd.DataFrame,
    lag: int = 12,
) -> list[str]:
    sig_col = f"acf{lag}_sig"
    ids = acf_info.loc[
        (acf_info["enough_len"]) & (acf_info[sig_col]),
        "unique_id",
    ].tolist()
    return ids


def sample_series_ids(
    filtered_ids: list[str],
    n_series: int = 200,
    seed: int = 42,
) -> list[str]:
    if len(filtered_ids) < n_series:
        raise ValueError(
            f"После фильтра осталось только {len(filtered_ids)} рядов, меньше чем n_series={n_series}"
        )

    rng = random.Random(seed)
    selected_ids = sorted(rng.sample(filtered_ids, n_series))
    return selected_ids


def select_series(
    df: pd.DataFrame,
    selected_ids: list[str],
) -> pd.DataFrame:
    df_selected = (
        df[df["unique_id"].isin(selected_ids)]
        .sort_values(["unique_id", "ds"])
        .reset_index(drop=True)
    )
    return df_selected


def select_info(
    acf_info: pd.DataFrame,
    selected_ids: list[str],
    lag: int = 12,
) -> pd.DataFrame:
    acf_col = f"acf{lag}"
    selected_info = (
        acf_info[acf_info["unique_id"].isin(selected_ids)]
        .sort_values([acf_col, "length"], ascending=[False, False])
        .reset_index(drop=True)
    )
    return selected_info


def save_selected_ids(
    selected_ids: list[str],
    selected_info: pd.DataFrame,
    out_dir: str = "artifacts",
) -> None:
    Path(out_dir).mkdir(exist_ok=True)

    pd.Series(selected_ids, name="unique_id").to_csv(
        f"{out_dir}/selected_ids_monthly_m4.csv",
        index=False,
    )
    selected_info.to_csv(
        f"{out_dir}/selected_ids_info.csv",
        index=False,
    )


def prepare_filtered_sample(
    data_dir: str = "data",
    lag: int = 12,
    alpha: float = 0.05,
    min_len: int = 48,
    n_series: int = 200,
    seed: int = 42,
    save: bool = True,
    out_dir: str = "artifacts",
) -> dict:
    set_seed(seed)

    df = load_monthly_m4(data_dir=data_dir)
    lengths = get_series_lengths(df)
    acf_info = build_acf_filter_table(
        df=df,
        lag=lag,
        alpha=alpha,
        min_len=min_len,
    )
    filtered_ids = get_filtered_ids(acf_info, lag=lag)
    selected_ids = sample_series_ids(
        filtered_ids=filtered_ids,
        n_series=n_series,
        seed=seed,
    )
    df_selected = select_series(df, selected_ids)
    selected_info = select_info(acf_info, selected_ids, lag=lag)

    if save:
        save_selected_ids(
            selected_ids=selected_ids,
            selected_info=selected_info,
            out_dir=out_dir,
        )

    return {
        "df": df,
        "lengths": lengths,
        "acf_info": acf_info,
        "filtered_ids": filtered_ids,
        "selected_ids": selected_ids,
        "df_selected": df_selected,
        "selected_info": selected_info,
    }


def _resolve_plot_ids(
    df: pd.DataFrame,
    uid=None,
    selected_ids: list[str] | None = None,
    n_random: int = 1,
    seed: int = 42,
) -> list[str]:
    all_ids = sorted(df["unique_id"].unique().tolist())

    if uid == "all":
        return selected_ids if selected_ids is not None else all_ids

    if uid == -1 or uid is None:
        base_ids = selected_ids if selected_ids is not None else all_ids
        rng = random.Random(seed)
        return rng.sample(base_ids, min(n_random, len(base_ids)))

    if isinstance(uid, (list, tuple, set, np.ndarray, pd.Series)):
        return list(uid)

    return [uid]


def plot_series_acf(
    df: pd.DataFrame,
    uid=None,
    selected_ids: list[str] | None = None,
    n_random: int = 1,
    seed: int = 42,
    lags: int = 24,
    max_plots: int | None = None,
) -> None:
    plot_ids = _resolve_plot_ids(
        df=df,
        uid=uid,
        selected_ids=selected_ids,
        n_random=n_random,
        seed=seed,
    )

    if max_plots is not None:
        plot_ids = plot_ids[:max_plots]

    if len(plot_ids) == 0:
        raise ValueError("Нет рядов для отрисовки")

    fig, axes = plt.subplots(len(plot_ids), 2, figsize=(14, 4 * len(plot_ids)))

    if len(plot_ids) == 1:
        axes = np.array([axes])

    for i, one_id in enumerate(plot_ids):
        g = df[df["unique_id"] == one_id].copy().sort_values("ds")

        axes[i, 0].plot(g["ds"], g["y"])
        axes[i, 0].set_title(f"{one_id} | len={len(g)}")

        sm.graphics.tsa.plot_acf(g["y"], lags=lags, ax=axes[i, 1])
        axes[i, 1].set_title(f"{one_id} | ACF")

    plt.tight_layout()
    plt.show()