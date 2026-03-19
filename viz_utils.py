import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _sort_df_for_plot(df, value_col, ascending=True):
    out = df.copy()
    if value_col in out.columns:
        out = out.sort_values(value_col, ascending=ascending).reset_index(drop=True)
    return out


def plot_metric_bar(
    df,
    x_col,
    y_col,
    title=None,
    ascending=True,
    rotation=45,
    figsize=(10, 4),
    annotate=True,
):
    plot_df = _sort_df_for_plot(df, y_col, ascending=ascending)

    plt.figure(figsize=figsize)
    bars = plt.bar(plot_df[x_col].astype(str), plot_df[y_col].values)
    plt.title(title if title is not None else y_col)
    plt.ylabel(y_col)
    plt.xticks(rotation=rotation)
    plt.tight_layout()

    if annotate:
        for bar, val in zip(bars, plot_df[y_col].values):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.3f}" if isinstance(val, (int, float, np.number)) else str(val),
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.show()


def plot_two_metric_bars(
    df,
    x_col,
    metric_1,
    metric_2,
    title_1=None,
    title_2=None,
    rotation=45,
    figsize=(14, 4),
    ascending_1=True,
    ascending_2=True,
):
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    df1 = _sort_df_for_plot(df, metric_1, ascending=ascending_1)
    df2 = _sort_df_for_plot(df, metric_2, ascending=ascending_2)

    bars1 = axes[0].bar(df1[x_col].astype(str), df1[metric_1].values)
    axes[0].set_title(title_1 if title_1 is not None else metric_1)
    axes[0].set_ylabel(metric_1)
    axes[0].tick_params(axis="x", rotation=rotation)

    bars2 = axes[1].bar(df2[x_col].astype(str), df2[metric_2].values)
    axes[1].set_title(title_2 if title_2 is not None else metric_2)
    axes[1].set_ylabel(metric_2)
    axes[1].tick_params(axis="x", rotation=rotation)

    for ax, bars, vals in [
        (axes[0], bars1, df1[metric_1].values),
        (axes[1], bars2, df2[metric_2].values),
    ]:
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.tight_layout()
    plt.show()


def plot_wins_bar(
    wins_df,
    model_col="model",
    wins_col="n_wins",
    title="Number of wins",
    rotation=45,
    figsize=(10, 4),
    annotate=True,
):
    plot_df = wins_df.sort_values(wins_col, ascending=False).reset_index(drop=True)

    plt.figure(figsize=figsize)
    bars = plt.bar(plot_df[model_col].astype(str), plot_df[wins_col].values)
    plt.title(title)
    plt.ylabel(wins_col)
    plt.xticks(rotation=rotation)
    plt.tight_layout()

    if annotate:
        for bar, val in zip(bars, plot_df[wins_col].values):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{int(val)}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.show()


def plot_share_wins_bar(
    wins_df,
    model_col="model",
    share_col="share_wins",
    title="Share of wins",
    rotation=45,
    figsize=(10, 4),
    annotate=True,
):
    plot_df = wins_df.sort_values(share_col, ascending=False).reset_index(drop=True)

    plt.figure(figsize=figsize)
    bars = plt.bar(plot_df[model_col].astype(str), plot_df[share_col].values)
    plt.title(title)
    plt.ylabel(share_col)
    plt.xticks(rotation=rotation)
    plt.tight_layout()

    if annotate:
        for bar, val in zip(bars, plot_df[share_col].values):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.2%}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.show()


def plot_metric_heatmap(
    df,
    value_col,
    index_col="history",
    columns_col="model_horizon",
    title=None,
    figsize=(6, 5),
    annotate=True,
    cmap="viridis",
    fmt=".3f",
):
    pivot = df.pivot(index=index_col, columns=columns_col, values=value_col)
    pivot = pivot.sort_index().sort_index(axis=1)

    plt.figure(figsize=figsize)
    plt.imshow(pivot.values, aspect="auto", cmap=cmap)

    plt.xticks(range(len(pivot.columns)), pivot.columns)
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.xlabel(columns_col)
    plt.ylabel(index_col)
    plt.title(title if title is not None else value_col)

    plt.colorbar()

    if annotate:
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.iloc[i, j]
                if pd.notna(val):
                    plt.text(j, i, format(val, fmt), ha="center", va="center", fontsize=9)

    plt.tight_layout()
    plt.show()


def plot_two_heatmaps(
    df,
    value_col_1,
    value_col_2,
    index_col="history",
    columns_col="model_horizon",
    title_1=None,
    title_2=None,
    figsize=(12, 5),
    annotate=True,
    cmap="viridis",
    fmt=".3f",
):
    p1 = df.pivot(index=index_col, columns=columns_col, values=value_col_1).sort_index().sort_index(axis=1)
    p2 = df.pivot(index=index_col, columns=columns_col, values=value_col_2).sort_index().sort_index(axis=1)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    im1 = axes[0].imshow(p1.values, aspect="auto", cmap=cmap)
    axes[0].set_xticks(range(len(p1.columns)))
    axes[0].set_xticklabels(p1.columns)
    axes[0].set_yticks(range(len(p1.index)))
    axes[0].set_yticklabels(p1.index)
    axes[0].set_xlabel(columns_col)
    axes[0].set_ylabel(index_col)
    axes[0].set_title(title_1 if title_1 is not None else value_col_1)

    if annotate:
        for i in range(p1.shape[0]):
            for j in range(p1.shape[1]):
                val = p1.iloc[i, j]
                if pd.notna(val):
                    axes[0].text(j, i, format(val, fmt), ha="center", va="center", fontsize=8)

    im2 = axes[1].imshow(p2.values, aspect="auto", cmap=cmap)
    axes[1].set_xticks(range(len(p2.columns)))
    axes[1].set_xticklabels(p2.columns)
    axes[1].set_yticks(range(len(p2.index)))
    axes[1].set_yticklabels(p2.index)
    axes[1].set_xlabel(columns_col)
    axes[1].set_ylabel(index_col)
    axes[1].set_title(title_2 if title_2 is not None else value_col_2)

    if annotate:
        for i in range(p2.shape[0]):
            for j in range(p2.shape[1]):
                val = p2.iloc[i, j]
                if pd.notna(val):
                    axes[1].text(j, i, format(val, fmt), ha="center", va="center", fontsize=8)

    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()


def plot_metric_boxplot(
    metrics_dict,
    metric="mase",
    figsize=(10, 5),
    rotation=45,
    title=None,
):
    labels = []
    values = []

    for model_name, df_metrics in metrics_dict.items():
        labels.append(model_name)
        values.append(df_metrics[metric].dropna().values)

    plt.figure(figsize=figsize)
    plt.boxplot(values, labels=labels)
    plt.xticks(rotation=rotation)
    plt.ylabel(metric)
    plt.title(title if title is not None else f"Boxplot of {metric}")
    plt.tight_layout()
    plt.show()


def plot_metric_histograms(
    metrics_dict,
    metric="mase",
    bins=20,
    ncols=2,
    figsize_per_plot=(5, 3),
):
    model_names = list(metrics_dict.keys())
    n = len(model_names)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize_per_plot[0] * ncols, figsize_per_plot[1] * nrows),
    )

    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = np.array([axes])
    elif ncols == 1:
        axes = np.array([[ax] for ax in axes])

    axes_flat = axes.flatten()

    for ax, model_name in zip(axes_flat, model_names):
        vals = metrics_dict[model_name][metric].dropna().values
        ax.hist(vals, bins=bins)
        ax.set_title(model_name)
        ax.set_xlabel(metric)

    for ax in axes_flat[len(model_names):]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def plot_forecast_comparison(
    train_df,
    truth_df,
    pred_dict,
    uid,
    id_col="unique_id",
    date_col="ds",
    target_col="y",
    figsize=(12, 5),
    title=None,
):
    g_train = train_df[train_df[id_col] == uid].sort_values(date_col).copy()
    g_truth = truth_df[truth_df[id_col] == uid].sort_values(date_col).copy()

    plt.figure(figsize=figsize)
    plt.plot(g_train[date_col], g_train[target_col], label="train")
    plt.plot(g_truth[date_col], g_truth[target_col], label="truth", linewidth=2)

    for model_name, pred_df in pred_dict.items():
        g_pred = pred_df[pred_df[id_col] == uid].sort_values(date_col).copy()
        plt.plot(g_pred[date_col], g_pred["y_pred"], label=model_name)

    plt.legend()
    plt.title(title if title is not None else f"Forecast comparison | {uid}")
    plt.tight_layout()
    plt.show()


def plot_forecast_comparison_many(
    train_df,
    truth_df,
    pred_dict,
    uids,
    id_col="unique_id",
    date_col="ds",
    target_col="y",
    figsize_per_plot=(12, 4),
):
    for uid in uids:
        plot_forecast_comparison(
            train_df=train_df,
            truth_df=truth_df,
            pred_dict=pred_dict,
            uid=uid,
            id_col=id_col,
            date_col=date_col,
            target_col=target_col,
            figsize=figsize_per_plot,
        )


def plot_stage1_summary(
    summary_df,
    wins_df=None,
    model_col="model",
    metric_1="mean_mase",
    metric_2="mean_smape",
):
    plot_two_metric_bars(
        df=summary_df,
        x_col=model_col,
        metric_1=metric_1,
        metric_2=metric_2,
        title_1=f"Stage 1: {metric_1}",
        title_2=f"Stage 1: {metric_2}",
        ascending_1=True,
        ascending_2=True,
    )

    if wins_df is not None:
        plot_wins_bar(
            wins_df=wins_df,
            model_col="model",
            wins_col="n_wins",
            title="Stage 1: wins",
        )


def plot_stage2_heatmaps_for_config(
    grid_df,
    config_name,
    metric_1="mean_mase",
    metric_2="mean_smape",
    config_col="config_name",
):
    one = grid_df[grid_df[config_col] == config_name].copy()

    plot_two_heatmaps(
        df=one,
        value_col_1=metric_1,
        value_col_2=metric_2,
        index_col="history",
        columns_col="model_horizon",
        title_1=f"{config_name}: {metric_1}",
        title_2=f"{config_name}: {metric_2}",
    )


def plot_stage3_summary(
    summary_df,
    wins_col="n_grid_wins",
    model_col="config_name",
    metric_1="avg_grid_mase",
    metric_2="avg_grid_smape",
):
    plot_two_metric_bars(
        df=summary_df,
        x_col=model_col,
        metric_1=metric_1,
        metric_2=metric_2,
        title_1=f"Stage 3: {metric_1}",
        title_2=f"Stage 3: {metric_2}",
        ascending_1=True,
        ascending_2=True,
    )

    if wins_col in summary_df.columns:
        plot_metric_bar(
            df=summary_df,
            x_col=model_col,
            y_col=wins_col,
            title="Stage 3: wins on grid",
            ascending=False,
        )


def plot_final_comparison(
    summary_df,
    wins_df=None,
    model_col="model",
    metric_1="mean_mase",
    metric_2="mean_smape",
):
    plot_two_metric_bars(
        df=summary_df,
        x_col=model_col,
        metric_1=metric_1,
        metric_2=metric_2,
        title_1=f"Final: {metric_1}",
        title_2=f"Final: {metric_2}",
        ascending_1=True,
        ascending_2=True,
    )

    if wins_df is not None:
        plot_wins_bar(
            wins_df=wins_df,
            model_col="model",
            wins_col="n_wins",
            title="Final: wins",
        )