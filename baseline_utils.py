import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import Naive, SeasonalNaive, AutoETS, AutoTheta


def prepare_statsforecast_train(
    train_df,
    id_col="unique_id",
    date_col="ds",
    target_col="y",
):
    df = train_df[[id_col, date_col, target_col]].copy()
    df = df.sort_values([id_col, date_col]).reset_index(drop=True)
    return df


def prepare_truth_for_eval(
    truth_df,
    id_col="unique_id",
    date_col="ds",
    target_col="y",
):
    df = truth_df[[id_col, date_col, target_col]].copy()
    df = df.sort_values([id_col, date_col]).reset_index(drop=True)
    return df


def get_baseline_models(season_length):
    models = [
        Naive(),
        SeasonalNaive(season_length=season_length),
        AutoETS(season_length=season_length),
        AutoTheta(season_length=season_length),
    ]
    return models


def fit_predict_baselines(
    train_df,
    h,
    season_length,
    freq="M",
    id_col="unique_id",
    date_col="ds",
    target_col="y",
):
    train_sf = prepare_statsforecast_train(
        train_df=train_df,
        id_col=id_col,
        date_col=date_col,
        target_col=target_col,
    )

    models = get_baseline_models(season_length=season_length)

    sf = StatsForecast(
        models=models,
        freq=freq,
        n_jobs=-1,
    )

    forecasts = sf.forecast(df=train_sf, h=h)
    forecasts = forecasts.sort_values([id_col, date_col]).reset_index(drop=True)
    return forecasts


def reshape_one_baseline_pred(
    forecasts_df,
    model_name,
    id_col="unique_id",
    date_col="ds",
):
    out = forecasts_df[[id_col, date_col, model_name]].copy()
    out = out.rename(columns={model_name: "y_pred"})
    out = out.sort_values([id_col, date_col]).reset_index(drop=True)
    return out


def reshape_all_baseline_preds(
    forecasts_df,
    id_col="unique_id",
    date_col="ds",
):
    model_cols = [c for c in forecasts_df.columns if c not in [id_col, date_col]]
    pred_dict = {}

    for model_name in model_cols:
        pred_dict[model_name] = reshape_one_baseline_pred(
            forecasts_df=forecasts_df,
            model_name=model_name,
            id_col=id_col,
            date_col=date_col,
        )

    return pred_dict


def run_all_baselines(
    train_df,
    h,
    season_length,
    freq="M",
    id_col="unique_id",
    date_col="ds",
    target_col="y",
):
    forecasts_df = fit_predict_baselines(
        train_df=train_df,
        h=h,
        season_length=season_length,
        freq=freq,
        id_col=id_col,
        date_col=date_col,
        target_col=target_col,
    )

    pred_dict = reshape_all_baseline_preds(
        forecasts_df=forecasts_df,
        id_col=id_col,
        date_col=date_col,
    )

    return {
        "forecasts_df": forecasts_df,
        "pred_dict": pred_dict,
    }


def sanity_check_baseline_forecasts(
    forecasts_df,
    h,
    n_ids_expected=None,
    id_col="unique_id",
    date_col="ds",
):
    if n_ids_expected is not None:
        assert forecasts_df[id_col].nunique() == n_ids_expected

    sizes = forecasts_df.groupby(id_col).size()
    assert (sizes == h).all()

    assert forecasts_df[[id_col, date_col]].duplicated().sum() == 0

    print("baseline forecasts look ok")