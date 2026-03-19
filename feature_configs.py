from copy import deepcopy


def get_base_config():
    return {
        "name": "lags_only",
        "use_lags": True,
        "lags": [1, 2, 3, 6],
        "use_seasonal_lags": False,
        "seasonal_lags": [],
        "use_calendar": False,
        "calendar_features": [],
        "use_fourier": False,
        "fourier_period": 12,
        "fourier_order": 0,
    }


def make_lags_only_config():
    return get_base_config()


def make_lags_seasonal_config():
    cfg = deepcopy(get_base_config())
    cfg["name"] = "lags_seasonal"
    cfg["use_seasonal_lags"] = True
    cfg["seasonal_lags"] = [12, 24]
    return cfg


def make_lags_calendar_config():
    cfg = deepcopy(get_base_config())
    cfg["name"] = "lags_calendar"
    cfg["use_calendar"] = True
    cfg["calendar_features"] = ["month", "quarter"]
    return cfg


def make_lags_fourier_config():
    cfg = deepcopy(get_base_config())
    cfg["name"] = "lags_fourier"
    cfg["use_fourier"] = True
    cfg["fourier_period"] = 12
    cfg["fourier_order"] = 2
    return cfg


def make_lags_seasonal_calendar_config():
    cfg = deepcopy(get_base_config())
    cfg["name"] = "lags_seasonal_calendar"
    cfg["use_seasonal_lags"] = True
    cfg["seasonal_lags"] = [12, 24]
    cfg["use_calendar"] = True
    cfg["calendar_features"] = ["month", "quarter"]
    return cfg


def make_lags_seasonal_fourier_config():
    cfg = deepcopy(get_base_config())
    cfg["name"] = "lags_seasonal_fourier"
    cfg["use_seasonal_lags"] = True
    cfg["seasonal_lags"] = [12, 24]
    cfg["use_fourier"] = True
    cfg["fourier_period"] = 12
    cfg["fourier_order"] = 2
    return cfg


def get_all_feature_configs():
    return [
        make_lags_only_config(),
        make_lags_seasonal_config(),
        make_lags_calendar_config(),
        make_lags_fourier_config(),
        make_lags_seasonal_calendar_config(),
        make_lags_seasonal_fourier_config(),
    ]


def get_feature_config_by_name(name):
    for cfg in get_all_feature_configs():
        if cfg["name"] == name:
            return cfg
    raise ValueError(f"Unknown config name: {name}")


def get_feature_config_names():
    return [cfg["name"] for cfg in get_all_feature_configs()]