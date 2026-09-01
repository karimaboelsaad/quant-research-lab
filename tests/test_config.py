import copy

import pytest

from src.config import DEFAULT_CONFIG_PATH,PROJECT_ROOT,load_config,resolve_project_path,validate_config


def test_default_config_is_valid():
    config=load_config()

    assert config["data"]["multi_asset_tickers"]==["SPY","TLT","GLD"]
    assert config["pairs"]["selection_policy"]=="predefined_pair"
    assert config["portfolio"]["inverse_volatility_estimation"]=="pre_oos_frozen"


def test_resolve_project_path_uses_repository_root():
    assert resolve_project_path("data/example.csv")==PROJECT_ROOT/"data"/"example.csv"
    assert resolve_project_path(DEFAULT_CONFIG_PATH)==DEFAULT_CONFIG_PATH


def test_load_config_rejects_missing_file(tmp_path):
    with pytest.raises(ValueError,match="does not exist"):
        load_config(tmp_path/"missing.json")


def test_validate_config_rejects_non_adjusted_prices():
    config=copy.deepcopy(load_config())
    config["data"]["adjusted_prices"]=False

    with pytest.raises(ValueError,match="adjusted prices"):
        validate_config(config)


def test_validate_config_rejects_invalid_pair_grid():
    config=copy.deepcopy(load_config())
    config["pairs"]["exit_thresholds"]=[2.0]

    with pytest.raises(ValueError,match="pair exit threshold"):
        validate_config(config)


def test_validate_config_rejects_insufficient_training_history():
    config=copy.deepcopy(load_config())
    config["walk_forward"]["initial_train_size"]=100

    with pytest.raises(ValueError,match="largest strategy lookback"):
        validate_config(config)


def test_validate_config_rejects_negative_cost():
    config=copy.deepcopy(load_config())
    config["momentum"]["cost_rate"]=-0.001

    with pytest.raises(ValueError,match="cannot be negative"):
        validate_config(config)
