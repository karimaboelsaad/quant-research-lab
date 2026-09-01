import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT=Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH=PROJECT_ROOT/"config"/"default.json"


def resolve_project_path(path):
    path=Path(path)

    if path.is_absolute():
        return path

    return PROJECT_ROOT/path


def _validate_positive_integer(value,name):
    if not isinstance(value,int) or isinstance(value,bool) or value<=0:
        raise ValueError(f"{name} must be a positive integer.")


def _validate_number_list(values,name,minimum=None,strict_minimum=False):
    if not isinstance(values,list) or not values:
        raise ValueError(f"{name} must be a non-empty list.")

    if any(not isinstance(value,(int,float)) or isinstance(value,bool) for value in values):
        raise ValueError(f"{name} must contain only numbers.")

    if minimum is not None:
        invalid=any(value<=minimum for value in values) if strict_minimum else any(value<minimum for value in values)

        if invalid:
            comparison="greater than" if strict_minimum else "at least"
            raise ValueError(f"{name} values must be {comparison} {minimum}.")


def _validate_cost(value,name):
    if not isinstance(value,(int,float)) or isinstance(value,bool) or value<0:
        raise ValueError(f"{name} cannot be negative.")


def validate_config(config):
    if not isinstance(config,dict):
        raise ValueError("Configuration must be a dictionary.")

    required_sections=["project","data","walk_forward","momentum","mean_reversion","pairs","portfolio","dynamic","execution","robustness","reporting"]

    for section in required_sections:
        if section not in config or not isinstance(config[section],dict):
            raise ValueError(f"Configuration must contain a {section} section.")

    project=config["project"]
    _validate_positive_integer(project.get("annualisation_periods"),"annualisation_periods")

    if not isinstance(project.get("random_seed"),int) or isinstance(project.get("random_seed"),bool):
        raise ValueError("random_seed must be an integer.")

    if not project.get("primary_hypothesis"):
        raise ValueError("primary_hypothesis cannot be empty.")

    data=config["data"]

    try:
        start_date=pd.Timestamp(data.get("start_date"))
        end_date=pd.Timestamp(data.get("end_date"))
    except (TypeError,ValueError):
        raise ValueError("Data dates must be valid.") from None

    if start_date>=end_date:
        raise ValueError("Data start_date must be earlier than end_date.")

    multi_asset_tickers=data.get("multi_asset_tickers")
    pair_tickers=data.get("pair_tickers")

    if not isinstance(multi_asset_tickers,list) or len(multi_asset_tickers)<2 or len(set(multi_asset_tickers))!=len(multi_asset_tickers):
        raise ValueError("multi_asset_tickers must contain at least two unique tickers.")

    if not isinstance(pair_tickers,list) or len(pair_tickers)!=2 or len(set(pair_tickers))!=2:
        raise ValueError("pair_tickers must contain exactly two unique tickers.")

    if data.get("single_asset_ticker") not in multi_asset_tickers:
        raise ValueError("single_asset_ticker must be one of multi_asset_tickers.")

    if data.get("adjusted_prices") is not True:
        raise ValueError("The final experiment requires adjusted prices.")

    if data.get("calendar_policy")!="intersection":
        raise ValueError("calendar_policy must be intersection.")

    for path_name in ["single_asset_path","multi_asset_path","pair_path"]:
        if not isinstance(data.get(path_name),str) or not data[path_name]:
            raise ValueError(f"{path_name} must be a non-empty path.")

    walk_forward=config["walk_forward"]

    for size_name in ["initial_train_size","validation_size","test_size"]:
        _validate_positive_integer(walk_forward.get(size_name),size_name)

    if not isinstance(walk_forward.get("include_partial_final_window"),bool):
        raise ValueError("include_partial_final_window must be boolean.")

    momentum=config["momentum"]
    mean_reversion=config["mean_reversion"]
    pairs=config["pairs"]

    _validate_number_list(momentum.get("lookbacks"),"momentum lookbacks",0,True)
    _validate_number_list(mean_reversion.get("lookbacks"),"mean-reversion lookbacks",1,True)
    _validate_number_list(pairs.get("lookbacks"),"pair lookbacks",1,True)
    _validate_number_list(mean_reversion.get("entry_thresholds"),"mean-reversion entry_thresholds")
    _validate_number_list(mean_reversion.get("exit_thresholds"),"mean-reversion exit_thresholds")
    _validate_number_list(pairs.get("entry_thresholds"),"pair entry_thresholds",0,True)
    _validate_number_list(pairs.get("exit_thresholds"),"pair exit_thresholds",0)

    if any(entry>=exit for entry in mean_reversion["entry_thresholds"] for exit in mean_reversion["exit_thresholds"]):
        raise ValueError("Every mean-reversion entry threshold must be below every exit threshold.")

    if any(exit>=entry for entry in pairs["entry_thresholds"] for exit in pairs["exit_thresholds"]):
        raise ValueError("Every pair exit threshold must be below every entry threshold.")

    candidate_counts={
        "momentum":len(momentum["lookbacks"]),
        "mean_reversion":len(mean_reversion["lookbacks"])*len(mean_reversion["entry_thresholds"])*len(mean_reversion["exit_thresholds"]),
        "pairs":len(pairs["lookbacks"])*len(pairs["entry_thresholds"])*len(pairs["exit_thresholds"])
    }

    for section_name,section in [("momentum",momentum),("mean_reversion",mean_reversion),("pairs",pairs)]:
        _validate_positive_integer(section.get("top_candidates"),f"{section_name} top_candidates")

        if section["top_candidates"]>candidate_counts[section_name]:
            raise ValueError(f"{section_name} top_candidates exceeds its parameter grid.")

        _validate_cost(section.get("cost_rate"),f"{section_name} cost_rate")

    largest_lookback=max(momentum["lookbacks"]+mean_reversion["lookbacks"]+pairs["lookbacks"])

    if walk_forward["initial_train_size"]<=largest_lookback:
        raise ValueError("initial_train_size must exceed the largest strategy lookback.")

    if pairs.get("selection_policy")!="predefined_pair":
        raise ValueError("pairs selection_policy must be predefined_pair.")

    portfolio=config["portfolio"]
    _validate_cost(portfolio.get("cost_rate"),"portfolio cost_rate")

    if portfolio.get("inverse_volatility_estimation")!="pre_oos_frozen":
        raise ValueError("inverse-volatility weights must be estimated pre-OOS and frozen.")

    dynamic=config["dynamic"]
    _validate_positive_integer(dynamic.get("score_lookback"),"dynamic score_lookback")
    _validate_cost(dynamic.get("overlay_cost_rate"),"dynamic overlay_cost_rate")

    robustness=config["robustness"]
    _validate_positive_integer(robustness.get("bootstrap_resamples"),"bootstrap_resamples")
    _validate_number_list(robustness.get("cost_rates"),"robustness cost_rates",0)

    confidence_level=robustness.get("confidence_level")

    if not isinstance(confidence_level,(int,float)) or isinstance(confidence_level,bool) or not 0<confidence_level<1:
        raise ValueError("confidence_level must be between 0 and 1.")

    _validate_positive_integer(config["reporting"].get("minimum_oos_observations"),"minimum_oos_observations")

    return config


def load_config(path=DEFAULT_CONFIG_PATH):
    path=resolve_project_path(path)

    try:
        with path.open("r",encoding="utf-8") as file:
            config=json.load(file)
    except FileNotFoundError:
        raise ValueError(f"Configuration file does not exist: {path}") from None
    except json.JSONDecodeError as error:
        raise ValueError(f"Configuration file is invalid JSON: {error}") from None

    return validate_config(config)
