import pandas as pd

from src.results import standardize_oos_results,standardize_pair_oos_results
from src.dynamic_portfolio import run_dynamic_portfolio,validate_dynamic_strategy_returns
from src.analyse import calculate_returns
from src.data import validate_pair_prices,validate_single_asset_prices
from src.mean_reversion_walk_forward import run_mean_reversion_walk_forward
from src.momentum_walk_forward import run_momentum_walk_forward
from src.pairs_walk_forward import run_pair_walk_forward
from src.metrics import calculate_cumulative_value
from src.portfolio import(
    apply_portfolio_transaction_costs,
    calculate_asset_returns,
    calculate_equal_weights,
    calculate_portfolio_returns,
    calculate_pre_oos_inverse_volatility_weights,
    calculate_rebalancing_turnover
)

def prepare_asset_data(price_panel,asset):
    required_columns=["Date",asset]

    missing_columns=[
        column for column in required_columns
        if column not in price_panel.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing asset price columns: {missing_columns}")

    asset_df=price_panel[required_columns].copy()
    asset_df=asset_df.rename(columns={asset:"Close"})

    asset_df=validate_single_asset_prices(asset_df)
    asset_df=calculate_returns(asset_df)

    return asset_df


def standardize_momentum_walk_forward_results(walk_forward_results,asset):
    full_results=walk_forward_results["full_results"]

    if not full_results:
        raise ValueError("Momentum walk-forward results cannot be empty.")

    standardized_periods=[]

    for window_id,result in enumerate(full_results,start=1):
        standardized_period=standardize_oos_results(result["test_results"],asset,"momentum",window_id,result["best_lookback"])

        standardized_periods.append(standardized_period)

    combined_results=pd.concat(standardized_periods,ignore_index=True)

    if combined_results["Date"].duplicated().any():
        raise ValueError("Momentum OOS windows contain duplicate dates.")

    if not combined_results["Date"].is_monotonic_increasing:
        raise ValueError("Momentum OOS dates must be chronological.")

    return combined_results


def standardize_mean_reversion_walk_forward_results(walk_forward_results,asset):
    full_results=walk_forward_results["full_results"]

    if not full_results:
        raise ValueError("Mean-reversion walk-forward results cannot be empty.")

    standardized_periods=[]

    for window_id,result in enumerate(full_results,start=1):
        best_parameters=result["best_parameters"]

        standardized_period=standardize_oos_results(result["test_results"],asset,"mean_reversion",window_id,best_parameters["Lookback"],best_parameters["EntryThreshold"],best_parameters["ExitThreshold"])

        standardized_periods.append(standardized_period)

    combined_results=pd.concat(standardized_periods,ignore_index=True)

    if combined_results["Date"].duplicated().any():
        raise ValueError("Mean-reversion OOS windows contain duplicate dates.")

    if not combined_results["Date"].is_monotonic_increasing:
        raise ValueError("Mean-reversion OOS dates must be chronological.")

    return combined_results


def align_oos_candidate_streams(oos_results,assets):
    if not oos_results:
        raise ValueError("OOS results cannot be empty.")

    if not assets:
        raise ValueError("Assets cannot be empty.")

    required_columns=[
        "Date",
        "Asset",
        "Strategy",
        "NetReturn",
        "EvaluationType"
    ]

    missing_columns=[
        column for column in required_columns
        if any(column not in result.columns for result in oos_results)
    ]

    if missing_columns:
        raise ValueError(f"Missing standardized OOS columns: {missing_columns}")

    combined_results=pd.concat([result[required_columns] for result in oos_results],ignore_index=True)

    if not (combined_results["EvaluationType"]=="walk_forward_test").all():
        raise ValueError("Every candidate stream must be a walk-forward test.")

    expected_streams={
        (asset,strategy)
        for asset in assets
        for strategy in ["momentum","mean_reversion"]
    }

    actual_streams=set(zip(combined_results["Asset"],combined_results["Strategy"]))

    if actual_streams!=expected_streams:
        raise ValueError("Every asset must have momentum and mean-reversion OOS results.")

    if combined_results.duplicated(["Date","Asset","Strategy"]).any():
        raise ValueError("Candidate streams contain duplicate dates.")

    expected_results_per_date=len(expected_streams)
    results_per_date=combined_results.groupby("Date").size()

    if not (results_per_date==expected_results_per_date).all():
        raise ValueError("Every candidate stream must have identical OOS dates.")

    aligned_results=combined_results.pivot(index="Date",columns=["Asset","Strategy"],values="NetReturn")

    strategy_names={
        "momentum":"Momentum",
        "mean_reversion":"MeanReversion"
    }

    aligned_results.columns=[
        f"{asset}_{strategy_names[strategy]}Return"
        for asset,strategy in aligned_results.columns
    ]

    aligned_results=aligned_results.reset_index()

    return_columns=[]

    for asset in assets:
        return_columns+=[
            f"{asset}_MomentumReturn",
            f"{asset}_MeanReversionReturn"
        ]

    aligned_results=aligned_results[["Date"]+return_columns]

    return validate_dynamic_strategy_returns(aligned_results,assets)


def run_asset_oos_strategies(price_panel,asset,config):
    asset_df=prepare_asset_data(price_panel,asset)

    walk_forward_config=config["walk_forward"]
    momentum_config=config["momentum"]
    mean_reversion_config=config["mean_reversion"]

    momentum_results=run_momentum_walk_forward(asset_df,momentum_config["lookbacks"],momentum_config["top_candidates"],momentum_config["cost_rate"],walk_forward_config["initial_train_size"],walk_forward_config["validation_size"],walk_forward_config["test_size"])

    mean_reversion_results=run_mean_reversion_walk_forward(asset_df,mean_reversion_config["lookbacks"],mean_reversion_config["entry_thresholds"],mean_reversion_config["exit_thresholds"],mean_reversion_config["top_candidates"],mean_reversion_config["cost_rate"],walk_forward_config["initial_train_size"],walk_forward_config["validation_size"],walk_forward_config["test_size"])

    momentum_oos=standardize_momentum_walk_forward_results(momentum_results,asset)
    mean_reversion_oos=standardize_mean_reversion_walk_forward_results(mean_reversion_results,asset)

    return {
        "momentum":momentum_oos,
        "mean_reversion":mean_reversion_oos,
        "momentum_walk_forward":momentum_results,
        "mean_reversion_walk_forward":mean_reversion_results
    }


def run_multi_asset_oos_strategies(price_panel,assets,config):
    if not assets:
        raise ValueError("Assets cannot be empty.")

    if len(assets)!=len(set(assets)):
        raise ValueError("Assets must be unique.")

    oos_results=[]
    walk_forward_results={}

    for asset in assets:
        asset_results=run_asset_oos_strategies(price_panel,asset,config)

        oos_results.append(asset_results["momentum"])
        oos_results.append(asset_results["mean_reversion"])
        walk_forward_results[asset]={
            "momentum":asset_results["momentum_walk_forward"],
            "mean_reversion":asset_results["mean_reversion_walk_forward"]
        }

    standalone_oos_results=pd.concat(oos_results,ignore_index=True)
    candidate_returns=align_oos_candidate_streams(oos_results,assets)

    return {
        "standalone_oos_results":standalone_oos_results,
        "candidate_returns":candidate_returns,
        "walk_forward_results":walk_forward_results
    }


def run_dynamic_oos_portfolio(standalone_oos_results,assets,config):
    candidate_returns=align_oos_candidate_streams([standalone_oos_results],assets)

    required_detail_columns=["GrossReturn","InternalCost","Turnover"]
    missing_detail_columns=[column for column in required_detail_columns if column not in standalone_oos_results.columns]

    if missing_detail_columns:
        raise ValueError(f"Standalone OOS results are missing dynamic detail columns: {missing_detail_columns}")

    candidate_details=standalone_oos_results.pivot(index="Date",columns=["Asset","Strategy"],values=required_detail_columns)

    strategy_names={
        "momentum":"Momentum",
        "mean_reversion":"MeanReversion"
    }

    detail_names={
        "GrossReturn":"GrossReturn",
        "InternalCost":"InternalCost",
        "Turnover":"Turnover"
    }

    candidate_details.columns=[
        f"{asset}_{strategy_names[strategy]}{detail_names[detail]}"
        for detail,asset,strategy in candidate_details.columns
    ]

    candidate_details=candidate_details.reset_index()

    dynamic_input=candidate_returns.merge(candidate_details,on="Date",validate="one_to_one")

    dynamic_config=config["dynamic"]

    dynamic_results=run_dynamic_portfolio(dynamic_input,assets,dynamic_config["score_lookback"],dynamic_config["overlay_cost_rate"])

    dynamic_results["DynamicUnderlyingCost"]=0.0
    dynamic_results["DynamicUnderlyingTurnover"]=0.0
    dynamic_results["GrossDynamicPortfolioReturn"]=0.0

    for asset in assets:
        for strategy in ["Momentum","MeanReversion"]:
            weight_column=f"{asset}_{strategy}Weight"
            cost_column=f"{asset}_{strategy}InternalCost"
            turnover_column=f"{asset}_{strategy}Turnover"
            gross_return_column=f"{asset}_{strategy}GrossReturn"

            dynamic_results["DynamicUnderlyingCost"]+=dynamic_results[weight_column]*dynamic_results[cost_column]
            dynamic_results["DynamicUnderlyingTurnover"]+=dynamic_results[weight_column]*dynamic_results[turnover_column]
            dynamic_results["GrossDynamicPortfolioReturn"]+=dynamic_results[weight_column]*dynamic_results[gross_return_column]

    dynamic_results["DynamicReturnAfterUnderlyingCost"]=dynamic_results["DynamicPortfolioReturn"]
    dynamic_results["DynamicOverlayCost"]=dynamic_results["DynamicTransactionCost"]

    return dynamic_results



def create_dynamic_comparison_periods(dynamic_results,score_lookback):
    if dynamic_results.empty:
        raise ValueError("Dynamic results cannot be empty.")

    if "Date" not in dynamic_results.columns:
        raise ValueError("Dynamic results must contain Date.")

    if not isinstance(score_lookback,int) or isinstance(score_lookback,bool) or not 1<=score_lookback<len(dynamic_results):
        raise ValueError("Score lookback must leave at least one comparison row.")

    full_results=dynamic_results.copy().reset_index(drop=True)

    full_results["ScoreWarmup"]=full_results.index<score_lookback
    full_results["CommonComparison"]=~full_results["ScoreWarmup"]

    common_results=full_results.loc[full_results["CommonComparison"]].copy()
    common_results=common_results.reset_index(drop=True)

    return {
        "full_results":full_results,
        "common_results":common_results,
        "common_start_date":common_results["Date"].iloc[0]
    }


def run_static_oos_baselines(price_panel,assets,oos_dates,common_start_date,config):
    if not assets:
        raise ValueError("Assets cannot be empty.")

    required_columns=["Date"]+assets
    missing_columns=[column for column in required_columns if column not in price_panel.columns]

    if missing_columns:
        raise ValueError(f"Missing baseline price columns: {missing_columns}")

    oos_dates=pd.Series(pd.to_datetime(oos_dates,errors="coerce")).reset_index(drop=True)

    if oos_dates.empty or oos_dates.isna().any():
        raise ValueError("OOS dates must be valid and non-empty.")

    if oos_dates.duplicated().any() or not oos_dates.is_monotonic_increasing:
        raise ValueError("OOS dates must be unique and chronological.")

    portfolio_df=calculate_asset_returns(price_panel[required_columns].copy())

    if not oos_dates.isin(portfolio_df["Date"]).all():
        raise ValueError("Every OOS date must exist in the price panel.")

    oos_start_position=int(portfolio_df.index[portfolio_df["Date"]==oos_dates.iloc[0]][0])

    oos_df=portfolio_df.loc[portfolio_df["Date"].isin(oos_dates)].copy()
    oos_df=oos_df.reset_index(drop=True)

    if oos_df["Date"].tolist()!=oos_dates.tolist():
        raise ValueError("Baseline and strategy OOS dates must be identical.")

    common_start_date=pd.Timestamp(common_start_date)

    if common_start_date not in oos_dates.tolist():
        raise ValueError("Common comparison start date must exist in the OOS period.")

    equal_weights=calculate_equal_weights(portfolio_df)
    inverse_volatility_weights=calculate_pre_oos_inverse_volatility_weights(portfolio_df,oos_start_position)

    cost_rate=config["portfolio"]["cost_rate"]

    equal_weight_results=calculate_portfolio_returns(oos_df,equal_weights)
    equal_weight_results=calculate_rebalancing_turnover(equal_weight_results,equal_weights)
    equal_weight_results=apply_portfolio_transaction_costs(equal_weight_results,cost_rate)

    inverse_volatility_results=calculate_portfolio_returns(oos_df,inverse_volatility_weights)
    inverse_volatility_results=calculate_rebalancing_turnover(inverse_volatility_results,inverse_volatility_weights)
    inverse_volatility_results=apply_portfolio_transaction_costs(inverse_volatility_results,cost_rate)

    equal_weight_common=equal_weight_results.loc[equal_weight_results["Date"]>=common_start_date].copy().reset_index(drop=True)
    inverse_volatility_common=inverse_volatility_results.loc[inverse_volatility_results["Date"]>=common_start_date].copy().reset_index(drop=True)

    for common_results in [equal_weight_common,inverse_volatility_common]:
        common_results["PortfolioCumulativeValue"]=calculate_cumulative_value(common_results["PortfolioReturn"])
        common_results["NetPortfolioCumulativeValue"]=calculate_cumulative_value(common_results["NetPortfolioReturn"])

    return {
        "equal_weight_weights":equal_weights,
        "inverse_volatility_weights":inverse_volatility_weights,
        "equal_weight_full_results":equal_weight_results,
        "inverse_volatility_full_results":inverse_volatility_results,
        "equal_weight_common_results":equal_weight_common,
        "inverse_volatility_common_results":inverse_volatility_common
    }


def standardize_pair_walk_forward_results(walk_forward_results,asset_a,asset_b):
    full_results=walk_forward_results["full_results"]

    if not full_results:
        raise ValueError("Pair walk-forward results cannot be empty.")

    standardized_periods=[]

    for window_id,result in enumerate(full_results,start=1):
        best_parameters=result["best_parameters"]
        standardized_period=standardize_pair_oos_results(result["test_results"],asset_a,asset_b,window_id,best_parameters["Lookback"],best_parameters["EntryThreshold"],best_parameters["ExitThreshold"])
        standardized_periods.append(standardized_period)

    combined_results=pd.concat(standardized_periods,ignore_index=True)

    if combined_results["Date"].duplicated().any():
        raise ValueError("Pair OOS windows contain duplicate dates.")

    if not combined_results["Date"].is_monotonic_increasing:
        raise ValueError("Pair OOS dates must be chronological.")

    return combined_results


def run_pair_oos_strategy(pair_panel,pair_assets,config):
    if not isinstance(pair_assets,list) or len(pair_assets)!=2 or len(set(pair_assets))!=2:
        raise ValueError("Pair assets must contain exactly two unique names.")

    pair_df=validate_pair_prices(pair_panel)
    walk_forward_config=config["walk_forward"]
    pair_config=config["pairs"]

    walk_forward_results=run_pair_walk_forward(pair_df,pair_config["lookbacks"],pair_config["entry_thresholds"],pair_config["exit_thresholds"],pair_config["top_candidates"],pair_config["cost_rate"],walk_forward_config["initial_train_size"],walk_forward_config["validation_size"],walk_forward_config["test_size"])

    oos_results=standardize_pair_walk_forward_results(walk_forward_results,pair_assets[0],pair_assets[1])

    return {
        "walk_forward_results":walk_forward_results,
        "oos_results":oos_results
    }


def run_complete_oos_integration(price_panel,pair_panel,config):
    assets=config["data"]["multi_asset_tickers"]
    pair_assets=config["data"]["pair_tickers"]

    multi_asset_results=run_multi_asset_oos_strategies(price_panel,assets,config)
    standalone_oos_results=multi_asset_results["standalone_oos_results"]
    candidate_returns=multi_asset_results["candidate_returns"]
    multi_asset_walk_forward_results=multi_asset_results["walk_forward_results"]

    dynamic_results=run_dynamic_oos_portfolio(standalone_oos_results,assets,config)
    comparison_periods=create_dynamic_comparison_periods(dynamic_results,config["dynamic"]["score_lookback"])
    common_start_date=comparison_periods["common_start_date"]

    standalone_common_results=standalone_oos_results.loc[standalone_oos_results["Date"]>=common_start_date].copy().reset_index(drop=True)
    candidate_common_returns=candidate_returns.loc[candidate_returns["Date"]>=common_start_date].copy().reset_index(drop=True)

    baselines=run_static_oos_baselines(price_panel,assets,candidate_returns["Date"],common_start_date,config)
    pair_results=run_pair_oos_strategy(pair_panel,pair_assets,config)

    return {
        "standalone_oos_results":standalone_oos_results,
        "standalone_common_results":standalone_common_results,
        "candidate_returns":candidate_returns,
        "candidate_common_returns":candidate_common_returns,
        "multi_asset_walk_forward_results":multi_asset_walk_forward_results,
        "dynamic_full_results":comparison_periods["full_results"],
        "dynamic_common_results":comparison_periods["common_results"],
        "common_start_date":common_start_date,
        "baselines":baselines,
        "pair_walk_forward_results":pair_results["walk_forward_results"],
        "pair_oos_results":pair_results["oos_results"]
    }
