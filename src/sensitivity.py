import numpy as np
import pandas as pd

from src.metrics import calculate_cumulative_value,calculate_performance_metrics


def calculate_frozen_cost_sensitivity(dynamic_results,cost_rates):
    if dynamic_results.empty:
        raise ValueError("Dynamic results cannot be empty.")

    required_columns=[
        "GrossDynamicPortfolioReturn",
        "DynamicUnderlyingTurnover",
        "DynamicTurnover"
    ]
    missing_columns=[column for column in required_columns if column not in dynamic_results.columns]

    if missing_columns:
        raise ValueError(f"Dynamic results are missing cost-sensitivity columns: {missing_columns}")

    if not cost_rates or any(not isinstance(cost_rate,(int,float)) or isinstance(cost_rate,bool) or cost_rate<0 for cost_rate in cost_rates):
        raise ValueError("Cost rates must contain non-negative numbers.")

    results=[]
    gross_returns=dynamic_results["GrossDynamicPortfolioReturn"].astype(float)
    underlying_turnover=dynamic_results["DynamicUnderlyingTurnover"].astype(float)
    overlay_turnover=dynamic_results["DynamicTurnover"].astype(float)

    for cost_rate in cost_rates:
        underlying_cost=underlying_turnover*cost_rate
        overlay_cost=overlay_turnover*cost_rate
        net_returns=gross_returns-underlying_cost-overlay_cost
        metrics=calculate_performance_metrics(net_returns)

        results.append({
            "System":"dynamic_portfolio",
            "CostRate":cost_rate,
            "CostBps":cost_rate*10000,
            "Observations":metrics["observations"],
            "GrossTotalReturn":calculate_cumulative_value(gross_returns).iloc[-1]-1,
            "NetTotalReturn":metrics["total_return"],
            "AnnualisedReturn":metrics["annualised_return"],
            "AnnualisedVolatility":metrics["annualised_volatility"],
            "SharpeRatio":metrics["sharpe_ratio"],
            "MaxDrawdown":metrics["max_drawdown"],
            "UnderlyingTurnover":underlying_turnover.sum(),
            "OverlayTurnover":overlay_turnover.sum(),
            "UnderlyingCost":underlying_cost.sum(),
            "OverlayCost":overlay_cost.sum(),
            "TotalCost":underlying_cost.sum()+overlay_cost.sum(),
            "EvaluationType":"frozen_policy_common_oos"
        })

    return pd.DataFrame(results)


def _build_window_row(window_df,asset,strategy,window_id,position_column,cointegration_p_value=np.nan):
    gross_returns=window_df["GrossReturn"].astype(float)
    net_returns=window_df["NetReturn"].astype(float)
    metrics=calculate_performance_metrics(net_returns)
    observations=len(window_df)

    return {
        "Asset":asset,
        "Strategy":strategy,
        "WindowID":window_id,
        "TestStart":window_df["Date"].iloc[0],
        "TestEnd":window_df["Date"].iloc[-1],
        "Observations":observations,
        "SelectedLookback":window_df["SelectedLookback"].iloc[0],
        "SelectedEntryThreshold":window_df["SelectedEntryThreshold"].iloc[0],
        "SelectedExitThreshold":window_df["SelectedExitThreshold"].iloc[0],
        "RegressionAlpha":window_df["RegressionAlpha"].iloc[0] if "RegressionAlpha" in window_df.columns else np.nan,
        "RegressionBeta":window_df["RegressionBeta"].iloc[0] if "RegressionBeta" in window_df.columns else np.nan,
        "PretestCointegrationPValue":cointegration_p_value,
        "GrossTotalReturn":calculate_cumulative_value(gross_returns).iloc[-1]-1,
        "NetTotalReturn":metrics["total_return"],
        "AnnualisedReturn":metrics["annualised_return"],
        "AnnualisedVolatility":metrics["annualised_volatility"],
        "SharpeRatio":metrics["sharpe_ratio"] if observations>=20 else np.nan,
        "Turnover":window_df["Turnover"].sum(),
        "InternalCost":window_df["InternalCost"].sum(),
        "ActiveFraction":(window_df[position_column]!=0).mean(),
        "EvaluationType":"walk_forward_test"
    }


def build_walk_forward_window_table(standalone_oos_results,pair_oos_results,pair_walk_forward_results=None):
    if standalone_oos_results.empty or pair_oos_results.empty:
        raise ValueError("Standalone and pair OOS results cannot be empty.")

    rows=[]

    for (asset,strategy,window_id),window_df in standalone_oos_results.groupby(["Asset","Strategy","WindowID"],sort=True):
        rows.append(_build_window_row(window_df,asset,strategy,window_id,"Position"))

    pair_diagnostics={}

    if pair_walk_forward_results is not None:
        for window_id,result in enumerate(pair_walk_forward_results["full_results"],start=1):
            pair_diagnostics[window_id]=result.get("pretest_cointegration",{}).get("p_value",np.nan)

    pair_asset=f"{pair_oos_results['AssetA'].iloc[0]}/{pair_oos_results['AssetB'].iloc[0]}"

    for window_id,window_df in pair_oos_results.groupby("WindowID",sort=True):
        rows.append(_build_window_row(window_df,pair_asset,"pairs",window_id,"SpreadPosition",pair_diagnostics.get(window_id,np.nan)))

    table=pd.DataFrame(rows).sort_values(["Strategy","Asset","WindowID"]).reset_index(drop=True)
    table["PositiveProfitShare"]=0.0
    table["BestWindowProfitConcentration"]=np.nan
    table["FractionPositiveWindows"]=np.nan
    table["MedianWindowNetReturn"]=np.nan

    for _,indices in table.groupby(["Asset","Strategy"]).groups.items():
        group_returns=table.loc[indices,"NetTotalReturn"]
        positive_returns=group_returns.clip(lower=0)
        total_positive=positive_returns.sum()
        shares=positive_returns/total_positive if total_positive>0 else positive_returns*0

        table.loc[indices,"PositiveProfitShare"]=shares
        table.loc[indices,"BestWindowProfitConcentration"]=shares.max() if total_positive>0 else np.nan
        table.loc[indices,"FractionPositiveWindows"]=(group_returns>0).mean()
        table.loc[indices,"MedianWindowNetReturn"]=group_returns.median()

    return table


def _parameter_key(strategy,lookback,entry_threshold,exit_threshold):
    if strategy=="momentum":
        return f"lookback={int(lookback)}"

    return f"lookback={int(lookback)},entry={float(entry_threshold):g},exit={float(exit_threshold):g}"


def build_parameter_stability_table(multi_asset_walk_forward_results,pair_walk_forward_results,pair_assets):
    if not multi_asset_walk_forward_results:
        raise ValueError("Multi-asset walk-forward results cannot be empty.")

    if not isinstance(pair_assets,list) or len(pair_assets)!=2:
        raise ValueError("Pair assets must contain two names.")

    rows=[]

    def add_strategy(asset,strategy,walk_forward_results):
        for window_id,result in enumerate(walk_forward_results["full_results"],start=1):
            validation_results=result["validation_results"].copy()
            validation_results["ValidationRank"]=validation_results["NetStrategyReturn"].rank(method="min",ascending=False).astype(int)

            if strategy=="momentum":
                best_lookback=result["best_lookback"]
                best_entry=np.nan
                best_exit=np.nan
            else:
                best_parameters=result["best_parameters"]
                best_lookback=best_parameters["Lookback"]
                best_entry=best_parameters["EntryThreshold"]
                best_exit=best_parameters["ExitThreshold"]

            for _,candidate in validation_results.iterrows():
                lookback=int(candidate["Lookback"])
                entry_threshold=candidate["EntryThreshold"] if "EntryThreshold" in candidate else np.nan
                exit_threshold=candidate["ExitThreshold"] if "ExitThreshold" in candidate else np.nan
                selected=lookback==best_lookback

                if strategy!="momentum":
                    selected=selected and np.isclose(entry_threshold,best_entry) and np.isclose(exit_threshold,best_exit)

                rows.append({
                    "Asset":asset,
                    "Strategy":strategy,
                    "WindowID":window_id,
                    "Lookback":lookback,
                    "EntryThreshold":entry_threshold,
                    "ExitThreshold":exit_threshold,
                    "ParameterKey":_parameter_key(strategy,lookback,entry_threshold,exit_threshold),
                    "ValidationNetReturn":candidate["NetStrategyReturn"],
                    "ValidationRank":candidate["ValidationRank"],
                    "Selected":selected
                })

    for asset,strategy_results in multi_asset_walk_forward_results.items():
        add_strategy(asset,"momentum",strategy_results["momentum"])
        add_strategy(asset,"mean_reversion",strategy_results["mean_reversion"])

    add_strategy(f"{pair_assets[0]}/{pair_assets[1]}","pairs",pair_walk_forward_results)

    table=pd.DataFrame(rows)
    group_columns=["Asset","Strategy","ParameterKey"]
    selection_counts=table.loc[table["Selected"]].groupby(group_columns).size().to_dict()
    window_counts=table.groupby(["Asset","Strategy"])["WindowID"].nunique().to_dict()

    table["SelectionCount"]=[selection_counts.get((row.Asset,row.Strategy,row.ParameterKey),0) for row in table.itertuples()]
    table["WindowCount"]=[window_counts[(row.Asset,row.Strategy)] for row in table.itertuples()]
    table["SelectionFrequency"]=table["SelectionCount"]/table["WindowCount"]
    table["MedianValidationNetReturn"]=table.groupby(group_columns)["ValidationNetReturn"].transform("median")
    table["MedianValidationRank"]=table.groupby(group_columns)["ValidationRank"].transform("median")
    table["EvaluationType"]="walk_forward_validation"

    return table.sort_values(["Strategy","Asset","WindowID","ValidationRank"]).reset_index(drop=True)
