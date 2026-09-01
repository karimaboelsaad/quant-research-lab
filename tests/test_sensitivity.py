import numpy as np
import pandas as pd

from src.sensitivity import build_parameter_stability_table,build_walk_forward_window_table,calculate_frozen_cost_sensitivity


def test_frozen_cost_sensitivity_reprices_one_unchanged_policy():
    dynamic_results=pd.DataFrame({
        "GrossDynamicPortfolioReturn":[0.02,-0.01,0.01],
        "DynamicUnderlyingTurnover":[1.0,0.5,0.0],
        "DynamicTurnover":[0.5,0.0,1.0]
    })

    results=calculate_frozen_cost_sensitivity(dynamic_results,[0.0,0.001,0.002])

    assert results["NetTotalReturn"].is_monotonic_decreasing
    assert results["UnderlyingTurnover"].nunique()==1
    assert results["OverlayTurnover"].nunique()==1
    assert results["CostBps"].tolist()==[0.0,10.0,20.0]


def test_walk_forward_window_table_reports_each_window_and_profit_concentration():
    dates=pd.date_range("2020-01-01",periods=4,freq="B")
    standalone=pd.DataFrame({
        "Date":dates,
        "Asset":["SPY"]*4,
        "Strategy":["momentum"]*4,
        "WindowID":[1,1,2,2],
        "GrossReturn":[0.02,0.01,-0.01,-0.01],
        "NetReturn":[0.019,0.009,-0.011,-0.011],
        "Turnover":[1.0,0.0,0.0,1.0],
        "InternalCost":[0.001,0.0,0.0,0.001],
        "Position":[1,1,0,0],
        "SelectedLookback":[20,20,60,60],
        "SelectedEntryThreshold":[np.nan]*4,
        "SelectedExitThreshold":[np.nan]*4
    })
    pair=pd.DataFrame({
        "Date":dates[:2],
        "AssetA":["KO"]*2,
        "AssetB":["PEP"]*2,
        "WindowID":[1,1],
        "GrossReturn":[0.01,0.01],
        "NetReturn":[0.009,0.009],
        "Turnover":[1.0,0.0],
        "InternalCost":[0.001,0.0],
        "SpreadPosition":[1,1],
        "SelectedLookback":[20,20],
        "SelectedEntryThreshold":[1.5,1.5],
        "SelectedExitThreshold":[0.5,0.5],
        "RegressionAlpha":[1.0,1.0],
        "RegressionBeta":[0.8,0.8]
    })
    pair_walk_forward={"full_results":[{"pretest_cointegration":{"p_value":0.04}}]}

    results=build_walk_forward_window_table(standalone,pair,pair_walk_forward)

    assert len(results)==3
    assert results.loc[results["Strategy"]=="momentum","FractionPositiveWindows"].eq(0.5).all()
    assert results.loc[(results["Strategy"]=="momentum")&(results["WindowID"]==1),"PositiveProfitShare"].iloc[0]==1.0
    assert results.loc[results["Strategy"]=="pairs","PretestCointegrationPValue"].iloc[0]==0.04


def _walk_forward_strategy(strategy):
    validation=pd.DataFrame({
        "Lookback":[20,60],
        "NetStrategyReturn":[0.10,0.05]
    })

    if strategy!="momentum":
        validation["EntryThreshold"]=[-1.0,-1.5] if strategy=="mean_reversion" else [1.0,1.5]
        validation["ExitThreshold"]=[0.0,0.5]

    result={"validation_results":validation}

    if strategy=="momentum":
        result["best_lookback"]=20
    else:
        result["best_parameters"]={"Lookback":20,"EntryThreshold":validation["EntryThreshold"].iloc[0],"ExitThreshold":0.0}

    return {"full_results":[result]}


def test_parameter_stability_counts_frozen_window_selections():
    multi={
        "SPY":{
            "momentum":_walk_forward_strategy("momentum"),
            "mean_reversion":_walk_forward_strategy("mean_reversion")
        }
    }
    pair=_walk_forward_strategy("pairs")

    results=build_parameter_stability_table(multi,pair,["KO","PEP"])

    selected=results.loc[results["Selected"]]
    assert len(selected)==3
    assert selected["SelectionFrequency"].eq(1.0).all()
    assert results["EvaluationType"].eq("walk_forward_validation").all()
