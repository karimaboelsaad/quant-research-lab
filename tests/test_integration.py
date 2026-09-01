import pandas as pd
import numpy as np
import pytest

from src.integration import(
    align_oos_candidate_streams,
    create_dynamic_comparison_periods,
    prepare_asset_data,
    run_asset_oos_strategies,
    run_complete_oos_integration,
    run_dynamic_oos_portfolio,
    run_multi_asset_oos_strategies,
    run_pair_oos_strategy,
    run_static_oos_baselines,
    standardize_mean_reversion_walk_forward_results,
    standardize_momentum_walk_forward_results
)

from src.pairs import calculate_pair_regression

def test_standardize_momentum_walk_forward_results():
    first_test_period=pd.DataFrame({
        "Date":["2020-01-02"],
        "StrategyReturn":[0.01],
        "TransactionCost":[0.001],
        "NetStrategyReturn":[0.009],
        "Turnover":[1.0],
        "Position":[1]
    })

    second_test_period=pd.DataFrame({
        "Date":["2020-01-03"],
        "StrategyReturn":[-0.01],
        "TransactionCost":[0.001],
        "NetStrategyReturn":[-0.011],
        "Turnover":[1.0],
        "Position":[0]
    })

    walk_forward_results={
        "full_results":[
            {
                "best_lookback":20,
                "test_results":first_test_period
            },
            {
                "best_lookback":60,
                "test_results":second_test_period
            }
        ]
    }

    result=standardize_momentum_walk_forward_results(walk_forward_results,"SPY")

    assert len(result)==2
    assert result["Date"].is_monotonic_increasing
    assert result["Asset"].tolist()==["SPY","SPY"]
    assert result["Strategy"].tolist()==["momentum","momentum"]
    assert result["WindowID"].tolist()==[1,2]
    assert result["SelectedLookback"].tolist()==[20,60]
    assert result["NetReturn"].tolist()==[0.009,-0.011]
    assert result["EvaluationType"].tolist()==[
        "walk_forward_test",
        "walk_forward_test"
    ]


def test_standardize_mean_reversion_walk_forward_results():
    test_period=pd.DataFrame({
        "Date":["2020-01-02"],
        "StrategyReturn":[0.01],
        "TransactionCost":[0.001],
        "NetStrategyReturn":[0.009],
        "Turnover":[1.0],
        "Position":[1]
    })

    walk_forward_results={
        "full_results":[
            {
                "best_parameters":{
                    "Lookback":20,
                    "EntryThreshold":-1.5,
                    "ExitThreshold":0.0
                },
                "test_results":test_period
            }
        ]
    }

    result=standardize_mean_reversion_walk_forward_results(walk_forward_results,"SPY")

    assert len(result)==1
    assert result["Asset"].tolist()==["SPY"]
    assert result["Strategy"].tolist()==["mean_reversion"]
    assert result["WindowID"].tolist()==[1]
    assert result["SelectedLookback"].tolist()==[20]
    assert result["SelectedEntryThreshold"].tolist()==[-1.5]
    assert result["SelectedExitThreshold"].tolist()==[0.0]
    assert result["EvaluationType"].tolist()==["walk_forward_test"]


def test_align_oos_candidate_streams():
    dates=["2020-01-02","2020-01-03"]

    return_values={
        ("SPY","momentum"):[0.01,0.02],
        ("SPY","mean_reversion"):[0.03,0.04],
        ("TLT","momentum"):[0.05,0.06],
        ("TLT","mean_reversion"):[0.07,0.08]
    }

    oos_results=[]

    for asset_strategy,returns in return_values.items():
        asset,strategy=asset_strategy

        stream=pd.DataFrame({
            "Date":dates,
            "Asset":[asset,asset],
            "Strategy":[strategy,strategy],
            "NetReturn":returns,
            "EvaluationType":["walk_forward_test","walk_forward_test"]
        })

        oos_results.append(stream)

    result=align_oos_candidate_streams(oos_results,["SPY","TLT"])

    assert result.columns.tolist()==[
        "Date",
        "SPY_MomentumReturn",
        "SPY_MeanReversionReturn",
        "TLT_MomentumReturn",
        "TLT_MeanReversionReturn"
    ]

    assert result["SPY_MomentumReturn"].tolist()==[0.01,0.02]
    assert result["SPY_MeanReversionReturn"].tolist()==[0.03,0.04]
    assert result["TLT_MomentumReturn"].tolist()==[0.05,0.06]
    assert result["TLT_MeanReversionReturn"].tolist()==[0.07,0.08]


def test_prepare_asset_data():
    price_panel=pd.DataFrame({
        "Date":["2020-01-02","2020-01-03","2020-01-06"],
        "SPY":[100.0,110.0,121.0],
        "TLT":[50.0,51.0,52.0]
    })

    result=prepare_asset_data(price_panel,"SPY")

    assert result.columns.tolist()==[
        "Date",
        "Close",
        "DailyReturn",
        "CumulativeValue"
    ]

    assert result["Close"].tolist()==[100.0,110.0,121.0]
    assert pd.isna(result["DailyReturn"].iloc[0])
    assert result["DailyReturn"].iloc[1:].round(6).tolist()==[0.1,0.1]
    assert result["CumulativeValue"].round(6).tolist()==[1.0,1.1,1.21]


def test_run_asset_oos_strategies_with_synthetic_prices():
    rows=50

    price_panel=pd.DataFrame({
        "Date":pd.date_range("2020-01-01",periods=rows,freq="B"),
        "SPY":100+np.arange(rows)*0.2+np.sin(np.arange(rows))*2
    })

    config={
        "walk_forward":{
            "initial_train_size":20,
            "validation_size":10,
            "test_size":10
        },
        "momentum":{
            "lookbacks":[2,3],
            "top_candidates":1,
            "cost_rate":0.001
        },
        "mean_reversion":{
            "lookbacks":[3,5],
            "entry_thresholds":[-1.0],
            "exit_thresholds":[0.0],
            "top_candidates":1,
            "cost_rate":0.001
        }
    }

    result=run_asset_oos_strategies(price_panel,"SPY",config)

    momentum=result["momentum"]
    mean_reversion=result["mean_reversion"]

    assert len(momentum)==20
    assert len(mean_reversion)==20
    assert momentum["Date"].equals(mean_reversion["Date"])
    assert momentum["WindowID"].unique().tolist()==[1,2]
    assert mean_reversion["WindowID"].unique().tolist()==[1,2]
    assert (momentum["Asset"]=="SPY").all()
    assert (mean_reversion["Asset"]=="SPY").all()
    assert (momentum["EvaluationType"]=="walk_forward_test").all()
    assert (mean_reversion["EvaluationType"]=="walk_forward_test").all()



def test_run_multi_asset_oos_strategies_with_synthetic_prices():
    rows=40
    steps=np.arange(rows)

    price_panel=pd.DataFrame({
        "Date":pd.date_range("2020-01-01",periods=rows,freq="B"),
        "SPY":100+steps*0.2+np.sin(steps)*2,
        "TLT":80+steps*0.1+np.cos(steps)*1.5
    })

    config={
        "walk_forward":{
            "initial_train_size":20,
            "validation_size":10,
            "test_size":10
        },
        "momentum":{
            "lookbacks":[2,3],
            "top_candidates":1,
            "cost_rate":0.001
        },
        "mean_reversion":{
            "lookbacks":[3,5],
            "entry_thresholds":[-1.0],
            "exit_thresholds":[0.0],
            "top_candidates":1,
            "cost_rate":0.001
        },
        "dynamic":{
            "score_lookback":3,
            "overlay_cost_rate":0.001
        },
        "portfolio":{
            "cost_rate":0.001
        }
    }

    result=run_multi_asset_oos_strategies(price_panel,["SPY","TLT"],config)

    standalone_results=result["standalone_oos_results"]
    candidate_returns=result["candidate_returns"]

    assert len(standalone_results)==40
    assert len(candidate_returns)==10

    assert set(zip(standalone_results["Asset"],standalone_results["Strategy"]))=={
        ("SPY","momentum"),
        ("SPY","mean_reversion"),
        ("TLT","momentum"),
        ("TLT","mean_reversion")
    }

    assert candidate_returns.columns.tolist()==[
        "Date",
        "SPY_MomentumReturn",
        "SPY_MeanReversionReturn",
        "TLT_MomentumReturn",
        "TLT_MeanReversionReturn"
    ]

    assert (standalone_results["EvaluationType"]=="walk_forward_test").all()
    assert candidate_returns["Date"].is_unique
    assert candidate_returns["Date"].is_monotonic_increasing

    dynamic_results=run_dynamic_oos_portfolio(standalone_results,["SPY","TLT"],config)

    assert len(dynamic_results)==10
    assert (dynamic_results["DynamicUnderlyingCost"]>=0).all()
    assert (dynamic_results["DynamicOverlayCost"]>=0).all()

    np.testing.assert_allclose(
        dynamic_results["DynamicReturnAfterUnderlyingCost"],
        dynamic_results["DynamicPortfolioReturn"]
    )

    np.testing.assert_allclose(
        dynamic_results["GrossDynamicPortfolioReturn"]
        -dynamic_results["DynamicUnderlyingCost"]
        -dynamic_results["DynamicOverlayCost"],
        dynamic_results["NetDynamicPortfolioReturn"]
    )

    assert dynamic_results["DynamicGrossExposure"].iloc[:3].tolist()==[0.0,0.0,0.0]

    comparison_periods=create_dynamic_comparison_periods(dynamic_results,config["dynamic"]["score_lookback"])
    baselines=run_static_oos_baselines(price_panel,["SPY","TLT"],candidate_returns["Date"],comparison_periods["common_start_date"],config)

    assert baselines["equal_weight_weights"]=={
        "SPYReturn":0.5,
        "TLTReturn":0.5
    }
    assert sum(baselines["inverse_volatility_weights"].values())==pytest.approx(1.0)
    assert len(baselines["equal_weight_full_results"])==10
    assert len(baselines["inverse_volatility_full_results"])==10
    assert len(baselines["equal_weight_common_results"])==7
    assert len(baselines["inverse_volatility_common_results"])==7
    assert baselines["equal_weight_full_results"]["Turnover"].iloc[0]==pytest.approx(1.0)
    assert baselines["inverse_volatility_full_results"]["Turnover"].iloc[0]==pytest.approx(1.0)
    assert baselines["equal_weight_common_results"]["Date"].iloc[0]==comparison_periods["common_start_date"]
    assert baselines["inverse_volatility_common_results"]["Date"].iloc[0]==comparison_periods["common_start_date"]


def test_create_dynamic_comparison_periods_keeps_post_warmup_cash():
    dynamic_results=pd.DataFrame({
        "Date":pd.date_range("2020-01-01",periods=5,freq="B"),
        "DynamicGrossExposure":[0.0,0.0,0.0,0.0,1.0]
    })

    periods=create_dynamic_comparison_periods(dynamic_results,3)

    full_results=periods["full_results"]
    common_results=periods["common_results"]

    assert full_results["ScoreWarmup"].tolist()==[True,True,True,False,False]
    assert full_results["CommonComparison"].tolist()==[False,False,False,True,True]
    assert len(common_results)==2
    assert common_results["DynamicGrossExposure"].tolist()==[0.0,1.0]
    assert periods["common_start_date"]==pd.Timestamp("2020-01-06")


def test_align_oos_candidate_streams_rejects_non_oos_provenance():
    streams=[]

    for asset in ["SPY","TLT"]:
        for strategy in ["momentum","mean_reversion"]:
            evaluation_type="manual" if asset=="TLT" and strategy=="momentum" else "walk_forward_test"

            streams.append(pd.DataFrame({
                "Date":pd.date_range("2020-01-01",periods=2,freq="B"),
                "Asset":[asset,asset],
                "Strategy":[strategy,strategy],
                "NetReturn":[0.0,0.01],
                "EvaluationType":[evaluation_type,evaluation_type]
            }))

    with pytest.raises(ValueError,match="walk-forward test"):
        align_oos_candidate_streams(streams,["SPY","TLT"])


def test_align_oos_candidate_streams_rejects_mismatched_dates():
    streams=[]

    for asset in ["SPY","TLT"]:
        for strategy in ["momentum","mean_reversion"]:
            dates=pd.date_range("2020-01-01",periods=2,freq="B")

            if asset=="TLT" and strategy=="mean_reversion":
                dates=dates[:1]

            streams.append(pd.DataFrame({
                "Date":dates,
                "Asset":[asset]*len(dates),
                "Strategy":[strategy]*len(dates),
                "NetReturn":[0.0]*len(dates),
                "EvaluationType":["walk_forward_test"]*len(dates)
            }))

    with pytest.raises(ValueError,match="identical OOS dates"):
        align_oos_candidate_streams(streams,["SPY","TLT"])


def test_static_inverse_volatility_weights_ignore_oos_price_changes():
    rows=20
    steps=np.arange(rows)
    dates=pd.date_range("2020-01-01",periods=rows,freq="B")

    price_panel=pd.DataFrame({
        "Date":dates,
        "SPY":100+steps+np.sin(steps),
        "TLT":80+steps*0.5+np.cos(steps)
    })

    changed_panel=price_panel.copy()
    changed_panel.loc[15:,"SPY"]=[500.0,300.0,700.0,250.0,800.0]
    changed_panel.loc[15:,"TLT"]=[40.0,120.0,35.0,130.0,30.0]

    oos_dates=dates[15:]
    config={"portfolio":{"cost_rate":0.001}}

    original=run_static_oos_baselines(price_panel,["SPY","TLT"],oos_dates,oos_dates[1],config)
    changed=run_static_oos_baselines(changed_panel,["SPY","TLT"],oos_dates,oos_dates[1],config)

    assert changed["inverse_volatility_weights"]==pytest.approx(original["inverse_volatility_weights"])


def test_run_pair_oos_strategy_uses_real_walk_forward_regressions():
    rows=60
    steps=np.arange(rows)
    dates=pd.date_range("2020-01-01",periods=rows,freq="B")
    close_b=50+steps*0.15+np.sin(steps/3)

    pair_panel=pd.DataFrame({
        "Date":dates,
        "CloseA":5+1.2*close_b+1.5*np.sin(steps/2)+0.5*np.cos(steps/5),
        "CloseB":close_b
    })

    config={
        "walk_forward":{
            "initial_train_size":30,
            "validation_size":10,
            "test_size":10
        },
        "pairs":{
            "lookbacks":[3,5],
            "entry_thresholds":[1.0],
            "exit_thresholds":[0.5],
            "top_candidates":1,
            "cost_rate":0.001
        }
    }

    result=run_pair_oos_strategy(pair_panel,["KO","PEP"],config)["oos_results"]
    first_regression=calculate_pair_regression(pair_panel.iloc[:40])
    second_regression=calculate_pair_regression(pair_panel.iloc[:50])

    assert result["Date"].tolist()==dates[40:].tolist()
    assert result["WindowID"].unique().tolist()==[1,2]
    assert result.loc[result["WindowID"]==1,"RegressionAlpha"].unique().tolist()==pytest.approx([first_regression["alpha"]])
    assert result.loc[result["WindowID"]==1,"RegressionBeta"].unique().tolist()==pytest.approx([first_regression["beta"]])
    assert result.loc[result["WindowID"]==2,"RegressionAlpha"].unique().tolist()==pytest.approx([second_regression["alpha"]])
    assert result.loc[result["WindowID"]==2,"RegressionBeta"].unique().tolist()==pytest.approx([second_regression["beta"]])
    assert (result["EvaluationType"]=="walk_forward_test").all()


def test_run_complete_oos_integration_with_synthetic_prices():
    rows=60
    steps=np.arange(rows)
    dates=pd.date_range("2020-01-01",periods=rows,freq="B")

    price_panel=pd.DataFrame({
        "Date":dates,
        "SPY":100+steps*0.2+np.sin(steps)*2,
        "TLT":80+steps*0.1+np.cos(steps)*1.5
    })

    close_b=50+steps*0.15+np.sin(steps/3)
    pair_panel=pd.DataFrame({
        "Date":dates,
        "CloseA":5+1.2*close_b+1.5*np.sin(steps/2)+0.5*np.cos(steps/5),
        "CloseB":close_b
    })

    config={
        "data":{
            "multi_asset_tickers":["SPY","TLT"],
            "pair_tickers":["KO","PEP"]
        },
        "walk_forward":{
            "initial_train_size":30,
            "validation_size":10,
            "test_size":10
        },
        "momentum":{
            "lookbacks":[2,3],
            "top_candidates":1,
            "cost_rate":0.001
        },
        "mean_reversion":{
            "lookbacks":[3,5],
            "entry_thresholds":[-1.0],
            "exit_thresholds":[0.0],
            "top_candidates":1,
            "cost_rate":0.001
        },
        "pairs":{
            "lookbacks":[3,5],
            "entry_thresholds":[1.0],
            "exit_thresholds":[0.5],
            "top_candidates":1,
            "cost_rate":0.001
        },
        "dynamic":{
            "score_lookback":3,
            "overlay_cost_rate":0.001
        },
        "portfolio":{
            "cost_rate":0.001
        }
    }

    result=run_complete_oos_integration(price_panel,pair_panel,config)

    assert len(result["candidate_returns"])==20
    assert len(result["candidate_common_returns"])==17
    assert len(result["dynamic_full_results"])==20
    assert len(result["dynamic_common_results"])==17
    assert len(result["pair_oos_results"])==20
    assert result["common_start_date"]==result["candidate_returns"]["Date"].iloc[3]
    assert (result["standalone_oos_results"]["EvaluationType"]=="walk_forward_test").all()
    assert (result["pair_oos_results"]["EvaluationType"]=="walk_forward_test").all()
    assert result["standalone_common_results"]["Date"].min()==result["common_start_date"]
    assert result["baselines"]["equal_weight_common_results"]["Date"].iloc[0]==result["common_start_date"]
    assert result["baselines"]["inverse_volatility_common_results"]["Date"].iloc[0]==result["common_start_date"]
