import pandas as pd
import pytest

from src.analyse import(
    calculate_returns,
    validate_data
)

from src.backtest import calculate_backtest_statistics

from src.mean_reversion_optimise import(
    evaluate_mean_reversion_period,
    evaluate_mean_reversion_parameters_on_period,
    select_top_mean_reversion_parameters,
    select_best_mean_reversion_parameters,
    evaluate_mean_reversion_candidates_on_period,
    run_mean_reversion_optimisation
)


def make_prepared_dataframe(prices):
    raw_df=pd.DataFrame({
        "Date":pd.date_range(start="2024-01-01",periods=len(prices),freq="D").strftime("%Y-%m-%d").tolist(),
        "Close":prices
    })

    df=validate_data(raw_df)
    df=calculate_returns(df)

    return df


def make_long_dataframe():
    prices=[
        100,102,99,103,98,104,100,105,101,106,
        102,107,101,108,103,109,104,110,105,111,
        106,112,107,113,106,114,108,115,109,116,
        110,117,109,118,111,119,112,120,113,121
    ]

    return make_prepared_dataframe(prices)


def test_evaluate_mean_reversion_period_returns_requested_rows():
    df=make_long_dataframe()

    result=evaluate_mean_reversion_period(df,10,20,3,-1.0,0.0,0.001)

    assert len(result)==10
    assert list(result.index)==list(range(10,20))


def test_evaluate_mean_reversion_period_resets_cumulative_values():
    df=make_long_dataframe()

    result=evaluate_mean_reversion_period(df,10,20,3,-1.0,0.0,0.001)

    expected_asset=(1+result["DailyReturn"].fillna(0)).cumprod()
    expected_strategy=(1+result["StrategyReturn"]).cumprod()
    expected_net=(1+result["NetStrategyReturn"]).cumprod()

    assert result["CumulativeValue"].tolist()==pytest.approx(expected_asset.tolist())
    assert result["StrategyCumulativeValue"].tolist()==pytest.approx(expected_strategy.tolist())
    assert result["NetStrategyCumulativeValue"].tolist()==pytest.approx(expected_net.tolist())


def test_evaluate_mean_reversion_period_preserves_previous_state():
    df=make_long_dataframe()

    full_result=evaluate_mean_reversion_period(df,0,20,3,-1.0,0.0,0.001)
    period_result=evaluate_mean_reversion_period(df,10,20,3,-1.0,0.0,0.001)

    assert period_result.iloc[0]["Position"]==full_result.loc[10,"Position"]


def test_evaluate_mean_reversion_period_does_not_modify_input():
    df=make_long_dataframe()

    original_df=df.copy(deep=True)

    evaluate_mean_reversion_period(df,10,20,3,-1.0,0.0,0.001)

    pd.testing.assert_frame_equal(df,original_df)


@pytest.mark.parametrize(
    "start_position,end_position",
    [
        (-1,10),
        (10,10),
        (15,10),
        (0,41)
    ]
)
def test_evaluate_mean_reversion_period_rejects_invalid_boundaries(start_position, end_position):
    df=make_long_dataframe()

    with pytest.raises(ValueError):
        evaluate_mean_reversion_period(df,start_position,end_position,3,-1.0,0.0,0.001)


def test_evaluate_mean_reversion_parameters_on_period():
    df=make_long_dataframe()

    result=evaluate_mean_reversion_parameters_on_period(df,0,20,[2,3],[-1.5,-1.0],[0.0,0.5],0.001)

    assert len(result)==8

    assert result.columns.tolist()==[
        "Lookback",
        "EntryThreshold",
        "ExitThreshold",
        "NetStrategyReturn"
    ]


def test_parameter_results_match_backtest_statistics():
    df=make_long_dataframe()

    result=evaluate_mean_reversion_parameters_on_period(df,0,20,[2],[-1.0],[0.0],0.001)

    period_result=evaluate_mean_reversion_period(df,0,20,2,-1.0,0.0,0.001)
    statistics=calculate_backtest_statistics(period_result)

    assert result.iloc[0]["NetStrategyReturn"]==pytest.approx(statistics["net_strategy_return"])


def test_parameter_evaluation_skips_invalid_threshold_combinations():
    df=make_long_dataframe()

    result=evaluate_mean_reversion_parameters_on_period(df,0,20,[2],[-1.0,0.5],[0.0],0.001)

    assert len(result)==1
    assert result.iloc[0]["EntryThreshold"]==pytest.approx(-1.0)
    assert result.iloc[0]["ExitThreshold"]==pytest.approx(0.0)


def test_select_top_mean_reversion_parameters():
    results=pd.DataFrame({
        "Lookback":[
            5,
            10,
            20
        ],
        "EntryThreshold":[
            -1.0,
            -1.5,
            -2.0
        ],
        "ExitThreshold":[
            0.0,
            0.5,
            0.0
        ],
        "NetStrategyReturn":[
            0.05,
            0.20,
            0.10
        ]
    })

    selected=select_top_mean_reversion_parameters(results,n=2)

    assert selected["Lookback"].tolist()==[
        10,
        20
    ]


def test_select_top_mean_reversion_parameters_does_not_modify_input():
    results=pd.DataFrame({
        "Lookback":[5,10,20],
        "EntryThreshold":[-1.0,-1.5,-2.0],
        "ExitThreshold":[0.0,0.5,0.0],
        "NetStrategyReturn":[0.05,0.20,0.10]
    })

    original_results=results.copy(deep=True)

    select_top_mean_reversion_parameters(results,n=2)

    pd.testing.assert_frame_equal(results,original_results)


def test_select_top_mean_reversion_parameters_rejects_empty_results():
    results=pd.DataFrame()

    with pytest.raises(ValueError,match="Cannot select parameters"):
        select_top_mean_reversion_parameters(results,n=1)


@pytest.mark.parametrize("n",[0,-1,-10])
def test_select_top_mean_reversion_parameters_rejects_invalid_n(n):
    results=pd.DataFrame({
        "Lookback":[2,3],
        "EntryThreshold":[-1.0,-1.5],
        "ExitThreshold":[0.0,0.0],
        "NetStrategyReturn":[0.1,0.2]
    })

    with pytest.raises(ValueError,match="n must be positive"):
        select_top_mean_reversion_parameters(results,n)


def test_select_top_mean_reversion_parameters_rejects_too_large_n():
    results=pd.DataFrame({
        "Lookback":[2,3],
        "EntryThreshold":[-1.0,-1.5],
        "ExitThreshold":[0.0,0.0],
        "NetStrategyReturn":[0.1,0.2]
    })

    with pytest.raises(ValueError,match="n cannot exceed"):
        select_top_mean_reversion_parameters(results,n=3)


def test_select_best_mean_reversion_parameters():
    results=pd.DataFrame({
        "Lookback":[
            5,
            10,
            20
        ],
        "EntryThreshold":[
            -1.0,
            -1.5,
            -2.0
        ],
        "ExitThreshold":[
            0.0,
            0.5,
            0.0
        ],
        "NetStrategyReturn":[
            0.05,
            0.20,
            0.10
        ]
    })

    best=select_best_mean_reversion_parameters(results)

    assert best["Lookback"]==10
    assert isinstance(best["Lookback"],int)
    assert best["EntryThreshold"]==pytest.approx(-1.5)
    assert best["ExitThreshold"]==pytest.approx(0.5)


def test_select_best_mean_reversion_parameters_rejects_empty_results():
    results=pd.DataFrame()

    with pytest.raises(ValueError,match="Cannot select parameters"):
        select_best_mean_reversion_parameters(results)


def test_evaluate_candidates_only_evaluates_exact_combinations():
    df=make_long_dataframe()

    candidates=pd.DataFrame({
        "Lookback":[
            2,
            4
        ],
        "EntryThreshold":[
            -1.0,
            -2.0
        ],
        "ExitThreshold":[
            0.0,
            0.5
        ],
        "NetStrategyReturn":[
            0.1,
            0.2
        ]
    })

    result=evaluate_mean_reversion_candidates_on_period(df,20,30,candidates,0.001)

    assert len(result)==2

    combinations=list(zip(
        result["Lookback"],
        result["EntryThreshold"],
        result["ExitThreshold"]
    ))

    assert combinations==[
        (2,-1.0,0.0),
        (4,-2.0,0.5)
    ]


def test_run_mean_reversion_optimisation():
    df=make_long_dataframe()

    result=run_mean_reversion_optimisation(df,[2,3,4],[-1.5,-1.0],[0.0,0.5],3,0.001)

    assert set(result.keys())=={
        "training_results",
        "candidate_parameters",
        "validation_results",
        "best_parameters",
        "test_results",
        "test_statistics"
    }

    assert len(result["training_results"])==12
    assert len(result["candidate_parameters"])==3
    assert len(result["validation_results"])==3
    assert len(result["test_results"])==8
    assert result["test_statistics"]["observations"]==8


def test_optimisation_best_parameters_come_from_candidates():
    df=make_long_dataframe()

    result=run_mean_reversion_optimisation(df,[2,3,4],[-1.5,-1.0],[0.0,0.5],3,0.001)

    best=result["best_parameters"]
    candidates=result["candidate_parameters"]

    matches=(
        (candidates["Lookback"]==best["Lookback"])
        & (candidates["EntryThreshold"]==best["EntryThreshold"])
        & (candidates["ExitThreshold"]==best["ExitThreshold"])
    )

    assert matches.any()


def test_run_mean_reversion_optimisation_does_not_modify_input():
    df=make_long_dataframe()

    original_df=df.copy(deep=True)

    run_mean_reversion_optimisation(df,[2,3],[-1.5,-1.0],[0.0,0.5],2,0.001)

    pd.testing.assert_frame_equal(df,original_df)