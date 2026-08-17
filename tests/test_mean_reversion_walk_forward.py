import pandas as pd
import pytest

from src.analyse import(
    calculate_returns,
    validate_data
)

from src.mean_reversion_walk_forward import(
    run_mean_reversion_walk_forward_window,
    run_mean_reversion_walk_forward
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


def test_run_mean_reversion_walk_forward_window():
    df=make_long_dataframe()

    window={
        "train_end":20,
        "validation_end":25,
        "test_end":30
    }

    result=run_mean_reversion_walk_forward_window(df,window,[2,3,4],[-1.5,-1.0],[0.0,0.5],3,0.001)

    assert set(result.keys())=={
        "training_results",
        "validation_results",
        "candidate_parameters",
        "best_parameters",
        "test_results",
        "test_statistics"
    }

    assert len(result["training_results"])==12
    assert len(result["candidate_parameters"])==3
    assert len(result["validation_results"])==3
    assert len(result["test_results"])==5

    assert list(result["test_results"].index)==[
        25,
        26,
        27,
        28,
        29
    ]

    assert result["test_statistics"]["observations"]==5


def test_walk_forward_best_parameters_come_from_candidates():
    df=make_long_dataframe()

    window={
        "train_end":20,
        "validation_end":25,
        "test_end":30
    }

    result=run_mean_reversion_walk_forward_window(df,window,[2,3,4],[-1.5,-1.0],[0.0,0.5],3,0.001)

    best=result["best_parameters"]
    candidates=result["candidate_parameters"]

    matches=(
        (candidates["Lookback"]==best["Lookback"])
        & (candidates["EntryThreshold"]==best["EntryThreshold"])
        & (candidates["ExitThreshold"]==best["ExitThreshold"])
    )

    assert matches.any()


def test_run_mean_reversion_walk_forward_window_does_not_modify_input():
    df=make_long_dataframe()

    original_df=df.copy(deep=True)

    window={
        "train_end":20,
        "validation_end":25,
        "test_end":30
    }

    run_mean_reversion_walk_forward_window(df,window,[2,3],[-1.5,-1.0],[0.0,0.5],2,0.001)

    pd.testing.assert_frame_equal(df,original_df)


def test_run_mean_reversion_walk_forward():
    df=make_long_dataframe()

    result=run_mean_reversion_walk_forward(df,[2,3,4],[-1.5,-1.0],[0.0,0.5],3,0.001,20,5,5)

    assert set(result.keys())=={
        "windows",
        "full_results",
        "combined_test_results",
        "statistics"
    }

    assert len(result["windows"])==3
    assert len(result["full_results"])==3

    combined=result["combined_test_results"]

    assert len(combined)==15
    assert list(combined.index)==list(range(25,40))
    assert result["statistics"]["observations"]==15


def test_mean_reversion_walk_forward_combines_test_periods_in_order():
    df=make_long_dataframe()

    result=run_mean_reversion_walk_forward(df,[2,3],[-1.5,-1.0],[0.0,0.5],2,0.001,20,5,5)

    combined=result["combined_test_results"]

    assert combined.index.is_monotonic_increasing
    assert combined.index.is_unique


def test_mean_reversion_walk_forward_recalculates_cumulative_values():
    df=make_long_dataframe()

    result=run_mean_reversion_walk_forward(df,[2,3],[-1.5,-1.0],[0.0,0.5],2,0.001,20,5,5)

    combined=result["combined_test_results"]

    expected_asset=(1+combined["DailyReturn"].fillna(0)).cumprod()
    expected_strategy=(1+combined["StrategyReturn"]).cumprod()
    expected_net=(1+combined["NetStrategyReturn"]).cumprod()

    assert combined["CumulativeValue"].tolist()==pytest.approx(expected_asset.tolist())
    assert combined["StrategyCumulativeValue"].tolist()==pytest.approx(expected_strategy.tolist())
    assert combined["NetStrategyCumulativeValue"].tolist()==pytest.approx(expected_net.tolist())


def test_mean_reversion_walk_forward_cumulative_values_do_not_reset():
    df=make_long_dataframe()

    result=run_mean_reversion_walk_forward(df,[2,3],[-1.5,-1.0],[0.0,0.5],2,0.001,20,5,5)

    combined=result["combined_test_results"]

    first_window_end=combined.iloc[4]["NetStrategyCumulativeValue"]
    second_window_start=combined.iloc[5]["NetStrategyCumulativeValue"]

    expected_second_start=first_window_end*(1+combined.iloc[5]["NetStrategyReturn"])

    assert second_window_start==pytest.approx(expected_second_start)


def test_mean_reversion_walk_forward_returns_risk_statistics():
    df=make_long_dataframe()

    result=run_mean_reversion_walk_forward(df,[2,3],[-1.5,-1.0],[0.0,0.5],2,0.001,20,5,5)

    required_statistics=[
        "net_strategy_return",
        "annualised_return",
        "annualised_volatility",
        "sharpe_ratio",
        "max_drawdown"
    ]

    for statistic in required_statistics:
        assert statistic in result["statistics"]


def test_each_walk_forward_window_has_selected_parameters():
    df=make_long_dataframe()

    result=run_mean_reversion_walk_forward(df,[2,3,4],[-1.5,-1.0],[0.0,0.5],3,0.001,20,5,5)

    for window_result in result["full_results"]:
        best=window_result["best_parameters"]
        candidates=window_result["candidate_parameters"]

        matches=(
            (candidates["Lookback"]==best["Lookback"])
            & (candidates["EntryThreshold"]==best["EntryThreshold"])
            & (candidates["ExitThreshold"]==best["ExitThreshold"])
        )

        assert matches.any()


def test_mean_reversion_walk_forward_does_not_modify_input():
    df=make_long_dataframe()

    original_df=df.copy(deep=True)

    run_mean_reversion_walk_forward(df,[2,3],[-1.5,-1.0],[0.0,0.5],2,0.001,20,5,5)

    pd.testing.assert_frame_equal(df,original_df)