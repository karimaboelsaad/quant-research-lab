import pandas as pd
import pytest

from src.analyse import(
    calculate_returns,
    validate_data
)

from src.walk_forward import generate_walk_forward_windows

from src.momentum_walk_forward import(
    run_momentum_walk_forward_window,
    run_momentum_walk_forward
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
        100,102,101,104,103,106,105,108,107,110,
        109,112,111,114,113,116,115,118,117,120,
        119,122,121,124,123,126,125,128,127,130,
        129,132,131,134,133,136,135,138,137,140
    ]

    return make_prepared_dataframe(prices)


def test_generate_walk_forward_windows():
    windows=generate_walk_forward_windows(number_of_rows=1000,initial_train_size=400,validation_size=100,test_size=100)

    assert len(windows)==5

    assert windows[0]=={
        "train_end":400,
        "validation_end":500,
        "test_end":600
    }

    assert windows[-1]=={
        "train_end":800,
        "validation_end":900,
        "test_end":1000
    }


def test_generate_walk_forward_windows_expands_training():
    windows=generate_walk_forward_windows(number_of_rows=30,initial_train_size=10,validation_size=5,test_size=5)

    assert windows==[
        {
            "train_end":10,
            "validation_end":15,
            "test_end":20
        },
        {
            "train_end":15,
            "validation_end":20,
            "test_end":25
        },
        {
            "train_end":20,
            "validation_end":25,
            "test_end":30
        }
    ]


@pytest.mark.parametrize(
    "number_of_rows,initial_train_size,validation_size,test_size",
    [
        (0,10,5,5),
        (-1,10,5,5),
        (30,0,5,5),
        (30,-1,5,5),
        (30,10,0,5),
        (30,10,-1,5),
        (30,10,5,0),
        (30,10,5,-1)
    ]
)
def test_generate_walk_forward_windows_rejects_invalid_sizes(number_of_rows, initial_train_size, validation_size, test_size):
    with pytest.raises(ValueError):
        generate_walk_forward_windows(number_of_rows,initial_train_size,validation_size,test_size)


def test_generate_walk_forward_windows_rejects_insufficient_rows():
    with pytest.raises(ValueError,match="Not enough rows"):
        generate_walk_forward_windows(number_of_rows=15,initial_train_size=10,validation_size=5,test_size=5)


def test_run_momentum_walk_forward_window():
    df=make_long_dataframe()

    window={
        "train_end":20,
        "validation_end":25,
        "test_end":30
    }

    result=run_momentum_walk_forward_window(df,window,lookbacks=[1,2,3,4],n=2,cost_rate=0.001)

    assert set(result.keys())=={
        "training_results",
        "validation_results",
        "candidate_lookbacks",
        "best_lookback",
        "test_results",
        "test_statistics"
    }

    assert len(result["training_results"])==4
    assert len(result["candidate_lookbacks"])==2
    assert len(result["validation_results"])==2
    assert result["best_lookback"] in result["candidate_lookbacks"]
    assert len(result["test_results"])==5

    assert list(result["test_results"].index)==[
        25,
        26,
        27,
        28,
        29
    ]

    assert result["test_statistics"]["observations"]==5


def test_run_momentum_walk_forward_window_does_not_modify_input():
    df=make_long_dataframe()
    original_df=df.copy(deep=True)

    window={
        "train_end":20,
        "validation_end":25,
        "test_end":30
    }

    run_momentum_walk_forward_window(df,window,lookbacks=[1,2,3],n=2,cost_rate=0.001)

    pd.testing.assert_frame_equal(df,original_df)


def test_run_momentum_walk_forward():
    df=make_long_dataframe()

    result=run_momentum_walk_forward(df,lookbacks=[1,2,3,4],n=2,cost_rate=0.001,initial_train_size=20,validation_size=5,test_size=5)

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


def test_run_momentum_walk_forward_combines_test_periods_in_order():
    df=make_long_dataframe()

    result=run_momentum_walk_forward(df,lookbacks=[1,2,3],n=2,cost_rate=0.001,initial_train_size=20,validation_size=5,test_size=5)

    combined=result["combined_test_results"]

    assert combined.index.is_monotonic_increasing
    assert combined.index.is_unique


def test_walk_forward_boundary_turnover_uses_prior_actual_position():
    df=make_long_dataframe()

    result=run_momentum_walk_forward(df,lookbacks=[1,2,3],n=2,cost_rate=0.001,initial_train_size=20,validation_size=5,test_size=5)

    combined=result["combined_test_results"].reset_index(drop=True)

    for boundary in [5,10]:
        previous_position=combined.loc[boundary-1,"Position"]
        current_position=combined.loc[boundary,"Position"]

        assert combined.loc[boundary,"Turnover"]==pytest.approx(abs(current_position-previous_position))


def test_run_momentum_walk_forward_recalculates_cumulative_values():
    df=make_long_dataframe()

    result=run_momentum_walk_forward(df,lookbacks=[1,2,3],n=2,cost_rate=0.001,initial_train_size=20,validation_size=5,test_size=5)

    combined=result["combined_test_results"]

    expected_asset=(1+combined["DailyReturn"].fillna(0)).cumprod()
    expected_strategy=(1+combined["StrategyReturn"]).cumprod()
    expected_net=(1+combined["NetStrategyReturn"]).cumprod()

    assert combined["CumulativeValue"].tolist()==pytest.approx(expected_asset.tolist())
    assert combined["StrategyCumulativeValue"].tolist()==pytest.approx(expected_strategy.tolist())
    assert combined["NetStrategyCumulativeValue"].tolist()==pytest.approx(expected_net.tolist())


def test_run_momentum_walk_forward_cumulative_values_do_not_reset_between_windows():
    df=make_long_dataframe()

    result=run_momentum_walk_forward(df,lookbacks=[1,2,3],n=2,cost_rate=0.001,initial_train_size=20,validation_size=5,test_size=5)

    combined=result["combined_test_results"]

    first_window_end=combined.iloc[4]["NetStrategyCumulativeValue"]
    second_window_start=combined.iloc[5]["NetStrategyCumulativeValue"]

    expected_second_start=first_window_end*(1+combined.iloc[5]["NetStrategyReturn"])

    assert second_window_start==pytest.approx(expected_second_start)


def test_run_momentum_walk_forward_returns_risk_statistics():
    df=make_long_dataframe()

    result=run_momentum_walk_forward(df,lookbacks=[1,2,3],n=2,cost_rate=0.001,initial_train_size=20,validation_size=5,test_size=5)

    statistics=result["statistics"]

    required_statistics=[
        "net_strategy_return",
        "annualised_return",
        "annualised_volatility",
        "sharpe_ratio",
        "max_drawdown"
    ]

    for statistic in required_statistics:
        assert statistic in statistics


def test_run_momentum_walk_forward_each_window_has_selected_lookback():
    df=make_long_dataframe()

    result=run_momentum_walk_forward(df,lookbacks=[1,2,3,4],n=2,cost_rate=0.001,initial_train_size=20,validation_size=5,test_size=5)

    for window_result in result["full_results"]:
        assert window_result["best_lookback"] in window_result["candidate_lookbacks"]


def test_run_momentum_walk_forward_does_not_modify_input():
    df=make_long_dataframe()
    original_df=df.copy(deep=True)

    run_momentum_walk_forward(df,lookbacks=[1,2,3],n=2,cost_rate=0.001,initial_train_size=20,validation_size=5,test_size=5)

    pd.testing.assert_frame_equal(df,original_df)
