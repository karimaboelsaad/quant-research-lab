import pandas as pd
import pytest

from src.analyse import(
    calculate_returns,
    validate_data
)

from src.backtest import calculate_backtest_statistics

from src.momentum_optimise import(
    evaluate_lookbacks_on_period,
    evaluate_momentum_period,
    run_momentum_optimisation,
    select_best_lookback,
    select_top_lookbacks
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
        100,
        102,
        101,
        104,
        103,
        106,
        105,
        108,
        107,
        110,
        109,
        112,
        111,
        114,
        113,
        116,
        115,
        118,
        117,
        120,
        119,
        122,
        121,
        124,
        123,
        126,
        125,
        128,
        127,
        130
    ]

    return make_prepared_dataframe(prices)


def test_evaluate_period_uses_warmup_history():
    df=make_prepared_dataframe([
        100,
        100,
        100,
        110,
        121,
        120,
        132,
        130
    ])

    result=evaluate_momentum_period(df,start_position=4,end_position=7,lookback=2,cost_rate=0.001)

    assert list(result.index)==[
        4,
        5,
        6
    ]

    assert result.iloc[0]["Position"]==1


def test_evaluate_period_preserves_initial_trade_cost():
    df=make_prepared_dataframe([
        100,
        100,
        100,
        110,
        121,
        120,
        132
    ])

    result=evaluate_momentum_period(df,start_position=4,end_position=7,lookback=2,cost_rate=0.001)

    first_row=result.iloc[0]

    assert first_row["Position"]==1
    assert first_row["Turnover"]==pytest.approx(1.0)
    assert first_row["TransactionCost"]==pytest.approx(0.001)


def test_evaluate_period_preserves_position_continuity():
    df=make_prepared_dataframe([
        100,
        100,
        105,
        110,
        115,
        120,
        125,
        130
    ])

    result=evaluate_momentum_period(df,start_position=5,end_position=8,lookback=2,cost_rate=0.001)

    first_row=result.iloc[0]

    assert first_row["Position"]==1
    assert first_row["Turnover"]==pytest.approx(0.0)
    assert first_row["TransactionCost"]==pytest.approx(0.0)


def test_evaluate_period_returns_only_requested_rows():
    df=make_prepared_dataframe([
        100,
        101,
        102,
        103,
        104,
        105,
        106
    ])

    result=evaluate_momentum_period(df,start_position=3,end_position=6,lookback=2,cost_rate=0.0)

    assert len(result)==3

    assert list(result.index)==[
        3,
        4,
        5
    ]


def test_evaluate_period_resets_cumulative_values():
    df=make_prepared_dataframe([
        100,
        100,
        100,
        110,
        121,
        120,
        132
    ])

    result=evaluate_momentum_period(df,start_position=4,end_position=7,lookback=2,cost_rate=0.001)

    first_daily_return=121/110-1

    expected_asset_value=1+first_daily_return
    expected_gross_value=1+first_daily_return
    expected_net_value=1+first_daily_return-0.001

    assert result.iloc[0]["CumulativeValue"]==pytest.approx(expected_asset_value)
    assert result.iloc[0]["StrategyCumulativeValue"]==pytest.approx(expected_gross_value)
    assert result.iloc[0]["NetStrategyCumulativeValue"]==pytest.approx(expected_net_value)


def test_training_period_starts_with_cumulative_value_one():
    df=make_prepared_dataframe([
        100,
        105,
        110,
        115
    ])

    result=evaluate_momentum_period(df,start_position=0,end_position=3,lookback=1,cost_rate=0.0)

    assert pd.isna(result.iloc[0]["DailyReturn"])
    assert result.iloc[0]["CumulativeValue"]==pytest.approx(1.0)
    assert result.iloc[0]["StrategyCumulativeValue"]==pytest.approx(1.0)
    assert result.iloc[0]["NetStrategyCumulativeValue"]==pytest.approx(1.0)


def test_evaluate_period_does_not_modify_input():
    df=make_prepared_dataframe([
        100,
        101,
        102,
        103,
        104,
        105
    ])

    original_df=df.copy(deep=True)

    evaluate_momentum_period(df,start_position=3,end_position=6,lookback=2,cost_rate=0.001)

    pd.testing.assert_frame_equal(df,original_df)


@pytest.mark.parametrize("lookback",[0,-1,-10])
def test_evaluate_period_rejects_invalid_lookback(lookback):
    df=make_prepared_dataframe([
        100,
        101,
        102,
        103
    ])

    with pytest.raises(ValueError,match="Lookback must be positive"):
        evaluate_momentum_period(df,start_position=0,end_position=3,lookback=lookback,cost_rate=0.0)


@pytest.mark.parametrize("start_position,end_position",[(-1,3),(2,2),(3,2),(0,5)])
def test_evaluate_period_rejects_invalid_boundaries(start_position, end_position):
    df=make_prepared_dataframe([
        100,
        101,
        102,
        103
    ])

    with pytest.raises(ValueError,match="Invalid evaluation period"):
        evaluate_momentum_period(df,start_position=start_position,end_position=end_position,lookback=1,cost_rate=0.0)


def test_evaluate_period_rejects_negative_cost_rate():
    df=make_prepared_dataframe([
        100,
        101,
        102,
        103,
        104
    ])

    with pytest.raises(ValueError,match="cannot be negative"):
        evaluate_momentum_period(df,start_position=2,end_position=5,lookback=1,cost_rate=-0.001)


def test_evaluate_lookbacks_on_period():
    df=make_prepared_dataframe([
        100,
        101,
        102,
        103,
        102,
        104,
        106,
        105
    ])

    lookbacks=[
        1,
        2,
        3
    ]

    result=evaluate_lookbacks_on_period(df,start_position=4,end_position=8,lookbacks=lookbacks,cost_rate=0.001)

    assert result.columns.tolist()==[
        "Lookback",
        "NetStrategyReturn"
    ]

    assert result["Lookback"].tolist()==lookbacks
    assert len(result)==3

    for lookback in lookbacks:
        period_result=evaluate_momentum_period(df,start_position=4,end_position=8,lookback=lookback,cost_rate=0.001)

        expected_stats=calculate_backtest_statistics(period_result)

        actual_return=result.loc[result["Lookback"]==lookback,"NetStrategyReturn"].iloc[0]

        assert actual_return==pytest.approx(expected_stats["net_strategy_return"])


def test_evaluate_lookbacks_on_period_does_not_modify_input():
    df=make_long_dataframe()

    original_df=df.copy(deep=True)

    evaluate_lookbacks_on_period(df,start_position=10,end_position=20,lookbacks=[1,2,3],cost_rate=0.001)

    pd.testing.assert_frame_equal(df,original_df)


def test_select_best_lookback():
    results=pd.DataFrame({
        "Lookback":[
            5,
            10,
            20
        ],
        "NetStrategyReturn":[
            0.04,
            0.12,
            0.08
        ]
    },index=[
        10,
        20,
        30
    ])

    best_lookback=select_best_lookback(results)

    assert best_lookback==10
    assert isinstance(best_lookback,int)


def test_select_best_lookback_rejects_empty_results():
    results=pd.DataFrame(columns=[
        "Lookback",
        "NetStrategyReturn"
    ])

    with pytest.raises(ValueError,match="Cannot select lookback"):
        select_best_lookback(results)


def test_select_top_lookbacks():
    results=pd.DataFrame({
        "Lookback":[
            5,
            10,
            20,
            50
        ],
        "NetStrategyReturn":[
            0.04,
            0.12,
            0.08,
            -0.02
        ]
    })

    selected=select_top_lookbacks(results,n=2)

    assert selected==[
        10,
        20
    ]

    assert all(isinstance(value,int) for value in selected)


def test_select_top_lookbacks_does_not_modify_input():
    results=pd.DataFrame({
        "Lookback":[
            5,
            10,
            20
        ],
        "NetStrategyReturn":[
            0.04,
            0.12,
            0.08
        ]
    })

    original_results=results.copy(deep=True)

    select_top_lookbacks(results,n=2)

    pd.testing.assert_frame_equal(results,original_results)


def test_select_top_lookbacks_rejects_empty_results():
    results=pd.DataFrame(columns=[
        "Lookback",
        "NetStrategyReturn"
    ])

    with pytest.raises(ValueError,match="Cannot select lookbacks"):
        select_top_lookbacks(results,n=1)


@pytest.mark.parametrize("invalid_n",[0,-1,-10])
def test_select_top_lookbacks_rejects_non_positive_n(invalid_n):
    results=pd.DataFrame({
        "Lookback":[
            5,
            10,
            20
        ],
        "NetStrategyReturn":[
            0.04,
            0.12,
            0.08
        ]
    })

    with pytest.raises(ValueError,match="n must be positive"):
        select_top_lookbacks(results,n=invalid_n)


def test_select_top_lookbacks_rejects_n_that_is_too_large():
    results=pd.DataFrame({
        "Lookback":[
            5,
            10
        ],
        "NetStrategyReturn":[
            0.04,
            0.12
        ]
    })

    with pytest.raises(ValueError,match="n cannot exceed"):
        select_top_lookbacks(results,n=3)


def test_run_momentum_optimisation_runs_complete_pipeline():
    df=make_long_dataframe()

    result=run_momentum_optimisation(df,lookbacks=[1,2,3,4],n=2,cost_rate=0.001,train_ratio=0.6,validation_ratio=0.2)

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
    assert len(result["test_results"])==6

    assert list(result["test_results"].index)==[
        24,
        25,
        26,
        27,
        28,
        29
    ]

    assert result["test_statistics"]["observations"]==6


def test_run_momentum_optimisation_returns_complete_test_backtest():
    df=make_long_dataframe()

    result=run_momentum_optimisation(df,lookbacks=[1,2,3,4],n=2,cost_rate=0.001)

    test_results=result["test_results"]

    required_columns=[
        "LookbackReturn",
        "Signal",
        "Position",
        "StrategyReturn",
        "StrategyCumulativeValue",
        "Turnover",
        "TransactionCost",
        "NetStrategyReturn",
        "NetStrategyCumulativeValue"
    ]

    for column in required_columns:
        assert column in test_results.columns


def test_run_momentum_optimisation_returns_correct_test_statistics():
    df=make_long_dataframe()

    result=run_momentum_optimisation(df,lookbacks=[1,2,3,4],n=2,cost_rate=0.001)

    test_results=result["test_results"]
    test_statistics=result["test_statistics"]

    expected_net_return=test_results["NetStrategyCumulativeValue"].iloc[-1]-1

    assert test_statistics["net_strategy_return"]==pytest.approx(expected_net_return)
    assert test_statistics["lookback"]==result["best_lookback"]
    assert test_statistics["cost_rate"]==pytest.approx(0.001)


def test_run_momentum_optimisation_returns_risk_statistics():
    df=make_long_dataframe()

    result=run_momentum_optimisation(df,lookbacks=[1,2,3,4],n=2,cost_rate=0.001)

    test_statistics=result["test_statistics"]

    required_statistics=[
        "annualised_return",
        "annualised_volatility",
        "sharpe_ratio",
        "maximum_drawdown"
    ]

    for statistic in required_statistics:
        assert statistic in test_statistics


def test_run_momentum_optimisation_does_not_modify_input():
    df=make_long_dataframe()

    original_df=df.copy(deep=True)

    run_momentum_optimisation(df,lookbacks=[1,2,3,4],n=2,cost_rate=0.001)

    pd.testing.assert_frame_equal(df,original_df)


def test_run_momentum_optimisation_rejects_too_many_candidates():
    df=make_long_dataframe()

    with pytest.raises(ValueError,match="n cannot exceed"):
        run_momentum_optimisation(df,lookbacks=[1,2],n=3,cost_rate=0.001)