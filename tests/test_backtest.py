import pandas as pd
import numpy as np
import pytest

from src.backtest import(
    calculate_backtest_statistics,
    calculate_positions,
    calculate_strategy_returns,
    calculate_transaction_costs,
    run_backtest
)

from src.metrics import calculate_performance_metrics


def test_calculate_positions_delays_signal():
    df=pd.DataFrame({
        "Signal":[0,1,1,0]
    })

    result=calculate_positions(df)

    assert result["Position"].tolist()==[0,0,1,1]


def test_calculate_positions_does_not_modify_input():
    df=pd.DataFrame({
        "Signal":[0,1,1,0]
    })
    original=df.copy(deep=True)

    calculate_positions(df)

    pd.testing.assert_frame_equal(df,original)


def test_calculate_strategy_returns():
    df=pd.DataFrame({
        "Position":[0,0,1,1],
        "DailyReturn":[np.nan,0.05,0.10,-0.05]
    })

    result=calculate_strategy_returns(df)

    assert result["StrategyReturn"].tolist()==pytest.approx([
        0.0,
        0.0,
        0.10,
        -0.05
    ])

    assert result["StrategyCumulativeValue"].tolist()==pytest.approx([
        1.0,
        1.0,
        1.10,
        1.045
    ])


def test_strategy_returns_support_short_positions():
    df=pd.DataFrame({
        "Position":[0,-1,-1,1],
        "DailyReturn":[0.0,0.05,-0.10,0.02]
    })

    result=calculate_strategy_returns(df)

    assert result["StrategyReturn"].tolist()==pytest.approx([
        0.0,
        -0.05,
        0.10,
        0.02
    ])


def test_calculate_transaction_costs():
    df=pd.DataFrame({
        "Position":[0,1,1,0],
        "StrategyReturn":[0.0,0.03,-0.02,0.0]
    })

    result=calculate_transaction_costs(df,0.001)

    assert result["Turnover"].tolist()==pytest.approx([
        0.0,
        1.0,
        0.0,
        1.0
    ])

    assert result["TransactionCost"].tolist()==pytest.approx([
        0.0,
        0.001,
        0.0,
        0.001
    ])

    assert result["NetStrategyReturn"].tolist()==pytest.approx([
        0.0,
        0.029,
        -0.02,
        -0.001
    ])

    assert result["NetStrategyCumulativeValue"].tolist()==pytest.approx([
        1.0,
        1.029,
        1.00842,
        1.00741158
    ])


@pytest.mark.parametrize("invalid_cost_rate",[-0.001,-0.01,-1])
def test_transaction_costs_reject_negative_rates(invalid_cost_rate):
    df=pd.DataFrame({
        "Position":[0,1,0],
        "StrategyReturn":[0.0,0.05,0.0]
    })

    with pytest.raises(ValueError,match="cannot be negative"):
        calculate_transaction_costs(df,invalid_cost_rate)


def test_zero_transaction_cost_does_not_change_returns():
    df=pd.DataFrame({
        "Position":[0,1,1,0],
        "StrategyReturn":[0.0,0.05,-0.02,0.0]
    })

    result=calculate_transaction_costs(df,0)

    assert result["TransactionCost"].tolist()==pytest.approx([
        0.0,
        0.0,
        0.0,
        0.0
    ])

    assert result["NetStrategyReturn"].tolist()==pytest.approx(
        result["StrategyReturn"].tolist()
    )


def test_position_change_from_long_to_short_has_turnover_two():
    df=pd.DataFrame({
        "Position":[0,1,-1],
        "StrategyReturn":[0.0,0.02,0.03]
    })

    result=calculate_transaction_costs(df,0.001)

    assert result["Turnover"].tolist()==pytest.approx([
        0.0,
        1.0,
        2.0
    ])


def test_calculate_backtest_statistics():
    net_returns=pd.Series([
        0.0,
        0.01,
        0.039603960396,
        0.028571428571
    ])

    df=pd.DataFrame({
        "CumulativeValue":[1.0,1.05,1.10,1.20],
        "StrategyCumulativeValue":[1.0,1.02,1.06,1.10],
        "NetStrategyReturn":net_returns,
        "Turnover":[0.0,1.0,0.0,1.0],
        "Position":[1,-1,0,1]
    })

    stats=calculate_backtest_statistics(df)
    core_statistics=calculate_performance_metrics(net_returns)

    assert stats["observations"]==core_statistics["observations"]
    assert stats["buy_and_hold_return"]==pytest.approx(0.20)
    assert stats["gross_strategy_return"]==pytest.approx(0.10)
    assert stats["net_strategy_return"]==pytest.approx(core_statistics["total_return"])
    assert stats["annualised_return"]==pytest.approx(core_statistics["annualised_return"])
    assert stats["annualised_volatility"]==pytest.approx(core_statistics["annualised_volatility"])
    assert stats["sharpe_ratio"]==pytest.approx(core_statistics["sharpe_ratio"])
    assert stats["max_drawdown"]==pytest.approx(core_statistics["max_drawdown"])
    assert stats["total_turnover"]==pytest.approx(2.0)
    assert stats["trade_events"]==2
    assert stats["time_in_market"]==pytest.approx(0.75)


def test_statistics_use_canonical_names():
    df=pd.DataFrame({
        "CumulativeValue":[1.0,1.0],
        "StrategyCumulativeValue":[1.0,1.0],
        "NetStrategyReturn":[0.0,0.0],
        "Turnover":[0.0,0.0],
        "Position":[0,0]
    })

    stats=calculate_backtest_statistics(df)

    assert "annualised_return" in stats
    assert "annualised_volatility" in stats
    assert "max_drawdown" in stats
    assert "maximum_drawdown" not in stats


def test_zero_volatility_sharpe_is_nan():
    df=pd.DataFrame({
        "CumulativeValue":[1.0,1.0,1.0],
        "StrategyCumulativeValue":[1.0,1.0,1.0],
        "NetStrategyReturn":[0.0,0.0,0.0],
        "Turnover":[0.0,0.0,0.0],
        "Position":[0,0,0]
    })

    stats=calculate_backtest_statistics(df)

    assert stats["annualised_volatility"]==pytest.approx(0.0)
    assert np.isnan(stats["sharpe_ratio"])


def test_time_in_market_counts_short_positions():
    df=pd.DataFrame({
        "CumulativeValue":[1.0,1.0],
        "StrategyCumulativeValue":[1.0,1.0],
        "NetStrategyReturn":[0.0,0.0],
        "Turnover":[1.0,2.0],
        "Position":[1,-1]
    })

    stats=calculate_backtest_statistics(df)

    assert stats["time_in_market"]==pytest.approx(1.0)


def test_run_backtest_runs_complete_pipeline():
    df=pd.DataFrame({
        "Signal":[0,1,1,0],
        "DailyReturn":[np.nan,0.05,0.10,-0.05]
    })

    result=run_backtest(df,0.001)

    assert result["Position"].tolist()==[0,0,1,1]

    assert result["StrategyReturn"].tolist()==pytest.approx([
        0.0,
        0.0,
        0.10,
        -0.05
    ])

    assert result["Turnover"].tolist()==pytest.approx([
        0.0,
        0.0,
        1.0,
        0.0
    ])

    assert result["TransactionCost"].tolist()==pytest.approx([
        0.0,
        0.0,
        0.001,
        0.0
    ])

    assert result["NetStrategyReturn"].tolist()==pytest.approx([
        0.0,
        0.0,
        0.099,
        -0.05
    ])

    assert result["NetStrategyCumulativeValue"].tolist()==pytest.approx([
        1.0,
        1.0,
        1.099,
        1.04405
    ])


def test_run_backtest_does_not_modify_input():
    df=pd.DataFrame({
        "Signal":[0,1,1,0],
        "DailyReturn":[np.nan,0.05,0.10,-0.05]
    })
    original=df.copy(deep=True)

    run_backtest(df,0.001)

    pd.testing.assert_frame_equal(df,original)


def test_run_backtest_is_deterministic():
    df=pd.DataFrame({
        "Signal":[0,1,1,0],
        "DailyReturn":[np.nan,0.05,0.10,-0.05]
    })

    first_result=run_backtest(df,0.001)
    second_result=run_backtest(df,0.001)

    pd.testing.assert_frame_equal(first_result,second_result)


def test_statistics_rejects_empty_dataframe():
    with pytest.raises(ValueError,match="Cannot calculate statistics"):
        calculate_backtest_statistics(pd.DataFrame())



def test_transaction_costs_use_previous_execution_position():
    df=pd.DataFrame({
        "Position":[1,1],
        "StrategyReturn":[0.02,0.01]
    })

    result=calculate_transaction_costs(df,0.001,previous_position=0)

    assert result["Turnover"].tolist()==pytest.approx([
        1.0,
        0.0
    ])

    assert result["TransactionCost"].tolist()==pytest.approx([
        0.001,
        0.0
    ])
