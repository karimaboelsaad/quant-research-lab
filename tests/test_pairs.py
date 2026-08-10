import numpy as np
import pandas as pd
import pytest

import src.pairs as pairs

from src.pairs import(
    calculate_pair_regression,
    calculate_pair_spread,
    calculate_spread_z_score,
    calculate_pair_signal,
    calculate_pair_positions,
    calculate_pair_returns,
    apply_pair_transaction_costs,
    run_pair_strategy,
    calculate_pair_statistics
)


def test_calculate_pair_regression():
    df=pd.DataFrame({
        "CloseA":[25,45,65,85],
        "CloseB":[10,20,30,40]
    })

    result=calculate_pair_regression(df)

    assert result["alpha"]==pytest.approx(5)
    assert result["beta"]==pytest.approx(2)


def test_calculate_pair_regression_zero_variance():
    df=pd.DataFrame({
        "CloseA":[10,20,30],
        "CloseB":[5,5,5]
    })

    with pytest.raises(ValueError):
        calculate_pair_regression(df)


def test_calculate_pair_spread():
    df=pd.DataFrame({
        "CloseA":[25,45,65],
        "CloseB":[10,20,30]
    })

    result=calculate_pair_spread(df,5,2)

    expected_predicted=pd.Series([25,45,65],name="PredictedA")
    expected_spread=pd.Series([0,0,0],name="Spread")

    pd.testing.assert_series_equal(result["PredictedA"],expected_predicted)
    pd.testing.assert_series_equal(result["Spread"],expected_spread)


def test_calculate_spread_z_score():
    df=pd.DataFrame({
        "Spread":[1,2,3,4]
    })

    result=calculate_spread_z_score(df,3)

    assert pd.isna(result["SpreadZScore"].iloc[0])
    assert pd.isna(result["SpreadZScore"].iloc[1])
    assert result["SpreadZScore"].iloc[2]==pytest.approx(1)
    assert result["SpreadZScore"].iloc[3]==pytest.approx(1)


def test_calculate_spread_z_score_invalid_lookback():
    df=pd.DataFrame({
        "Spread":[1,2,3]
    })

    with pytest.raises(ValueError):
        calculate_spread_z_score(df,1)

    with pytest.raises(ValueError):
        calculate_spread_z_score(df,4)


def test_calculate_pair_signal():
    df=pd.DataFrame({
        "SpreadZScore":[0,-2.1,-1,-0.4,0,2.1,1,0.4,0]
    })

    result=calculate_pair_signal(df,2,0.5)

    expected=pd.Series(
        [0,1,1,0,0,-1,-1,0,0],
        name="Signal"
    )

    pd.testing.assert_series_equal(result["Signal"],expected)


def test_calculate_pair_signal_invalid_thresholds():
    df=pd.DataFrame({
        "ZScore":[0,1,-1]
    })

    with pytest.raises(ValueError):
        calculate_pair_signal(df,0,0)

    with pytest.raises(ValueError):
        calculate_pair_signal(df,2,-0.5)

    with pytest.raises(ValueError):
        calculate_pair_signal(df,2,2)


def test_calculate_pair_positions():
    df=pd.DataFrame({
        "Signal":[0,1,1,0,-1]
    })

    result=calculate_pair_positions(df,2)

    expected_spread=pd.Series(
        [0.0,0.0,1.0,1.0,0.0],
        name="SpreadPosition"
    )

    expected_a=pd.Series(
        [0.0,0.0,1.0,1.0,0.0],
        name="PositionA"
    )

    expected_b=pd.Series(
        [0.0,0.0,-2.0,-2.0,0.0],
        name="PositionB"
    )

    pd.testing.assert_series_equal(result["SpreadPosition"],expected_spread)
    pd.testing.assert_series_equal(result["PositionA"],expected_a)
    pd.testing.assert_series_equal(result["PositionB"],expected_b)


def test_calculate_pair_returns():
    df=pd.DataFrame({
        "CloseA":[100,103],
        "CloseB":[40,41],
        "PositionA":[0,1],
        "PositionB":[0,-2]
    })

    result=calculate_pair_returns(df)

    assert result["StrategyReturn"].iloc[0]==0
    assert result["PairPnL"].iloc[1]==pytest.approx(1)
    assert result["GrossExposure"].iloc[1]==pytest.approx(180)
    assert result["StrategyReturn"].iloc[1]==pytest.approx(1/180)
    assert result["StrategyCumulativeValue"].iloc[1]==pytest.approx(1+1/180)


def test_apply_pair_transaction_costs():
    df=pd.DataFrame({
        "SpreadPosition":[0,1,1,0,-1,1],
        "StrategyReturn":[0,0,0,0,0,0]
    })

    result=apply_pair_transaction_costs(df,0.001)

    expected_turnover=pd.Series(
        [0.0,1.0,0.0,1.0,1.0,2.0],
        name="Turnover"
    )

    pd.testing.assert_series_equal(result["Turnover"],expected_turnover)

    assert result["TransactionCost"].iloc[1]==pytest.approx(0.001)
    assert result["TransactionCost"].iloc[5]==pytest.approx(0.002)
    assert result["NetStrategyReturn"].iloc[5]==pytest.approx(-0.002)


def test_spread_stationarity(monkeypatch):
    df=pd.DataFrame({
        "Spread":[1,-1,0.5,-0.5,0.2]
    })

    def fake_adfuller(spread):
        return (
            -3.5,
            0.02,
            1,
            5,
            {
                "1%":-3.6,
                "5%":-2.9,
                "10%":-2.6
            },
            10
        )

    monkeypatch.setattr(pairs,"adfuller",fake_adfuller)

    result=pairs.test_spread_stationarity(df)

    assert result["adf_statistic"]==-3.5
    assert result["p_value"]==0.02
    assert result["is_stationary"]


def test_pair_cointegration(monkeypatch):
    df=pd.DataFrame({
        "CloseA":[10,12,14,16,18],
        "CloseB":[5,6,7,8,9]
    })

    def fake_coint(close_a, close_b):
        return (
            -4.2,
            0.01,
            np.array([-3.9,-3.3,-3.0])
        )

    monkeypatch.setattr(pairs,"coint",fake_coint)

    result=pairs.test_pair_cointegration(df)

    assert result["test_statistic"]==-4.2
    assert result["p_value"]==0.01
    assert result["is_cointegrated"]


def test_run_pair_strategy():
    df=pd.DataFrame({
        "CloseA":[21,23,24,27,27,31,32,35],
        "CloseB":[10,11,12,13,14,15,16,17]
    })

    result=run_pair_strategy(
        df,
        1,
        2,
        3,
        1,
        0.5,
        0.001
    )

    expected_columns=[
        "Spread",
        "SpreadZScore",
        "Signal",
        "SpreadPosition",
        "PositionA",
        "PositionB",
        "PairPnL",
        "GrossExposure",
        "StrategyReturn",
        "StrategyCumulativeValue",
        "Turnover",
        "TransactionCost",
        "NetStrategyReturn",
        "NetStrategyCumulativeValue"
    ]

    for column in expected_columns:
        assert column in result.columns

    assert len(result)==len(df)


def test_calculate_pair_statistics():
    strategy_returns=pd.Series([0,0.01,-0.01,0.02])
    net_returns=pd.Series([0,0.01,-0.02,0.01])

    df=pd.DataFrame({
        "StrategyReturn":strategy_returns,
        "NetStrategyReturn":net_returns,
        "StrategyCumulativeValue":(1+strategy_returns).cumprod(),
        "NetStrategyCumulativeValue":(1+net_returns).cumprod(),
        "Turnover":[0,1,0,1],
        "SpreadPosition":[0,1,1,0]
    })

    result=calculate_pair_statistics(df)

    assert result["observations"]==4
    assert result["gross_return"]==pytest.approx(
        df["StrategyCumulativeValue"].iloc[-1]-1
    )
    assert result["net_return"]==pytest.approx(
        df["NetStrategyCumulativeValue"].iloc[-1]-1
    )
    assert result["total_turnover"]==2
    assert result["trade_events"]==2
    assert result["time_in_market"]==pytest.approx(0.5)
    assert result["max_drawdown"]<=0