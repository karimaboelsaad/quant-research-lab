import pandas as pd
import pytest

from src.backtest import (
    calculate_backtest_statistics,
    calculate_positions,
    calculate_strategy_returns,
    calculate_transaction_costs,
    run_backtest,
)


def test_calculate_positions_delays_signal():
    df = pd.DataFrame(
        {
            "Signal": [0, 1, 1, 0],
        }
    )

    result = calculate_positions(df)

    assert result["Position"].tolist() == [
        0,
        0,
        1,
        1,
    ]


def test_calculate_positions_does_not_modify_input():
    df = pd.DataFrame(
        {
            "Signal": [0, 1, 1, 0],
        }
    )

    original_df = df.copy(deep=True)

    calculate_positions(df)

    pd.testing.assert_frame_equal(
        df,
        original_df,
    )


def test_calculate_strategy_returns():
    df = pd.DataFrame(
        {
            "Position": [0, 0, 1, 1],
            "DailyReturn": [
                float("nan"),
                0.05,
                0.10,
                -0.05,
            ],
        }
    )

    result = calculate_strategy_returns(df)

    assert result["StrategyReturn"].tolist() == pytest.approx(
        [
            0.0,
            0.0,
            0.10,
            -0.05,
        ]
    )

    assert result["StrategyCumulativeValue"].tolist() == pytest.approx(
        [
            1.0,
            1.0,
            1.10,
            1.045,
        ]
    )


def test_strategy_returns_support_short_positions():
    df = pd.DataFrame(
        {
            "Position": [0, -1, -1, 1],
            "DailyReturn": [
                0.0,
                0.05,
                -0.10,
                0.02,
            ],
        }
    )

    result = calculate_strategy_returns(df)

    assert result["StrategyReturn"].tolist() == pytest.approx(
        [
            0.0,
            -0.05,
            0.10,
            0.02,
        ]
    )

    assert result["StrategyCumulativeValue"].tolist() == pytest.approx(
        [
            1.0,
            0.95,
            1.045,
            1.0659,
        ]
    )


def test_calculate_transaction_costs():
    df = pd.DataFrame(
        {
            "Position": [0, 1, 1, 0],
            "StrategyReturn": [
                0.0,
                0.03,
                -0.02,
                0.0,
            ],
        }
    )

    result = calculate_transaction_costs(
        df,
        cost_rate=0.001,
    )

    assert result["Turnover"].tolist() == pytest.approx(
        [
            0.0,
            1.0,
            0.0,
            1.0,
        ]
    )

    assert result["TransactionCost"].tolist() == pytest.approx(
        [
            0.0,
            0.001,
            0.0,
            0.001,
        ]
    )

    assert result["NetStrategyReturn"].tolist() == pytest.approx(
        [
            0.0,
            0.029,
            -0.02,
            -0.001,
        ]
    )

    assert result["NetStrategyCumulativeValue"].tolist() == pytest.approx(
        [
            1.0,
            1.029,
            1.00842,
            1.00741158,
        ]
    )


@pytest.mark.parametrize(
    "invalid_cost_rate",
    [
        -0.001,
        -0.01,
        -1,
    ],
)
def test_transaction_costs_reject_negative_rates(
    invalid_cost_rate,
):
    df = pd.DataFrame(
        {
            "Position": [0, 1, 0],
            "StrategyReturn": [0.0, 0.05, 0.0],
        }
    )

    with pytest.raises(ValueError):
        calculate_transaction_costs(
            df,
            cost_rate=invalid_cost_rate,
        )


def test_zero_transaction_cost_does_not_change_returns():
    df = pd.DataFrame(
        {
            "Position": [0, 1, 1, 0],
            "StrategyReturn": [
                0.0,
                0.05,
                -0.02,
                0.0,
            ],
        }
    )

    result = calculate_transaction_costs(
        df,
        cost_rate=0,
    )

    assert result["TransactionCost"].tolist() == pytest.approx(
        [
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )

    assert result["NetStrategyReturn"].tolist() == pytest.approx(
        result["StrategyReturn"].tolist()
    )


def test_position_change_from_long_to_short_has_turnover_two():
    df = pd.DataFrame(
        {
            "Position": [0, 1, -1],
            "StrategyReturn": [0.0, 0.02, 0.03],
        }
    )

    result = calculate_transaction_costs(
        df,
        cost_rate=0.001,
    )

    assert result["Turnover"].tolist() == pytest.approx(
        [
            0.0,
            1.0,
            2.0,
        ]
    )

    assert result["TransactionCost"].tolist() == pytest.approx(
        [
            0.0,
            0.001,
            0.002,
        ]
    )


def test_calculate_backtest_statistics():
    df = pd.DataFrame(
        {
            "CumulativeValue": [
                1.0,
                1.05,
                1.10,
                1.20,
            ],
            "StrategyCumulativeValue": [
                1.0,
                1.02,
                1.06,
                1.10,
            ],
            "NetStrategyCumulativeValue": [
                1.0,
                1.01,
                1.05,
                1.08,
            ],
            "Turnover": [
                0.0,
                1.0,
                0.0,
                1.0,
            ],
            "Position": [
                1,
                -1,
                0,
                1,
            ],
        }
    )

    stats = calculate_backtest_statistics(df)

    assert stats["observations"] == 4
    assert stats["buy_and_hold_return"] == pytest.approx(0.20)
    assert stats["gross_strategy_return"] == pytest.approx(0.10)
    assert stats["net_strategy_return"] == pytest.approx(0.08)
    assert stats["total_turnover"] == pytest.approx(2.0)
    assert stats["trade_events"] == 2
    assert stats["time_in_market"] == pytest.approx(0.75)


def test_time_in_market_counts_short_positions():
    df = pd.DataFrame(
        {
            "CumulativeValue": [1.0, 1.0],
            "StrategyCumulativeValue": [1.0, 1.0],
            "NetStrategyCumulativeValue": [1.0, 1.0],
            "Turnover": [1.0, 2.0],
            "Position": [1, -1],
        }
    )

    stats = calculate_backtest_statistics(df)

    assert stats["time_in_market"] == pytest.approx(1.0)


def test_run_backtest_runs_complete_pipeline():
    df = pd.DataFrame(
        {
            "Signal": [0, 1, 1, 0],
            "DailyReturn": [
                float("nan"),
                0.05,
                0.10,
                -0.05,
            ],
        }
    )

    result = run_backtest(
        df,
        cost_rate=0.001,
    )

    assert result["Position"].tolist() == [
        0,
        0,
        1,
        1,
    ]

    assert result["StrategyReturn"].tolist() == pytest.approx(
        [
            0.0,
            0.0,
            0.10,
            -0.05,
        ]
    )

    assert result["StrategyCumulativeValue"].tolist() == pytest.approx(
        [
            1.0,
            1.0,
            1.10,
            1.045,
        ]
    )

    assert result["Turnover"].tolist() == pytest.approx(
        [
            0.0,
            0.0,
            1.0,
            0.0,
        ]
    )

    assert result["TransactionCost"].tolist() == pytest.approx(
        [
            0.0,
            0.0,
            0.001,
            0.0,
        ]
    )

    assert result["NetStrategyReturn"].tolist() == pytest.approx(
        [
            0.0,
            0.0,
            0.099,
            -0.05,
        ]
    )

    assert result["NetStrategyCumulativeValue"].tolist() == pytest.approx(
        [
            1.0,
            1.0,
            1.099,
            1.04405,
        ]
    )


def test_run_backtest_does_not_modify_input():
    df = pd.DataFrame(
        {
            "Signal": [0, 1, 1, 0],
            "DailyReturn": [
                float("nan"),
                0.05,
                0.10,
                -0.05,
            ],
        }
    )

    original_df = df.copy(deep=True)

    run_backtest(
        df,
        cost_rate=0.001,
    )

    pd.testing.assert_frame_equal(
        df,
        original_df,
    )


def test_run_backtest_is_deterministic():
    df = pd.DataFrame(
        {
            "Signal": [0, 1, 1, 0],
            "DailyReturn": [
                float("nan"),
                0.05,
                0.10,
                -0.05,
            ],
        }
    )

    first_result = run_backtest(
        df,
        cost_rate=0.001,
    )

    second_result = run_backtest(
        df,
        cost_rate=0.001,
    )

    pd.testing.assert_frame_equal(
        first_result,
        second_result,
    )