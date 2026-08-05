import math

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

    assert result[
        "StrategyCumulativeValue"
    ].tolist() == pytest.approx(
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

    assert result[
        "StrategyCumulativeValue"
    ].tolist() == pytest.approx(
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

    assert result[
        "TransactionCost"
    ].tolist() == pytest.approx(
        [
            0.0,
            0.001,
            0.0,
            0.001,
        ]
    )

    assert result[
        "NetStrategyReturn"
    ].tolist() == pytest.approx(
        [
            0.0,
            0.029,
            -0.02,
            -0.001,
        ]
    )

    assert result[
        "NetStrategyCumulativeValue"
    ].tolist() == pytest.approx(
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
            "StrategyReturn": [
                0.0,
                0.05,
                0.0,
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="cannot be negative",
    ):
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

    assert result[
        "TransactionCost"
    ].tolist() == pytest.approx(
        [
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )

    assert result[
        "NetStrategyReturn"
    ].tolist() == pytest.approx(
        result["StrategyReturn"].tolist()
    )


def test_position_change_from_long_to_short_has_turnover_two():
    df = pd.DataFrame(
        {
            "Position": [0, 1, -1],
            "StrategyReturn": [
                0.0,
                0.02,
                0.03,
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
            2.0,
        ]
    )

    assert result[
        "TransactionCost"
    ].tolist() == pytest.approx(
        [
            0.0,
            0.001,
            0.002,
        ]
    )


def test_calculate_backtest_statistics():
    net_returns = pd.Series(
        [
            0.0,
            0.01,
            0.039603960396,
            0.028571428571,
        ],
        dtype=float,
    )

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
            "NetStrategyReturn": net_returns,
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

    expected_annualised_return = (
        1.08 ** (252 / 4) - 1
    )

    expected_annualised_volatility = (
        net_returns.std() * math.sqrt(252)
    )

    expected_sharpe = (
        net_returns.mean()
        / net_returns.std()
        * math.sqrt(252)
    )

    assert stats["observations"] == 4

    assert stats[
        "buy_and_hold_return"
    ] == pytest.approx(0.20)

    assert stats[
        "gross_strategy_return"
    ] == pytest.approx(0.10)

    assert stats[
        "net_strategy_return"
    ] == pytest.approx(0.08)

    assert stats[
        "annualised_return"
    ] == pytest.approx(
        expected_annualised_return
    )

    assert stats[
        "annualised_volatility"
    ] == pytest.approx(
        expected_annualised_volatility
    )

    assert stats[
        "sharpe_ratio"
    ] == pytest.approx(expected_sharpe)

    assert stats[
        "maximum_drawdown"
    ] == pytest.approx(0.0)

    assert stats[
        "total_turnover"
    ] == pytest.approx(2.0)

    assert stats["trade_events"] == 2

    assert stats[
        "time_in_market"
    ] == pytest.approx(0.75)


def test_time_in_market_counts_short_positions():
    df = pd.DataFrame(
        {
            "CumulativeValue": [
                1.0,
                1.0,
            ],
            "StrategyCumulativeValue": [
                1.0,
                1.0,
            ],
            "NetStrategyReturn": [
                0.0,
                0.0,
            ],
            "NetStrategyCumulativeValue": [
                1.0,
                1.0,
            ],
            "Turnover": [
                1.0,
                2.0,
            ],
            "Position": [
                1,
                -1,
            ],
        }
    )

    stats = calculate_backtest_statistics(df)

    assert stats[
        "time_in_market"
    ] == pytest.approx(1.0)


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

    assert result[
        "StrategyReturn"
    ].tolist() == pytest.approx(
        [
            0.0,
            0.0,
            0.10,
            -0.05,
        ]
    )

    assert result[
        "StrategyCumulativeValue"
    ].tolist() == pytest.approx(
        [
            1.0,
            1.0,
            1.10,
            1.045,
        ]
    )

    assert result[
        "Turnover"
    ].tolist() == pytest.approx(
        [
            0.0,
            0.0,
            1.0,
            0.0,
        ]
    )

    assert result[
        "TransactionCost"
    ].tolist() == pytest.approx(
        [
            0.0,
            0.0,
            0.001,
            0.0,
        ]
    )

    assert result[
        "NetStrategyReturn"
    ].tolist() == pytest.approx(
        [
            0.0,
            0.0,
            0.099,
            -0.05,
        ]
    )

    assert result[
        "NetStrategyCumulativeValue"
    ].tolist() == pytest.approx(
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


def make_statistics_dataframe(net_returns):
    net_returns = pd.Series(
        net_returns,
        dtype=float,
    )

    cumulative_value = (
        1 + net_returns
    ).cumprod()

    return pd.DataFrame(
        {
            "CumulativeValue": cumulative_value,
            "StrategyCumulativeValue": cumulative_value,
            "NetStrategyReturn": net_returns,
            "NetStrategyCumulativeValue": cumulative_value,
            "Turnover": (
                [0.0] * len(net_returns)
            ),
            "Position": (
                [1] * len(net_returns)
            ),
        }
    )


def test_statistics_calculates_annualised_return():
    df = make_statistics_dataframe(
        [0.01, 0.01]
    )

    stats = calculate_backtest_statistics(df)

    final_value = 1.01 ** 2

    expected_return = (
        final_value ** (252 / 2)
        - 1
    )

    assert stats[
        "annualised_return"
    ] == pytest.approx(expected_return)


def test_statistics_calculates_annualised_volatility():
    returns = pd.Series(
        [
            0.01,
            -0.01,
            0.02,
        ],
        dtype=float,
    )

    df = make_statistics_dataframe(returns)

    stats = calculate_backtest_statistics(df)

    expected_volatility = (
        returns.std() * math.sqrt(252)
    )

    assert stats[
        "annualised_volatility"
    ] == pytest.approx(expected_volatility)


def test_statistics_calculates_sharpe_ratio():
    returns = pd.Series(
        [
            0.01,
            -0.01,
            0.02,
        ],
        dtype=float,
    )

    df = make_statistics_dataframe(returns)

    stats = calculate_backtest_statistics(df)

    expected_sharpe = (
        returns.mean()
        / returns.std()
        * math.sqrt(252)
    )

    assert stats[
        "sharpe_ratio"
    ] == pytest.approx(expected_sharpe)


def test_statistics_sets_sharpe_to_zero_when_volatility_is_zero():
    df = make_statistics_dataframe(
        [
            0.0,
            0.0,
            0.0,
        ]
    )

    stats = calculate_backtest_statistics(df)

    assert stats[
        "annualised_volatility"
    ] == pytest.approx(0.0)

    assert stats[
        "sharpe_ratio"
    ] == pytest.approx(0.0)


def test_statistics_handles_single_observation():
    df = make_statistics_dataframe(
        [0.01]
    )

    stats = calculate_backtest_statistics(df)

    assert stats["observations"] == 1

    assert stats[
        "annualised_volatility"
    ] == pytest.approx(0.0)

    assert stats[
        "sharpe_ratio"
    ] == pytest.approx(0.0)


def test_statistics_calculates_maximum_drawdown():
    df = make_statistics_dataframe(
        [
            0.10,
            -0.20,
            0.05,
        ]
    )

    stats = calculate_backtest_statistics(df)

    assert stats[
        "maximum_drawdown"
    ] == pytest.approx(-0.20)


def test_drawdown_includes_loss_from_initial_capital():
    df = make_statistics_dataframe(
        [
            -0.10,
            0.05,
        ]
    )

    stats = calculate_backtest_statistics(df)

    assert stats[
        "maximum_drawdown"
    ] == pytest.approx(-0.10)


def test_statistics_rejects_empty_dataframe():
    with pytest.raises(
        ValueError,
        match="Cannot calculate statistics",
    ):
        calculate_backtest_statistics(
            pd.DataFrame()
        )