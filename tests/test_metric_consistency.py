import pandas as pd
import pytest

from src.backtest import calculate_backtest_statistics
from src.pairs import calculate_pair_statistics
from src.portfolio import calculate_portfolio_statistics
from src.dynamic_portfolio import calculate_dynamic_portfolio_statistics
from src.metrics import calculate_performance_metrics


def test_all_strategy_families_use_identical_performance_metrics():
    returns=pd.Series([
        0.01,
        -0.02,
        0.03,
        -0.01
    ],dtype=float)

    cumulative_value=(1+returns).cumprod()

    backtest_df=pd.DataFrame({
        "CumulativeValue":cumulative_value,
        "StrategyCumulativeValue":cumulative_value,
        "NetStrategyReturn":returns,
        "Turnover":[0.0]*len(returns),
        "Position":[1]*len(returns)
    })

    pair_df=pd.DataFrame({
        "StrategyReturn":returns,
        "NetStrategyReturn":returns,
        "Turnover":[0.0]*len(returns),
        "SpreadPosition":[1]*len(returns)
    })

    portfolio_df=pd.DataFrame({
        "PortfolioReturn":returns,
        "PortfolioCumulativeValue":cumulative_value,
        "Turnover":[0.0]*len(returns),
        "TransactionCost":[0.0]*len(returns),
        "NetPortfolioReturn":returns,
        "NetPortfolioCumulativeValue":cumulative_value
    })

    dynamic_df=pd.DataFrame({
        "DynamicGrossExposure":[1.0]*len(returns),
        "DynamicPortfolioReturn":returns,
        "DynamicTurnover":[0.0]*len(returns),
        "DynamicTransactionCost":[0.0]*len(returns),
        "NetDynamicPortfolioReturn":returns
    })

    core_statistics=calculate_performance_metrics(returns)
    backtest_statistics=calculate_backtest_statistics(backtest_df)
    pair_statistics=calculate_pair_statistics(pair_df)
    portfolio_statistics=calculate_portfolio_statistics(portfolio_df)
    dynamic_statistics=calculate_dynamic_portfolio_statistics(
        dynamic_df,
        2,
        0.001
    )

    strategy_statistics=[
        backtest_statistics,
        pair_statistics,
        portfolio_statistics,
        dynamic_statistics
    ]

    for statistics in strategy_statistics:
        assert statistics["annualised_return"]==pytest.approx(
            core_statistics["annualised_return"]
        )
        assert statistics["annualised_volatility"]==pytest.approx(
            core_statistics["annualised_volatility"]
        )
        assert statistics["sharpe_ratio"]==pytest.approx(
            core_statistics["sharpe_ratio"]
        )
        assert statistics["max_drawdown"]==pytest.approx(
            core_statistics["max_drawdown"]
        )