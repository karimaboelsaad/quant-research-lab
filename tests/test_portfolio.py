import numpy as np
import pandas as pd
import pytest

from src.portfolio import(
    calculate_asset_returns,
    calculate_correlation_matrix,
    calculate_covariance_matrix,
    calculate_equal_weights,
    calculate_portfolio_returns,
    calculate_portfolio_volatility,
    calculate_inverse_volatility_weights,
    calculate_pre_oos_inverse_volatility_weights,
    calculate_rebalancing_turnover,
    apply_portfolio_transaction_costs,
    calculate_portfolio_statistics,
    load_portfolio_prices,
    save_portfolio_output
)

from src.metrics import calculate_performance_metrics


def test_calculate_asset_returns():
    df=pd.DataFrame({
        "Date":pd.date_range("2024-01-01",periods=3),
        "A":[100,110,99],
        "B":[50,55,60.5]
    })

    result=calculate_asset_returns(df)

    assert "AReturn" in result.columns
    assert "BReturn" in result.columns
    assert pd.isna(result["AReturn"].iloc[0])
    assert pd.isna(result["BReturn"].iloc[0])
    assert result["AReturn"].iloc[1]==pytest.approx(0.1)
    assert result["AReturn"].iloc[2]==pytest.approx(-0.1)
    assert result["BReturn"].iloc[1]==pytest.approx(0.1)
    assert result["BReturn"].iloc[2]==pytest.approx(0.1)


def test_calculate_asset_returns_rejects_empty_dataframe():
    with pytest.raises(ValueError):
        calculate_asset_returns(pd.DataFrame())


def test_calculate_correlation_matrix():
    df=pd.DataFrame({
        "AReturn":[np.nan,0.01,0.02,0.03],
        "BReturn":[np.nan,0.02,0.04,0.06]
    })

    result=calculate_correlation_matrix(df)

    assert result.shape==(2,2)
    assert result.loc["AReturn","AReturn"]==pytest.approx(1)
    assert result.loc["BReturn","BReturn"]==pytest.approx(1)
    assert result.loc["AReturn","BReturn"]==pytest.approx(1)


def test_calculate_correlation_matrix_requires_two_return_columns():
    df=pd.DataFrame({
        "AReturn":[0.01,0.02]
    })

    with pytest.raises(ValueError):
        calculate_correlation_matrix(df)


def test_calculate_covariance_matrix():
    df=pd.DataFrame({
        "AReturn":[0.01,0.02,0.03],
        "BReturn":[0.02,0.04,0.06]
    })

    result=calculate_covariance_matrix(df)

    assert result.loc["AReturn","AReturn"]==pytest.approx(0.0001)
    assert result.loc["AReturn","BReturn"]==pytest.approx(0.0002)
    assert result.loc["BReturn","AReturn"]==pytest.approx(0.0002)
    assert result.loc["BReturn","BReturn"]==pytest.approx(0.0004)


def test_calculate_equal_weights():
    df=pd.DataFrame({
        "AReturn":[0.01,0.02],
        "BReturn":[0.02,0.01],
        "CReturn":[0.03,0.01]
    })

    result=calculate_equal_weights(df)

    assert result["AReturn"]==pytest.approx(1/3)
    assert result["BReturn"]==pytest.approx(1/3)
    assert result["CReturn"]==pytest.approx(1/3)
    assert sum(result.values())==pytest.approx(1)


def test_calculate_portfolio_returns():
    df=pd.DataFrame({
        "AReturn":[0.10,-0.10],
        "BReturn":[0.00,0.20]
    })

    portfolio_weights={
        "AReturn":0.6,
        "BReturn":0.4
    }

    result=calculate_portfolio_returns(df,portfolio_weights)

    assert result["PortfolioReturn"].iloc[0]==pytest.approx(0.06)
    assert result["PortfolioReturn"].iloc[1]==pytest.approx(0.02)
    assert result["PortfolioCumulativeValue"].iloc[0]==pytest.approx(1.06)
    assert result["PortfolioCumulativeValue"].iloc[1]==pytest.approx(1.0812)


def test_calculate_portfolio_returns_rejects_invalid_weights():
    df=pd.DataFrame({
        "AReturn":[0.01],
        "BReturn":[0.02]
    })

    portfolio_weights={
        "AReturn":0.6,
        "BReturn":0.6
    }

    with pytest.raises(ValueError):
        calculate_portfolio_returns(df,portfolio_weights)


def test_calculate_portfolio_returns_rejects_missing_column():
    df=pd.DataFrame({
        "AReturn":[0.01]
    })

    portfolio_weights={
        "AReturn":0.5,
        "BReturn":0.5
    }

    with pytest.raises(ValueError):
        calculate_portfolio_returns(df,portfolio_weights)


def test_calculate_portfolio_volatility():
    cov_matrix=pd.DataFrame(
        [
            [0.0004,0],
            [0,0.0009]
        ],
        index=["AReturn","BReturn"],
        columns=["AReturn","BReturn"]
    )

    portfolio_weights={
        "AReturn":0.5,
        "BReturn":0.5
    }

    result=calculate_portfolio_volatility(cov_matrix,portfolio_weights)

    expected=np.sqrt(
        (0.5**2)*0.0004+
        (0.5**2)*0.0009
    )

    assert result==pytest.approx(expected)


def test_calculate_inverse_volatility_weights():
    df=pd.DataFrame({
        "AReturn":[0,0.01,-0.01],
        "BReturn":[0,0.02,-0.02]
    })

    result=calculate_inverse_volatility_weights(df)

    assert result["AReturn"]==pytest.approx(2/3)
    assert result["BReturn"]==pytest.approx(1/3)
    assert sum(result.values())==pytest.approx(1)


def test_calculate_inverse_volatility_weights_rejects_zero_volatility():
    df=pd.DataFrame({
        "AReturn":[0,0,0],
        "BReturn":[0.01,0.02,0.03]
    })

    with pytest.raises(ValueError):
        calculate_inverse_volatility_weights(df)


def test_pre_oos_inverse_volatility_weights_ignore_future_returns():
    original=pd.DataFrame({
        "AReturn":[0.01,-0.01,0.02,-0.02,0.01,0.02],
        "BReturn":[0.005,-0.005,0.01,-0.01,0.005,0.01]
    })
    changed_future=original.copy()
    changed_future.loc[4:,"AReturn"]=[0.50,-0.50]
    changed_future.loc[4:,"BReturn"]=[-0.40,0.40]

    original_weights=calculate_pre_oos_inverse_volatility_weights(original,4)
    changed_weights=calculate_pre_oos_inverse_volatility_weights(changed_future,4)

    assert changed_weights==pytest.approx(original_weights)


def test_calculate_rebalancing_turnover():
    df=pd.DataFrame({
        "AReturn":[0,0.10],
        "BReturn":[0,0],
        "PortfolioReturn":[0,0.05]
    })

    portfolio_weights={
        "AReturn":0.5,
        "BReturn":0.5
    }

    result=calculate_rebalancing_turnover(df,portfolio_weights)

    expected=(
        abs(0.5-(0.5*1.10/1.05))
        +
        abs(0.5-(0.5/1.05))
    )

    assert result["Turnover"].iloc[0]==pytest.approx(1)
    assert result["Turnover"].iloc[1]==pytest.approx(expected)


def test_apply_portfolio_transaction_costs():
    df=pd.DataFrame({
        "PortfolioReturn":[0.01,0.02],
        "Turnover":[0.5,1.0]
    })

    result=apply_portfolio_transaction_costs(df,0.001)

    assert result["TransactionCost"].iloc[0]==pytest.approx(0.0005)
    assert result["TransactionCost"].iloc[1]==pytest.approx(0.001)

    assert result["NetPortfolioReturn"].iloc[0]==pytest.approx(0.0095)
    assert result["NetPortfolioReturn"].iloc[1]==pytest.approx(0.019)

    expected=(1.0095)*(1.019)

    assert result["NetPortfolioCumulativeValue"].iloc[-1]==pytest.approx(expected)


def test_apply_portfolio_transaction_costs_rejects_negative_cost():
    df=pd.DataFrame({
        "PortfolioReturn":[0.01],
        "Turnover":[0.1]
    })

    with pytest.raises(ValueError):
        apply_portfolio_transaction_costs(df,-0.001)


def test_calculate_portfolio_statistics():
    portfolio_returns=pd.Series([0.01,-0.005,0.02])
    turnover=pd.Series([0,0.2,0.1])
    transaction_cost=turnover*0.001
    net_returns=portfolio_returns-transaction_cost

    df=pd.DataFrame({
        "PortfolioReturn":portfolio_returns,
        "PortfolioCumulativeValue":(1+portfolio_returns).cumprod(),
        "Turnover":turnover,
        "TransactionCost":transaction_cost,
        "NetPortfolioReturn":net_returns,
        "NetPortfolioCumulativeValue":(1+net_returns).cumprod()
    })

    result=calculate_portfolio_statistics(df)
    core_statistics=calculate_performance_metrics(net_returns)

    assert result["observations"]==3
    assert result["gross_portfolio_return"]==pytest.approx(
        (1+portfolio_returns).prod()-1
    )
    assert result["net_portfolio_return"]==pytest.approx(
        core_statistics["total_return"]
    )
    assert result["annualised_return"]==pytest.approx(
        core_statistics["annualised_return"]
    )
    assert result["annualised_volatility"]==pytest.approx(
        core_statistics["annualised_volatility"]
    )
    assert result["sharpe_ratio"]==pytest.approx(
        core_statistics["sharpe_ratio"]
    )
    assert result["max_drawdown"]==pytest.approx(
        core_statistics["max_drawdown"]
    )
    assert result["total_turnover"]==pytest.approx(turnover.sum())
    assert result["total_transaction_cost"]==pytest.approx(
        transaction_cost.sum()
    )

def test_load_portfolio_prices(tmp_path):
    filepath=tmp_path/"portfolio_prices.csv"

    pd.DataFrame({
        "Date":["2024-01-01","2024-01-02"],
        "A":[100,101],
        "B":[50,51]
    }).to_csv(filepath,index=False)

    result=load_portfolio_prices(filepath)

    assert len(result)==2
    assert pd.api.types.is_datetime64_any_dtype(result["Date"])
    assert list(result.columns)==["Date","A","B"]


def test_load_portfolio_prices_requires_two_assets(tmp_path):
    filepath=tmp_path/"portfolio_prices.csv"

    pd.DataFrame({
        "Date":["2024-01-01"],
        "A":[100]
    }).to_csv(filepath,index=False)

    with pytest.raises(ValueError):
        load_portfolio_prices(filepath)


def test_save_portfolio_output(tmp_path):
    filepath=tmp_path/"portfolio_results.csv"

    df=pd.DataFrame({
        "Date":["2024-01-01"],
        "PortfolioReturn":[0.01]
    })

    save_portfolio_output(df,filepath)

    result=pd.read_csv(filepath)

    assert filepath.exists()
    assert len(result)==1
    assert result["PortfolioReturn"].iloc[0]==pytest.approx(0.01)
