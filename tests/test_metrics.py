import pandas as pd
import numpy as np
import pytest

from src.metrics import(
    validate_returns,
    calculate_cumulative_value,
    calculate_annualised_return,
    calculate_annualised_volatility,
    calculate_sharpe_ratio,
    calculate_drawdown,
    calculate_max_drawdown,
    calculate_performance_metrics
)


def test_validate_returns_accepts_valid_returns():
    returns=[0.01,-0.02,0.03]

    result=validate_returns(returns)

    assert isinstance(result,pd.Series)
    assert result.tolist()==[0.01,-0.02,0.03]


def test_validate_returns_rejects_empty_returns():
    with pytest.raises(ValueError):
        validate_returns([])


def test_validate_returns_rejects_nonnumeric_returns():
    returns=[0.01,"invalid",0.03]

    with pytest.raises(ValueError):
        validate_returns(returns)


def test_validate_returns_rejects_nan():
    returns=[0.01,np.nan,0.03]

    with pytest.raises(ValueError):
        validate_returns(returns)


def test_validate_returns_rejects_positive_infinity():
    returns=[0.01,np.inf,0.03]

    with pytest.raises(ValueError):
        validate_returns(returns)


def test_validate_returns_rejects_negative_infinity():
    returns=[0.01,-np.inf,0.03]

    with pytest.raises(ValueError):
        validate_returns(returns)


def test_validate_returns_rejects_negative_one():
    returns=[0.01,-1.0,0.03]

    with pytest.raises(ValueError):
        validate_returns(returns)


def test_validate_returns_rejects_less_than_negative_one():
    returns=[0.01,-1.2,0.03]

    with pytest.raises(ValueError):
        validate_returns(returns)


def test_validate_returns_does_not_modify_input_series():
    returns=pd.Series([0.01,-0.02,0.03])
    original=returns.copy()

    validate_returns(returns)

    pd.testing.assert_series_equal(returns,original)


def test_calculate_cumulative_value():
    returns=pd.Series([0.10,-0.10])

    result=calculate_cumulative_value(returns)

    expected=pd.Series([1.10,0.99])

    pd.testing.assert_series_equal(result,expected)


def test_calculate_annualized_return():
    returns=pd.Series([0.01,0.02,-0.01])

    cumulative_value=(1.01*1.02*0.99)
    expected=(cumulative_value**(252/3))-1

    result=calculate_annualised_return(returns)

    assert np.isclose(result,expected)


def test_calculate_annualized_volatility():
    returns=pd.Series([0.01,-0.02,0.03,-0.01])

    expected=returns.std()*np.sqrt(252)

    result=calculate_annualised_volatility(returns)

    assert np.isclose(result,expected)


def test_calculate_annualized_volatility_single_observation_is_nan():
    returns=pd.Series([0.01])

    result=calculate_annualised_volatility(returns)

    assert np.isnan(result)


def test_calculate_sharpe_ratio():
    returns=pd.Series([0.01,-0.02,0.03,-0.01])

    expected=returns.mean()/returns.std()*np.sqrt(252)

    result=calculate_sharpe_ratio(returns)

    assert np.isclose(result,expected)


def test_calculate_sharpe_ratio_zero_volatility_is_nan():
    returns=pd.Series([0.01,0.01,0.01])

    result=calculate_sharpe_ratio(returns)

    assert np.isnan(result)


def test_calculate_sharpe_ratio_single_observation_is_nan():
    returns=pd.Series([0.01])

    result=calculate_sharpe_ratio(returns)

    assert np.isnan(result)


def test_first_loss_produces_drawdown_from_initial_nav():
    returns=pd.Series([-0.10,0.0])

    result=calculate_drawdown(returns)

    assert np.isclose(result.iloc[0],-0.10)


def test_first_loss_produces_correct_max_drawdown():
    returns=pd.Series([-0.10,0.0])

    result=calculate_max_drawdown(returns)

    assert np.isclose(result,-0.10)


def test_drawdown_uses_previous_running_peak():
    returns=pd.Series([0.10,-0.20,0.10])

    result=calculate_drawdown(returns)

    expected_first=0.0
    expected_second=0.88/1.10-1
    expected_third=0.968/1.10-1

    assert np.isclose(result.iloc[0],expected_first)
    assert np.isclose(result.iloc[1],expected_second)
    assert np.isclose(result.iloc[2],expected_third)


def test_calculate_performance_metrics():
    returns=pd.Series([0.01,-0.02,0.03])

    statistics=calculate_performance_metrics(returns)

    assert statistics["observations"]==3
    assert np.isclose(
        statistics["total_return"],
        (1.01*0.98*1.03)-1
    )
    assert np.isclose(
        statistics["annualised_return"],
        calculate_annualised_return(returns)
    )
    assert np.isclose(
        statistics["annualised_volatility"],
        calculate_annualised_volatility(returns)
    )
    assert np.isclose(
        statistics["sharpe_ratio"],
        calculate_sharpe_ratio(returns)
    )
    assert np.isclose(
        statistics["max_drawdown"],
        calculate_max_drawdown(returns)
    )


def test_performance_metrics_use_canonical_keys():
    returns=pd.Series([0.01,-0.02,0.03])

    statistics=calculate_performance_metrics(returns)

    assert set(statistics.keys())=={
        "observations",
        "total_return",
        "annualised_return",
        "annualised_volatility",
        "sharpe_ratio",
        "max_drawdown"
    }