import pandas as pd
import numpy as np


PERIODS_PER_YEAR=252
RISK_FREE_RATE=0.0
INITIAL_NAV=1.0


def validate_returns(returns):
    if isinstance(returns,pd.Series):
        returns=returns.copy()
    else:
        returns=pd.Series(returns)

    if returns.empty:
        raise ValueError(
            "Returns cannot be empty."
        )

    try:
        returns=pd.to_numeric(returns,errors="raise").astype(float)
    except (ValueError,TypeError):
        raise ValueError(
            "Returns must be numeric."
        )

    if not np.isfinite(returns.to_numpy()).all():
        raise ValueError(
            "Returns must be finite."
        )

    if (returns<=-1).any():
        raise ValueError(
            "Returns must be greater than -100%."
        )

    return returns


def calculate_cumulative_value(returns):
    returns=validate_returns(returns)

    cumulative_value=(1+returns).cumprod()*INITIAL_NAV

    return cumulative_value


def calculate_annualised_return(returns):
    returns=validate_returns(returns)

    observations=len(returns)
    cumulative_value=calculate_cumulative_value(returns)
    final_value=cumulative_value.iloc[-1]

    annualised_return=(final_value**(PERIODS_PER_YEAR/observations))-1

    return annualised_return


def calculate_annualised_volatility(returns):
    returns=validate_returns(returns)

    daily_volatility=returns.std()

    if pd.isna(daily_volatility):
        return np.nan

    annualised_volatility=daily_volatility*np.sqrt(PERIODS_PER_YEAR)

    return annualised_volatility


def calculate_sharpe_ratio(returns):
    returns=validate_returns(returns)

    daily_volatility=returns.std()

    if pd.isna(daily_volatility) or daily_volatility==0:
        return np.nan

    excess_returns=returns-RISK_FREE_RATE/PERIODS_PER_YEAR
    sharpe_ratio=excess_returns.mean()/daily_volatility*np.sqrt(PERIODS_PER_YEAR)

    return sharpe_ratio


def calculate_drawdown(returns):
    returns=validate_returns(returns)

    cumulative_value=calculate_cumulative_value(returns)
    running_max=cumulative_value.cummax().clip(lower=INITIAL_NAV)
    drawdown=cumulative_value/running_max-1

    return drawdown


def calculate_max_drawdown(returns):
    drawdown=calculate_drawdown(returns)

    max_drawdown=drawdown.min()

    return max_drawdown


def calculate_performance_metrics(returns):
    returns=validate_returns(returns)

    cumulative_value=calculate_cumulative_value(returns)

    statistics={
        "observations":len(returns),
        "total_return":cumulative_value.iloc[-1]-INITIAL_NAV,
        "annualised_return":calculate_annualised_return(returns),
        "annualised_volatility":calculate_annualised_volatility(returns),
        "sharpe_ratio":calculate_sharpe_ratio(returns),
        "max_drawdown":calculate_max_drawdown(returns)
    }

    return statistics