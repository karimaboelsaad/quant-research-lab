import math

def calculate_positions(df):
    df = df.copy()

    df["Position"]=df["Signal"].shift(1).fillna(0).astype(int)
    return df


def calculate_strategy_returns(df):
    df = df.copy()

    df["StrategyReturn"]=(df["Position"]*df["DailyReturn"]).fillna(0)
    df["StrategyCumulativeValue"]=(1+df["StrategyReturn"]).cumprod()
    return df

def calculate_transaction_costs(df, cost_rate):
    if cost_rate<0:
        raise ValueError("The transaction cost rate cannot be negative.")
    df = df.copy()

    df["Turnover"]=(df["Position"]-df["Position"].shift(1).fillna(0)).abs()
    df["TransactionCost"]=df["Turnover"]*cost_rate
    df["NetStrategyReturn"]=df["StrategyReturn"]-df["TransactionCost"]
    df["NetStrategyCumulativeValue"]=(1+df["NetStrategyReturn"]).cumprod()

    return df

def calculate_backtest_statistics(df):
    if df.empty:
        raise ValueError(
            "Cannot calculate statistics for an empty DataFrame."
        )

    observations = len(df)

    net_returns = df["NetStrategyReturn"]

    net_cumulative_value = df["NetStrategyCumulativeValue"]

    final_net_value = net_cumulative_value.iloc[-1]

    annualised_return = (final_net_value**(252/observations)-1)

    daily_volatility = net_returns.std()

    if math.isnan(daily_volatility):
        daily_volatility = 0.0

    annualised_volatility = (daily_volatility*math.sqrt(252))

    if daily_volatility == 0:
        sharpe_ratio = 0.0
    else:
        sharpe_ratio = net_returns.mean()/daily_volatility*math.sqrt(252)

    running_peak = net_cumulative_value.cummax().clip(lower=1.0)

    drawdown = net_cumulative_value/running_peak-1

    maximum_drawdown = drawdown.min()

    return {
        "observations": observations,
        "buy_and_hold_return": (df["CumulativeValue"].iloc[-1] - 1),
        "gross_strategy_return": (df["StrategyCumulativeValue"].iloc[-1] - 1),
        "net_strategy_return": (final_net_value - 1),
        "annualised_return": annualised_return,
        "annualised_volatility": annualised_volatility,
        "sharpe_ratio": sharpe_ratio,
        "maximum_drawdown": maximum_drawdown,
        "total_turnover": df["Turnover"].sum(),
        "trade_events": (df["Turnover"] > 0).sum(),
        "time_in_market": (df["Position"].abs().mean()),
    }

def run_backtest(df, cost_rate):
    df = calculate_positions(df)
    df = calculate_strategy_returns(df)
    df = calculate_transaction_costs(df, cost_rate)

    return df