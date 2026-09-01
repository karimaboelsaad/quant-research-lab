from src.metrics import(
    calculate_cumulative_value,
    calculate_performance_metrics
)


def calculate_positions(df):
    """Delay each signal so today's information can only earn tomorrow's return."""
    df=df.copy()

    df["Position"]=df["Signal"].shift(1).fillna(0).astype(int)

    return df


def calculate_strategy_returns(df):
    df=df.copy()

    df["StrategyReturn"]=(df["Position"]*df["DailyReturn"]).fillna(0)
    df["StrategyCumulativeValue"]=calculate_cumulative_value(df["StrategyReturn"])

    return df


def calculate_transaction_costs(df,cost_rate,previous_position=0):
    if cost_rate<0:
        raise ValueError(
            "The transaction cost rate cannot be negative."
        )

    df=df.copy()

    previous_positions=df["Position"].shift(1,fill_value=previous_position)

    df["Turnover"]=(df["Position"]-previous_positions).abs()
    df["TransactionCost"]=df["Turnover"]*cost_rate
    df["NetStrategyReturn"]=df["StrategyReturn"]-df["TransactionCost"]
    df["NetStrategyCumulativeValue"]=calculate_cumulative_value(df["NetStrategyReturn"])

    return df

def calculate_backtest_statistics(df):
    if df.empty:
        raise ValueError(
            "Cannot calculate statistics for an empty DataFrame."
        )

    core_statistics=calculate_performance_metrics(df["NetStrategyReturn"])

    return {
        "observations":core_statistics["observations"],
        "buy_and_hold_return":df["CumulativeValue"].iloc[-1]-1,
        "gross_strategy_return":df["StrategyCumulativeValue"].iloc[-1]-1,
        "net_strategy_return":core_statistics["total_return"],
        "annualised_return":core_statistics["annualised_return"],
        "annualised_volatility":core_statistics["annualised_volatility"],
        "sharpe_ratio":core_statistics["sharpe_ratio"],
        "max_drawdown":core_statistics["max_drawdown"],
        "total_turnover":df["Turnover"].sum(),
        "trade_events":(df["Turnover"]>0).sum(),
        "time_in_market":df["Position"].abs().mean()
    }

def run_backtest(df,cost_rate):
    df=calculate_positions(df)
    df=calculate_strategy_returns(df)
    df=calculate_transaction_costs(df,cost_rate)

    return df
