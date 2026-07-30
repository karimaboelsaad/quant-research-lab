import pandas as pd

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
    stats = {
        "observations": len(df),
        "buy_and_hold_return": df["CumulativeValue"].iloc[-1] - 1,
        "gross_strategy_return": df["StrategyCumulativeValue"].iloc[-1] - 1,
        "net_strategy_return": df["NetStrategyCumulativeValue"].iloc[-1] - 1,
        "total_turnover": df["Turnover"].sum(),
        "trade_events": (df["Turnover"] > 0).sum(),
        "time_in_market": df["Position"].abs().mean()
    }

    return stats

def run_backtest(df, cost_rate):
    df = calculate_positions(df)
    df = calculate_strategy_returns(df)
    df = calculate_transaction_costs(df, cost_rate)

    return df