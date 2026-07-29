import pandas as pd
from pathlib import Path
from src.analyse import (
    load_data,
    validate_data,
    calculate_returns,
)

def calculate_momentum_signal(df, lookback):
    df = df.copy()

    previous_prices = df["Close"].shift(lookback)
    df["LookbackReturn"] = (df["Close"] / previous_prices) - 1
    df["Signal"] = (df["LookbackReturn"] > 0).astype(int)

    return df


def calculate_strategy_returns(df):
    df = df.copy()

    df["Position"]=df["Signal"].shift(1).fillna(0).astype(int)
    df["StrategyReturn"]=df["Position"]*df["DailyReturn"].fillna(0)
    df["StrategyCumulativeValue"]=(1 + df["StrategyReturn"]).cumprod()

    return df


def calculate_transaction_costs(df, cost_rate):
    if cost_rate < 0:
        raise ValueError("The transaction cost rate cannot be negative.")
    df = df.copy()

    df["Turnover"]=(df["Position"]-df["Position"].shift(1).fillna(0)).abs()
    df["TransactionCost"]=df["Turnover"]*cost_rate
    df["NetStrategyReturn"]=df["StrategyReturn"]-df["TransactionCost"]
    df["NetStrategyCumulativeValue"]=(1+df["NetStrategyReturn"]).cumprod()

    return df


def calculate_momentum_statistics(df, lookback, cost_rate):
    stats = {
        "lookback": lookback,
        "cost_rate": cost_rate,
        "observations": len(df),
        "buy_and_hold_return": df["CumulativeValue"].iloc[-1] - 1,
        "gross_strategy_return": df["StrategyCumulativeValue"].iloc[-1] - 1,
        "net_strategy_return": df["NetStrategyCumulativeValue"].iloc[-1] - 1,
        "total_turnover": df["Turnover"].sum(),
        "trade_events": (df["Turnover"] > 0).sum(),
        "time_in_market": df["Position"].mean()
    }

    return stats


def print_momentum_report(stats):
    print(f"""
MOMENTUM STRATEGY REPORT
------------------------
Lookback period:        {stats["lookback"]} periods
Transaction cost:       {stats["cost_rate"]:.2%}
Observations:           {stats["observations"]}
Buy-and-hold return:    {stats["buy_and_hold_return"]:.2%}
Gross strategy return:  {stats["gross_strategy_return"]:.2%}
Net strategy return:    {stats["net_strategy_return"]:.2%}
Total turnover:         {stats["total_turnover"]:.0f}
Trade events:           {stats["trade_events"]}
Time in market:         {stats["time_in_market"]:.2%}
""")
    

def save_momentum_output(df):
    output_path = Path("output/momentum_results.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_columns = [
        "Date",
        "Close",
        "DailyReturn",
        "CumulativeValue",
        "LookbackReturn",
        "Signal",
        "Position",
        "StrategyReturn",
        "StrategyCumulativeValue",
        "Turnover",
        "TransactionCost",
        "NetStrategyReturn",
        "NetStrategyCumulativeValue",
    ]


    df[output_columns].to_csv(
        output_path,
        index=False,
        float_format="%.6f",
    )


def main():
    filepath = "data/prices.csv"
    lookback = 3
    cost_rate = 0.001

    df = load_data(filepath)
    df = validate_data(df)
    df = calculate_returns(df)

    df = calculate_momentum_signal(df, lookback)
    df = calculate_strategy_returns(df)
    df = calculate_transaction_costs(df, cost_rate)

    stats = calculate_momentum_statistics(
        df,
        lookback,
        cost_rate,
    )

    print_momentum_report(stats)
    save_momentum_output(df)


if __name__ == "__main__":
    main()

    