import pandas as pd
from pathlib import Path
from src.analyse import (
    load_data,
    validate_data,
    calculate_returns,
)
from src.backtest import (
    calculate_backtest_statistics,
    run_backtest,
)


def calculate_momentum_signal(df, lookback):
    df = df.copy()

    previous_prices = df["Close"].shift(lookback)
    df["LookbackReturn"] = (df["Close"] / previous_prices) - 1
    df["Signal"] = (df["LookbackReturn"] > 0).astype(int)

    return df

def calculate_momentum_statistics(df, lookback, cost_rate):
    stats = calculate_backtest_statistics(df)

    stats["lookback"] = lookback
    stats["cost_rate"] = cost_rate

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

    df = run_backtest(df, cost_rate)

    stats = calculate_momentum_statistics(
        df,
        lookback,
        cost_rate,
    )

    print_momentum_report(stats)
    save_momentum_output(df)
    

if __name__ == "__main__":
    main()

    