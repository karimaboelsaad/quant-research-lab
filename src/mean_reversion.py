from pathlib import Path

from src.analyse import(
    load_data,
    validate_data,
    calculate_returns
)

from src.backtest import(
    calculate_backtest_statistics,
    run_backtest
)


def calculate_z_score(df, lookback):
    if lookback<=1 or lookback>len(df):
        raise ValueError(
            "Lookback must be at least 2 and no larger than the size of the dataframe."
        )

    df=df.copy()

    df["RollingMean"]=df["Close"].rolling(lookback).mean()
    df["RollingStd"]=df["Close"].rolling(lookback).std()
    df["ZScore"]=(df["Close"]-df["RollingMean"])/df["RollingStd"]

    return df


def calculate_mean_reversion_signal(df, entry_threshold, exit_threshold):
    if entry_threshold>=exit_threshold:
        raise ValueError(
            "Exit threshold must be larger than entry threshold."
        )

    df=df.copy()

    signals=[]
    state=0

    for i in range(len(df)):
        if state==0 and df["ZScore"].iloc[i]<=entry_threshold:
            state=1
            signals.append(1)

        elif state==1 and df["ZScore"].iloc[i]>=exit_threshold:
            state=0
            signals.append(0)

        else:
            signals.append(state)

    df["Signal"]=signals

    return df


def run_mean_reversion(df, lookback, entry_threshold, exit_threshold, cost_rate):
    df=calculate_z_score(df,lookback)
    df=calculate_mean_reversion_signal(df,entry_threshold,exit_threshold)
    df=run_backtest(df,cost_rate)

    return df


def calculate_mean_reversion_statistics(df, lookback, entry_threshold, exit_threshold, cost_rate):
    stats=calculate_backtest_statistics(df)

    stats["lookback"]=lookback
    stats["entry_threshold"]=entry_threshold
    stats["exit_threshold"]=exit_threshold
    stats["cost_rate"]=cost_rate

    return stats


def print_mean_reversion_report(stats):
    print(f"""
MEAN REVERSION STRATEGY REPORT
------------------------------
Lookback period:        {stats["lookback"]} periods
Entry threshold:        {stats["entry_threshold"]}
Exit threshold:         {stats["exit_threshold"]}
Transaction cost:       {stats["cost_rate"]:.2%}
Observations:           {stats["observations"]}
Buy-and-hold return:    {stats["buy_and_hold_return"]:.2%}
Gross strategy return:  {stats["gross_strategy_return"]:.2%}
Net strategy return:    {stats["net_strategy_return"]:.2%}
Total turnover:         {stats["total_turnover"]:.0f}
Trade events:           {stats["trade_events"]}
Time in market:         {stats["time_in_market"]:.2%}
""")


def save_mean_reversion_output(df):
    output_path=Path("output/mean_reversion_results.csv")
    output_path.parent.mkdir(parents=True,exist_ok=True)

    output_columns=[
        "Date",
        "Close",
        "DailyReturn",
        "CumulativeValue",
        "RollingMean",
        "RollingStd",
        "ZScore",
        "Signal",
        "Position",
        "StrategyReturn",
        "StrategyCumulativeValue",
        "Turnover",
        "TransactionCost",
        "NetStrategyReturn",
        "NetStrategyCumulativeValue"
    ]

    df[output_columns].to_csv(output_path,index=False,float_format="%.6f")


def main():
    filepath="data/prices.csv"
    lookback=3
    entry_threshold=-1.0
    exit_threshold=0.0
    cost_rate=0.001

    df=load_data(filepath)
    df=validate_data(df)
    df=calculate_returns(df)
    df=run_mean_reversion(df,lookback,entry_threshold,exit_threshold,cost_rate)

    stats=calculate_mean_reversion_statistics(df,lookback,entry_threshold,exit_threshold,cost_rate)

    print_mean_reversion_report(stats)
    save_mean_reversion_output(df)


if __name__=="__main__":
    main()