import pandas as pd
from pathlib import Path
from statsmodels.tsa.stattools import(
    adfuller,
    coint
) 


def calculate_pair_regression(df):
    if df.empty:
        raise ValueError(
            "Dataframe cannot be empty."
        )
    
    df=df.copy()
    covariance=df["CloseB"].cov(df["CloseA"])
    b_variance=df["CloseB"].var()

    if b_variance==0:
        raise ValueError(
            "B variance cannot be 0."
        )
    
    beta=covariance/b_variance
    alpha=df["CloseA"].mean()-beta*df["CloseB"].mean()

    return {
        "alpha":alpha,
        "beta":beta
    }


def calculate_pair_spread(df, alpha, beta):
    if df.empty:
        raise ValueError(
            "Dataframe cannot be empty."
        )
    
    df=df.copy()
    df["PredictedA"]=alpha+df["CloseB"]*beta
    df["Spread"]=df["CloseA"]-df["PredictedA"]

    return df


def calculate_spread_z_score(df, lookback):
    if lookback<=1 or lookback>len(df):
        raise ValueError(
            "Lookback must be at least 2 and no larger than the size of the dataframe."
        )

    df=df.copy()

    df["SpreadRollingMean"]=df["Spread"].rolling(lookback).mean()
    df["SpreadRollingStd"]=df["Spread"].rolling(lookback).std()
    df["SpreadZScore"]=(df["Spread"]-df["SpreadRollingMean"])/df["SpreadRollingStd"]

    return df


def calculate_pair_signal(df, entry_threshold, exit_threshold):
    if entry_threshold<=0:
        raise ValueError(
            "Entry threshold must be positive."
        )

    if exit_threshold<0 or exit_threshold>=entry_threshold:
        raise ValueError(
            "Exit threshold must be non-negative and smaller than entry threshold."
        )

    df=df.copy()

    signals=[]
    state=0

    for i in range(len(df)):
        z_score=df["SpreadZScore"].iloc[i]

        if state==0 and z_score<=-entry_threshold:
            state=1
        elif state==0 and z_score>=entry_threshold:
            state=-1
        elif state==1 and z_score>=-exit_threshold:
            state=0
        elif state==-1 and z_score<=exit_threshold:
            state=0

        signals.append(state)

    df["Signal"]=signals

    return df


def calculate_pair_positions(df, beta):
    if df.empty:
        raise ValueError(
            "Dataframe cannot be empty."
        )
    
    df=df.copy()
    df["SpreadPosition"]=df["Signal"].shift(1).fillna(0)
    df["PositionA"]=df["SpreadPosition"]
    df["PositionB"]=df["SpreadPosition"]*(-beta)

    return df

 
def calculate_pair_returns(df):
    if df.empty:
        raise ValueError(
            "Dataframe cannot be empty."
        )
    
    df=df.copy()
    df["PairPnL"]=df["PositionA"]*(df["CloseA"]-df["CloseA"].shift(1))+df["PositionB"]*(df["CloseB"]-df["CloseB"].shift(1))
    df["GrossExposure"]=abs(df["PositionA"])*df["CloseA"].shift(1)+abs(df["PositionB"])*df["CloseB"].shift(1)
    df["StrategyReturn"]=df["PairPnL"]/df["GrossExposure"]
    df["StrategyReturn"]=df["StrategyReturn"].fillna(0)
    df["StrategyCumulativeValue"]=(1+df["StrategyReturn"]).cumprod()

    return df


def apply_pair_transaction_costs(df, cost_rate):
    if df.empty:
        raise ValueError(
            "Dataframe cannot be empty."
        )
    if cost_rate<0:
        raise ValueError(
            "Cost rate cannot be negative."
        )
    
    df=df.copy()
    df["Turnover"]=abs(df["SpreadPosition"]-df["SpreadPosition"].shift(1)).fillna(0)
    df["TransactionCost"]=df["Turnover"]*cost_rate
    df["NetStrategyReturn"]=df["StrategyReturn"]-df["TransactionCost"]
    df["NetStrategyCumulativeValue"]=(1+df["NetStrategyReturn"]).cumprod()

    return df


def test_spread_stationarity(df):
    if df.empty:
        raise ValueError(
            "Dataframe cannot be empty."
        )

    spread=df["Spread"].dropna()

    if len(spread)<3:
        raise ValueError(
            "Spread must contain at least 3 valid observations."
        )

    result=adfuller(spread)

    adf_statistic=result[0]
    p_value=result[1]
    critical_values=result[4]

    return {
        "adf_statistic":adf_statistic,
        "p_value":p_value,
        "critical_values":critical_values,
        "is_stationary":p_value<0.05
    }

def test_pair_cointegration(df):
    if df.empty:
        raise ValueError(
            "Dataframe cannot be empty."
        )

    pair=df[["CloseA","CloseB"]].dropna()

    if len(pair)<3:
        raise ValueError(
            "Pair must contain at least 3 valid observations."
        )

    result=coint(pair["CloseA"],pair["CloseB"])

    test_statistic=result[0]
    p_value=result[1]
    critical_values=result[2]

    return {
        "test_statistic":test_statistic,
        "p_value":p_value,
        "critical_values":critical_values,
        "is_cointegrated":p_value<0.05
    }



def run_pair_strategy(df, alpha, beta, lookback, entry_threshold, exit_threshold, cost_rate):
    if df.empty:
        raise ValueError(
            "Dataframe cannot be empty."
        )

    df=df.copy()

    df=calculate_pair_spread(df,alpha,beta)
    df=calculate_spread_z_score(df,lookback)
    df=calculate_pair_signal(df,entry_threshold,exit_threshold)
    df=calculate_pair_positions(df,beta)
    df=calculate_pair_returns(df)
    df=apply_pair_transaction_costs(df,cost_rate)

    return df


def calculate_pair_statistics(df):
    if df.empty:
        raise ValueError(
            "Dataframe cannot be empty."
        )

    observations=len(df)

    gross_return=df["StrategyCumulativeValue"].iloc[-1]-1
    net_return=df["NetStrategyCumulativeValue"].iloc[-1]-1

    annualised_return=(1+net_return)**(252/observations)-1

    daily_volatility=df["NetStrategyReturn"].std()
    annualised_volatility=daily_volatility*(252**0.5)

    if daily_volatility==0:
        sharpe_ratio=0
    else:
        sharpe_ratio=(df["NetStrategyReturn"].mean()/daily_volatility)*(252**0.5)

    running_max=df["NetStrategyCumulativeValue"].cummax()
    drawdown=df["NetStrategyCumulativeValue"]/running_max-1
    max_drawdown=drawdown.min()

    total_turnover=df["Turnover"].sum()
    trade_events=(df["SpreadPosition"].diff().fillna(0)!=0).sum()
    time_in_market=(df["SpreadPosition"]!=0).mean()

    return {
        "observations":observations,
        "gross_return":gross_return,
        "net_return":net_return,
        "annualised_return":annualised_return,
        "annualised_volatility":annualised_volatility,
        "sharpe_ratio":sharpe_ratio,
        "max_drawdown":max_drawdown,
        "total_turnover":total_turnover,
        "trade_events":trade_events,
        "time_in_market":time_in_market
    }


def calculate_pair_strategy_statistics(df, alpha, beta, lookback, entry_threshold, exit_threshold, cost_rate):
    stats=calculate_pair_statistics(df)

    stationarity=test_spread_stationarity(df)
    cointegration=test_pair_cointegration(df)

    stats["alpha"]=alpha
    stats["beta"]=beta
    stats["lookback"]=lookback
    stats["entry_threshold"]=entry_threshold
    stats["exit_threshold"]=exit_threshold
    stats["cost_rate"]=cost_rate
    stats["adf_statistic"]=stationarity["adf_statistic"]
    stats["adf_p_value"]=stationarity["p_value"]
    stats["is_stationary"]=stationarity["is_stationary"]
    stats["cointegration_statistic"]=cointegration["test_statistic"]
    stats["cointegration_p_value"]=cointegration["p_value"]
    stats["is_cointegrated"]=cointegration["is_cointegrated"]

    return stats


def print_pair_report(stats):
    print(f"""
PAIR TRADING STRATEGY REPORT
----------------------------
Alpha:                  {stats["alpha"]:.4f}
Beta:                   {stats["beta"]:.4f}
Lookback period:        {stats["lookback"]} periods
Entry threshold:        {stats["entry_threshold"]}
Exit threshold:         {stats["exit_threshold"]}
Transaction cost:       {stats["cost_rate"]:.2%}
Observations:           {stats["observations"]}
ADF statistic:          {stats["adf_statistic"]:.4f}
ADF p-value:            {stats["adf_p_value"]:.4f}
Stationary:             {stats["is_stationary"]}
Cointegration statistic:{stats["cointegration_statistic"]:.4f}
Cointegration p-value:  {stats["cointegration_p_value"]:.4f}
Cointegrated:           {stats["is_cointegrated"]}
Gross strategy return:  {stats["gross_return"]:.2%}
Net strategy return:    {stats["net_return"]:.2%}
Annualised return:      {stats["annualised_return"]:.2%}
Annualised volatility:  {stats["annualised_volatility"]:.2%}
Sharpe ratio:           {stats["sharpe_ratio"]:.4f}
Max drawdown:           {stats["max_drawdown"]:.2%}
Total turnover:         {stats["total_turnover"]:.0f}
Trade events:           {stats["trade_events"]}
Time in market:         {stats["time_in_market"]:.2%}
""")


def save_pair_output(df):
    output_path=Path("output/pair_results.csv")
    output_path.parent.mkdir(parents=True,exist_ok=True)

    output_columns=[
        "Date",
        "CloseA",
        "CloseB",
        "PredictedA",
        "Spread",
        "SpreadRollingMean",
        "SpreadRollingStd",
        "SpreadZScore",
        "Signal",
        "SpreadPosition",
        "PositionA",
        "PositionB",
        "PairPnL",
        "GrossExposure",
        "StrategyReturn",
        "StrategyCumulativeValue",
        "Turnover",
        "TransactionCost",
        "NetStrategyReturn",
        "NetStrategyCumulativeValue"
    ]

    df[output_columns].to_csv(output_path,index=False,float_format="%.6f")


def main():
    filepath="data/pair_prices.csv"
    lookback=20
    entry_threshold=2.0
    exit_threshold=0.5
    cost_rate=0.001

    df=pd.read_csv(filepath)
    df["Date"]=pd.to_datetime(df["Date"])

    regression=calculate_pair_regression(df)

    alpha=regression["alpha"]
    beta=regression["beta"]

    df=run_pair_strategy(df,alpha,beta,lookback,entry_threshold,exit_threshold,cost_rate)

    stats=calculate_pair_strategy_statistics(df,alpha,beta,lookback,entry_threshold,exit_threshold,cost_rate)

    print_pair_report(stats)
    save_pair_output(df)


if __name__=="__main__":
    main()