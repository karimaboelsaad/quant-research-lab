from pathlib import Path

import pandas as pd
import numpy as np

from src.metrics import(
    calculate_cumulative_value,
    calculate_performance_metrics
)


def calculate_rolling_strategy_scores(df,assets,lookback):
    if df.empty:
        raise ValueError(
            "Dataframe cannot be empty."
        )

    if not assets:
        raise ValueError(
            "Assets cannot be empty."
        )

    if lookback<2 or lookback>len(df):
        raise ValueError(
            "Lookback must be at least 2 and no larger than the size of the dataframe."
        )

    df=df.copy()

    for asset in assets:
        momentum_return=f"{asset}_MomentumReturn"
        mean_reversion_return=f"{asset}_MeanReversionReturn"

        if momentum_return not in df.columns or mean_reversion_return not in df.columns:
            raise ValueError(
                "Every asset must have momentum and mean reversion return columns."
            )

        for strategy in ["Momentum","MeanReversion"]:
            return_column=f"{asset}_{strategy}Return"
            past_returns=df[return_column].shift(1)

            df[f"{asset}_{strategy}PerformanceMean"]=past_returns.rolling(lookback).mean()
            df[f"{asset}_{strategy}PerformanceStd"]=past_returns.rolling(lookback).std()

            volatility=df[f"{asset}_{strategy}PerformanceStd"].replace(0,np.nan)

            df[f"{asset}_{strategy}Score"]=(df[f"{asset}_{strategy}PerformanceMean"]/volatility)*np.sqrt(252)

    return df


def select_best_strategy_per_asset(df,assets):
    if df.empty:
        raise ValueError(
            "Dataframe cannot be empty."
        )

    if not assets:
        raise ValueError(
            "Assets cannot be empty."
        )

    df=df.copy()

    for asset in assets:
        momentum_score=f"{asset}_MomentumScore"
        mean_reversion_score=f"{asset}_MeanReversionScore"

        if momentum_score not in df.columns or mean_reversion_score not in df.columns:
            raise ValueError(
                "Every asset must have momentum and mean reversion score columns."
            )

        momentum_comparison=df[momentum_score].fillna(-np.inf)
        mean_reversion_comparison=df[mean_reversion_score].fillna(-np.inf)

        df[f"{asset}_MomentumSelected"]=(
            (df[momentum_score]>0)&
            (momentum_comparison>=mean_reversion_comparison)
        ).astype(int)

        df[f"{asset}_MeanReversionSelected"]=(
            (df[mean_reversion_score]>0)&
            (mean_reversion_comparison>momentum_comparison)
        ).astype(int)

    return df


def calculate_selected_strategy_returns(df,assets):
    if df.empty:
        raise ValueError(
            "Dataframe cannot be empty."
        )

    if not assets:
        raise ValueError(
            "Assets cannot be empty."
        )

    df=df.copy()

    for asset in assets:
        momentum_return=f"{asset}_MomentumReturn"
        mean_reversion_return=f"{asset}_MeanReversionReturn"
        momentum_selected=f"{asset}_MomentumSelected"
        mean_reversion_selected=f"{asset}_MeanReversionSelected"

        required_columns=[
            momentum_return,
            mean_reversion_return,
            momentum_selected,
            mean_reversion_selected
        ]

        for column in required_columns:
            if column not in df.columns:
                raise ValueError(
                    f"Dataframe must contain {column}."
                )

        df[f"{asset}_SelectedStrategyReturn"]=df[momentum_selected]*df[momentum_return].fillna(0)+df[mean_reversion_selected]*df[mean_reversion_return].fillna(0)

    return df


def calculate_dynamic_strategy_weights(df,assets):
    if df.empty:
        raise ValueError(
            "Dataframe cannot be empty."
        )

    if not assets:
        raise ValueError(
            "Assets cannot be empty."
        )

    df=df.copy()

    active_assets=pd.Series(0,index=df.index,dtype=float)

    for asset in assets:
        momentum_selected=f"{asset}_MomentumSelected"
        mean_reversion_selected=f"{asset}_MeanReversionSelected"

        if momentum_selected not in df.columns or mean_reversion_selected not in df.columns:
            raise ValueError(
                "Every asset must have strategy selection columns."
            )

        active_assets+=df[momentum_selected]+df[mean_reversion_selected]

    denominator=active_assets.replace(0,np.nan)

    for asset in assets:
        asset_selected=df[f"{asset}_MomentumSelected"]+df[f"{asset}_MeanReversionSelected"]

        df[f"{asset}_Weight"]=(asset_selected/denominator).fillna(0)
        df[f"{asset}_MomentumWeight"]=df[f"{asset}_Weight"]*df[f"{asset}_MomentumSelected"]
        df[f"{asset}_MeanReversionWeight"]=df[f"{asset}_Weight"]*df[f"{asset}_MeanReversionSelected"]

    df["DynamicGrossExposure"]=0.0

    for asset in assets:
        df["DynamicGrossExposure"]+=df[f"{asset}_Weight"]

    return df


def calculate_dynamic_portfolio_returns(df,assets):
    if df.empty:
        raise ValueError(
            "Dataframe cannot be empty."
        )

    if not assets:
        raise ValueError(
            "Assets cannot be empty."
        )

    df=df.copy()

    df["DynamicPortfolioReturn"]=0.0

    for asset in assets:
        selected_return=f"{asset}_SelectedStrategyReturn"
        weight=f"{asset}_Weight"

        if selected_return not in df.columns or weight not in df.columns:
            raise ValueError(
                "Every asset must have a selected strategy return and weight column."
            )

        df["DynamicPortfolioReturn"]+=df[weight]*df[selected_return].fillna(0)

    df["DynamicPortfolioCumulativeValue"]=calculate_cumulative_value(
        df["DynamicPortfolioReturn"]
    )

    return df


def calculate_dynamic_turnover(df,assets):
    if df.empty:
        raise ValueError(
            "Dataframe cannot be empty."
        )

    if not assets:
        raise ValueError(
            "Assets cannot be empty."
        )

    if "DynamicPortfolioReturn" not in df.columns:
        raise ValueError(
            "Dataframe must contain DynamicPortfolioReturn."
        )

    df=df.copy()

    previous_portfolio_growth=(1+df["DynamicPortfolioReturn"].shift(1)).fillna(1)

    if (previous_portfolio_growth<=0).any():
        raise ValueError(
            "Portfolio value cannot fall to zero or below."
        )

    df["DynamicTurnover"]=0.0

    for asset in assets:
        for strategy in ["Momentum","MeanReversion"]:
            weight_column=f"{asset}_{strategy}Weight"
            return_column=f"{asset}_{strategy}Return"

            if weight_column not in df.columns or return_column not in df.columns:
                raise ValueError(
                    "Every strategy must have weight and return columns."
                )

            previous_weight=df[weight_column].shift(1).fillna(0)
            previous_return=df[return_column].shift(1).fillna(0)

            drifted_weight=previous_weight*(1+previous_return)/previous_portfolio_growth

            df["DynamicTurnover"]+=abs(df[weight_column]-drifted_weight)

    return df


def apply_dynamic_transaction_costs(df,cost_rate):
    if df.empty:
        raise ValueError(
            "Dataframe cannot be empty."
        )

    if cost_rate<0:
        raise ValueError(
            "Cost rate cannot be negative."
        )

    if "DynamicPortfolioReturn" not in df.columns or "DynamicTurnover" not in df.columns:
        raise ValueError(
            "Dataframe must contain DynamicPortfolioReturn and DynamicTurnover."
        )

    df=df.copy()

    df["DynamicTransactionCost"]=df["DynamicTurnover"]*cost_rate
    df["NetDynamicPortfolioReturn"]=df["DynamicPortfolioReturn"]-df["DynamicTransactionCost"]
    df["NetDynamicPortfolioCumulativeValue"]=calculate_cumulative_value(
        df["NetDynamicPortfolioReturn"]
    )

    return df


def run_dynamic_portfolio(df,assets,lookback,cost_rate):
    df=calculate_rolling_strategy_scores(df,assets,lookback)
    df=select_best_strategy_per_asset(df,assets)
    df=calculate_selected_strategy_returns(df,assets)
    df=calculate_dynamic_strategy_weights(df,assets)
    df=calculate_dynamic_portfolio_returns(df,assets)
    df=calculate_dynamic_turnover(df,assets)
    df=apply_dynamic_transaction_costs(df,cost_rate)

    return df


def calculate_dynamic_portfolio_statistics(df,lookback,cost_rate):
    if df.empty:
        raise ValueError(
            "Dataframe cannot be empty."
        )

    required_columns=[
        "DynamicGrossExposure",
        "DynamicPortfolioReturn",
        "DynamicTurnover",
        "DynamicTransactionCost",
        "NetDynamicPortfolioReturn"
    ]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(
                f"Dataframe must contain {column}."
            )

    active_rows=df["DynamicGrossExposure"]>0

    if not active_rows.any():
        raise ValueError(
            "Portfolio was never active."
        )

    first_active=np.flatnonzero(active_rows.to_numpy())[0]
    evaluation_df=df.iloc[first_active:].copy()

    core_statistics=calculate_performance_metrics(
        evaluation_df["NetDynamicPortfolioReturn"]
    )

    gross_cumulative_value=calculate_cumulative_value(
        evaluation_df["DynamicPortfolioReturn"]
    )

    time_in_market=(evaluation_df["DynamicGrossExposure"]>0).mean()

    stats={
        "lookback":lookback,
        "cost_rate":cost_rate,
        "observations":core_statistics["observations"],
        "gross_portfolio_return":gross_cumulative_value.iloc[-1]-1,
        "net_portfolio_return":core_statistics["total_return"],
        "annualised_return":core_statistics["annualised_return"],
        "annualised_volatility":core_statistics["annualised_volatility"],
        "sharpe_ratio":core_statistics["sharpe_ratio"],
        "max_drawdown":core_statistics["max_drawdown"],
        "total_turnover":evaluation_df["DynamicTurnover"].sum(),
        "total_transaction_cost":evaluation_df["DynamicTransactionCost"].sum(),
        "time_in_market":time_in_market
    }

    return stats


def print_dynamic_portfolio_report(stats):
    print(f"""
DYNAMIC PORTFOLIO REPORT
------------------------
Score lookback:          {stats["lookback"]} periods
Transaction cost:        {stats["cost_rate"]:.2%}
Observations:            {stats["observations"]}
Gross portfolio return:  {stats["gross_portfolio_return"]:.2%}
Net portfolio return:    {stats["net_portfolio_return"]:.2%}
Annualised return:       {stats["annualised_return"]:.2%}
Annualised volatility:   {stats["annualised_volatility"]:.2%}
Sharpe ratio:            {stats["sharpe_ratio"]:.2f}
Maximum drawdown:        {stats["max_drawdown"]:.2%}
Total turnover:          {stats["total_turnover"]:.2f}
Total transaction cost:  {stats["total_transaction_cost"]:.2%}
Time in market:          {stats["time_in_market"]:.2%}
""")


def load_dynamic_strategy_returns(filepath,assets):
    df=pd.read_csv(filepath)

    if df.empty:
        raise ValueError(
            "Dataframe cannot be empty."
        )

    if "Date" not in df.columns:
        raise ValueError(
            "Dataframe must contain Date."
        )

    for asset in assets:
        momentum_return=f"{asset}_MomentumReturn"
        mean_reversion_return=f"{asset}_MeanReversionReturn"

        if momentum_return not in df.columns or mean_reversion_return not in df.columns:
            raise ValueError(
                "Every asset must have momentum and mean reversion return columns."
            )

    df["Date"]=pd.to_datetime(df["Date"])

    if df["Date"].duplicated().any():
        raise ValueError(
            "Dates cannot contain duplicates."
        )

    df=df.sort_values("Date").reset_index(drop=True)

    return df


def save_dynamic_portfolio_output(df,assets):
    output_path=Path("output/dynamic_portfolio_results.csv")
    output_path.parent.mkdir(parents=True,exist_ok=True)

    output_columns=["Date"]

    for asset in assets:
        output_columns+=[
            f"{asset}_MomentumReturn",
            f"{asset}_MomentumPerformanceMean",
            f"{asset}_MomentumPerformanceStd",
            f"{asset}_MomentumScore",
            f"{asset}_MeanReversionReturn",
            f"{asset}_MeanReversionPerformanceMean",
            f"{asset}_MeanReversionPerformanceStd",
            f"{asset}_MeanReversionScore",
            f"{asset}_MomentumSelected",
            f"{asset}_MeanReversionSelected",
            f"{asset}_SelectedStrategyReturn",
            f"{asset}_Weight",
            f"{asset}_MomentumWeight",
            f"{asset}_MeanReversionWeight"
        ]

    output_columns+=[
        "DynamicGrossExposure",
        "DynamicPortfolioReturn",
        "DynamicPortfolioCumulativeValue",
        "DynamicTurnover",
        "DynamicTransactionCost",
        "NetDynamicPortfolioReturn",
        "NetDynamicPortfolioCumulativeValue"
    ]

    df[output_columns].to_csv(output_path,index=False,float_format="%.6f")


def main():
    filepath="data/dynamic_strategy_returns.csv"
    assets=["A","B","C"]
    lookback=60
    cost_rate=0.001

    df=load_dynamic_strategy_returns(filepath,assets)
    df=run_dynamic_portfolio(df,assets,lookback,cost_rate)

    stats=calculate_dynamic_portfolio_statistics(df,lookback,cost_rate)

    print_dynamic_portfolio_report(stats)
    save_dynamic_portfolio_output(df,assets)


if __name__=="__main__":
    main()