import pandas as pd
import numpy as np
from pathlib import Path

from src.data import(
    calculate_price_returns,
    load_multi_asset_prices
)

from src.metrics import(
    calculate_cumulative_value,
    calculate_performance_metrics
)


def calculate_asset_returns(df):
    if df.empty:
        raise ValueError(
            "Dataframe cannot be empty."
        )

    asset_columns=[
        column
        for column in df.columns
        if column!="Date"
    ]

    if len(asset_columns)<2:
        raise ValueError(
            "Dataframe must contain at least 2 asset price columns."
        )

    df=calculate_price_returns(df,asset_columns)

    return df


def calculate_correlation_matrix(df):
    if df.empty:
        raise ValueError(
            "Dataframe cannot be empty."
        )

    df=df.copy()
    return_columns=[column for column in df.columns if column.endswith("Return")]

    if len(return_columns)<2:
        raise ValueError(
            "There must be at least 2 return columns."
        )

    corr_matrix=df[return_columns].corr()

    return corr_matrix


def calculate_covariance_matrix(df):
    if df.empty:
        raise ValueError(
            "Dataframe cannot be empty."
        )

    df=df.copy()
    return_columns=[column for column in df.columns if column.endswith("Return")]

    if len(return_columns)<2:
        raise ValueError(
            "There must be at least 2 return columns."
        )

    cov_matrix=df[return_columns].cov()

    return cov_matrix


def calculate_equal_weights(df):
    if df.empty:
        raise ValueError(
            "Dataframe cannot be empty."
        )

    df=df.copy()
    return_columns=[column for column in df.columns if column.endswith("Return")]

    if len(return_columns)<2:
        raise ValueError(
            "There must be at least 2 return columns."
        )

    portfolio_weights={}

    for column in return_columns:
        portfolio_weights[column]=1/len(return_columns)

    return portfolio_weights


def calculate_portfolio_returns(df,portfolio_weights):
    if df.empty:
        raise ValueError(
            "Dataframe cannot be empty."
        )

    if not portfolio_weights:
        raise ValueError(
            "Portfolio weights cannot be empty."
        )

    for column in portfolio_weights:
        if column not in df.columns:
            raise ValueError(
                "Every portfolio weight must correspond to a return column in the dataframe."
            )

    if not np.isclose(sum(portfolio_weights.values()),1):
        raise ValueError(
            "Portfolio weights must sum to 1."
        )

    df=df.copy()

    df["PortfolioReturn"]=0.0

    for column,weight in portfolio_weights.items():
        df["PortfolioReturn"]+=df[column].fillna(0)*weight

    df["PortfolioCumulativeValue"]=calculate_cumulative_value(
        df["PortfolioReturn"]
    )

    return df


def calculate_portfolio_volatility(cov_matrix,portfolio_weights):
    if cov_matrix.empty:
        raise ValueError(
            "Covariance matrix cannot be empty."
        )

    if not portfolio_weights:
        raise ValueError(
            "Portfolio weights cannot be empty."
        )

    for column in cov_matrix.columns:
        if column not in portfolio_weights:
            raise ValueError(
                "Every covariance matrix column must have a corresponding portfolio weight."
            )

    if not np.isclose(sum(portfolio_weights.values()),1):
        raise ValueError(
            "Portfolio weights must sum to 1."
        )

    weights=np.array([
        portfolio_weights[column]
        for column in cov_matrix.columns
    ])

    portfolio_variance=weights @ cov_matrix.values @ weights
    portfolio_volatility=np.sqrt(portfolio_variance)

    return portfolio_volatility


def calculate_inverse_volatility_weights(df):
    if df.empty:
        raise ValueError(
            "Dataframe cannot be empty."
        )

    return_columns=[column for column in df.columns if column.endswith("Return")]

    if len(return_columns)<2:
        raise ValueError(
            "There must be at least 2 return columns."
        )

    inverse_volatilities={}

    for column in return_columns:
        volatility=df[column].std()

        if pd.isna(volatility) or volatility==0:
            raise ValueError(
                "Asset volatility must be positive and valid."
            )

        inverse_volatilities[column]=1/volatility

    total_inverse_volatility=sum(inverse_volatilities.values())

    portfolio_weights={}

    for column in return_columns:
        portfolio_weights[column]=inverse_volatilities[column]/total_inverse_volatility

    return portfolio_weights


def calculate_pre_oos_inverse_volatility_weights(df,oos_start_position):
    if not 1<oos_start_position<len(df):
        raise ValueError(
            "OOS start must leave at least two prior rows and one OOS row."
        )

    estimation_df=df.iloc[:oos_start_position].copy()

    return calculate_inverse_volatility_weights(estimation_df)


def calculate_rebalancing_turnover(df,portfolio_weights):
    """Measure trades from drifted holdings back to fixed target weights."""
    if df.empty:
        raise ValueError(
            "Dataframe cannot be empty."
        )

    if not portfolio_weights:
        raise ValueError(
            "Portfolio weights cannot be empty."
        )

    for column in portfolio_weights:
        if column not in df.columns:
            raise ValueError(
                "Every portfolio weight must correspond to a return column in the dataframe."
            )

    if "PortfolioReturn" not in df.columns:
        raise ValueError(
            "Dataframe must contain PortfolioReturn."
        )

    if not np.isclose(sum(portfolio_weights.values()),1):
        raise ValueError(
            "Portfolio weights must sum to 1."
        )

    df=df.copy()

    df["Turnover"]=0.0

    for column,weight in portfolio_weights.items():
        drifted_weight=weight*(1+df[column].fillna(0))/(1+df["PortfolioReturn"])

        df["Turnover"]+=abs(weight-drifted_weight)

    df.loc[df.index[0],"Turnover"]=sum(abs(weight) for weight in portfolio_weights.values())

    return df


def apply_portfolio_transaction_costs(df,cost_rate):
    if df.empty:
        raise ValueError(
            "Dataframe cannot be empty."
        )

    if cost_rate<0:
        raise ValueError(
            "Cost rate cannot be negative."
        )

    if "PortfolioReturn" not in df.columns or "Turnover" not in df.columns:
        raise ValueError(
            "Dataframe must contain PortfolioReturn and Turnover."
        )

    df=df.copy()

    df["TransactionCost"]=df["Turnover"]*cost_rate
    df["NetPortfolioReturn"]=df["PortfolioReturn"]-df["TransactionCost"]
    df["NetPortfolioCumulativeValue"]=calculate_cumulative_value(
        df["NetPortfolioReturn"]
    )

    return df


def calculate_portfolio_statistics(df):
    if df.empty:
        raise ValueError(
            "Dataframe cannot be empty."
        )

    required_columns=[
        "PortfolioReturn",
        "PortfolioCumulativeValue",
        "Turnover",
        "TransactionCost",
        "NetPortfolioReturn",
        "NetPortfolioCumulativeValue"
    ]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(
                f"Dataframe must contain {column}."
            )

    core_statistics=calculate_performance_metrics(
        df["NetPortfolioReturn"]
    )

    gross_cumulative_value=calculate_cumulative_value(
        df["PortfolioReturn"]
    )

    statistics={
        "observations":core_statistics["observations"],
        "gross_portfolio_return":gross_cumulative_value.iloc[-1]-1,
        "net_portfolio_return":core_statistics["total_return"],
        "annualised_return":core_statistics["annualised_return"],
        "annualised_volatility":core_statistics["annualised_volatility"],
        "sharpe_ratio":core_statistics["sharpe_ratio"],
        "max_drawdown":core_statistics["max_drawdown"],
        "total_turnover":df["Turnover"].sum(),
        "total_transaction_cost":df["TransactionCost"].sum()
    }

    return statistics


def load_portfolio_prices(path):
    df=load_multi_asset_prices(path)

    return df


def save_portfolio_output(df,path):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    df.to_csv(path,index=False)


def main():
    input_path="data/portfolio_prices.csv"
    output_path="output/portfolio_results.csv"

    cost_rate=.001

    df=load_portfolio_prices(input_path)
    df=calculate_asset_returns(df)

    portfolio_weights=calculate_equal_weights(df)

    cov_matrix=calculate_covariance_matrix(df)
    portfolio_volatility=calculate_portfolio_volatility(cov_matrix,portfolio_weights)

    df=calculate_portfolio_returns(df,portfolio_weights)
    df=calculate_rebalancing_turnover(df,portfolio_weights)
    df=apply_portfolio_transaction_costs(df,cost_rate)

    statistics=calculate_portfolio_statistics(df)

    save_portfolio_output(df,output_path)

    print("Portfolio weights:")
    print(portfolio_weights)

    print("\nPortfolio daily volatility:")
    print(portfolio_volatility)

    print("\nPortfolio statistics:")
    print(statistics)


if __name__=="__main__":
    main()
