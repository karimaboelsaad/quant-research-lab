import pandas as pd
import numpy as np

def calculate_asset_returns(df):
    if df.empty:
        raise ValueError(
            "Dataframe cannot be empty."
        )
    
    df=df.copy()
    asset_columns=[column for column in df.columns if column!="Date"]

    for column in asset_columns:
        df[f"{column}Return"]=df[column].pct_change()

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


def calculate_portfolio_returns(df, portfolio_weights):
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

    for column, weight in portfolio_weights.items():
        df["PortfolioReturn"]+=df[column].fillna(0)*weight

    df["PortfolioCumulativeValue"]=(1+df["PortfolioReturn"]).cumprod()

    return df


def calculate_portfolio_volatility(cov_matrix, portfolio_weights):
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



def calculate_rebalancing_turnover(df, portfolio_weights):
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

    for column, weight in portfolio_weights.items():
        drifted_weight=weight*(1+df[column].fillna(0))/(1+df["PortfolioReturn"])

        df["Turnover"]+=abs(weight-drifted_weight)

    return df



def apply_portfolio_transaction_costs(df, cost_rate):
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
    df["NetPortfolioCumulativeValue"]=(1+df["NetPortfolioReturn"]).cumprod()

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

    observations=len(df)

    gross_return=df["PortfolioCumulativeValue"].iloc[-1]-1
    net_return=df["NetPortfolioCumulativeValue"].iloc[-1]-1

    annualized_return=(df["NetPortfolioCumulativeValue"].iloc[-1]**(252/observations))-1

    daily_volatility=df["NetPortfolioReturn"].std()
    annualized_volatility=daily_volatility*np.sqrt(252)

    if daily_volatility==0 or pd.isna(daily_volatility):
        sharpe_ratio=np.nan
    else:
        sharpe_ratio=(df["NetPortfolioReturn"].mean()/daily_volatility)*np.sqrt(252)

    running_max=df["NetPortfolioCumulativeValue"].cummax()
    drawdown=df["NetPortfolioCumulativeValue"]/running_max-1
    max_drawdown=drawdown.min()

    statistics={
        "Observations":observations,
        "GrossReturn":gross_return,
        "NetReturn":net_return,
        "AnnualizedReturn":annualized_return,
        "AnnualizedVolatility":annualized_volatility,
        "SharpeRatio":sharpe_ratio,
        "MaxDrawdown":max_drawdown,
        "TotalTurnover":df["Turnover"].sum(),
        "TotalTransactionCost":df["TransactionCost"].sum()
    }

    return statistics


def load_portfolio_prices(path):
    df=pd.read_csv(path)

    if df.empty:
        raise ValueError(
            "Dataframe cannot be empty."
        )

    if "Date" not in df.columns:
        raise ValueError(
            "Dataframe must contain Date."
        )

    if len(df.columns)<3:
        raise ValueError(
            "Dataframe must contain at least 2 asset price columns."
        )

    df["Date"]=pd.to_datetime(df["Date"])

    return df


def save_portfolio_output(df, path):
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