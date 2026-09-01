import pandas as pd
import numpy as np


def validate_dates(df,date_column="Date"):
    if df.empty:
        raise ValueError(
            "Dataframe cannot be empty."
        )

    if date_column not in df.columns:
        raise ValueError(
            f"Dataframe must contain a {date_column} column."
        )

    df=df.copy()

    try:
        df[date_column]=pd.to_datetime(df[date_column],errors="raise")
    except (ValueError,TypeError):
        raise ValueError(
            "Dates must be valid."
        )

    if df[date_column].isna().any():
        raise ValueError(
            "Dates cannot contain missing values."
        )

    if df[date_column].duplicated().any():
        raise ValueError(
            "Dates must be unique."
        )

    df=df.sort_values(date_column).reset_index(drop=True)

    return df


def validate_price_columns(df,price_columns):
    if not price_columns:
        raise ValueError(
            "At least one price column is required."
        )

    missing_columns=[
        column
        for column in price_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing price columns: {missing_columns}"
        )

    df=df.copy()

    for column in price_columns:
        if df[column].isna().any():
            raise ValueError(
                f"{column} cannot contain missing prices."
            )

        try:
            df[column]=pd.to_numeric(df[column],errors="raise").astype(float)
        except (ValueError,TypeError):
            raise ValueError(
                f"{column} must contain numeric prices."
            )

        if not np.isfinite(df[column].to_numpy()).all():
            raise ValueError(
                f"{column} must contain finite prices."
            )

        if (df[column]<=0).any():
            raise ValueError(
                f"{column} prices must be positive."
            )

    return df


def validate_price_data(df,price_columns,date_column="Date"):
    df=validate_dates(df,date_column)
    df=validate_price_columns(df,price_columns)

    return df


def validate_single_asset_prices(df,date_column="Date",price_column="Close"):
    df=validate_price_data(df,[price_column],date_column)

    return df


def validate_pair_prices(df,date_column="Date",price_columns=("CloseA","CloseB")):
    if len(price_columns)!=2:
        raise ValueError(
            "Pair data must contain exactly two price columns."
        )

    df=validate_price_data(df,list(price_columns),date_column)

    return df


def validate_multi_asset_prices(df,date_column="Date",asset_columns=None):
    if asset_columns is None:
        asset_columns=[
            column
            for column in df.columns
            if column!=date_column
        ]

    if len(asset_columns)<2:
        raise ValueError(
            "Multi-asset data must contain at least two assets."
        )

    df=validate_price_data(df,asset_columns,date_column)

    return df


def validate_aligned_dates(dataframes,date_column="Date"):
    if len(dataframes)<2:
        raise ValueError(
            "At least two dataframes are required."
        )

    validated_dataframes=[
        validate_dates(df,date_column)
        for df in dataframes
    ]

    reference_dates=validated_dataframes[0][date_column]

    for df in validated_dataframes[1:]:
        if not reference_dates.equals(df[date_column]):
            raise ValueError(
                "Dataframes must contain identical aligned dates."
            )

    return validated_dataframes


def validate_return_data(df,return_columns,date_column="Date"):
    df=validate_dates(df,date_column)

    if not return_columns:
        raise ValueError(
            "At least one return column is required."
        )

    missing_columns=[column for column in return_columns if column not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing return columns: {missing_columns}")

    df=df.copy()

    for column in return_columns:
        if df[column].isna().any():
            raise ValueError(f"{column} cannot contain missing returns.")

        try:
            df[column]=pd.to_numeric(df[column],errors="raise").astype(float)
        except (ValueError,TypeError):
            raise ValueError(f"{column} must contain numeric returns.") from None

        if not np.isfinite(df[column].to_numpy()).all():
            raise ValueError(f"{column} must contain finite returns.")

        if (df[column]<=-1).any():
            raise ValueError(f"{column} returns must be greater than -100%.")

    return df


def calculate_price_returns(df,price_columns,return_columns=None,date_column="Date"):
    df=validate_price_data(df,price_columns,date_column)

    if return_columns is None:
        return_columns=[
            f"{column}Return"
            for column in price_columns
        ]

    if len(return_columns)!=len(price_columns):
        raise ValueError(
            "Each price column must have one return column."
        )

    df=df.copy()

    for price_column,return_column in zip(price_columns,return_columns):
        df[return_column]=df[price_column].pct_change(fill_method=None)

    return df


def load_single_asset_prices(path,date_column="Date",price_column="Close"):
    df=pd.read_csv(path)
    df=validate_single_asset_prices(df,date_column,price_column)

    return df


def load_pair_prices(path,date_column="Date",price_columns=("CloseA","CloseB")):
    df=pd.read_csv(path)
    df=validate_pair_prices(df,date_column,price_columns)

    return df


def load_multi_asset_prices(path,date_column="Date",asset_columns=None):
    df=pd.read_csv(path)
    df=validate_multi_asset_prices(df,date_column,asset_columns)

    return df
