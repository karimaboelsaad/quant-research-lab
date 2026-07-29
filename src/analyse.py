import pandas as pd
from pathlib import Path


def load_data(csv_name):
    try:
        df = pd.read_csv(csv_name)

    except pd.errors.EmptyDataError:
        raise ValueError("The CSV file is completely blank.") from None

    except pd.errors.ParserError as error:
        raise ValueError("The CSV file could not be parsed.") from error

    if not {"Date", "Close"}.issubset(df.columns):
        raise ValueError(
            "The CSV file must contain 'Date' and 'Close' columns."
        )

    if df.empty:
        raise ValueError("The CSV file contains no price observations.")

    return df


def validate_data(df):
    df = df.copy()

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

    if df["Date"].isna().any():
        raise ValueError(
            "The CSV file contains missing or invalid dates."
        )

    if df["Close"].isna().any():
        raise ValueError(
            "The CSV file contains missing or invalid closing prices."
        )

    if not (df["Close"] > 0).all():
        raise ValueError(
            "The CSV file contains zero or negative closing prices."
        )

    if not df["Date"].is_unique:
        raise ValueError(
            "The CSV file contains duplicate dates."
        )

    if len(df) < 3:
        raise ValueError(
            "The CSV file must contain at least three price observations."
        )

    df = df.sort_values("Date").reset_index(drop=True)

    return df


def calculate_returns(df):
    df = df.copy()

    df["DailyReturn"] = df["Close"].pct_change(fill_method=None)
    df["CumulativeValue"] = (
        1 + df["DailyReturn"].fillna(0)
    ).cumprod()

    return df


def calculate_drawdowns(df):
    df = df.copy()

    df["RunningMaximum"] = df["CumulativeValue"].cummax()
    df["Drawdown"] = (
        df["CumulativeValue"] / df["RunningMaximum"]
    ) - 1

    return df


def calculate_summary_statistics(df):
    best_index = df["DailyReturn"].idxmax()
    worst_index = df["DailyReturn"].idxmin()

    daily_volatility = df["DailyReturn"].std()

    stats = {
        "start_date": df["Date"].iloc[0],
        "end_date": df["Date"].iloc[-1],
        "observations": len(df),
        "starting_price": df["Close"].iloc[0],
        "ending_price": df["Close"].iloc[-1],
        "total_return": df["CumulativeValue"].iloc[-1] - 1,
        "average_daily_return": df["DailyReturn"].mean(),
        "daily_volatility": daily_volatility,
        "annualised_volatility": daily_volatility * (252 ** 0.5),
        "maximum_drawdown": df["Drawdown"].min(),
        "best_return": df["DailyReturn"].max(),
        "best_date": df.loc[best_index, "Date"],
        "worst_return": df["DailyReturn"].min(),
        "worst_date": df.loc[worst_index, "Date"],
    }

    return stats


def print_report(stats):
    print(f"""
ASSET PERFORMANCE REPORT
------------------------
Period:                 {stats["start_date"]:%d %B %Y} to {stats["end_date"]:%d %B %Y}
Observations:           {stats["observations"]}
Starting price:         £{stats["starting_price"]:.2f}
Ending price:           £{stats["ending_price"]:.2f}
Total return:           {stats["total_return"]:.2%}
Average daily return:   {stats["average_daily_return"]:.2%}
Daily volatility:       {stats["daily_volatility"]:.2%}
Annualised volatility:  {stats["annualised_volatility"]:.2%}
Maximum drawdown:       {abs(stats["maximum_drawdown"]):.2%}
Best day:               {stats["best_return"]:.2%} on {stats["best_date"]:%d %B %Y}
Worst day:              {stats["worst_return"]:.2%} on {stats["worst_date"]:%d %B %Y}
""")


def save_output(df):
    output_path = Path("output/analysed_prices.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_columns = [
        "Date",
        "Close",
        "DailyReturn",
        "CumulativeValue",
        "RunningMaximum",
        "Drawdown",
    ]

    df[output_columns].to_csv(
        output_path,
        index=False,
        float_format="%.6f",
    )


def main():
    filepath = "data/prices.csv"

    df = load_data(filepath)
    df = validate_data(df)
    df = calculate_returns(df)
    df = calculate_drawdowns(df)

    stats = calculate_summary_statistics(df)

    print_report(stats)
    save_output(df)


if __name__ == "__main__":
    main()