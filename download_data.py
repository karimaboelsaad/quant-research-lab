import pandas as pd
import yfinance as yf

from src.config import DEFAULT_CONFIG_PATH,load_config,resolve_project_path
from src.data import validate_multi_asset_prices,validate_pair_prices,validate_single_asset_prices


def download_adjusted_close(ticker,start_date,end_date):
    downloaded=yf.download(ticker,start=start_date,end=end_date,auto_adjust=True,actions=False,progress=False,threads=False,keepna=False,multi_level_index=False,timeout=30)

    if downloaded.empty or "Close" not in downloaded.columns:
        raise ValueError(f"No adjusted close data was downloaded for {ticker}.")

    close=downloaded["Close"].copy()
    close.index=pd.to_datetime(close.index).tz_localize(None)
    close.name=ticker

    if close.isna().any():
        raise ValueError(f"Downloaded adjusted close data for {ticker} contains missing values.")

    return close


def build_common_price_panel(tickers,start_date,end_date):
    series=[download_adjusted_close(ticker,start_date,end_date) for ticker in tickers]
    panel=pd.concat(series,axis=1,join="inner").dropna().reset_index()
    panel=panel.rename(columns={panel.columns[0]:"Date"})

    return validate_multi_asset_prices(panel,asset_columns=tickers)


def save_price_data(df,path):
    path=resolve_project_path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    df.to_csv(path,index=False,float_format="%.10f",date_format="%Y-%m-%d")

    return path


def download_research_data(config_path=DEFAULT_CONFIG_PATH):
    config=load_config(config_path)
    data_config=config["data"]
    start_date=data_config["start_date"]
    end_date=data_config["end_date"]

    multi_asset_df=build_common_price_panel(data_config["multi_asset_tickers"],start_date,end_date)
    pair_panel=build_common_price_panel(data_config["pair_tickers"],start_date,end_date)
    pair_df=pair_panel.rename(columns={data_config["pair_tickers"][0]:"CloseA",data_config["pair_tickers"][1]:"CloseB"})
    pair_df=validate_pair_prices(pair_df)
    single_asset_df=multi_asset_df[["Date",data_config["single_asset_ticker"]]].rename(columns={data_config["single_asset_ticker"]:"Close"})
    single_asset_df=validate_single_asset_prices(single_asset_df)

    output_paths={
        "single_asset":save_price_data(single_asset_df,data_config["single_asset_path"]),
        "multi_asset":save_price_data(multi_asset_df,data_config["multi_asset_path"]),
        "pair":save_price_data(pair_df,data_config["pair_path"])
    }

    return {
        "output_paths":output_paths,
        "single_asset_rows":len(single_asset_df),
        "multi_asset_rows":len(multi_asset_df),
        "pair_rows":len(pair_df)
    }


def main():
    result=download_research_data()

    print("Downloaded adjusted research data:")
    print(f"Single asset rows: {result['single_asset_rows']}")
    print(f"Multi-asset rows:  {result['multi_asset_rows']}")
    print(f"Pair rows:         {result['pair_rows']}")

    for name,path in result["output_paths"].items():
        print(f"{name}: {path}")


if __name__=="__main__":
    main()
