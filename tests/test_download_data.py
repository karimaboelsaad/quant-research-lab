import pandas as pd

import download_data

from src.config import load_config


def test_build_common_price_panel_uses_date_intersection(monkeypatch):
    series={
        "A":pd.Series([100.0,101.0,102.0],index=pd.to_datetime(["2024-01-01","2024-01-02","2024-01-03"]),name="A"),
        "B":pd.Series([50.0,51.0,52.0],index=pd.to_datetime(["2024-01-02","2024-01-03","2024-01-04"]),name="B")
    }

    monkeypatch.setattr(download_data,"download_adjusted_close",lambda ticker,start_date,end_date:series[ticker])

    result=download_data.build_common_price_panel(["A","B"],"2024-01-01","2024-01-05")

    assert result["Date"].tolist()==pd.to_datetime(["2024-01-02","2024-01-03"]).tolist()
    assert result[["A","B"]].values.tolist()==[[101.0,50.0],[102.0,51.0]]


def test_save_price_data_creates_parent_directory(tmp_path):
    df=pd.DataFrame({"Date":pd.to_datetime(["2024-01-01"]),"Close":[100.0]})
    path=tmp_path/"nested"/"prices.csv"

    result=download_data.save_price_data(df,path)

    assert result==path
    assert path.exists()


def test_default_config_uses_ignored_research_directory():
    config=load_config()

    assert config["data"]["single_asset_path"].startswith("data/research/")
    assert config["data"]["multi_asset_path"].startswith("data/research/")
    assert config["data"]["pair_path"].startswith("data/research/")
