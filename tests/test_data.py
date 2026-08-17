import pandas as pd
import numpy as np
import pytest

from src.data import(
    validate_dates,
    validate_price_columns,
    validate_price_data,
    validate_single_asset_prices,
    validate_pair_prices,
    validate_multi_asset_prices,
    validate_aligned_dates,
    calculate_price_returns
)


def test_validate_dates_sorts_chronologically():
    df=pd.DataFrame({
        "Date":["2024-01-03","2024-01-01","2024-01-02"],
        "Close":[110,100,105]
    })

    result=validate_dates(df)

    expected_dates=pd.to_datetime([
        "2024-01-01",
        "2024-01-02",
        "2024-01-03"
    ])

    assert result["Date"].tolist()==expected_dates.tolist()
    assert result.index.tolist()==[0,1,2]


def test_validate_dates_rejects_empty_dataframe():
    df=pd.DataFrame()

    with pytest.raises(ValueError):
        validate_dates(df)


def test_validate_dates_requires_date_column():
    df=pd.DataFrame({
        "Close":[100,105,110]
    })

    with pytest.raises(ValueError):
        validate_dates(df)


def test_validate_dates_rejects_invalid_dates():
    df=pd.DataFrame({
        "Date":["2024-01-01","invalid","2024-01-03"],
        "Close":[100,105,110]
    })

    with pytest.raises(ValueError):
        validate_dates(df)


def test_validate_dates_rejects_duplicate_dates():
    df=pd.DataFrame({
        "Date":["2024-01-01","2024-01-01","2024-01-03"],
        "Close":[100,105,110]
    })

    with pytest.raises(ValueError):
        validate_dates(df)


def test_validate_dates_does_not_modify_input():
    df=pd.DataFrame({
        "Date":["2024-01-02","2024-01-01"],
        "Close":[105,100]
    })
    original=df.copy()

    validate_dates(df)

    pd.testing.assert_frame_equal(df,original)


def test_validate_price_columns_accepts_valid_prices():
    df=pd.DataFrame({
        "Close":[100,105,110]
    })

    result=validate_price_columns(df,["Close"])

    assert result["Close"].tolist()==[100.0,105.0,110.0]


def test_validate_price_columns_requires_requested_columns():
    df=pd.DataFrame({
        "CloseA":[100,105,110]
    })

    with pytest.raises(ValueError):
        validate_price_columns(df,["CloseA","CloseB"])


def test_validate_price_columns_rejects_missing_prices():
    df=pd.DataFrame({
        "Close":[100,np.nan,110]
    })

    with pytest.raises(ValueError):
        validate_price_columns(df,["Close"])


def test_validate_price_columns_rejects_nonnumeric_prices():
    df=pd.DataFrame({
        "Close":[100,"invalid",110]
    })

    with pytest.raises(ValueError):
        validate_price_columns(df,["Close"])


def test_validate_price_columns_rejects_zero_prices():
    df=pd.DataFrame({
        "Close":[100,0,110]
    })

    with pytest.raises(ValueError):
        validate_price_columns(df,["Close"])


def test_validate_price_columns_rejects_negative_prices():
    df=pd.DataFrame({
        "Close":[100,-5,110]
    })

    with pytest.raises(ValueError):
        validate_price_columns(df,["Close"])


def test_validate_price_columns_rejects_infinite_prices():
    df=pd.DataFrame({
        "Close":[100,np.inf,110]
    })

    with pytest.raises(ValueError):
        validate_price_columns(df,["Close"])


def test_validate_price_data_validates_and_sorts():
    df=pd.DataFrame({
        "Date":["2024-01-03","2024-01-01","2024-01-02"],
        "Close":[110,100,105]
    })

    result=validate_price_data(df,["Close"])

    assert result["Date"].is_monotonic_increasing
    assert result["Close"].tolist()==[100.0,105.0,110.0]


def test_validate_single_asset_prices():
    df=pd.DataFrame({
        "Date":["2024-01-01","2024-01-02","2024-01-03"],
        "Close":[100,105,110]
    })

    result=validate_single_asset_prices(df)

    assert list(result.columns)==["Date","Close"]


def test_validate_pair_prices():
    df=pd.DataFrame({
        "Date":["2024-01-01","2024-01-02","2024-01-03"],
        "CloseA":[100,105,110],
        "CloseB":[50,51,52]
    })

    result=validate_pair_prices(df)

    assert len(result)==3


def test_validate_pair_prices_requires_two_price_columns():
    df=pd.DataFrame({
        "Date":["2024-01-01","2024-01-02"],
        "CloseA":[100,105],
        "CloseB":[50,51]
    })

    with pytest.raises(ValueError):
        validate_pair_prices(df,price_columns=("CloseA",))


def test_validate_multi_asset_prices():
    df=pd.DataFrame({
        "Date":["2024-01-01","2024-01-02","2024-01-03"],
        "A":[100,105,110],
        "B":[50,51,52],
        "C":[200,201,203]
    })

    result=validate_multi_asset_prices(df)

    assert len(result)==3


def test_validate_multi_asset_prices_requires_two_assets():
    df=pd.DataFrame({
        "Date":["2024-01-01","2024-01-02"],
        "A":[100,105]
    })

    with pytest.raises(ValueError):
        validate_multi_asset_prices(df)


def test_validate_aligned_dates_accepts_same_dates():
    df_a=pd.DataFrame({
        "Date":["2024-01-02","2024-01-01"],
        "A":[102,100]
    })

    df_b=pd.DataFrame({
        "Date":["2024-01-01","2024-01-02"],
        "B":[50,51]
    })

    result=validate_aligned_dates([df_a,df_b])

    assert len(result)==2
    assert result[0]["Date"].equals(result[1]["Date"])


def test_validate_aligned_dates_rejects_different_dates():
    df_a=pd.DataFrame({
        "Date":["2024-01-01","2024-01-02"],
        "A":[100,102]
    })

    df_b=pd.DataFrame({
        "Date":["2024-01-01","2024-01-03"],
        "B":[50,51]
    })

    with pytest.raises(ValueError):
        validate_aligned_dates([df_a,df_b])


def test_validate_aligned_dates_requires_two_dataframes():
    df=pd.DataFrame({
        "Date":["2024-01-01","2024-01-02"],
        "A":[100,102]
    })

    with pytest.raises(ValueError):
        validate_aligned_dates([df])


def test_calculate_price_returns_single_asset():
    df=pd.DataFrame({
        "Date":["2024-01-01","2024-01-02","2024-01-03"],
        "Close":[100,105,110.25]
    })

    result=calculate_price_returns(df,["Close"],["Return"])

    assert np.isnan(result["Return"].iloc[0])
    assert np.isclose(result["Return"].iloc[1],0.05)
    assert np.isclose(result["Return"].iloc[2],0.05)


def test_calculate_price_returns_multiple_assets():
    df=pd.DataFrame({
        "Date":["2024-01-01","2024-01-02","2024-01-03"],
        "A":[100,110,121],
        "B":[200,190,199.5]
    })

    result=calculate_price_returns(
        df,
        ["A","B"],
        ["AReturn","BReturn"]
    )

    assert np.isnan(result["AReturn"].iloc[0])
    assert np.isnan(result["BReturn"].iloc[0])
    assert np.isclose(result["AReturn"].iloc[1],0.10)
    assert np.isclose(result["AReturn"].iloc[2],0.10)
    assert np.isclose(result["BReturn"].iloc[1],-0.05)
    assert np.isclose(result["BReturn"].iloc[2],0.05)


def test_calculate_price_returns_generates_default_names():
    df=pd.DataFrame({
        "Date":["2024-01-01","2024-01-02"],
        "A":[100,110],
        "B":[200,220]
    })

    result=calculate_price_returns(df,["A","B"])

    assert "AReturn" in result.columns
    assert "BReturn" in result.columns


def test_calculate_price_returns_requires_matching_return_columns():
    df=pd.DataFrame({
        "Date":["2024-01-01","2024-01-02"],
        "A":[100,110],
        "B":[200,220]
    })

    with pytest.raises(ValueError):
        calculate_price_returns(
            df,
            ["A","B"],
            ["AReturn"]
        )


def test_missing_price_never_becomes_zero_return():
    df=pd.DataFrame({
        "Date":["2024-01-01","2024-01-02","2024-01-03"],
        "Close":[100,np.nan,110]
    })

    with pytest.raises(ValueError):
        calculate_price_returns(df,["Close"],["Return"])