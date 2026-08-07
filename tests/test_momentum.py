import pandas as pd
import pytest

from src.analyse import(
    calculate_returns,
    validate_data
)

from src.momentum import(
    calculate_momentum_signal,
    calculate_momentum_statistics,
    main,
    print_momentum_report,
    run_momentum,
    save_momentum_output
)


def make_price_dataframe(prices, dates=None):
    if dates is None:
        dates=pd.date_range(start="2024-01-01",periods=len(prices),freq="D").strftime("%Y-%m-%d").tolist()

    return pd.DataFrame({
        "Date":dates,
        "Close":prices
    })


def run_momentum_pipeline(prices, lookback=3, cost_rate=0.001):
    raw_df=make_price_dataframe(prices)

    df=validate_data(raw_df)
    df=calculate_returns(df)
    df=run_momentum(df,lookback,cost_rate)

    return df


def test_momentum_signal_calculates_lookback_returns():
    raw_df=make_price_dataframe([100,110,120,108,90,99])

    df=validate_data(raw_df)
    df=calculate_returns(df)

    result=calculate_momentum_signal(df,lookback=3)

    assert result["LookbackReturn"].iloc[:3].isna().all()

    assert result["LookbackReturn"].iloc[3:].tolist()==pytest.approx([
        0.08,
        -0.1818181818,
        -0.175
    ])

    assert result["Signal"].tolist()==[
        0,
        0,
        0,
        1,
        0,
        0
    ]


def test_momentum_signal_handles_positive_zero_and_negative_returns():
    raw_df=make_price_dataframe([100,105,105,95])

    df=validate_data(raw_df)
    df=calculate_returns(df)

    result=calculate_momentum_signal(df,lookback=1)

    assert pd.isna(result["LookbackReturn"].iloc[0])

    assert result["LookbackReturn"].iloc[1:].tolist()==pytest.approx([
        0.05,
        0.0,
        -0.0952380952
    ])

    assert result["Signal"].tolist()==[
        0,
        1,
        0,
        0
    ]


def test_momentum_signal_does_not_modify_input():
    raw_df=make_price_dataframe([100,110,120,108])

    df=validate_data(raw_df)
    df=calculate_returns(df)

    original_df=df.copy(deep=True)

    calculate_momentum_signal(df,lookback=2)

    pd.testing.assert_frame_equal(df,original_df)


def test_run_momentum():
    raw_df=make_price_dataframe([100,110,120,108,90,99])

    df=validate_data(raw_df)
    df=calculate_returns(df)

    result=run_momentum(df,lookback=3,cost_rate=0.001)

    assert "LookbackReturn" in result.columns
    assert "Signal" in result.columns
    assert "Position" in result.columns
    assert "NetStrategyReturn" in result.columns
    assert "NetStrategyCumulativeValue" in result.columns


def test_complete_momentum_pipeline():
    df=run_momentum_pipeline(prices=[100,110,120,108,90,99],lookback=3,cost_rate=0.001)

    assert df["Signal"].tolist()==[
        0,
        0,
        0,
        1,
        0,
        0
    ]

    assert df["Position"].tolist()==[
        0,
        0,
        0,
        0,
        1,
        0
    ]

    assert df["StrategyReturn"].tolist()==pytest.approx([
        0.0,
        0.0,
        0.0,
        0.0,
        -1/6,
        0.0
    ])

    assert df["Turnover"].tolist()==pytest.approx([
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        1.0
    ])

    assert df["TransactionCost"].tolist()==pytest.approx([
        0.0,
        0.0,
        0.0,
        0.0,
        0.001,
        0.001
    ])

    assert df["NetStrategyCumulativeValue"].iloc[-1]==pytest.approx(0.831501)


def test_momentum_statistics():
    df=run_momentum_pipeline(prices=[100,110,120,108,90,99],lookback=3,cost_rate=0.001)

    stats=calculate_momentum_statistics(df,lookback=3,cost_rate=0.001)

    assert stats["lookback"]==3
    assert stats["cost_rate"]==pytest.approx(0.001)
    assert stats["observations"]==6
    assert stats["buy_and_hold_return"]==pytest.approx(-0.01)
    assert stats["gross_strategy_return"]==pytest.approx(-1/6)
    assert stats["net_strategy_return"]==pytest.approx(-0.168499)
    assert stats["total_turnover"]==pytest.approx(2)
    assert stats["trade_events"]==2
    assert stats["time_in_market"]==pytest.approx(1/6)


def test_transaction_costs_reduce_strategy_value():
    df=run_momentum_pipeline(prices=[100,110,120,108,90,99],lookback=3,cost_rate=0.001)

    gross_final_value=df["StrategyCumulativeValue"].iloc[-1]
    net_final_value=df["NetStrategyCumulativeValue"].iloc[-1]

    assert net_final_value<gross_final_value


def test_print_momentum_report(capsys):
    stats={
        "lookback":3,
        "cost_rate":0.001,
        "observations":6,
        "buy_and_hold_return":-0.01,
        "gross_strategy_return":-1/6,
        "net_strategy_return":-0.168499,
        "total_turnover":2,
        "trade_events":2,
        "time_in_market":1/6
    }

    print_momentum_report(stats)

    terminal_output=capsys.readouterr().out

    assert "MOMENTUM STRATEGY REPORT" in terminal_output
    assert "3 periods" in terminal_output
    assert "0.10%" in terminal_output
    assert "-1.00%" in terminal_output
    assert "-16.67%" in terminal_output
    assert "-16.85%" in terminal_output
    assert "16.67%" in terminal_output


def test_save_momentum_output_writes_required_columns(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    df=run_momentum_pipeline(prices=[100,110,120,108,90,99],lookback=3,cost_rate=0.001)

    df["UnusedColumn"]=123

    save_momentum_output(df)

    output_path=tmp_path/"output"/"momentum_results.csv"

    assert output_path.exists()

    saved_df=pd.read_csv(output_path)

    assert saved_df.columns.tolist()==[
        "Date",
        "Close",
        "DailyReturn",
        "CumulativeValue",
        "LookbackReturn",
        "Signal",
        "Position",
        "StrategyReturn",
        "StrategyCumulativeValue",
        "Turnover",
        "TransactionCost",
        "NetStrategyReturn",
        "NetStrategyCumulativeValue"
    ]

    assert "UnusedColumn" not in saved_df.columns
    assert len(saved_df)==6
    assert saved_df["NetStrategyCumulativeValue"].iloc[-1]==pytest.approx(0.831501,abs=0.000001)


def test_momentum_pipeline_does_not_modify_raw_input():
    raw_df=make_price_dataframe([100,110,120,108,90,99])

    original_df=raw_df.copy(deep=True)

    df=validate_data(raw_df)
    df=calculate_returns(df)

    run_momentum(df,lookback=3,cost_rate=0.001)

    pd.testing.assert_frame_equal(raw_df,original_df)


def test_momentum_pipeline_is_deterministic():
    first_result=run_momentum_pipeline(prices=[100,110,120,108,90,99])
    second_result=run_momentum_pipeline(prices=[100,110,120,108,90,99])

    pd.testing.assert_frame_equal(first_result,second_result)


def test_main_runs_complete_momentum_pipeline(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)

    data_directory=tmp_path/"data"
    data_directory.mkdir()

    input_path=data_directory/"prices.csv"

    original_contents=(
        "Date,Close\n"
        "2024-01-05,108\n"
        "2024-01-02,100\n"
        "2024-01-04,120\n"
        "2024-01-03,110\n"
        "2024-01-08,90\n"
        "2024-01-09,99\n"
    )

    input_path.write_text(original_contents,encoding="utf-8")

    main()

    terminal_output=capsys.readouterr().out

    assert "MOMENTUM STRATEGY REPORT" in terminal_output
    assert "Buy-and-hold return:    -1.00%" in terminal_output
    assert "Gross strategy return:  -16.67%" in terminal_output
    assert "Net strategy return:    -16.85%" in terminal_output
    assert "Trade events:           2" in terminal_output

    output_path=tmp_path/"output"/"momentum_results.csv"

    assert output_path.exists()

    saved_df=pd.read_csv(output_path)

    assert len(saved_df)==6

    assert saved_df["Date"].tolist()==[
        "2024-01-02",
        "2024-01-03",
        "2024-01-04",
        "2024-01-05",
        "2024-01-08",
        "2024-01-09"
    ]

    assert input_path.read_text(encoding="utf-8")==original_contents