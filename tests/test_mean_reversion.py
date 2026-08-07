import pandas as pd
import pytest

from src.analyse import(
    calculate_returns,
    validate_data
)

from src.mean_reversion import(
    calculate_z_score,
    calculate_mean_reversion_signal,
    run_mean_reversion,
    calculate_mean_reversion_statistics,
    print_mean_reversion_report,
    save_mean_reversion_output,
    main
)


def make_price_dataframe(prices, dates=None):
    if dates is None:
        dates=pd.date_range(start="2024-01-01",periods=len(prices),freq="D").strftime("%Y-%m-%d").tolist()

    return pd.DataFrame({
        "Date":dates,
        "Close":prices
    })


def make_prepared_dataframe(prices):
    raw_df=make_price_dataframe(prices)

    df=validate_data(raw_df)
    df=calculate_returns(df)

    return df


def test_calculate_z_score():
    df=make_prepared_dataframe([
        100,
        102,
        101,
        105,
        103
    ])

    result=calculate_z_score(df,lookback=3)

    assert result["RollingMean"].iloc[:2].isna().all()
    assert result["RollingStd"].iloc[:2].isna().all()
    assert result["ZScore"].iloc[:2].isna().all()

    assert result["RollingMean"].iloc[2]==pytest.approx(101)
    assert result["RollingStd"].iloc[2]==pytest.approx(1)
    assert result["ZScore"].iloc[2]==pytest.approx(0)


@pytest.mark.parametrize("lookback",[0,1,-1,-10])
def test_calculate_z_score_rejects_invalid_lookback(lookback):
    df=make_prepared_dataframe([
        100,
        101,
        102,
        103
    ])

    with pytest.raises(ValueError,match="Lookback must be at least 2"):
        calculate_z_score(df,lookback)


def test_calculate_z_score_rejects_lookback_larger_than_dataframe():
    df=make_prepared_dataframe([
        100,
        101,
        102
    ])

    with pytest.raises(ValueError):
        calculate_z_score(df,lookback=4)


def test_calculate_z_score_does_not_modify_input():
    df=make_prepared_dataframe([
        100,
        101,
        102,
        103
    ])

    original_df=df.copy(deep=True)

    calculate_z_score(df,lookback=2)

    pd.testing.assert_frame_equal(df,original_df)


def test_mean_reversion_signal_is_stateful():
    df=pd.DataFrame({
        "ZScore":[
            -2.0,
            -1.2,
            -0.2,
            0.2,
            -2.0,
            -0.5,
            0.5
        ]
    })

    result=calculate_mean_reversion_signal(df,entry_threshold=-1.0,exit_threshold=0.0)

    assert result["Signal"].tolist()==[
        1,
        1,
        1,
        0,
        1,
        1,
        0
    ]


def test_mean_reversion_signal_uses_threshold_equality():
    df=pd.DataFrame({
        "ZScore":[
            -1.0,
            -0.5,
            0.0
        ]
    })

    result=calculate_mean_reversion_signal(df,entry_threshold=-1.0,exit_threshold=0.0)

    assert result["Signal"].tolist()==[
        1,
        1,
        0
    ]


def test_mean_reversion_signal_handles_nan():
    df=pd.DataFrame({
        "ZScore":[
            float("nan"),
            float("nan"),
            -2.0,
            -0.5,
            0.5
        ]
    })

    result=calculate_mean_reversion_signal(df,entry_threshold=-1.0,exit_threshold=0.0)

    assert result["Signal"].tolist()==[
        0,
        0,
        1,
        1,
        0
    ]


@pytest.mark.parametrize(
    "entry_threshold,exit_threshold",
    [
        (0.0,0.0),
        (1.0,0.0),
        (-1.0,-2.0)
    ]
)
def test_mean_reversion_signal_rejects_invalid_thresholds(entry_threshold, exit_threshold):
    df=pd.DataFrame({
        "ZScore":[
            -1,
            0,
            1
        ]
    })

    with pytest.raises(ValueError,match="Exit threshold must be larger"):
        calculate_mean_reversion_signal(df,entry_threshold,exit_threshold)


def test_mean_reversion_signal_does_not_modify_input():
    df=pd.DataFrame({
        "ZScore":[
            -2,
            -1,
            0,
            1
        ]
    })

    original_df=df.copy(deep=True)

    calculate_mean_reversion_signal(df,entry_threshold=-1,exit_threshold=0)

    pd.testing.assert_frame_equal(df,original_df)


def test_run_mean_reversion_runs_complete_pipeline():
    df=make_prepared_dataframe([
        100,
        102,
        98,
        103,
        97,
        104,
        100
    ])

    result=run_mean_reversion(df,lookback=3,entry_threshold=-1.0,exit_threshold=0.0,cost_rate=0.001)

    required_columns=[
        "RollingMean",
        "RollingStd",
        "ZScore",
        "Signal",
        "Position",
        "StrategyReturn",
        "StrategyCumulativeValue",
        "Turnover",
        "TransactionCost",
        "NetStrategyReturn",
        "NetStrategyCumulativeValue"
    ]

    for column in required_columns:
        assert column in result.columns


def test_run_mean_reversion_delays_signal():
    df=make_prepared_dataframe([
        100,
        100,
        90,
        100,
        100
    ])

    result=run_mean_reversion(df,lookback=3,entry_threshold=-1.0,exit_threshold=0.0,cost_rate=0.0)

    assert result.iloc[2]["Signal"]==1
    assert result.iloc[2]["Position"]==0
    assert result.iloc[3]["Position"]==1


def test_run_mean_reversion_does_not_modify_input():
    df=make_prepared_dataframe([
        100,
        102,
        98,
        103,
        97,
        104
    ])

    original_df=df.copy(deep=True)

    run_mean_reversion(df,lookback=3,entry_threshold=-1.0,exit_threshold=0.0,cost_rate=0.001)

    pd.testing.assert_frame_equal(df,original_df)


def test_mean_reversion_statistics():
    df=make_prepared_dataframe([
        100,
        102,
        98,
        103,
        97,
        104
    ])

    df=run_mean_reversion(df,lookback=3,entry_threshold=-1.0,exit_threshold=0.0,cost_rate=0.001)

    stats=calculate_mean_reversion_statistics(df,lookback=3,entry_threshold=-1.0,exit_threshold=0.0,cost_rate=0.001)

    assert stats["lookback"]==3
    assert stats["entry_threshold"]==pytest.approx(-1.0)
    assert stats["exit_threshold"]==pytest.approx(0.0)
    assert stats["cost_rate"]==pytest.approx(0.001)
    assert stats["observations"]==6


def test_print_mean_reversion_report(capsys):
    df=make_prepared_dataframe([
        100,
        102,
        98,
        103,
        97,
        104
    ])

    df=run_mean_reversion(df,lookback=3,entry_threshold=-1.0,exit_threshold=0.0,cost_rate=0.001)

    stats=calculate_mean_reversion_statistics(df,lookback=3,entry_threshold=-1.0,exit_threshold=0.0,cost_rate=0.001)

    print_mean_reversion_report(stats)

    terminal_output=capsys.readouterr().out

    assert "MEAN REVERSION STRATEGY REPORT" in terminal_output
    assert "3 periods" in terminal_output
    assert "Entry threshold:        -1.0" in terminal_output
    assert "Exit threshold:         0.0" in terminal_output
    assert "0.10%" in terminal_output


def test_save_mean_reversion_output_writes_required_columns(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    df=make_prepared_dataframe([
        100,
        102,
        98,
        103,
        97,
        104
    ])

    df=run_mean_reversion(df,lookback=3,entry_threshold=-1.0,exit_threshold=0.0,cost_rate=0.001)

    df["UnusedColumn"]=123

    save_mean_reversion_output(df)

    output_path=tmp_path/"output"/"mean_reversion_results.csv"

    assert output_path.exists()

    saved_df=pd.read_csv(output_path)

    assert saved_df.columns.tolist()==[
        "Date",
        "Close",
        "DailyReturn",
        "CumulativeValue",
        "RollingMean",
        "RollingStd",
        "ZScore",
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


def test_main_runs_complete_mean_reversion_pipeline(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)

    data_directory=tmp_path/"data"
    data_directory.mkdir()

    input_path=data_directory/"prices.csv"

    original_contents=(
        "Date,Close\n"
        "2024-01-05,97\n"
        "2024-01-02,100\n"
        "2024-01-04,103\n"
        "2024-01-03,102\n"
        "2024-01-08,104\n"
        "2024-01-09,100\n"
    )

    input_path.write_text(original_contents,encoding="utf-8")

    main()

    terminal_output=capsys.readouterr().out

    assert "MEAN REVERSION STRATEGY REPORT" in terminal_output

    output_path=tmp_path/"output"/"mean_reversion_results.csv"

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