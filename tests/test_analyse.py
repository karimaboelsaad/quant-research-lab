import pandas as pd
import pytest

from src.analyse import (
    calculate_drawdowns,
    calculate_returns,
    calculate_summary_statistics,
    load_data,
    main,
    save_output,
    validate_data,
)


def make_price_dataframe(prices, dates=None):
    """Create a small raw price DataFrame for testing."""
    if dates is None:
        dates = (
            pd.date_range(
                start="2024-01-01",
                periods=len(prices),
                freq="D",
            )
            .strftime("%Y-%m-%d")
            .tolist()
        )

    return pd.DataFrame(
        {
            "Date": dates,
            "Close": prices,
        }
    )


def run_calculation_pipeline(raw_df):
    """Run the calculation stages without printing or saving."""
    df = validate_data(raw_df)
    df = calculate_returns(df)
    df = calculate_drawdowns(df)
    return df


def test_load_valid_csv(tmp_path):
    csv_path = tmp_path / "prices.csv"

    csv_path.write_text(
        "Date,Close\n"
        "2024-01-03,110\n"
        "2024-01-01,100\n"
        "2024-01-02,105\n",
        encoding="utf-8",
    )

    df = load_data(csv_path)

    assert list(df.columns) == ["Date", "Close"]
    assert len(df) == 3
    assert df["Date"].tolist() == [
        "2024-01-03",
        "2024-01-01",
        "2024-01-02",
    ]
    assert df["Close"].tolist() == [110, 100, 105]


@pytest.mark.parametrize(
    "csv_contents",
    [
        (
            "Date\n"
            "2024-01-01\n"
            "2024-01-02\n"
            "2024-01-03\n"
        ),
        (
            "Close\n"
            "100\n"
            "110\n"
            "120\n"
        ),
    ],
)
def test_load_rejects_missing_required_columns(tmp_path, csv_contents):
    csv_path = tmp_path / "missing_column.csv"
    csv_path.write_text(csv_contents, encoding="utf-8")

    with pytest.raises(ValueError):
        load_data(csv_path)


@pytest.mark.parametrize(
    "csv_contents",
    [
        "",
        "Date,Close\n",
    ],
)
def test_load_rejects_empty_files(tmp_path, csv_contents):
    csv_path = tmp_path / "empty.csv"
    csv_path.write_text(csv_contents, encoding="utf-8")

    with pytest.raises(ValueError):
        load_data(csv_path)


@pytest.mark.parametrize(
    "invalid_date",
    [
        None,
        "",
        "banana",
        "not-a-date",
    ],
)
def test_validate_rejects_missing_or_invalid_dates(invalid_date):
    raw_df = make_price_dataframe(
        prices=[100, 110, 120],
        dates=[
            "2024-01-01",
            invalid_date,
            "2024-01-03",
        ],
    )

    with pytest.raises(ValueError):
        validate_data(raw_df)


@pytest.mark.parametrize(
    "invalid_price",
    [
        None,
        "",
        "hello",
        "not-a-number",
    ],
)
def test_validate_rejects_missing_or_invalid_prices(invalid_price):
    raw_df = make_price_dataframe(
        prices=[100, invalid_price, 120]
    )

    with pytest.raises(ValueError):
        validate_data(raw_df)


@pytest.mark.parametrize(
    "invalid_price",
    [
        0,
        -1,
        -100,
    ],
)
def test_validate_rejects_non_positive_prices(invalid_price):
    raw_df = make_price_dataframe(
        prices=[100, invalid_price, 120]
    )

    with pytest.raises(ValueError):
        validate_data(raw_df)


def test_validate_rejects_duplicate_dates():
    raw_df = make_price_dataframe(
        prices=[100, 110, 120],
        dates=[
            "2024-01-01",
            "2024-01-01",
            "2024-01-03",
        ],
    )

    with pytest.raises(ValueError):
        validate_data(raw_df)


@pytest.mark.parametrize(
    "prices",
    [
        [100],
        [100, 110],
    ],
)
def test_validate_rejects_too_few_observations(prices):
    raw_df = make_price_dataframe(prices)

    with pytest.raises(ValueError):
        validate_data(raw_df)


def test_validation_sorts_dates_and_preserves_input():
    raw_df = make_price_dataframe(
        prices=[120, 100, 110],
        dates=[
            "2024-01-03",
            "2024-01-01",
            "2024-01-02",
        ],
    )

    original_df = raw_df.copy(deep=True)

    validated_df = validate_data(raw_df)

    # The input object should remain unchanged.
    pd.testing.assert_frame_equal(raw_df, original_df)

    assert validated_df["Date"].tolist() == [
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-01-03"),
    ]

    assert validated_df["Close"].tolist() == [
        100,
        110,
        120,
    ]

    assert validated_df.index.tolist() == [0, 1, 2]


def test_returns_and_cumulative_value():
    raw_df = make_price_dataframe(
        prices=[100, 110, 121]
    )

    processed_df = run_calculation_pipeline(raw_df)

    # The first return must remain missing.
    assert pd.isna(processed_df["DailyReturn"].iloc[0])

    assert processed_df["DailyReturn"].iloc[1:].tolist() == (
        pytest.approx([0.10, 0.10])
    )

    assert processed_df["CumulativeValue"].tolist() == (
        pytest.approx([1.00, 1.10, 1.21])
    )

    stats = calculate_summary_statistics(processed_df)

    assert stats["total_return"] == pytest.approx(0.21)
    assert stats["starting_price"] == 100
    assert stats["ending_price"] == 121
    assert stats["observations"] == 3


def test_constant_prices_produce_zero_returns_and_volatility():
    raw_df = make_price_dataframe(
        prices=[100, 100, 100]
    )

    processed_df = run_calculation_pipeline(raw_df)
    stats = calculate_summary_statistics(processed_df)

    assert pd.isna(processed_df["DailyReturn"].iloc[0])

    assert processed_df["DailyReturn"].iloc[1:].tolist() == (
        pytest.approx([0.0, 0.0])
    )

    assert processed_df["CumulativeValue"].tolist() == (
        pytest.approx([1.0, 1.0, 1.0])
    )

    assert processed_df["RunningMaximum"].tolist() == (
        pytest.approx([1.0, 1.0, 1.0])
    )

    assert processed_df["Drawdown"].tolist() == (
        pytest.approx([0.0, 0.0, 0.0])
    )

    assert stats["total_return"] == pytest.approx(0.0)
    assert stats["daily_volatility"] == pytest.approx(0.0)
    assert stats["annualised_volatility"] == pytest.approx(0.0)
    assert stats["maximum_drawdown"] == pytest.approx(0.0)


def test_known_maximum_drawdown():
    raw_df = make_price_dataframe(
        prices=[100, 120, 90]
    )

    processed_df = run_calculation_pipeline(raw_df)
    stats = calculate_summary_statistics(processed_df)

    assert processed_df["CumulativeValue"].tolist() == (
        pytest.approx([1.0, 1.2, 0.9])
    )

    assert processed_df["RunningMaximum"].tolist() == (
        pytest.approx([1.0, 1.2, 1.2])
    )

    assert processed_df["Drawdown"].tolist() == (
        pytest.approx([0.0, 0.0, -0.25])
    )

    assert stats["maximum_drawdown"] == pytest.approx(-0.25)


def test_summary_statistics_identify_best_and_worst_days():
    raw_df = make_price_dataframe(
        prices=[100, 110, 99, 108.9],
        dates=[
            "2024-01-01",
            "2024-01-02",
            "2024-01-03",
            "2024-01-04",
        ],
    )

    processed_df = run_calculation_pipeline(raw_df)
    stats = calculate_summary_statistics(processed_df)

    assert stats["best_return"] == pytest.approx(0.10)
    assert stats["best_date"] == pd.Timestamp("2024-01-02")

    assert stats["worst_return"] == pytest.approx(-0.10)
    assert stats["worst_date"] == pd.Timestamp("2024-01-03")

    assert stats["start_date"] == pd.Timestamp("2024-01-01")
    assert stats["end_date"] == pd.Timestamp("2024-01-04")


def test_save_output_writes_required_columns(
    tmp_path,
    monkeypatch,
):
    # Make the temporary directory act as the project root.
    monkeypatch.chdir(tmp_path)

    raw_df = make_price_dataframe(
        prices=[100, 110, 90]
    )

    processed_df = run_calculation_pipeline(raw_df)

    # Add an unrelated column to ensure save_output excludes it.
    processed_df["UnusedColumn"] = [1, 2, 3]

    save_output(processed_df)

    output_path = (
        tmp_path
        / "output"
        / "analysed_prices.csv"
    )

    assert output_path.exists()

    saved_df = pd.read_csv(output_path)

    assert list(saved_df.columns) == [
        "Date",
        "Close",
        "DailyReturn",
        "CumulativeValue",
        "RunningMaximum",
        "Drawdown",
    ]

    assert len(saved_df) == 3
    assert pd.isna(saved_df["DailyReturn"].iloc[0])


def test_pipeline_is_deterministic(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)

    raw_df = make_price_dataframe(
        prices=[100, 120, 108, 90, 99],
        dates=[
            "2024-01-01",
            "2024-01-02",
            "2024-01-03",
            "2024-01-04",
            "2024-01-05",
        ],
    )

    first_result = run_calculation_pipeline(raw_df)
    second_result = run_calculation_pipeline(raw_df)

    pd.testing.assert_frame_equal(
        first_result,
        second_result,
    )

    save_output(first_result)

    output_path = (
        tmp_path
        / "output"
        / "analysed_prices.csv"
    )

    first_file_contents = output_path.read_text(
        encoding="utf-8"
    )

    save_output(second_result)

    second_file_contents = output_path.read_text(
        encoding="utf-8"
    )

    assert first_file_contents == second_file_contents


def test_main_runs_complete_pipeline(
    tmp_path,
    monkeypatch,
    capsys,
):
    monkeypatch.chdir(tmp_path)

    data_directory = tmp_path / "data"
    data_directory.mkdir()

    input_path = data_directory / "prices.csv"

    original_contents = (
        "Date,Close\n"
        "2024-01-05,108\n"
        "2024-01-02,100\n"
        "2024-01-04,120\n"
        "2024-01-03,110\n"
        "2024-01-08,90\n"
        "2024-01-09,99\n"
    )

    input_path.write_text(
        original_contents,
        encoding="utf-8",
    )

    main()

    terminal_output = capsys.readouterr().out

    assert "ASSET PERFORMANCE REPORT" in terminal_output
    assert "Total return:           -1.00%" in terminal_output
    assert "Maximum drawdown:       25.00%" in terminal_output
    assert "Best day:               10.00%" in terminal_output
    assert "Worst day:              -16.67%" in terminal_output

    output_path = (
        tmp_path
        / "output"
        / "analysed_prices.csv"
    )

    assert output_path.exists()

    # The raw input file must never be modified.
    assert input_path.read_text(
        encoding="utf-8"
    ) == original_contents