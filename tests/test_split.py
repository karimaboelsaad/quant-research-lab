import pandas as pd
import pytest

from src.split import split_data


def make_dataframe(number_of_rows):
    return pd.DataFrame(
        {
            "Date": pd.date_range(
                start="2024-01-01",
                periods=number_of_rows,
                freq="D",
            ),
            "Close": range(
                100,
                100 + number_of_rows,
            ),
        }
    )


def test_split_data_uses_default_ratios():
    df = make_dataframe(10)

    train_df, validation_df, test_df = split_data(df)

    assert len(train_df) == 6
    assert len(validation_df) == 2
    assert len(test_df) == 2


def test_split_data_uses_custom_ratios():
    df = make_dataframe(20)

    train_df, validation_df, test_df = split_data(
        df,
        train_ratio=0.5,
        validation_ratio=0.25,
    )

    assert len(train_df) == 10
    assert len(validation_df) == 5
    assert len(test_df) == 5


def test_split_data_preserves_chronological_order():
    df = make_dataframe(10)

    train_df, validation_df, test_df = split_data(df)

    assert train_df["Date"].tolist() == (
        df["Date"].iloc[:6].tolist()
    )

    assert validation_df["Date"].tolist() == (
        df["Date"].iloc[6:8].tolist()
    )

    assert test_df["Date"].tolist() == (
        df["Date"].iloc[8:].tolist()
    )


def test_split_data_does_not_lose_rows():
    df = make_dataframe(11)

    train_df, validation_df, test_df = split_data(df)

    combined_df = pd.concat(
        [
            train_df,
            validation_df,
            test_df,
        ]
    )

    pd.testing.assert_frame_equal(
        combined_df,
        df,
    )


def test_split_data_has_no_overlapping_rows():
    df = make_dataframe(10)

    train_df, validation_df, test_df = split_data(df)

    train_indices = set(train_df.index)
    validation_indices = set(validation_df.index)
    test_indices = set(test_df.index)

    assert train_indices.isdisjoint(
        validation_indices
    )

    assert train_indices.isdisjoint(
        test_indices
    )

    assert validation_indices.isdisjoint(
        test_indices
    )


def test_split_data_does_not_modify_input():
    df = make_dataframe(10)
    original_df = df.copy(deep=True)

    split_data(df)

    pd.testing.assert_frame_equal(
        df,
        original_df,
    )


@pytest.mark.parametrize(
    "train_ratio, validation_ratio",
    [
        (0, 0.2),
        (-0.1, 0.2),
        (1, 0.2),
        (1.1, 0.2),
        (0.6, 0),
        (0.6, -0.1),
        (0.6, 1),
        (0.6, 1.1),
        (0.8, 0.2),
        (0.9, 0.2),
    ],
)
def test_split_data_rejects_invalid_ratios(
    train_ratio,
    validation_ratio,
):
    df = make_dataframe(10)

    with pytest.raises(ValueError):
        split_data(
            df,
            train_ratio=train_ratio,
            validation_ratio=validation_ratio,
        )


def test_split_data_rejects_dataframe_that_is_too_small():
    df = make_dataframe(3)

    with pytest.raises(ValueError):
        split_data(
            df,
            train_ratio=0.6,
            validation_ratio=0.2,
        )