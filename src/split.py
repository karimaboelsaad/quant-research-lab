import pandas as pd

def split_data(df,train_ratio=0.6,validation_ratio=0.2):

    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1.")

    if not 0 < validation_ratio < 1:
        raise ValueError("validation_ratio must be between 0 and 1.")

    if train_ratio + validation_ratio >= 1:
        raise ValueError("The train and validation ratios must add up to less than 1.")

    number_of_rows = len(df)

    train_end = int(number_of_rows * train_ratio)
    validation_end = train_end + int(number_of_rows * validation_ratio)
    train_df = df.iloc[:train_end].copy()

    validation_df = df.iloc[train_end:validation_end].copy()

    test_df = df.iloc[validation_end:].copy()

    if (train_df.empty or validation_df.empty or test_df.empty):
        raise ValueError("The DataFrame is too small for the chosen split ratios.")

    return train_df, validation_df, test_df

