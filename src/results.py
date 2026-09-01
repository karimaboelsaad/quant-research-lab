import pandas as pd


OOS_RESULT_COLUMNS=[
    "Date",
    "Asset",
    "Strategy",
    "GrossReturn",
    "InternalCost",
    "NetReturn",
    "Turnover",
    "Position",
    "WindowID",
    "SelectedLookback",
    "SelectedEntryThreshold",
    "SelectedExitThreshold",
    "EvaluationType"
]

PAIR_OOS_RESULT_COLUMNS=[
    "Date",
    "AssetA",
    "AssetB",
    "Strategy",
    "GrossReturn",
    "InternalCost",
    "NetReturn",
    "Turnover",
    "SpreadPosition",
    "PositionA",
    "PositionB",
    "GrossExposure",
    "RegressionAlpha",
    "RegressionBeta",
    "WindowID",
    "SelectedLookback",
    "SelectedEntryThreshold",
    "SelectedExitThreshold",
    "EvaluationType"
]


def standardize_oos_results(df,asset,strategy,window_id,selected_lookback,selected_entry_threshold=None,selected_exit_threshold=None):
    required_columns=[
        "Date",
        "StrategyReturn",
        "TransactionCost",
        "NetStrategyReturn",
        "Turnover",
        "Position"
    ]

    missing_columns=[
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing OOS result columns: {missing_columns}")

    if df.empty:
        raise ValueError("OOS results cannot be empty.")

    result=df[required_columns].copy()

    result=result.rename(columns={
        "StrategyReturn":"GrossReturn",
        "TransactionCost":"InternalCost",
        "NetStrategyReturn":"NetReturn"
    })

    result["Date"]=pd.to_datetime(result["Date"],errors="coerce")

    if result["Date"].isna().any():
        raise ValueError("OOS results contain invalid dates.")

    if result["Date"].duplicated().any():
        raise ValueError("OOS result dates must be unique.")

    result["Asset"]=asset
    result["Strategy"]=strategy
    result["WindowID"]=window_id
    result["SelectedLookback"]=selected_lookback
    result["SelectedEntryThreshold"]=float("nan") if selected_entry_threshold is None else float(selected_entry_threshold)
    result["SelectedExitThreshold"]=float("nan") if selected_exit_threshold is None else float(selected_exit_threshold)
    result["EvaluationType"]="walk_forward_test"

    return result[OOS_RESULT_COLUMNS].reset_index(drop=True)


def standardize_pair_oos_results(df,asset_a,asset_b,window_id,selected_lookback,selected_entry_threshold,selected_exit_threshold):
    if not asset_a or not asset_b or asset_a==asset_b:
        raise ValueError("Pair assets must be two different non-empty names.")

    required_columns=[
        "Date",
        "StrategyReturn",
        "TransactionCost",
        "NetStrategyReturn",
        "Turnover",
        "SpreadPosition",
        "PositionA",
        "PositionB",
        "GrossExposure",
        "RegressionAlpha",
        "RegressionBeta"
    ]

    missing_columns=[column for column in required_columns if column not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing pair OOS result columns: {missing_columns}")

    if df.empty:
        raise ValueError("Pair OOS results cannot be empty.")

    result=df[required_columns].copy()

    result=result.rename(columns={
        "StrategyReturn":"GrossReturn",
        "TransactionCost":"InternalCost",
        "NetStrategyReturn":"NetReturn"
    })

    result["Date"]=pd.to_datetime(result["Date"],errors="coerce")

    if result["Date"].isna().any():
        raise ValueError("Pair OOS results contain invalid dates.")

    if result["Date"].duplicated().any():
        raise ValueError("Pair OOS result dates must be unique.")

    result["AssetA"]=asset_a
    result["AssetB"]=asset_b
    result["Strategy"]="pairs"
    result["WindowID"]=window_id
    result["SelectedLookback"]=selected_lookback
    result["SelectedEntryThreshold"]=float(selected_entry_threshold)
    result["SelectedExitThreshold"]=float(selected_exit_threshold)
    result["EvaluationType"]="walk_forward_test"

    return result[PAIR_OOS_RESULT_COLUMNS].reset_index(drop=True)
