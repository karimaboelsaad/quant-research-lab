import pandas as pd

from src.backtest import calculate_backtest_statistics
from src.mean_reversion import run_mean_reversion
from src.split import split_data


def evaluate_mean_reversion_period(df, start_position, end_position, lookback, entry_threshold, exit_threshold, cost_rate):
    if not 0<=start_position<end_position<=len(df):
        raise ValueError(
            "The start position needs at least 0 and smaller than the end position which needs to be at most the size of the dataframe."
        )

    df=df.iloc[:end_position].copy()

    df=run_mean_reversion(df,lookback,entry_threshold,exit_threshold,cost_rate)

    df=df.iloc[start_position:].copy()

    df["CumulativeValue"]=(1+df["DailyReturn"].fillna(0)).cumprod()
    df["StrategyCumulativeValue"]=(1+df["StrategyReturn"]).cumprod()
    df["NetStrategyCumulativeValue"]=(1+df["NetStrategyReturn"]).cumprod()

    return df


def evaluate_mean_reversion_parameters_on_period(df, start_position, end_position, lookbacks, entry_thresholds, exit_thresholds, cost_rate):
    results=[]

    for lookback in lookbacks:
        for entry_threshold in entry_thresholds:
            for exit_threshold in exit_thresholds:
                if exit_threshold>entry_threshold:
                    result_df=evaluate_mean_reversion_period(df,start_position,end_position,lookback,entry_threshold,exit_threshold,cost_rate)

                    stats=calculate_backtest_statistics(result_df)

                    results.append({
                        "Lookback":lookback,
                        "EntryThreshold":entry_threshold,
                        "ExitThreshold":exit_threshold,
                        "NetStrategyReturn":stats["net_strategy_return"]
                    })

    return pd.DataFrame(results)


def select_top_mean_reversion_parameters(df, n):
    if df.empty:
        raise ValueError(
            "Cannot select parameters from empty results."
        )

    if n<=0:
        raise ValueError(
            "n must be positive."
        )

    if n>len(df):
        raise ValueError(
            "n cannot exceed the number of results."
        )

    return df.sort_values("NetStrategyReturn",ascending=False).head(n)


def select_best_mean_reversion_parameters(df):
    if df.empty:
        raise ValueError(
            "Cannot select parameters from empty results."
        )

    best_row=df.loc[df["NetStrategyReturn"].idxmax()]

    return {
        "Lookback":int(best_row["Lookback"]),
        "EntryThreshold":best_row["EntryThreshold"],
        "ExitThreshold":best_row["ExitThreshold"]
    }


def evaluate_mean_reversion_candidates_on_period(df, start_position, end_position, candidates_df, cost_rate):
    results=[]

    for _, candidate in candidates_df.iterrows():
        lookback=int(candidate["Lookback"])
        entry_threshold=candidate["EntryThreshold"]
        exit_threshold=candidate["ExitThreshold"]

        result_df=evaluate_mean_reversion_period(df,start_position,end_position,lookback,entry_threshold,exit_threshold,cost_rate)

        stats=calculate_backtest_statistics(result_df)

        results.append({
            "Lookback":lookback,
            "EntryThreshold":entry_threshold,
            "ExitThreshold":exit_threshold,
            "NetStrategyReturn":stats["net_strategy_return"]
        })

    return pd.DataFrame(results)


def run_mean_reversion_optimisation(df, lookbacks, entry_thresholds, exit_thresholds, n, cost_rate):
    train_df, validation_df, _=split_data(df)

    train_end=len(train_df)
    validation_end=train_end+len(validation_df)

    training_results=evaluate_mean_reversion_parameters_on_period(df,0,train_end,lookbacks,entry_thresholds,exit_thresholds,cost_rate)

    candidate_parameters=select_top_mean_reversion_parameters(training_results,n)

    validation_results=evaluate_mean_reversion_candidates_on_period(df,train_end,validation_end,candidate_parameters,cost_rate)

    best_parameters=select_best_mean_reversion_parameters(validation_results)

    best_lookback=best_parameters["Lookback"]
    best_entry_threshold=best_parameters["EntryThreshold"]
    best_exit_threshold=best_parameters["ExitThreshold"]

    test_results=evaluate_mean_reversion_period(df,validation_end,len(df),best_lookback,best_entry_threshold,best_exit_threshold,cost_rate)

    test_statistics=calculate_backtest_statistics(test_results)

    return {
        "training_results":training_results,
        "candidate_parameters":candidate_parameters,
        "validation_results":validation_results,
        "best_parameters":best_parameters,
        "test_results":test_results,
        "test_statistics":test_statistics
    }