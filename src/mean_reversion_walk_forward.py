import pandas as pd

from src.backtest import calculate_backtest_statistics

from src.mean_reversion_optimise import(
    evaluate_mean_reversion_parameters_on_period,
    evaluate_mean_reversion_candidates_on_period,
    evaluate_mean_reversion_period,
    select_top_mean_reversion_parameters,
    select_best_mean_reversion_parameters
)

from src.walk_forward import calculate_common_training_start,generate_walk_forward_windows


def run_mean_reversion_walk_forward_window(df,window,lookbacks,entry_thresholds,exit_thresholds,n,cost_rate,previous_position=0):
    train_end=window["train_end"]
    validation_end=window["validation_end"]
    test_end=window["test_end"]
    training_start=calculate_common_training_start(lookbacks,train_end)

    training_results=evaluate_mean_reversion_parameters_on_period(df,training_start,train_end,lookbacks,entry_thresholds,exit_thresholds,cost_rate)

    candidate_parameters=select_top_mean_reversion_parameters(training_results,n)

    validation_results=evaluate_mean_reversion_candidates_on_period(df,train_end,validation_end,candidate_parameters,cost_rate)

    best_parameters=select_best_mean_reversion_parameters(validation_results)

    best_lookback=best_parameters["Lookback"]
    best_entry_threshold=best_parameters["EntryThreshold"]
    best_exit_threshold=best_parameters["ExitThreshold"]

    test_results=evaluate_mean_reversion_period(df,validation_end,test_end,best_lookback,best_entry_threshold,best_exit_threshold,cost_rate,previous_position)

    test_statistics=calculate_backtest_statistics(test_results)

    return {
        "training_results":training_results,
        "validation_results":validation_results,
        "candidate_parameters":candidate_parameters,
        "best_parameters":best_parameters,
        "test_results":test_results,
        "test_statistics":test_statistics
    }


def run_mean_reversion_walk_forward(df, lookbacks, entry_thresholds, exit_thresholds, n, cost_rate, initial_train_size, validation_size, test_size):
    windows=generate_walk_forward_windows(len(df),initial_train_size,validation_size,test_size)

    test_periods=[]
    full_results=[]
    previous_position=0

    for window in windows:
        result=run_mean_reversion_walk_forward_window(df,window,lookbacks,entry_thresholds,exit_thresholds,n,cost_rate,previous_position)

        full_results.append(result)
        test_periods.append(result["test_results"])
        previous_position=int(result["test_results"]["Position"].iloc[-1])

    combined_test_results=pd.concat(test_periods)

    combined_test_results["CumulativeValue"]=(1+combined_test_results["DailyReturn"].fillna(0)).cumprod()
    combined_test_results["StrategyCumulativeValue"]=(1+combined_test_results["StrategyReturn"]).cumprod()
    combined_test_results["NetStrategyCumulativeValue"]=(1+combined_test_results["NetStrategyReturn"]).cumprod()

    statistics=calculate_backtest_statistics(combined_test_results)

    return {
        "windows":windows,
        "full_results":full_results,
        "combined_test_results":combined_test_results,
        "statistics":statistics
    }
