import pandas as pd

from src.pairs import calculate_pair_statistics,test_pair_cointegration
from src.pairs_optimise import(
    evaluate_pair_period,
    evaluate_pair_parameters_on_period,
    select_top_pair_parameters,
    select_best_pair_parameters,
    evaluate_pair_candidates_on_period
)
from src.walk_forward import calculate_common_training_start,generate_walk_forward_windows


def run_pair_walk_forward_window(df,window,lookbacks,entry_thresholds,exit_thresholds,n,cost_rate,previous_position_a=0,previous_position_b=0):
    train_end=window["train_end"]
    validation_end=window["validation_end"]
    test_end=window["test_end"]
    training_start=calculate_common_training_start(lookbacks,train_end)

    training_results=evaluate_pair_parameters_on_period(df,0,train_end,lookbacks,entry_thresholds,exit_thresholds,cost_rate,training_start)

    candidate_parameters=select_top_pair_parameters(training_results,n)

    validation_results=evaluate_pair_candidates_on_period(df,train_end,validation_end,candidate_parameters,cost_rate)

    best_parameters=select_best_pair_parameters(validation_results)

    best_lookback=best_parameters["Lookback"]
    best_entry_threshold=best_parameters["EntryThreshold"]
    best_exit_threshold=best_parameters["ExitThreshold"]

    test_results=evaluate_pair_period(df,validation_end,test_end,best_lookback,best_entry_threshold,best_exit_threshold,cost_rate,previous_position_a,previous_position_b)

    test_statistics=calculate_pair_statistics(test_results)
    pretest_cointegration=test_pair_cointegration(df.iloc[:validation_end])

    return {
        "training_results":training_results,
        "validation_results":validation_results,
        "candidate_parameters":candidate_parameters,
        "best_parameters":best_parameters,
        "test_results":test_results,
        "test_statistics":test_statistics,
        "pretest_cointegration":pretest_cointegration
    }


def run_pair_walk_forward(df, lookbacks, entry_thresholds, exit_thresholds, n, cost_rate, initial_train_size, validation_size, test_size):
    windows=generate_walk_forward_windows(len(df),initial_train_size,validation_size,test_size)

    test_periods=[]
    full_results=[]
    previous_position_a=0
    previous_position_b=0

    for window in windows:
        result=run_pair_walk_forward_window(df,window,lookbacks,entry_thresholds,exit_thresholds,n,cost_rate,previous_position_a,previous_position_b)

        full_results.append(result)
        test_periods.append(result["test_results"])
        previous_position_a=result["test_results"]["PositionA"].iloc[-1]
        previous_position_b=result["test_results"]["PositionB"].iloc[-1]

    combined_test_results=pd.concat(test_periods)

    combined_test_results["StrategyCumulativeValue"]=(1+combined_test_results["StrategyReturn"]).cumprod()
    combined_test_results["NetStrategyCumulativeValue"]=(1+combined_test_results["NetStrategyReturn"]).cumprod()

    statistics=calculate_pair_statistics(combined_test_results)

    return {
        "windows":windows,
        "full_results":full_results,
        "combined_test_results":combined_test_results,
        "statistics":statistics
    }
