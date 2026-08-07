import pandas as pd

from src.backtest import calculate_backtest_statistics

from src.momentum_optimise import(
    evaluate_lookbacks_on_period,
    evaluate_momentum_period,
    select_top_lookbacks,
    select_best_lookback
)

from src.walk_forward import generate_walk_forward_windows


def run_momentum_walk_forward_window(df, window, lookbacks, n, cost_rate):
    train_end=window["train_end"]
    validation_end=window["validation_end"]
    test_end=window["test_end"]

    training_results=evaluate_lookbacks_on_period(df,0,train_end,lookbacks,cost_rate)

    candidate_lookbacks=select_top_lookbacks(training_results,n)

    validation_results=evaluate_lookbacks_on_period(df,train_end,validation_end,candidate_lookbacks,cost_rate)

    best_lookback=select_best_lookback(validation_results)

    test_results=evaluate_momentum_period(df,validation_end,test_end,best_lookback,cost_rate)

    test_statistics=calculate_backtest_statistics(test_results)

    return {
        "training_results":training_results,
        "validation_results":validation_results,
        "candidate_lookbacks":candidate_lookbacks,
        "best_lookback":best_lookback,
        "test_results":test_results,
        "test_statistics":test_statistics
    }


def run_momentum_walk_forward(df, lookbacks, n, cost_rate, initial_train_size, validation_size, test_size):
    windows=generate_walk_forward_windows(len(df),initial_train_size,validation_size,test_size)

    test_periods=[]
    full_results=[]

    for window in windows:
        result=run_momentum_walk_forward_window(df,window,lookbacks,n,cost_rate)

        full_results.append(result)
        test_periods.append(result["test_results"])

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