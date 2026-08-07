import pandas as pd
from src.optimise import(
    evaluate_lookbacks_on_period,
    evaluate_momentum_period,
    select_top_lookbacks,
    select_best_lookback
)
from src.backtest import calculate_backtest_statistics




def generate_walk_forward_windows(number_of_rows,initial_train_size,validation_size,test_size):
    if number_of_rows <= 0:
        raise ValueError(
            "number_of_rows must be positive."
        )

    if initial_train_size <= 0:
        raise ValueError(
            "initial_train_size must be positive."
        )

    if validation_size <= 0:
        raise ValueError(
            "validation_size must be positive."
        )

    if test_size <= 0:
        raise ValueError(
            "test_size must be positive."
        )
    
    minimum_required = (
        initial_train_size
        + validation_size
        + test_size
    )

    if minimum_required > number_of_rows:
        raise ValueError(
            "Not enough rows for one complete window."
        )

    results = []
    number_of_windows = ((number_of_rows-initial_train_size-validation_size)//test_size)
    for i in range (number_of_windows):
        train_end=initial_train_size+test_size*i
        validation_end=train_end+validation_size
        test_end=validation_end+test_size
        

        results.append(
            {
                "train_end":train_end,
                "validation_end":validation_end,
                "test_end":test_end
            }
        )
    
    return results


def run_walk_forward_window(df, window, lookbacks, n, cost_rate):
    train_end = window["train_end"]
    validation_end = window["validation_end"]
    test_end = window["test_end"]

    train_results = evaluate_lookbacks_on_period(df, 0, train_end, lookbacks, cost_rate)

    validation_lookbacks = select_top_lookbacks(train_results, n)

    validation_results = evaluate_lookbacks_on_period(df, train_end, validation_end, validation_lookbacks, cost_rate)

    best_lookback = select_best_lookback(validation_results)

    test_results = evaluate_momentum_period(df, validation_end, test_end, best_lookback, cost_rate)

    test_statistics = calculate_backtest_statistics(test_results)

    return {
        "training_results": train_results,
        "validation_results": validation_results,
        "candidate_lookbacks": validation_lookbacks,
        "best_lookback": best_lookback,
        "test_results": test_results,
        "test_statistics": test_statistics
    }



def run_walk_forward(df, lookbacks, n, cost_rate, initial_train_size, validation_size, test_size):
    windows=generate_walk_forward_windows(len(df),initial_train_size,validation_size,test_size)
    test_periods=[]
    full_results=[]
    for window in windows:
        result=run_walk_forward_window(df,window,lookbacks,n,cost_rate)
        full_results.append(result)
        test_periods.append(result["test_results"])

    combined_test_results=pd.concat(test_periods)

    combined_test_results["CumulativeValue"]=(1 + combined_test_results["DailyReturn"].fillna(0)).cumprod()   
    combined_test_results["StrategyCumulativeValue"]=(1 + combined_test_results["StrategyReturn"]).cumprod()
    combined_test_results["NetStrategyCumulativeValue"]=(1 + combined_test_results["NetStrategyReturn"]).cumprod()
    
    statistics=calculate_backtest_statistics(combined_test_results)

    return {
        "windows":windows,
        "full_results":full_results,
        "combined_test_results":combined_test_results,
        "statistics":statistics
    }