import pandas as pd

from src.backtest import (
    calculate_backtest_statistics,
    run_backtest,)

from src.momentum import calculate_momentum_signal

from src.split import split_data


def evaluate_momentum_period(df,start_position,end_position,lookback,cost_rate):
    if lookback <= 0:
        raise ValueError("Lookback must be positive.")

    if not 0 <= start_position < end_position <= len(df):
        raise ValueError(
            "Invalid evaluation period."
            )

    warmup_start = max(0,start_position - lookback - 2)

    context_df = df.iloc[warmup_start:end_position].copy()

    context_results = calculate_momentum_signal(context_df,lookback)

    context_results = run_backtest(context_results,cost_rate)

    warmup_rows = (start_position - warmup_start)

    period_results = context_results.iloc[warmup_rows:].copy()

    period_results["CumulativeValue"] = (1 + period_results["DailyReturn"].fillna(0)).cumprod()

    period_results["StrategyCumulativeValue"] = (1 + period_results["StrategyReturn"]).cumprod()

    period_results["NetStrategyCumulativeValue"] = (1 + period_results["NetStrategyReturn"]) .cumprod()

    return period_results


def evaluate_lookbacks_on_period(df,start_position,end_position,lookbacks,cost_rate):
    results = []
    for lookback in lookbacks:
        strategy_df = evaluate_momentum_period(df,start_position,end_position,lookback,cost_rate)
        stats = calculate_backtest_statistics(strategy_df)
        results.append(
            {
                "Lookback":lookback,
                "NetStrategyReturn": stats["net_strategy_return"],
            }
        )

    
    return pd.DataFrame(results)


def select_best_lookback(df):
    if df.empty:
        raise ValueError(
            "Cannot select lookback from empty set."
        )
    

    index = df["NetStrategyReturn"].idxmax()
    return int(df.loc[index, "Lookback"])


def select_top_lookbacks(df, n):
    if df.empty:
        raise ValueError(
            "Cannot select lookbacks from empty results."
        )

    if n <= 0:
        raise ValueError(
            "n must be positive."
        )

    if n > len(df):
        raise ValueError(
            "n cannot exceed the number of results."
        )
    
    top_results = df.sort_values("NetStrategyReturn",ascending=False).head(n)
    lookbacks = top_results["Lookback"].astype(int).tolist()

    return lookbacks

def run_optimisation(df,lookbacks,n,cost_rate,train_ratio=0.6,validation_ratio=0.2):
    train_df, validation_df,_ = split_data(df,train_ratio,validation_ratio)

    train_end = len(train_df)

    validation_end = (train_end + len(validation_df))

    training_results = evaluate_lookbacks_on_period(df,start_position=0,end_position=train_end,lookbacks=lookbacks,cost_rate=cost_rate)

    candidate_lookbacks = select_top_lookbacks(training_results,n,)

    validation_results = evaluate_lookbacks_on_period(df,start_position=train_end,end_position=validation_end,lookbacks=candidate_lookbacks,cost_rate=cost_rate)

    best_lookback = select_best_lookback(validation_results)

    test_results = evaluate_momentum_period(df, start_position=validation_end,end_position=len(df),lookback=best_lookback,cost_rate=cost_rate)

    test_statistics = calculate_backtest_statistics(test_results)

    test_statistics["lookback"] = best_lookback
    test_statistics["cost_rate"] = cost_rate

    return {
        "training_results": training_results,
        "validation_results": validation_results,
        "candidate_lookbacks": candidate_lookbacks,
        "best_lookback": best_lookback,
        "test_results": test_results,
        "test_statistics": test_statistics,
    }