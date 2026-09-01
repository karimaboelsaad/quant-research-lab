import pandas as pd

from src.pairs import(
    apply_pair_transaction_costs,
    calculate_pair_regression,
    run_pair_strategy,
    calculate_pair_statistics
)
from src.split import split_data
from src.walk_forward import calculate_common_training_start


def validate_pair_parameter_grid(lookbacks,entry_thresholds,exit_thresholds):
    if not lookbacks or any(not isinstance(lookback,int) or isinstance(lookback,bool) or lookback<=1 for lookback in lookbacks):
        raise ValueError("Pair lookbacks must contain integers greater than 1.")

    if not entry_thresholds or any(not isinstance(entry,(int,float)) or isinstance(entry,bool) or entry<=0 for entry in entry_thresholds):
        raise ValueError("Pair entry thresholds must be positive.")

    if not exit_thresholds or any(not isinstance(exit_threshold,(int,float)) or isinstance(exit_threshold,bool) or exit_threshold<0 for exit_threshold in exit_thresholds):
        raise ValueError("Pair exit thresholds must be non-negative.")

    if any(exit_threshold>=entry for entry in entry_thresholds for exit_threshold in exit_thresholds):
        raise ValueError("Every pair exit threshold must be smaller than every entry threshold.")


def evaluate_pair_period(df,start_position,end_position,lookback,entry_threshold,exit_threshold,cost_rate,previous_position_a=0,previous_position_b=0,scoring_start_position=None):
    if not 0<=start_position<end_position<=len(df):
        raise ValueError(
            "The start position needs at least 0 and smaller than the end position which needs to be at most the size of the dataframe."
        )

    if start_position==0:
        regression_df=df.iloc[:end_position].copy()
    else:
        regression_df=df.iloc[:start_position].copy()

    regression=calculate_pair_regression(regression_df)

    alpha=regression["alpha"]
    beta=regression["beta"]

    df=df.iloc[:end_position].copy()

    df=run_pair_strategy(df,alpha,beta,lookback,entry_threshold,exit_threshold,cost_rate)

    df["RegressionAlpha"]=alpha
    df["RegressionBeta"]=beta

    if scoring_start_position is None:
        scoring_start_position=start_position

    if not start_position<=scoring_start_position<end_position:
        raise ValueError("Scoring start must fall inside the evaluation period.")

    df=df.iloc[scoring_start_position:].copy()

    # Regression/signal history determines the new desired hedge, while these
    # carried leg units represent what was actually held before the boundary.
    df=apply_pair_transaction_costs(df,cost_rate,previous_position_a,previous_position_b)

    df["StrategyCumulativeValue"]=(1+df["StrategyReturn"]).cumprod()
    df["NetStrategyCumulativeValue"]=(1+df["NetStrategyReturn"]).cumprod()

    return df


def evaluate_pair_parameters_on_period(df,start_position,end_position,lookbacks,entry_thresholds,exit_thresholds,cost_rate,scoring_start_position=None):
    validate_pair_parameter_grid(lookbacks,entry_thresholds,exit_thresholds)

    results=[]

    for lookback in lookbacks:
        for entry_threshold in entry_thresholds:
            for exit_threshold in exit_thresholds:
                if scoring_start_position is None:
                    result_df=evaluate_pair_period(df,start_position,end_position,lookback,entry_threshold,exit_threshold,cost_rate)
                else:
                    result_df=evaluate_pair_period(df,start_position,end_position,lookback,entry_threshold,exit_threshold,cost_rate,scoring_start_position=scoring_start_position)

                stats=calculate_pair_statistics(result_df)

                results.append({
                    "Lookback":lookback,
                    "EntryThreshold":entry_threshold,
                    "ExitThreshold":exit_threshold,
                    "NetStrategyReturn":stats["net_return"]
                })

    return pd.DataFrame(results)


def select_top_pair_parameters(df, n):
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


def select_best_pair_parameters(df):
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


def evaluate_pair_candidates_on_period(df, start_position, end_position, candidates_df, cost_rate):
    results=[]

    for _, candidate in candidates_df.iterrows():
        lookback=int(candidate["Lookback"])
        entry_threshold=candidate["EntryThreshold"]
        exit_threshold=candidate["ExitThreshold"]

        result_df=evaluate_pair_period(df,start_position,end_position,lookback,entry_threshold,exit_threshold,cost_rate)

        stats=calculate_pair_statistics(result_df)

        results.append({
            "Lookback":lookback,
            "EntryThreshold":entry_threshold,
            "ExitThreshold":exit_threshold,
            "NetStrategyReturn":stats["net_return"]
        })

    return pd.DataFrame(results)


def run_pair_optimisation(df, lookbacks, entry_thresholds, exit_thresholds, n, cost_rate):
    train_df, validation_df, _=split_data(df)

    train_end=len(train_df)
    validation_end=train_end+len(validation_df)
    training_start=calculate_common_training_start(lookbacks,train_end)

    training_results=evaluate_pair_parameters_on_period(df,0,train_end,lookbacks,entry_thresholds,exit_thresholds,cost_rate,training_start)

    candidate_parameters=select_top_pair_parameters(training_results,n)

    validation_results=evaluate_pair_candidates_on_period(df,train_end,validation_end,candidate_parameters,cost_rate)

    best_parameters=select_best_pair_parameters(validation_results)

    best_lookback=best_parameters["Lookback"]
    best_entry_threshold=best_parameters["EntryThreshold"]
    best_exit_threshold=best_parameters["ExitThreshold"]

    test_results=evaluate_pair_period(df,validation_end,len(df),best_lookback,best_entry_threshold,best_exit_threshold,cost_rate)

    test_statistics=calculate_pair_statistics(test_results)

    return {
        "training_results":training_results,
        "candidate_parameters":candidate_parameters,
        "validation_results":validation_results,
        "best_parameters":best_parameters,
        "test_results":test_results,
        "test_statistics":test_statistics
    }
