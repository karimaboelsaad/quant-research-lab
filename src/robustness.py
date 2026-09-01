import numpy as np
import pandas as pd

from src.metrics import PERIODS_PER_YEAR,validate_returns


def calculate_expected_block_length(observations,multiplier=1.0):
    if not isinstance(observations,int) or isinstance(observations,bool) or observations<2:
        raise ValueError("Observations must be an integer of at least 2.")

    if not isinstance(multiplier,(int,float)) or isinstance(multiplier,bool) or multiplier<=0:
        raise ValueError("Block-length multiplier must be positive.")

    base_length=max(5,round(observations**(1/3)))
    adjusted_length=max(2,round(base_length*multiplier))
    maximum_length=max(2,observations//10)

    return min(adjusted_length,maximum_length)


def _stationary_bootstrap_indices(observations,expected_block_length,rng):
    indices=np.empty(observations,dtype=int)
    filled=0
    restart_probability=1/expected_block_length

    while filled<observations:
        start=int(rng.integers(0,observations))
        block_length=int(rng.geometric(restart_probability))
        block_length=min(block_length,observations-filled)
        indices[filled:filled+block_length]=(start+np.arange(block_length))%observations
        filled+=block_length

    return indices


def _sharpe_ratio(returns):
    volatility=np.std(returns,ddof=1)

    if not np.isfinite(volatility) or volatility==0:
        return np.nan

    return np.mean(returns)/volatility*np.sqrt(PERIODS_PER_YEAR)


def _bootstrap_statistics(returns):
    mean_return=np.mean(returns)

    return {
        "MeanDailyReturnBps":mean_return*10000,
        "AnnualisedMeanReturn":mean_return*PERIODS_PER_YEAR,
        "SharpeRatio":_sharpe_ratio(returns)
    }


def stationary_bootstrap_intervals(returns,resamples,confidence_level,seed,minimum_observations=100):
    returns=validate_returns(returns).to_numpy()

    if not isinstance(resamples,int) or isinstance(resamples,bool) or resamples<=0:
        raise ValueError("Bootstrap resamples must be a positive integer.")

    if not isinstance(seed,int) or isinstance(seed,bool):
        raise ValueError("Bootstrap seed must be an integer.")

    if not 0<confidence_level<1:
        raise ValueError("Confidence level must be between 0 and 1.")

    if not isinstance(minimum_observations,int) or isinstance(minimum_observations,bool) or minimum_observations<=0:
        raise ValueError("Minimum observations must be a positive integer.")

    observations=len(returns)
    observed_statistics=_bootstrap_statistics(returns)

    if observations<minimum_observations:
        return pd.DataFrame([
            {
                "Statistic":statistic,
                "Observed":observed,
                "Lower":np.nan,
                "Median":np.nan,
                "Upper":np.nan,
                "ConfidenceLevel":confidence_level,
                "ExpectedBlockLength":np.nan,
                "BlockVariant":"primary",
                "Resamples":resamples,
                "Seed":seed,
                "Observations":observations,
                "Status":"insufficient_sample",
                "EvaluationType":"common_oos"
            }
            for statistic,observed in observed_statistics.items()
        ])

    variants=[("half",0.5),("primary",1.0),("double",2.0)]
    lower_percentile=(1-confidence_level)/2*100
    upper_percentile=(1+confidence_level)/2*100
    rows=[]

    for variant_number,(variant,multiplier) in enumerate(variants):
        block_length=calculate_expected_block_length(observations,multiplier)
        rng=np.random.default_rng(seed+variant_number)
        samples={statistic:np.empty(resamples) for statistic in observed_statistics}

        for resample in range(resamples):
            indices=_stationary_bootstrap_indices(observations,block_length,rng)
            statistics=_bootstrap_statistics(returns[indices])

            for statistic,value in statistics.items():
                samples[statistic][resample]=value

        for statistic,observed in observed_statistics.items():
            finite_samples=samples[statistic][np.isfinite(samples[statistic])]

            if finite_samples.size==0:
                lower=median=upper=np.nan
                status="undefined_flat_returns"
            else:
                lower,median,upper=np.percentile(finite_samples,[lower_percentile,50,upper_percentile])
                status="ok"

            rows.append({
                "Statistic":statistic,
                "Observed":observed,
                "Lower":lower,
                "Median":median,
                "Upper":upper,
                "ConfidenceLevel":confidence_level,
                "ExpectedBlockLength":block_length,
                "BlockVariant":variant,
                "Resamples":resamples,
                "Seed":seed,
                "Observations":observations,
                "Status":status,
                "EvaluationType":"common_oos"
            })

    return pd.DataFrame(rows)


def _circular_block_permutation_indices(observations,block_length,rng):
    offset=int(rng.integers(0,observations))
    ordered_indices=(offset+np.arange(observations))%observations
    blocks=[ordered_indices[start:start+block_length] for start in range(0,observations,block_length)]
    order=rng.permutation(len(blocks))

    return np.concatenate([blocks[index] for index in order])


def _calculate_overlay_turnover(weights,returns,portfolio_returns,previous_weights,previous_returns,previous_portfolio_return):
    turnover=np.empty(len(portfolio_returns))

    for row in range(len(portfolio_returns)):
        previous_growth=1+previous_portfolio_return

        if previous_growth<=0:
            raise ValueError("Portfolio value cannot fall to zero or below.")

        drifted_weights=previous_weights*(1+previous_returns)/previous_growth
        turnover[row]=np.abs(weights[row]-drifted_weights).sum()
        previous_weights=weights[row]
        previous_returns=returns[row]
        previous_portfolio_return=portfolio_returns[row]

    return turnover


def run_dynamic_timing_placebo(dynamic_results,assets,overlay_cost_rate,resamples,confidence_level,seed,minimum_observations=100):
    if dynamic_results.empty:
        raise ValueError("Dynamic results cannot be empty.")

    if not assets:
        raise ValueError("Assets cannot be empty.")

    if overlay_cost_rate<0:
        raise ValueError("Overlay cost rate cannot be negative.")

    if not isinstance(resamples,int) or isinstance(resamples,bool) or resamples<=0:
        raise ValueError("Placebo resamples must be a positive integer.")

    if not isinstance(seed,int) or isinstance(seed,bool):
        raise ValueError("Placebo seed must be an integer.")

    if not isinstance(minimum_observations,int) or isinstance(minimum_observations,bool) or minimum_observations<=0:
        raise ValueError("Minimum observations must be a positive integer.")

    if not 0<confidence_level<1:
        raise ValueError("Confidence level must be between 0 and 1.")

    return_columns=[]
    weight_columns=[]

    for asset in assets:
        for strategy in ["Momentum","MeanReversion"]:
            return_columns.append(f"{asset}_{strategy}Return")
            weight_columns.append(f"{asset}_{strategy}Weight")

    required_columns=return_columns+weight_columns+["CommonComparison"]
    missing_columns=[column for column in required_columns if column not in dynamic_results.columns]

    if missing_columns:
        raise ValueError(f"Dynamic results are missing placebo columns: {missing_columns}")

    evaluation_positions=np.flatnonzero(dynamic_results["CommonComparison"].to_numpy())

    if evaluation_positions.size==0 or (evaluation_positions.size>1 and not np.all(np.diff(evaluation_positions)==1)):
        raise ValueError("Common comparison rows must be one non-empty contiguous period.")

    start_position=int(evaluation_positions[0])
    returns=dynamic_results.iloc[evaluation_positions][return_columns].to_numpy(dtype=float)
    weights=dynamic_results.iloc[evaluation_positions][weight_columns].to_numpy(dtype=float)
    observations=len(evaluation_positions)

    if start_position==0:
        previous_weights=np.zeros(len(weight_columns))
        previous_returns=np.zeros(len(return_columns))
        previous_portfolio_return=0.0
    else:
        previous_weights=dynamic_results.iloc[start_position-1][weight_columns].to_numpy(dtype=float)
        previous_returns=dynamic_results.iloc[start_position-1][return_columns].to_numpy(dtype=float)
        previous_portfolio_return=float((previous_weights*previous_returns).sum())

    observed_portfolio_returns=(weights*returns).sum(axis=1)
    observed_turnover=_calculate_overlay_turnover(weights,returns,observed_portfolio_returns,previous_weights.copy(),previous_returns.copy(),previous_portfolio_return)
    observed_net_returns=observed_portfolio_returns-observed_turnover*overlay_cost_rate
    observed_sharpe=_sharpe_ratio(observed_net_returns)
    block_length=calculate_expected_block_length(max(observations,2))

    base_row={
        "Test":"dynamic_block_timing_placebo",
        "Statistic":"SharpeRatio",
        "Observed":observed_sharpe,
        "ExpectedBlockLength":block_length,
        "Resamples":resamples,
        "Seed":seed,
        "Observations":observations,
        "EvaluationType":"common_oos"
    }

    if observations<minimum_observations:
        return pd.DataFrame([{
            **base_row,
            "NullMean":np.nan,
            "NullLower":np.nan,
            "NullMedian":np.nan,
            "NullUpper":np.nan,
            "RawPValue":np.nan,
            "AdjustedPValue":np.nan,
            "Status":"insufficient_sample"
        }])

    if not np.isfinite(observed_sharpe):
        return pd.DataFrame([{
            **base_row,
            "NullMean":np.nan,
            "NullLower":np.nan,
            "NullMedian":np.nan,
            "NullUpper":np.nan,
            "RawPValue":np.nan,
            "AdjustedPValue":np.nan,
            "Status":"undefined_flat_returns"
        }])

    rng=np.random.default_rng(seed)
    null_sharpes=np.empty(resamples)

    for resample in range(resamples):
        permutation=_circular_block_permutation_indices(observations,block_length,rng)
        permuted_returns=returns[permutation]
        portfolio_returns=(weights*permuted_returns).sum(axis=1)
        turnover=_calculate_overlay_turnover(weights,permuted_returns,portfolio_returns,previous_weights.copy(),previous_returns.copy(),previous_portfolio_return)
        net_returns=portfolio_returns-turnover*overlay_cost_rate
        null_sharpes[resample]=_sharpe_ratio(net_returns)

    finite_null=null_sharpes[np.isfinite(null_sharpes)]

    if finite_null.size==0:
        return pd.DataFrame([{
            **base_row,
            "NullMean":np.nan,
            "NullLower":np.nan,
            "NullMedian":np.nan,
            "NullUpper":np.nan,
            "RawPValue":np.nan,
            "AdjustedPValue":np.nan,
            "Status":"undefined_flat_placebos"
        }])

    lower_percentile=(1-confidence_level)/2*100
    upper_percentile=(1+confidence_level)/2*100
    raw_p_value=(1+np.sum(finite_null>=observed_sharpe))/(len(finite_null)+1)
    lower,median,upper=np.percentile(finite_null,[lower_percentile,50,upper_percentile])

    return pd.DataFrame([{
        **base_row,
        "NullMean":finite_null.mean(),
        "NullLower":lower,
        "NullMedian":median,
        "NullUpper":upper,
        "RawPValue":raw_p_value,
        "AdjustedPValue":raw_p_value,
        "Status":"ok"
    }])
