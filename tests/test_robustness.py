import numpy as np
import pandas as pd

from src.robustness import calculate_expected_block_length,run_dynamic_timing_placebo,stationary_bootstrap_intervals


def test_calculate_expected_block_length_caps_at_ten_blocks():
    assert calculate_expected_block_length(100)==5
    assert calculate_expected_block_length(1000)==10
    assert calculate_expected_block_length(1000,2)==20


def test_stationary_bootstrap_is_reproducible_and_reports_block_variants():
    rng=np.random.default_rng(7)
    returns=pd.Series(rng.normal(0.0005,0.01,300))

    first=stationary_bootstrap_intervals(returns,200,0.95,123)
    second=stationary_bootstrap_intervals(returns,200,0.95,123)

    pd.testing.assert_frame_equal(first,second)
    assert set(first["BlockVariant"])=={"half","primary","double"}
    assert (first["Lower"]<=first["Median"]).all()
    assert (first["Median"]<=first["Upper"]).all()
    assert (first["Status"]=="ok").all()


def test_stationary_bootstrap_flags_short_and_flat_samples():
    short=stationary_bootstrap_intervals(pd.Series([0.01]*20),20,0.95,1,100)
    flat=stationary_bootstrap_intervals(pd.Series([0.0]*120),20,0.95,1,100)

    assert (short["Status"]=="insufficient_sample").all()
    assert flat.loc[flat["Statistic"]=="SharpeRatio","Status"].eq("undefined_flat_returns").all()


def _build_dynamic_placebo_frame(predictive):
    rng=np.random.default_rng(15)
    observations=301
    selection=rng.integers(0,2,observations)
    noise=rng.normal(0,0.002,observations)

    if predictive:
        momentum_returns=np.where(selection==1,0.004,-0.004)+noise
        mean_reversion_returns=np.where(selection==0,0.004,-0.004)-noise
        momentum_weights=selection.astype(float)
        mean_reversion_weights=1-momentum_weights
    else:
        momentum_returns=rng.normal(0.0002,0.01,observations)
        mean_reversion_returns=rng.normal(0.0002,0.01,observations)
        momentum_weights=np.full(observations,0.5)
        mean_reversion_weights=np.full(observations,0.5)

    return pd.DataFrame({
        "A_MomentumReturn":momentum_returns,
        "A_MeanReversionReturn":mean_reversion_returns,
        "A_MomentumWeight":momentum_weights,
        "A_MeanReversionWeight":mean_reversion_weights,
        "CommonComparison":[False]+[True]*(observations-1)
    })


def test_dynamic_timing_placebo_detects_predictive_timing():
    results=run_dynamic_timing_placebo(_build_dynamic_placebo_frame(True),["A"],0.0,400,0.95,44)

    assert results.loc[0,"Status"]=="ok"
    assert results.loc[0,"Observed"]>results.loc[0,"NullUpper"]
    assert results.loc[0,"RawPValue"]<0.01


def test_dynamic_timing_placebo_does_not_reward_constant_allocation():
    results=run_dynamic_timing_placebo(_build_dynamic_placebo_frame(False),["A"],0.0,100,0.95,44)

    assert results.loc[0,"Status"]=="ok"
    assert results.loc[0,"RawPValue"]>0.1
