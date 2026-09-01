import pandas as pd
import pytest

import src.pairs_optimise as pairs_optimise

from src.pairs_optimise import(
    evaluate_pair_period,
    evaluate_pair_parameters_on_period,
    select_top_pair_parameters,
    select_best_pair_parameters,
    evaluate_pair_candidates_on_period,
    run_pair_optimisation
)


def make_pair_df():
    return pd.DataFrame({
        "CloseA":[21,23,25,27,29,31,33,35],
        "CloseB":[10,11,12,13,14,15,16,17]
    })


def fake_strategy_result(df):
    df=df.copy()

    df["StrategyReturn"]=0.01
    df["PositionA"]=0.0
    df["PositionB"]=0.0
    df["SpreadPosition"]=0.0
    df["PreviousCloseA"]=df["CloseA"].shift(1)
    df["PreviousCloseB"]=df["CloseB"].shift(1)
    df["NetStrategyReturn"]=df["StrategyReturn"]
    df["StrategyCumulativeValue"]=(1+df["StrategyReturn"]).cumprod()
    df["NetStrategyCumulativeValue"]=(1+df["NetStrategyReturn"]).cumprod()

    return df


def test_evaluate_pair_period_uses_pre_period_regression(monkeypatch):
    df=make_pair_df()

    captured={}

    def fake_regression(regression_df):
        captured["length"]=len(regression_df)

        return {
            "alpha":1,
            "beta":2
        }

    def fake_run(df, alpha, beta, lookback, entry_threshold, exit_threshold, cost_rate):
        return fake_strategy_result(df)

    monkeypatch.setattr(
        pairs_optimise,
        "calculate_pair_regression",
        fake_regression
    )

    monkeypatch.setattr(
        pairs_optimise,
        "run_pair_strategy",
        fake_run
    )

    result=evaluate_pair_period(
        df,
        4,
        6,
        3,
        2,
        0.5,
        0.001
    )

    assert captured["length"]==4
    assert len(result)==2
    assert result["RegressionAlpha"].tolist()==[1,1]
    assert result["RegressionBeta"].tolist()==[2,2]
    assert result["StrategyCumulativeValue"].iloc[0]==pytest.approx(1.01)
    assert result["NetStrategyCumulativeValue"].iloc[0]==pytest.approx(1.01)


def test_evaluate_pair_training_period_uses_training_data_for_regression(monkeypatch):
    df=make_pair_df()

    captured={}

    def fake_regression(regression_df):
        captured["length"]=len(regression_df)

        return {
            "alpha":1,
            "beta":2
        }

    def fake_run(df, alpha, beta, lookback, entry_threshold, exit_threshold, cost_rate):
        return fake_strategy_result(df)

    monkeypatch.setattr(
        pairs_optimise,
        "calculate_pair_regression",
        fake_regression
    )

    monkeypatch.setattr(
        pairs_optimise,
        "run_pair_strategy",
        fake_run
    )

    evaluate_pair_period(
        df,
        0,
        4,
        3,
        2,
        0.5,
        0.001
    )

    assert captured["length"]==4


def test_evaluate_pair_period_invalid_positions():
    df=make_pair_df()

    with pytest.raises(ValueError):
        evaluate_pair_period(
            df,
            -1,
            4,
            3,
            2,
            0.5,
            0.001
        )

    with pytest.raises(ValueError):
        evaluate_pair_period(
            df,
            4,
            4,
            3,
            2,
            0.5,
            0.001
        )


def test_evaluate_pair_parameters_on_period(monkeypatch):
    df=make_pair_df()

    def fake_evaluate(df, start_position, end_position, lookback, entry_threshold, exit_threshold, cost_rate):
        return pd.DataFrame({
            "Score":[lookback+entry_threshold-exit_threshold]
        })

    def fake_statistics(result_df):
        return {
            "net_return":result_df["Score"].iloc[0]
        }

    monkeypatch.setattr(
        pairs_optimise,
        "evaluate_pair_period",
        fake_evaluate
    )

    monkeypatch.setattr(
        pairs_optimise,
        "calculate_pair_statistics",
        fake_statistics
    )

    result=evaluate_pair_parameters_on_period(
        df,
        0,
        4,
        [5,10],
        [1,2],
        [0.25,0.5],
        0.001
    )

    assert len(result)==8

    assert set(result.columns)=={
        "Lookback",
        "EntryThreshold",
        "ExitThreshold",
        "NetStrategyReturn"
    }

    assert (result["ExitThreshold"]<result["EntryThreshold"]).all()


def test_evaluate_pair_parameters_rejects_invalid_grid():
    df=make_pair_df()

    with pytest.raises(ValueError,match="smaller than every entry"):
        evaluate_pair_parameters_on_period(df,0,4,[2,3],[1,2],[0.5,1],0.001)


def test_select_top_pair_parameters():
    df=pd.DataFrame({
        "Lookback":[5,10,15],
        "EntryThreshold":[1,1.5,2],
        "ExitThreshold":[0.5,0.5,1],
        "NetStrategyReturn":[0.1,0.3,0.2]
    })

    result=select_top_pair_parameters(df,2)

    assert len(result)==2
    assert result.iloc[0]["NetStrategyReturn"]==0.3
    assert result.iloc[1]["NetStrategyReturn"]==0.2


def test_select_top_pair_parameters_invalid_n():
    df=pd.DataFrame({
        "NetStrategyReturn":[0.1,0.2]
    })

    with pytest.raises(ValueError):
        select_top_pair_parameters(df,0)

    with pytest.raises(ValueError):
        select_top_pair_parameters(df,3)


def test_select_best_pair_parameters():
    df=pd.DataFrame({
        "Lookback":[5,10,15],
        "EntryThreshold":[1,2,2.5],
        "ExitThreshold":[0.5,0.5,1],
        "NetStrategyReturn":[0.1,0.4,0.2]
    })

    result=select_best_pair_parameters(df)

    assert result=={
        "Lookback":10,
        "EntryThreshold":2,
        "ExitThreshold":0.5
    }


def test_evaluate_pair_candidates_on_period(monkeypatch):
    df=make_pair_df()

    candidates=pd.DataFrame({
        "Lookback":[5,10],
        "EntryThreshold":[1,2],
        "ExitThreshold":[0.5,1]
    })

    def fake_evaluate(df, start_position, end_position, lookback, entry_threshold, exit_threshold, cost_rate):
        return pd.DataFrame({
            "Score":[lookback]
        })

    def fake_statistics(result_df):
        return {
            "net_return":result_df["Score"].iloc[0]
        }

    monkeypatch.setattr(
        pairs_optimise,
        "evaluate_pair_period",
        fake_evaluate
    )

    monkeypatch.setattr(
        pairs_optimise,
        "calculate_pair_statistics",
        fake_statistics
    )

    result=evaluate_pair_candidates_on_period(
        df,
        4,
        6,
        candidates,
        0.001
    )

    assert len(result)==2
    assert result["Lookback"].tolist()==[5,10]
    assert result["NetStrategyReturn"].tolist()==[5,10]


def test_run_pair_optimisation(monkeypatch):
    df=make_pair_df()

    train_df=df.iloc[:4]
    validation_df=df.iloc[4:6]
    test_df=df.iloc[6:]

    training_results=pd.DataFrame({
        "Lookback":[2,3],
        "EntryThreshold":[1,2],
        "ExitThreshold":[0.5,1],
        "NetStrategyReturn":[0.1,0.2]
    })

    candidate_parameters=training_results.iloc[[1]].copy()

    validation_results=pd.DataFrame({
        "Lookback":[3],
        "EntryThreshold":[2],
        "ExitThreshold":[1],
        "NetStrategyReturn":[0.3]
    })

    best_parameters={
        "Lookback":3,
        "EntryThreshold":2,
        "ExitThreshold":1
    }

    test_results=pd.DataFrame({
        "StrategyReturn":[0.01,0.02],
        "NetStrategyReturn":[0.009,0.019],
        "StrategyCumulativeValue":[1.01,1.0302],
        "NetStrategyCumulativeValue":[1.009,1.028171],
        "Turnover":[0,1],
        "SpreadPosition":[0,1]
    })

    monkeypatch.setattr(
        pairs_optimise,
        "split_data",
        lambda df:(train_df,validation_df,test_df)
    )

    monkeypatch.setattr(
        pairs_optimise,
        "evaluate_pair_parameters_on_period",
        lambda *args:training_results
    )

    monkeypatch.setattr(
        pairs_optimise,
        "select_top_pair_parameters",
        lambda *args:candidate_parameters
    )

    monkeypatch.setattr(
        pairs_optimise,
        "evaluate_pair_candidates_on_period",
        lambda *args:validation_results
    )

    monkeypatch.setattr(
        pairs_optimise,
        "select_best_pair_parameters",
        lambda *args:best_parameters
    )

    monkeypatch.setattr(
        pairs_optimise,
        "evaluate_pair_period",
        lambda *args:test_results
    )

    monkeypatch.setattr(
        pairs_optimise,
        "calculate_pair_statistics",
        lambda df:{"net_return":0.028171}
    )

    result=run_pair_optimisation(
        df,
        [2,3],
        [1,2],
        [0,0.5],
        1,
        0.001
    )

    assert result["best_parameters"]==best_parameters
    assert result["test_statistics"]["net_return"]==0.028171
    assert result["training_results"].equals(training_results)
    assert result["validation_results"].equals(validation_results)
