import pandas as pd
import pytest

import src.pairs_walk_forward as pairs_walk_forward

from src.pairs_walk_forward import(
    run_pair_walk_forward_window,
    run_pair_walk_forward
)


def test_run_pair_walk_forward_window(monkeypatch):
    df=pd.DataFrame({
        "CloseA":[20+i+(i%3)*0.2 for i in range(20)],
        "CloseB":range(10,30)
    })

    window={
        "train_end":10,
        "validation_end":15,
        "test_end":20
    }

    training_results=pd.DataFrame({
        "Lookback":[5,9],
        "EntryThreshold":[1,2],
        "ExitThreshold":[0.5,1],
        "NetStrategyReturn":[0.1,0.2]
    })

    candidate_parameters=training_results.iloc[[1]].copy()

    validation_results=pd.DataFrame({
        "Lookback":[9],
        "EntryThreshold":[2],
        "ExitThreshold":[1],
        "NetStrategyReturn":[0.3]
    })

    best_parameters={
        "Lookback":9,
        "EntryThreshold":2,
        "ExitThreshold":1
    }

    test_results=pd.DataFrame({
        "StrategyReturn":[0.01,0.02],
        "NetStrategyReturn":[0.009,0.019],
        "StrategyCumulativeValue":[1.01,1.0302],
        "NetStrategyCumulativeValue":[1.009,1.028171],
        "Turnover":[0,1],
        "SpreadPosition":[0,1],
        "PositionA":[0,1],
        "PositionB":[0,-2]
    })

    captured={}

    monkeypatch.setattr(
        pairs_walk_forward,
        "evaluate_pair_parameters_on_period",
        lambda *args:training_results
    )

    monkeypatch.setattr(
        pairs_walk_forward,
        "select_top_pair_parameters",
        lambda *args:candidate_parameters
    )

    monkeypatch.setattr(
        pairs_walk_forward,
        "evaluate_pair_candidates_on_period",
        lambda *args:validation_results
    )

    monkeypatch.setattr(
        pairs_walk_forward,
        "select_best_pair_parameters",
        lambda *args:best_parameters
    )

    def fake_evaluate_period(df,start_position,end_position,lookback,entry_threshold,exit_threshold,cost_rate,previous_position_a,previous_position_b):
        captured["start_position"]=start_position
        captured["end_position"]=end_position
        captured["lookback"]=lookback
        captured["entry_threshold"]=entry_threshold
        captured["exit_threshold"]=exit_threshold
        captured["previous_position_a"]=previous_position_a
        captured["previous_position_b"]=previous_position_b

        return test_results

    monkeypatch.setattr(
        pairs_walk_forward,
        "evaluate_pair_period",
        fake_evaluate_period
    )

    monkeypatch.setattr(
        pairs_walk_forward,
        "calculate_pair_statistics",
        lambda df:{"net_return":0.028171}
    )

    result=run_pair_walk_forward_window(
        df,
        window,
        [5,9],
        [1,2],
        [0,0.5],
        1,
        0.001
    )

    assert captured["start_position"]==15
    assert captured["end_position"]==20
    assert captured["lookback"]==9
    assert captured["entry_threshold"]==2
    assert captured["exit_threshold"]==1
    assert captured["previous_position_a"]==0
    assert captured["previous_position_b"]==0

    assert result["best_parameters"]==best_parameters
    assert result["test_results"].equals(test_results)
    assert result["test_statistics"]["net_return"]==0.028171


def test_run_pair_walk_forward(monkeypatch):
    df=pd.DataFrame({
        "CloseA":range(20,50),
        "CloseB":range(10,40)
    })

    windows=[
        {
            "train_end":10,
            "validation_end":15,
            "test_end":20
        },
        {
            "train_end":15,
            "validation_end":20,
            "test_end":25
        }
    ]

    first_test=pd.DataFrame({
        "StrategyReturn":[0.01,0.02],
        "NetStrategyReturn":[0.009,0.019],
        "PositionA":[0.0,1.0],
        "PositionB":[0.0,-2.0]
    })

    second_test=pd.DataFrame({
        "StrategyReturn":[-0.01,0.03],
        "NetStrategyReturn":[-0.011,0.029],
        "PositionA":[1.0,0.0],
        "PositionB":[-3.0,0.0]
    })

    window_results=[
        {
            "test_results":first_test
        },
        {
            "test_results":second_test
        }
    ]

    monkeypatch.setattr(
        pairs_walk_forward,
        "generate_walk_forward_windows",
        lambda *args:windows
    )

    call_count={"value":0}
    previous_positions=[]

    def fake_window(*args):
        previous_positions.append(args[-2:])
        result=window_results[call_count["value"]]
        call_count["value"]+=1

        return result

    monkeypatch.setattr(
        pairs_walk_forward,
        "run_pair_walk_forward_window",
        fake_window
    )

    monkeypatch.setattr(
        pairs_walk_forward,
        "calculate_pair_statistics",
        lambda df:{
            "observations":len(df),
            "net_return":df["NetStrategyCumulativeValue"].iloc[-1]-1
        }
    )

    result=run_pair_walk_forward(
        df,
        [5,10],
        [1,2],
        [0.5,1],
        1,
        0.001,
        10,
        5,
        5
    )

    combined=result["combined_test_results"]

    assert len(result["windows"])==2
    assert len(result["full_results"])==2
    assert len(combined)==4

    expected_gross=(1.01)*(1.02)*(0.99)*(1.03)
    expected_net=(1.009)*(1.019)*(0.989)*(1.029)

    assert combined["StrategyCumulativeValue"].iloc[-1]==pytest.approx(expected_gross)
    assert combined["NetStrategyCumulativeValue"].iloc[-1]==pytest.approx(expected_net)

    assert result["statistics"]["observations"]==4
    assert previous_positions==[(0,0),(1.0,-2.0)]


def test_run_pair_walk_forward_passes_window_sizes(monkeypatch):
    df=pd.DataFrame({
        "CloseA":range(20,50),
        "CloseB":range(10,40)
    })

    captured={}

    def fake_generate(length, initial_train_size, validation_size, test_size):
        captured["length"]=length
        captured["initial_train_size"]=initial_train_size
        captured["validation_size"]=validation_size
        captured["test_size"]=test_size

        return [
            {
                "train_end":10,
                "validation_end":15,
                "test_end":20
            }
        ]

    monkeypatch.setattr(
        pairs_walk_forward,
        "generate_walk_forward_windows",
        fake_generate
    )

    monkeypatch.setattr(
        pairs_walk_forward,
        "run_pair_walk_forward_window",
        lambda *args:{
            "test_results":pd.DataFrame({
                "StrategyReturn":[0],
                "NetStrategyReturn":[0],
                "PositionA":[0],
                "PositionB":[0]
            })
        }
    )

    monkeypatch.setattr(
        pairs_walk_forward,
        "calculate_pair_statistics",
        lambda df:{"observations":len(df)}
    )

    run_pair_walk_forward(
        df,
        [5],
        [2],
        [0.5],
        1,
        0.001,
        10,
        5,
        5
    )

    assert captured["length"]==len(df)
    assert captured["initial_train_size"]==10
    assert captured["validation_size"]==5
    assert captured["test_size"]==5
