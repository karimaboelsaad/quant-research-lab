import pandas as pd

from src.results import(
    OOS_RESULT_COLUMNS,
    PAIR_OOS_RESULT_COLUMNS,
    standardize_oos_results,
    standardize_pair_oos_results
)


def test_standardize_momentum_oos_results():
    df=pd.DataFrame({
        "Date":["2020-01-02","2020-01-03"],
        "StrategyReturn":[0.01,-0.02],
        "TransactionCost":[0.001,0.0],
        "NetStrategyReturn":[0.009,-0.02],
        "Turnover":[1.0,0.0],
        "Position":[1,1]
    })

    result=standardize_oos_results(
        df,
        asset="SPY",
        strategy="momentum",
        window_id=1,
        selected_lookback=60
    )

    assert result.columns.tolist()==OOS_RESULT_COLUMNS
    assert result["Asset"].tolist()==["SPY","SPY"]
    assert result["Strategy"].tolist()==["momentum","momentum"]
    assert result["GrossReturn"].tolist()==[0.01,-0.02]
    assert result["InternalCost"].tolist()==[0.001,0.0]
    assert result["NetReturn"].tolist()==[0.009,-0.02]
    assert result["WindowID"].tolist()==[1,1]
    assert result["SelectedLookback"].tolist()==[60,60]
    assert result["SelectedEntryThreshold"].isna().all()
    assert result["SelectedExitThreshold"].isna().all()
    assert result["EvaluationType"].tolist()==[
        "walk_forward_test",
        "walk_forward_test"
    ]


def test_standardize_pair_oos_results():
    df=pd.DataFrame({
        "Date":["2020-01-02","2020-01-03"],
        "StrategyReturn":[0.01,-0.02],
        "TransactionCost":[0.001,0.0],
        "NetStrategyReturn":[0.009,-0.02],
        "Turnover":[1.0,0.0],
        "SpreadPosition":[1.0,1.0],
        "PositionA":[1.0,1.0],
        "PositionB":[-1.2,-1.2],
        "GrossExposure":[100.0,102.0],
        "RegressionAlpha":[5.0,5.0],
        "RegressionBeta":[1.2,1.2]
    })

    result=standardize_pair_oos_results(df,"KO","PEP",1,20,1.5,0.5)

    assert result.columns.tolist()==PAIR_OOS_RESULT_COLUMNS
    assert result["AssetA"].tolist()==["KO","KO"]
    assert result["AssetB"].tolist()==["PEP","PEP"]
    assert result["Strategy"].tolist()==["pairs","pairs"]
    assert result["RegressionAlpha"].tolist()==[5.0,5.0]
    assert result["RegressionBeta"].tolist()==[1.2,1.2]
    assert result["SelectedLookback"].tolist()==[20,20]
    assert result["SelectedEntryThreshold"].tolist()==[1.5,1.5]
    assert result["SelectedExitThreshold"].tolist()==[0.5,0.5]
    assert result["EvaluationType"].tolist()==["walk_forward_test","walk_forward_test"]
