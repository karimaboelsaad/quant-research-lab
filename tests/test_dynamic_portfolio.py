import numpy as np
import pandas as pd
import pytest

from src.dynamic_portfolio import(
    calculate_rolling_strategy_scores,
    select_best_strategy_per_asset,
    calculate_selected_strategy_returns,
    calculate_dynamic_strategy_weights,
    calculate_dynamic_portfolio_returns,
    calculate_dynamic_turnover,
    apply_dynamic_transaction_costs,
    run_dynamic_portfolio,
    calculate_dynamic_portfolio_statistics,
    load_dynamic_strategy_returns,
    save_dynamic_portfolio_output
)


def create_strategy_dataframe():
    return pd.DataFrame({
        "Date":pd.date_range("2024-01-01",periods=5),
        "A_MomentumReturn":[0.01,0.02,0.015,0.01,0.005],
        "A_MeanReversionReturn":[-0.01,-0.005,-0.01,-0.005,-0.01],
        "B_MomentumReturn":[-0.01,-0.005,-0.01,-0.005,-0.01],
        "B_MeanReversionReturn":[0.01,0.015,0.02,0.01,0.015]
    })


def test_calculate_rolling_strategy_scores():
    df=pd.DataFrame({
        "A_MomentumReturn":[0.01,0.03,0.02,-0.01],
        "A_MeanReversionReturn":[0.02,0,0.01,0.02]
    })

    result=calculate_rolling_strategy_scores(df,["A"],2)

    expected_mean=(0.01+0.03)/2
    expected_std=np.std([0.01,0.03],ddof=1)
    expected_score=(expected_mean/expected_std)*np.sqrt(252)

    assert pd.isna(result["A_MomentumScore"].iloc[0])
    assert pd.isna(result["A_MomentumScore"].iloc[1])

    assert result["A_MomentumPerformanceMean"].iloc[2]==pytest.approx(expected_mean)
    assert result["A_MomentumPerformanceStd"].iloc[2]==pytest.approx(expected_std)
    assert result["A_MomentumScore"].iloc[2]==pytest.approx(expected_score)


def test_calculate_rolling_strategy_scores_does_not_use_current_return():
    df=pd.DataFrame({
        "A_MomentumReturn":[0.01,0.03,100],
        "A_MeanReversionReturn":[0.01,0.02,100]
    })

    result=calculate_rolling_strategy_scores(df,["A"],2)

    assert result["A_MomentumPerformanceMean"].iloc[2]==pytest.approx(0.02)


def test_calculate_rolling_strategy_scores_rejects_invalid_lookback():
    df=create_strategy_dataframe()

    with pytest.raises(ValueError):
        calculate_rolling_strategy_scores(df,["A","B"],1)


def test_select_best_strategy_per_asset():
    df=pd.DataFrame({
        "A_MomentumScore":[1,-1,1,np.nan,1],
        "A_MeanReversionScore":[0.5,-0.5,1,2,np.nan]
    })

    result=select_best_strategy_per_asset(df,["A"])

    assert result["A_MomentumSelected"].tolist()==[1,0,1,0,1]
    assert result["A_MeanReversionSelected"].tolist()==[0,0,0,1,0]


def test_select_best_strategy_selects_neither_when_both_negative():
    df=pd.DataFrame({
        "A_MomentumScore":[-0.2],
        "A_MeanReversionScore":[-0.5]
    })

    result=select_best_strategy_per_asset(df,["A"])

    assert result["A_MomentumSelected"].iloc[0]==0
    assert result["A_MeanReversionSelected"].iloc[0]==0


def test_calculate_selected_strategy_returns():
    df=pd.DataFrame({
        "A_MomentumReturn":[0.02,0.03,0.04],
        "A_MeanReversionReturn":[-0.01,0.01,0.02],
        "A_MomentumSelected":[1,0,0],
        "A_MeanReversionSelected":[0,1,0]
    })

    result=calculate_selected_strategy_returns(df,["A"])

    assert result["A_SelectedStrategyReturn"].iloc[0]==pytest.approx(0.02)
    assert result["A_SelectedStrategyReturn"].iloc[1]==pytest.approx(0.01)
    assert result["A_SelectedStrategyReturn"].iloc[2]==pytest.approx(0)


def test_calculate_dynamic_strategy_weights():
    df=pd.DataFrame({
        "A_MomentumSelected":[1,0],
        "A_MeanReversionSelected":[0,0],
        "B_MomentumSelected":[0,0],
        "B_MeanReversionSelected":[1,0],
        "C_MomentumSelected":[0,0],
        "C_MeanReversionSelected":[0,0]
    })

    result=calculate_dynamic_strategy_weights(df,["A","B","C"])

    assert result["A_Weight"].iloc[0]==pytest.approx(0.5)
    assert result["B_Weight"].iloc[0]==pytest.approx(0.5)
    assert result["C_Weight"].iloc[0]==pytest.approx(0)

    assert result["A_MomentumWeight"].iloc[0]==pytest.approx(0.5)
    assert result["B_MeanReversionWeight"].iloc[0]==pytest.approx(0.5)

    assert result["DynamicGrossExposure"].iloc[0]==pytest.approx(1)

    assert result["A_Weight"].iloc[1]==pytest.approx(0)
    assert result["B_Weight"].iloc[1]==pytest.approx(0)
    assert result["C_Weight"].iloc[1]==pytest.approx(0)
    assert result["DynamicGrossExposure"].iloc[1]==pytest.approx(0)


def test_calculate_dynamic_portfolio_returns():
    df=pd.DataFrame({
        "A_SelectedStrategyReturn":[0.02,-0.01],
        "A_Weight":[0.5,0.5],
        "B_SelectedStrategyReturn":[-0.01,0.03],
        "B_Weight":[0.5,0.5]
    })

    result=calculate_dynamic_portfolio_returns(df,["A","B"])

    assert result["DynamicPortfolioReturn"].iloc[0]==pytest.approx(0.005)
    assert result["DynamicPortfolioReturn"].iloc[1]==pytest.approx(0.01)

    expected=(1.005)*(1.01)

    assert result["DynamicPortfolioCumulativeValue"].iloc[-1]==pytest.approx(expected)


def test_calculate_dynamic_turnover_initial_position():
    df=pd.DataFrame({
        "DynamicPortfolioReturn":[0],
        "A_MomentumWeight":[1],
        "A_MeanReversionWeight":[0],
        "A_MomentumReturn":[0],
        "A_MeanReversionReturn":[0]
    })

    result=calculate_dynamic_turnover(df,["A"])

    assert result["DynamicTurnover"].iloc[0]==pytest.approx(1)


def test_calculate_dynamic_turnover_strategy_switch():
    df=pd.DataFrame({
        "DynamicPortfolioReturn":[0,0],
        "A_MomentumWeight":[1,0],
        "A_MeanReversionWeight":[0,1],
        "A_MomentumReturn":[0,0],
        "A_MeanReversionReturn":[0,0]
    })

    result=calculate_dynamic_turnover(df,["A"])

    assert result["DynamicTurnover"].iloc[0]==pytest.approx(1)
    assert result["DynamicTurnover"].iloc[1]==pytest.approx(2)


def test_calculate_dynamic_turnover_accounts_for_weight_drift():
    df=pd.DataFrame({
        "DynamicPortfolioReturn":[0.10,0],
        "A_MomentumWeight":[1,1],
        "A_MeanReversionWeight":[0,0],
        "A_MomentumReturn":[0.10,0],
        "A_MeanReversionReturn":[0,0]
    })

    result=calculate_dynamic_turnover(df,["A"])

    assert result["DynamicTurnover"].iloc[0]==pytest.approx(1)
    assert result["DynamicTurnover"].iloc[1]==pytest.approx(0)


def test_apply_dynamic_transaction_costs():
    df=pd.DataFrame({
        "DynamicPortfolioReturn":[0.01,0.02],
        "DynamicTurnover":[1,0.5]
    })

    result=apply_dynamic_transaction_costs(df,0.001)

    assert result["DynamicTransactionCost"].iloc[0]==pytest.approx(0.001)
    assert result["DynamicTransactionCost"].iloc[1]==pytest.approx(0.0005)

    assert result["NetDynamicPortfolioReturn"].iloc[0]==pytest.approx(0.009)
    assert result["NetDynamicPortfolioReturn"].iloc[1]==pytest.approx(0.0195)


def test_apply_dynamic_transaction_costs_rejects_negative_cost():
    df=pd.DataFrame({
        "DynamicPortfolioReturn":[0.01],
        "DynamicTurnover":[1]
    })

    with pytest.raises(ValueError):
        apply_dynamic_transaction_costs(df,-0.001)


def test_run_dynamic_portfolio():
    df=create_strategy_dataframe()

    result=run_dynamic_portfolio(
        df,
        ["A","B"],
        2,
        0.001
    )

    required_columns=[
        "A_MomentumScore",
        "A_MomentumSelected",
        "A_SelectedStrategyReturn",
        "A_Weight",
        "B_MeanReversionScore",
        "B_MeanReversionSelected",
        "B_SelectedStrategyReturn",
        "B_Weight",
        "DynamicPortfolioReturn",
        "DynamicTurnover",
        "DynamicTransactionCost",
        "NetDynamicPortfolioReturn"
    ]

    for column in required_columns:
        assert column in result.columns

    assert result["A_MomentumSelected"].iloc[2]==1
    assert result["B_MeanReversionSelected"].iloc[2]==1

    assert result["A_Weight"].iloc[2]==pytest.approx(0.5)
    assert result["B_Weight"].iloc[2]==pytest.approx(0.5)


def test_calculate_dynamic_portfolio_statistics():
    df=create_strategy_dataframe()

    df=run_dynamic_portfolio(
        df,
        ["A","B"],
        2,
        0.001
    )

    stats=calculate_dynamic_portfolio_statistics(
        df,
        2,
        0.001
    )

    assert stats["lookback"]==2
    assert stats["cost_rate"]==pytest.approx(0.001)
    assert stats["observations"]==3
    assert stats["time_in_market"]==pytest.approx(1)
    assert stats["total_turnover"]>=0
    assert stats["total_transaction_cost"]>=0


def test_load_dynamic_strategy_returns(tmp_path):
    filepath=tmp_path/"dynamic_strategy_returns.csv"

    pd.DataFrame({
        "Date":["2024-01-02","2024-01-01"],
        "A_MomentumReturn":[0.02,0.01],
        "A_MeanReversionReturn":[0.01,-0.01],
        "B_MomentumReturn":[0.03,0.02],
        "B_MeanReversionReturn":[0.01,0.02]
    }).to_csv(filepath,index=False)

    result=load_dynamic_strategy_returns(filepath,["A","B"])

    assert pd.api.types.is_datetime64_any_dtype(result["Date"])
    assert result["Date"].iloc[0]==pd.Timestamp("2024-01-01")
    assert result["Date"].iloc[1]==pd.Timestamp("2024-01-02")


def test_load_dynamic_strategy_returns_rejects_duplicate_dates(tmp_path):
    filepath=tmp_path/"dynamic_strategy_returns.csv"

    pd.DataFrame({
        "Date":["2024-01-01","2024-01-01"],
        "A_MomentumReturn":[0.01,0.02],
        "A_MeanReversionReturn":[0.01,0.02]
    }).to_csv(filepath,index=False)

    with pytest.raises(ValueError):
        load_dynamic_strategy_returns(filepath,["A"])


def test_save_dynamic_portfolio_output(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)

    df=create_strategy_dataframe()

    df=run_dynamic_portfolio(
        df,
        ["A","B"],
        2,
        0.001
    )

    save_dynamic_portfolio_output(df,["A","B"])

    filepath=tmp_path/"output"/"dynamic_portfolio_results.csv"

    assert filepath.exists()

    result=pd.read_csv(filepath)

    assert len(result)==len(df)
    assert "DynamicPortfolioReturn" in result.columns
    assert "NetDynamicPortfolioReturn" in result.columns
    assert "A_MomentumScore" in result.columns
    assert "B_MeanReversionScore" in result.columns