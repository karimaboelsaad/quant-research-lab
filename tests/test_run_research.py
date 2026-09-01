import copy
import json

import numpy as np
import pandas as pd

from run_research import REQUIRED_ARTIFACTS,run_research
from src.config import load_config


def test_run_research_creates_complete_artifact_set(tmp_path):
    rng=np.random.default_rng(21)
    observations=90
    dates=pd.date_range("2020-01-01",periods=observations,freq="B")
    asset_a=100*np.exp(np.cumsum(rng.normal(0.0003,0.01,observations)))
    asset_b=80*np.exp(np.cumsum(rng.normal(0.0002,0.008,observations)))
    asset_c=120*np.exp(np.cumsum(rng.normal(0.0001,0.009,observations)))
    pair_b=50*np.exp(np.cumsum(rng.normal(0.0002,0.006,observations)))
    pair_a=5+1.2*pair_b+rng.normal(0,0.5,observations)

    single_path=tmp_path/"single.csv"
    multi_path=tmp_path/"multi.csv"
    pair_path=tmp_path/"pair.csv"
    pd.DataFrame({"Date":dates,"Close":asset_a}).to_csv(single_path,index=False)
    pd.DataFrame({"Date":dates,"A":asset_a,"B":asset_b,"C":asset_c}).to_csv(multi_path,index=False)
    pd.DataFrame({"Date":dates,"CloseA":pair_a,"CloseB":pair_b}).to_csv(pair_path,index=False)

    config=copy.deepcopy(load_config())
    config["data"].update({
        "multi_asset_tickers":["A","B","C"],
        "pair_tickers":["PAIR_A","PAIR_B"],
        "single_asset_ticker":"A",
        "single_asset_path":str(single_path),
        "multi_asset_path":str(multi_path),
        "pair_path":str(pair_path)
    })
    config["walk_forward"].update({"initial_train_size":40,"validation_size":15,"test_size":15})
    config["momentum"].update({"lookbacks":[5,10],"top_candidates":1})
    config["mean_reversion"].update({"lookbacks":[5,10],"entry_thresholds":[-1.0],"exit_thresholds":[0.0],"top_candidates":1})
    config["pairs"].update({"lookbacks":[5,10],"entry_thresholds":[1.0],"exit_thresholds":[0.0],"top_candidates":1})
    config["dynamic"]["score_lookback"]=5
    config["robustness"]["bootstrap_resamples"]=20
    config["reporting"]["minimum_oos_observations"]=10
    config_path=tmp_path/"config.json"
    config_path.write_text(json.dumps(config),encoding="utf-8")
    output_dir=tmp_path/"output"

    results=run_research(config_path,output_dir)

    assert all((output_dir/path).exists() for path in REQUIRED_ARTIFACTS)
    assert (output_dir/"run_manifest.json").exists()
    assert len(list((output_dir/"figures").glob("*.png")))==4
    assert {"System","ComparisonPeriod","NetTotalReturn","SharpeRatio","TotalTurnover","TotalCost"}.issubset(results["summary_metrics"].columns)
    assert {"Statistic","Observed","Lower","Upper","ExpectedBlockLength","Status"}.issubset(results["bootstrap_intervals"].columns)
    assert {"RawPValue","NullLower","NullUpper","Status"}.issubset(results["randomisation_tests"].columns)
    assert {"ValidationRank","SelectionFrequency","Selected"}.issubset(results["parameter_stability"].columns)
    assert set(results["summary_metrics"]["ComparisonPeriod"])=={"common_oos","separate_pair_oos"}
    assert results["manifest"]["run_metadata"]["evaluation_type"]=="frozen_walk_forward_oos"
