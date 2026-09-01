import copy
import json

import pandas as pd

from src.config import load_config
from src.manifest import build_run_manifest,calculate_file_sha256,save_run_manifest


def test_calculate_file_sha256_is_deterministic(tmp_path):
    path=tmp_path/"example.txt"
    path.write_text("quant",encoding="utf-8")

    assert calculate_file_sha256(path)==calculate_file_sha256(path)
    assert len(calculate_file_sha256(path))==64


def test_build_and_save_run_manifest(tmp_path):
    config=copy.deepcopy(load_config())

    for filename,columns in [
        ("prices.csv",{"Close":[100.0,101.0,102.0,103.0,104.0,105.0,106.0,107.0]}),
        ("portfolio.csv",{"SPY":[100.0,101.0,102.0,103.0,104.0,105.0,106.0,107.0],"TLT":[80.0,81.0,82.0,83.0,84.0,85.0,86.0,87.0],"GLD":[150.0,151.0,152.0,153.0,154.0,155.0,156.0,157.0]}),
        ("pair.csv",{"CloseA":[50.0,51.0,52.0,53.0,54.0,55.0,56.0,57.0],"CloseB":[60.0,61.0,62.0,63.0,64.0,65.0,66.0,67.0]})
    ]:
        pd.DataFrame({"Date":pd.date_range("2024-01-01",periods=8),**columns}).to_csv(tmp_path/filename,index=False)

    config["data"]["single_asset_path"]=str(tmp_path/"prices.csv")
    config["data"]["multi_asset_path"]=str(tmp_path/"portfolio.csv")
    config["data"]["pair_path"]=str(tmp_path/"pair.csv")
    config["walk_forward"]["initial_train_size"]=3
    config["walk_forward"]["validation_size"]=2
    config["walk_forward"]["test_size"]=2
    config["momentum"]["lookbacks"]=[2]
    config["momentum"]["top_candidates"]=1
    config["mean_reversion"]["lookbacks"]=[2]
    config["pairs"]["lookbacks"]=[2]

    config_path=tmp_path/"config.json"
    config_path.write_text(json.dumps(config),encoding="utf-8")

    manifest=build_run_manifest(config_path,run_metadata={"evaluation_type":"test"})
    output_path=save_run_manifest(manifest,tmp_path/"manifest.json")

    assert manifest["data_files"]["multi_asset"]["rows"]==8
    assert manifest["run_metadata"]["evaluation_type"]=="test"
    assert manifest["config_sha256"]==calculate_file_sha256(config_path)
    assert output_path.exists()
