import hashlib
import json
import platform
import subprocess
from datetime import datetime,timezone
from importlib.metadata import PackageNotFoundError,version
from pathlib import Path

import pandas as pd

from src.config import DEFAULT_CONFIG_PATH,PROJECT_ROOT,load_config,resolve_project_path
from src.walk_forward import calculate_unused_walk_forward_rows


def calculate_file_sha256(path):
    digest=hashlib.sha256()

    with Path(path).open("rb") as file:
        for chunk in iter(lambda:file.read(1024*1024),b""):
            digest.update(chunk)

    return digest.hexdigest()


def get_package_versions(package_names):
    versions={}

    for package_name in package_names:
        try:
            versions[package_name]=version(package_name)
        except PackageNotFoundError:
            versions[package_name]=None

    return versions


def get_git_state():
    try:
        commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=PROJECT_ROOT,text=True).strip()
        status=subprocess.check_output(["git","status","--porcelain"],cwd=PROJECT_ROOT,text=True)
    except (FileNotFoundError,subprocess.CalledProcessError):
        return {"commit":None,"dirty":None}

    return {"commit":commit,"dirty":bool(status.strip())}


def describe_data_file(path):
    path=resolve_project_path(path)

    if not path.exists():
        raise ValueError(f"Data file does not exist: {path}")

    df=pd.read_csv(path)

    if df.empty or "Date" not in df.columns:
        raise ValueError(f"Data file must contain dated observations: {path}")

    dates=pd.to_datetime(df["Date"],errors="raise")

    try:
        relative_path=str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        relative_path=str(path)

    return {
        "path":relative_path,
        "sha256":calculate_file_sha256(path),
        "bytes":path.stat().st_size,
        "rows":len(df),
        "start_date":dates.min().date().isoformat(),
        "end_date":dates.max().date().isoformat()
    }


def build_run_manifest(config_path=DEFAULT_CONFIG_PATH,run_metadata=None):
    config_path=resolve_project_path(config_path)
    config=load_config(config_path)
    data_config=config["data"]
    data_files={
        "single_asset":describe_data_file(data_config["single_asset_path"]),
        "multi_asset":describe_data_file(data_config["multi_asset_path"]),
        "pair":describe_data_file(data_config["pair_path"])
    }
    window_config=config["walk_forward"]

    for description in data_files.values():
        description["unused_trailing_rows"]=calculate_unused_walk_forward_rows(description["rows"],window_config["initial_train_size"],window_config["validation_size"],window_config["test_size"])

    manifest={
        "generated_at_utc":datetime.now(timezone.utc).isoformat(),
        "config_path":str(config_path.relative_to(PROJECT_ROOT)) if config_path.is_relative_to(PROJECT_ROOT) else str(config_path),
        "config_sha256":calculate_file_sha256(config_path),
        "config":config,
        "data_files":data_files,
        "python_version":platform.python_version(),
        "package_versions":get_package_versions(["pandas","numpy","statsmodels","yfinance","matplotlib","pytest"]),
        "git":get_git_state(),
        "run_metadata":run_metadata or {}
    }

    return manifest


def save_run_manifest(manifest,path="output/run_manifest.json"):
    path=resolve_project_path(path)
    path.parent.mkdir(parents=True,exist_ok=True)

    with path.open("w",encoding="utf-8") as file:
        json.dump(manifest,file,indent=2,sort_keys=True)

    return path


def main():
    path=save_run_manifest(build_run_manifest())
    print(f"Saved run manifest: {path}")


if __name__=="__main__":
    main()
