import argparse

from src.config import PROJECT_ROOT,load_config,resolve_project_path
from src.data import load_multi_asset_prices,load_pair_prices
from src.integration import run_complete_oos_integration
from src.manifest import build_run_manifest,calculate_file_sha256,save_run_manifest
from src.reporting import build_final_report,build_oos_strategy_streams,build_summary_metrics,plot_dynamic_allocation,plot_oos_performance,plot_robustness,plot_walk_forward_stability,save_report
from src.robustness import run_dynamic_timing_placebo,stationary_bootstrap_intervals
from src.sensitivity import build_parameter_stability_table,build_walk_forward_window_table,calculate_frozen_cost_sensitivity


REQUIRED_ARTIFACTS=[
    "oos_strategy_streams.csv","dynamic_portfolio.csv","walk_forward_windows.csv","summary_metrics.csv","bootstrap_intervals.csv",
    "randomisation_tests.csv","cost_sensitivity.csv","parameter_stability.csv","final_report.md","figures/oos_performance.png",
    "figures/walk_forward_stability.png","figures/dynamic_allocation.png","figures/robustness.png"
]


def _save_csv(df,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    df.to_csv(path,index=False,float_format="%.10f")


def _describe_artifacts(output_dir):
    descriptions=[]

    for relative_path in REQUIRED_ARTIFACTS:
        path=output_dir/relative_path
        descriptions.append({"path":str(path.relative_to(PROJECT_ROOT)) if path.is_relative_to(PROJECT_ROOT) else str(path),"sha256":calculate_file_sha256(path),"bytes":path.stat().st_size})

    return descriptions


def run_research(config_path="config/default.json",output_dir="output"):
    config_path=resolve_project_path(config_path)
    output_dir=resolve_project_path(output_dir)
    config=load_config(config_path)
    assets=config["data"]["multi_asset_tickers"]
    pair_assets=config["data"]["pair_tickers"]
    price_panel=load_multi_asset_prices(resolve_project_path(config["data"]["multi_asset_path"]),asset_columns=assets)
    pair_panel=load_pair_prices(resolve_project_path(config["data"]["pair_path"]))
    integration=run_complete_oos_integration(price_panel,pair_panel,config)
    robustness=config["robustness"]
    minimum_observations=config["reporting"]["minimum_oos_observations"]
    seed=config["project"]["random_seed"]

    oos_streams=build_oos_strategy_streams(integration)
    window_table=build_walk_forward_window_table(integration["standalone_oos_results"],integration["pair_oos_results"],integration["pair_walk_forward_results"])
    parameter_stability=build_parameter_stability_table(integration["multi_asset_walk_forward_results"],integration["pair_walk_forward_results"],pair_assets)
    cost_sensitivity=calculate_frozen_cost_sensitivity(integration["dynamic_common_results"],robustness["cost_rates"])
    bootstrap_results=stationary_bootstrap_intervals(integration["dynamic_common_results"]["NetDynamicPortfolioReturn"],robustness["bootstrap_resamples"],robustness["confidence_level"],seed,minimum_observations)
    randomisation_results=run_dynamic_timing_placebo(integration["dynamic_full_results"],assets,config["dynamic"]["overlay_cost_rate"],robustness["bootstrap_resamples"],robustness["confidence_level"],seed,minimum_observations)
    summary_metrics=build_summary_metrics(integration)

    tables={
        "oos_strategy_streams.csv":oos_streams,
        "dynamic_portfolio.csv":integration["dynamic_full_results"],
        "walk_forward_windows.csv":window_table,
        "summary_metrics.csv":summary_metrics,
        "bootstrap_intervals.csv":bootstrap_results,
        "randomisation_tests.csv":randomisation_results,
        "cost_sensitivity.csv":cost_sensitivity,
        "parameter_stability.csv":parameter_stability
    }

    for filename,table in tables.items():
        _save_csv(table,output_dir/filename)

    plot_oos_performance(integration,output_dir/"figures"/"oos_performance.png")
    plot_walk_forward_stability(window_table,output_dir/"figures"/"walk_forward_stability.png")
    plot_dynamic_allocation(integration["dynamic_full_results"],assets,output_dir/"figures"/"dynamic_allocation.png")
    plot_robustness(bootstrap_results,randomisation_results,cost_sensitivity,output_dir/"figures"/"robustness.png")
    report=build_final_report(config,summary_metrics,bootstrap_results,randomisation_results,cost_sensitivity,window_table)
    save_report(report,output_dir/"final_report.md")

    dynamic_common=integration["dynamic_common_results"]
    pair_oos=integration["pair_oos_results"]
    run_metadata={
        "primary_hypothesis":config["project"]["primary_hypothesis"],
        "evaluation_type":"frozen_walk_forward_oos",
        "oos_start_date":integration["candidate_returns"]["Date"].iloc[0].date().isoformat(),
        "oos_end_date":integration["candidate_returns"]["Date"].iloc[-1].date().isoformat(),
        "common_start_date":dynamic_common["Date"].iloc[0].date().isoformat(),
        "common_end_date":dynamic_common["Date"].iloc[-1].date().isoformat(),
        "pair_oos_start_date":pair_oos["Date"].iloc[0].date().isoformat(),
        "pair_oos_end_date":pair_oos["Date"].iloc[-1].date().isoformat(),
        "artifacts":_describe_artifacts(output_dir)
    }
    manifest=build_run_manifest(config_path,run_metadata)
    save_run_manifest(manifest,output_dir/"run_manifest.json")

    return {"integration":integration,"summary_metrics":summary_metrics,"bootstrap_intervals":bootstrap_results,"randomisation_tests":randomisation_results,"cost_sensitivity":cost_sensitivity,"parameter_stability":parameter_stability,"walk_forward_windows":window_table,"manifest":manifest}


def main():
    parser=argparse.ArgumentParser(description="Run the frozen walk-forward research pipeline.")
    parser.add_argument("--config",default="config/default.json",help="Path to the frozen JSON configuration.")
    parser.add_argument("--output-dir",default="output",help="Directory for generated research artifacts.")
    args=parser.parse_args()
    results=run_research(args.config,args.output_dir)
    dynamic=results["summary_metrics"].loc[results["summary_metrics"]["System"]=="dynamic_portfolio"].iloc[0]

    print(f"Research run complete: {resolve_project_path(args.output_dir)}")
    print(f"Common OOS dynamic return: {dynamic['NetTotalReturn']:.2%}")
    print(f"Common OOS dynamic Sharpe: {dynamic['SharpeRatio']:.2f}")


if __name__=="__main__":
    main()
