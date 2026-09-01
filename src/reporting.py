from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from src.metrics import calculate_cumulative_value,calculate_drawdown,calculate_performance_metrics


def build_oos_strategy_streams(integration_results):
    standalone=integration_results["standalone_oos_results"].copy()
    standalone["RegressionAlpha"]=np.nan
    standalone["RegressionBeta"]=np.nan

    pair=integration_results["pair_oos_results"].copy()
    pair["Asset"]=pair["AssetA"]+"/"+pair["AssetB"]
    pair["Position"]=pair["SpreadPosition"]

    columns=[
        "Date","Asset","Strategy","GrossReturn","InternalCost","NetReturn","Turnover","Position","WindowID",
        "SelectedLookback","SelectedEntryThreshold","SelectedExitThreshold","RegressionAlpha","RegressionBeta","EvaluationType"
    ]

    return pd.concat([standalone[columns],pair[columns]],ignore_index=True).sort_values(["Date","Asset","Strategy"]).reset_index(drop=True)


def _summary_row(system,asset,evaluation_type,comparison_period,dates,gross_returns,net_returns,turnover,internal_cost,overlay_turnover,overlay_cost,active_fraction,primary):
    metrics=calculate_performance_metrics(net_returns)

    return {
        "System":system,
        "Asset":asset,
        "EvaluationType":evaluation_type,
        "ComparisonPeriod":comparison_period,
        "StartDate":pd.to_datetime(dates).iloc[0],
        "EndDate":pd.to_datetime(dates).iloc[-1],
        "Observations":metrics["observations"],
        "GrossTotalReturn":calculate_cumulative_value(gross_returns).iloc[-1]-1,
        "NetTotalReturn":metrics["total_return"],
        "AnnualisedReturn":metrics["annualised_return"],
        "AnnualisedVolatility":metrics["annualised_volatility"],
        "SharpeRatio":metrics["sharpe_ratio"],
        "MaxDrawdown":metrics["max_drawdown"],
        "UnderlyingTurnover":float(np.sum(turnover)),
        "OverlayTurnover":float(np.sum(overlay_turnover)),
        "TotalTurnover":float(np.sum(turnover)+np.sum(overlay_turnover)),
        "UnderlyingCost":float(np.sum(internal_cost)),
        "OverlayCost":float(np.sum(overlay_cost)),
        "TotalCost":float(np.sum(internal_cost)+np.sum(overlay_cost)),
        "ActiveFraction":active_fraction,
        "PrimaryComparison":primary
    }


def build_summary_metrics(integration_results):
    rows=[]
    dynamic=integration_results["dynamic_common_results"]
    portfolio_assets="/".join(integration_results["standalone_oos_results"]["Asset"].drop_duplicates())

    rows.append(_summary_row(
        "dynamic_portfolio",portfolio_assets,"walk_forward_test","common_oos",dynamic["Date"],dynamic["GrossDynamicPortfolioReturn"],
        dynamic["NetDynamicPortfolioReturn"],dynamic["DynamicUnderlyingTurnover"],dynamic["DynamicUnderlyingCost"],dynamic["DynamicTurnover"],
        dynamic["DynamicOverlayCost"],(dynamic["DynamicGrossExposure"]>0).mean(),True
    ))

    baseline_specs=[
        ("equal_weight","equal_weight_common_results"),
        ("inverse_volatility","inverse_volatility_common_results")
    ]

    for system,key in baseline_specs:
        results=integration_results["baselines"][key]
        rows.append(_summary_row(
            system,portfolio_assets,"walk_forward_test","common_oos",results["Date"],results["PortfolioReturn"],results["NetPortfolioReturn"],
            results["Turnover"],results["TransactionCost"],np.zeros(len(results)),np.zeros(len(results)),1.0,True
        ))

    standalone=integration_results["standalone_common_results"]

    for (asset,strategy),results in standalone.groupby(["Asset","Strategy"],sort=True):
        rows.append(_summary_row(
            strategy,asset,"walk_forward_test","common_oos",results["Date"],results["GrossReturn"],results["NetReturn"],results["Turnover"],
            results["InternalCost"],np.zeros(len(results)),np.zeros(len(results)),(results["Position"]!=0).mean(),True
        ))

    pair=integration_results["pair_oos_results"]
    pair_asset=f"{pair['AssetA'].iloc[0]}/{pair['AssetB'].iloc[0]}"
    rows.append(_summary_row(
        "pairs",pair_asset,"walk_forward_test","separate_pair_oos",pair["Date"],pair["GrossReturn"],pair["NetReturn"],pair["Turnover"],
        pair["InternalCost"],np.zeros(len(pair)),np.zeros(len(pair)),(pair["SpreadPosition"]!=0).mean(),False
    ))

    return pd.DataFrame(rows)


def _finish_figure(figure,path):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    figure.tight_layout()
    figure.savefig(path,dpi=160,bbox_inches="tight")
    plt.close(figure)

    return path


def plot_oos_performance(integration_results,path):
    dynamic=integration_results["dynamic_common_results"]
    baselines=integration_results["baselines"]
    series={
        "Dynamic":(dynamic["Date"],dynamic["NetDynamicPortfolioReturn"]),
        "Equal weight":(baselines["equal_weight_common_results"]["Date"],baselines["equal_weight_common_results"]["NetPortfolioReturn"]),
        "Inverse volatility":(baselines["inverse_volatility_common_results"]["Date"],baselines["inverse_volatility_common_results"]["NetPortfolioReturn"])
    }
    figure,axes=plt.subplots(2,1,figsize=(11,8),sharex=True)

    for label,(dates,returns) in series.items():
        axes[0].plot(dates,calculate_cumulative_value(returns),label=label,linewidth=1.6)
        axes[1].plot(dates,calculate_drawdown(returns),label=label,linewidth=1.3)

    axes[0].set_title("Common out-of-sample performance after costs")
    axes[0].set_ylabel("Value of 1.0")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.2)
    axes[1].set_title("Drawdown from the previous peak")
    axes[1].set_ylabel("Drawdown")
    axes[1].set_xlabel("Date")
    axes[1].grid(alpha=0.2)

    return _finish_figure(figure,path)


def plot_walk_forward_stability(window_table,path):
    figure,axis=plt.subplots(figsize=(12,7))

    for (asset,strategy),group in window_table.groupby(["Asset","Strategy"],sort=True):
        label=f"{asset} {strategy.replace('_',' ')}"
        axis.plot(group["WindowID"],group["NetTotalReturn"],marker="o",linewidth=1.2,label=label)

        for row in group.itertuples():
            axis.annotate(f"L{int(row.SelectedLookback)}",(row.WindowID,row.NetTotalReturn),xytext=(0,5),textcoords="offset points",fontsize=7,ha="center")

    axis.axhline(0,color="black",linewidth=0.8)
    axis.set_title("Walk-forward test return by window")
    axis.set_xlabel("Window")
    axis.set_ylabel("Net return")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False,ncol=2,fontsize=8)

    return _finish_figure(figure,path)


def plot_dynamic_allocation(dynamic_results,assets,path):
    dates=dynamic_results["Date"]
    asset_weights=np.vstack([dynamic_results[f"{asset}_Weight"].to_numpy() for asset in assets])
    momentum_weight=sum(dynamic_results[f"{asset}_MomentumWeight"] for asset in assets)
    mean_reversion_weight=sum(dynamic_results[f"{asset}_MeanReversionWeight"] for asset in assets)
    cash_weight=(1-dynamic_results["DynamicGrossExposure"]).clip(lower=0)
    figure,axes=plt.subplots(2,1,figsize=(12,8),sharex=True)

    axes[0].stackplot(dates,*asset_weights,labels=assets,alpha=0.85)
    axes[0].set_title("Dynamic allocation by asset")
    axes[0].set_ylabel("Weight")
    axes[0].set_ylim(0,1)
    axes[0].legend(frameon=False,ncol=len(assets))
    axes[1].stackplot(dates,momentum_weight,mean_reversion_weight,cash_weight,labels=["Momentum","Mean reversion","Cash"],alpha=0.85)
    axes[1].set_title("Dynamic allocation by strategy")
    axes[1].set_ylabel("Weight")
    axes[1].set_xlabel("Date")
    axes[1].set_ylim(0,1)
    axes[1].legend(frameon=False,ncol=3)

    return _finish_figure(figure,path)


def plot_robustness(bootstrap_results,randomisation_results,cost_sensitivity,path):
    primary=bootstrap_results.loc[(bootstrap_results["Statistic"]=="SharpeRatio")&(bootstrap_results["BlockVariant"]=="primary")].iloc[0]
    placebo=randomisation_results.iloc[0]
    figure,axes=plt.subplots(1,3,figsize=(14,4.5))

    if primary["Status"]=="ok":
        axes[0].vlines(0,primary["Lower"],primary["Upper"],linewidth=2)
        axes[0].scatter([0],[primary["Observed"]],zorder=3)
    else:
        axes[0].text(0.5,0.5,primary["Status"],ha="center",va="center",transform=axes[0].transAxes)

    axes[0].axhline(0,color="black",linewidth=0.8)
    axes[0].set_title("Bootstrap Sharpe interval")
    axes[0].set_xticks([])

    if placebo["Status"]=="ok":
        axes[1].axhspan(placebo["NullLower"],placebo["NullUpper"],alpha=0.2,label="95% timing null")
        axes[1].scatter([0],[placebo["Observed"]],label="Observed",zorder=3)
        axes[1].axhline(placebo["NullMedian"],linestyle="--",linewidth=1,label="Null median")
        axes[1].legend(frameon=False,fontsize=8)
    else:
        axes[1].text(0.5,0.5,placebo["Status"],ha="center",va="center",transform=axes[1].transAxes)

    axes[1].set_title("Block timing placebo")
    axes[1].set_xticks([])
    axes[2].plot(cost_sensitivity["CostBps"],cost_sensitivity["NetTotalReturn"],marker="o")
    axes[2].set_title("Frozen-policy cost sensitivity")
    axes[2].set_xlabel("Cost per turnover (bps)")
    axes[2].set_ylabel("Net total return")
    axes[2].grid(alpha=0.2)

    return _finish_figure(figure,path)


def _format_percent(value):
    return "n/a" if pd.isna(value) else f"{value:.2%}"


def _format_number(value):
    return "n/a" if pd.isna(value) else f"{value:.2f}"


def _metrics_markdown(summary_metrics):
    rows=["| System | Period | Net return | Annual return | Volatility | Sharpe | Max drawdown |","|---|---|---:|---:|---:|---:|---:|"]

    for row in summary_metrics.itertuples():
        rows.append(f"| {row.System} ({row.Asset}) | {row.ComparisonPeriod} | {_format_percent(row.NetTotalReturn)} | {_format_percent(row.AnnualisedReturn)} | {_format_percent(row.AnnualisedVolatility)} | {_format_number(row.SharpeRatio)} | {_format_percent(row.MaxDrawdown)} |")

    return "\n".join(rows)


def build_final_report(config,summary_metrics,bootstrap_results,randomisation_results,cost_sensitivity,window_table):
    dynamic=summary_metrics.loc[summary_metrics["System"]=="dynamic_portfolio"].iloc[0]
    comparators=summary_metrics.loc[summary_metrics["PrimaryComparison"]&(summary_metrics["System"]!="dynamic_portfolio")]
    placebo=randomisation_results.iloc[0]
    primary_bootstrap=bootstrap_results.loc[(bootstrap_results["Statistic"]=="SharpeRatio")&(bootstrap_results["BlockVariant"]=="primary")].iloc[0]
    descriptive_support=bool(pd.notna(dynamic["SharpeRatio"]) and dynamic["SharpeRatio"]>comparators["SharpeRatio"].max())
    conclusion="The frozen dynamic portfolio had the highest descriptive Sharpe among the common-period comparisons." if descriptive_support else "The frozen dynamic portfolio did not have the highest descriptive Sharpe among the common-period comparisons."
    placebo_text=f"The block timing placebo produced a one-sided p-value of {_format_number(placebo['RawPValue'])}." if placebo["Status"]=="ok" else f"The timing placebo was not interpretable ({placebo['Status']})."
    bootstrap_text=f"Its primary stationary-bootstrap Sharpe interval was [{_format_number(primary_bootstrap['Lower'])}, {_format_number(primary_bootstrap['Upper'])}]." if primary_bootstrap["Status"]=="ok" else f"The bootstrap Sharpe interval was not interpretable ({primary_bootstrap['Status']})."
    common_start=pd.Timestamp(dynamic["StartDate"]).date().isoformat()
    common_end=pd.Timestamp(dynamic["EndDate"]).date().isoformat()
    median_positive=window_table.groupby(["Asset","Strategy"])["FractionPositiveWindows"].first().median()

    return f"""# Final research report

## Research question

Can several simple trading hypotheses be selected chronologically, evaluated repeatedly out of sample, and combined into a lagged dynamic portfolio that improves net risk-adjusted performance over static and standalone alternatives?

This report is the output of one frozen configuration. The result was not re-optimised after changing transaction costs or after running the robustness checks.

## Result

The primary common comparison covers {common_start} to {common_end}. {conclusion} {bootstrap_text} {placebo_text}

That is evidence about this sample and this implementation, not proof of persistent alpha or a deployable trading system.

{_metrics_markdown(summary_metrics)}

## What was tested

- Momentum and mean-reversion parameters were chosen using chronological training and validation periods, then evaluated on the following test window.
- The KO/PEP pair was evaluated separately with regression coefficients frozen before each test period.
- The dynamic portfolio used only standardized walk-forward test streams. Its strategy score used a {config['dynamic']['score_lookback']}-day lookback and was shifted so the current return could not choose itself.
- Candidate trading costs and the extra dynamic allocation-overlay cost were recorded separately.
- Equal-weight and inverse-volatility baselines used the same common comparison dates. Inverse-volatility weights were estimated before OOS and then frozen.

Across asset/strategy groups, the median fraction of profitable walk-forward windows was {_format_percent(median_positive)}. Window-level results and parameter selection frequencies are saved beside this report so performance concentration is visible rather than hidden inside one final number.

## Robustness checks

The stationary bootstrap used {config['robustness']['bootstrap_resamples']} resamples, seed {config['project']['random_seed']}, and three expected block-length variants. Blocking preserves short-run dependence better than independently shuffling daily returns. The timing placebo moved aligned blocks of candidate returns relative to the already frozen allocation decisions; it asks whether the observed timing was stronger than could be expected from a broken timing relationship.

Cost sensitivity repriced the unchanged policy at {', '.join(f'{rate*10000:g}' for rate in config['robustness']['cost_rates'])} basis points per unit of gross-notional turnover. It did not select new parameters or new allocations at each cost.

## Important assumptions

- Prices are adjusted daily closes from Yahoo Finance through `yfinance`, aligned on a complete common calendar.
- Decisions are made at close t with idealized execution at that close and earn the close-t to close-(t+1) return.
- Annualisation uses {config['project']['annualisation_periods']} trading periods and a zero risk-free rate.
- The dynamic layer is a sleeve abstraction. Internal strategy costs and allocation-overlay costs may conservatively charge some switching twice because there is no asset-level netting engine.
- KO/PEP was predefined. Cointegration diagnostics are reported as pre-test evidence, not used to choose a pair after seeing OOS results.

## Limitations

This is daily-bar research with simple proportional costs. It omits bid/ask spreads, market impact, borrow fees, taxes, capacity, intraday execution, and live operational constraints. The same historical OOS sequence was used while developing earlier milestones, so it should not be described as a pristine institutional holdout. Bootstrap intervals describe sampling uncertainty conditional on this design; they do not correct for every research choice or establish causality.

## Reproduction

From the repository root:

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python run_research.py --config config/default.json
```

The run manifest records the exact configuration, data hashes, package versions, seed, Git state, dates, and generated artifacts.
"""


def save_report(report,path):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(report,encoding="utf-8")

    return path
