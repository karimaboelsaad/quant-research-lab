# Final research report

## Research question

Can several simple trading hypotheses be selected chronologically, evaluated repeatedly out of sample, and combined into a lagged dynamic portfolio that improves net risk-adjusted performance over static and standalone alternatives?

This report is the output of one frozen configuration. The result was not re-optimised after changing transaction costs or after running the robustness checks.

## Result

The primary common comparison covers 2015-03-23 to 2025-12-31. The frozen dynamic portfolio did not have the highest descriptive Sharpe among the common-period comparisons. Its primary stationary-bootstrap Sharpe interval was [-1.05, 0.20]. The block timing placebo produced a one-sided p-value of 0.90.

That is evidence about this sample and this implementation, not proof of persistent alpha or a deployable trading system.

| System | Period | Net return | Annual return | Volatility | Sharpe | Max drawdown |
|---|---|---:|---:|---:|---:|---:|
| dynamic_portfolio (SPY/TLT/GLD) | common_oos | -33.26% | -3.69% | 8.24% | -0.41 | -45.34% |
| equal_weight (SPY/TLT/GLD) | common_oos | 146.44% | 8.74% | 9.48% | 0.93 | -22.86% |
| inverse_volatility (SPY/TLT/GLD) | common_oos | 139.15% | 8.44% | 9.45% | 0.90 | -23.51% |
| mean_reversion (GLD) | common_oos | 21.13% | 1.80% | 6.57% | 0.30 | -13.73% |
| momentum (GLD) | common_oos | 143.96% | 8.64% | 12.36% | 0.73 | -21.58% |
| mean_reversion (SPY) | common_oos | 35.21% | 2.84% | 13.96% | 0.27 | -30.65% |
| momentum (SPY) | common_oos | 73.67% | 5.26% | 12.72% | 0.47 | -31.58% |
| mean_reversion (TLT) | common_oos | -18.21% | -1.85% | 9.57% | -0.15 | -34.66% |
| momentum (TLT) | common_oos | -21.71% | -2.25% | 10.69% | -0.16 | -35.63% |
| pairs (KO/PEP) | separate_pair_oos | -14.31% | -1.39% | 5.84% | -0.21 | -26.20% |

## What was tested

- Momentum and mean-reversion parameters were chosen using chronological training and validation periods, then evaluated on the following test window.
- The KO/PEP pair was evaluated separately with regression coefficients frozen before each test period.
- The dynamic portfolio used only standardized walk-forward test streams. Its strategy score used a 60-day lookback and was shifted so the current return could not choose itself.
- Candidate trading costs and the extra dynamic allocation-overlay cost were recorded separately.
- Equal-weight and inverse-volatility baselines used the same common comparison dates. Inverse-volatility weights were estimated before OOS and then frozen.

Across asset/strategy groups, the median fraction of profitable walk-forward windows was 50.00%. Window-level results and parameter selection frequencies are saved beside this report so performance concentration is visible rather than hidden inside one final number.

## Robustness checks

The stationary bootstrap used 5000 resamples, seed 20260820, and three expected block-length variants. Blocking preserves short-run dependence better than independently shuffling daily returns. The timing placebo moved aligned blocks of candidate returns relative to the already frozen allocation decisions; it asks whether the observed timing was stronger than could be expected from a broken timing relationship.

Cost sensitivity repriced the unchanged policy at 0, 5, 10, 20 basis points per unit of gross-notional turnover. It did not select new parameters or new allocations at each cost.

## Important assumptions

- Prices are adjusted daily closes from Yahoo Finance through `yfinance`, aligned on a complete common calendar.
- Decisions are made at close t with idealized execution at that close and earn the close-t to close-(t+1) return.
- Annualisation uses 252 trading periods and a zero risk-free rate.
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
