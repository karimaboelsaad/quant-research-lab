# Quant Project

This project analyses historical closing-price data and tests momentum, mean-reversion, and pairs-trading strategies.

It loads and validates price data, calculates returns, runs backtests with transaction costs, performs chronological train-validation-test optimisation, evaluates strategies using expanding-window walk-forward testing, and combines strategy returns through static and dynamic multi-asset portfolio construction.

The momentum and mean-reversion strategies operate on a single asset. The pairs strategy studies the statistical relationship between two assets and trades deviations in their estimated spread.

## Input Data

### Single-Asset Data

Momentum, mean reversion, and the basic price analyser use:

`data/prices.csv`

It must contain at least three rows and two columns named `Date` and `Close`.

Example:

```csv
Date,Close
2024-01-01,100
2024-01-02,105
2024-01-03,110
```

Dates must be valid, closing prices must be positive numbers, and duplicate dates are not allowed.

The rows do not need to be ordered because the program sorts them automatically.

### Pair Data

Pairs trading uses:

`data/pair_prices.csv`

The file contains two aligned closing-price series:

```csv
Date,CloseA,CloseB
2024-01-01,101,50
2024-01-02,103,51
2024-01-03,106,52
```

`CloseA` is treated as the dependent asset and `CloseB` as the explanatory asset when estimating the pair regression.

Both prices must correspond to the same date so that the relationship between the assets can be calculated correctly.

### Portfolio Data

Static portfolio analysis uses:

`data/portfolio_prices.csv`

The file must contain a `Date` column and at least two asset price columns.

Example:

```csv
Date,A,B,C
2024-01-01,100,80,150
2024-01-02,102,79,151
2024-01-03,101,81,152
```

Dynamic portfolio allocation currently uses:

`data/dynamic_strategy_returns.csv`

which contains previously generated momentum and mean-reversion return streams for several assets.

Example columns include:

```text
Date
A_MomentumReturn
A_MeanReversionReturn
B_MomentumReturn
B_MeanReversionReturn
```

## Price Analyser

The basic price analyser is located in `src/analyse.py`.

It calculates:

- Daily returns
- Cumulative value
- Running maximum
- Drawdown
- Annualised volatility

Run it from the project root with:

```bash
python src/analyse.py
```

The processed data is saved to:

`output/analysed_prices.csv`

## Backtesting Engine

The reusable single-asset backtesting logic is located in `src/backtest.py`.

Each single-asset strategy first creates a `Signal` column. The backtesting engine then:

1. Delays signals to create positions
2. Calculates gross strategy returns
3. Measures position turnover
4. Applies transaction costs
5. Calculates net strategy returns
6. Calculates cumulative portfolio values
7. Produces performance and risk statistics

Signals are delayed by one period:

```text
Position[t] = Signal[t - 1]
```

This prevents a strategy from using information from the current period to earn the current period's return.

The reported statistics include:

- Buy-and-hold return
- Gross strategy return
- Net strategy return
- Annualised return
- Annualised volatility
- Sharpe ratio
- Maximum drawdown
- Total turnover
- Number of trade events
- Time in the market

Annualised calculations assume 252 trading periods per year.

The Sharpe ratio currently assumes a risk-free rate of zero.

Keeping the backtesting logic separate allows the single-asset strategies to use the same engine without repeating the calculations.

Pairs trading uses separate two-leg return calculations because one spread position represents simultaneous positions in two assets.

## Momentum Strategy

The momentum strategy is located in `src/momentum.py`.

It compares the current closing price with the price from a chosen number of periods earlier.

The lookback return is:

```text
LookbackReturn = CurrentPrice / PreviousPrice - 1
```

When the lookback return is positive, the strategy holds the asset. Otherwise, it stays in cash.

The complete momentum strategy is handled by:

`run_momentum()`

Run the strategy with:

```bash
python -m src.momentum
```

The results are saved to:

`output/momentum_results.csv`

## Momentum Warm-Up History

Momentum signals require earlier prices.

Validation and test periods therefore use rows from before their starting positions as warm-up history. This allows the first signals, positions, and transaction costs in each period to be calculated using the required historical information.

Warm-up rows are used only as historical context. Their returns are not included in the performance of the evaluated period.

Cumulative values are recalculated after the warm-up rows are removed, so each evaluated period measures only its own returns.

## Momentum Optimisation

The momentum optimisation logic is located in:

`src/momentum_optimise.py`

The optimisation process is:

```text
Evaluate all lookbacks on training data
→ keep the strongest candidates
→ evaluate those candidates on validation data
→ select one final lookback
→ evaluate it once on test data
```

`evaluate_momentum_period()` evaluates one momentum lookback over a specified period while preserving earlier rows as warm-up history.

`evaluate_lookbacks_on_period()` compares several lookbacks over the same period.

`select_top_lookbacks()` keeps the strongest candidates from the training results.

`select_best_lookback()` chooses the strongest candidate from the validation results.

`run_momentum_optimisation()` connects the complete training, validation, and test process.

The test set is used only after the final lookback has been selected.

## Mean-Reversion Strategy

The mean-reversion strategy is located in `src/mean_reversion.py`.

It measures how far the current closing price is from its recent rolling mean using a z-score.

The z-score is:

```text
ZScore = (Close - RollingMean) / RollingStd
```

The strategy uses two thresholds:

```text
ZScore <= EntryThreshold → enter long
ZScore >= ExitThreshold  → exit to cash
```

Unlike the momentum signal, the mean-reversion signal is stateful.

Once the strategy enters a position, it remains long until the exit condition is reached.

For example:

```text
EntryThreshold = -1.0
ExitThreshold = 0.0
```

could produce:

```text
ZScore:  -1.5  -0.8  -0.3   0.2
Signal:     1     1     1     0
```

The complete mean-reversion strategy is handled by:

`run_mean_reversion()`

Run the strategy with:

```bash
python -m src.mean_reversion
```

The results are saved to:

`output/mean_reversion_results.csv`

## Mean-Reversion Historical State

Mean reversion requires more than just a fixed number of warm-up rows because its signal is stateful.

For example, a position may have been entered before the beginning of a validation or test period and still be open when that period begins.

When a mean-reversion period is evaluated, the strategy is therefore run using all historical data available up to the end of the requested period.

Only the requested rows are then retained for performance evaluation.

This preserves the strategy state at chronological boundaries while ensuring that earlier returns are not included in the evaluated period's performance.

## Mean-Reversion Optimisation

The mean-reversion optimisation logic is located in:

`src/mean_reversion_optimise.py`

The strategy has three parameters:

```text
Lookback
EntryThreshold
ExitThreshold
```

The optimisation process is:

```text
Evaluate all valid parameter combinations on training data
→ keep the strongest candidate combinations
→ evaluate those exact candidates on validation data
→ select one final parameter combination
→ evaluate it once on test data
```

`evaluate_mean_reversion_period()` evaluates one complete parameter combination over a specified period.

`evaluate_mean_reversion_parameters_on_period()` evaluates the full parameter grid.

`select_top_mean_reversion_parameters()` keeps the strongest candidate combinations from the training results.

`evaluate_mean_reversion_candidates_on_period()` evaluates those exact combinations on the validation period.

`select_best_mean_reversion_parameters()` selects the strongest validation candidate.

`run_mean_reversion_optimisation()` connects the complete training, validation, and test process.

Candidate combinations remain together during validation.

For example:

```text
Lookback 20, Entry -1.5, Exit 0.0
Lookback 50, Entry -2.0, Exit 0.5
```

are treated as two individual candidates rather than creating new combinations from their separate parameter values.

The test set is used only after the final parameter combination has been selected.

## Pairs Trading Strategy

The pairs-trading strategy is located in:

`src/pairs.py`

Instead of trading one asset based only on its own price history, pairs trading studies the relationship between two assets.

The relationship is estimated using ordinary least-squares regression:

```text
CloseA = Alpha + Beta × CloseB + Error
```

`Alpha` is the regression intercept.

`Beta` is the hedge ratio describing how much `CloseA` tends to change relative to `CloseB`.

The estimated value of `CloseA` is:

```text
PredictedA = Alpha + Beta × CloseB
```

The pair spread is the regression residual:

```text
Spread = CloseA - PredictedA
```

or equivalently:

```text
Spread = CloseA - Alpha - Beta × CloseB
```

The trading strategy looks for unusually large deviations in this spread.

## Pair Spread Z-Score

The spread is standardised using a rolling z-score:

```text
ZScore = (Spread - RollingMean) / RollingStd
```

The pairs strategy uses symmetric entry conditions.

For example:

```text
EntryThreshold = 2.0
ExitThreshold = 0.5
```

A sufficiently negative spread produces a long-spread signal:

```text
ZScore <= -EntryThreshold → Signal = 1
```

A sufficiently positive spread produces a short-spread signal:

```text
ZScore >= EntryThreshold → Signal = -1
```

The strategy exits when the spread moves back toward its normal range.

For a positive hedge ratio:

```text
Signal = 1
→ long A
→ short Beta units of B

Signal = -1
→ short A
→ long Beta units of B

Signal = 0
→ flat
```

Like mean reversion, the pairs signal is stateful.

## Pair Regression

The pair regression is calculated by:

`calculate_pair_regression()`

It estimates:

```text
Alpha
Beta
```

using historical observations of `CloseA` and `CloseB`.

The regression coefficients are kept separate from `run_pair_strategy()`.

This is important for out-of-sample testing.

For example:

```text
Training data
→ estimate Alpha and Beta

Validation data
→ use the already-estimated Alpha and Beta
```

The validation period is not allowed to estimate the relationship that is supposedly being tested on that same future period.

The same principle applies to test periods.

## Pair Stationarity and Cointegration

Pairs trading relies on the idea that two individually wandering price series may still maintain a relatively stable long-run relationship.

Two related statistical diagnostics are included.

### Augmented Dickey-Fuller Test

`test_spread_stationarity()` applies an Augmented Dickey-Fuller test to the calculated spread.

The null hypothesis is that the spread contains a unit root and is non-stationary.

A sufficiently small p-value provides evidence against that null hypothesis.

The ADF test therefore asks:

```text
Is this particular spread stationary?
```

### Cointegration Test

`test_pair_cointegration()` applies an Engle-Granger cointegration test to `CloseA` and `CloseB`.

It asks whether the two price series have evidence of a stable long-run linear relationship despite potentially being non-stationary individually.

The null hypothesis is that the two price series are not cointegrated.

A sufficiently small p-value provides evidence against that null hypothesis.

The two diagnostics are related but serve slightly different purposes:

```text
ADF
→ tests the calculated spread directly

Cointegration
→ formally tests whether A and B have a stationary long-run relationship
```

The cointegration test is treated as the primary formal test of the pair relationship.

## Pair Backtesting

Pairs trading requires separate return calculations because a spread position contains two asset positions.

The one-period pair P&L is:

```text
PairPnL =
PositionA × ChangeInA
+
PositionB × ChangeInB
```

Gross exposure is:

```text
GrossExposure =
|PositionA| × PreviousCloseA
+
|PositionB| × PreviousCloseB
```

The strategy return is:

```text
StrategyReturn = PairPnL / GrossExposure
```

Signals are still delayed by one period:

```text
SpreadPosition[t] = Signal[t - 1]
```

so current information cannot earn the current period's return.

Transaction costs are currently based on changes in the spread position.

For example:

```text
0 → 1     turnover = 1
1 → 0     turnover = 1
1 → -1    turnover = 2
```

The complete strategy pipeline is handled by:

`run_pair_strategy()`

Run the standalone strategy with:

```bash
python -m src.pairs
```

The results are saved to:

`output/pair_results.csv`

The standalone run estimates the pair regression using the complete supplied dataset, so it should be treated as an exploratory in-sample backtest.

The optimisation and walk-forward pipelines provide the more meaningful out-of-sample evaluation.

## Pair Optimisation

The pair optimisation logic is located in:

`src/pairs_optimise.py`

The user-supplied strategy parameters are:

```text
Lookback
EntryThreshold
ExitThreshold
```

`Alpha` and `Beta` are not grid-search parameters.

They are estimated from historical price data using regression.

The optimisation process is:

```text
Estimate the pair relationship from available training data
→ evaluate all valid strategy parameter combinations on training data
→ keep the strongest candidate combinations
→ evaluate those candidates on validation data
→ select the strongest candidate
→ evaluate it on test data
```

`evaluate_pair_period()` evaluates one parameter combination while preserving the required historical context.

For validation periods, the regression is fitted using data before the validation period.

For test periods, the regression is fitted using all data available before the test period.

This prevents validation and test prices from being used to estimate their own regression coefficients.

`evaluate_pair_parameters_on_period()` evaluates the complete parameter grid.

`select_top_pair_parameters()` keeps the strongest training candidates.

`evaluate_pair_candidates_on_period()` evaluates those candidates on validation data.

`select_best_pair_parameters()` chooses the strongest validation candidate.

`run_pair_optimisation()` connects the complete training-validation-test process.

## Data Splitting

The chronological data-splitting logic is located in `src/split.py`.

For the basic optimisation pipelines, historical data is divided into:

- Training data for comparing candidate parameters
- Validation data for choosing between the strongest candidates
- Test data for one final evaluation

The data is never shuffled because future prices must not be used to make decisions about earlier periods.

The default split is:

```text
60% training
20% validation
20% testing
```

## Walk-Forward Window Generation

The generic walk-forward window logic is located in:

`src/walk_forward.py`

This file only generates chronological train-validation-test boundaries.

It does not contain strategy-specific logic.

The project uses an expanding training window.

Example:

```text
Window 1:
Training      0–399
Validation  400–499
Test        500–599

Window 2:
Training      0–499
Validation  500–599
Test        600–699

Window 3:
Training      0–599
Validation  600–699
Test        700–799
```

The training period grows as more historical data becomes available.

`generate_walk_forward_windows()` creates these chronological window boundaries.

## Momentum Walk-Forward Evaluation

The momentum walk-forward logic is located in:

`src/momentum_walk_forward.py`

A single train-validation-test split can produce results that depend heavily on one particular test period.

Walk-forward evaluation repeats the optimisation process across multiple points in time.

For each momentum window, the process is:

```text
Evaluate all lookbacks on training data
→ keep the strongest candidates
→ evaluate them on validation data
→ select the best lookback
→ use that lookback on the unseen test period
```

The selected lookback is allowed to change between windows.

For example:

```text
Test period 1 → lookback 5
Test period 2 → lookback 20
Test period 3 → lookback 10
```

`run_momentum_walk_forward_window()` performs training, validation, lookback selection, and testing for one window.

`run_momentum_walk_forward()` executes every window and combines all unseen test periods into one out-of-sample performance history.

The cumulative values are recalculated after the test periods are combined so capital continues between windows rather than resetting to 1 at the start of each test period.

Overall performance statistics are then calculated across the full combined out-of-sample period.

## Mean-Reversion Walk-Forward Evaluation

The mean-reversion walk-forward logic is located in:

`src/mean_reversion_walk_forward.py`

For each mean-reversion window, the process is:

```text
Evaluate all parameter combinations on training data
→ keep the strongest candidate combinations
→ evaluate those candidates on validation data
→ select the best parameter combination
→ use that combination on the unseen test period
```

The selected:

```text
Lookback
EntryThreshold
ExitThreshold
```

can change between windows.

`run_mean_reversion_walk_forward_window()` performs optimisation and testing for one window.

`run_mean_reversion_walk_forward()` executes every window and combines all unseen test periods into one out-of-sample performance history.

As with momentum, cumulative values are recalculated across the combined test periods so capital continues between windows.

## Pair Walk-Forward Evaluation

The pair walk-forward logic is located in:

`src/pairs_walk_forward.py`

For each pair window, the process is:

```text
Estimate the pair relationship from historical data
→ evaluate parameter combinations on training data
→ keep the strongest candidates
→ evaluate those candidates on validation data
→ select the best parameter combination
→ estimate the relationship using all available pre-test data
→ evaluate the strategy on the unseen test period
```

The selected:

```text
Lookback
EntryThreshold
ExitThreshold
```

can change between windows.

The regression coefficients can also change as additional historical information becomes available.

However, each validation or test period uses coefficients estimated without looking into that period's future prices.

`run_pair_walk_forward_window()` performs the complete optimisation and test process for one window.

`run_pair_walk_forward()` executes every window and combines the unseen test periods into a single out-of-sample performance history.

Cumulative strategy values are recalculated after the test periods are combined so capital continues across walk-forward windows.

## Multi-Asset Portfolio Construction

General multi-asset portfolio logic is located in:

`src/portfolio.py`

Returns are calculated independently for every asset.

The portfolio module can calculate:

- Asset returns
- Correlation matrices
- Covariance matrices
- Equal portfolio weights
- Inverse-volatility weights
- Portfolio returns
- Portfolio volatility
- Rebalancing turnover
- Transaction costs
- Portfolio performance statistics

Equal weighting assigns:

```text
Weight = 1 / NumberOfAssets
```

Inverse-volatility weighting assigns more capital to assets with lower historical volatility:

```text
Weight_i =
(1 / Volatility_i)
/
Sum(1 / Volatility_j)
```

This is a simple inverse-volatility allocation rather than full risk-parity optimisation.

Portfolio variance is calculated using:

```text
PortfolioVariance = wᵀ Σ w
```

where `w` is the vector of portfolio weights and `Σ` is the asset covariance matrix.

Portfolio volatility is:

```text
PortfolioVolatility = sqrt(PortfolioVariance)
```

Run the standalone portfolio example with:

```bash
python -m src.portfolio
```

The results are saved to:

`output/portfolio_results.csv`

## Portfolio Rebalancing and Costs

Fixed target weights drift when assets produce different returns.

The drifted weight of an asset is calculated as:

```text
DriftedWeight =
TargetWeight × (1 + AssetReturn)
/
(1 + PortfolioReturn)
```

Turnover measures the amount of trading required to restore the target weights.

Transaction costs are applied proportionally:

```text
TransactionCost = Turnover × CostRate
```

and:

```text
NetPortfolioReturn =
PortfolioReturn - TransactionCost
```

## Dynamic Multi-Strategy Portfolio

Dynamic strategy allocation is located in:

`src/dynamic_portfolio.py`

Instead of assigning one strategy permanently to each asset, the dynamic portfolio compares recent momentum and mean-reversion performance separately for every asset.

For each strategy, recent risk-adjusted performance is measured using a rolling annualised Sharpe-style score:

```text
Score =
RollingMeanReturn
/
RollingReturnStd
× sqrt(252)
```

Strategy returns are shifted by one period before calculating the score:

```text
PastReturns[t] = StrategyReturn[t - 1]
```

This ensures that the strategy selected for the current period cannot use that period's realised return.

## Dynamic Strategy Selection

Momentum and mean reversion are compared independently for every asset.

The strategy with the strongest positive score is selected.

For example:

```text
A Momentum Score        1.3
A Mean-Reversion Score  0.4
→ select A momentum

B Momentum Score       -0.2
B Mean-Reversion Score  0.8
→ select B mean reversion

C Momentum Score       -0.3
C Mean-Reversion Score -0.1
→ select neither
```

If both strategies have non-positive scores, the asset remains inactive.

Capital is equally divided across active assets.

For example:

```text
A active
B active
C inactive

A Weight = 0.5
B Weight = 0.5
C Weight = 0
```

If no asset has a positive selected strategy, the portfolio remains in cash.

The selected strategy return for each asset is:

```text
SelectedStrategyReturn =
MomentumSelected × MomentumReturn
+
MeanReversionSelected × MeanReversionReturn
```

The portfolio return is then the weighted sum of the selected strategy returns.

## Dynamic Portfolio Turnover

The dynamic portfolio tracks strategy-specific weights such as:

```text
A_MomentumWeight
A_MeanReversionWeight
```

This allows turnover to capture both asset rebalancing and changes between strategies.

Previous strategy weights are adjusted for their realised returns:

```text
DriftedWeight[t] =
PreviousWeight × (1 + PreviousStrategyReturn)
/
(1 + PreviousPortfolioReturn)
```

Turnover is then calculated from the difference between the drifted previous allocation and the new target allocation.

This means that switching from momentum to mean reversion requires exiting one strategy allocation and entering the other.

Dynamic transaction costs are deducted before net portfolio performance is calculated.

The reported statistics include:

- Gross portfolio return
- Net portfolio return
- Annualised return
- Annualised volatility
- Sharpe ratio
- Maximum drawdown
- Total turnover
- Total transaction cost
- Time in the market

Run the dynamic portfolio with:

```bash
python -m src.dynamic_portfolio
```

The results are saved to:

`output/dynamic_portfolio_results.csv`

The dynamic allocator currently expects momentum and mean-reversion return streams to have already been generated.

A later integration step will connect the existing strategy and walk-forward pipelines directly to the dynamic portfolio.

## Project Structure

```text
quant-project/
├── data/
│   ├── prices.csv
│   ├── pair_prices.csv
│   ├── portfolio_prices.csv
│   └── dynamic_strategy_returns.csv
├── output/
│   ├── analysed_prices.csv
│   ├── momentum_results.csv
│   ├── mean_reversion_results.csv
│   ├── pair_results.csv
│   ├── portfolio_results.csv
│   └── dynamic_portfolio_results.csv
├── src/
│   ├── analyse.py
│   ├── backtest.py
│   ├── split.py
│   ├── walk_forward.py
│   ├── momentum.py
│   ├── momentum_optimise.py
│   ├── momentum_walk_forward.py
│   ├── mean_reversion.py
│   ├── mean_reversion_optimise.py
│   ├── mean_reversion_walk_forward.py
│   ├── pairs.py
│   ├── pairs_optimise.py
│   ├── pairs_walk_forward.py
│   ├── portfolio.py
│   └── dynamic_portfolio.py
├── tests/
│   ├── test_analyse.py
│   ├── test_backtest.py
│   ├── test_split.py
│   ├── test_walk_forward.py
│   ├── test_momentum.py
│   ├── test_momentum_optimise.py
│   ├── test_momentum_walk_forward.py
│   ├── test_mean_reversion.py
│   ├── test_mean_reversion_optimise.py
│   ├── test_mean_reversion_walk_forward.py
│   ├── test_pairs.py
│   ├── test_pairs_optimise.py
│   ├── test_pairs_walk_forward.py
│   ├── test_portfolio.py
│   └── test_dynamic_portfolio.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Running the Tests

Run the complete test suite with:

```bash
python -m pytest -v
```

Run a specific test file with:

```bash
python -m pytest tests/test_analyse.py -v
python -m pytest tests/test_backtest.py -v
python -m pytest tests/test_split.py -v
python -m pytest tests/test_walk_forward.py -v

python -m pytest tests/test_momentum.py -v
python -m pytest tests/test_momentum_optimise.py -v
python -m pytest tests/test_momentum_walk_forward.py -v

python -m pytest tests/test_mean_reversion.py -v
python -m pytest tests/test_mean_reversion_optimise.py -v
python -m pytest tests/test_mean_reversion_walk_forward.py -v

python -m pytest tests/test_pairs.py -v
python -m pytest tests/test_pairs_optimise.py -v
python -m pytest tests/test_pairs_walk_forward.py -v

python -m pytest tests/test_portfolio.py -v
python -m pytest tests/test_dynamic_portfolio.py -v
```

The pairs strategy introduces `statsmodels` for the ADF and Engle-Granger cointegration tests.

## Current Limitations

The project currently:

- Uses closing prices only
- Uses fixed parameter grids supplied by the researcher
- Uses expanding-window walk-forward evaluation only
- Uses simplified proportional transaction costs
- Does not model bid-ask spreads
- Does not model slippage
- Does not model market impact
- Supports equal and inverse-volatility static portfolio weighting but not general portfolio optimisation
- Uses equal weighting across active assets in the dynamic portfolio
- Does not model leverage explicitly
- Assumes 252 trading periods per year
- Assumes a zero risk-free rate when calculating Sharpe ratios
- Does not yet perform bootstrap or permutation-based statistical robustness testing
- Uses OLS to estimate a single linear hedge ratio for pairs trading
- Does not dynamically update the pair hedge ratio inside an individual test period
- Uses a simplified pair transaction-cost model based on spread-position turnover
- Requires dynamic strategy-return streams to be generated separately before running the dynamic allocator

There is also a transaction-cost detail at walk-forward boundaries that can be improved.

If selected strategy parameters change between consecutive walk-forward windows, the first transaction cost in the new window is currently based on the position generated by the newly selected strategy rather than explicitly using the final deployed position from the previous test window.

For pairs trading, the same issue can also arise when the estimated hedge ratio changes between consecutive windows.

This can be refined later when the execution model is made more realistic.

## Current Progress

Completed components include:

```text
Price loading and analysis
→ reusable single-asset backtesting
→ momentum
→ mean reversion
→ train-validation-test optimisation
→ historical warm-up and state handling
→ expanding-window walk-forward evaluation
→ pairs trading
→ regression and hedge-ratio estimation
→ stationarity and cointegration testing
→ pair optimisation and walk-forward evaluation
→ multi-asset return analysis
→ correlation and covariance analysis
→ equal and inverse-volatility portfolio weighting
→ portfolio risk and rebalancing
→ dynamic per-asset strategy selection
→ dynamic multi-asset allocation
→ portfolio turnover and transaction costs
```

The remaining major stages are integrating the existing strategy pipelines directly into the dynamic portfolio and performing stronger statistical robustness testing.