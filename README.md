# Quant Project

This project analyses historical closing-price data for a single asset and tests momentum and mean-reversion trading strategies.

It loads and validates price data, calculates returns, runs reusable backtests with transaction costs, performs chronological train-validation-test optimisation, and evaluates strategies using expanding-window walk-forward testing.

## Input Data

Place the input file at:

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

The reusable backtesting logic is located in `src/backtest.py`.

Each strategy first creates a `Signal` column. The backtesting engine then:

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

Keeping the backtesting logic separate allows different strategies to use the same engine without repeating the calculations.

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

## Data Splitting

The chronological data-splitting logic is located in `src/split.py`.

For the basic optimisation pipelines, the historical data is divided into:

- Training data for comparing all candidate parameters
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

It does not contain momentum-specific or mean-reversion-specific strategy logic.

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

## Project Structure

```text
quant-project/
├── data/
│   └── prices.csv
├── output/
│   ├── analysed_prices.csv
│   ├── momentum_results.csv
│   └── mean_reversion_results.csv
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
│   └── mean_reversion_walk_forward.py
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
│   └── test_mean_reversion_walk_forward.py
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
```

The complete test suite currently passes.

## Current Limitations

The project currently:

- Supports one asset at a time
- Uses closing prices only
- Implements long-or-cash strategies
- Implements momentum and mean-reversion strategies
- Uses fixed parameter grids supplied by the researcher
- Uses expanding-window walk-forward evaluation only
- Uses a fixed proportional transaction-cost model
- Does not model bid-ask spreads
- Does not model slippage
- Does not model market impact
- Does not support position sizing
- Does not support leverage
- Assumes 252 trading periods per year
- Assumes a zero risk-free rate when calculating the Sharpe ratio
- Does not yet perform statistical significance or robustness testing
- Does not yet support portfolio-level or multi-asset strategies

There is also a transaction-cost detail at walk-forward boundaries that can be improved.

If the selected strategy parameters change between consecutive walk-forward windows, the first transaction cost in the new window is currently based on the position generated by the newly selected strategy rather than explicitly using the final deployed position from the previous test window.

This can be refined later when the execution model is made more realistic.

