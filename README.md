# Quant Project

This project analyses historical closing-price data for a single asset and tests a momentum trading strategy.

It loads and validates price data, calculates returns, runs backtests with transaction costs, compares momentum lookback periods, and evaluates a selected strategy using chronological training, validation, and test periods.

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

## Momentum Strategy

The momentum strategy is located in `src/momentum.py`.

It compares the current closing price with the price from a chosen number of periods earlier.

When the lookback return is positive, the strategy holds the asset. Otherwise, it stays in cash.

Signals are delayed by one period so the strategy cannot use current information to earn the current period's return.

Run the strategy with:

```bash
python -m src.momentum
```

The results are saved to:

`output/momentum_results.csv`

## Backtesting Engine

The reusable backtesting logic is located in `src/backtest.py`.

The strategy first creates a `Signal` column. The backtesting engine then:

1. Delays signals to create positions
2. Calculates gross strategy returns
3. Measures position turnover
4. Applies transaction costs
5. Calculates net strategy returns
6. Calculates cumulative portfolio values
7. Produces performance and risk statistics

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

Annualised calculations assume 252 trading periods per year. The Sharpe ratio currently assumes a risk-free rate of zero.

Keeping the backtesting logic separate allows future strategies to use the same engine without repeating the calculations.

## Data Splitting

The chronological data-splitting logic is located in `src/split.py`.

It divides the historical data into three periods:

- Training data for comparing all candidate lookbacks
- Validation data for choosing between the strongest candidates
- Test data for one final evaluation

The data is never shuffled because future prices must not be used to make decisions about earlier periods.

The default split is:

```text
60% training
20% validation
20% testing
```

## Warm-Up History

Momentum signals require earlier prices.

Validation and test periods therefore use rows from before their starting positions as warm-up history. This allows the first signals, positions, and transaction costs in each period to be calculated correctly.

Warm-up rows are used only as historical context. Their returns are not included in the performance of the evaluated period.

Cumulative values are recalculated after the warm-up rows are removed, so each evaluated period measures only its own returns.

## Strategy Optimisation

The optimisation logic is located in `src/optimise.py`.

The optimisation process is:

```text
Split the data chronologically
→ evaluate all lookbacks on training data
→ keep the strongest candidates
→ evaluate those candidates on validation data
→ select one final lookback
→ evaluate it once on test data
```

`evaluate_momentum_period()` evaluates one momentum lookback over a specified period while preserving earlier rows as warm-up history.

`evaluate_lookbacks_on_period()` compares several lookbacks over the same period.

`select_top_lookbacks()` keeps the strongest candidates from the training results.

`select_best_lookback()` chooses the strongest candidate from the validation results.

`run_optimisation()` connects the complete training, validation, and test process.

The test set is used only after the final lookback has been selected.

## Project Structure

```text
quant-project/
├── data/
│   └── prices.csv
├── output/
│   ├── analysed_prices.csv
│   └── momentum_results.csv
├── src/
│   ├── analyse.py
│   ├── backtest.py
│   ├── momentum.py
│   ├── optimise.py
│   └── split.py
├── tests/
│   ├── test_analyse.py
│   ├── test_backtest.py
│   ├── test_momentum.py
│   ├── test_optimise.py
│   └── test_split.py
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
python -m pytest tests/test_momentum.py -v
python -m pytest tests/test_split.py -v
python -m pytest tests/test_optimise.py -v
```

## Current Limitations

The project currently:

- Supports one asset at a time
- Uses closing prices only
- Implements only a long-or-cash momentum strategy
- Uses fixed training, validation, and test periods rather than walk-forward evaluation
- Assumes 252 trading periods per year
- Assumes a zero risk-free rate when calculating the Sharpe ratio