# Quant Project

This project analyses historical closing-price data for a single asset and tests a simple momentum trading strategy.

It loads prices from a CSV file, validates the data, calculates returns and risk statistics, runs a backtest with transaction costs, prints reports in the terminal, and saves the results as CSV files.

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

Dates must be valid, closing prices must be positive numbers, and duplicate dates are not allowed. The rows do not need to be ordered because the program sorts them automatically.

## Price Analyser

The basic analyser is located in `src/analyse.py`.

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

It measures how much the price has changed over a chosen lookback period. When the lookback return is positive, the strategy holds the asset. Otherwise, it stays in cash.

Signals are delayed by one period so the strategy cannot use current information to earn the current period's return.

Run the strategy with:

```bash
python -m src.momentum
```

The program reports the buy-and-hold return, gross strategy return, net strategy return, turnover, trade events, and time in the market.

The results are saved to:

`output/momentum_results.csv`

## Backtesting Engine

The reusable backtesting logic is located in `src/backtest.py`.

The momentum strategy creates a `Signal` column. The backtesting engine then:

1. Converts signals into delayed positions
2. Calculates gross strategy returns
3. Measures turnover
4. Applies transaction costs
5. Calculates net returns and cumulative value
6. Produces general backtest statistics

Keeping the backtesting logic separate means future strategies can use the same engine without repeating the calculations.

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
│   └── momentum.py
├── tests/
│   ├── test_analyse.py
│   ├── test_backtest.py
│   └── test_momentum.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Running the Tests

Run the complete test suite with:

```bash
python -m pytest -v
```

Run only the analyser tests with:

```bash
python -m pytest tests/test_analyse.py -v
```

Run only the backtesting tests with:

```bash
python -m pytest tests/test_backtest.py -v
```

Run only the momentum tests with:

```bash
python -m pytest tests/test_momentum.py -v
```

## Current Limitations

The project currently supports one asset at a time and uses closing prices only. Annualised volatility assumes 252 trading days per year.