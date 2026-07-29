# Quant Project

This project analyses historical price data for a single asset.

It reads a CSV file containing dates and closing prices, checks that the data is valid, calculates returns and risk statistics, prints reports in the terminal, and saves the results to CSV files.

The basic analyser is in `src/analyse.py`, and its tests are in `tests/test_analyse.py`.

The input file should be placed at:

`data/prices.csv`

It must contain two columns called `Date` and `Close`, with at least three rows of data.

Example:

```csv
Date,Close
2024-01-01,100
2024-01-02,105
2024-01-03,110
```

Dates must be valid, prices must be positive numbers, and duplicate dates are not allowed. The rows do not need to be sorted because the program sorts them automatically.

To run the basic analyser, use:

```bash
python src/analyse.py
```

The program prints a summary report and saves the processed data to:

`output/analysed_prices.csv`

The output includes daily returns, cumulative value, running maximum, and drawdown.

## Momentum Strategy

The project also includes a simple long-only momentum strategy in `src/momentum.py`.

The strategy:

- Calculates the return over a chosen lookback period
- Holds the asset when the lookback return is positive
- Stays in cash when the lookback return is zero or negative
- Delays signals by one period to avoid look-ahead bias
- Calculates turnover and transaction costs
- Compares buy-and-hold, gross strategy returns, and net strategy returns

To run the momentum strategy, use:

```bash
python -m src.momentum
```

The momentum results are saved to:

`output/momentum_results.csv`

The momentum tests are located in:

`tests/test_momentum.py`

## Running the Tests

To run all tests, use:

```bash
python -m pytest -v
```

To run only the momentum tests, use:

```bash
python -m pytest tests/test_momentum.py -v
```

The project currently only supports one asset at a time, uses closing prices only, and assumes 252 trading days per year when calculating annualised volatility.