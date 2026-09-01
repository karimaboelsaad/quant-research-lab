# Data

## Final research inputs

The frozen experiment in `config/default.json` uses:

- multi-asset panel: SPY, TLT, and GLD;
- predefined pair: KO and PEP;
- single-asset example: SPY;
- fixed download interval: 1 January 2010 through 31 December 2025;
- daily prices automatically adjusted for corporate actions;
- an intersection calendar, retaining only dates available for every instrument in a panel.

Generate the files with:

```bash
python download_data.py
```

The command creates:

```text
data/research/prices.csv
data/research/portfolio_prices.csv
data/research/pair_prices.csv
```

These generated files are intentionally ignored by Git. `yfinance` is an independent open-source client for Yahoo Finance and states that downloaded Yahoo data is intended for personal use. The repository therefore stores the reproducible downloader and configuration rather than redistributing the downloaded observations.

The downloader explicitly passes `auto_adjust=True`. Consequently, the saved `Close` values are the adjusted close series returned by `yfinance`, rather than unadjusted exchange closes. The configured end date is exclusive, so `2026-01-01` includes observations only through 31 December 2025.

Yahoo may revise historical adjustments. Every final run must therefore record file hashes in `output/run_manifest.json`; the hash identifies the exact observations used even when the download instructions remain unchanged.

KO and PEP are predefined before the final OOS experiment. Cointegration diagnostics may be reported using training/pre-test observations, but OOS cointegration results must not be used to replace the pair after results are viewed.

## Legacy demonstration files

The CSV files directly under `data/` are small legacy fixtures used by the original standalone module entry points. They are not adequate for final empirical or robustness claims. The final research pipeline reads the paths specified in `config/default.json`, not these fixtures.

## Missing-data rule

The final research inputs require complete, positive, finite adjusted prices on a common calendar. A missing price or strategy return is an error. A numeric strategy return of zero means the strategy was in cash; missing and zero are never treated as equivalent.
