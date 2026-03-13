## TradingSystem

Cross-sectional quantitative trading sandbox with:
- **ML ranking strategy** (`strategy.py`, `backtest.py`)
- **Momentum, Moving-Average Crossover, and Bollinger Bands strategies**
  (with dedicated backtest scripts and comparison plots)
- **Market-data pipeline** for downloading and feature‑engineering Alpaca data (`data.py`)

### 1. Project structure (high level)

- **`data.py`**: `DataDownloader` (Alpaca stocks/crypto) and `DataCleaner` with feature engineering
- **`strategy.py`**:
  - `RankingStrategy` – regression‑based cross‑sectional stock ranking
  - `RankingModelStrategy` – LightGBM `LGBMRanker` (LambdaRank) ranking model
  - `MomentumStrategy` – indicator‑based momentum ranking
- **`backtest.py`**: Generic `Backtester` that wires `DataGateway`, `OrderManager`,
  `MatchingEngine`, and `OrderGateway`. Also includes plotting and comparison helpers.
- **`ma_crossover_strategy.py`** / **`backtest_ma_crossover.py`**:
  MA crossover strategy and its backtest/parameter studies.
- **`bollinger_bands_strategy.py`** / **`backtest_bollinger_bands.py`**:
  Bollinger Bands strategy and its backtest/parameter studies.
- **`backtest_momentum.py`**: Backtest runner for `MomentumStrategy`.
- **PNG reports**: Saved equity curves and parameter/signal comparisons.

### 2. Environment setup

- **Python**: 3.10+ recommended.
- **Install deps** (from project root):

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows PowerShell
pip install -r requirements.txt
```

- **Alpaca API keys** (required only if you use live data download in `data.py`):
  - See `ENV_SETUP.md` for step‑by‑step instructions (PowerShell, Windows GUI, `.env`).

### 3. Data

- Most examples assume a cleaned, multi‑stock CSV named **`multi_stock_dataset.csv`**
  with columns such as:
  - `Datetime`, `Ticker`, `Open`, `High`, `Low`, `Close`, `Volume`
  - Engineered features (e.g. `Return_1`, `Return_5`, `Momentum_20`, `MA_ratio`,
    `Volatility_10`, `RSI`, `CS_Return_Rank`, `CS_Vol_Rank`, `Target`)
- You can:
  - Use your own dataset with the same schema, or
  - Use `DataDownloader` + `DataCleaner` in `data.py` to create a similar file.

### 4. Running the main ML ranking backtest

From the project root (with your environment activated and `multi_stock_dataset.csv` present):

```bash
python backtest.py
```

This will:
- Train `RankingStrategy` on the first 70% of the time series
- Backtest on the remaining 30% using the full trading pipeline
- Print performance statistics
- Save an equity‑curve and trade‑distribution plot to `backtest_report.png`
- Run a small parameter sweep over `top_k` and plot comparisons

### 5. Running the technical‑indicator strategies

All of these also expect `multi_stock_dataset.csv`:

- **Momentum strategy**:

```bash
python backtest_momentum.py
```

Generates `momentum_backtest_report.png` and `momentum_parameter_comparison.png`.

- **Moving‑average crossover strategy**:

```bash
python backtest_ma_crossover.py
```

Generates `ma_crossover_backtest_report.png`, `ma_crossover_parameter_comparison.png`,
and `ma_crossover_mode_comparison.png`.

- **Bollinger Bands strategy**:

```bash
python backtest_bollinger_bands.py
```

Generates `bollinger_bands_backtest_report.png`, parameter and mode/signal comparison plots.

### 6. Using `strategy.py` directly

You can also run `strategy.py` to compare the regression‑based `RankingStrategy` vs.
the LightGBM `RankingModelStrategy` on `multi_stock_dataset.csv`:

```bash
python strategy.py
```

This prints total return and Sharpe for both variants and shows sample signals.

### 7. Notes and caveats

- This is a **research/backtesting** codebase, not production trading software.
- No transaction costs, slippage modeling, or borrow/short‑availability checks
are included unless you extend the matching/ordering components.
- Always validate with your own data and risk assumptions before deploying
anything derived from this project.