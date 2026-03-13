"""
Alpaca Trading Integration: Market Data, PostgreSQL Storage, and Strategy Execution.

- Step 4: Retrieve market data from Alpaca (stocks or crypto).
- Step 5: Store bars in PostgreSQL (by asset and timeframe), UTC timestamps, clean data.
- Step 6: Run your Part 1 strategy (train/predict/signals) and submit orders via Alpaca.

Usage:
  Set env: ALPACA_API_KEY, ALPACA_SECRET_KEY, DATABASE_URL (e.g. postgresql://user:pass@host:5432/db)
  Optional: ALPACA_PAPER=true (default), PG_HOST, PG_PORT, PG_USER, PG_PASSWORD, PG_DATABASE

  # Fetch data and save to DB
  python alpaca_trading.py fetch --symbols AAPL MSFT GOOGL --days 30

  # Run strategy and place orders (paper by default)
  python alpaca_trading.py run --strategy momentum --symbols AAPL MSFT GOOGL --top-k 2
"""

from __future__ import annotations

import os
import argparse
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd
import numpy as np

# Alpaca data (historical bars)
from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed

# Alpaca trading (orders)
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, OrderType, TimeInForce

try:
    import psycopg2
    from psycopg2.extras import execute_values
except ImportError:
    psycopg2 = None  # type: ignore

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Local strategy and data prep
from strategy import MomentumStrategy, RankingStrategy, RankingModelStrategy
from data_multi_stock import DataCleaner

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_FEED = DataFeed.IEX


def _get_pg_connection():
    """Build PostgreSQL connection from DATABASE_URL or PG_* env vars."""
    url = os.getenv("DATABASE_URL")
    if url:
        return psycopg2.connect(url)
    return psycopg2.connect(
        host=os.getenv("PG_HOST", "localhost"),
        port=os.getenv("PG_PORT", "5432"),
        user=os.getenv("PG_USER", "postgres"),
        password=os.getenv("PG_PASSWORD", ""),
        dbname=os.getenv("PG_DATABASE", "trading"),
    )


def _pg_available() -> bool:
    """Return True if PostgreSQL is configured and reachable."""
    ok, _ = _pg_availability()
    return ok


def _pg_availability() -> tuple[bool, str]:
    """
    Return (True, "") if PostgreSQL is configured and reachable.
    Otherwise return (False, reason_string) so the user can see why DB isn't used.
    """
    if psycopg2 is None:
        return False, "psycopg2 not installed (pip install psycopg2-binary)"
    if not os.getenv("DATABASE_URL") and not os.getenv("PG_HOST") and not os.getenv("PG_USER"):
        return False, "DATABASE_URL or PG_* env vars not set (add DATABASE_URL to .env)"
    try:
        conn = _get_pg_connection()
        conn.close()
        return True, ""
    except Exception as e:
        return False, f"PostgreSQL connection failed: {e}"


# ---------------------------------------------------------------------------
# Step 5: PostgreSQL storage (organize by asset & timeframe, UTC)
# ---------------------------------------------------------------------------


class MarketDataStore:
    """
    Save and load market bars in PostgreSQL.
    Organizes by asset and timeframe; all timestamps stored and returned in UTC.
    """

    TABLE = "market_bars"

    def __init__(self, connection=None):
        if psycopg2 is None:
            raise ImportError("psycopg2 is required for PostgreSQL storage. Install with: pip install psycopg2-binary")
        self._connection = connection

    @property
    def conn(self):
        if self._connection is None or getattr(self._connection, "closed", True):
            self._connection = _get_pg_connection()
        return self._connection

    def create_tables(self):
        """Create market_bars table if it does not exist."""
        with self.conn.cursor() as cur:
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.TABLE} (
                    id BIGSERIAL PRIMARY KEY,
                    asset VARCHAR(32) NOT NULL,
                    timeframe VARCHAR(16) NOT NULL,
                    bar_time_utc TIMESTAMPTZ NOT NULL,
                    open NUMERIC(20, 8) NOT NULL,
                    high NUMERIC(20, 8) NOT NULL,
                    low NUMERIC(20, 8) NOT NULL,
                    close NUMERIC(20, 8) NOT NULL,
                    volume NUMERIC(24, 4) NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    UNIQUE (asset, timeframe, bar_time_utc)
                );
                CREATE INDEX IF NOT EXISTS idx_market_bars_asset_timeframe_time
                ON {self.TABLE} (asset, timeframe, bar_time_utc);
            """)
            self.conn.commit()

    def save_bars(
        self,
        df: pd.DataFrame,
        asset: str,
        timeframe: str,
        datetime_column: str = "Datetime",
        symbol_column: Optional[str] = None,
    ):
        """
        Save bars to PostgreSQL. All timestamps converted to UTC.

        If df has a symbol column (e.g. 'Ticker'), pass symbol_column and each row
        is stored with its symbol as asset. Otherwise every row is stored under `asset`.
        """
        df = df.copy()
        df[datetime_column] = pd.to_datetime(df[datetime_column], utc=True)
        if not all(c in df.columns for c in ["Open", "High", "Low", "Close", "Volume"]):
            raise ValueError("DataFrame must contain Open, High, Low, Close, Volume")

        rows = []
        for _, row in df.iterrows():
            ts = row[datetime_column]
            ts = pd.Timestamp(ts)
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            else:
                ts = ts.tz_convert("UTC")
            ts = ts.to_pydatetime()
            sym = str(row[symbol_column]).strip() if symbol_column and symbol_column in row.index else (asset or "UNKNOWN")
            o = float(row["Open"])
            h = float(row["High"])
            lo = float(row["Low"])
            c = float(row["Close"])
            v = float(row["Volume"])
            rows.append((sym, timeframe, ts, o, h, lo, c, v))

        if not rows:
            return
        with self.conn.cursor() as cur:
            execute_values(
                cur,
                f"""
                INSERT INTO {self.TABLE} (asset, timeframe, bar_time_utc, open, high, low, close, volume)
                VALUES %s
                ON CONFLICT (asset, timeframe, bar_time_utc) DO UPDATE SET
                    open = EXCLUDED.open, high = EXCLUDED.high, low = EXCLUDED.low,
                    close = EXCLUDED.close, volume = EXCLUDED.volume
                """,
                rows,
            )
            self.conn.commit()

    def load_bars(
        self,
        assets: list[str],
        timeframe: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Load bars from PostgreSQL for given assets and timeframe.
        Returns DataFrame with columns: Datetime, Ticker, Open, High, Low, Close, Volume.
        All datetimes are timezone-aware UTC.
        """
        if not assets:
            return pd.DataFrame(columns=["Datetime", "Ticker", "Open", "High", "Low", "Close", "Volume"])

        placeholders = ",".join(["%s"] * len(assets))
        conditions = ["asset IN (" + placeholders + ")", "timeframe = %s"]
        params: list = list(assets) + [timeframe]
        if start is not None:
            conditions.append("bar_time_utc >= %s")
            params.append(start if start.tzinfo else start.replace(tzinfo=timezone.utc))
        if end is not None:
            conditions.append("bar_time_utc <= %s")
            params.append(end if end.tzinfo else end.replace(tzinfo=timezone.utc))

        q = f"""
            SELECT asset, bar_time_utc, open, high, low, close, volume
            FROM {self.TABLE}
            WHERE {" AND ".join(conditions)}
            ORDER BY bar_time_utc, asset
        """
        with self.conn.cursor() as cur:
            cur.execute(q, params)
            rows = cur.fetchall()

        if not rows:
            return pd.DataFrame(columns=["Datetime", "Ticker", "Open", "High", "Low", "Close", "Volume"])

        df = pd.DataFrame(
            rows,
            columns=["Ticker", "Datetime", "Open", "High", "Low", "Close", "Volume"],
        )
        df["Datetime"] = pd.to_datetime(df["Datetime"], utc=True)
        return df.sort_values(["Datetime", "Ticker"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Step 4: Retrieve market data from Alpaca
# ---------------------------------------------------------------------------

def get_alpaca_clients(api_key: Optional[str] = None, secret_key: Optional[str] = None):
    api_key = api_key or os.getenv("ALPACA_API_KEY")
    secret_key = secret_key or os.getenv("ALPACA_SECRET_KEY")
    if not api_key or not secret_key:
        raise ValueError("Set ALPACA_API_KEY and ALPACA_SECRET_KEY")
    return (
        StockHistoricalDataClient(api_key=api_key, secret_key=secret_key),
        CryptoHistoricalDataClient(api_key=api_key, secret_key=secret_key),
    )


def fetch_stock_bars(
    symbols: list[str],
    start: datetime,
    end: datetime,
    timeframe: TimeFrame = TimeFrame.Day,
    feed: DataFeed = DEFAULT_FEED,
    stock_client: Optional[StockHistoricalDataClient] = None,
) -> pd.DataFrame:
    """Fetch stock bars from Alpaca. Returns DataFrame with Datetime, Ticker, Open, High, Low, Close, Volume."""
    if stock_client is None:
        stock_client, _ = get_alpaca_clients()
    # Normalize to UTC
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    end = min(end, datetime.now(timezone.utc))
    if timeframe == TimeFrame.Day:
        start = start.replace(hour=0, minute=0, second=0, microsecond=0)
        end = end.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)

    req = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=timeframe,
        start=start,
        end=end,
        feed=feed,
    )
    bars = stock_client.get_stock_bars(req)
    if bars.df.empty:
        return pd.DataFrame(columns=["Datetime", "Ticker", "Open", "High", "Low", "Close", "Volume"])
    df = bars.df.reset_index()
    df = df.rename(columns={
        "timestamp": "Datetime",
        "symbol": "Ticker",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    })
    df = df[["Datetime", "Ticker", "Open", "High", "Low", "Close", "Volume"]]
    df["Datetime"] = pd.to_datetime(df["Datetime"]).dt.tz_convert("UTC")
    return df.sort_values(["Datetime", "Ticker"]).reset_index(drop=True)


def fetch_crypto_bars(
    symbols: list[str],
    start: datetime,
    end: datetime,
    timeframe: TimeFrame = TimeFrame.Day,
    crypto_client: Optional[CryptoHistoricalDataClient] = None,
) -> pd.DataFrame:
    """Fetch crypto bars from Alpaca. Returns DataFrame with Datetime, Ticker, Open, High, Low, Close, Volume."""
    if crypto_client is None:
        _, crypto_client = get_alpaca_clients()
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    end = min(end, datetime.now(timezone.utc))
    req = CryptoBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=timeframe,
        start=start,
        end=end,
    )
    bars = crypto_client.get_crypto_bars(req)
    if bars.df.empty:
        return pd.DataFrame(columns=["Datetime", "Ticker", "Open", "High", "Low", "Close", "Volume"])
    df = bars.df.reset_index()
    df = df.rename(columns={
        "timestamp": "Datetime",
        "symbol": "Ticker",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    })
    df = df[["Datetime", "Ticker", "Open", "High", "Low", "Close", "Volume"]]
    df["Datetime"] = pd.to_datetime(df["Datetime"]).dt.tz_convert("UTC")
    return df.sort_values(["Datetime", "Ticker"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Step 6: Strategy + Alpaca orders
# ---------------------------------------------------------------------------

def prepare_features_for_strategy(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Build strategy-ready features (Return_1, Momentum_20, MA_ratio, RSI, Target, etc.) using data_multi_stock pipeline."""
    cleaner = DataCleaner()
    return cleaner.prepare_data(df_raw)


def run_strategy_and_submit_orders(
    strategy,
    df_with_features: pd.DataFrame,
    trading_client: TradingClient,
    notional_per_side: float = 1000.0,
    dry_run: bool = False,
):
    """
    Run strategy and REBALANCE positions to match target weights.
    """

    if df_with_features.empty:
        print("No data to run strategy.")
        return []

    # ---------------------------------------
    # Latest timestamp
    # ---------------------------------------
    latest_date = pd.Timestamp(df_with_features["Datetime"].max())
    df_latest = df_with_features[df_with_features["Datetime"] == latest_date].copy()

    # ---------------------------------------
    # Strategy pipeline
    # ---------------------------------------
    if hasattr(strategy, "normalize_features"):
        df_with_features = strategy.normalize_features(df_with_features)
        df_latest = df_with_features[df_with_features["Datetime"] == latest_date].copy()

    if hasattr(strategy, "train"):
        strategy.train(df_with_features)

    df_latest = strategy.predict(df_latest)
    df_latest = strategy.generate_signals(df_latest)
    df_latest = strategy.position_sizing(df_latest)

    # ---------------------------------------
    # Current positions
    # ---------------------------------------
    current_positions = {}

    if not dry_run:
        try:
            positions = trading_client.get_all_positions()

            for p in positions:

                symbol = str(p.symbol).upper()
                qty = float(p.qty)

                current_positions[symbol] = qty

            if current_positions:
                print("Current positions:", current_positions)

        except Exception as e:
            print("Failed to fetch positions:", e)

    # ---------------------------------------
    # Rebalance loop
    # ---------------------------------------
    orders_placed = []

    for _, row in df_latest.iterrows():

        ticker = str(row.get("Ticker", "")).upper()

        if not ticker:
            continue

        price = float(row.get("Close", 0))
        weight = float(row.get("weight", 0))

        if price <= 0:
            continue

        # Target shares
        target_qty = weight * (notional_per_side / price)

        target_qty = int(round(target_qty))

        current_qty = int(round(current_positions.get(ticker, 0)))

        order_qty = target_qty - current_qty

        if order_qty == 0:
            continue

        side = OrderSide.BUY if order_qty > 0 else OrderSide.SELL
        qty = abs(order_qty)

        if qty == 0:
            continue

        if dry_run:

            print(
                f"[DRY RUN] {side.value} {ticker} qty={qty} "
                f"(current={current_qty} target={target_qty})"
            )

            orders_placed.append(
                {
                    "symbol": ticker,
                    "side": side.value,
                    "qty": qty,
                    "dry_run": True,
                }
            )

            continue

        try:

            req = MarketOrderRequest(
                symbol=ticker,
                qty=qty,
                side=side,
                type=OrderType.MARKET,
                time_in_force=TimeInForce.DAY,
            )

            order = trading_client.submit_order(req)

            print(
                f"Submitted {side.value} {ticker} qty={qty} "
                f"(current={current_qty} target={target_qty})"
            )

            orders_placed.append(
                {
                    "symbol": ticker,
                    "side": side.value,
                    "qty": qty,
                    "order_id": str(order.id),
                }
            )

        except Exception as e:

            print(f"Order failed {ticker}: {e}")

    return orders_placed

def get_trading_client(paper: bool = True, api_key: Optional[str] = None, secret_key: Optional[str] = None) -> TradingClient:
    paper = os.getenv("ALPACA_PAPER", "true").lower() in ("1", "true", "yes") if paper is None else paper
    api_key = api_key or os.getenv("ALPACA_API_KEY")
    secret_key = secret_key or os.getenv("ALPACA_SECRET_KEY")
    if not api_key or not secret_key:
        raise ValueError("Set ALPACA_API_KEY and ALPACA_SECRET_KEY")
    return TradingClient(api_key, secret_key, paper=paper)


# ---------------------------------------------------------------------------
# CLI: fetch (Step 4 + 5) and run (Step 6)
# ---------------------------------------------------------------------------

def cmd_fetch(args):
    """Fetch market data from Alpaca and save to PostgreSQL or CSV (if DB unavailable)."""
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=args.days)
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    timeframe = getattr(TimeFrame, args.timeframe, TimeFrame.Day)
    asset_type = (args.asset_type or "stock").lower()

    if asset_type == "crypto":
        df = fetch_crypto_bars(symbols, start, end, timeframe=timeframe)
    else:
        df = fetch_stock_bars(symbols, start, end, timeframe=timeframe)

    if df.empty:
        print("No bars returned from Alpaca.")
        return

    tf_key = args.timeframe if args.timeframe else "Day"
    use_csv = getattr(args, "csv", False) or not _pg_available()

    if use_csv:
        if not getattr(args, "csv", False):
            _, reason = _pg_availability()
            print(f"PostgreSQL not used: {reason}")
        out_path = getattr(args, "output", None) or "market_data.csv"
        df.to_csv(out_path, index=False)
        print(f"Saved {len(df)} bars to {out_path}.")
    else:
        store = MarketDataStore()
        store.create_tables()
        store.save_bars(df, asset="", timeframe=tf_key, symbol_column="Ticker")
        print(f"Saved {len(df)} bars to PostgreSQL (timeframe={tf_key}).")


def cmd_run(args):
    """Load data from PostgreSQL or CSV, build features, run strategy, submit orders to Alpaca."""
    data_csv = getattr(args, "data_csv", None)
    if data_csv:
        df_raw = pd.read_csv(data_csv)
        df_raw["Datetime"] = pd.to_datetime(df_raw["Datetime"])
        if "Ticker" not in df_raw.columns and "Symbol" in df_raw.columns:
            df_raw = df_raw.rename(columns={"Symbol": "Ticker"})
        print(f"Loaded {len(df_raw)} bars from {data_csv}.")
    else:
        if not _pg_available():
            _, reason = _pg_availability()
            print(f"PostgreSQL not available: {reason}")
            print("Either fix the above or run with: python alpaca_trading.py run --data-csv market_data.csv ...")
            return
        store = MarketDataStore()
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
        timeframe = args.timeframe or "Day"
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=args.days)
        df_raw = store.load_bars(symbols, timeframe, start=start, end=end)
        if df_raw.empty:
            print("No bars in DB. Run 'fetch' first or use --data-csv path/to/market_data.csv")
            return
        print(f"Loaded {len(df_raw)} bars from PostgreSQL.")

    df_features = prepare_features_for_strategy(df_raw)
    if df_features.empty:
        print("No rows after feature preparation.")
        return

    strategy_name = (args.strategy or "momentum").lower()
    if strategy_name == "ranking":
        # RankingModelStrategy = LGBMRanker (same as backtest_ranking_strat.py)
        strategy = RankingModelStrategy(top_k=args.top_k)
    elif strategy_name == "ranking-regression":
        strategy = RankingStrategy(top_k=args.top_k)
    else:
        strategy = MomentumStrategy(top_k=args.top_k)

    paper = not args.live
    trading_client = get_trading_client(paper=paper)
    run_strategy_and_submit_orders(
        strategy,
        df_features,
        trading_client,
        notional_per_side=args.notional,
        dry_run=args.dry_run,
    )


def main():
    parser = argparse.ArgumentParser(description="Alpaca data + PostgreSQL + strategy execution")
    sub = parser.add_subparsers(dest="command", required=True)

    # fetch: pull from Alpaca, save to DB or CSV
    p_fetch = sub.add_parser("fetch", help="Fetch market data from Alpaca; save to PostgreSQL or CSV if DB unavailable")
    p_fetch.add_argument("--symbols", default="AAPL,MSFT,GOOGL", help="Comma-separated symbols")
    p_fetch.add_argument("--days", type=int, default=30, help="Days of history")
    p_fetch.add_argument("--timeframe", default="Day", help="Alpaca timeframe: Minute, Hour, Day")
    p_fetch.add_argument("--asset-type", default="stock", choices=("stock", "crypto"), help="Asset type")
    p_fetch.add_argument("--csv", action="store_true", help="Save to CSV instead of PostgreSQL")
    p_fetch.add_argument("--output", "-o", help="Output path when saving to CSV (default: market_data.csv)")
    p_fetch.set_defaults(func=cmd_fetch)

    # run: load from DB or CSV, run strategy, place orders
    p_run = sub.add_parser("run", help="Load data from DB or CSV, run strategy, submit orders to Alpaca")
    p_run.add_argument("--symbols", default="AAPL,MSFT,GOOGL", help="Comma-separated symbols (used when loading from DB)")
    p_run.add_argument("--data-csv", help="Load market data from this CSV instead of PostgreSQL (e.g. market_data.csv)")
    p_run.add_argument("--days", type=int, default=90, help="Days of history for features and training")
    p_run.add_argument("--timeframe", default="Day", help="Timeframe key in DB (e.g. Day)")
    p_run.add_argument("--strategy", default="ranking", choices=("momentum", "ranking", "ranking-regression"),
        help="Strategy: ranking (LGBMRanker, same as backtest_ranking_strat), momentum, or ranking-regression")
    p_run.add_argument("--top-k", type=int, default=2, help="Top/bottom K for long/short")
    p_run.add_argument("--notional", type=float, default=1000.0, help="Notional per position (approx)")
    p_run.add_argument("--live", action="store_true", help="Use live trading (default: paper)")
    p_run.add_argument("--dry-run", action="store_true", help="Print orders only, do not submit")
    p_run.add_argument("--update-positions", action="store_true",
        help="Submit orders for all signals, including symbols that already have a position (default: skip existing)")
    p_run.set_defaults(func=cmd_run)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
