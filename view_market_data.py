"""
View market data stored in PostgreSQL (market_bars table).

Usage:
  python view_market_data.py
  python view_market_data.py --symbols AAPL,MSFT --limit 20
  python view_market_data.py --timeframe Day --last 7
"""

import os
import argparse

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import psycopg2
    import pandas as pd
except ImportError as e:
    print("Required: pip install psycopg2-binary pandas")
    raise SystemExit(1) from e


def get_connection():
    url = os.getenv("DATABASE_URL")
    if not url:
        raise SystemExit(
            "DATABASE_URL not set. Add it to .env or set the environment variable."
        )
    return psycopg2.connect(url)


def view_market_data(
    symbols: list[str] | None = None,
    timeframe: str | None = None,
    limit: int = 50,
    last_days: int | None = None,
):
    conn = get_connection()
    try:
        conditions = []
        params = []
        if symbols:
            placeholders = ",".join(["%s"] * len(symbols))
            conditions.append(f"asset IN ({placeholders})")
            params.extend(symbols)
        if timeframe:
            conditions.append("timeframe = %s")
            params.append(timeframe)
        if last_days is not None:
            conditions.append("bar_time_utc >= NOW() - make_interval(days => %s)")
            params.append(last_days)

        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        params.append(limit)
        query = f"""
            SELECT asset, timeframe, bar_time_utc, open, high, low, close, volume
            FROM market_bars
            {where}
            ORDER BY bar_time_utc DESC, asset
            LIMIT %s
        """
        df = pd.read_sql(query, conn, params=params)
        return df
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description="View market data from PostgreSQL")
    parser.add_argument(
        "--symbols", "-s",
        help="Comma-separated symbols (e.g. AAPL,MSFT). Default: all.",
    )
    parser.add_argument(
        "--timeframe", "-t",
        help="Timeframe filter (e.g. Day). Default: all.",
    )
    parser.add_argument(
        "--limit", "-n",
        type=int,
        default=50,
        help="Max rows to show (default: 50).",
    )
    parser.add_argument(
        "--last",
        type=int,
        metavar="DAYS",
        help="Only show bars from the last N days.",
    )
    parser.add_argument(
        "--no-index",
        action="store_true",
        help="Hide DataFrame index when printing.",
    )
    args = parser.parse_args()

    symbols = [x.strip() for x in args.symbols.split(",")] if args.symbols else None

    df = view_market_data(
        symbols=symbols,
        timeframe=args.timeframe,
        limit=args.limit,
        last_days=args.last,
    )

    if df.empty:
        print("No rows found.")
        return

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", 20)
    print(df.to_string(index=not args.no_index))
    print(f"\n({len(df)} rows)")


if __name__ == "__main__":
    main()
