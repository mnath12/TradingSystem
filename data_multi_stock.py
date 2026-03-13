"""
Data Download and Preparation Module
Supports MULTI-STOCK datasets for cross-sectional ML strategies.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed
from typing import Optional, List


DEFAULT_STOCK_FEED = DataFeed.IEX

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import requests


from io import StringIO


@staticmethod
def get_sp500_tickers():
    """
    Fetch S&P 500 tickers from Wikipedia reliably.
    """

    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    response = requests.get(url, headers=headers)

    html = StringIO(response.text)

    tables = pd.read_html(html)

    # First table contains the S&P500 list
    sp500_table = tables[0]

    tickers = sp500_table["Symbol"].tolist()

    # Alpaca uses '-' instead of '.'
    tickers = [t.replace(".", "-") for t in tickers]

    return tickers

class DataDownloader:
    """Downloads market data from Alpaca."""

    def __init__(self, api_key: Optional[str] = None, secret_key: Optional[str] = None):

        self.api_key = api_key or os.getenv("ALPACA_API_KEY")
        self.secret_key = secret_key or os.getenv("ALPACA_SECRET_KEY")

        if not self.api_key or not self.secret_key:
            raise ValueError("Missing Alpaca credentials.")

        self.stock_client = StockHistoricalDataClient(
            api_key=self.api_key,
            secret_key=self.secret_key
        )

    @staticmethod
    def _normalize_request_times(start_date, end_date, timeframe):

        now = datetime.now(timezone.utc)

        if start_date.tzinfo is None:
            start_date = start_date.replace(tzinfo=timezone.utc)

        if end_date.tzinfo is None:
            end_date = end_date.replace(tzinfo=timezone.utc)

        if end_date > now:
            end_date = now

        if timeframe == TimeFrame.Day:
            start_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
            end_date = end_date.replace(hour=0, minute=0, second=0, microsecond=0)
            end_date += timedelta(days=1)

        return start_date, end_date

    def filter_valid_symbols(self, symbols):
        """
        Remove symbols that Alpaca does not support.
        """

        valid = []
        invalid = []

        for s in symbols:
            try:
                request = StockBarsRequest(
                    symbol_or_symbols=[s],
                    timeframe=TimeFrame.Day,
                    start=datetime(2024,1,1, tzinfo=timezone.utc),
                    end=datetime(2024,1,10, tzinfo=timezone.utc),
                    feed=DEFAULT_STOCK_FEED
                )

                self.stock_client.get_stock_bars(request)

                valid.append(s)

            except Exception:
                invalid.append(s)

        print(f"Valid symbols: {len(valid)}")
        print(f"Removed invalid symbols: {invalid}")

        return valid

    def download_multiple_stocks(
        self,
        symbols,
        start_date,
        end_date,
        timeframe=TimeFrame.Day,
        feed=DEFAULT_STOCK_FEED,
        batch_size=100
    ):
        def chunk_list(lst, size):
            for i in range(0, len(lst), size):
                yield lst[i:i + size]

        start_date, end_date = self._normalize_request_times(
            start_date, end_date, timeframe
        )

        all_data = []

        for batch in chunk_list(symbols, batch_size):

            print(f"Downloading batch of {len(batch)} symbols")

            request = StockBarsRequest(
                symbol_or_symbols=batch,
                timeframe=timeframe,
                start=start_date,
                end=end_date,
                feed=feed
            )

            bars = self.stock_client.get_stock_bars(request)

            df = bars.df.reset_index()

            if df.empty:
                continue

            df = df.rename(columns={
                "timestamp": "Datetime",
                "symbol": "Ticker",
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume"
            })

            df = df[["Datetime", "Ticker", "Open", "High", "Low", "Close", "Volume"]]

            all_data.append(df)

        full_df = pd.concat(all_data)

        full_df = full_df.sort_values(["Datetime", "Ticker"]).reset_index(drop=True)

        return full_df

class DataCleaner:
    """Feature engineering for multi-asset datasets."""

    def clean_data(self, df: pd.DataFrame):

        df = df.copy()

        df["Datetime"] = pd.to_datetime(df["Datetime"])

        df = df.dropna()
        df = df.drop_duplicates()

        df = df.sort_values(["Ticker", "Datetime"])

        return df

    def add_returns(self, df: pd.DataFrame):

        df = df.copy()

        df["Return_1"] = df.groupby("Ticker")["Close"].pct_change()

        df["Return_5"] = df.groupby("Ticker")["Close"].pct_change(5)

        df["Return_10"] = df.groupby("Ticker")["Close"].pct_change(10)

        return df

    def add_moving_averages(self, df: pd.DataFrame):

        df = df.copy()

        df["SMA_5"] = (
            df.groupby("Ticker")["Close"]
            .transform(lambda x: x.rolling(5).mean())
        )

        df["SMA_20"] = (
            df.groupby("Ticker")["Close"]
            .transform(lambda x: x.rolling(20).mean())
        )

        df["MA_ratio"] = df["SMA_5"] / df["SMA_20"]

        return df

    def add_volatility(self, df: pd.DataFrame):

        df = df.copy()

        df["Volatility_10"] = (
            df.groupby("Ticker")["Return_1"]
            .transform(lambda x: x.rolling(10).std())
        )

        return df

    def add_rsi(self, df: pd.DataFrame, period: int = 14):

        df = df.copy()

        def rsi(series):

            delta = series.diff()

            gain = delta.clip(lower=0).rolling(period).mean()
            loss = -delta.clip(upper=0).rolling(period).mean()

            rs = gain / loss

            return 100 - (100 / (1 + rs))

        df["RSI"] = df.groupby("Ticker")["Close"].transform(rsi)

        return df

    def add_target(self, df: pd.DataFrame):

        df = df.copy()

        df["Target"] = (
            df.groupby("Ticker")["Close"]
            .pct_change()
            .shift(-1)
        )

        return df

    def add_cross_sectional_features(self, df: pd.DataFrame):

        df = df.copy()

        df["CS_Return_Rank"] = (
            df.groupby("Datetime")["Return_5"]
            .rank()
        )

        df["CS_Vol_Rank"] = (
            df.groupby("Datetime")["Volatility_10"]
            .rank()
        )

        return df

    def add_volume_features(self, df: pd.DataFrame):
        df = df.copy()
        df["Volume_MA_20"] = (
            df.groupby("Ticker")["Volume"]
            .transform(lambda x: x.rolling(20).mean())
        )

        df["Volume_Ratio"] = df["Volume"] / df["Volume_MA_20"]
        return df

    def add_momentum_features(self, df: pd.DataFrame):
        df = df.copy()
        df["Momentum_20"] = (
            df.groupby("Ticker")["Close"]
            .pct_change(20)
        )
        return df

    def cross_sectional_normalization(self, df: pd.DataFrame):
        df = df.copy()
        feature_cols = [
            "Return_1",
            "Return_5",
            "Return_10",
            "Momentum_20",
            "MA_ratio",
            "Volatility_10",
            "RSI",
            "Volume_Ratio"
        ]

        df[feature_cols] = df.groupby("Datetime")[feature_cols].transform(
            lambda x: (x - x.mean()) / x.std()
        )
        return df

    def prepare_data(self, df: pd.DataFrame):

        df = self.clean_data(df)

        df = self.add_returns(df)

        df = self.add_moving_averages(df)

        df = self.add_volatility(df)

        df = self.add_rsi(df)

        df = self.add_target(df)

        df = self.add_cross_sectional_features(df)

        df = self.add_volume_features(df)

        df = self.add_momentum_features(df)

        df = self.cross_sectional_normalization(df)

        df = df.dropna()

        return df


def main():

    downloader = DataDownloader()

    print("Fetching S&P 500 ticker list...")

    symbols = get_sp500_tickers()

    print("Filtering unsupported symbols...")

    symbols = downloader.filter_valid_symbols(symbols)

    print(f"Total symbols: {len(symbols)}")

    end_date = datetime.now(timezone.utc)

    start_date = datetime(2015,1,1, tzinfo=timezone.utc)

    print("Downloading historical data...")

    df_raw = downloader.download_multiple_stocks(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        timeframe=TimeFrame.Day
    )

    print("Rows downloaded:", len(df_raw))

    df_raw["DollarVolume"] = df_raw["Close"] * df_raw["Volume"]

    df_raw = df_raw[df_raw["DollarVolume"] > 10_000_000]

    print("Rows after dollar volume filter:", len(df_raw))

    cleaner = DataCleaner()

    print("Generating features...")

    df_clean = cleaner.prepare_data(df_raw)

    df_clean.to_csv("sp500_dataset.csv", index=False)

    print("Dataset saved: sp500_dataset.csv")

if __name__ == "__main__":
    main()