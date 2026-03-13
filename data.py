"""
Data Download and Preparation Module
Downloads intraday market data using Alpaca API and saves to CSV.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed
from typing import Optional

# Default feed: IEX is free; SIP requires a paid subscription
DEFAULT_STOCK_FEED = DataFeed.IEX

# Try to load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, skip .env loading


class DataDownloader:
    """Downloads intraday market data from Alpaca API."""
    
    def __init__(self, api_key: Optional[str] = None, secret_key: Optional[str] = None):
        """
        Initialize the data downloader with Alpaca API credentials.
        
        Args:
            api_key: Alpaca API key (defaults to ALPACA_API_KEY env var)
            secret_key: Alpaca secret key (defaults to ALPACA_SECRET_KEY env var)
        """
        self.api_key = api_key or os.getenv('ALPACA_API_KEY')
        self.secret_key = secret_key or os.getenv('ALPACA_SECRET_KEY')
        
        if not self.api_key or not self.secret_key:
            raise ValueError(
                "Alpaca API credentials required. Set ALPACA_API_KEY and "
                "ALPACA_SECRET_KEY environment variables or pass them as arguments."
            )
        
        # Initialize clients
        self.stock_client = StockHistoricalDataClient(
            api_key=self.api_key,
            secret_key=self.secret_key
        )
        self.crypto_client = CryptoHistoricalDataClient(
            api_key=self.api_key,
            secret_key=self.secret_key
        )

    @staticmethod
    def _normalize_request_times(
        start_date: datetime,
        end_date: datetime,
        timeframe: TimeFrame,
    ) -> tuple[datetime, datetime]:
        """
        Normalize start/end to UTC for Alpaca API. For daily bars, use date boundaries
        so the range is unambiguous. Ensures end is not in the future (Alpaca returns
        empty when end is ahead of available data).
        """
        now_utc = datetime.now(timezone.utc)

        # Ensure we have timezone-aware datetimes (Alpaca treats naive as UTC)
        if start_date.tzinfo is None:
            start_date = start_date.replace(tzinfo=timezone.utc)
        else:
            start_date = start_date.astimezone(timezone.utc)
        if end_date.tzinfo is None:
            end_date = end_date.replace(tzinfo=timezone.utc)
        else:
            end_date = end_date.astimezone(timezone.utc)

        # Cap end at now so we never request future data (avoids empty response)
        if end_date > now_utc:
            end_date = now_utc

        # For daily (or longer) bars, use start-of-day boundaries so the API returns
        # the expected bars. Daily bar for date D is typically timestamped at
        # midnight ET (e.g. 04:00 or 05:00 UTC).
        if timeframe == TimeFrame.Day or getattr(timeframe, "unit", None) == "Day":
            start_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
            # End: start of next day so the last full day is included
            end_date = end_date.replace(hour=0, minute=0, second=0, microsecond=0)
            end_date = end_date + timedelta(days=1)

        return start_date, end_date

    def download_stock_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: TimeFrame = TimeFrame.Minute,
        feed: DataFeed = DEFAULT_STOCK_FEED,
        output_file: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Download intraday stock data from Alpaca.

        Uses IEX feed by default (free). Use feed=DataFeed.SIP only if you have
        a subscription that permits querying SIP data.

        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            timeframe: Timeframe for bars (default: 1 minute)
            feed: Data feed - IEX (free) or SIP (paid). Default: IEX.
            output_file: Optional CSV file path to save data

        Returns:
            DataFrame with columns: Datetime, Open, High, Low, Close, Volume
        """
        start_date, end_date = self._normalize_request_times(start_date, end_date, timeframe)

        # Create request (feed=IEX avoids "subscription does not permit querying recent SIP data")
        request_params = StockBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=timeframe,
            start=start_date,
            end=end_date,
            feed=feed
        )

        # Fetch data
        bars = self.stock_client.get_stock_bars(request_params)
        
        # Convert to DataFrame
        df = bars.df
        
        if df.empty:
            raise ValueError(f"No data found for symbol {symbol} in the specified date range.")
        
        # Reset index to get symbol and timestamp as columns
        df = df.reset_index()
        
        # Rename and select columns
        df = df.rename(columns={
            'timestamp': 'Datetime',
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        })
        
        # Select only required columns
        df = df[['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']]
        
        # Sort by datetime
        df = df.sort_values('Datetime').reset_index(drop=True)
        
        # Save to CSV if output file specified
        if output_file:
            df.to_csv(output_file, index=False)
            print(f"Data saved to {output_file}")
        
        return df
    
    def download_crypto_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: TimeFrame = TimeFrame.Minute,
        output_file: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Download intraday cryptocurrency data from Alpaca.
        
        Args:
            symbol: Crypto symbol (e.g., 'BTCUSD')
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            timeframe: Timeframe for bars (default: 1 minute)
            output_file: Optional CSV file path to save data
        
        Returns:
            DataFrame with columns: Datetime, Open, High, Low, Close, Volume
        """
        start_date, end_date = self._normalize_request_times(start_date, end_date, timeframe)

        # Create request
        request_params = CryptoBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=timeframe,
            start=start_date,
            end=end_date
        )

        # Fetch data
        bars = self.crypto_client.get_crypto_bars(request_params)
        
        # Convert to DataFrame
        df = bars.df
        
        if df.empty:
            raise ValueError(f"No data found for symbol {symbol} in the specified date range.")
        
        # Reset index to get symbol and timestamp as columns
        df = df.reset_index()
        
        # Rename and select columns
        df = df.rename(columns={
            'timestamp': 'Datetime',
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        })
        
        # Select only required columns
        df = df[['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']]
        
        # Sort by datetime
        df = df.sort_values('Datetime').reset_index(drop=True)
        
        # Save to CSV if output file specified
        if output_file:
            df.to_csv(output_file, index=False)
            print(f"Data saved to {output_file}")
        
        return df
    
    def download_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        asset_type: str = 'stock',
        timeframe: TimeFrame = TimeFrame.Minute,
        feed: DataFeed = DEFAULT_STOCK_FEED,
        output_file: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Download intraday data (stock or crypto) from Alpaca.

        For stocks, uses IEX feed by default (free). Use feed=DataFeed.SIP only if
        your subscription permits querying SIP data.

        Args:
            symbol: Asset symbol (e.g., 'AAPL' for stocks, 'BTCUSD' for crypto)
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            asset_type: 'stock' or 'crypto' (default: 'stock')
            timeframe: Timeframe for bars (default: 1 minute)
            feed: Stock data feed - IEX (free) or SIP (paid). Ignored for crypto.
            output_file: Optional CSV file path to save data

        Returns:
            DataFrame with columns: Datetime, Open, High, Low, Close, Volume
        """
        if asset_type.lower() == 'stock':
            return self.download_stock_data(
                symbol, start_date, end_date, timeframe, feed=feed, output_file=output_file
            )
        elif asset_type.lower() == 'crypto':
            return self.download_crypto_data(symbol, start_date, end_date, timeframe, output_file)
        else:
            raise ValueError(f"Invalid asset_type: {asset_type}. Must be 'stock' or 'crypto'.")

    def download_multiple_stocks(
            self,
            symbols: list,
            start_date: datetime,
            end_date: datetime,
            timeframe: TimeFrame = TimeFrame.Day,
            feed: DataFeed = DEFAULT_STOCK_FEED
        ) -> pd.DataFrame:

        start_date, end_date = self._normalize_request_times(start_date, end_date, timeframe)

        request_params = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=timeframe,
            start=start_date,
            end=end_date,
            feed=feed
        )

        bars = self.stock_client.get_stock_bars(request_params)

        df = bars.df

        if df.empty:
            raise ValueError("No data returned")

        df = df.reset_index()

        df = df.rename(columns={
            'timestamp': 'Datetime',
            'symbol': 'Ticker',
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        })

        df = df[['Datetime', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']]

        df = df.sort_values(['Datetime','Ticker']).reset_index(drop=True)

        return df


class DataCleaner:
    """Cleans and organizes market data, adding derived features for analysis."""
    
    def __init__(self):
        """Initialize the data cleaner."""
        pass
    
    def clean_data(
        self,
        df: pd.DataFrame,
        datetime_column: str = 'Datetime',
        remove_duplicates: bool = True,
        remove_missing: bool = True
    ) -> pd.DataFrame:
        """
        Clean raw market data by removing missing/duplicate rows and setting datetime index.
        
        Args:
            df: Raw DataFrame with market data
            datetime_column: Name of the datetime column (default: 'Datetime')
            remove_duplicates: Whether to remove duplicate rows (default: True)
            remove_missing: Whether to remove rows with missing values (default: True)
        
        Returns:
            Cleaned DataFrame with Datetime as index, sorted chronologically
        """
        # Make a copy to avoid modifying original
        df_clean = df.copy()
        
        # Convert datetime column to datetime type if it's not already
        if datetime_column in df_clean.columns:
            df_clean[datetime_column] = pd.to_datetime(df_clean[datetime_column])
        else:
            raise ValueError(f"Column '{datetime_column}' not found in DataFrame.")
        
        # Remove missing values
        if remove_missing:
            initial_rows = len(df_clean)
            df_clean = df_clean.dropna()
            removed = initial_rows - len(df_clean)
            if removed > 0:
                print(f"Removed {removed} rows with missing values.")
        
        # Remove duplicates
        if remove_duplicates:
            initial_rows = len(df_clean)
            df_clean = df_clean.drop_duplicates()
            removed = initial_rows - len(df_clean)
            if removed > 0:
                print(f"Removed {removed} duplicate rows.")
        
        # Set Datetime as index
        df_clean.set_index(datetime_column, inplace=True)
        
        # Sort chronologically
        df_clean.sort_index(inplace=True)
        
        print(f"Data cleaned: {len(df_clean)} rows remaining.")
        return df_clean
    
    def add_returns(self, df: pd.DataFrame, periods: list = [1, 5, 10, 20]) -> pd.DataFrame:
        """
        Add return features (simple and log returns) to the DataFrame.
        
        Args:
            df: DataFrame with Close prices
            periods: List of periods for returns (default: [1, 5, 10, 20])
        
        Returns:
            DataFrame with added return columns
        """
        df_features = df.copy()
        
        for period in periods:
            # Simple returns (percentage change)
            df_features[f'Return_{period}'] = df_features['Close'].pct_change(periods=period)
            
            # Log returns
            df_features[f'LogReturn_{period}'] = np.log(
                df_features['Close'] / df_features['Close'].shift(periods=period)
            )
        
        return df_features
    
    def add_moving_averages(
        self,
        df: pd.DataFrame,
        windows: list = [5, 10, 20, 50, 200],
        ma_types: list = ['SMA', 'EMA']
    ) -> pd.DataFrame:
        """
        Add moving average features to the DataFrame.
        
        Args:
            df: DataFrame with price data
            windows: List of window sizes for moving averages (default: [5, 10, 20, 50, 200])
            ma_types: Types of moving averages to calculate (default: ['SMA', 'EMA'])
        
        Returns:
            DataFrame with added moving average columns
        """
        df_features = df.copy()
        
        for window in windows:
            if 'SMA' in ma_types:
                # Simple Moving Average
                df_features[f'SMA_{window}'] = df_features['Close'].rolling(window=window).mean()
            
            if 'EMA' in ma_types:
                # Exponential Moving Average
                df_features[f'EMA_{window}'] = df_features['Close'].ewm(span=window, adjust=False).mean()
        
        return df_features
    
    def add_volatility(
        self,
        df: pd.DataFrame,
        windows: list = [5, 10, 20],
        use_returns: bool = True
    ) -> pd.DataFrame:
        """
        Add volatility features (rolling standard deviation) to the DataFrame.
        
        Args:
            df: DataFrame with price or return data
            windows: List of window sizes for volatility calculation (default: [5, 10, 20])
            use_returns: If True, calculate volatility from returns; if False, from prices (default: True)
        
        Returns:
            DataFrame with added volatility columns
        """
        df_features = df.copy()
        
        for window in windows:
            if use_returns:
                # Calculate volatility from returns
                if 'Return_1' in df_features.columns:
                    df_features[f'Volatility_{window}'] = (
                        df_features['Return_1'].rolling(window=window).std()
                    )
                else:
                    # Calculate returns if not present
                    returns = df_features['Close'].pct_change()
                    df_features[f'Volatility_{window}'] = returns.rolling(window=window).std()
            else:
                # Calculate volatility from prices
                df_features[f'Volatility_{window}'] = (
                    df_features['Close'].rolling(window=window).std()
                )
        
        return df_features
    
    def add_rsi(self, df: pd.DataFrame, periods: list = [14]) -> pd.DataFrame:
        """
        Add Relative Strength Index (RSI) indicator.
        
        Args:
            df: DataFrame with Close prices
            periods: List of periods for RSI calculation (default: [14])
        
        Returns:
            DataFrame with added RSI columns
        """
        df_features = df.copy()
        
        for period in periods:
            delta = df_features['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            rs = gain / loss
            df_features[f'RSI_{period}'] = 100 - (100 / (1 + rs))
        
        return df_features
    
    def add_macd(
        self,
        df: pd.DataFrame,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9
    ) -> pd.DataFrame:
        """
        Add MACD (Moving Average Convergence Divergence) indicator.
        
        Args:
            df: DataFrame with Close prices
            fast_period: Fast EMA period (default: 12)
            slow_period: Slow EMA period (default: 26)
            signal_period: Signal line EMA period (default: 9)
        
        Returns:
            DataFrame with added MACD columns
        """
        df_features = df.copy()
        
        # Calculate EMAs
        ema_fast = df_features['Close'].ewm(span=fast_period, adjust=False).mean()
        ema_slow = df_features['Close'].ewm(span=slow_period, adjust=False).mean()
        
        # MACD line
        df_features['MACD'] = ema_fast - ema_slow
        
        # Signal line
        df_features['MACD_Signal'] = df_features['MACD'].ewm(span=signal_period, adjust=False).mean()
        
        # Histogram
        df_features['MACD_Histogram'] = df_features['MACD'] - df_features['MACD_Signal']
        
        return df_features
    
    def add_bollinger_bands(
        self,
        df: pd.DataFrame,
        window: int = 20,
        num_std: float = 2.0
    ) -> pd.DataFrame:
        """
        Add Bollinger Bands indicator.
        
        Args:
            df: DataFrame with Close prices
            window: Window size for moving average (default: 20)
            num_std: Number of standard deviations for bands (default: 2.0)
        
        Returns:
            DataFrame with added Bollinger Bands columns
        """
        df_features = df.copy()
        
        # Calculate moving average and standard deviation
        df_features[f'BB_Middle_{window}'] = df_features['Close'].rolling(window=window).mean()
        bb_std = df_features['Close'].rolling(window=window).std()
        
        # Upper and lower bands
        df_features[f'BB_Upper_{window}'] = (
            df_features[f'BB_Middle_{window}'] + (bb_std * num_std)
        )
        df_features[f'BB_Lower_{window}'] = (
            df_features[f'BB_Middle_{window}'] - (bb_std * num_std)
        )
        
        # Band width and position
        df_features[f'BB_Width_{window}'] = (
            df_features[f'BB_Upper_{window}'] - df_features[f'BB_Lower_{window}']
        )
        df_features[f'BB_Position_{window}'] = (
            (df_features['Close'] - df_features[f'BB_Lower_{window}']) /
            (df_features[f'BB_Upper_{window}'] - df_features[f'BB_Lower_{window}'])
        )
        
        return df_features
    
    def add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add additional price-based features.
        
        Args:
            df: DataFrame with OHLC data
        
        Returns:
            DataFrame with added price feature columns
        """
        df_features = df.copy()
        
        # Price changes
        df_features['Price_Change'] = df_features['Close'] - df_features['Open']
        df_features['Price_Change_Pct'] = (
            (df_features['Close'] - df_features['Open']) / df_features['Open']
        )
        
        # High-Low spread
        df_features['HL_Spread'] = df_features['High'] - df_features['Low']
        df_features['HL_Spread_Pct'] = (
            (df_features['High'] - df_features['Low']) / df_features['Close']
        )
        
        # Body size (Open-Close difference)
        df_features['Body_Size'] = abs(df_features['Close'] - df_features['Open'])
        df_features['Body_Size_Pct'] = (
            abs(df_features['Close'] - df_features['Open']) / df_features['Close']
        )
        
        # Upper and lower shadows
        df_features['Upper_Shadow'] = df_features['High'] - df_features[['Open', 'Close']].max(axis=1)
        df_features['Lower_Shadow'] = df_features[['Open', 'Close']].min(axis=1) - df_features['Low']
        
        return df_features
    
    def add_volume_features(self, df: pd.DataFrame, windows: list = [5, 10, 20]) -> pd.DataFrame:
        """
        Add volume-based features.
        
        Args:
            df: DataFrame with Volume data
            windows: List of window sizes for volume moving averages (default: [5, 10, 20])
        
        Returns:
            DataFrame with added volume feature columns
        """
        df_features = df.copy()
        
        # Volume moving averages
        for window in windows:
            df_features[f'Volume_MA_{window}'] = (
                df_features['Volume'].rolling(window=window).mean()
            )
        
        # Volume ratio (current volume vs average)
        if 'Volume_MA_20' in df_features.columns:
            df_features['Volume_Ratio'] = (
                df_features['Volume'] / df_features['Volume_MA_20']
            )
        
        # Price-Volume relationship
        df_features['Price_Volume'] = df_features['Close'] * df_features['Volume']
        
        return df_features
    
    def prepare_data(
        self,
        df: pd.DataFrame,
        datetime_column: str = 'Datetime',
        add_returns: bool = True,
        add_moving_averages: bool = True,
        add_volatility: bool = True,
        add_rsi: bool = True,
        add_macd: bool = True,
        add_bollinger_bands: bool = True,
        add_price_features: bool = True,
        add_volume_features: bool = True,
        remove_duplicates: bool = True,
        remove_missing: bool = True
    ) -> pd.DataFrame:
        """
        Complete data preparation pipeline: clean data and add all derived features.
        
        Args:
            df: Raw DataFrame with market data
            datetime_column: Name of the datetime column (default: 'Datetime')
            add_returns: Whether to add return features (default: True)
            add_moving_averages: Whether to add moving average features (default: True)
            add_volatility: Whether to add volatility features (default: True)
            add_rsi: Whether to add RSI indicator (default: True)
            add_macd: Whether to add MACD indicator (default: True)
            add_bollinger_bands: Whether to add Bollinger Bands (default: True)
            add_price_features: Whether to add price-based features (default: True)
            add_volume_features: Whether to add volume-based features (default: True)
            remove_duplicates: Whether to remove duplicate rows (default: True)
            remove_missing: Whether to remove rows with missing values (default: True)
        
        Returns:
            Fully prepared DataFrame ready for analysis
        """
        print("Starting data preparation pipeline...")
        
        # Step 1: Clean data
        df_clean = self.clean_data(
            df,
            datetime_column=datetime_column,
            remove_duplicates=remove_duplicates,
            remove_missing=remove_missing
        )
        
        # Step 2: Add derived features
        if add_returns:
            print("Adding return features...")
            df_clean = self.add_returns(df_clean)
        
        if add_moving_averages:
            print("Adding moving averages...")
            df_clean = self.add_moving_averages(df_clean)
        
        if add_volatility:
            print("Adding volatility features...")
            df_clean = self.add_volatility(df_clean)
        
        if add_rsi:
            print("Adding RSI indicator...")
            df_clean = self.add_rsi(df_clean)
        
        if add_macd:
            print("Adding MACD indicator...")
            df_clean = self.add_macd(df_clean)
        
        if add_bollinger_bands:
            print("Adding Bollinger Bands...")
            df_clean = self.add_bollinger_bands(df_clean)
        
        if add_price_features:
            print("Adding price features...")
            df_clean = self.add_price_features(df_clean)
        
        if add_volume_features:
            print("Adding volume features...")
            df_clean = self.add_volume_features(df_clean)
        
        # Remove rows with NaN values created by feature engineering
        initial_rows = len(df_clean)
        df_clean = df_clean.dropna()
        removed = initial_rows - len(df_clean)
        if removed > 0:
            print(f"Removed {removed} rows with NaN values from feature engineering.")
        
        print(f"\nData preparation complete!")
        print(f"Final dataset: {len(df_clean)} rows, {len(df_clean.columns)} columns")
        print(f"Date range: {df_clean.index.min()} to {df_clean.index.max()}")
        
        return df_clean


def main():
    """
    Example usage of the DataDownloader and DataCleaner classes.
    """
    try:
        # Step 1: Download data
        downloader = DataDownloader()
        
        # Set date range (last 180 days); use UTC so daily bar request is unambiguous
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=180)
        
        # Download stock data
        print("=" * 60)
        print("STEP 1: Downloading AAPL stock data...")
        print("=" * 60)
        df_raw = downloader.download_data(
            symbol='AAPL',
            start_date=start_date,
            end_date=end_date,
            asset_type='stock',
            timeframe=TimeFrame.Day,
            output_file='aapl_daily_data.csv'
        )
        
        print(f"\nDownloaded {len(df_raw)} records")
        print(f"Date range: {df_raw['Datetime'].min()} to {df_raw['Datetime'].max()}")
        
        # Step 2: Clean and prepare data
        print("\n" + "=" * 60)
        print("STEP 2: Cleaning and preparing data...")
        print("=" * 60)
        cleaner = DataCleaner()
        df_clean = cleaner.prepare_data(
            df_raw,
            datetime_column='Datetime',
            add_returns=True,
            add_moving_averages=True,
            add_volatility=True,
            add_rsi=True,
            add_macd=True,
            add_bollinger_bands=True,
            add_price_features=True,
            add_volume_features=True
        )
        
        # Save cleaned data
        output_file = 'aapl_daily_data_cleaned.csv'
        df_clean.to_csv(output_file)
        print(f"\nCleaned data saved to {output_file}")
        
        # Display summary
        print("\n" + "=" * 60)
        print("DATA SUMMARY")
        print("=" * 60)
        print(f"\nShape: {df_clean.shape}")
        print(f"\nColumns ({len(df_clean.columns)}):")
        for i, col in enumerate(df_clean.columns, 1):
            print(f"  {i}. {col}")
        
        print(f"\n\nFirst few rows:")
        print(df_clean.head())
        
        print(f"\n\nLast few rows:")
        print(df_clean.tail())
        
        print(f"\n\nBasic statistics:")
        print(df_clean[['Open', 'High', 'Low', 'Close', 'Volume']].describe())
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure to set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables.")


if __name__ == "__main__":
    main()

