"""
Moving Average Crossover Trading Strategy
Uses short-term and long-term moving averages to generate trading signals.

Strategy Logic:
- Golden Cross: When short MA crosses above long MA -> Buy signal
- Death Cross: When short MA crosses below long MA -> Sell signal
- Can be applied cross-sectionally to rank stocks by MA crossover strength
"""

import pandas as pd
import numpy as np


class MACrossoverStrategy:
    """
    Moving Average Crossover trading strategy.
    
    This strategy uses two moving averages (short-term and long-term) to identify
    trend changes. When the short MA crosses above the long MA (golden cross),
    it signals an uptrend and generates buy signals. When the short MA crosses
    below the long MA (death cross), it signals a downtrend and generates sell signals.
    
    Key Components:
    - Short-term MA (fast): Typically 5-20 periods
    - Long-term MA (slow): Typically 20-200 periods
    - Crossover detection: Identifies when MAs cross
    - Cross-sectional ranking: Can rank stocks by crossover strength
    - Position sizing: Equal weights or based on crossover strength
    """

    def __init__(
        self,
        short_window=5,
        long_window=20,
        top_k=3,
        use_cross_sectional=True,
        ma_type="SMA"
    ):
        """
        Initialize MA Crossover strategy.
        
        Args:
            short_window: Period for short-term moving average (default: 5)
            long_window: Period for long-term moving average (default: 20)
            top_k: Number of top/bottom stocks to trade in cross-sectional mode (default: 3)
            use_cross_sectional: If True, ranks stocks cross-sectionally (default: True)
            ma_type: Type of MA to use - "SMA" or "EMA" (default: "SMA")
        """
        self.short_window = short_window
        self.long_window = long_window
        self.top_k = top_k
        self.use_cross_sectional = use_cross_sectional
        self.ma_type = ma_type.upper()
        
        if self.short_window >= self.long_window:
            raise ValueError("short_window must be less than long_window")
        
        # Features used (will be created if not present)
        self.short_ma_col = f"{self.ma_type}_{self.short_window}"
        self.long_ma_col = f"{self.ma_type}_{self.long_window}"
        self.features = [self.short_ma_col, self.long_ma_col, "Close"]

    # ---------------------------------------------------
    # Data loading
    # ---------------------------------------------------

    def load_data(self, filepath):
        """Load data from CSV file."""
        df = pd.read_csv(filepath)
        df["Datetime"] = pd.to_datetime(df["Datetime"])
        return df

    # ---------------------------------------------------
    # Calculate moving averages if not present
    # ---------------------------------------------------

    def _calculate_moving_averages(self, df):
        """
        Calculate moving averages if they don't exist in the dataset.
        """
        df = df.copy()
        
        # Group by ticker to calculate MAs per stock
        for ticker, group in df.groupby("Ticker"):
            ticker_idx = group.index
            
            if self.ma_type == "SMA":
                # Simple Moving Average
                if self.short_ma_col not in df.columns:
                    df.loc[ticker_idx, self.short_ma_col] = (
                        group["Close"].rolling(window=self.short_window).mean()
                    )
                if self.long_ma_col not in df.columns:
                    df.loc[ticker_idx, self.long_ma_col] = (
                        group["Close"].rolling(window=self.long_window).mean()
                    )
            elif self.ma_type == "EMA":
                # Exponential Moving Average
                if self.short_ma_col not in df.columns:
                    df.loc[ticker_idx, self.short_ma_col] = (
                        group["Close"].ewm(span=self.short_window, adjust=False).mean()
                    )
                if self.long_ma_col not in df.columns:
                    df.loc[ticker_idx, self.long_ma_col] = (
                        group["Close"].ewm(span=self.long_window, adjust=False).mean()
                    )
        
        return df

    # ---------------------------------------------------
    # Feature normalization (optional)
    # ---------------------------------------------------

    def normalize_features(self, df):
        """
        Normalize features cross-sectionally (optional).
        Not typically needed for MA crossover, but included for compatibility.
        """
        df = df.copy()
        return df

    # ---------------------------------------------------
    # Train (no-op for MA crossover strategy)
    # ---------------------------------------------------

    def train(self, df):
        """
        Training method for compatibility with backtest framework.
        MA crossover strategy doesn't require training, but we validate data here.
        """
        # Calculate MAs if needed
        df = self._calculate_moving_averages(df)
        
        # Validate that required columns exist
        required_cols = [self.short_ma_col, self.long_ma_col, "Close"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Missing required columns: {missing_cols}. "
                f"Please ensure moving averages are calculated."
            )
        
        # No actual training needed for MA crossover strategy
        pass

    # ---------------------------------------------------
    # Prediction: Calculate crossover signals
    # ---------------------------------------------------

    def predict(self, df):
        """
        Calculate MA crossover signals and strength.
        
        Creates a 'prediction' column that represents:
        - Crossover strength: (short_MA - long_MA) / long_MA
        - Positive values: short MA above long MA (bullish)
        - Negative values: short MA below long MA (bearish)
        - Magnitude indicates strength of the signal
        
        Returns DataFrame with 'prediction' column.
        """
        df = df.copy()
        
        # Calculate MAs if needed
        df = self._calculate_moving_averages(df)
        
        # Calculate crossover signal strength
        # Normalized by long MA to make it comparable across stocks
        df["prediction"] = (
            (df[self.short_ma_col] - df[self.long_ma_col]) / df[self.long_ma_col]
        )
        
        # Handle NaN values (when MAs haven't been calculated yet)
        df["prediction"] = df["prediction"].fillna(0)
        
        return df

    # ---------------------------------------------------
    # Generate trading signals
    # ---------------------------------------------------

    def generate_signals(self, df):
        """
        Generate trading signals based on MA crossovers.
        
        Two modes:
        1. Cross-sectional (use_cross_sectional=True):
           - Ranks stocks by crossover strength at each time point
           - Long top K stocks (strongest bullish crossovers)
           - Short bottom K stocks (strongest bearish crossovers)
        
        2. Absolute (use_cross_sectional=False):
           - Long when short MA > long MA (golden cross)
           - Short when short MA < long MA (death cross)
           - No signal when MAs are equal or too close
        
        Returns DataFrame with 'signal' column (1=long, -1=short, 0=no signal).
        """
        df = df.copy()
        
        # Ensure prediction column exists
        if "prediction" not in df.columns:
            df = self.predict(df)
        
        signals = []
        
        for date, group in df.groupby("Datetime"):
            group = group.copy()
            
            if self.use_cross_sectional:
                # Cross-sectional ranking mode
                # Rank stocks by crossover strength (higher = more bullish)
                group["rank"] = group["prediction"].rank(ascending=False)
                
                # Initialize signals to zero
                group["signal"] = 0
                
                # Long top K stocks (strongest bullish crossovers)
                long_idx = group["rank"] <= self.top_k
                
                # Short bottom K stocks (strongest bearish crossovers)
                total_stocks = len(group)
                short_idx = group["rank"] >= (total_stocks - self.top_k + 1)
                
                # Assign signals
                group.loc[long_idx, "signal"] = 1
                group.loc[short_idx, "signal"] = -1
            else:
                # Absolute crossover mode
                # Long when short MA > long MA (golden cross)
                # Short when short MA < long MA (death cross)
                group["signal"] = 0
                
                # Golden cross: short MA above long MA
                golden_cross = group["prediction"] > 0
                group.loc[golden_cross, "signal"] = 1
                
                # Death cross: short MA below long MA
                death_cross = group["prediction"] < 0
                group.loc[death_cross, "signal"] = -1
            
            signals.append(group)
        
        df = pd.concat(signals)
        
        return df

    # ---------------------------------------------------
    # Position sizing
    # ---------------------------------------------------

    def position_sizing(self, df):
        """
        Calculate position weights for each signal.
        
        If cross-sectional mode:
        - Equal weights: each position gets 1/top_k of the allocation
        
        If absolute mode:
        - Weight based on crossover strength (normalized)
        
        Returns DataFrame with 'weight' column.
        """
        df = df.copy()
        
        if self.use_cross_sectional:
            # Equal weight per position: signal / top_k
            df["weight"] = df["signal"] / self.top_k
        else:
            # Weight based on crossover strength
            # Normalize prediction to [-1, 1] range for weighting
            max_pred = df["prediction"].abs().max()
            if max_pred > 0:
                normalized_pred = df["prediction"] / max_pred
            else:
                normalized_pred = df["prediction"]
            
            # Use signal direction with normalized strength
            df["weight"] = np.sign(df["signal"]) * np.abs(normalized_pred) * 0.1  # Scale to reasonable size
        
        return df

    # ---------------------------------------------------
    # Strategy returns
    # ---------------------------------------------------

    def compute_returns(self, df):
        """
        Calculate strategy returns.
        
        Strategy return = sum of (weight * target return) for all stocks at each time point.
        
        Returns Series of portfolio returns indexed by Datetime.
        """
        df = df.copy()
        
        # Calculate strategy return: weight * target return
        df["strategy_return"] = df["weight"] * df["Target"]
        
        # Aggregate to portfolio level (sum across all stocks at each time)
        portfolio = df.groupby("Datetime")["strategy_return"].sum()
        
        return portfolio

    # ---------------------------------------------------
    # Sharpe ratio
    # ---------------------------------------------------

    def sharpe_ratio(self, returns):
        """
        Calculate annualized Sharpe ratio.
        
        Sharpe = sqrt(252) * mean(returns) / std(returns)
        Assumes 252 trading days per year.
        """
        mean = returns.mean()
        std = returns.std()
        
        if std == 0:
            return 0.0
        
        sharpe = np.sqrt(252) * mean / std
        
        return sharpe

    # ---------------------------------------------------
    # Additional utility: Detect actual crossovers
    # ---------------------------------------------------

    def detect_crossovers(self, df):
        """
        Detect actual crossover events (when MAs cross each other).
        
        Returns DataFrame with additional columns:
        - 'crossover': 1 for golden cross, -1 for death cross, 0 for no crossover
        - 'crossover_strength': Magnitude of the crossover
        """
        df = df.copy()
        
        # Calculate MAs if needed
        df = self._calculate_moving_averages(df)
        
        # Calculate previous period's MA relationship
        df["prev_prediction"] = df.groupby("Ticker")["prediction"].shift(1)
        
        # Detect crossovers
        df["crossover"] = 0
        
        # Golden cross: prediction > 0 and prev_prediction <= 0
        golden_cross = (df["prediction"] > 0) & (df["prev_prediction"] <= 0)
        df.loc[golden_cross, "crossover"] = 1
        
        # Death cross: prediction < 0 and prev_prediction >= 0
        death_cross = (df["prediction"] < 0) & (df["prev_prediction"] >= 0)
        df.loc[death_cross, "crossover"] = -1
        
        # Crossover strength
        df["crossover_strength"] = df["prediction"].abs()
        
        # Clean up
        df = df.drop(columns=["prev_prediction"])
        
        return df


# -------------------------------------------------------
# Example execution
# -------------------------------------------------------

def main():
    """
    Example usage of MA Crossover Strategy.
    """
    print("=" * 60)
    print("Moving Average Crossover Strategy Example")
    print("=" * 60)
    
    # Initialize strategy
    # Using 5-day and 20-day SMAs (common configuration)
    strategy = MACrossoverStrategy(
        short_window=5,
        long_window=20,
        top_k=3,
        use_cross_sectional=True,
        ma_type="SMA"
    )
    
    print(f"\nStrategy Configuration:")
    print(f"  - Short MA: {strategy.short_window} periods ({strategy.ma_type})")
    print(f"  - Long MA: {strategy.long_window} periods ({strategy.ma_type})")
    print(f"  - Top K stocks: {strategy.top_k}")
    print(f"  - Cross-sectional mode: {strategy.use_cross_sectional}")
    
    print("\nLoading dataset...")
    df = strategy.load_data("multi_stock_dataset.csv")
    
    # Calculate MAs if needed
    df = strategy._calculate_moving_averages(df)
    
    # Train (validates data)
    print("Validating data...")
    strategy.train(df)
    
    # Generate predictions
    print("Calculating MA crossover signals...")
    df = strategy.predict(df)
    
    # Detect crossovers
    print("Detecting crossover events...")
    df = strategy.detect_crossovers(df)
    
    # Generate signals
    print("Generating trading signals...")
    df = strategy.generate_signals(df)
    
    # Position sizing
    df = strategy.position_sizing(df)
    
    # Show sample results
    print("\n" + "-" * 60)
    print("Sample Results:")
    print("-" * 60)
    
    sample_cols = ["Datetime", "Ticker", "Close", 
                   strategy.short_ma_col, strategy.long_ma_col,
                   "prediction", "crossover", "signal", "weight"]
    available_cols = [col for col in sample_cols if col in df.columns]
    
    print(df[available_cols].head(20))
    
    # Show crossover statistics
    if "crossover" in df.columns:
        print("\n" + "-" * 60)
        print("Crossover Statistics:")
        print("-" * 60)
        crossover_counts = df["crossover"].value_counts()
        print(f"Golden Crosses (buy signals): {crossover_counts.get(1, 0)}")
        print(f"Death Crosses (sell signals): {crossover_counts.get(-1, 0)}")
        print(f"No crossovers: {crossover_counts.get(0, 0)}")
    
    # Compute returns if Target column exists
    if "Target" in df.columns:
        returns = strategy.compute_returns(df)
        sharpe = strategy.sharpe_ratio(returns)
        
        print("\n" + "-" * 60)
        print("Strategy Performance:")
        print("-" * 60)
        print(f"Total Return: {returns.sum():.4f}")
        print(f"Sharpe Ratio: {sharpe:.4f}")
    
    print("\n" + "=" * 60)
    print("Example Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

