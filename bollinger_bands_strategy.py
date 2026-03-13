"""
Bollinger Bands Trading Strategy
Uses Bollinger Bands to identify overbought and oversold conditions.

Strategy Logic:
- Lower Band Touch: When price touches or goes below lower band -> Buy signal (oversold)
- Upper Band Touch: When price touches or goes above upper band -> Sell signal (overbought)
- BB Position: Uses position within bands (0 = lower, 1 = upper) to generate signals
- Can be applied cross-sectionally to rank stocks by BB signals
"""

import pandas as pd
import numpy as np


class BollingerBandsStrategy:
    """
    Bollinger Bands trading strategy.
    
    Bollinger Bands consist of:
    - Middle Band: Moving average (typically SMA)
    - Upper Band: Middle + (num_std * standard deviation)
    - Lower Band: Middle - (num_std * standard deviation)
    
    Trading Signals:
    - Buy when price touches/goes below lower band (oversold condition)
    - Sell when price touches/goes above upper band (overbought condition)
    - Can use BB Position (0-1 scale) for more nuanced signals
    
    Key Components:
    - Window: Period for moving average and standard deviation
    - Num Std: Number of standard deviations for bands (typically 2.0)
    - Cross-sectional ranking: Can rank stocks by BB signals
    - Position sizing: Based on distance from bands or equal weights
    """

    def __init__(
        self,
        window=20,
        num_std=2.0,
        top_k=3,
        use_cross_sectional=True,
        signal_type="touch",  # "touch", "position", or "bounce"
        min_band_width_pct=0.0  # Minimum band width as % of price (filters low volatility)
    ):
        """
        Initialize Bollinger Bands strategy.
        
        Args:
            window: Period for moving average and standard deviation (default: 20)
            num_std: Number of standard deviations for bands (default: 2.0)
            top_k: Number of top/bottom stocks to trade in cross-sectional mode (default: 3)
            use_cross_sectional: If True, ranks stocks cross-sectionally (default: True)
            signal_type: Type of signal generation:
                - "touch": Signals when price touches bands
                - "position": Uses BB_Position (0-1) for signals
                - "bounce": Expects price to bounce back from bands
            min_band_width_pct: Minimum band width as % of price to filter signals (default: 0.0)
        """
        self.window = window
        self.num_std = num_std
        self.top_k = top_k
        self.use_cross_sectional = use_cross_sectional
        self.signal_type = signal_type.lower()
        self.min_band_width_pct = min_band_width_pct
        
        if self.signal_type not in ["touch", "position", "bounce"]:
            raise ValueError("signal_type must be 'touch', 'position', or 'bounce'")
        
        # Column names for Bollinger Bands
        self.middle_col = f"BB_Middle_{self.window}"
        self.upper_col = f"BB_Upper_{self.window}"
        self.lower_col = f"BB_Lower_{self.window}"
        self.width_col = f"BB_Width_{self.window}"
        self.position_col = f"BB_Position_{self.window}"
        
        self.features = [
            self.middle_col,
            self.upper_col,
            self.lower_col,
            self.width_col,
            self.position_col,
            "Close"
        ]

    # ---------------------------------------------------
    # Data loading
    # ---------------------------------------------------

    def load_data(self, filepath):
        """Load data from CSV file."""
        df = pd.read_csv(filepath)
        df["Datetime"] = pd.to_datetime(df["Datetime"])
        return df

    # ---------------------------------------------------
    # Calculate Bollinger Bands if not present
    # ---------------------------------------------------

    def _calculate_bollinger_bands(self, df):
        """
        Calculate Bollinger Bands if they don't exist in the dataset.
        """
        df = df.copy()
        
        # Group by ticker to calculate BB per stock
        for ticker, group in df.groupby("Ticker"):
            ticker_idx = group.index
            
            # Calculate if columns don't exist
            if self.middle_col not in df.columns:
                # Middle band: Moving average
                df.loc[ticker_idx, self.middle_col] = (
                    group["Close"].rolling(window=self.window).mean()
                )
            
            if self.upper_col not in df.columns or self.lower_col not in df.columns:
                # Calculate standard deviation
                bb_std = group["Close"].rolling(window=self.window).std()
                
                # Get middle band (calculate if needed)
                if self.middle_col not in df.columns:
                    middle = group["Close"].rolling(window=self.window).mean()
                else:
                    middle = df.loc[ticker_idx, self.middle_col]
                
                # Upper and lower bands
                if self.upper_col not in df.columns:
                    df.loc[ticker_idx, self.upper_col] = middle + (bb_std * self.num_std)
                
                if self.lower_col not in df.columns:
                    df.loc[ticker_idx, self.lower_col] = middle - (bb_std * self.num_std)
            
            # Band width
            if self.width_col not in df.columns:
                if self.upper_col in df.columns and self.lower_col in df.columns:
                    df.loc[ticker_idx, self.width_col] = (
                        df.loc[ticker_idx, self.upper_col] - 
                        df.loc[ticker_idx, self.lower_col]
                    )
            
            # BB Position: (Close - Lower) / (Upper - Lower)
            # 0 = at lower band, 1 = at upper band
            if self.position_col not in df.columns:
                if (self.upper_col in df.columns and 
                    self.lower_col in df.columns and
                    self.width_col in df.columns):
                    width = df.loc[ticker_idx, self.width_col]
                    # Avoid division by zero
                    width = width.replace(0, np.nan)
                    df.loc[ticker_idx, self.position_col] = (
                        (group["Close"] - df.loc[ticker_idx, self.lower_col]) / width
                    )
        
        return df

    # ---------------------------------------------------
    # Feature normalization (optional)
    # ---------------------------------------------------

    def normalize_features(self, df):
        """
        Normalize features cross-sectionally (optional).
        Not typically needed for Bollinger Bands, but included for compatibility.
        """
        df = df.copy()
        return df

    # ---------------------------------------------------
    # Train (no-op for Bollinger Bands strategy)
    # ---------------------------------------------------

    def train(self, df):
        """
        Training method for compatibility with backtest framework.
        Bollinger Bands strategy doesn't require training, but we validate data here.
        """
        # Calculate BB if needed
        df = self._calculate_bollinger_bands(df)
        
        # Validate that required columns exist
        required_cols = [self.upper_col, self.lower_col, "Close"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Missing required columns: {missing_cols}. "
                f"Please ensure Bollinger Bands are calculated."
            )
        
        # No actual training needed for Bollinger Bands strategy
        pass

    # ---------------------------------------------------
    # Prediction: Calculate BB signals
    # ---------------------------------------------------

    def predict(self, df):
        """
        Calculate Bollinger Bands signals and strength.
        
        Creates a 'prediction' column that represents:
        - For "touch" mode: Distance from bands (negative = below lower, positive = above upper)
        - For "position" mode: BB_Position normalized to [-1, 1] range
        - For "bounce" mode: Expected bounce strength from bands
        
        Returns DataFrame with 'prediction' column.
        """
        df = df.copy()
        
        # Calculate BB if needed
        df = self._calculate_bollinger_bands(df)
        
        if self.signal_type == "touch":
            # Distance from bands
            # Negative when below lower band (oversold), positive when above upper band (overbought)
            df["prediction"] = (
                (df["Close"] - df[self.lower_col]) / df["Close"] -
                (df[self.upper_col] - df["Close"]) / df["Close"]
            )
            # Normalize: -1 when at lower band, +1 when at upper band
            df["prediction"] = df["prediction"] * 2
        
        elif self.signal_type == "position":
            # Use BB_Position directly (0 = lower band, 1 = upper band)
            # Convert to [-1, 1] range: 0 -> -1, 1 -> +1
            if self.position_col in df.columns:
                df["prediction"] = (df[self.position_col] - 0.5) * 2
            else:
                # Calculate position if not present
                width = df[self.upper_col] - df[self.lower_col]
                width = width.replace(0, np.nan)
                position = (df["Close"] - df[self.lower_col]) / width
                df["prediction"] = (position - 0.5) * 2
        
        elif self.signal_type == "bounce":
            # Expect price to bounce back from bands
            # Strong buy signal when far below lower band
            # Strong sell signal when far above upper band
            lower_dist = (df["Close"] - df[self.lower_col]) / df["Close"]
            upper_dist = (df[self.upper_col] - df["Close"]) / df["Close"]
            df["prediction"] = upper_dist - lower_dist  # Negative = oversold, Positive = overbought
        
        # Handle NaN values
        df["prediction"] = df["prediction"].fillna(0)
        
        # Filter by minimum band width if specified
        if self.min_band_width_pct > 0 and self.width_col in df.columns:
            min_width = df["Close"] * self.min_band_width_pct
            narrow_bands = df[self.width_col] < min_width
            df.loc[narrow_bands, "prediction"] = 0  # No signal for narrow bands
        
        return df

    # ---------------------------------------------------
    # Generate trading signals
    # ---------------------------------------------------

    def generate_signals(self, df):
        """
        Generate trading signals based on Bollinger Bands.
        
        Two modes:
        1. Cross-sectional (use_cross_sectional=True):
           - Ranks stocks by BB signal strength at each time point
           - Long top K stocks (most oversold / strongest buy signals)
           - Short bottom K stocks (most overbought / strongest sell signals)
        
        2. Absolute (use_cross_sectional=False):
           - Long when price at/below lower band (oversold)
           - Short when price at/above upper band (overbought)
           - No signal when price is between bands
        
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
                # Rank stocks by prediction (lower = more oversold/buy signal)
                # For BB, negative prediction = oversold (buy), positive = overbought (sell)
                group["rank"] = group["prediction"].rank(ascending=True)  # Lower rank = more oversold
                
                # Initialize signals to zero
                group["signal"] = 0
                
                # Long bottom K stocks (most oversold / strongest buy signals)
                long_idx = group["rank"] <= self.top_k
                
                # Short top K stocks (most overbought / strongest sell signals)
                total_stocks = len(group)
                short_idx = group["rank"] >= (total_stocks - self.top_k + 1)
                
                # Assign signals
                group.loc[long_idx, "signal"] = 1
                group.loc[short_idx, "signal"] = -1
            else:
                # Absolute mode
                group["signal"] = 0
                
                # Long when price at/below lower band (oversold)
                if self.signal_type == "touch":
                    oversold = group["Close"] <= group[self.lower_col]
                    group.loc[oversold, "signal"] = 1
                    
                    overbought = group["Close"] >= group[self.upper_col]
                    group.loc[overbought, "signal"] = -1
                
                elif self.signal_type == "position":
                    # Long when BB_Position < 0.2 (near lower band)
                    # Short when BB_Position > 0.8 (near upper band)
                    if self.position_col in group.columns:
                        oversold = group[self.position_col] < 0.2
                        overbought = group[self.position_col] > 0.8
                    else:
                        # Calculate position
                        width = group[self.upper_col] - group[self.lower_col]
                        width = width.replace(0, np.nan)
                        position = (group["Close"] - group[self.lower_col]) / width
                        oversold = position < 0.2
                        overbought = position > 0.8
                    
                    group.loc[oversold, "signal"] = 1
                    group.loc[overbought, "signal"] = -1
                
                elif self.signal_type == "bounce":
                    # Long when far below lower band (expect bounce up)
                    # Short when far above upper band (expect bounce down)
                    lower_threshold = group[self.lower_col] * 0.98  # 2% below lower band
                    upper_threshold = group[self.upper_col] * 1.02  # 2% above upper band
                    
                    oversold = group["Close"] < lower_threshold
                    overbought = group["Close"] > upper_threshold
                    
                    group.loc[oversold, "signal"] = 1
                    group.loc[overbought, "signal"] = -1
            
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
        - Weight based on distance from bands (normalized)
        
        Returns DataFrame with 'weight' column.
        """
        df = df.copy()
        
        if self.use_cross_sectional:
            # Equal weight per position: signal / top_k
            df["weight"] = df["signal"] / self.top_k
        else:
            # Weight based on distance from bands
            # Normalize prediction to reasonable size
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
    # Additional utility: Detect band touches
    # ---------------------------------------------------

    def detect_band_touches(self, df):
        """
        Detect when price touches or crosses Bollinger Bands.
        
        Returns DataFrame with additional columns:
        - 'lower_touch': 1 when price touches/crosses below lower band
        - 'upper_touch': 1 when price touches/crosses above upper band
        - 'band_squeeze': 1 when band width is narrow (low volatility)
        """
        df = df.copy()
        
        # Calculate BB if needed
        df = self._calculate_bollinger_bands(df)
        
        # Detect touches
        df["lower_touch"] = (df["Close"] <= df[self.lower_col]).astype(int)
        df["upper_touch"] = (df["Close"] >= df[self.upper_col]).astype(int)
        
        # Band squeeze: when width is below average (low volatility)
        if self.width_col in df.columns:
            avg_width = df.groupby("Ticker")[self.width_col].transform(lambda x: x.rolling(20).mean())
            df["band_squeeze"] = (df[self.width_col] < avg_width * 0.8).astype(int)
        else:
            df["band_squeeze"] = 0
        
        return df


# -------------------------------------------------------
# Example execution
# -------------------------------------------------------

def main():
    """
    Example usage of Bollinger Bands Strategy.
    """
    print("=" * 60)
    print("Bollinger Bands Strategy Example")
    print("=" * 60)
    
    # Initialize strategy
    # Using 20-period BB with 2 standard deviations (standard configuration)
    strategy = BollingerBandsStrategy(
        window=20,
        num_std=2.0,
        top_k=3,
        use_cross_sectional=True,
        signal_type="touch",
        min_band_width_pct=0.0
    )
    
    print(f"\nStrategy Configuration:")
    print(f"  - Window: {strategy.window} periods")
    print(f"  - Number of Std Devs: {strategy.num_std}")
    print(f"  - Top K stocks: {strategy.top_k}")
    print(f"  - Cross-sectional mode: {strategy.use_cross_sectional}")
    print(f"  - Signal type: {strategy.signal_type}")
    
    print("\nLoading dataset...")
    df = strategy.load_data("multi_stock_dataset.csv")
    
    # Calculate BB if needed
    df = strategy._calculate_bollinger_bands(df)
    
    # Train (validates data)
    print("Validating data...")
    strategy.train(df)
    
    # Generate predictions
    print("Calculating Bollinger Bands signals...")
    df = strategy.predict(df)
    
    # Detect band touches
    print("Detecting band touches...")
    df = strategy.detect_band_touches(df)
    
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
                   strategy.lower_col, strategy.upper_col,
                   "prediction", "lower_touch", "upper_touch", "signal", "weight"]
    available_cols = [col for col in sample_cols if col in df.columns]
    
    print(df[available_cols].head(20))
    
    # Show touch statistics
    if "lower_touch" in df.columns:
        print("\n" + "-" * 60)
        print("Band Touch Statistics:")
        print("-" * 60)
        lower_touches = df["lower_touch"].sum()
        upper_touches = df["upper_touch"].sum()
        print(f"Lower band touches (buy signals): {lower_touches}")
        print(f"Upper band touches (sell signals): {upper_touches}")
    
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

