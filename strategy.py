"""
ML Ranking Trading Strategy
Uses cross-sectional ranking of predicted returns to generate signals.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
import lightgbm as lgb


class RankingStrategy:
    """
    Cross-sectional ML ranking strategy.
    """

    def __init__(self, model=None, top_k=3):

        self.model = model if model is not None else GradientBoostingRegressor(
            n_estimators=700,
            max_depth=15,
            learning_rate=0.05,
        )


        self.top_k = top_k

        self.features = [
            "Return_1",
            "Return_5",
            "Return_10",
            "Momentum_20",
            "MA_ratio",
            "Volatility_10",
            "RSI",
            "CS_Return_Rank",
            "CS_Vol_Rank"
        ]

        self.target = "Target"

    # ---------------------------------------------------
    # Data loading
    # ---------------------------------------------------

    def load_data(self, filepath):

        df = pd.read_csv(filepath)

        df["Datetime"] = pd.to_datetime(df["Datetime"])

        return df

    # ---------------------------------------------------
    # Cross-sectional normalization
    # ---------------------------------------------------

    def normalize_features(self, df):

        df = df.copy()

        df[self.features] = df.groupby("Datetime")[self.features].transform(
            lambda x: (x - x.mean()) / x.std()
        )

        return df

    # ---------------------------------------------------
    # Train ML model
    # ---------------------------------------------------

    def train(self, df):

        X = df[self.features]
        y = df[self.target]

        self.model.fit(X, y)

    # ---------------------------------------------------
    # Prediction
    # ---------------------------------------------------

    def predict(self, df):

        df = df.copy()

        df["prediction"] = self.model.predict(df[self.features])

        return df

    # ---------------------------------------------------
    # Generate trading signals
    # ---------------------------------------------------

    def generate_signals(self, df):

        df = df.copy()

        signals = []

        for date, group in df.groupby("Datetime"):

            group = group.copy()

            group["rank"] = group["prediction"].rank()

            group["signal"] = 0

            long_idx = group["rank"] >= len(group) - self.top_k + 1
            short_idx = group["rank"] <= self.top_k

            group.loc[long_idx, "signal"] = 1
            group.loc[short_idx, "signal"] = -1

            signals.append(group)

        df = pd.concat(signals)

        return df

    # ---------------------------------------------------
    # Position sizing
    # ---------------------------------------------------

    def position_sizing(self, df):

        df = df.copy()

        df["weight"] = df["signal"] / self.top_k

        return df

    # ---------------------------------------------------
    # Strategy returns
    # ---------------------------------------------------

    def compute_returns(self, df):

        df = df.copy()

        df["strategy_return"] = df["weight"] * df["Target"]

        portfolio = df.groupby("Datetime")["strategy_return"].sum()

        return portfolio

    # ---------------------------------------------------
    # Sharpe ratio
    # ---------------------------------------------------

    def sharpe_ratio(self, returns):

        mean = returns.mean()
        std = returns.std()

        sharpe = np.sqrt(252) * mean / std

        return sharpe


class RankingModelStrategy:
    """
    Ranking Model Strategy using LightGBM's LGBMRanker.
    
    This strategy treats stock selection as a ranking problem rather than
    a regression problem. Instead of predicting returns directly, it learns
    to rank stocks correctly within each time period (date).
    
    Key Advantages:
    - Better suited for cross-sectional ranking problems
    - Focuses on relative ordering rather than absolute values
    - Uses LightGBM's ranking objective (LambdaRank)
    - Requires grouping by date for proper ranking training
    
    The model learns to rank stocks by their future returns, then we select
    the top K and bottom K stocks for long/short positions.
    """

    def __init__(self, model=None, top_k=3):
        """
        Initialize ranking model strategy.
        
        Args:
            model: Pre-initialized LGBMRanker model (optional)
            top_k: Number of top/bottom stocks to trade (default: 3)
        """
        self.model = model if model else lgb.LGBMRanker(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=10,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='lambdarank',
            metric='ndcg',  # Normalized Discounted Cumulative Gain
            boosting_type='gbdt',
            verbose=-1
        )
        
        self.top_k = top_k
        
        self.features = [
            "Return_1",
            "Return_5",
            "Return_10",
            "Momentum_20",
            "MA_ratio",
            "Volatility_10",
            "RSI",
            "CS_Return_Rank",
            "CS_Vol_Rank"
        ]
        
        self.target = "Target"

    # ---------------------------------------------------
    # Data loading
    # ---------------------------------------------------

    def load_data(self, filepath):
        """Load data from CSV file."""
        df = pd.read_csv(filepath)
        df["Datetime"] = pd.to_datetime(df["Datetime"])
        return df

    # ---------------------------------------------------
    # Cross-sectional normalization
    # ---------------------------------------------------

    def normalize_features(self, df):
        """
        Normalize features cross-sectionally.
        This is important for ranking models to ensure features are comparable
        across stocks at each time point.
        """
        df = df.copy()
        
        # Only normalize features that exist
        available_features = [f for f in self.features if f in df.columns]
        if available_features:
            df[available_features] = df.groupby("Datetime")[available_features].transform(
                lambda x: (x - x.mean()) / (x.std() + 1e-8)  # Add epsilon to avoid division by zero
            )
        
        return df

    # ---------------------------------------------------
    # Prepare data for ranking model
    # ---------------------------------------------------

    def _prepare_ranking_data(self, df):
        """
        Prepare data for LightGBM ranking model.
        
        For ranking models, we need:
        1. Features (X)
        2. Target (y) - MUST be integer relevance scores for LGBMRanker
        3. Groups - list of group sizes (number of stocks per date)
        
        LightGBM's LGBMRanker requires integer labels. We convert returns to
        integer relevance scores (0-4) by ranking stocks within each date group.
        
        Returns:
            X: Feature matrix
            y: Integer relevance scores (0-4 scale based on return quantiles)
            groups: List of group sizes for each date
        """
        # Sort by Datetime to ensure groups are in order
        df = df.sort_values(["Datetime", "Ticker"]).copy()
        
        # Get available features
        available_features = [f for f in self.features if f in df.columns]
        X = df[available_features].fillna(0).values
        
        # Convert returns to integer relevance scores
        # LightGBM ranking requires integer labels, not float returns
        # We'll create relevance scores based on return quantiles within each date
        df["relevance"] = 0
        
        for date, group in df.groupby("Datetime"):
            # Rank returns within this date group
            # Higher return = higher relevance score
            returns = group[self.target].fillna(0)
            
            # Convert to integer relevance scores (0-4 scale)
            # Top 20% = 4, Next 20% = 3, Middle 20% = 2, Next 20% = 1, Bottom 20% = 0
            quantiles = returns.quantile([0.8, 0.6, 0.4, 0.2])
            
            relevance = pd.Series(0, index=group.index)
            relevance[returns >= quantiles[0.8]] = 4  # Top 20%
            relevance[(returns >= quantiles[0.6]) & (returns < quantiles[0.8])] = 3  # Next 20%
            relevance[(returns >= quantiles[0.4]) & (returns < quantiles[0.6])] = 2  # Middle 20%
            relevance[(returns >= quantiles[0.2]) & (returns < quantiles[0.4])] = 1  # Next 20%
            relevance[returns < quantiles[0.2]] = 0  # Bottom 20%
            
            df.loc[group.index, "relevance"] = relevance.values
        
        # Use integer relevance scores as target (required by LGBMRanker)
        y = df["relevance"].astype(int).values
        
        # Groups: number of stocks per date
        groups = df.groupby("Datetime").size().values
        
        return X, y, groups

    # ---------------------------------------------------
    # Train ranking model
    # ---------------------------------------------------

    def train(self, df):
        """
        Train the ranking model.
        
        For LGBMRanker, we need to provide:
        - X: features
        - y: target (integer relevance scores, 0-4)
        - group: list of group sizes (stocks per date)
        
        The model learns to rank stocks within each date group.
        """
        # Prepare data for ranking
        X, y, groups = self._prepare_ranking_data(df)
        
        # Train the ranking model
        # group parameter tells LightGBM which samples belong to the same query/date
        self.model.fit(
            X, y,
            group=groups,
            eval_set=[(X, y)],
            eval_group=[groups],
            eval_names=['train'],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )

    # ---------------------------------------------------
    # Prediction
    # ---------------------------------------------------

    def predict(self, df):
        """
        Generate ranking predictions.
        
        The model outputs relevance scores that can be used to rank stocks.
        Higher scores indicate stocks that should be ranked higher (better returns).
        
        Returns DataFrame with 'prediction' column (ranking scores).
        """
        df = df.copy()
        
        # Sort by Datetime to match training order
        df = df.sort_values(["Datetime", "Ticker"])
        
        # Get available features
        available_features = [f for f in self.features if f in df.columns]
        X = df[available_features].fillna(0).values
        
        # Get predictions (ranking scores)
        predictions = self.model.predict(X)
        
        # Add predictions to dataframe
        df["prediction"] = predictions
        
        return df

    # ---------------------------------------------------
    # Generate trading signals
    # ---------------------------------------------------

    def generate_signals(self, df):
        """
        Generate trading signals based on ranking predictions.
        
        For each time point (date):
        - Ranks stocks by prediction score (higher = better)
        - Long signal (+1) for top K stocks (highest ranking scores)
        - Short signal (-1) for bottom K stocks (lowest ranking scores)
        - No signal (0) for middle stocks
        
        Returns DataFrame with 'signal' column.
        """
        df = df.copy()
        
        signals = []
        
        for date, group in df.groupby("Datetime"):
            group = group.copy()
            
            # Rank stocks by prediction score (higher = better ranking)
            group["rank"] = group["prediction"].rank(ascending=False)
            
            # Initialize signals to zero
            group["signal"] = 0
            
            # Long top K stocks (highest ranking scores)
            long_idx = group["rank"] <= self.top_k
            
            # Short bottom K stocks (lowest ranking scores)
            total_stocks = len(group)
            short_idx = group["rank"] >= (total_stocks - self.top_k + 1)
            
            # Assign signals
            group.loc[long_idx, "signal"] = 1
            group.loc[short_idx, "signal"] = -1
            
            signals.append(group)
        
        df = pd.concat(signals)
        
        return df

    # ---------------------------------------------------
    # Position sizing
    # ---------------------------------------------------

    def position_sizing(self, df):
        """
        Calculate position weights for each signal.
        
        Equal weights: each position gets 1/top_k of the allocation.
        Long positions get positive weight, short positions get negative weight.
        
        Returns DataFrame with 'weight' column.
        """
        df = df.copy()
        
        # Equal weight per position: signal / top_k
        df["weight"] = df["signal"] / self.top_k
        
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


class RankingModelStrategy:
    """
    Ranking Model Strategy using LightGBM's LGBMRanker.
    
    This strategy treats stock selection as a ranking problem rather than
    a regression problem. Instead of predicting returns directly, it learns
    to rank stocks correctly within each time period (date).
    
    Key Advantages:
    - Better suited for cross-sectional ranking problems
    - Focuses on relative ordering rather than absolute values
    - Uses LightGBM's ranking objective (LambdaRank)
    - Requires grouping by date for proper ranking training
    
    The model learns to rank stocks by their future returns, then we select
    the top K and bottom K stocks for long/short positions.
    """

    def __init__(self, model=None, top_k=3):
        """
        Initialize ranking model strategy.
        
        Args:
            model: Pre-initialized LGBMRanker model (optional)
            top_k: Number of top/bottom stocks to trade (default: 3)
        """
        self.model = model if model else lgb.LGBMRanker(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=10,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='lambdarank',
            metric='ndcg',  # Normalized Discounted Cumulative Gain
            boosting_type='gbdt',
            verbose=-1
        )
        
        self.top_k = top_k
        
        self.features = [
            "Return_1",
            "Return_5",
            "Return_10",
            "Momentum_20",
            "MA_ratio",
            "Volatility_10",
            "RSI",
            "CS_Return_Rank",
            "CS_Vol_Rank"
        ]
        
        self.target = "Target"

    # ---------------------------------------------------
    # Data loading
    # ---------------------------------------------------

    def load_data(self, filepath):
        """Load data from CSV file."""
        df = pd.read_csv(filepath)
        df["Datetime"] = pd.to_datetime(df["Datetime"])
        return df

    # ---------------------------------------------------
    # Cross-sectional normalization
    # ---------------------------------------------------

    def normalize_features(self, df):
        """
        Normalize features cross-sectionally.
        This is important for ranking models to ensure features are comparable
        across stocks at each time point.
        """
        df = df.copy()
        
        # Only normalize features that exist
        available_features = [f for f in self.features if f in df.columns]
        if available_features:
            df[available_features] = df.groupby("Datetime")[available_features].transform(
                lambda x: (x - x.mean()) / (x.std() + 1e-8)  # Add epsilon to avoid division by zero
            )
        
        return df

    # ---------------------------------------------------
    # Prepare data for ranking model
    # ---------------------------------------------------

    def _prepare_ranking_data(self, df):
        """
        Prepare data for LightGBM ranking model.
        
        For ranking models, we need:
        1. Features (X)
        2. Target (y) - MUST be integer relevance scores for LGBMRanker
        3. Groups - list of group sizes (number of stocks per date)
        
        LightGBM's LGBMRanker requires integer labels. We convert returns to
        integer relevance scores (0-4) by ranking stocks within each date group.
        
        Returns:
            X: Feature matrix
            y: Integer relevance scores (0-4 scale based on return quantiles)
            groups: List of group sizes for each date
        """
        # Sort by Datetime to ensure groups are in order
        df = df.sort_values(["Datetime", "Ticker"]).copy()
        
        # Get available features
        available_features = [f for f in self.features if f in df.columns]
        X = df[available_features].fillna(0).values
        
        # Convert returns to integer relevance scores
        # LightGBM ranking requires integer labels, not float returns
        # We'll create relevance scores based on return quantiles within each date
        df["relevance"] = 0
        
        for date, group in df.groupby("Datetime"):
            # Rank returns within this date group
            # Higher return = higher relevance score
            returns = group[self.target].fillna(0)
            
            # Convert to integer relevance scores (0-4 scale)
            # Top 20% = 4, Next 20% = 3, Middle 20% = 2, Next 20% = 1, Bottom 20% = 0
            quantiles = returns.quantile([0.8, 0.6, 0.4, 0.2])
            
            relevance = pd.Series(0, index=group.index)
            relevance[returns >= quantiles[0.8]] = 4  # Top 20%
            relevance[(returns >= quantiles[0.6]) & (returns < quantiles[0.8])] = 3  # Next 20%
            relevance[(returns >= quantiles[0.4]) & (returns < quantiles[0.6])] = 2  # Middle 20%
            relevance[(returns >= quantiles[0.2]) & (returns < quantiles[0.4])] = 1  # Next 20%
            relevance[returns < quantiles[0.2]] = 0  # Bottom 20%
            
            df.loc[group.index, "relevance"] = relevance.values
        
        # Use integer relevance scores as target (required by LGBMRanker)
        y = df["relevance"].astype(int).values
        
        # Groups: number of stocks per date
        groups = df.groupby("Datetime").size().values
        
        return X, y, groups

    # ---------------------------------------------------
    # Train ranking model
    # ---------------------------------------------------

    def train(self, df):
        """
        Train the ranking model.
        
        For LGBMRanker, we need to provide:
        - X: features
        - y: target (integer relevance scores, 0-4)
        - group: list of group sizes (stocks per date)
        
        The model learns to rank stocks within each date group.
        """
        # Prepare data for ranking
        X, y, groups = self._prepare_ranking_data(df)
        
        # Train the ranking model
        # group parameter tells LightGBM which samples belong to the same query/date
        self.model.fit(
            X, y,
            group=groups,
            eval_set=[(X, y)],
            eval_group=[groups],
            eval_names=['train'],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )

    # ---------------------------------------------------
    # Prediction
    # ---------------------------------------------------

    def predict(self, df):
        """
        Generate ranking predictions.
        
        The model outputs relevance scores that can be used to rank stocks.
        Higher scores indicate stocks that should be ranked higher (better returns).
        
        Returns DataFrame with 'prediction' column (ranking scores).
        """
        df = df.copy()
        
        # Sort by Datetime to match training order
        df = df.sort_values(["Datetime", "Ticker"])
        
        # Get available features
        available_features = [f for f in self.features if f in df.columns]
        X = df[available_features].fillna(0).values
        
        # Get predictions (ranking scores)
        predictions = self.model.predict(X)
        
        # Add predictions to dataframe
        df["prediction"] = predictions
        
        return df

    # ---------------------------------------------------
    # Generate trading signals
    # ---------------------------------------------------

    def generate_signals(self, df):
        """
        Generate trading signals based on ranking predictions.
        
        For each time point (date):
        - Ranks stocks by prediction score (higher = better)
        - Long signal (+1) for top K stocks (highest ranking scores)
        - Short signal (-1) for bottom K stocks (lowest ranking scores)
        - No signal (0) for middle stocks
        
        Returns DataFrame with 'signal' column.
        """
        df = df.copy()
        
        signals = []
        
        for date, group in df.groupby("Datetime"):
            group = group.copy()
            
            # Rank stocks by prediction score (higher = better ranking)
            group["rank"] = group["prediction"].rank(ascending=False)
            
            # Initialize signals to zero
            group["signal"] = 0
            
            # Long top K stocks (highest ranking scores)
            long_idx = group["rank"] <= self.top_k
            
            # Short bottom K stocks (lowest ranking scores)
            total_stocks = len(group)
            short_idx = group["rank"] >= (total_stocks - self.top_k + 1)
            
            # Assign signals
            group.loc[long_idx, "signal"] = 1
            group.loc[short_idx, "signal"] = -1
            
            signals.append(group)
        
        df = pd.concat(signals)
        
        return df

    # ---------------------------------------------------
    # Position sizing
    # ---------------------------------------------------

    def position_sizing(self, df):
        """
        Calculate position weights for each signal.
        
        Equal weights: each position gets 1/top_k of the allocation.
        Long positions get positive weight, short positions get negative weight.
        
        Returns DataFrame with 'weight' column.
        """
        df = df.copy()
        
        # Equal weight per position: signal / top_k
        df["weight"] = df["signal"] / self.top_k
        
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


class MomentumStrategy:
    """
    Momentum-based trading strategy.
    
    This strategy uses momentum indicators to identify stocks with strong
    upward or downward price trends. It ranks stocks by their momentum score
    and takes long positions in high momentum stocks and short positions in
    low momentum stocks.
    
    Key Components:
    - Uses multiple momentum indicators (returns, moving averages, RSI)
    - Combines indicators into a composite momentum score
    - Cross-sectional ranking: compares stocks at each time point
    - Long top performers, short bottom performers
    """

    def __init__(self, top_k=3, momentum_lookback=20, rsi_threshold=50):
        """
        Initialize momentum strategy.
        
        Args:
            top_k: Number of top/bottom stocks to trade (default: 3)
            momentum_lookback: Lookback period for momentum calculation (default: 20)
            rsi_threshold: RSI threshold for momentum confirmation (default: 50)
        """
        self.top_k = top_k
        self.momentum_lookback = momentum_lookback
        self.rsi_threshold = rsi_threshold
        
        # Features used for momentum calculation
        self.features = [
            "Return_1",
            "Return_5",
            "Return_10",
            "Momentum_20",
            "MA_ratio",
            "RSI",
            "CS_Return_Rank",
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
    # Feature normalization (optional for momentum)
    # ---------------------------------------------------

    def normalize_features(self, df):
        """
        Normalize features cross-sectionally (optional for momentum strategy).
        This helps compare momentum across different stocks at each time point.
        """
        df = df.copy()
        
        # Only normalize if features exist
        available_features = [f for f in self.features if f in df.columns]
        if available_features:
            df[available_features] = df.groupby("Datetime")[available_features].transform(
                lambda x: (x - x.mean()) / (x.std() + 1e-8)  # Add small epsilon to avoid division by zero
            )
        
        return df

    # ---------------------------------------------------
    # Train (no-op for momentum strategy)
    # ---------------------------------------------------

    def train(self, df):
        """
        Training method for compatibility with backtest framework.
        Momentum strategy doesn't require training, but we validate data here.
        """
        # Validate that required features exist
        missing_features = [f for f in self.features if f not in df.columns]
        if missing_features:
            print(f"Warning: Missing features {missing_features}. Strategy may not work optimally.")
        # No actual training needed for momentum strategy
        pass

    # ---------------------------------------------------
    # Prediction: Calculate momentum score
    # ---------------------------------------------------

    def predict(self, df):
        """
        Calculate momentum score for each stock.
        
        The momentum score combines:
        - Short-term returns (Return_1, Return_5, Return_10)
        - Medium-term momentum (Momentum_20)
        - Moving average ratio (MA_ratio) - indicates trend strength
        - RSI - momentum confirmation
        - Cross-sectional return rank - relative performance
        
        Returns DataFrame with 'prediction' column (momentum score).
        """
        df = df.copy()
        
        # Initialize prediction column
        df["prediction"] = 0.0
        
        # Calculate momentum score for each stock at each time point
        for date, group in df.groupby("Datetime"):
            group = group.copy()
            
            # Start with zero momentum score
            momentum_score = pd.Series(0.0, index=group.index)
            
            # Weight different momentum indicators
            # Short-term returns (recent performance)
            if "Return_1" in group.columns:
                momentum_score += 0.15 * group["Return_1"].fillna(0)
            if "Return_5" in group.columns:
                momentum_score += 0.25 * group["Return_5"].fillna(0)
            if "Return_10" in group.columns:
                momentum_score += 0.20 * group["Return_10"].fillna(0)
            
            # Medium-term momentum
            if "Momentum_20" in group.columns:
                momentum_score += 0.20 * group["Momentum_20"].fillna(0)
            
            # Moving average ratio (trend strength)
            # MA_ratio > 1 means price above short-term MA (bullish)
            if "MA_ratio" in group.columns:
                ma_signal = (group["MA_ratio"].fillna(1) - 1) * 0.10
                momentum_score += ma_signal
            
            # RSI momentum confirmation
            # RSI > 50 is bullish, RSI < 50 is bearish
            if "RSI" in group.columns:
                rsi_signal = ((group["RSI"].fillna(50) - self.rsi_threshold) / 50) * 0.05
                momentum_score += rsi_signal
            
            # Cross-sectional return rank (relative performance)
            # Higher rank = better relative performance
            if "CS_Return_Rank" in group.columns:
                # Normalize rank to [-1, 1] range
                rank_normalized = (group["CS_Return_Rank"].fillna(0.5) - 0.5) * 2
                momentum_score += 0.05 * rank_normalized
            
            # Store prediction (momentum score)
            df.loc[group.index, "prediction"] = momentum_score.values
        
        return df

    # ---------------------------------------------------
    # Generate trading signals
    # ---------------------------------------------------

    def generate_signals(self, df):
        """
        Generate trading signals based on momentum ranking.
        
        For each time point:
        - Ranks stocks by momentum score (prediction)
        - Long signal (+1) for top K stocks (highest momentum)
        - Short signal (-1) for bottom K stocks (lowest momentum)
        - No signal (0) for middle stocks
        
        Returns DataFrame with 'signal' column.
        """
        df = df.copy()
        
        signals = []
        
        for date, group in df.groupby("Datetime"):
            group = group.copy()
            
            # Rank stocks by momentum score (higher = better)
            group["rank"] = group["prediction"].rank(ascending=False)
            
            # Initialize signals to zero
            group["signal"] = 0
            
            # Long top K stocks (highest momentum)
            # Rank 1 is best, so we want ranks <= top_k
            long_idx = group["rank"] <= self.top_k
            
            # Short bottom K stocks (lowest momentum)
            # Rank N is worst, so we want ranks >= (N - top_k + 1)
            total_stocks = len(group)
            short_idx = group["rank"] >= (total_stocks - self.top_k + 1)
            
            # Assign signals
            group.loc[long_idx, "signal"] = 1
            group.loc[short_idx, "signal"] = -1
            
            signals.append(group)
        
        df = pd.concat(signals)
        
        return df

    # ---------------------------------------------------
    # Position sizing
    # ---------------------------------------------------

    def position_sizing(self, df):
        """
        Calculate position weights for each signal.
        
        Equal weights: each position gets 1/top_k of the allocation.
        Long positions get positive weight, short positions get negative weight.
        
        Returns DataFrame with 'weight' column.
        """
        df = df.copy()
        
        # Equal weight per position: signal / top_k
        # This ensures total absolute weight = 1.0 (0.5 long + 0.5 short if top_k stocks each)
        df["weight"] = df["signal"] / self.top_k
        
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


# -------------------------------------------------------
# Example execution
# -------------------------------------------------------

def main():
    """
    Example usage of RankingStrategy and RankingModelStrategy.
    """
    print("=" * 60)
    print("Ranking Strategy Example")
    print("=" * 60)
    
    # Example 1: Original RankingStrategy (regression-based)
    print("\n[1] Testing RankingStrategy (Regression-based)...")
    strategy_reg = RankingStrategy(top_k=3)
    
    print("Loading dataset...")
    df = strategy_reg.load_data("multi_stock_dataset.csv")
    df = strategy_reg.normalize_features(df)
    
    # Train/test split (time-based)
    split_date = df["Datetime"].quantile(0.8)
    train = df[df["Datetime"] <= split_date]
    test = df[df["Datetime"] > split_date]
    
    print("Training regression model...")
    strategy_reg.train(train)
    
    print("Generating predictions...")
    test_reg = strategy_reg.predict(test)
    test_reg = strategy_reg.generate_signals(test_reg)
    test_reg = strategy_reg.position_sizing(test_reg)
    
    returns_reg = strategy_reg.compute_returns(test_reg)
    sharpe_reg = strategy_reg.sharpe_ratio(returns_reg)
    
    print(f"Regression Strategy - Total Return: {returns_reg.sum():.4f}, Sharpe: {sharpe_reg:.4f}")
    
    # Example 2: RankingModelStrategy (ranking-based)
    print("\n[2] Testing RankingModelStrategy (Ranking-based)...")
    strategy_rank = RankingModelStrategy(top_k=3)
    
    print("Loading dataset...")
    df_rank = strategy_rank.load_data("multi_stock_dataset.csv")
    df_rank = strategy_rank.normalize_features(df_rank)
    
    # Train/test split (time-based)
    train_rank = df_rank[df_rank["Datetime"] <= split_date]
    test_rank = df_rank[df_rank["Datetime"] > split_date]
    
    print("Training ranking model (with date grouping)...")
    strategy_rank.train(train_rank)
    
    print("Generating ranking predictions...")
    test_rank = strategy_rank.predict(test_rank)
    test_rank = strategy_rank.generate_signals(test_rank)
    test_rank = strategy_rank.position_sizing(test_rank)
    
    returns_rank = strategy_rank.compute_returns(test_rank)
    sharpe_rank = strategy_rank.sharpe_ratio(returns_rank)
    
    print(f"Ranking Strategy - Total Return: {returns_rank.sum():.4f}, Sharpe: {sharpe_rank:.4f}")
    
    print("\n" + "=" * 60)
    print("Comparison:")
    print("=" * 60)
    print(f"Regression Strategy: Return={returns_reg.sum():.4f}, Sharpe={sharpe_reg:.4f}")
    print(f"Ranking Strategy:    Return={returns_rank.sum():.4f}, Sharpe={sharpe_rank:.4f}")
    
    print("\nFirst few signals from Ranking Model:")
    print(test_rank[["Datetime", "Ticker", "prediction", "rank", "signal"]].head())


if __name__ == "__main__":
    main()