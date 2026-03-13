"""
Simple test script for Moving Average Crossover Strategy backtest
Minimal version without plotting dependencies.
"""

from backtest import Backtester, print_performance_stats
from ma_crossover_strategy import MACrossoverStrategy


def main():
    """
    Run a simple backtest with MACrossoverStrategy.
    """
    print("=" * 60)
    print("Moving Average Crossover Strategy Backtest")
    print("=" * 60)
    
    # Initialize MA Crossover strategy
    # Using 5-day and 20-day SMAs (common golden cross/death cross configuration)
    strategy = MACrossoverStrategy(
        short_window=5,      # Short-term MA: 5 periods
        long_window=20,      # Long-term MA: 20 periods
        top_k=3,             # Trade top 3 and bottom 3 stocks
        use_cross_sectional=True,  # Rank stocks cross-sectionally
        ma_type="SMA"        # Use Simple Moving Average
    )
    
    print(f"\nStrategy Configuration:")
    print(f"  - Short MA: {strategy.short_window} periods ({strategy.ma_type})")
    print(f"  - Long MA: {strategy.long_window} periods ({strategy.ma_type})")
    print(f"  - Top K stocks: {strategy.top_k}")
    print(f"  - Cross-sectional mode: {strategy.use_cross_sectional}")
    print(f"\nStrategy Logic:")
    print(f"  - Golden Cross: When {strategy.short_window}-day MA crosses above {strategy.long_window}-day MA -> Buy")
    print(f"  - Death Cross: When {strategy.short_window}-day MA crosses below {strategy.long_window}-day MA -> Sell")
    print(f"  - Cross-sectional: Ranks stocks by crossover strength at each time point")
    
    # Create backtester
    backtester = Backtester(
        strategy=strategy,
        initial_capital=1_000_000.0,  # $1M starting capital
        train_ratio=0.7,  # 70% training, 30% testing
        orders_per_minute_limit=100,
        position_limit_notional=5_000_000.0,
        matching_engine_seed=42,  # For reproducibility
    )
    
    print("\nRunning backtest...")
    print("-" * 60)
    
    # Run backtest
    result = backtester.run(
        data_path="multi_stock_dataset.csv",
        position_pct_per_name=0.05,  # 5% of equity per stock position
        min_order_size=1.0,
    )
    
    # Print performance statistics
    print("\n")
    print_performance_stats(result)
    
    # Show some sample trades
    if result.trades:
        print("\n" + "-" * 60)
        print("Sample Trades (first 10):")
        print("-" * 60)
        for i, trade in enumerate(result.trades[:10]):
            side_str = "BUY" if trade.get('side') == 'BID' else "SELL"
            print(f"Trade {i+1}: {trade.get('ticker', 'N/A')} | "
                  f"{side_str} | "
                  f"Price: ${trade.get('price', 0):.2f} | "
                  f"Size: {trade.get('size', 0):.2f} | "
                  f"PnL: ${trade.get('pnl', 0):.2f}")
    
    print("\n" + "=" * 60)
    print("Backtest Complete!")
    print("=" * 60)
    
    return result


if __name__ == "__main__":
    result = main()

