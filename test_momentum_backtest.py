"""
Simple test script for Momentum Strategy backtest
Minimal version without plotting dependencies.
"""

from backtest import Backtester, print_performance_stats
from strategy import MomentumStrategy


def main():
    """
    Run a simple backtest with MomentumStrategy.
    """
    print("=" * 60)
    print("Momentum Strategy Backtest")
    print("=" * 60)
    
    # Initialize momentum strategy
    # top_k=3 means we'll long the top 3 and short the bottom 3 stocks
    strategy = MomentumStrategy(top_k=5, rsi_threshold=30)
    
    print(f"\nStrategy Configuration:")
    print(f"  - Top K stocks to trade: {strategy.top_k}")
    print(f"  - RSI threshold: {strategy.rsi_threshold}")
    print(f"  - Momentum lookback: {strategy.momentum_lookback}")
    
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
            print(f"Trade {i+1}: {trade.get('ticker', 'N/A')} | "
                  f"{trade.get('side', 'N/A')} | "
                  f"Price: ${trade.get('price', 0):.2f} | "
                  f"Size: {trade.get('size', 0):.2f} | "
                  f"PnL: ${trade.get('pnl', 0):.2f}")
    
    print("\n" + "=" * 60)
    print("Backtest Complete!")
    print("=" * 60)
    
    return result


if __name__ == "__main__":
    result = main()

