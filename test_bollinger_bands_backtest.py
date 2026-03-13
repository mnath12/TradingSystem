"""
Simple test script for Bollinger Bands Strategy backtest
Minimal version without plotting dependencies.
"""

from backtest import Backtester, print_performance_stats
from bollinger_bands_strategy import BollingerBandsStrategy


def main():
    """
    Run a simple backtest with BollingerBandsStrategy.
    """
    print("=" * 60)
    print("Bollinger Bands Strategy Backtest")
    print("=" * 60)
    
    # Initialize Bollinger Bands strategy
    # Using 20-period BB with 2 standard deviations (standard configuration)
    strategy = BollingerBandsStrategy(
        window=20,           # 20-period moving average
        num_std=2.0,          # 2 standard deviations for bands
        top_k=3,              # Trade top 3 and bottom 3 stocks
        use_cross_sectional=True,  # Rank stocks cross-sectionally
        signal_type="touch",  # Signal when price touches bands
        min_band_width_pct=0.0  # No minimum band width filter
    )
    
    print(f"\nStrategy Configuration:")
    print(f"  - Window: {strategy.window} periods")
    print(f"  - Number of Std Devs: {strategy.num_std}")
    print(f"  - Top K stocks: {strategy.top_k}")
    print(f"  - Cross-sectional mode: {strategy.use_cross_sectional}")
    print(f"  - Signal type: {strategy.signal_type}")
    print(f"\nStrategy Logic:")
    print(f"  - Lower Band Touch: Price touches/goes below lower band -> Buy (oversold)")
    print(f"  - Upper Band Touch: Price touches/goes above upper band -> Sell (overbought)")
    print(f"  - Cross-sectional: Ranks stocks by BB signal strength at each time point")
    
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

