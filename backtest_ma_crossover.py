"""
Backtest script for Moving Average Crossover Strategy
Runs a full backtest using the MACrossoverStrategy with performance reporting.
"""

from backtest import Backtester, print_performance_stats, plot_equity_curve, plot_trade_distribution, compare_results
from ma_crossover_strategy import MACrossoverStrategy
import matplotlib.pyplot as plt


def main():
    """
    Run backtest with MACrossoverStrategy.
    """
    print("=" * 60)
    print("Moving Average Crossover Strategy Backtest")
    print("=" * 60)
    
    # Initialize MA Crossover strategy
    # Using 5-day and 20-day SMAs (common configuration)
    strategy = MACrossoverStrategy(
        short_window=5,
        long_window=20,
        top_k=3,
        use_cross_sectional=True,  # Rank stocks cross-sectionally
        ma_type="SMA"
    )
    
    print(f"\nStrategy Configuration:")
    print(f"  - Short MA: {strategy.short_window} periods ({strategy.ma_type})")
    print(f"  - Long MA: {strategy.long_window} periods ({strategy.ma_type})")
    print(f"  - Top K stocks: {strategy.top_k}")
    print(f"  - Cross-sectional mode: {strategy.use_cross_sectional}")
    
    # Create backtester
    backtester = Backtester(
        strategy=strategy,
        initial_capital=1_000_000.0,
        train_ratio=0.7,  # Use 70% for training, 30% for testing
        orders_per_minute_limit=100,
        position_limit_notional=5_000_000.0,
        matching_engine_seed=42,  # For reproducibility
        audit_log_path="ma_crossover_backtest_order_audit.csv",
    )
    
    print("\nRunning backtest...")
    print("-" * 60)
    
    # Run backtest
    result = backtester.run(
        data_path="multi_stock_dataset.csv",
        position_pct_per_name=0.05,  # 5% of equity per stock
        min_order_size=1.0,
    )
    
    # Print performance statistics
    print("\n")
    print_performance_stats(result)
    
    # Generate plots
    try:
        print("\nGenerating performance plots...")
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Equity curve
        plot_equity_curve(
            result, 
            ax=axes[0], 
            title=f"MA Crossover Strategy ({strategy.short_window}/{strategy.long_window}) - Equity Curve"
        )
        
        # Trade distribution
        plot_trade_distribution(
            result.trades, 
            ax=axes[1], 
            title="MA Crossover Strategy - Trade Distribution"
        )
        
        plt.tight_layout()
        plt.savefig("ma_crossover_backtest_report.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("Saved ma_crossover_backtest_report.png")
        
    except ImportError:
        print("matplotlib not installed; skipping plots.")
    
    # Parameter sensitivity analysis: compare different MA configurations
    print("\n" + "=" * 60)
    print("Parameter Sensitivity Analysis: Testing different MA configurations")
    print("=" * 60)
    
    results = []
    labels = []
    
    # Test different MA combinations
    ma_configs = [
        (5, 20, "5/20 SMA"),
        (10, 30, "10/30 SMA"),
        (10, 50, "10/50 SMA"),
    ]
    
    for short, long_win, label in ma_configs:
        print(f"\nTesting {label}...")
        try:
            s = MACrossoverStrategy(
                short_window=short,
                long_window=long_win,
                top_k=3,
                use_cross_sectional=True,
                ma_type="SMA"
            )
            bt = Backtester(
                s,
                initial_capital=1_000_000.0,
                train_ratio=0.7,
                matching_engine_seed=42,
            )
            r = bt.run("multi_stock_dataset.csv", position_pct_per_name=0.05)
            results.append(r)
            labels.append(label)
        except Exception as e:
            print(f"Error testing {label}: {e}")
            continue
    
    # Compare results
    if results:
        print("\n")
        compare_results(results, labels=labels, plot=True)
        
        try:
            plt.savefig("ma_crossover_parameter_comparison.png", dpi=150, bbox_inches='tight')
            plt.close()
            print("Saved ma_crossover_parameter_comparison.png")
        except:
            pass
    
    # Compare cross-sectional vs absolute mode
    print("\n" + "=" * 60)
    print("Comparing Cross-Sectional vs Absolute Mode")
    print("=" * 60)
    
    mode_results = []
    mode_labels = []
    
    for use_cs in [True, False]:
        mode_name = "Cross-Sectional" if use_cs else "Absolute"
        print(f"\nTesting {mode_name} mode...")
        try:
            s = MACrossoverStrategy(
                short_window=10,
                long_window=30,
                top_k=3,
                use_cross_sectional=use_cs,
                ma_type="SMA"
            )
            bt = Backtester(
                s,
                initial_capital=1_000_000.0,
                train_ratio=0.7,
                matching_engine_seed=42,
            )
            r = bt.run("multi_stock_dataset.csv", position_pct_per_name=0.05)
            mode_results.append(r)
            mode_labels.append(mode_name)
        except Exception as e:
            print(f"Error testing {mode_name} mode: {e}")
            continue
    
    if mode_results:
        print("\n")
        compare_results(mode_results, labels=mode_labels, plot=True)
        
        try:
            plt.savefig("ma_crossover_mode_comparison.png", dpi=150, bbox_inches='tight')
            plt.close()
            print("Saved ma_crossover_mode_comparison.png")
        except:
            pass
    
    print("\n" + "=" * 60)
    print("Backtest Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

