"""
Backtest script for Momentum Strategy
Runs a full backtest using the MomentumStrategy with performance reporting.
"""

from backtest import Backtester, print_performance_stats, plot_equity_curve, plot_trade_distribution, compare_results
from strategy import MomentumStrategy
import matplotlib.pyplot as plt


def main():
    """
    Run backtest with MomentumStrategy.
    """
    print("=" * 60)
    print("Momentum Strategy Backtest")
    print("=" * 60)
    
    # Initialize momentum strategy
    strategy = MomentumStrategy(top_k=3, rsi_threshold=50)
    
    # Create backtester
    backtester = Backtester(
        strategy=strategy,
        initial_capital=1_000_000.0,
        train_ratio=0.7,  # Use 70% for training, 30% for testing
        orders_per_minute_limit=100,
        position_limit_notional=5_000_000.0,
        matching_engine_seed=42,  # For reproducibility
        audit_log_path="momentum_backtest_order_audit.csv",
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
        plot_equity_curve(result, ax=axes[0], title="Momentum Strategy - Equity Curve")
        
        # Trade distribution
        plot_trade_distribution(result.trades, ax=axes[1], title="Momentum Strategy - Trade Distribution")
        
        plt.tight_layout()
        plt.savefig("momentum_backtest_report.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("Saved momentum_backtest_report.png")
        
    except ImportError:
        print("matplotlib not installed; skipping plots.")
    
    # Parameter sensitivity analysis: compare different top_k values
    print("\n" + "=" * 60)
    print("Parameter Sensitivity Analysis: Testing different top_k values")
    print("=" * 60)
    
    results = []
    labels = []
    
    for top_k in [2, 3, 5]:
        print(f"\nTesting top_k={top_k}...")
        s = MomentumStrategy(top_k=top_k, rsi_threshold=50)
        bt = Backtester(
            s,
            initial_capital=1_000_000.0,
            train_ratio=0.7,
            matching_engine_seed=42,
        )
        r = bt.run("multi_stock_dataset.csv", position_pct_per_name=0.05)
        results.append(r)
        labels.append(f"top_k={top_k}")
    
    # Compare results
    print("\n")
    compare_results(results, labels=labels, plot=True)
    
    try:
        plt.savefig("momentum_parameter_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("Saved momentum_parameter_comparison.png")
    except:
        pass
    
    print("\n" + "=" * 60)
    print("Backtest Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

