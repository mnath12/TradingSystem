"""
Backtest script for Bollinger Bands Strategy
Runs a full backtest using the BollingerBandsStrategy with performance reporting.
"""

from backtest import Backtester, print_performance_stats, plot_equity_curve, plot_trade_distribution, compare_results
from bollinger_bands_strategy import BollingerBandsStrategy
import matplotlib.pyplot as plt


def main():
    """
    Run backtest with BollingerBandsStrategy.
    """
    print("=" * 60)
    print("Bollinger Bands Strategy Backtest")
    print("=" * 60)
    
    # Initialize Bollinger Bands strategy
    # Using 20-period BB with 2 standard deviations (standard configuration)
    strategy = BollingerBandsStrategy(
        window=30,
        num_std=1.5,
        top_k=5,
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
    
    # Create backtester
    backtester = Backtester(
        strategy=strategy,
        initial_capital=1_000_000.0,
        train_ratio=0.7,  # Use 70% for training, 30% for testing
        orders_per_minute_limit=100,
        position_limit_notional=5_000_000.0,
        matching_engine_seed=42,  # For reproducibility
        audit_log_path="bollinger_bands_backtest_order_audit.csv",
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
            title=f"Bollinger Bands Strategy ({strategy.window} periods, {strategy.num_std} std) - Equity Curve"
        )
        
        # Trade distribution
        plot_trade_distribution(
            result.trades, 
            ax=axes[1], 
            title="Bollinger Bands Strategy - Trade Distribution"
        )
        
        plt.tight_layout()
        plt.savefig("bollinger_bands_backtest_report.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("Saved bollinger_bands_backtest_report.png")
        
    except ImportError:
        print("matplotlib not installed; skipping plots.")
    
    # Parameter sensitivity analysis: compare different BB configurations
    print("\n" + "=" * 60)
    print("Parameter Sensitivity Analysis: Testing different BB configurations")
    print("=" * 60)
    
    results = []
    labels = []
    
    # Test different BB configurations
    bb_configs = [
        (20, 2.0, "20/2.0"),
        (20, 1.5, "20/1.5"),
        (20, 2.5, "20/2.5"),
        (15, 2.0, "15/2.0"),
    ]
    
    for window, num_std, label in bb_configs:
        print(f"\nTesting Window={window}, Std={num_std}...")
        try:
            s = BollingerBandsStrategy(
                window=window,
                num_std=num_std,
                top_k=3,
                use_cross_sectional=True,
                signal_type="touch",
            )
            bt = Backtester(
                s,
                initial_capital=1_000_000.0,
                train_ratio=0.7,
                matching_engine_seed=42,
            )
            r = bt.run("multi_stock_dataset.csv", position_pct_per_name=0.05)
            results.append(r)
            labels.append(f"BB_{label}")
        except Exception as e:
            print(f"Error testing {label}: {e}")
            continue
    
    # Compare results
    if results:
        print("\n")
        compare_results(results, labels=labels, plot=True)
        
        try:
            plt.savefig("bollinger_bands_parameter_comparison.png", dpi=150, bbox_inches='tight')
            plt.close()
            print("Saved bollinger_bands_parameter_comparison.png")
        except:
            pass
    
    # Compare different signal types
    print("\n" + "=" * 60)
    print("Comparing Different Signal Types")
    print("=" * 60)
    
    signal_results = []
    signal_labels = []
    
    for signal_type in ["touch", "position", "bounce"]:
        print(f"\nTesting signal_type={signal_type}...")
        try:
            s = BollingerBandsStrategy(
                window=20,
                num_std=2.0,
                top_k=3,
                use_cross_sectional=True,
                signal_type=signal_type,
            )
            bt = Backtester(
                s,
                initial_capital=1_000_000.0,
                train_ratio=0.7,
                matching_engine_seed=42,
            )
            r = bt.run("multi_stock_dataset.csv", position_pct_per_name=0.05)
            signal_results.append(r)
            signal_labels.append(f"Signal: {signal_type}")
        except Exception as e:
            print(f"Error testing {signal_type}: {e}")
            continue
    
    if signal_results:
        print("\n")
        compare_results(signal_results, labels=signal_labels, plot=True)
        
        try:
            plt.savefig("bollinger_bands_signal_type_comparison.png", dpi=150, bbox_inches='tight')
            plt.close()
            print("Saved bollinger_bands_signal_type_comparison.png")
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
            s = BollingerBandsStrategy(
                window=20,
                num_std=2.0,
                top_k=3,
                use_cross_sectional=use_cs,
                signal_type="touch",
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
            plt.savefig("bollinger_bands_mode_comparison.png", dpi=150, bbox_inches='tight')
            plt.close()
            print("Saved bollinger_bands_mode_comparison.png")
        except:
            pass
    
    print("\n" + "=" * 60)
    print("Backtest Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

