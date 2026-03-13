"""
Backtest script for Ranking Model Strategy
Runs a full backtest using the RankingModelStrategy (LGBMRanker) with performance reporting.
Also compares with the regression-based RankingStrategy.
"""

from backtest import Backtester, print_performance_stats, plot_equity_curve, plot_trade_distribution, compare_results
from strategy import RankingModelStrategy, RankingStrategy
import matplotlib.pyplot as plt


def main():
    """
    Run backtest with RankingModelStrategy and compare with RankingStrategy.
    """
    print("=" * 60)
    print("Ranking Model Strategy Backtest")
    print("=" * 60)
    
    # Initialize Ranking Model strategy (using LGBMRanker)
    strategy = RankingModelStrategy(top_k=3)
    
    print(f"\nStrategy Configuration:")
    print(f"  - Model: LightGBM LGBMRanker")
    print(f"  - Objective: lambdarank (LambdaRank)")
    print(f"  - Metric: NDCG (Normalized Discounted Cumulative Gain)")
    print(f"  - Top K stocks: {strategy.top_k}")
    print(f"  - Features: {len(strategy.features)} features")
    
    # Create backtester
    backtester = Backtester(
        strategy=strategy,
        initial_capital=1_000_000.0,
        train_ratio=0.7,  # Use 70% for training, 30% for testing
        orders_per_minute_limit=100,
        position_limit_notional=5_000_000.0,
        matching_engine_seed=42,  # For reproducibility
        audit_log_path="ranking_model_backtest_order_audit.csv",
    )
    
    print("\nRunning backtest...")
    print("-" * 60)
    print("Note: Ranking model groups stocks by date for training.")
    
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
            title="Ranking Model Strategy (LGBMRanker) - Equity Curve"
        )
        
        # Trade distribution
        plot_trade_distribution(
            result.trades, 
            ax=axes[1], 
            title="Ranking Model Strategy - Trade Distribution"
        )
        
        plt.tight_layout()
        plt.savefig("ranking_model_backtest_report.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("Saved ranking_model_backtest_report.png")
        
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
        try:
            s = RankingModelStrategy(top_k=top_k)
            bt = Backtester(
                s,
                initial_capital=1_000_000.0,
                train_ratio=0.7,
                matching_engine_seed=42,
            )
            r = bt.run("multi_stock_dataset.csv", position_pct_per_name=0.05)
            results.append(r)
            labels.append(f"top_k={top_k}")
        except Exception as e:
            print(f"Error testing top_k={top_k}: {e}")
            continue
    
    # Compare results
    if results:
        print("\n")
        compare_results(results, labels=labels, plot=True)
        
        try:
            plt.savefig("ranking_model_parameter_comparison.png", dpi=150, bbox_inches='tight')
            plt.close()
            print("Saved ranking_model_parameter_comparison.png")
        except:
            pass
    
    # Compare Ranking Model vs Regression-based Ranking Strategy
    print("\n" + "=" * 60)
    print("Comparing Ranking Model vs Regression-based Ranking Strategy")
    print("=" * 60)
    
    comparison_results = []
    comparison_labels = []
    
    # Test Ranking Model Strategy
    print("\nTesting RankingModelStrategy (LGBMRanker)...")
    try:
        s_rank = RankingModelStrategy(top_k=3)
        bt_rank = Backtester(
            s_rank,
            initial_capital=1_000_000.0,
            train_ratio=0.7,
            matching_engine_seed=42,
        )
        r_rank = bt_rank.run("multi_stock_dataset.csv", position_pct_per_name=0.05)
        comparison_results.append(r_rank)
        comparison_labels.append("Ranking Model (LGBMRanker)")
    except Exception as e:
        print(f"Error testing RankingModelStrategy: {e}")
    
    # Test Regression-based Ranking Strategy
    print("\nTesting RankingStrategy (LGBMRegressor)...")
    try:
        s_reg = RankingStrategy(top_k=3)
        bt_reg = Backtester(
            s_reg,
            initial_capital=1_000_000.0,
            train_ratio=0.7,
            matching_engine_seed=42,
        )
        r_reg = bt_reg.run("multi_stock_dataset.csv", position_pct_per_name=0.05)
        comparison_results.append(r_reg)
        comparison_labels.append("Regression (LGBMRegressor)")
    except Exception as e:
        print(f"Error testing RankingStrategy: {e}")
    
    # Compare results
    if len(comparison_results) > 1:
        print("\n")
        compare_results(comparison_results, labels=comparison_labels, plot=True)
        
        try:
            plt.savefig("ranking_vs_regression_comparison.png", dpi=150, bbox_inches='tight')
            plt.close()
            print("Saved ranking_vs_regression_comparison.png")
        except:
            pass
        
        # Print detailed comparison
        print("\n" + "-" * 60)
        print("Detailed Comparison:")
        print("-" * 60)
        for r, label in zip(comparison_results, comparison_labels):
            d = r.to_dict()
            print(f"\n{label}:")
            print(f"  Total Return: {d['total_return']:.2%}")
            print(f"  Sharpe Ratio: {d['sharpe_ratio']:.4f}")
            print(f"  Max Drawdown: {d['max_drawdown']:.2%}")
            print(f"  Win Rate: {d['win_rate']:.2%}")
            print(f"  Num Trades: {d['num_trades']}")
    
    print("\n" + "=" * 60)
    print("Backtest Complete!")
    print("=" * 60)
    print("\nKey Differences:")
    print("  - Ranking Model: Uses LGBMRanker with date grouping")
    print("  - Regression Model: Uses LGBMRegressor (predicts absolute returns)")
    print("  - Ranking Model focuses on relative ordering within each date")
    print("  - Regression Model focuses on predicting return magnitudes")


if __name__ == "__main__":
    main()

