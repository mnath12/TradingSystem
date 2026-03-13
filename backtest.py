"""
Strategy Backtesting
Evaluates trading strategy using historical data, Gateway, OrderManager,
MatchingEngine, and OrderGateway. Tracks performance and reports metrics and plots.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from gateway import DataGateway
from order_manager import OrderManager, ValidationResult
from order_gateway import OrderGateway
from matching_engine import MatchingEngine, ExecutionStatus, ExecutionResult
from order_book import Side


# ---------------------------------------------------------------------------
# Strategy interface: any strategy with train + signal DataFrame (Datetime, Ticker, signal, weight, Close)
# ---------------------------------------------------------------------------

def _normalize_dt_key(dt: Any) -> str:
    """Normalize datetime for lookup (e.g. from CSV string or pandas Timestamp)."""
    if hasattr(dt, "strftime"):
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    return str(dt).replace("+00:00", "").strip()[:19]


# ---------------------------------------------------------------------------
# Backtest result and metrics
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    """Result of a single backtest run."""
    initial_capital: float
    final_equity: float
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    num_trades: int
    num_wins: int
    num_losses: int
    equity_curve: pd.Series
    trades: List[Dict[str, Any]]
    params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "initial_capital": self.initial_capital,
            "final_equity": self.final_equity,
            "total_return": self.total_return,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "num_trades": self.num_trades,
            "num_wins": self.num_wins,
            "num_losses": self.num_losses,
            "params": self.params,
        }


def compute_metrics(
    equity_curve: pd.Series,
    trades: List[Dict[str, Any]],
    initial_capital: float,
) -> Dict[str, float]:
    """Compute P&L, Sharpe, drawdown, win/loss from equity curve and trades list."""
    if equity_curve is None or len(equity_curve) == 0:
        return {
            "total_return": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "num_trades": 0,
            "num_wins": 0,
            "num_losses": 0,
            "final_equity": initial_capital,
        }
    eq = equity_curve.dropna()
    if len(eq) < 2:
        total_return = (eq.iloc[-1] - initial_capital) / initial_capital if len(eq) else 0.0
        return {
            "total_return": total_return,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "num_trades": len(trades),
            "num_wins": 0,
            "num_losses": 0,
            "final_equity": float(eq.iloc[-1]) if len(eq) else initial_capital,
        }
    returns = eq.pct_change().dropna()
    if returns.std() == 0:
        sharpe = 0.0
    else:
        sharpe = np.sqrt(252) * returns.mean() / returns.std()
    peak = eq.expanding(min_periods=1).max()
    drawdown = (peak - eq) / peak.replace(0, np.nan)
    max_dd = drawdown.max() if len(drawdown) else 0.0

    final_equity = float(eq.iloc[-1])
    total_return = (final_equity - initial_capital) / initial_capital

    # Win/loss: from trade PnL if present, else from bar returns (positive return = win)
    wins = sum(1 for t in trades if t.get("pnl", 0) > 0)
    losses = sum(1 for t in trades if t.get("pnl", 0) < 0)
    num_trades = len(trades)
    if num_trades and (wins + losses) > 0:
        win_rate = wins / (wins + losses)
        gross_profit = sum(t.get("pnl", 0) for t in trades if t.get("pnl", 0) > 0)
        gross_loss = abs(sum(t.get("pnl", 0) for t in trades if t.get("pnl", 0) < 0))
        profit_factor = gross_profit / gross_loss if gross_loss else (float("inf") if gross_profit else 0.0)
    else:
        # Fallback: win rate from positive bar returns
        pos_returns = (returns > 0).sum()
        win_rate = pos_returns / len(returns) if len(returns) else 0.0
        profit_factor = 0.0

    return {
        "total_return": total_return,
        "sharpe_ratio": float(sharpe),
        "max_drawdown": float(max_dd),
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "num_trades": num_trades,
        "num_wins": wins,
        "num_losses": losses,
        "final_equity": final_equity,
    }


# ---------------------------------------------------------------------------
# Backtester
# ---------------------------------------------------------------------------

class Backtester:
    """
    Integrates Gateway, OrderManager, MatchingEngine, and OrderGateway
    to backtest a user-defined strategy on historical data.
    """

    def __init__(
        self,
        strategy: Any,
        initial_capital: float = 1_000_000.0,
        train_ratio: float = 0.7,
        orders_per_minute_limit: int = 100,
        position_limit_notional: float = 5_000_000.0,
        matching_engine_seed: Optional[int] = None,
        audit_log_path: Optional[str] = None,
    ):
        """
        Args:
            strategy: Strategy instance (e.g. RankingStrategy) with train(), predict(), generate_signals(), position_sizing();
                      and attribute 'features' for required columns.
            initial_capital: Starting cash.
            train_ratio: Fraction of data used for training (time-based split).
            orders_per_minute_limit: OrderManager limit.
            position_limit_notional: Max buy/sell position notional for OrderManager.
            matching_engine_seed: Random seed for MatchingEngine.
            audit_log_path: If set, OrderGateway writes audit log to this file.
        """
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.train_ratio = train_ratio
        self.orders_per_minute_limit = orders_per_minute_limit
        self.position_limit_notional = position_limit_notional
        self.matching_engine_seed = matching_engine_seed
        self.audit_log_path = audit_log_path

    def _prepare_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run strategy predict -> generate_signals -> position_sizing. Returns df with signal, weight, Close."""
        df = df.copy()
        if "prediction" not in df.columns:
            df = self.strategy.predict(df)
        if "signal" not in df.columns:
            df = self.strategy.generate_signals(df)
        if "weight" not in df.columns:
            df = self.strategy.position_sizing(df)
        return df

    def _build_signal_lookup(self, signals_df: pd.DataFrame) -> Dict[Tuple[str, str], Dict[str, Any]]:
        """Build (datetime_key, ticker) -> {signal, weight, Close}."""
        lookup = {}
        for _, row in signals_df.iterrows():
            dt_key = _normalize_dt_key(row["Datetime"])
            ticker = row.get("Ticker", "")
            close = row.get("Close")
            if pd.isna(close):
                continue
            lookup[(dt_key, ticker)] = {
                "signal": row.get("signal", 0),
                "weight": row.get("weight", 0),
                "Close": float(close),
            }
        return lookup

    def run(
        self,
        data_path: str,
        position_pct_per_name: float = 0.05,
        min_order_size: float = 1.0,
    ) -> BacktestResult:
        """
        Run backtest: load data, train strategy, stream test data via Gateway,
        generate orders from signals, execute via MatchingEngine, track P&L and equity.

        Args:
            data_path: Path to cleaned CSV (e.g. multi_stock_dataset.csv).
            position_pct_per_name: Target position size as fraction of current equity per name (per signal).
            min_order_size: Minimum order size in shares.

        Returns:
            BacktestResult with equity curve, trades, and metrics.
        """
        # Load and prepare data
        if hasattr(self.strategy, "load_data"):
            df = self.strategy.load_data(data_path)
        else:
            df = pd.read_csv(data_path)
            if "Datetime" in df.columns:
                df["Datetime"] = pd.to_datetime(df["Datetime"])

        if hasattr(self.strategy, "normalize_features"):
            df = self.strategy.normalize_features(df)

        # Time-based split
        df = df.sort_values("Datetime").reset_index(drop=True)
        split_idx = int(len(df) * self.train_ratio)
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:].copy()

        # Train and get signals for test period
        self.strategy.train(train_df)
        test_df = self._prepare_signals(test_df)
        signal_lookup = self._build_signal_lookup(test_df)

        # Components
        cash = float(self.initial_capital)
        positions: Dict[str, float] = {}  # ticker -> shares
        cost_basis: Dict[str, Dict[str, float]] = {}  # ticker -> {"shares": float, "cost": float}
        last_price: Dict[str, float] = {}  # ticker -> last Close
        equity_curve_ts: List[float] = []
        equity_curve_val: List[float] = []
        trades: List[Dict[str, Any]] = []

        om = OrderManager(
            initial_capital=self.initial_capital,
            orders_per_minute_limit=self.orders_per_minute_limit,
            max_buy_position=self.position_limit_notional,
            max_sell_position=self.position_limit_notional,
            position_limit_is_notional=True,
        )
        om.available_capital = cash

        engine = MatchingEngine(seed=self.matching_engine_seed)
        gateway = DataGateway(sort_by_time=True)
        audit = OrderGateway(filepath=self.audit_log_path) if self.audit_log_path else None

        test_start = test_df["Datetime"].min()
        test_end = test_df["Datetime"].max()
        bar_index = 0.0

        for row in gateway.stream(data_path):
            # Only process rows in test period
            dt_val = row.get("Datetime")
            if dt_val is not None:
                try:
                    dt = pd.to_datetime(dt_val)
                    if dt < test_start or dt > test_end:
                        continue
                except Exception:
                    pass
            dt_raw = row.get("Datetime")
            ticker = row.get("Ticker", "")
            close = row.get("Close")
            if close is None or pd.isna(close):
                continue
            close = float(close)
            last_price[ticker] = close
            dt_key = _normalize_dt_key(dt_raw) if dt_raw is not None else ""
            key = (dt_key, ticker)
            sig_info = signal_lookup.get(key)
            if not sig_info or sig_info["signal"] == 0:
                # Update equity at this bar
                equity = cash + sum(positions.get(t, 0) * last_price.get(t, 0) for t in positions)
                equity_curve_ts.append(bar_index)
                equity_curve_val.append(equity)
                bar_index += 1
                continue

            signal = sig_info["signal"]
            weight = sig_info["weight"]
            price = close
            equity_before = cash + sum(positions.get(t, 0) * last_price.get(t, 0) for t in positions)
            target_notional = equity_before * abs(weight) * position_pct_per_name
            size = max(min_order_size, target_notional / price) if price > 0 else 0
            size = round(size, 4)
            if size <= 0:
                equity_curve_ts.append(bar_index)
                equity_curve_val.append(equity_before)
                bar_index += 1
                continue

            side = Side.BID if signal > 0 else Side.ASK
            ts = bar_index
            vr = om.validate_order(side, price, size, timestamp=ts)
            if not vr.valid:
                equity_curve_ts.append(bar_index)
                equity_curve_val.append(equity_before)
                bar_index += 1
                continue

            engine.set_time(ts)
            res = engine.submit_order(side, price, size, timestamp=ts)
            om.record_order_sent(timestamp=ts)

            if audit:
                audit.log_sent(res.order_id, res.side, res.price, res.size, timestamp=ts)

            def _record_fill_and_trade(
                _cash: float,
                _positions: Dict[str, float],
                _cost_basis: Dict[str, Dict[str, float]],
                _ticker: str,
                _side: Side,
                fill_price: float,
                fill_size: float,
            ) -> Tuple[float, float, Dict[str, float], Dict[str, Dict[str, float]]]:
                """Update cash, positions, cost_basis; return (new_cash, realized_pnl, new_positions, new_cost_basis)."""
                pnl = 0.0
                _positions = dict(_positions)
                _cost_basis = {k: dict(v) for k, v in _cost_basis.items()}
                if _side == Side.BID:
                    _cash -= fill_price * fill_size
                    _positions[_ticker] = _positions.get(_ticker, 0) + fill_size
                    if _ticker not in _cost_basis:
                        _cost_basis[_ticker] = {"shares": 0.0, "cost": 0.0}
                    _cost_basis[_ticker]["shares"] += fill_size
                    _cost_basis[_ticker]["cost"] += fill_price * fill_size
                else:
                    _cash += fill_price * fill_size
                    _positions[_ticker] = _positions.get(_ticker, 0) - fill_size
                    if _ticker in _cost_basis and _cost_basis[_ticker]["shares"] > 0:
                        avg_cost = _cost_basis[_ticker]["cost"] / _cost_basis[_ticker]["shares"]
                        close_shares = min(fill_size, _cost_basis[_ticker]["shares"])
                        pnl = (fill_price - avg_cost) * close_shares
                        _cost_basis[_ticker]["shares"] -= close_shares
                        _cost_basis[_ticker]["cost"] -= avg_cost * close_shares
                        if _cost_basis[_ticker]["shares"] <= 0:
                            _cost_basis[_ticker] = {"shares": 0.0, "cost": 0.0}
                return (_cash, pnl, _positions, _cost_basis)

            if res.status == ExecutionStatus.FILLED and res.filled_size > 0:
                om.record_fill(side, res.filled_price or price, res.filled_size)
                fill_price = res.filled_price or price
                cash, pnl, positions, cost_basis = _record_fill_and_trade(
                    cash, positions, cost_basis, ticker, side, fill_price, res.filled_size
                )
                trades.append({
                    "timestamp": ts,
                    "datetime": dt_raw,
                    "ticker": ticker,
                    "side": res.side,
                    "price": fill_price,
                    "size": res.filled_size,
                    "order_id": res.order_id,
                    "pnl": pnl,
                })
                if audit:
                    audit.log_filled(
                        res.order_id, res.side, res.price, res.size,
                        filled_size=res.filled_size, remaining_size=0, timestamp=ts,
                    )
            elif res.status == ExecutionStatus.PARTIALLY_FILLED and res.filled_size > 0:
                om.record_fill(side, res.filled_price or price, res.filled_size)
                fill_price = res.filled_price or price
                cash, pnl, positions, cost_basis = _record_fill_and_trade(
                    cash, positions, cost_basis, ticker, side, fill_price, res.filled_size
                )
                trades.append({
                    "timestamp": ts,
                    "datetime": dt_raw,
                    "ticker": ticker,
                    "side": res.side,
                    "price": fill_price,
                    "size": res.filled_size,
                    "order_id": res.order_id,
                    "pnl": pnl,
                })
                if audit:
                    audit.log_partial_fill(
                        res.order_id, res.side, res.price, res.size,
                        filled_size=res.filled_size, remaining_size=res.remaining_size, timestamp=ts,
                    )
            else:
                if audit:
                    audit.log_cancelled(
                        res.order_id, res.side, res.price, res.size,
                        remaining_size=res.remaining_size, timestamp=ts,
                    )

            equity = cash + sum(positions.get(t, 0) * last_price.get(t, 0) for t in positions)
            equity_curve_ts.append(bar_index)
            equity_curve_val.append(equity)
            bar_index += 1

        # Build equity series (index = bar index)
        if equity_curve_ts:
            eq_series = pd.Series(equity_curve_val, index=equity_curve_ts).sort_index()
        else:
            eq_series = pd.Series([self.initial_capital], index=[0.0])

        # Compute trade PnL if we have position closes (simplified: no cost basis here; use 0)
        # For win/loss we can use trade direction and later mark-to-market or assume round-trip
        metrics = compute_metrics(eq_series, trades, self.initial_capital)
        final_equity = cash + sum(positions.get(t, 0) * last_price.get(t, 0) for t in positions)

        return BacktestResult(
            initial_capital=self.initial_capital,
            final_equity=metrics.get("final_equity", final_equity),
            total_return=metrics["total_return"],
            sharpe_ratio=metrics["sharpe_ratio"],
            max_drawdown=metrics["max_drawdown"],
            win_rate=metrics["win_rate"],
            profit_factor=metrics["profit_factor"],
            num_trades=metrics["num_trades"],
            num_wins=metrics["num_wins"],
            num_losses=metrics["num_losses"],
            equity_curve=eq_series,
            trades=trades,
            params={
                "train_ratio": self.train_ratio,
                "position_pct_per_name": position_pct_per_name,
            },
        )


# ---------------------------------------------------------------------------
# Reporting: equity curve, trade distribution, stats, comparison
# ---------------------------------------------------------------------------

def plot_equity_curve(result: BacktestResult, ax=None, title: str = "Equity Curve"):
    """Plot equity curve. Requires matplotlib."""
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    result.equity_curve.plot(ax=ax, label="Equity")
    ax.set_title(title)
    ax.set_ylabel("Equity")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax


def plot_trade_distribution(trades: List[Dict[str, Any]], ax=None, title: str = "Trade distribution"):
    """Plot histogram of trade sizes or PnL if available."""
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    if not trades:
        ax.set_title(title + " (no trades)")
        return ax
    pnls = [t.get("pnl", 0) for t in trades]
    sizes = [t.get("size", 0) for t in trades]
    if any(p != 0 for p in pnls):
        ax.hist(pnls, bins=min(50, max(10, len(trades))), edgecolor="black", alpha=0.7)
        ax.set_ylabel("Count")
        ax.set_xlabel("Trade PnL")
    else:
        ax.hist(sizes, bins=min(50, max(10, len(trades))), edgecolor="black", alpha=0.7)
        ax.set_ylabel("Count")
        ax.set_xlabel("Trade size")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return ax


def print_performance_stats(result: BacktestResult):
    """Print key performance statistics."""
    d = result.to_dict()
    print("Performance Statistics")
    print("-" * 40)
    print(f"Initial capital:  {d['initial_capital']:,.2f}")
    print(f"Final equity:     {d['final_equity']:,.2f}")
    print(f"Total return:     {d['total_return']:.2%}")
    print(f"Sharpe ratio:     {d['sharpe_ratio']:.4f}")
    print(f"Max drawdown:     {d['max_drawdown']:.2%}")
    print(f"Win rate:         {d['win_rate']:.2%}")
    print(f"Profit factor:    {d['profit_factor']:.4f}")
    print(f"Num trades:       {d['num_trades']} (wins: {d['num_wins']}, losses: {d['num_losses']})")
    print(f"Params:           {d.get('params', {})}")


def compare_results(
    results: List[BacktestResult],
    labels: Optional[List[str]] = None,
    plot: bool = True,
):
    """
    Compare multiple backtest runs (e.g. different strategy parameters).
    Prints a comparison table and optionally plots equity curves.
    """
    import matplotlib.pyplot as plt
    if labels is None:
        labels = [f"Run {i+1}" for i in range(len(results))]
    rows = []
    for r, label in zip(results, labels):
        d = r.to_dict()
        rows.append({
            "Label": label,
            "Total Return": d["total_return"],
            "Sharpe": d["sharpe_ratio"],
            "Max DD": d["max_drawdown"],
            "Win Rate": d["win_rate"],
            "Trades": d["num_trades"],
        })
    comp_df = pd.DataFrame(rows)
    print("Strategy comparison")
    print(comp_df.to_string(index=False))
    if plot and results:
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        for r, label in zip(results, labels):
            r.equity_curve.plot(ax=ax, label=label, alpha=0.8)
        ax.set_title("Equity curve comparison")
        ax.set_ylabel("Equity")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig, ax
    return None, None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    from strategy import RankingStrategy
    strategy = RankingStrategy(top_k=3)
    backtester = Backtester(
        strategy=strategy,
        initial_capital=1_000_000.0,
        train_ratio=0.7,
        matching_engine_seed=42,
        audit_log_path="backtest_order_audit.csv",
    )
    result = backtester.run("multi_stock_dataset.csv", position_pct_per_name=0.05)
    print_performance_stats(result)
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        plot_equity_curve(result, ax=axes[0])
        plot_trade_distribution(result.trades, ax=axes[1])
        plt.tight_layout()
        plt.savefig("backtest_report.png", dpi=150)
        plt.close()
        print("Saved backtest_report.png")
    except ImportError:
        print("matplotlib not installed; skipping plots.")
    # Parameter sensitivity: compare top_k
    results = []
    for top_k in [2, 3, 5, 7, 9, 15]:
        s = RankingStrategy(top_k=top_k)
        bt = Backtester(s, initial_capital=1_000_000.0, train_ratio=0.7, matching_engine_seed=42)
        r = bt.run("multi_stock_dataset.csv", position_pct_per_name=0.05)
        results.append(r)
    compare_results(results, labels=["top_k=2", "top_k=3", "top_k=5", "top_k=7", "top_k=9", "top_k=15"], plot=True)


if __name__ == "__main__":
    main()
