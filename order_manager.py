"""
Order Manager: validation and risk controls.
Validates orders for capital sufficiency and risk limits before execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
from collections import deque
import time

# Use order_book.Side when coupling with order book
try:
    from order_book import Side
except ImportError:
    from enum import Enum
    class Side(str, Enum):
        BID = "BID"
        ASK = "ASK"


@dataclass
class ValidationResult:
    """Result of order validation."""
    valid: bool
    reason: Optional[str] = None  # Set when valid is False


class OrderManager:
    """
    Validates and records orders before execution.
    - Capital sufficiency: enough capital to execute (for buys).
    - Risk limits: orders per minute, total buy position limit, total sell position limit.
    """

    def __init__(
        self,
        initial_capital: float,
        orders_per_minute_limit: int = 30,
        max_buy_position: float = 1_000_000.0,   # max total quantity or notional for buys
        max_sell_position: float = 1_000_000.0,    # max total quantity or notional for sells
        position_limit_is_notional: bool = False,  # if True, limits are in $; else in quantity
    ):
        """
        Args:
            initial_capital: Starting cash (used for buy capital check).
            orders_per_minute_limit: Max orders allowed per minute (risk limit).
            max_buy_position: Max cumulative buy quantity (or notional if position_limit_is_notional).
            max_sell_position: Max cumulative sell quantity (or notional if position_limit_is_notional).
            position_limit_is_notional: If True, position limits are in dollar notional; else in quantity.
        """
        self.available_capital = float(initial_capital)
        self.orders_per_minute_limit = orders_per_minute_limit
        self.max_buy_position = float(max_buy_position)
        self.max_sell_position = float(max_sell_position)
        self.position_limit_is_notional = position_limit_is_notional
        # Cumulative position (quantity or notional) used this session for limit checks
        self._current_buy_position: float = 0.0
        self._current_sell_position: float = 0.0
        # Orders per minute: timestamps of orders in the current window
        self._order_timestamps: deque = deque()

    def _trim_old_minutes(self, current_time: Optional[float] = None) -> None:
        """Remove timestamps older than one minute."""
        now = current_time if current_time is not None else time.time()
        cutoff = now - 60.0
        while self._order_timestamps and self._order_timestamps[0] < cutoff:
            self._order_timestamps.popleft()

    def validate_order(
        self,
        side: Side,
        price: float,
        size: float,
        timestamp: Optional[float] = None,
    ) -> ValidationResult:
        """
        Check if the order passes capital and risk limits.

        Returns:
            ValidationResult(valid=True) or ValidationResult(valid=False, reason="...").
        """
        if price <= 0 or size <= 0:
            return ValidationResult(valid=False, reason="Price and size must be positive.")

        # Orders per minute
        self._trim_old_minutes(timestamp)
        now = timestamp if timestamp is not None else time.time()
        if len(self._order_timestamps) >= self.orders_per_minute_limit:
            return ValidationResult(
                valid=False,
                reason=f"Orders per minute limit reached ({self.orders_per_minute_limit}).",
            )

        notional = price * size
        if self.position_limit_is_notional:
            buy_impact = notional if side == Side.BID else 0.0
            sell_impact = notional if side == Side.ASK else 0.0
        else:
            buy_impact = size if side == Side.BID else 0.0
            sell_impact = size if side == Side.ASK else 0.0

        if side == Side.BID:
            if self.available_capital < notional:
                return ValidationResult(
                    valid=False,
                    reason=f"Insufficient capital: need {notional:.2f}, available {self.available_capital:.2f}.",
                )
            if self._current_buy_position + buy_impact > self.max_buy_position:
                return ValidationResult(
                    valid=False,
                    reason=f"Buy position limit exceeded: would be {self._current_buy_position + buy_impact:.2f}, max {self.max_buy_position:.2f}.",
                )
        else:
            if self._current_sell_position + sell_impact > self.max_sell_position:
                return ValidationResult(
                    valid=False,
                    reason=f"Sell position limit exceeded: would be {self._current_sell_position + sell_impact:.2f}, max {self.max_sell_position:.2f}.",
                )

        return ValidationResult(valid=True)

    def record_order_sent(self, timestamp: Optional[float] = None) -> None:
        """Call when an order is sent (for orders-per-minute accounting)."""
        now = timestamp if timestamp is not None else time.time()
        self._order_timestamps.append(now)

    def record_fill(
        self,
        side: Side,
        price: float,
        size: float,
    ) -> None:
        """
        Update capital and position after a fill.
        - Buy: reduce capital by price*size; increase buy position.
        - Sell: increase capital by price*size; increase sell position.
        """
        notional = price * size
        if self.position_limit_is_notional:
            buy_inc = notional if side == Side.BID else 0.0
            sell_inc = notional if side == Side.ASK else 0.0
        else:
            buy_inc = size if side == Side.BID else 0.0
            sell_inc = size if side == Side.ASK else 0.0

        if side == Side.BID:
            self.available_capital -= notional
            self._current_buy_position += buy_inc
        else:
            self.available_capital += notional
            self._current_sell_position += sell_inc

    def record_cancel(self, side: Side, price: float, size: float) -> None:
        """
        Optional: reverse position/capital if a previously filled order is cancelled.
        Not used when cancel only removes unfilled order; use when cancelling fills.
        """
        # Typically cancel does not reverse fills; leave no-op unless you track pending.
        pass

    @property
    def current_buy_position(self) -> float:
        return self._current_buy_position

    @property
    def current_sell_position(self) -> float:
        return self._current_sell_position
