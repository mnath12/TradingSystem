"""
Matching Engine Simulator
Simulates realistic order execution outcomes: filled, partially filled, or cancelled.
Returns execution details for each order.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Tuple

try:
    from order_book import Side, OrderBook
except ImportError:
    from enum import Enum
    class Side(str, Enum):
        BID = "BID"
        ASK = "ASK"
    OrderBook = None


class ExecutionStatus(str, Enum):
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


@dataclass
class ExecutionResult:
    """Execution details for a single order."""
    order_id: int
    status: ExecutionStatus
    side: str
    price: float
    size: float
    filled_size: float
    filled_price: Optional[float] = None  # average fill price if filled
    remaining_size: float = 0.0
    message: Optional[str] = None
    timestamp: float = 0.0

    def to_dict(self) -> dict:
        return {
            "order_id": self.order_id,
            "status": self.status.value,
            "side": self.side,
            "price": self.price,
            "size": self.size,
            "filled_size": self.filled_size,
            "filled_price": self.filled_price,
            "remaining_size": self.remaining_size,
            "message": self.message,
            "timestamp": self.timestamp,
        }


class MatchingEngine:
    """
    Simulates order matching and execution outcomes.
    Randomly determines whether orders are filled, partially filled, or cancelled.
    Returns execution details for each order.
    """

    def __init__(
        self,
        fill_probability: float = 0.6,
        partial_probability: float = 0.25,
        cancel_probability: float = 0.15,
        partial_fill_ratio_min: float = 0.2,
        partial_fill_ratio_max: float = 0.8,
        seed: Optional[int] = None,
    ):
        """
        Args:
            fill_probability: Probability of full fill (default 0.6).
            partial_probability: Probability of partial fill (default 0.25).
            cancel_probability: Probability of cancel (default 0.15). Probabilities are normalized if they don't sum to 1.
            partial_fill_ratio_min: Min fraction of size filled when partially filled.
            partial_fill_ratio_max: Max fraction of size filled when partially filled.
            seed: Random seed for reproducibility.
        """
        self.fill_probability = fill_probability
        self.partial_probability = partial_probability
        self.cancel_probability = cancel_probability
        self.partial_fill_ratio_min = partial_fill_ratio_min
        self.partial_fill_ratio_max = partial_fill_ratio_max
        self._rng = random.Random(seed)
        self._order_id_counter = 1
        self._current_time: float = 0.0

    def set_time(self, t: float) -> None:
        """Set current simulation time."""
        self._current_time = max(self._current_time, t)

    def _next_order_id(self) -> int:
        oid = self._order_id_counter
        self._order_id_counter += 1
        return oid

    def _roll_outcome(self) -> ExecutionStatus:
        """Randomly choose FILLED, PARTIALLY_FILLED, or CANCELLED."""
        total = self.fill_probability + self.partial_probability + self.cancel_probability
        if total <= 0:
            total = 1.0
        r = self._rng.random() * total
        if r < self.fill_probability:
            return ExecutionStatus.FILLED
        if r < self.fill_probability + self.partial_probability:
            return ExecutionStatus.PARTIALLY_FILLED
        return ExecutionStatus.CANCELLED

    def submit_order(
        self,
        side: Side,
        price: float,
        size: float,
        timestamp: Optional[float] = None,
    ) -> ExecutionResult:
        """
        Submit an order and simulate execution outcome.
        Randomly returns filled, partially filled, or cancelled.

        Returns:
            ExecutionResult with status, filled_size, filled_price, remaining_size, message.
        """
        if size <= 0 or price <= 0:
            return ExecutionResult(
                order_id=-1,
                status=ExecutionStatus.REJECTED,
                side=side.value,
                price=price,
                size=size,
                filled_size=0.0,
                remaining_size=size,
                message="Invalid price or size",
                timestamp=timestamp or self._current_time,
            )
        order_id = self._next_order_id()
        ts = timestamp if timestamp is not None else self._current_time
        self._current_time = ts

        status = self._roll_outcome()

        if status == ExecutionStatus.FILLED:
            return ExecutionResult(
                order_id=order_id,
                status=ExecutionStatus.FILLED,
                side=side.value,
                price=price,
                size=size,
                filled_size=size,
                filled_price=price,
                remaining_size=0.0,
                message="Filled",
                timestamp=ts,
            )

        if status == ExecutionStatus.PARTIALLY_FILLED:
            ratio = self._rng.uniform(self.partial_fill_ratio_min, self.partial_fill_ratio_max)
            filled_size = round(size * ratio, 6)
            if filled_size <= 0:
                filled_size = min(size, 1.0)
            remaining = max(0.0, size - filled_size)
            return ExecutionResult(
                order_id=order_id,
                status=ExecutionStatus.PARTIALLY_FILLED,
                side=side.value,
                price=price,
                size=size,
                filled_size=filled_size,
                filled_price=price,
                remaining_size=remaining,
                message=f"Partially filled {filled_size:.4f} of {size}",
                timestamp=ts,
            )

        # CANCELLED
        return ExecutionResult(
            order_id=order_id,
            status=ExecutionStatus.CANCELLED,
            side=side.value,
            price=price,
            size=size,
            filled_size=0.0,
            remaining_size=size,
            message="Cancelled by simulator",
            timestamp=ts,
        )

    def submit_orders_batch(
        self,
        orders: List[Tuple[Side, float, float]],
        timestamp: Optional[float] = None,
    ) -> List[ExecutionResult]:
        """
        Submit multiple orders and return execution details for each.

        Args:
            orders: List of (side, price, size).
            timestamp: Optional time for all orders.

        Returns:
            List of ExecutionResult, one per order.
        """
        results = []
        for side, price, size in orders:
            res = self.submit_order(side, price, size, timestamp)
            results.append(res)
        return results
