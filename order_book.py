"""
Order Book Implementation
Manages bid/ask orders with heaps (priority queues) and price-time priority matching.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any


class Side(str, Enum):
    BID = "BID"
    ASK = "ASK"


@dataclass
class Order:
    """User-facing order record."""
    order_id: int
    side: Side
    price: float
    size: float
    timestamp: float
    remaining: float  # unfilled size

    def to_dict(self) -> Dict[str, Any]:
        return {
            "order_id": self.order_id,
            "side": self.side.value,
            "price": self.price,
            "size": self.size,
            "timestamp": self.timestamp,
            "remaining": self.remaining,
        }


@dataclass
class Trade:
    """Result of matching a bid and ask."""
    bid_order_id: int
    ask_order_id: int
    price: float
    size: float
    timestamp: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bid_order_id": self.bid_order_id,
            "ask_order_id": self.ask_order_id,
            "price": self.price,
            "size": self.size,
            "timestamp": self.timestamp,
        }


class OrderBook:
    """
    Order book with price-time priority using heaps.

    - Bids: max-heap by price (simulated with -price in min-heap), then by time.
    - Asks: min-heap by price, then by time.
    - Supports add, modify, cancel, and matching.
    """

    def __init__(self, next_order_id: int = 1):
        # Bids: min-heap of (price_key, order_id, price, size, timestamp)
        # price_key for bids = -price so highest bid is at top
        self._bid_heap: List[Tuple[Tuple[float, float, int], int, float, float, float]] = []
        # Asks: min-heap of (price, timestamp, order_id, price, size, timestamp)
        self._ask_heap: List[Tuple[Tuple[float, float, int], int, float, float, float]] = []
        # order_id -> (side, heap_index_ref, price, size, timestamp)
        # We don't have direct heap index; we use a "stale" set and lazy removal
        self._orders: Dict[int, Tuple[Side, float, float, float]] = {}  # id -> (side, price, size, ts)
        self._next_order_id = next_order_id
        self._cancelled: set = set()
        self._current_time: float = 0.0

    def _timestamp(self) -> float:
        """Monotonic time for priority; can be overridden for replay."""
        self._current_time += 1.0
        return self._current_time

    def set_time(self, t: float) -> None:
        """Set current time (e.g. from backtest). Next add will use t+1 if not advanced."""
        self._current_time = max(self._current_time, t)

    def add_order(self, side: Side, price: float, size: float, timestamp: Optional[float] = None) -> int:
        """
        Add an order to the book.

        Args:
            side: BID or ASK.
            price: Limit price.
            size: Order size (quantity).
            timestamp: Optional time for priority; default uses internal monotonic time.

        Returns:
            order_id for this order.
        """
        if size <= 0 or price <= 0:
            raise ValueError("Price and size must be positive.")
        order_id = self._next_order_id
        self._next_order_id += 1
        ts = timestamp if timestamp is not None else self._timestamp()
        self._orders[order_id] = (side, price, size, ts)
        # Bids: best = highest price, then earliest -> key (-price, ts, order_id)
        # Asks: best = lowest price, then earliest -> key (price, ts, order_id)
        if side == Side.BID:
            key = (-float(price), ts, order_id)
            heapq.heappush(self._bid_heap, (key, order_id, price, size, ts))
        else:
            key = (float(price), ts, order_id)
            heapq.heappush(self._ask_heap, (key, order_id, price, size, ts))
        return order_id

    def cancel_order(self, order_id: int) -> bool:
        """
        Cancel an order by id. It is marked cancelled and removed from the book
        when it would otherwise be matched (lazy removal).

        Returns:
            True if the order existed and was cancelled.
        """
        if order_id not in self._orders:
            return False
        self._cancelled.add(order_id)
        del self._orders[order_id]
        return True

    def modify_order(
        self,
        order_id: int,
        new_price: Optional[float] = None,
        new_size: Optional[float] = None,
    ) -> Optional[int]:
        """
        Modify an existing order. Implemented as cancel + add so the modified
        order gets new time priority (goes to the back of the queue).

        Returns:
            New order_id if the order existed and was modified, None otherwise.
        """
        if order_id not in self._orders:
            return None
        side, price, size, _ = self._orders[order_id]
        if new_price is not None:
            if new_price <= 0:
                raise ValueError("Price must be positive.")
            price = new_price
        if new_size is not None:
            if new_size <= 0:
                raise ValueError("Size must be positive.")
            size = new_size
        self.cancel_order(order_id)
        new_id = self.add_order(side, price, size)
        return new_id

    def _peek_best_bid(self) -> Optional[Tuple[int, float, float, float]]:
        """Return (order_id, price, size, ts) or None after popping stale."""
        while self._bid_heap:
            key, order_id, price, size, ts = self._bid_heap[0]
            if order_id in self._cancelled:
                heapq.heappop(self._bid_heap)
                self._cancelled.discard(order_id)
                continue
            return (order_id, price, size, ts)
        return None

    def _peek_best_ask(self) -> Optional[Tuple[int, float, float, float]]:
        """Return (order_id, price, size, ts) or None after popping stale."""
        while self._ask_heap:
            key, order_id, price, size, ts = self._ask_heap[0]
            if order_id in self._cancelled:
                heapq.heappop(self._ask_heap)
                self._cancelled.discard(order_id)
                continue
            return (order_id, price, size, ts)
        return None

    def _pop_bid(self) -> Optional[Tuple[int, float, float, float]]:
        bid = self._peek_best_bid()
        if bid is None:
            return None
        heapq.heappop(self._bid_heap)
        return bid

    def _pop_ask(self) -> Optional[Tuple[int, float, float, float]]:
        ask = self._peek_best_ask()
        if ask is None:
            return None
        heapq.heappop(self._ask_heap)
        return ask

    def match(self, current_time: Optional[float] = None) -> List[Trade]:
        """
        Match orders while best bid >= best ask (price-time priority).
        Fills are at the price of the resting (earlier) order.

        Returns:
            List of Trade objects for this match run.
        """
        trades: List[Trade] = []
        t = current_time if current_time is not None else self._current_time
        while True:
            bid = self._peek_best_bid()
            ask = self._peek_best_ask()
            if bid is None or ask is None:
                break
            bid_id, bid_price, bid_size, bid_ts = bid
            ask_id, ask_price, ask_size, ask_ts = ask
            if bid_price < ask_price:
                break
            fill_size = min(bid_size, ask_size)
            # Match price: use resting (maker) order price
            match_price = ask_price
            trades.append(
                Trade(
                    bid_order_id=bid_id,
                    ask_order_id=ask_id,
                    price=match_price,
                    size=fill_size,
                    timestamp=t,
                )
            )
            # Reduce or remove top bid
            if bid_size <= fill_size:
                self._pop_bid()
                if bid_id in self._orders:
                    del self._orders[bid_id]
            else:
                heapq.heappop(self._bid_heap)
                new_bid_size = bid_size - fill_size
                self._orders[bid_id] = (Side.BID, bid_price, new_bid_size, bid_ts)
                heapq.heappush(
                    self._bid_heap,
                    ((-bid_price, bid_ts, bid_id), bid_id, bid_price, new_bid_size, bid_ts),
                )
            # Reduce or remove top ask
            if ask_size <= fill_size:
                self._pop_ask()
                if ask_id in self._orders:
                    del self._orders[ask_id]
            else:
                heapq.heappop(self._ask_heap)
                new_ask_size = ask_size - fill_size
                self._orders[ask_id] = (Side.ASK, ask_price, new_ask_size, ask_ts)
                heapq.heappush(
                    self._ask_heap,
                    ((ask_price, ask_ts, ask_id), ask_id, ask_price, new_ask_size, ask_ts),
                )
        return trades

    def best_bid(self) -> Optional[Tuple[float, float]]:
        """(price, size) of best bid or None."""
        b = self._peek_best_bid()
        return (b[1], b[2]) if b else None

    def best_ask(self) -> Optional[Tuple[float, float]]:
        """(price, size) of best ask or None."""
        a = self._peek_best_ask()
        return (a[1], a[2]) if a else None

    def spread(self) -> Optional[float]:
        """Best ask - best bid, or None if either side is empty."""
        bid = self.best_bid()
        ask = self.best_ask()
        if bid is None or ask is None:
            return None
        return ask[0] - bid[0]

    def get_order(self, order_id: int) -> Optional[Order]:
        """Return Order if present (remaining size may be reduced by matches)."""
        if order_id not in self._orders:
            return None
        side, price, size, ts = self._orders[order_id]
        return Order(order_id=order_id, side=side, price=price, size=size, timestamp=ts, remaining=size)

    def order_count(self) -> int:
        """Number of live orders in the book."""
        return len(self._orders)
