"""
Order Gateway: audit logging of all order events.
Writes every order event (sent, modified, cancelled, filled) to a file for audit and analysis.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional, Any, Dict, Union
from enum import Enum
from datetime import datetime
import threading


class OrderEventType(str, Enum):
    SENT = "SENT"
    MODIFIED = "MODIFIED"
    CANCELLED = "CANCELLED"
    FILLED = "FILLED"
    PARTIAL_FILL = "PARTIAL_FILL"


class OrderGateway:
    """
    Gateway that writes all order events to a file for audit and analysis.
    Logs when orders are sent, modified, cancelled, or filled.
    """

    DEFAULT_FIELDS = [
        "timestamp",
        "event_type",
        "order_id",
        "side",
        "price",
        "size",
        "filled_size",
        "remaining_size",
        "details",
    ]

    def __init__(
        self,
        filepath: Optional[Union[str, Path]] = None,
        append: bool = True,
    ):
        """
        Args:
            filepath: Path to audit CSV. If None, no file is written until set_file() or log_* is called with filepath.
            append: If True, append to existing file; else overwrite.
        """
        self._filepath: Optional[Path] = Path(filepath) if filepath else None
        self._append = append
        self._header_written = False
        self._lock = threading.Lock()

    def set_file(self, filepath: Union[str, Path], append: bool = True) -> None:
        """Set or change the audit file path."""
        self._filepath = Path(filepath)
        self._append = append
        self._header_written = False

    def _ensure_header(self) -> None:
        if not self._filepath or self._header_written:
            return
        with self._lock:
            if self._header_written:
                return
            exists = self._filepath.exists()
            has_content = exists and self._filepath.stat().st_size > 0
            with open(self._filepath, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=self.DEFAULT_FIELDS, extrasaction="ignore")
                if not has_content:
                    w.writeheader()
                self._header_written = True

    def _write_row(self, row: Dict[str, Any]) -> None:
        if not self._filepath:
            return
        self._ensure_header()
        with self._lock:
            with open(self._filepath, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=self.DEFAULT_FIELDS, extrasaction="ignore")
                w.writerow(row)

    def _ts(self, timestamp: Optional[float] = None) -> str:
        if timestamp is not None:
            try:
                return datetime.utcfromtimestamp(timestamp).isoformat() + "Z"
            except (OSError, ValueError):
                pass
        return datetime.utcnow().isoformat() + "Z"

    def log_sent(
        self,
        order_id: int,
        side: str,
        price: float,
        size: float,
        timestamp: Optional[float] = None,
        details: Optional[str] = None,
    ) -> None:
        """Log order sent (submitted)."""
        self._write_row({
            "timestamp": self._ts(timestamp),
            "event_type": OrderEventType.SENT.value,
            "order_id": order_id,
            "side": side,
            "price": price,
            "size": size,
            "filled_size": "",
            "remaining_size": size,
            "details": details or "",
        })

    def log_modified(
        self,
        order_id: int,
        side: str,
        old_price: float,
        old_size: float,
        new_price: float,
        new_size: float,
        new_order_id: Optional[int] = None,
        timestamp: Optional[float] = None,
        details: Optional[str] = None,
    ) -> None:
        """Log order modified (e.g. cancel-replace)."""
        details_str = details or ""
        if new_order_id is not None:
            details_str = (details_str + f" new_order_id={new_order_id}").strip()
        self._write_row({
            "timestamp": self._ts(timestamp),
            "event_type": OrderEventType.MODIFIED.value,
            "order_id": order_id,
            "side": side,
            "price": new_price,
            "size": new_size,
            "filled_size": "",
            "remaining_size": new_size,
            "details": f"old_price={old_price} old_size={old_size}" + (" " + details_str if details_str else ""),
        })

    def log_cancelled(
        self,
        order_id: int,
        side: str,
        price: float,
        size: float,
        remaining_size: Optional[float] = None,
        timestamp: Optional[float] = None,
        details: Optional[str] = None,
    ) -> None:
        """Log order cancelled."""
        self._write_row({
            "timestamp": self._ts(timestamp),
            "event_type": OrderEventType.CANCELLED.value,
            "order_id": order_id,
            "side": side,
            "price": price,
            "size": size,
            "filled_size": "",
            "remaining_size": (remaining_size if remaining_size is not None else size),
            "details": details or "",
        })

    def log_filled(
        self,
        order_id: int,
        side: str,
        price: float,
        size: float,
        filled_size: float,
        remaining_size: float = 0.0,
        timestamp: Optional[float] = None,
        details: Optional[str] = None,
    ) -> None:
        """Log full fill."""
        self._write_row({
            "timestamp": self._ts(timestamp),
            "event_type": OrderEventType.FILLED.value,
            "order_id": order_id,
            "side": side,
            "price": price,
            "size": size,
            "filled_size": filled_size,
            "remaining_size": remaining_size,
            "details": details or "",
        })

    def log_partial_fill(
        self,
        order_id: int,
        side: str,
        price: float,
        size: float,
        filled_size: float,
        remaining_size: float,
        timestamp: Optional[float] = None,
        details: Optional[str] = None,
    ) -> None:
        """Log partial fill."""
        self._write_row({
            "timestamp": self._ts(timestamp),
            "event_type": OrderEventType.PARTIAL_FILL.value,
            "order_id": order_id,
            "side": side,
            "price": price,
            "size": size,
            "filled_size": filled_size,
            "remaining_size": remaining_size,
            "details": details or "",
        })
