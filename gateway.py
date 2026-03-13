"""
Backtester Gateway: Data Ingestion
Streams cleaned CSV market data row-by-row to simulate a live market data feed.
"""

import csv
from pathlib import Path
from typing import Iterator, List, Union, Optional, Dict, Any
from datetime import datetime
import heapq


class DataGateway:
    """
    Gateway that reads cleaned CSV files from Part 1 and feeds market data
    into the system incrementally, row-by-row, to mimic real-time updates.
    """

    def __init__(
        self,
        datetime_column: str = "Datetime",
        sort_by_time: bool = True,
    ):
        """
        Args:
            datetime_column: Name of the datetime column for ordering.
            sort_by_time: If True, stream rows in chronological order (required when
                merging multiple files or when CSV is not pre-sorted).
        """
        self.datetime_column = datetime_column
        self.sort_by_time = sort_by_time

    def stream_file(self, filepath: Union[str, Path]) -> Iterator[Dict[str, Any]]:
        """
        Stream a single CSV file row-by-row.

        Yields each row as a dict with column names as keys. Numeric columns
        are returned as floats when possible; datetime is left as string unless
        the CSV was written with a standard format.

        Args:
            filepath: Path to cleaned CSV file.

        Yields:
            Dict mapping column names to values for each row.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {filepath}")

        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                return
            for row in reader:
                yield self._coerce_row(row)

    def stream_files(
        self,
        filepaths: List[Union[str, Path]],
    ) -> Iterator[Dict[str, Any]]:
        """
        Stream multiple CSV files in chronological order by datetime.

        Reads each file incrementally and merges by the datetime column so
        that rows are yielded in time order, simulating a single live feed
        from multiple sources.

        Args:
            filepaths: List of paths to cleaned CSV files.

        Yields:
            Rows from all files in ascending datetime order.
        """
        if not filepaths:
            return

        if len(filepaths) == 1:
            if self.sort_by_time:
                yield from self._stream_sorted_single(filepaths[0])
            else:
                yield from self.stream_file(filepaths[0])
            return

        # Multi-file: merge by datetime using a heap
        yield from self._stream_merged(filepaths)

    def _coerce_row(self, row: Dict[str, str]) -> Dict[str, Any]:
        """Coerce string values to appropriate types where possible."""
        out = {}
        for k, v in row.items():
            if v == "" or v is None:
                out[k] = None
                continue
            # Try numeric
            try:
                if "." in v or "e" in v.lower():
                    out[k] = float(v)
                else:
                    out[k] = int(v)
            except ValueError:
                out[k] = v
        return out

    def _stream_sorted_single(self, filepath: Union[str, Path]) -> Iterator[Dict[str, Any]]:
        """Stream a single file in sorted order by datetime (loads file into memory)."""
        rows = list(self.stream_file(filepath))
        if not rows or self.datetime_column not in rows[0]:
            yield from rows
            return
        try:
            sorted_rows = sorted(
                rows,
                key=lambda r: (
                    r[self.datetime_column]
                    if isinstance(r[self.datetime_column], (int, float))
                    else str(r[self.datetime_column])
                ),
            )
        except (TypeError, KeyError):
            yield from rows
            return
        yield from sorted_rows

    def _stream_merged(self, filepaths: List[Union[str, Path]]) -> Iterator[Dict[str, Any]]:
        """Merge multiple CSV streams by datetime using a min-heap."""
        iters = []
        for fp in filepaths:
            it = iter(self.stream_file(fp))
            try:
                first = next(it)
            except StopIteration:
                continue
            dt = first.get(self.datetime_column)
            # Use string comparison for datetime strings; ensure comparable
            key = (str(dt) if dt is not None else "", first)
            iters.append((key, it, first))

        if not iters:
            return

        heapq.heapify(iters)

        while iters:
            _key, it, row = heapq.heappop(iters)
            yield row
            try:
                nxt = next(it)
            except StopIteration:
                continue
            dt = nxt.get(self.datetime_column)
            key = (str(dt) if dt is not None else "", nxt)
            heapq.heappush(iters, (key, it, nxt))

    def stream(
        self,
        source: Union[str, Path, List[Union[str, Path]]],
    ) -> Iterator[Dict[str, Any]]:
        """
        Main entry: stream one or more cleaned CSV files row-by-row.

        Args:
            source: Single file path or list of file paths.

        Yields:
            Market data rows as dicts in chronological order (if sort_by_time).
        """
        if isinstance(source, (list, tuple)):
            yield from self.stream_files(list(source))
        else:
            if self.sort_by_time:
                yield from self._stream_sorted_single(source)
            else:
                yield from self.stream_file(source)


def parse_datetime(value: Any) -> Optional[datetime]:
    """
    Parse a datetime value from a CSV row (string or already datetime).

    Args:
        value: Cell value for the datetime column.

    Returns:
        datetime or None if parsing fails.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, (int, float)):
        try:
            return datetime.utcfromtimestamp(value)
        except (OSError, ValueError):
            return None
    s = str(value).strip()
    if not s:
        return None
    for fmt in (
        "%Y-%m-%d %H:%M:%S%z",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d",
    ):
        try:
            normalized = s.replace("+00:00", "").replace("Z", "").rstrip()
            fmt_naive = fmt.replace("%z", "").rstrip()
            return datetime.strptime(normalized, fmt_naive)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        pass
    return None
