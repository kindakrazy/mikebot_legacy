from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, List, Optional


@dataclass
class QueuedOrder:
    """
    Canonical queued order entry.

    Properties:
      - order: the original OrderRequest (or compatible object)
      - enqueued_at: UTC timestamp when inserted
      - sequence_id: strictly increasing integer for ordering

    This mirrors the behavior of the original Highstrike queue:
      - timestamped entries
      - stable ordering
      - queue metadata preserved
    """
    order: Any
    enqueued_at: datetime
    sequence_id: int


class OrderQueue:
    """
    Canonical mikebot v3 order queue.

    Features:
      - FIFO semantics
      - bounded size (oldest dropped on overflow)
      - thread‑safe push/pop/peek
      - sequence IDs for deterministic ordering
      - queue introspection utilities
    """

    def __init__(self, max_size: int = 1000) -> None:
        self.max_size = max_size
        self._queue: List[QueuedOrder] = []
        self._lock = threading.Lock()
        self._sequence_counter = 0

    # ----------------------------------------------------------------------
    # Queue operations
    # ----------------------------------------------------------------------

    def push(self, order: Any) -> QueuedOrder:
        """
        Insert an order into the queue.

        Behavior:
          - Assigns a strictly increasing sequence ID
          - Timestamps the entry
          - If queue is full, drop the oldest entry (Highstrike behavior)
        """
        with self._lock:
            self._sequence_counter += 1
            entry = QueuedOrder(
                order=order,
                enqueued_at=datetime.now(timezone.utc),
                sequence_id=self._sequence_counter,
            )

            if len(self._queue) >= self.max_size:
                self._queue.pop(0)

            self._queue.append(entry)
            return entry

    def pop(self) -> Optional[QueuedOrder]:
        """
        Pop the oldest order (FIFO).

        Returns None if queue is empty.
        """
        with self._lock:
            if not self._queue:
                return None
            return self._queue.pop(0)

    def peek(self) -> Optional[QueuedOrder]:
        """
        Return the oldest order without removing it.
        """
        with self._lock:
            if not self._queue:
                return None
            return self._queue[0]

    # ----------------------------------------------------------------------
    # Introspection
    # ----------------------------------------------------------------------

    def size(self) -> int:
        with self._lock:
            return len(self._queue)

    def is_empty(self) -> bool:
        with self._lock:
            return len(self._queue) == 0

    def snapshot(self) -> List[QueuedOrder]:
        """
        Return a shallow copy of the queue for inspection.
        """
        with self._lock:
            return list(self._queue)

    def clear(self) -> None:
        """
        Clear the queue.
        """
        with self._lock:
            self._queue.clear()

    # ----------------------------------------------------------------------
    # Utility
    # ----------------------------------------------------------------------

    def wait_for_order(self, timeout: float = 1.0) -> Optional[QueuedOrder]:
        """
        Block until an order is available or timeout expires.

        This mirrors the behavior used in Highstrike's MT4 bridge loops.
        """
        end = time.time() + timeout
        while time.time() < end:
            entry = self.pop()
            if entry is not None:
                return entry
            time.sleep(0.01)
        return None