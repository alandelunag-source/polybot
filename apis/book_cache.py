"""
In-memory order book cache.

Stores the full bid/ask ladder per token ID and applies incremental delta
updates as they arrive from the WebSocket feed, avoiding redundant REST calls.

Price levels are keyed by price string (e.g. "0.4500") to avoid float
precision drift when deleting levels. Size=0 in a delta means remove the level.
"""
from __future__ import annotations

import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# {price_str: size_float}
_PriceLadder = dict[str, float]


class BookState:
    """Bid/ask ladder for a single outcome token."""

    __slots__ = ("bids", "asks", "updated_at")

    def __init__(self) -> None:
        self.bids: _PriceLadder = {}
        self.asks: _PriceLadder = {}
        self.updated_at: float = 0.0

    def best_ask(self) -> Optional[float]:
        if not self.asks:
            return None
        return min(float(p) for p, s in self.asks.items() if s > 0)

    def best_bid(self) -> Optional[float]:
        if not self.bids:
            return None
        return max(float(p) for p, s in self.bids.items() if s > 0)

    def to_dict(self) -> dict:
        return {
            "bids": [{"price": p, "size": str(s)} for p, s in self.bids.items() if s > 0],
            "asks": [{"price": p, "size": str(s)} for p, s in self.asks.items() if s > 0],
            "updated_at": self.updated_at,
        }


class BookCache:
    """
    Thread-safe (asyncio single-threaded) cache of order book state for all
    subscribed outcome tokens.

    Usage:
        cache = BookCache()
        cache.apply_snapshot(token_id, bids=[...], asks=[...])
        cache.apply_delta(token_id, changes=[...])
        ask = cache.best_ask(token_id)
    """

    def __init__(self) -> None:
        self._books: dict[str, BookState] = {}

    # ------------------------------------------------------------------
    # Write operations (called from WS message handler)
    # ------------------------------------------------------------------

    def apply_snapshot(
        self,
        token_id: str,
        bids: list[dict],
        asks: list[dict],
    ) -> None:
        """
        Replace the full book for token_id with a fresh snapshot.
        Called on initial 'book' event after subscribing.
        """
        book = self._books.setdefault(token_id, BookState())
        book.bids = {
            entry["price"]: float(entry["size"])
            for entry in bids
            if float(entry.get("size", 0)) > 0
        }
        book.asks = {
            entry["price"]: float(entry["size"])
            for entry in asks
            if float(entry.get("size", 0)) > 0
        }
        book.updated_at = time.monotonic()
        logger.debug(
            "Snapshot applied: token=%s  bids=%d  asks=%d",
            token_id[:12],
            len(book.bids),
            len(book.asks),
        )

    def apply_delta(self, token_id: str, changes: list[dict]) -> None:
        """
        Apply incremental changes from a 'price_change' event.
        Each change has: {"price": str, "side": "BUY"|"SELL", "size": str}
        Size "0" or 0.0 means remove that price level.
        """
        book = self._books.get(token_id)
        if book is None:
            # Haven't received a snapshot yet — ignore delta
            logger.debug("Delta received before snapshot for token %s, ignoring", token_id[:12])
            return

        for change in changes:
            price = change["price"]
            size = float(change["size"])
            side = change.get("side", "").upper()

            ladder = book.bids if side == "BUY" else book.asks
            if size == 0:
                ladder.pop(price, None)
            else:
                ladder[price] = size

        book.updated_at = time.monotonic()

    # ------------------------------------------------------------------
    # Read operations (called from arb scanner callback)
    # ------------------------------------------------------------------

    def best_ask(self, token_id: str) -> Optional[float]:
        book = self._books.get(token_id)
        return book.best_ask() if book else None

    def best_bid(self, token_id: str) -> Optional[float]:
        book = self._books.get(token_id)
        return book.best_bid() if book else None

    def get_book(self, token_id: str) -> Optional[dict]:
        book = self._books.get(token_id)
        return book.to_dict() if book else None

    def age_seconds(self, token_id: str) -> Optional[float]:
        """Seconds since the last update for this token."""
        book = self._books.get(token_id)
        if book is None or book.updated_at == 0:
            return None
        return time.monotonic() - book.updated_at

    def tracked_tokens(self) -> list[str]:
        return list(self._books.keys())

    def __len__(self) -> int:
        return len(self._books)
