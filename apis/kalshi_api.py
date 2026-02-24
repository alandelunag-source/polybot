"""
Kalshi API client.

Auth: Bearer token via KALSHI_API_TOKEN env var.
Base: https://trading-api.kalshi.com/trade-api/v2

Key methods:
  get_markets()               -> list of normalized market dicts
  get_orderbook(ticker)       -> {yes_bid, yes_ask, no_bid, no_ask}
"""
from __future__ import annotations

import logging
from typing import Optional
import requests

from config import settings

logger = logging.getLogger(__name__)

BASE_URL = "https://trading-api.kalshi.com/trade-api/v2"


def _headers() -> dict:
    token = settings.KALSHI_API_TOKEN
    if not token:
        raise RuntimeError("KALSHI_API_TOKEN not set in .env")
    return {"Authorization": f"Bearer {token}", "Accept": "application/json"}


def get_markets(limit: int = 200, status: str = "open") -> list[dict]:
    """
    Return a list of normalized Kalshi market dicts.

    Normalized shape:
      {ticker, title, yes_bid, yes_ask, close_time, category, volume}
    """
    params = {"limit": limit, "status": status}
    try:
        r = requests.get(f"{BASE_URL}/markets", headers=_headers(), params=params, timeout=10)
        r.raise_for_status()
    except requests.RequestException as exc:
        logger.error("Kalshi get_markets failed: %s", exc)
        return []

    raw = r.json().get("markets", [])
    return [_normalize_market(m) for m in raw]


def get_orderbook(ticker: str) -> Optional[dict]:
    """
    Fetch the order book for a single Kalshi market.

    Returns:
      {yes_bid, yes_ask, no_bid, no_ask}  — prices in [0, 1]
      None if unavailable or error.
    """
    try:
        r = requests.get(
            f"{BASE_URL}/markets/{ticker}/orderbook",
            headers=_headers(),
            timeout=10,
        )
        r.raise_for_status()
    except requests.RequestException as exc:
        logger.debug("Kalshi orderbook fetch failed for %s: %s", ticker, exc)
        return None

    data = r.json().get("orderbook", {})
    return _parse_orderbook(data)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_market(m: dict) -> dict:
    """Extract the fields we care about from a raw Kalshi market object."""
    # Kalshi prices are in cents (0-99); convert to [0, 1]
    yes_bid = _to_prob(m.get("yes_bid"))
    yes_ask = _to_prob(m.get("yes_ask"))
    return {
        "ticker": m.get("ticker", ""),
        "title": m.get("title", m.get("question", "")),
        "yes_bid": yes_bid,
        "yes_ask": yes_ask,
        "close_time": m.get("close_time", ""),
        "category": m.get("category", ""),
        "volume": m.get("volume", 0),
    }


def _to_prob(cents) -> Optional[float]:
    """Convert Kalshi cent price (0-99) → probability (0.0-1.0)."""
    if cents is None:
        return None
    try:
        return float(cents) / 100.0
    except (TypeError, ValueError):
        return None


def _parse_orderbook(data: dict) -> dict:
    """
    Extract best bid/ask for YES and NO from Kalshi's orderbook response.

    Kalshi orderbook format:
      {yes: [[price_cents, quantity], ...], no: [[price_cents, quantity], ...]}
    YES bids are sorted descending; YES asks are the NO bids (inverted).
    """
    yes_levels = data.get("yes", [])
    no_levels = data.get("no", [])

    # Best YES bid = highest YES level
    yes_bid = _to_prob(yes_levels[0][0]) if yes_levels else None
    # Best NO bid = highest NO level; YES ask = 1 - best NO bid
    no_bid = _to_prob(no_levels[0][0]) if no_levels else None
    yes_ask = (1.0 - no_bid) if no_bid is not None else None
    # NO ask = 1 - best YES bid
    no_ask = (1.0 - yes_bid) if yes_bid is not None else None

    return {
        "yes_bid": yes_bid,
        "yes_ask": yes_ask,
        "no_bid": no_bid,
        "no_ask": no_ask,
    }
