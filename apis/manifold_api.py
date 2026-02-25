"""
Manifold Markets API client.

Public API — no authentication required for reading.
Base: https://api.manifold.markets/v0

Key methods:
  get_markets(limit, sort)   -> list of normalized market dicts
  get_orderbook(market_id)   -> {yes_bid, yes_ask, no_bid, no_ask}

Normalized market shape (matches kalshi_api contract):
  {ticker, title, yes_bid, yes_ask, close_time, volume, total_liquidity}

Spread estimation:
  Manifold uses a CFMM (AMM). There is no explicit order book.
  We estimate the effective bid/ask spread from pool liquidity:
    liquidity >= $1 000 → half-spread 1%
    liquidity >= $200   → half-spread 2%
    else                → half-spread 4%
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://api.manifold.markets/v0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _half_spread(total_liquidity: float) -> float:
    """Estimate half the bid-ask spread from pool liquidity (in USD)."""
    if total_liquidity >= 1_000:
        return 0.010
    elif total_liquidity >= 200:
        return 0.020
    else:
        return 0.040


def _normalize_market(m: dict) -> dict:
    """Extract the fields we care about from a raw Manifold market object."""
    prob = m.get("probability")
    liq = float(m.get("totalLiquidity") or 0)
    hs = _half_spread(liq)

    if prob is not None:
        yes_bid = max(0.01, float(prob) - hs)
        yes_ask = min(0.99, float(prob) + hs)
    else:
        yes_bid = None
        yes_ask = None

    close_ms = m.get("closeTime")
    close_str = ""
    if close_ms:
        try:
            close_str = datetime.fromtimestamp(
                close_ms / 1000, tz=timezone.utc
            ).isoformat()
        except Exception:
            close_str = str(close_ms)

    return {
        "ticker": m.get("id", ""),
        "title": m.get("question", ""),
        "yes_bid": yes_bid,
        "yes_ask": yes_ask,
        "close_time": close_str,
        "volume": float(m.get("volume") or 0),
        "total_liquidity": liq,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_markets(limit: int = 500) -> list[dict]:
    """
    Return a list of normalized Manifold binary market dicts,
    sorted by most-recently-bet (active markets first).

    Fetches up to `limit` markets and filters client-side to open BINARY
    markets only (the API no longer supports server-side contractType/filter).

    Normalized shape:
      {ticker, title, yes_bid, yes_ask, close_time, volume, total_liquidity}
    """
    params = {
        "limit": min(limit, 1000),
        "sort": "last-bet-time",
    }
    try:
        r = requests.get(f"{BASE_URL}/markets", params=params, timeout=15)
        r.raise_for_status()
    except requests.RequestException as exc:
        logger.error("Manifold get_markets failed: %s", exc)
        return []

    raw = r.json()
    if not isinstance(raw, list):
        logger.error("Manifold get_markets: unexpected response shape")
        return []

    return [
        _normalize_market(m)
        for m in raw
        if m.get("outcomeType") == "BINARY"
        and not m.get("isResolved", False)
        and m.get("probability") is not None
    ]


def get_orderbook(market_id: str) -> Optional[dict]:
    """
    Fetch the current probability for a single Manifold market and
    return it as a synthetic bid/ask orderbook.

    Returns:
      {yes_bid, yes_ask, no_bid, no_ask}  — prices in [0, 1]
      None if market is resolved, non-binary, or fetch fails.
    """
    try:
        r = requests.get(f"{BASE_URL}/market/{market_id}", timeout=10)
        r.raise_for_status()
    except requests.RequestException as exc:
        logger.debug("Manifold orderbook fetch failed for %s: %s", market_id, exc)
        return None

    m = r.json()
    if m.get("isResolved") or m.get("outcomeType") != "BINARY":
        return None

    prob = m.get("probability")
    if prob is None:
        return None

    liq = float(m.get("totalLiquidity") or 0)
    hs = _half_spread(liq)
    yes_bid = max(0.01, float(prob) - hs)
    yes_ask = min(0.99, float(prob) + hs)

    return {
        "yes_bid": yes_bid,
        "yes_ask": yes_ask,
        "no_bid": 1.0 - yes_ask,
        "no_ask": 1.0 - yes_bid,
    }
