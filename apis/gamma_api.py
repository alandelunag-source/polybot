"""
Polymarket Gamma API client for market discovery and metadata.
Endpoint: https://gamma-api.polymarket.com
"""
from __future__ import annotations

from typing import Any, Optional
import requests
from config import settings

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "polybot/1.0"})


def _get(path: str, params: Optional[dict] = None) -> Any:
    url = f"{settings.GAMMA_ENDPOINT}{path}"
    resp = SESSION.get(url, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


def get_active_markets(
    limit: int = 100,
    offset: int = 0,
    tag: Optional[str] = None,
) -> list[dict]:
    """
    Fetch active (open) Gamma markets with pagination.
    Returns a list of market dicts.
    """
    params: dict = {
        "active": "true",
        "closed": "false",
        "limit": limit,
        "offset": offset,
    }
    if tag:
        params["tag"] = tag
    return _get("/markets", params=params)


def search_markets(keyword: str, limit: int = 50) -> list[dict]:
    """Full-text search across market questions."""
    params = {"q": keyword, "limit": limit, "active": "true"}
    return _get("/markets", params=params)


def get_market_by_id(market_id: str) -> dict:
    """Fetch a single market by its Gamma market ID."""
    return _get(f"/markets/{market_id}")


def get_events(
    limit: int = 50,
    offset: int = 0,
    tag: Optional[str] = None,
) -> list[dict]:
    """Fetch grouped events (each event can contain multiple markets)."""
    params: dict = {"limit": limit, "offset": offset, "active": "true"}
    if tag:
        params["tag"] = tag
    return _get("/events", params=params)


def get_sports_markets(sport_keyword: str, limit: int = 100) -> list[dict]:
    """
    Convenience: search for sports-related markets by keyword.
    sport_keyword examples: 'NFL', 'NBA', 'Premier League', 'MLB'
    """
    return search_markets(sport_keyword, limit=limit)


def iter_all_active_markets(batch_size: int = 100) -> list[dict]:
    """
    Collect all active markets by paginating through the Gamma API.
    Returns the full list (up to MAX_MARKETS_PER_SCAN).
    """
    all_markets: list[dict] = []
    offset = 0
    cap = settings.MAX_MARKETS_PER_SCAN

    while len(all_markets) < cap:
        batch = get_active_markets(limit=batch_size, offset=offset)
        if not batch:
            break
        all_markets.extend(batch)
        if len(batch) < batch_size:
            break
        offset += batch_size

    return all_markets[:cap]
