"""
The Odds API client — fetches bookmaker odds and converts to implied probabilities.
Documentation: https://the-odds-api.com/liveapi/guides/v4/
"""
from __future__ import annotations

from typing import Any, Optional
import requests
from config import settings

SESSION = requests.Session()


def _get(path: str, params: Optional[dict] = None) -> Any:
    if not settings.ODDS_API_KEY:
        raise ValueError(
            "ODDS_API_KEY is not set. Get a free key at https://the-odds-api.com"
        )
    base_params = {"apiKey": settings.ODDS_API_KEY}
    if params:
        base_params.update(params)
    url = f"{settings.ODDS_API_BASE}{path}"
    resp = SESSION.get(url, params=base_params, timeout=15)
    resp.raise_for_status()
    return resp.json()


def get_sports() -> list[dict]:
    """List all available sports and their keys."""
    return _get("/sports")


def get_odds(
    sport: str,
    regions: str = "us,uk,eu",
    markets: str = "h2h",
    odds_format: str = "decimal",
) -> list[dict]:
    """
    Fetch odds for upcoming events in a sport.

    Args:
        sport: sport key, e.g. 'soccer_epl', 'americanfootball_nfl'
        regions: comma-separated bookmaker regions
        markets: 'h2h' (moneyline), 'spreads', 'totals'
        odds_format: 'decimal' or 'american'

    Returns:
        List of event dicts with bookmaker odds.
    """
    return _get(
        f"/sports/{sport}/odds",
        params={
            "regions": regions,
            "markets": markets,
            "oddsFormat": odds_format,
        },
    )


def decimal_to_implied_prob(decimal_odds: float) -> float:
    """Convert decimal odds to implied probability."""
    if decimal_odds <= 0:
        return 0.0
    return 1.0 / decimal_odds


def normalize_overround(probs: list[float]) -> list[float]:
    """Remove bookmaker overround by normalizing probabilities to sum to 1.0."""
    total = sum(probs)
    if total == 0:
        return probs
    return [p / total for p in probs]


def get_consensus_probs(sport: str) -> list[dict]:
    """
    Fetch odds for a sport and return consensus implied probabilities
    (average across bookmakers, overround-normalized) per event.

    Returns list of:
    {
        "event_id": str,
        "sport_key": str,
        "commence_time": str,
        "home_team": str,
        "away_team": str,
        "home_prob": float,
        "away_prob": float,
        "draw_prob": float | None,
        "bookmaker_count": int,
    }
    """
    events = get_odds(sport)
    results = []

    for event in events:
        home = event.get("home_team", "")
        away = event.get("away_team", "")
        bookmakers = event.get("bookmakers", [])
        if not bookmakers:
            continue

        home_probs, away_probs, draw_probs = [], [], []

        for bk in bookmakers:
            for market in bk.get("markets", []):
                if market.get("key") != "h2h":
                    continue
                outcomes = {o["name"]: o["price"] for o in market.get("outcomes", [])}
                h = outcomes.get(home)
                a = outcomes.get(away)
                d = outcomes.get("Draw")
                if h and a:
                    raw = [decimal_to_implied_prob(h), decimal_to_implied_prob(a)]
                    if d:
                        raw.append(decimal_to_implied_prob(d))
                    normed = normalize_overround(raw)
                    home_probs.append(normed[0])
                    away_probs.append(normed[1])
                    if d:
                        draw_probs.append(normed[2])

        if not home_probs:
            continue

        results.append(
            {
                "event_id": event.get("id", ""),
                "sport_key": event.get("sport_key", sport),
                "commence_time": event.get("commence_time", ""),
                "home_team": home,
                "away_team": away,
                "home_prob": sum(home_probs) / len(home_probs),
                "away_prob": sum(away_probs) / len(away_probs),
                "draw_prob": (
                    sum(draw_probs) / len(draw_probs) if draw_probs else None
                ),
                "bookmaker_count": len(home_probs),
            }
        )

    return results
