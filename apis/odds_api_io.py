"""
Odds-API.io client — 2,400 free requests/day, 250+ bookmakers.
Used exclusively by the standalone value betting strategy.
Docs: https://odds-api.io/sports-betting-api
"""
from __future__ import annotations

import time
import logging
from typing import Any, Optional
import requests
from config import settings

logger = logging.getLogger(__name__)

SESSION = requests.Session()
_BASE = "https://odds-api.io/v1"

# Process-lifetime cache of active sport keys
_active_sports_cache: Optional[list[str]] = None

# Per-event prob snapshots for line movement detection
# { event_id -> {"home_prob": float, "away_prob": float, "ts": float} }
_prob_snapshot: dict[str, dict] = {}


def _get(path: str, params: Optional[dict] = None) -> Any:
    if not settings.ODDS_API_IO_KEY:
        raise ValueError("ODDS_API_IO_KEY is not set. Get a free key at https://odds-api.io")
    base_params = {"apiKey": settings.ODDS_API_IO_KEY}
    if params:
        base_params.update(params)
    resp = SESSION.get(f"{_BASE}{path}", params=base_params, timeout=15)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Sport discovery
# ---------------------------------------------------------------------------

def get_active_sport_keys(exclude_outrights: bool = True) -> list[str]:
    """
    Return all active sport keys. Cached for the lifetime of the process.
    Falls back to settings.SUPPORTED_SPORTS on failure.
    """
    global _active_sports_cache
    if _active_sports_cache is not None:
        return _active_sports_cache

    try:
        sports = _get("/sports")
        keys = [
            s["key"] for s in sports
            if s.get("active", False)
            and not (exclude_outrights and s.get("has_outrights", False))
        ]
        _active_sports_cache = keys
        logger.info("[OddsApiIo] Discovered %d active sports", len(keys))
        return keys
    except Exception as exc:
        logger.warning("[OddsApiIo] Sport discovery failed, using SUPPORTED_SPORTS: %s", exc)
        return list(settings.SUPPORTED_SPORTS)


# ---------------------------------------------------------------------------
# Odds + line movement
# ---------------------------------------------------------------------------

def _decimal_to_prob(decimal_odds: float) -> float:
    return 1.0 / decimal_odds if decimal_odds > 0 else 0.0


def _normalize(probs: list[float]) -> list[float]:
    total = sum(probs)
    return [p / total for p in probs] if total > 0 else probs


def get_odds_with_movement(sport: str) -> list[dict]:
    """
    Fetch consensus odds for a sport and enrich with line movement vs
    the previous call (stored in _prob_snapshot).

    Returns list of event dicts:
    {
        "event_id", "sport_key", "commence_time",
        "home_team", "away_team",
        "home_prob", "away_prob", "draw_prob",
        "bookmaker_count",
        "home_line_move",   # current - previous (0.0 on first call)
        "away_line_move",
    }
    """
    try:
        events = _get(
            f"/sports/{sport}/odds",
            params={"regions": "us,uk,eu", "markets": "h2h", "oddsFormat": "decimal"},
        )
    except Exception as exc:
        logger.warning("[OddsApiIo] Odds fetch failed for %s: %s", sport, exc)
        return []

    now = time.monotonic()
    results = []

    for event in events:
        home = event.get("home_team", "")
        away = event.get("away_team", "")
        bookmakers = event.get("bookmakers", [])
        if not bookmakers:
            continue

        home_probs, away_probs, draw_probs = [], [], []
        for bk in bookmakers:
            for mkt in bk.get("markets", []):
                if mkt.get("key") != "h2h":
                    continue
                outcomes = {o["name"]: o["price"] for o in mkt.get("outcomes", [])}
                h = outcomes.get(home)
                a = outcomes.get(away)
                d = outcomes.get("Draw")
                if h and a:
                    raw = [_decimal_to_prob(h), _decimal_to_prob(a)]
                    if d:
                        raw.append(_decimal_to_prob(d))
                    normed = _normalize(raw)
                    home_probs.append(normed[0])
                    away_probs.append(normed[1])
                    if d:
                        draw_probs.append(normed[2])

        if not home_probs:
            continue

        home_prob = sum(home_probs) / len(home_probs)
        away_prob = sum(away_probs) / len(away_probs)
        draw_prob = (sum(draw_probs) / len(draw_probs)) if draw_probs else None

        eid = event.get("id", "")
        prev = _prob_snapshot.get(eid)
        home_move = (home_prob - prev["home_prob"]) if prev else 0.0
        away_move = (away_prob - prev["away_prob"]) if prev else 0.0

        _prob_snapshot[eid] = {"home_prob": home_prob, "away_prob": away_prob, "ts": now}

        results.append({
            "event_id":       eid,
            "sport_key":      event.get("sport_key", sport),
            "commence_time":  event.get("commence_time", ""),
            "home_team":      home,
            "away_team":      away,
            "home_prob":      home_prob,
            "away_prob":      away_prob,
            "draw_prob":      draw_prob,
            "bookmaker_count": len(home_probs),
            "home_line_move": home_move,
            "away_line_move": away_move,
        })

    return results
