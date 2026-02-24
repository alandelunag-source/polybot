"""
Odds-API.io client — 2,400 free req/day.
Free tier: max 2 selected bookmakers. Status filter: "pending" = upcoming.
Base: https://api.odds-api.io/v3/
"""
from __future__ import annotations

import time
import logging
from typing import Any, Optional
import requests
from config import settings

logger = logging.getLogger(__name__)
SESSION = requests.Session()
_BASE = "https://api.odds-api.io/v3"
_MIN_CALL_INTERVAL = 0.25  # seconds between API calls (avoid burst 429s)
_last_call_ts: float = 0.0

# Process-lifetime caches
_active_sports_cache: Optional[list[str]] = None   # "sport/league" composites
_leagues_cache: dict[str, list[str]] = {}           # sport_slug → [league_slug]
_selected_bookmakers: Optional[str] = None          # comma-joined bookmaker names

# Short-lived caches
_events_cache: dict[str, tuple[float, list]] = {}   # "sport/league" → (ts, events)
_odds_cache: dict[str, tuple[float, Any]] = {}      # event_id → (ts, response)
_EVENTS_TTL = 600.0
_ODDS_TTL   = 300.0

# Line movement snapshots: event_id → {home_prob, away_prob}
_prob_snapshot: dict[str, dict] = {}


def _get(path: str, params: Optional[dict] = None) -> Any:
    global _last_call_ts
    if not settings.ODDS_API_IO_KEY:
        raise ValueError("ODDS_API_IO_KEY not set. Get a free key at odds-api.io")
    # Simple rate limiter to avoid burst 429s
    wait = _MIN_CALL_INTERVAL - (time.monotonic() - _last_call_ts)
    if wait > 0:
        time.sleep(wait)
    p = {"apiKey": settings.ODDS_API_IO_KEY}
    if params:
        p.update(params)
    r = SESSION.get(f"{_BASE}{path}", params=p, timeout=15)
    _last_call_ts = time.monotonic()
    r.raise_for_status()
    return r.json()


def _get_selected_bookmakers() -> str:
    """Return comma-joined bookmaker names that are selected for this account."""
    global _selected_bookmakers
    if _selected_bookmakers is not None:
        return _selected_bookmakers
    try:
        data = _get("/bookmakers/selected")
        names = data.get("bookmakers", [])
        _selected_bookmakers = ",".join(names)
        logger.info("[OddsApiIo] selected bookmakers: %s", _selected_bookmakers)
    except Exception as exc:
        logger.warning("[OddsApiIo] bookmakers/selected failed: %s", exc)
        _selected_bookmakers = ""
    return _selected_bookmakers


# ---------------------------------------------------------------------------
# Discovery (process-lifetime cached)
# ---------------------------------------------------------------------------

def _get_leagues(sport_slug: str) -> list[str]:
    if sport_slug in _leagues_cache:
        return _leagues_cache[sport_slug]
    try:
        data = _get("/leagues", {"sport": sport_slug})
        slugs = [x["slug"] for x in data if x.get("eventsCount", 0) > 0]
        _leagues_cache[sport_slug] = slugs
        logger.info("[OddsApiIo] %s: %d leagues", sport_slug, len(slugs))
    except Exception as exc:
        logger.warning("[OddsApiIo] leagues failed for %s: %s", sport_slug, exc)
        _leagues_cache[sport_slug] = []
    return _leagues_cache[sport_slug]


def get_active_sport_keys() -> list[str]:
    """Return 'sport_slug/league_slug' composites for every league with events."""
    global _active_sports_cache
    if _active_sports_cache is not None:
        return _active_sports_cache
    try:
        sports = _get("/sports")
        sport_slugs = [s["slug"] for s in sports]
    except Exception as exc:
        logger.warning("[OddsApiIo] sports fetch failed: %s", exc)
        return []
    composites = [
        f"{slug}/{league}"
        for slug in sport_slugs
        for league in _get_leagues(slug)
    ]
    _active_sports_cache = composites
    logger.info("[OddsApiIo] %d sport/league combos", len(composites))
    return composites


# ---------------------------------------------------------------------------
# Odds parsing — actual v3 response format:
# {
#   "id": 123, "home": "Team A", "away": "Team B",
#   "bookmakers": {
#     "BookmakerName": [{"name": "ML", "odds": [{"home": "2.50", "draw": "3.10", "away": "2.80"}]}]
#   }
# }
# ---------------------------------------------------------------------------

def _decimal_to_prob(d) -> float:
    try:
        f = float(d)
        return 1.0 / f if f > 1.0 else 0.0
    except (TypeError, ValueError, ZeroDivisionError):
        return 0.0


def _normalize(probs: list[float]) -> list[float]:
    total = sum(probs)
    return [p / total for p in probs] if total > 0 else probs


def _parse_odds_response(data: dict) -> tuple[list, list, list]:
    """
    Returns ([home_probs], [away_probs], [draw_probs]) across bookmakers.
    Looks for the moneyline market ("ML", "1X2", "h2h", or similar).
    """
    bookmakers = data.get("bookmakers", {})
    if not isinstance(bookmakers, dict):
        return [], [], []

    home_p, away_p, draw_p = [], [], []
    ml_names = {"ml", "1x2", "h2h", "match winner", "moneyline", "match result"}

    for bk_name, markets in bookmakers.items():
        if not isinstance(markets, list):
            continue
        for mkt in markets:
            if mkt.get("name", "").lower() not in ml_names:
                continue
            for odds_entry in mkt.get("odds", []):
                h = _decimal_to_prob(odds_entry.get("home"))
                a = _decimal_to_prob(odds_entry.get("away"))
                d = _decimal_to_prob(odds_entry.get("draw"))
                if h > 0 and a > 0:
                    raw = [h, a]
                    if d > 0:
                        raw.append(d)
                    normed = _normalize(raw)
                    home_p.append(normed[0])
                    away_p.append(normed[1])
                    if d > 0:
                        draw_p.append(normed[2])
                    break  # first valid ML entry per bookmaker

    return home_p, away_p, draw_p


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def get_odds_with_movement(sport_league: str) -> list[dict]:
    """
    sport_league: "sport_slug/league_slug", e.g. "football/england-premier-league"

    Returns list of event dicts:
      event_id, sport_key, commence_time, home_team, away_team,
      home_prob, away_prob, draw_prob, bookmaker_count,
      home_line_move, away_line_move
    """
    parts = sport_league.split("/", 1)
    if len(parts) != 2:
        return []
    sport_slug, league_slug = parts

    now = time.monotonic()

    # --- Events (cached 10 min) ---
    cached = _events_cache.get(sport_league)
    if cached and now - cached[0] < _EVENTS_TTL:
        events = cached[1]
    else:
        try:
            raw = _get("/events", {"sport": sport_slug, "league": league_slug, "status": "pending"})
            events = raw if isinstance(raw, list) else raw.get("data") or raw.get("events") or []
            _events_cache[sport_league] = (now, events)
        except Exception as exc:
            logger.warning("[OddsApiIo] events failed %s: %s", sport_league, exc)
            return []

    if not events:
        return []

    bookmakers_param = _get_selected_bookmakers()
    if not bookmakers_param:
        logger.warning("[OddsApiIo] no bookmakers selected, skipping odds")
        return []

    results = []
    for evt in events:
        eid      = str(evt.get("id") or "")
        home     = evt.get("home") or evt.get("home_team") or ""
        away     = evt.get("away") or evt.get("away_team") or ""
        commence = evt.get("date") or evt.get("commence_time") or ""
        if not (eid and home and away):
            continue

        # --- Odds (cached 5 min) ---
        odds_cached = _odds_cache.get(eid)
        if odds_cached and now - odds_cached[0] < _ODDS_TTL:
            raw_odds = odds_cached[1]
        else:
            try:
                raw_odds = _get("/odds", {"eventId": eid, "bookmakers": bookmakers_param})
                _odds_cache[eid] = (now, raw_odds)
            except Exception as exc:
                logger.debug("[OddsApiIo] odds failed evt %s: %s", eid, exc)
                continue

        home_probs, away_probs, draw_probs = _parse_odds_response(raw_odds)
        if not home_probs:
            continue

        home_prob = sum(home_probs) / len(home_probs)
        away_prob = sum(away_probs) / len(away_probs)
        draw_prob = (sum(draw_probs) / len(draw_probs)) if draw_probs else None

        prev = _prob_snapshot.get(eid)
        home_move = (home_prob - prev["home_prob"]) if prev else 0.0
        away_move = (away_prob - prev["away_prob"]) if prev else 0.0
        _prob_snapshot[eid] = {"home_prob": home_prob, "away_prob": away_prob}

        results.append({
            "event_id":        eid,
            "sport_key":       sport_slug,
            "commence_time":   commence,
            "home_team":       home,
            "away_team":       away,
            "home_prob":       home_prob,
            "away_prob":       away_prob,
            "draw_prob":       draw_prob,
            "bookmaker_count": len(home_probs),
            "home_line_move":  home_move,
            "away_line_move":  away_move,
        })

    logger.info("[OddsApiIo] %s: %d events with odds", sport_league, len(results))
    return results
