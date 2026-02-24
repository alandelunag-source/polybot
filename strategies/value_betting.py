"""
Value betting strategy (v2).

Two modes:
  1. DOWNSTREAM (legacy): Takes DivergenceSignal list from sports_divergence.py
     and sizes them with Kelly. Entry point: find_value_bets(signals, bankroll)

  2. STANDALONE (new): Independent multi-sport edge-finding via Odds-API.io.
     Scores opportunities on edge, bookmaker consensus strength, and line
     movement (smart money proxy). Entry point: standalone_value_scan(bankroll)
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Optional

from strategies.sports_divergence import (
    DivergenceSignal,
    _find_matching_poly_market,
    _poly_mid_price,
)
from config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Existing downstream mode (unchanged)
# ---------------------------------------------------------------------------

@dataclass
class ValueBet:
    signal: Any           # DivergenceSignal | ValueSignal (duck-typed)
    edge: float
    kelly_fraction: float
    suggested_size_usdc: float
    capped_size_usdc: float

    def __str__(self) -> str:
        return (
            f"[VALUE BET] {self.signal.question[:55]}\n"
            f"  Side={self.signal.side}  Edge={self.edge:+.4f}  "
            f"Kelly={self.kelly_fraction:.4f}  "
            f"Size=${self.capped_size_usdc:.2f} USDC"
        )


def kelly_size(
    true_prob: float,
    market_price: float,
    bankroll: float,
    fraction: float = settings.KELLY_FRACTION,
) -> float:
    if market_price <= 0 or market_price >= 1:
        return 0.0
    b = (1.0 - market_price) / market_price
    p = true_prob
    q = 1.0 - p
    raw_kelly = (b * p - q) / b
    if raw_kelly <= 0:
        return 0.0
    return fraction * raw_kelly * bankroll


def find_value_bets(
    signals: list[DivergenceSignal],
    bankroll: float,
    min_edge: float = settings.DIVERGENCE_THRESHOLD,
) -> list[ValueBet]:
    bets: list[ValueBet] = []
    for sig in signals:
        edge = abs(sig.delta)
        if edge < min_edge:
            continue
        if sig.side == "YES":
            market_price = sig.poly_prob
            true_prob = sig.book_prob
        else:
            market_price = 1.0 - sig.poly_prob
            true_prob = 1.0 - sig.book_prob
        suggested = kelly_size(true_prob, market_price, bankroll)
        if suggested <= 0:
            continue
        capped = min(suggested, settings.MAX_POSITION_USDC)
        bet = ValueBet(
            signal=sig,
            edge=edge,
            kelly_fraction=suggested / bankroll if bankroll > 0 else 0,
            suggested_size_usdc=suggested,
            capped_size_usdc=capped,
        )
        bets.append(bet)
        logger.info("Value bet: %s", bet)
    bets.sort(key=lambda b: b.edge, reverse=True)
    logger.info("Value betting: %d bets identified from %d signals", len(bets), len(signals))
    return bets


# ---------------------------------------------------------------------------
# Standalone mode — new
# ---------------------------------------------------------------------------

@dataclass
class ValueSignal:
    """
    Independent value signal. Structurally compatible with DivergenceSignal:
    exposes market_id, poly_prob, side — the three fields main.py accesses
    on bet.signal for order placement.
    """
    # Required by main.py order placement
    market_id: str          # YES token_id
    question: str
    poly_prob: float
    side: str               # "YES" or "NO"

    # Context
    sport: str
    home_team: str
    away_team: str
    commence_time: str

    # Signals
    book_prob: float
    raw_edge: float         # book_prob - poly_prob
    bookmaker_count: int
    line_move: float        # current book_prob - previous (0.0 on first scan)

    # Composite score
    composite_score: float

    def __str__(self) -> str:
        return (
            f"[VALUE SIGNAL] {self.question[:55]}\n"
            f"  Side={self.side}  Edge={self.raw_edge:+.4f}  "
            f"Books={self.bookmaker_count}  LineMov={self.line_move:+.4f}  "
            f"Score={self.composite_score:.3f}"
        )


# --- Scoring functions (each returns 0.0 – 1.0) ---

def _score_edge(raw_edge: float) -> float:
    """Linear scale from VALUE_MIN_EDGE (→0) to 0.30 (→1)."""
    min_e = settings.VALUE_MIN_EDGE
    if raw_edge < min_e:
        return 0.0
    return min(1.0, (raw_edge - min_e) / (0.30 - min_e))


def _score_consensus(bookmaker_count: int) -> float:
    """0 below 3 books, scales to 1.0 at 15+ books."""
    if bookmaker_count < 3:
        return 0.0
    return min(1.0, (bookmaker_count - 3) / (15 - 3))


def _score_line_movement(line_move: float, side: str) -> float:
    """
    0.5 = neutral (no significant movement).
    Higher if move confirms our bet direction (smart money agrees).
    Lower if move goes against us (smart money disagrees).
    """
    threshold = settings.VALUE_LINE_MOVE_THRESHOLD
    if abs(line_move) < threshold:
        return 0.5
    confirms = (line_move > 0 and side == "YES") or (line_move < 0 and side == "NO")
    delta = min(abs(line_move) / 0.10, 0.5)  # max 0.5 swing
    return min(1.0, 0.5 + delta) if confirms else max(0.0, 0.5 - delta)


def _composite_score(edge_s: float, consensus_s: float, line_s: float) -> float:
    return (
        settings.VALUE_WEIGHT_EDGE      * edge_s
        + settings.VALUE_WEIGHT_CONSENSUS * consensus_s
        + settings.VALUE_WEIGHT_LINE      * line_s
    )


# ---------------------------------------------------------------------------
# Outright championship scan — the correct approach for Polymarket
#
# Polymarket has no per-match h2h markets. It has outright winner markets:
#   "Will the Carolina Hurricanes win the 2026 NHL Stanley Cup?"
#   "Will Arsenal win the 2025-26 English Premier League?"
# We compare these to bookmaker outright odds from The Odds API.
# ---------------------------------------------------------------------------

# Maps competition keywords → The Odds API outright sport key
_COMPETITION_MAP: dict[str, str] = {
    "stanley cup":              "icehockey_nhl_championship_winner",
    "nhl":                      "icehockey_nhl_championship_winner",
    "nba championship":         "basketball_nba_championship_winner",
    "nba finals":               "basketball_nba_championship_winner",
    "world series":             "baseball_mlb_world_series_winner",
    "super bowl":               "americanfootball_nfl_super_bowl_winner",
    "fifa world cup":           "soccer_fifa_world_cup_winner",
    "world cup":                "soccer_fifa_world_cup_winner",
    "masters":                  "golf_masters_tournament_winner",
    "pga championship":         "golf_pga_championship_winner",
    "the open":                 "golf_the_open_championship_winner",
    "us open":                  "golf_us_open_winner",
    "ncaab":                    "basketball_ncaab_championship_winner",
    "march madness":            "basketball_ncaab_championship_winner",
}

_OUTRIGHT_RE = re.compile(
    r"Will (?:the )?(.*?) win (?:the )?(.*?)(?:\?|$)", re.IGNORECASE
)

# Cache outright bookmaker probs so one competition = one API call per scan
_outright_cache: dict[str, dict[str, float]] = {}


def _parse_outright_question(question: str) -> Optional[tuple[str, str]]:
    """Return (team_name, competition_name) or None."""
    m = _OUTRIGHT_RE.match(question)
    return (m.group(1).strip(), m.group(2).strip()) if m else None


def _map_competition(comp: str) -> Optional[str]:
    cl = comp.lower()
    for keyword, sport_key in _COMPETITION_MAP.items():
        if keyword in cl:
            return sport_key
    return None


def _fuzzy_match_team(poly_team: str, book_probs: dict[str, float]) -> Optional[tuple[str, float]]:
    """Match Polymarket team name against bookmaker team names."""
    pl = poly_team.lower().strip()
    # Exact
    for bk, prob in book_probs.items():
        if pl == bk.lower():
            return bk, prob
    # Substring
    for bk, prob in book_probs.items():
        bl = bk.lower()
        if pl in bl or bl in pl:
            return bk, prob
    # Word overlap (≥2 words)
    pw = set(pl.split())
    for bk, prob in book_probs.items():
        if len(pw & set(bk.lower().split())) >= 2:
            return bk, prob
    return None


def scan_sport_for_value(
    sport: str,
    poly_markets: Optional[list[dict]] = None,
) -> list[ValueSignal]:
    """
    Legacy per-sport scan — kept for backward compatibility with tests.
    In practice, standalone_value_scan() uses the outright championship flow.
    """
    from apis import gamma_api
    from apis.odds_api import get_odds_with_movement

    try:
        events = get_odds_with_movement(sport)
    except Exception as exc:
        logger.warning("[ValueScan] Odds fetch failed for %s: %s", sport, exc)
        return []
    if not events:
        return []

    if poly_markets is None:
        sport_label = sport.replace("_", " ").title()
        try:
            poly_markets = gamma_api.get_sports_markets(sport_label)
        except Exception as exc:
            logger.warning("[ValueScan] Gamma fetch failed for %s: %s", sport, exc)
            return []

    signals: list[ValueSignal] = []

    for event in events:
        home = event["home_team"]
        away = event["away_team"]

        matched = _find_matching_poly_market(home, away, poly_markets)
        if not matched:
            continue

        from apis import extract_token_ids
        yes_tid, _ = extract_token_ids(matched)
        if not yes_tid:
            continue

        poly_prob = _poly_mid_price(yes_tid)
        if poly_prob is None:
            continue

        question = matched.get("question", "")

        if home.lower() in question.lower():
            book_prob  = event["home_prob"]
            line_move  = event["home_line_move"]
        elif away.lower() in question.lower():
            book_prob  = event["away_prob"]
            line_move  = event["away_line_move"]
        else:
            book_prob  = event["home_prob"]
            line_move  = event["home_line_move"]

        raw_edge = book_prob - poly_prob
        if abs(raw_edge) < settings.VALUE_MIN_EDGE:
            continue

        side = "YES" if raw_edge > 0 else "NO"
        edge_s      = _score_edge(abs(raw_edge))
        consensus_s = _score_consensus(event["bookmaker_count"])
        line_s      = _score_line_movement(line_move, side)
        score       = _composite_score(edge_s, consensus_s, line_s)

        if score < settings.VALUE_MIN_COMPOSITE_SCORE:
            continue

        sig = ValueSignal(
            market_id=yes_tid, question=question, poly_prob=poly_prob, side=side,
            sport=sport, home_team=home, away_team=away,
            commence_time=event.get("commence_time", ""),
            book_prob=book_prob, raw_edge=raw_edge,
            bookmaker_count=event["bookmaker_count"],
            line_move=line_move, composite_score=score,
        )
        signals.append(sig)
        logger.info("[ValueScan] Signal: %s", sig)

    logger.info("[ValueScan] %s: %d signals from %d events", sport, len(signals), len(events))
    return signals


def standalone_value_scan(bankroll: float) -> list[ValueBet]:
    """
    Scan Polymarket championship winner markets vs bookmaker outright odds.

    Strategy:
      1. Fetch active Polymarket markets and filter for "Will X win the Y?" patterns
      2. Map competition name → The Odds API outright sport key
      3. Fetch bookmaker consensus probability for each team (one call per competition)
      4. Score edge + consensus and filter by composite threshold
      5. Size with fractional Kelly and return sorted by composite score
    """
    from apis import gamma_api, extract_token_ids
    from apis.odds_api import get_outright_probs

    # --- Fetch Polymarket markets ---
    poly_markets: list[dict] = []
    for offset in range(0, 500, 100):
        try:
            batch = gamma_api.get_active_markets(limit=100, offset=offset)
        except Exception as exc:
            logger.warning("[ValueScan] Gamma fetch failed at offset %d: %s", offset, exc)
            break
        if not batch:
            break
        poly_markets.extend(batch)

    logger.info("[ValueScan] Polymarket: %d active markets fetched", len(poly_markets))

    # --- Parse outright candidates ---
    # (question, team, sport_key, yes_tid, poly_prob)
    candidates: list[tuple[str, str, str, str, float]] = []
    for mkt in poly_markets:
        question = mkt.get("question", "")
        parsed = _parse_outright_question(question)
        if not parsed:
            continue
        team, comp = parsed
        sport_key = _map_competition(comp)
        if not sport_key:
            continue
        yes_tid, _ = extract_token_ids(mkt)
        if not yes_tid:
            continue
        # Use Gamma bid/ask — avoids CLOB call (outrights often lack a CLOB orderbook)
        try:
            bid = float(mkt.get("bestBid") or 0)
            ask = float(mkt.get("bestAsk") or 1)
        except (TypeError, ValueError):
            continue
        if ask <= 0 or ask <= bid:
            continue
        poly_prob = (bid + ask) / 2.0
        candidates.append((question, team, sport_key, yes_tid, poly_prob))

    logger.info("[ValueScan] %d outright candidates matched", len(candidates))
    if not candidates:
        return []

    # --- Fetch bookmaker outright probs (one call per competition) ---
    sport_keys_needed = {c[2] for c in candidates}
    _outright_cache.clear()
    for sk in sport_keys_needed:
        try:
            _outright_cache[sk] = get_outright_probs(sk)
            logger.info("[ValueScan] %s: %d teams from bookmakers", sk, len(_outright_cache[sk]))
        except Exception as exc:
            logger.warning("[ValueScan] Outright odds failed for %s: %s", sk, exc)

    # --- Score each candidate ---
    signals: list[ValueSignal] = []
    for question, team, sport_key, yes_tid, poly_prob in candidates:
        book_probs = _outright_cache.get(sport_key, {})
        if not book_probs:
            continue

        match = _fuzzy_match_team(team, book_probs)
        if not match:
            continue
        _, book_prob = match

        raw_edge = book_prob - poly_prob
        if abs(raw_edge) < settings.VALUE_MIN_EDGE:
            continue

        side = "YES" if raw_edge > 0 else "NO"
        bk_count = len(book_probs)
        edge_s      = _score_edge(abs(raw_edge))
        consensus_s = _score_consensus(bk_count)
        line_s      = 0.5  # no line-movement signal for outrights
        score       = _composite_score(edge_s, consensus_s, line_s)

        if score < settings.VALUE_MIN_COMPOSITE_SCORE:
            continue

        sig = ValueSignal(
            market_id=yes_tid, question=question, poly_prob=poly_prob, side=side,
            sport=sport_key, home_team=team, away_team="",
            commence_time="", book_prob=book_prob, raw_edge=raw_edge,
            bookmaker_count=bk_count, line_move=0.0, composite_score=score,
        )
        signals.append(sig)
        logger.info("[ValueScan] Signal: %s", sig)

    # --- Convert to ValueBets ---
    bets: list[ValueBet] = []
    for sig in signals:
        market_price = sig.poly_prob if sig.side == "YES" else (1.0 - sig.poly_prob)
        true_prob    = sig.book_prob if sig.side == "YES" else (1.0 - sig.book_prob)
        suggested    = kelly_size(true_prob, market_price, bankroll)
        if suggested <= 0:
            continue
        capped = min(suggested, settings.MAX_POSITION_USDC)
        bets.append(ValueBet(
            signal=sig, edge=abs(sig.raw_edge),
            kelly_fraction=suggested / bankroll if bankroll > 0 else 0,
            suggested_size_usdc=suggested, capped_size_usdc=capped,
        ))

    bets.sort(key=lambda b: b.signal.composite_score, reverse=True)
    logger.info("[ValueScan] standalone_value_scan: %d bets from %d candidates", len(bets), len(candidates))
    return bets
