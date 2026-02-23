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


# --- Sport rotation ---

_sport_cursor: int = 0


def _get_sports_batch() -> list[str]:
    global _sport_cursor
    from apis.odds_api_io import get_active_sport_keys
    all_sports = get_active_sport_keys()
    if not all_sports:
        return []
    n = settings.VALUE_SPORTS_PER_CYCLE
    batch = all_sports[_sport_cursor: _sport_cursor + n]
    if len(batch) < n:
        batch += all_sports[: n - len(batch)]
    _sport_cursor = (_sport_cursor + n) % len(all_sports)
    return batch


# --- Core scan ---

def scan_sport_for_value(
    sport: str,
    poly_markets: Optional[list[dict]] = None,
) -> list[ValueSignal]:
    """
    Scan one sport for standalone value opportunities.
    Reuses Polymarket matching + mid-price logic from sports_divergence.
    """
    from apis import gamma_api
    from apis.odds_api_io import get_odds_with_movement

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

        # Determine which team YES maps to — same heuristic as sports_divergence
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

        # Hard gate: minimum edge required
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
            market_id=yes_tid,
            question=question,
            poly_prob=poly_prob,
            side=side,
            sport=sport,
            home_team=home,
            away_team=away,
            commence_time=event.get("commence_time", ""),
            book_prob=book_prob,
            raw_edge=raw_edge,
            bookmaker_count=event["bookmaker_count"],
            line_move=line_move,
            composite_score=score,
        )
        signals.append(sig)
        logger.info("[ValueScan] Signal: %s", sig)

    logger.info("[ValueScan] %s: %d signals from %d events", sport, len(signals), len(events))
    return signals


def standalone_value_scan(bankroll: float) -> list[ValueBet]:
    """
    Scan a rotating batch of sports for value opportunities.
    Returns ValueBet list sorted by composite_score descending.
    """
    sports = _get_sports_batch()
    if not sports:
        logger.warning("[ValueScan] No sports available")
        return []

    all_signals: list[ValueSignal] = []
    for sport in sports:
        try:
            all_signals.extend(scan_sport_for_value(sport))
        except Exception as exc:
            logger.warning("[ValueScan] Error scanning %s: %s", sport, exc)

    bets: list[ValueBet] = []
    for sig in all_signals:
        market_price = sig.poly_prob if sig.side == "YES" else (1.0 - sig.poly_prob)
        true_prob    = sig.book_prob if sig.side == "YES" else (1.0 - sig.book_prob)
        suggested    = kelly_size(true_prob, market_price, bankroll)
        if suggested <= 0:
            continue
        capped = min(suggested, settings.MAX_POSITION_USDC)
        bets.append(ValueBet(
            signal=sig,
            edge=sig.raw_edge,
            kelly_fraction=suggested / bankroll if bankroll > 0 else 0,
            suggested_size_usdc=suggested,
            capped_size_usdc=capped,
        ))

    bets.sort(key=lambda b: b.signal.composite_score, reverse=True)
    logger.info("[ValueScan] standalone_value_scan: %d bets across %s", len(bets), sports)
    return bets
