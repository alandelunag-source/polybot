"""
Sports odds divergence scanner.

Compares Polymarket implied probabilities against consensus bookmaker lines
from The Odds API. Flags markets where the divergence exceeds the threshold,
indicating potential value on one side.

Matching logic:
  - Search Gamma API for sports markets containing team names
  - Align by home/away team names (fuzzy string match)
  - Compare Polymarket mid-price with bookmaker implied probability
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Optional

from apis import gamma_api, odds_api, clob_client, extract_token_ids
from config import settings

logger = logging.getLogger(__name__)


@dataclass
class DivergenceSignal:
    market_id: str
    question: str
    team: str
    poly_prob: float       # Polymarket mid-price (implied probability)
    book_prob: float       # Consensus bookmaker implied probability
    delta: float           # book_prob - poly_prob  (positive = Poly underpriced)
    side: str              # "YES" if poly underpriced, "NO" if overpriced
    sport: str
    bookmaker_count: int

    def __str__(self) -> str:
        direction = "UNDERPRICED" if self.delta > 0 else "OVERPRICED"
        return (
            f"[DIVERGENCE] {self.question[:60]}\n"
            f"  Team: {self.team}  Side: {self.side}  ({direction})\n"
            f"  Polymarket={self.poly_prob:.4f}  Books={self.book_prob:.4f}  "
            f"Delta={self.delta:+.4f}"
        )


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _poly_mid_price(token_id: str) -> Optional[float]:
    """Return mid-price (average of best bid and best ask) for a token."""
    try:
        book = clob_client.get_order_book(token_id)
        bids = book.get("bids", [])
        asks = book.get("asks", [])
        best_bid = max((float(b["price"]) for b in bids), default=None)
        best_ask = min((float(a["price"]) for a in asks), default=None)
        if best_bid is not None and best_ask is not None:
            return (best_bid + best_ask) / 2
        return best_bid or best_ask
    except Exception as exc:
        logger.debug("Mid-price fetch failed: %s", exc)
        return None


def _find_matching_poly_market(
    home: str, away: str, poly_markets: list[dict]
) -> Optional[dict]:
    """
    Find the best-matching Polymarket market for a given home/away matchup.
    Uses fuzzy name matching on the market question.
    """
    best_score = 0.0
    best_market = None
    query = f"{home} {away}".lower()

    for mkt in poly_markets:
        question = mkt.get("question", "").lower()
        score = _similarity(query, question)
        if score > best_score:
            best_score = score
            best_market = mkt

    # Require reasonable match quality
    if best_score < 0.4:
        return None
    return best_market


def find_divergences(
    sport: str,
    poly_markets: Optional[list[dict]] = None,
) -> list[DivergenceSignal]:
    """
    Find sports markets where Polymarket price diverges from bookmaker consensus.

    Args:
        sport: The Odds API sport key, e.g. 'soccer_epl'
        poly_markets: Pre-fetched Gamma markets (fetched automatically if None)

    Returns:
        List of DivergenceSignal with flagged opportunities.
    """
    logger.info("Fetching bookmaker consensus for sport: %s", sport)
    try:
        book_events = odds_api.get_consensus_probs(sport)
    except Exception as exc:
        logger.warning("Odds API failed for %s: %s", sport, exc)
        return []

    if not book_events:
        logger.info("No bookmaker events found for %s", sport)
        return []

    if poly_markets is None:
        sport_label = sport.replace("_", " ").title()
        logger.info("Fetching Polymarket sports markets for: %s", sport_label)
        poly_markets = gamma_api.get_sports_markets(sport_label)

    signals: list[DivergenceSignal] = []

    for event in book_events:
        home = event["home_team"]
        away = event["away_team"]
        matched = _find_matching_poly_market(home, away, poly_markets)
        if not matched:
            logger.debug("No Polymarket match for: %s vs %s", home, away)
            continue

        yes_tid, _ = extract_token_ids(matched)
        if not yes_tid:
            continue

        poly_prob = _poly_mid_price(yes_tid)
        if poly_prob is None:
            continue

        # Determine which team the YES outcome corresponds to
        # Heuristic: if question contains home team name, YES = home win
        question = matched.get("question", "")
        if home.lower() in question.lower():
            book_prob = event["home_prob"]
            team = home
        elif away.lower() in question.lower():
            book_prob = event["away_prob"]
            team = away
        else:
            # Default to home
            book_prob = event["home_prob"]
            team = home

        delta = book_prob - poly_prob
        if abs(delta) < settings.DIVERGENCE_THRESHOLD:
            continue

        side = "YES" if delta > 0 else "NO"
        signal = DivergenceSignal(
            market_id=matched.get("id", ""),
            question=question,
            team=team,
            poly_prob=poly_prob,
            book_prob=book_prob,
            delta=delta,
            side=side,
            sport=sport,
            bookmaker_count=event["bookmaker_count"],
        )
        signals.append(signal)
        logger.info("Divergence found: %s", signal)

    logger.info(
        "Divergence scan for %s: %d signals from %d events",
        sport,
        len(signals),
        len(book_events),
    )
    return signals


def scan_all_sports() -> list[DivergenceSignal]:
    """Run divergence scan across all configured sports."""
    all_signals: list[DivergenceSignal] = []
    for sport in settings.SUPPORTED_SPORTS:
        all_signals.extend(find_divergences(sport))
    return all_signals
