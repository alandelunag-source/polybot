"""
Political arbitrage strategy: Polymarket vs Kalshi.

Finds the same real-world binary event on both platforms and surfaces
price discrepancies that exceed a configurable edge threshold.

Signal logic:
  - Fuzzy-match Poly market title vs Kalshi market title (ratio >= 70%)
  - Edge A: Poly YES bid > Kalshi YES ask  → buy YES on Kalshi, sell/fade Poly
  - Edge B: Kalshi YES bid > Poly YES ask  → buy YES on Poly, fade Kalshi
  - Gate: edge > POLITICAL_MIN_EDGE (default 3%)
          both markets liquid: spread < POLITICAL_MAX_SPREAD (default 5%)

Usage:
  python main.py scan --strategy political
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from config import settings

logger = logging.getLogger(__name__)


@dataclass
class PoliticalArbSignal:
    poly_market_id: str
    poly_question: str
    kalshi_ticker: str
    kalshi_title: str
    match_score: float          # fuzzy ratio 0-1

    poly_yes_bid: float
    poly_yes_ask: float
    kalshi_yes_bid: float
    kalshi_yes_ask: float

    edge: float                 # positive = actionable edge
    buy_on: str                 # "kalshi" or "poly"
    sell_on: str                # "poly" or "kalshi"

    def __str__(self) -> str:
        direction = f"Buy YES on {self.buy_on.upper()}, fade {self.sell_on.upper()}"
        return (
            f"[POLITICAL ARB] {self.poly_question[:60]}\n"
            f"  Matched: {self.kalshi_title[:60]}  (score={self.match_score:.0%})\n"
            f"  Poly YES bid={self.poly_yes_bid:.3f} ask={self.poly_yes_ask:.3f}\n"
            f"  Kalshi YES bid={self.kalshi_yes_bid:.3f} ask={self.kalshi_yes_ask:.3f}\n"
            f"  Edge={self.edge:.1%}  Action: {direction}"
        )


# ---------------------------------------------------------------------------
# Fuzzy matching (no external dep — pure stdlib ratio)
# ---------------------------------------------------------------------------

def _similarity(a: str, b: str) -> float:
    """
    Token-overlap similarity: |intersection| / |union| of word sets.
    Strips punctuation so "election?" == "election".
    Good enough for matching political event titles without external deps.
    """
    import re
    _clean = lambda s: set(re.sub(r"[^a-z0-9\s]", "", s.lower()).split())
    wa = _clean(a)
    wb = _clean(b)
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


# ---------------------------------------------------------------------------
# Core scan
# ---------------------------------------------------------------------------

def find_political_arb(
    poly_markets: list[dict],
    kalshi_markets: list[dict],
) -> list[PoliticalArbSignal]:
    """
    Cross-match Poly markets with Kalshi markets and find price edges.

    Args:
        poly_markets: raw Gamma API market dicts (must have 'question', 'id')
        kalshi_markets: normalized Kalshi market dicts from kalshi_api.get_markets()

    Returns:
        List of PoliticalArbSignal with edge > POLITICAL_MIN_EDGE.
    """
    from apis.clob_client import get_order_book  # local import to avoid circular

    signals: list[PoliticalArbSignal] = []
    min_edge = settings.POLITICAL_MIN_EDGE
    max_spread = settings.POLITICAL_MAX_SPREAD
    min_match = settings.POLITICAL_MIN_MATCH_SCORE

    for poly in poly_markets:
        poly_q = poly.get("question", "")
        poly_id = poly.get("id", "")
        if not poly_q or not poly_id:
            continue

        # Find best-matching Kalshi market
        best_score = 0.0
        best_kalshi: Optional[dict] = None
        for km in kalshi_markets:
            score = _similarity(poly_q, km.get("title", ""))
            if score > best_score:
                best_score = score
                best_kalshi = km

        if best_kalshi is None or best_score < min_match:
            continue

        # Get live Poly YES prices from CLOB
        from apis import extract_token_ids
        from apis.clob_client import get_order_book
        yes_tid, no_tid = extract_token_ids(poly)
        if not yes_tid:
            continue

        try:
            yes_book = get_order_book(yes_tid)
        except Exception as exc:
            logger.debug("CLOB fetch failed for %s: %s", poly_id, exc)
            continue

        asks = yes_book.get("asks", [])
        bids = yes_book.get("bids", [])
        if not asks or not bids:
            continue

        poly_yes_ask = min(float(a["price"]) for a in asks)
        poly_yes_bid = max(float(b["price"]) for b in bids)
        poly_spread = poly_yes_ask - poly_yes_bid

        # Get live Kalshi prices (orderbook)
        from apis.kalshi_api import get_orderbook
        ticker = best_kalshi["ticker"]
        ob = get_orderbook(ticker)
        if ob is None:
            continue

        kalshi_yes_bid = ob.get("yes_bid") or 0.0
        kalshi_yes_ask = ob.get("yes_ask") or 1.0
        kalshi_spread = kalshi_yes_ask - kalshi_yes_bid

        # Liquidity gate
        if poly_spread > max_spread or kalshi_spread > max_spread:
            continue

        # Edge A: Poly YES bid > Kalshi YES ask → buy on Kalshi
        edge_a = poly_yes_bid - kalshi_yes_ask
        # Edge B: Kalshi YES bid > Poly YES ask → buy on Poly
        edge_b = kalshi_yes_bid - poly_yes_ask

        best_edge = max(edge_a, edge_b)
        if best_edge < min_edge:
            continue

        if edge_a >= edge_b:
            buy_on, sell_on = "kalshi", "poly"
        else:
            buy_on, sell_on = "poly", "kalshi"

        sig = PoliticalArbSignal(
            poly_market_id=poly_id,
            poly_question=poly_q,
            kalshi_ticker=ticker,
            kalshi_title=best_kalshi.get("title", ""),
            match_score=best_score,
            poly_yes_bid=poly_yes_bid,
            poly_yes_ask=poly_yes_ask,
            kalshi_yes_bid=kalshi_yes_bid,
            kalshi_yes_ask=kalshi_yes_ask,
            edge=best_edge,
            buy_on=buy_on,
            sell_on=sell_on,
        )
        signals.append(sig)
        logger.info("Political arb signal: %s  edge=%.1f%%", poly_q[:50], best_edge * 100)

    signals.sort(key=lambda s: s.edge, reverse=True)
    return signals


def scan_political(poly_markets: Optional[list[dict]] = None) -> list[PoliticalArbSignal]:
    """
    Top-level entry point called from main.py.
    Fetches Kalshi markets, fetches Poly markets if not provided, then scans.
    """
    from apis.kalshi_api import get_markets as kalshi_markets
    from apis.gamma_api import iter_all_active_markets

    if poly_markets is None:
        logger.info("Fetching Poly markets...")
        poly_markets = iter_all_active_markets()

    logger.info("Fetching Kalshi markets...")
    kalshi = kalshi_markets()
    logger.info("Matching %d Poly × %d Kalshi markets...", len(poly_markets), len(kalshi))

    return find_political_arb(poly_markets, kalshi)
