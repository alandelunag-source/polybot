"""
Intra-market arbitrage scanner.

Polymarket binary markets have separate YES and NO order books.
If: best_ask_YES + best_ask_NO < 1.0  →  guaranteed profit at resolution.

Example:
  YES ask = $0.45, NO ask = $0.52  →  total cost $0.97, guaranteed $1.00 payout
  Profit = $0.03 per share (3.09% ROI)

Accounts for ~2% fee on each leg (Polymarket charges 2% on maker fills).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import logging

from apis import clob_client, gamma_api, extract_token_ids
from apis.book_cache import BookCache
from config import settings

logger = logging.getLogger(__name__)

POLYMARKET_FEE = 0.02  # 2% per side


@dataclass
class ArbitrageOpportunity:
    market_id: str
    question: str
    yes_token_id: str
    no_token_id: str
    yes_ask: float
    no_ask: float
    raw_spread: float        # 1 - yes_ask - no_ask  (before fees)
    net_spread: float        # after fees
    expected_profit_pct: float

    def __str__(self) -> str:
        return (
            f"[ARB] {self.question[:60]}\n"
            f"  YES ask={self.yes_ask:.4f}  NO ask={self.no_ask:.4f}\n"
            f"  Raw spread={self.raw_spread:.4f}  Net={self.net_spread:.4f}  "
            f"Profit={self.expected_profit_pct:.2f}%"
        )


def _best_ask(order_book: dict) -> Optional[float]:
    """Extract the best (lowest) ask price from an order book response."""
    asks = order_book.get("asks", [])
    if not asks:
        return None
    prices = [float(a.get("price", 1.0)) for a in asks]
    return min(prices)


def scan_market(market: dict) -> Optional[ArbitrageOpportunity]:
    """
    Check a single Gamma market dict for intra-market arbitrage.
    Returns an ArbitrageOpportunity if one exists, else None.
    """
    yes_tid, no_tid = extract_token_ids(market)
    if not yes_tid or not no_tid:
        return None

    try:
        yes_book = clob_client.get_order_book(yes_tid)
        no_book = clob_client.get_order_book(no_tid)
    except Exception as exc:
        logger.debug("Order book fetch failed for %s: %s", market.get("id"), exc)
        return None

    yes_ask = _best_ask(yes_book)
    no_ask = _best_ask(no_book)

    if yes_ask is None or no_ask is None:
        return None

    raw_spread = 1.0 - yes_ask - no_ask
    # Each leg costs fee * price; approximate net cost increase
    fee_cost = POLYMARKET_FEE * (yes_ask + no_ask)
    net_spread = raw_spread - fee_cost

    if net_spread < settings.ARB_MIN_SPREAD:
        return None

    cost_with_fees = yes_ask + no_ask + fee_cost
    expected_profit_pct = (net_spread / cost_with_fees) * 100

    return ArbitrageOpportunity(
        market_id=market.get("id", ""),
        question=market.get("question", "Unknown"),
        yes_token_id=yes_tid,
        no_token_id=no_tid,
        yes_ask=yes_ask,
        no_ask=no_ask,
        raw_spread=raw_spread,
        net_spread=net_spread,
        expected_profit_pct=expected_profit_pct,
    )


def check_arb_from_cache(
    market: dict,
    yes_token_id: str,
    no_token_id: str,
    cache: BookCache,
) -> Optional[ArbitrageOpportunity]:
    """
    Cache-backed arb check — zero REST calls, called on every WS price update.

    Reads best asks directly from the BookCache instead of hitting the CLOB API.
    Returns an ArbitrageOpportunity if one exists, else None.
    """
    yes_ask = cache.best_ask(yes_token_id)
    no_ask = cache.best_ask(no_token_id)

    if yes_ask is None or no_ask is None:
        return None

    raw_spread = 1.0 - yes_ask - no_ask
    fee_cost = POLYMARKET_FEE * (yes_ask + no_ask)
    net_spread = raw_spread - fee_cost

    if net_spread < settings.ARB_MIN_SPREAD:
        return None

    cost_with_fees = yes_ask + no_ask + fee_cost
    expected_profit_pct = (net_spread / cost_with_fees) * 100

    return ArbitrageOpportunity(
        market_id=market.get("id", ""),
        question=market.get("question", "Unknown"),
        yes_token_id=yes_token_id,
        no_token_id=no_token_id,
        yes_ask=yes_ask,
        no_ask=no_ask,
        raw_spread=raw_spread,
        net_spread=net_spread,
        expected_profit_pct=expected_profit_pct,
    )


def scan_markets(markets: Optional[list[dict]] = None) -> list[ArbitrageOpportunity]:
    """
    Scan a list of Gamma market dicts for arbitrage opportunities.
    If markets is None, fetches all active markets automatically.
    """
    if markets is None:
        logger.info("Fetching active markets from Gamma API...")
        markets = gamma_api.iter_all_active_markets()

    logger.info("Scanning %d markets for arbitrage...", len(markets))
    opportunities: list[ArbitrageOpportunity] = []

    for mkt in markets:
        opp = scan_market(mkt)
        if opp:
            opportunities.append(opp)
            logger.info("Found opportunity: %s", opp)

    logger.info(
        "Arbitrage scan complete: %d/%d markets flagged",
        len(opportunities),
        len(markets),
    )
    return opportunities
