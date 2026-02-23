"""
Value betting strategy.

Takes divergence signals from sports_divergence.py and:
  1. Filters by minimum edge (book_prob - poly_prob)
  2. Sizes positions using fractional Kelly criterion
  3. Respects risk limits from execution/risk.py

Kelly Criterion:
  f = (b*p - q) / b
  where:
    b = net odds (payout per unit risked - 1)
    p = true probability of winning (bookmaker consensus)
    q = 1 - p

For binary Polymarket outcomes priced at `price` USDC per share:
  b = (1 - price) / price   (you risk `price`, win `1 - price`)
  f* = (p - price) / (1 - price)   (simplified for binary bets)

We use fractional Kelly (default: 25%) to reduce variance.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from strategies.sports_divergence import DivergenceSignal
from config import settings

logger = logging.getLogger(__name__)


@dataclass
class ValueBet:
    signal: DivergenceSignal
    edge: float           # book_prob - poly_prob (true edge)
    kelly_fraction: float # fraction of bankroll suggested by Kelly
    suggested_size_usdc: float
    capped_size_usdc: float   # after applying MAX_POSITION_USDC

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
    """
    Compute the Kelly-optimal bet size as a dollar amount.

    Args:
        true_prob:    Estimated true probability of the outcome
        market_price: Current Polymarket price (cost per share)
        bankroll:     Available capital in USDC
        fraction:     Kelly multiplier (0.25 = quarter-Kelly)

    Returns:
        Suggested bet size in USDC (before position cap).
    """
    if market_price <= 0 or market_price >= 1:
        return 0.0
    # Net odds: for every $price risked, you win $(1 - price) if correct
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
    """
    Convert divergence signals into sized value bet recommendations.

    Args:
        signals:  Output of sports_divergence.find_divergences()
        bankroll: Total available USDC capital
        min_edge: Minimum edge required to place a bet

    Returns:
        Sorted list of ValueBet (highest edge first).
    """
    bets: list[ValueBet] = []

    for sig in signals:
        edge = abs(sig.delta)
        if edge < min_edge:
            continue

        # For YES bets: price = poly_prob; for NO bets: price = 1 - poly_prob
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
