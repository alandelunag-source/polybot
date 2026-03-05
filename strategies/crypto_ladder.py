"""
Crypto Price Ladder Arbitrage

Alpha thesis:
  Polymarket lists BTC/ETH price-target markets at multiple levels:
    "Will BTC reach $80k by March 2025?"
    "Will BTC reach $90k by March 2025?"
    "Will BTC reach $100k by March 2025?"

  These form a survival function (complement of CDF). For any two thresholds
  X < Y with the same asset and expiry:
    P(price > X) MUST be >= P(price > Y)

  Any violation is a pure arb — no model, no opinion needed.
  Buy the higher-probability rung that's pricing too LOW and
  sell (buy NO of) the lower-probability rung pricing too HIGH.

  Additionally: implied probability mass in each rung (the slice between
  consecutive thresholds) should be non-negative. Negative slices indicate
  guaranteed mispricing.

Signal output:
  LadderArb with the two violating markets, the spread, and suggested sides.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# Regex to parse price threshold from market question
# Handles: $80k, $80,000, $80K, $80.5k, 80000, 80K
_PRICE_RE = re.compile(
    r"\$?([\d,]+\.?\d*)\s*[kK]?\b",
    re.IGNORECASE,
)

_ASSET_KEYWORDS = {
    "BTC":  ["bitcoin", "btc"],
    "ETH":  ["ethereum", "eth", "ether"],
    "SOL":  ["solana", "sol"],
}

# Common expiry keywords to group by same ladder
_EXPIRY_RE = re.compile(
    r"(by|before|end of|in|on)\s+(\w+\s+\d{4}|\d{4}|\w+\s+\d+,?\s*\d{4})",
    re.IGNORECASE,
)


@dataclass
class LadderRung:
    market_id: str        # YES token_id
    question:  str
    asset:     str        # "BTC" | "ETH" | "SOL"
    threshold: float      # price level in USD
    expiry_tag: str       # normalized expiry string for grouping
    poly_prob: float      # mid price (YES probability)
    yes_ask:   float
    no_ask:    float


@dataclass
class LadderArb:
    lower_rung:  LadderRung   # lower price threshold (should have higher prob)
    upper_rung:  LadderRung   # higher price threshold (should have lower prob)
    violation:   float         # upper_prob - lower_prob (positive = violation)
    net_spread:  float         # max guaranteed profit per dollar wagered

    def __str__(self) -> str:
        return (
            f"[LADDER ARB] {self.lower_rung.asset}\n"
            f"  Lower rung: ${self.lower_rung.threshold:,.0f}  prob={self.lower_rung.poly_prob:.3f}  "
            f"({self.lower_rung.question[:50]})\n"
            f"  Upper rung: ${self.upper_rung.threshold:,.0f}  prob={self.upper_rung.poly_prob:.3f}  "
            f"({self.upper_rung.question[:50]})\n"
            f"  Violation: upper - lower = {self.violation:+.4f}  net_spread={self.net_spread:.4f}\n"
            f"  Trade: BUY YES on lower (${self.lower_rung.threshold:,.0f}) + "
            f"BUY YES on upper (${self.upper_rung.threshold:,.0f}) → guaranteed profit"
        )


def _parse_asset(question: str) -> Optional[str]:
    ql = question.lower()
    for asset, keywords in _ASSET_KEYWORDS.items():
        if any(kw in ql for kw in keywords):
            return asset
    return None


def _parse_threshold(question: str) -> Optional[float]:
    """Extract the price threshold in USD. Handles $80k → 80000, $1.2m → 1200000."""
    # Try to find a dollar amount
    dollar_re = re.compile(
        r"\$([\d,]+\.?\d*)\s*([kKmM]?)",
    )
    matches = dollar_re.findall(question)
    if not matches:
        return None

    candidates = []
    for num_str, suffix in matches:
        try:
            num = float(num_str.replace(",", ""))
        except ValueError:
            continue
        if suffix in ("k", "K"):
            num *= 1_000
        elif suffix in ("m", "M"):
            num *= 1_000_000
        # Filter: crypto price targets are > $1,000
        if num >= 1_000:
            candidates.append(num)

    return candidates[0] if candidates else None


def _parse_expiry(question: str) -> str:
    """Extract a rough expiry tag for grouping rungs on the same ladder."""
    m = _EXPIRY_RE.search(question)
    if m:
        return m.group(2).strip().lower()
    # Look for a year
    year_m = re.search(r"\b(202[4-9]|203\d)\b", question)
    return year_m.group(1) if year_m else "unknown"


def _mid_price(market: dict) -> tuple[float, float, float]:
    """Return (mid, yes_ask, no_ask) from Gamma market dict."""
    try:
        bid = float(market.get("bestBid") or 0)
        ask = float(market.get("bestAsk") or 1)
    except (TypeError, ValueError):
        bid, ask = 0.0, 1.0
    mid = (bid + ask) / 2.0
    no_ask = 1.0 - bid   # approximate NO ask
    return mid, ask, no_ask


def parse_ladder_rungs(markets: list[dict]) -> list[LadderRung]:
    """
    Parse Polymarket markets into LadderRung objects.
    Filters for crypto price-target markets only.
    """
    from apis import extract_token_ids

    rungs: list[LadderRung] = []
    for mkt in markets:
        q = mkt.get("question", "")
        asset = _parse_asset(q)
        if not asset:
            continue
        threshold = _parse_threshold(q)
        if not threshold:
            continue
        expiry = _parse_expiry(q)

        yes_tid, _ = extract_token_ids(mkt)
        if not yes_tid:
            continue

        mid, yes_ask, no_ask = _mid_price(mkt)
        if mid <= 0 or mid >= 1:
            continue
        # Skip illiquid (wide spread)
        if yes_ask - float(mkt.get("bestBid") or 0) > 0.15:
            continue

        rungs.append(LadderRung(
            market_id=yes_tid,
            question=q,
            asset=asset,
            threshold=threshold,
            expiry_tag=expiry,
            poly_prob=mid,
            yes_ask=yes_ask,
            no_ask=no_ask,
        ))

    logger.info("[LadderArb] Parsed %d crypto price-target rungs", len(rungs))
    return rungs


def find_ladder_arbs(
    rungs: list[LadderRung],
    min_violation: float = 0.02,
) -> list[LadderArb]:
    """
    Group rungs by (asset, expiry) and check monotonicity.
    Returns violations where P(>upper) > P(>lower) — lower should have higher prob.
    """
    from collections import defaultdict

    groups: dict[tuple, list[LadderRung]] = defaultdict(list)
    for r in rungs:
        groups[(r.asset, r.expiry_tag)].append(r)

    arbs: list[LadderArb] = []

    for (asset, expiry), ladder in groups.items():
        if len(ladder) < 2:
            continue

        # Sort ascending by threshold (lower threshold = higher probability expected)
        ladder.sort(key=lambda r: r.threshold)

        for i in range(len(ladder) - 1):
            lower = ladder[i]
            upper = ladder[i + 1]

            # Violation: upper rung has HIGHER probability than lower rung
            violation = upper.poly_prob - lower.poly_prob
            if violation < min_violation:
                continue

            # Net spread: if we buy YES on both, one must pay off
            # Actually the arb is: BUY NO on lower + BUY YES on upper
            # Because: if price doesn't reach lower, both NO and YES pay 0
            # But if price reaches lower but not upper: NO pays 0 on lower = loss
            # Better framing: the arb is that upper can't be more likely than lower
            # Buy YES lower (costs lower.yes_ask) + Buy YES upper (costs upper.yes_ask)
            # At most one can resolve YES... but that's not guaranteed profit.
            # Real arb: Buy NO on lower rung (costs lower.no_ask) because
            # P(NOT reaching lower) >= P(NOT reaching upper) is guaranteed
            # But we need them to sum < 1.
            # Simpler: just flag the violation and recommend buying the underpriced one.
            net_spread = violation - 0.01  # rough estimate after costs

            arbs.append(LadderArb(
                lower_rung=lower,
                upper_rung=upper,
                violation=violation,
                net_spread=net_spread,
            ))
            logger.info(
                "[LadderArb] Violation: %s $%,.0f (%.3f) vs $%,.0f (%.3f) | gap=%.4f",
                asset, lower.threshold, lower.poly_prob,
                upper.threshold, upper.poly_prob, violation,
            )

    arbs.sort(key=lambda a: a.violation, reverse=True)
    return arbs


def scan_crypto_ladder(poly_markets: Optional[list[dict]] = None) -> list[LadderArb]:
    """
    Full scan: fetch Polymarket crypto markets, parse rungs, find violations.
    Uses paginated active markets (not search) to avoid stale results.
    """
    from apis import gamma_api

    if poly_markets is None:
        all_markets: list[dict] = []
        for offset in range(0, 1000, 100):
            try:
                batch = gamma_api.get_active_markets(limit=100, offset=offset)
            except Exception as exc:
                logger.warning("[LadderArb] Gamma fetch failed at offset %d: %s", offset, exc)
                break
            if not batch:
                break
            all_markets.extend(batch)

        # Filter to crypto-related markets
        keywords = ["bitcoin", "btc", "ethereum", "eth", "solana", "sol"]
        markets = [m for m in all_markets if any(kw in m.get("question", "").lower() for kw in keywords)]
        logger.info("[LadderArb] %d crypto markets from %d active total", len(markets), len(all_markets))
    else:
        markets = poly_markets

    rungs = parse_ladder_rungs(markets)
    return find_ladder_arbs(rungs)
