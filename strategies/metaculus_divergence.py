"""
Metaculus Superforecaster Divergence

Alpha thesis:
  Metaculus aggregates forecasts from thousands of trained superforecasters
  on political, economic, scientific, and tech events. Their community
  median is historically well-calibrated (Brier score ~0.16 vs ~0.20 for
  prediction markets on shared questions).

  When Metaculus and Polymarket disagree by >8% on the same question,
  we bet on the Metaculus side — the wisdom of superforecasters vs the
  wisdom of the crowd.

  Metaculus API is completely free, no auth required.
  Endpoint: https://www.metaculus.com/api2/questions/

Signal logic:
  1. Fetch open Metaculus questions with community predictions
  2. Fetch active Polymarket markets
  3. Fuzzy-match questions by title/keyword overlap
  4. Where |meta_prob - poly_prob| > MIN_DIVERGENCE: fire signal
  5. Kelly size based on edge and match confidence
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

import requests

logger = logging.getLogger(__name__)

METACULUS_API = "https://www.metaculus.com/api2/questions/"
MIN_DIVERGENCE = 0.08       # 8% minimum gap to signal
MIN_MATCH_SCORE = 0.35      # minimum keyword overlap to consider a match
MAX_QUESTIONS = 200         # questions to fetch per scan
METACULUS_TIMEOUT = 15

# NOTE: Metaculus API now requires authentication.
# Set METACULUS_TOKEN in your .env file (free account at metaculus.com).
# Without a token, the scan will return 0 signals gracefully.


@dataclass
class MetaculusSignal:
    """A divergence between Metaculus and Polymarket on the same question."""
    poly_market_id:  str
    poly_question:   str
    poly_prob:       float
    meta_question:   str
    meta_prob:       float
    meta_url:        str
    meta_resolve_by: str
    divergence:      float    # meta_prob - poly_prob (positive = meta more bullish)
    side:            str      # "YES" if we should buy YES, "NO" otherwise
    match_score:     float
    edge:            float    # abs(divergence)

    def __str__(self) -> str:
        direction = "higher" if self.divergence > 0 else "lower"
        return (
            f"[METACULUS] {self.poly_question[:60]}\n"
            f"  Poly={self.poly_prob:.3f}  Metaculus={self.meta_prob:.3f}  "
            f"Divergence={self.divergence:+.3f} (Meta is {direction})\n"
            f"  Side={self.side}  Edge={self.edge:.3f}  Match={self.match_score:.2f}\n"
            f"  Metaculus: {self.meta_url}"
        )


# ---------------------------------------------------------------------------
# Metaculus API
# ---------------------------------------------------------------------------

def _get_auth_headers() -> dict:
    """Return auth headers if METACULUS_TOKEN is set in environment."""
    import os
    token = os.environ.get("METACULUS_TOKEN", "")
    if token:
        return {"Authorization": f"Token {token}"}
    return {"User-Agent": "polybot/1.0"}


def _fetch_metaculus_questions(limit: int = MAX_QUESTIONS) -> list[dict]:
    """
    Fetch open binary questions with community predictions from Metaculus.
    Returns list of question dicts.
    Requires METACULUS_TOKEN env var (free account at metaculus.com).
    """
    import os
    if not os.environ.get("METACULUS_TOKEN"):
        logger.warning("[Metaculus] No METACULUS_TOKEN set — skipping fetch. "
                       "Get a free token at metaculus.com/accounts/profile/")
        return []

    params = {
        "format":                    "json",
        "limit":                     min(limit, 100),
        "status":                    "open",
        "has_community_prediction":  "true",
        "type":                      "forecast",  # binary questions only
        "order_by":                  "-activity",  # most active first
    }
    questions: list[dict] = []
    url = METACULUS_API

    while url and len(questions) < limit:
        try:
            resp = requests.get(url, params=params, headers=_get_auth_headers(), timeout=METACULUS_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.warning("[Metaculus] Fetch failed: %s", exc)
            break

        results = data.get("results", [])
        questions.extend(results)
        url = data.get("next")      # pagination
        params = {}                 # next URL already has params baked in

    logger.info("[Metaculus] Fetched %d open questions", len(questions))
    return questions


def _extract_meta_prob(question: dict) -> Optional[float]:
    """Extract median community probability from a Metaculus question dict."""
    cp = question.get("community_prediction") or {}

    # New API: community_prediction.full.q2 is the median
    full = cp.get("full") or {}
    q2 = full.get("q2")
    if q2 is not None:
        return float(q2)

    # Fallback: prediction field
    pred = question.get("prediction")
    if pred is not None:
        return float(pred)

    return None


# ---------------------------------------------------------------------------
# Fuzzy question matching
# ---------------------------------------------------------------------------

_STOP_WORDS = {
    "will", "the", "a", "an", "be", "is", "are", "was", "were", "in", "on",
    "at", "to", "of", "and", "or", "by", "for", "with", "as", "from", "that",
    "this", "it", "its", "win", "winning", "reach", "hit", "exceed", "above",
    "before", "during", "after", "end", "year", "month", "day", "before",
}


def _tokenize(text: str) -> set[str]:
    """Lowercase, remove punctuation, filter stop words."""
    tokens = re.sub(r"[^\w\s]", " ", text.lower()).split()
    return {t for t in tokens if t not in _STOP_WORDS and len(t) > 2}


def _match_score(poly_q: str, meta_q: str) -> float:
    """Jaccard similarity on meaningful keywords."""
    pt = _tokenize(poly_q)
    mt = _tokenize(meta_q)
    if not pt or not mt:
        return 0.0
    intersection = pt & mt
    union = pt | mt
    return len(intersection) / len(union)


def _find_best_poly_match(
    meta_q: str,
    poly_markets: list[dict],
    min_score: float = MIN_MATCH_SCORE,
) -> Optional[tuple[dict, float]]:
    """Find the best-matching Polymarket market for a Metaculus question."""
    best_market = None
    best_score = min_score

    for mkt in poly_markets:
        poly_q = mkt.get("question", "")
        score = _match_score(poly_q, meta_q)
        if score > best_score:
            best_score = score
            best_market = mkt

    return (best_market, best_score) if best_market else None


# ---------------------------------------------------------------------------
# Main scan
# ---------------------------------------------------------------------------

def scan_metaculus_divergence(
    poly_markets: Optional[list[dict]] = None,
    min_divergence: float = MIN_DIVERGENCE,
) -> list[MetaculusSignal]:
    """
    Fetch Metaculus questions and find divergences with Polymarket prices.
    """
    from apis import gamma_api, extract_token_ids

    # Fetch Polymarket markets if not provided
    if poly_markets is None:
        poly_markets = []
        for offset in range(0, 500, 100):
            try:
                batch = gamma_api.get_active_markets(limit=100, offset=offset)
            except Exception as exc:
                logger.warning("[Metaculus] Gamma fetch failed: %s", exc)
                break
            if not batch:
                break
            poly_markets.extend(batch)

    logger.info("[Metaculus] Polymarket: %d active markets", len(poly_markets))

    # Build a lookup: poly question → (market dict, mid price, yes_token_id)
    poly_lookup: list[tuple[dict, float, str]] = []
    for mkt in poly_markets:
        yes_tid, _ = extract_token_ids(mkt)
        if not yes_tid:
            continue
        try:
            bid = float(mkt.get("bestBid") or 0)
            ask = float(mkt.get("bestAsk") or 1)
        except (TypeError, ValueError):
            continue
        if ask <= bid or ask <= 0:
            continue
        mid = (bid + ask) / 2.0
        poly_lookup.append((mkt, mid, yes_tid))

    # Fetch Metaculus questions
    meta_questions = _fetch_metaculus_questions()

    signals: list[MetaculusSignal] = []

    for meta_q in meta_questions:
        meta_prob = _extract_meta_prob(meta_q)
        if meta_prob is None:
            continue
        # Only consider reasonably contested questions (5% - 95%)
        if not (0.05 <= meta_prob <= 0.95):
            continue

        meta_title  = meta_q.get("title", "")
        meta_id     = meta_q.get("id", "")
        meta_url    = f"https://www.metaculus.com/questions/{meta_id}/"
        meta_close  = meta_q.get("close_time", "")

        # Find best Polymarket match
        poly_markets_list = [m for m, _, _ in poly_lookup]
        result = _find_best_poly_match(meta_title, poly_markets_list)
        if not result:
            continue

        best_mkt, match_score = result

        # Get the mid price from our lookup
        poly_entry = next(
            ((m, mid, tid) for m, mid, tid in poly_lookup if m is best_mkt), None
        )
        if not poly_entry:
            continue
        _, poly_prob, yes_tid = poly_entry

        divergence = meta_prob - poly_prob
        edge = abs(divergence)
        if edge < min_divergence:
            continue

        side = "YES" if divergence > 0 else "NO"

        sig = MetaculusSignal(
            poly_market_id  = yes_tid,
            poly_question   = best_mkt.get("question", ""),
            poly_prob       = poly_prob,
            meta_question   = meta_title,
            meta_prob       = meta_prob,
            meta_url        = meta_url,
            meta_resolve_by = meta_close,
            divergence      = divergence,
            side            = side,
            match_score     = match_score,
            edge            = edge,
        )
        signals.append(sig)
        logger.info("[Metaculus] Signal: %s", sig)

    signals.sort(key=lambda s: s.edge * s.match_score, reverse=True)
    logger.info("[Metaculus] %d divergence signals found", len(signals))
    return signals


def to_value_bets(
    signals: list[MetaculusSignal],
    bankroll: float,
) -> list:
    """Convert MetaculusSignals to ValueBet objects for unified output."""
    from strategies.value_betting import ValueBet, kelly_size
    from config import settings

    bets = []
    for sig in signals:
        market_price = sig.poly_prob if sig.side == "YES" else (1.0 - sig.poly_prob)
        true_prob    = sig.meta_prob if sig.side == "YES" else (1.0 - sig.meta_prob)
        # Discount Kelly by match confidence (lower match = less conviction)
        adjusted_bankroll = bankroll * sig.match_score
        suggested = kelly_size(true_prob, market_price, adjusted_bankroll)
        if suggested <= 0:
            continue
        capped = min(suggested, settings.MAX_POSITION_USDC)

        # Wrap in a duck-typed signal compatible with ValueBet
        class _Sig:
            def __init__(self, s):
                self.market_id = s.poly_market_id
                self.question  = s.poly_question
                self.poly_prob = s.poly_prob
                self.side      = s.side

        bets.append(ValueBet(
            signal=_Sig(sig),
            edge=sig.edge,
            kelly_fraction=suggested / bankroll if bankroll > 0 else 0,
            suggested_size_usdc=suggested,
            capped_size_usdc=capped,
        ))

    return bets
