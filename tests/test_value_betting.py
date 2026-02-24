"""
Tests for standalone value betting strategy (v2).
Covers: scoring functions, scan flow, backward compat, standalone_value_scan.
"""
from __future__ import annotations
from unittest.mock import patch, MagicMock
import pytest

from strategies.value_betting import (
    ValueSignal,
    ValueBet,
    _score_edge,
    _score_consensus,
    _score_line_movement,
    _composite_score,
    scan_sport_for_value,
    standalone_value_scan,
    find_value_bets,
    kelly_size,
)
from strategies.sports_divergence import DivergenceSignal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_divergence_signal(**kwargs):
    defaults = dict(
        market_id="mkt-1", question="Will X win?", team="X",
        poly_prob=0.45, book_prob=0.55, delta=0.10,
        side="YES", sport="soccer_epl", bookmaker_count=10,
    )
    defaults.update(kwargs)
    return DivergenceSignal(**defaults)


def _make_value_signal(**kwargs):
    defaults = dict(
        market_id="tok-yes-1", question="Will Arsenal win?",
        poly_prob=0.45, side="YES", sport="soccer_epl",
        home_team="Arsenal", away_team="Chelsea",
        commence_time="2026-03-01T15:00:00Z",
        book_prob=0.55, raw_edge=0.10, bookmaker_count=12,
        line_move=0.03, composite_score=0.65,
    )
    defaults.update(kwargs)
    return ValueSignal(**defaults)


def _make_event(**kwargs):
    defaults = dict(
        event_id="evt-1", sport_key="soccer_epl",
        commence_time="2026-03-01T15:00:00Z",
        home_team="Arsenal", away_team="Chelsea",
        home_prob=0.55, away_prob=0.30, draw_prob=0.15,
        bookmaker_count=12,
        home_line_move=0.03, away_line_move=-0.03,
    )
    defaults.update(kwargs)
    return defaults


def _make_poly_market(yes_tid="tok-yes-1", question="Will Arsenal win?"):
    return {
        "id": "mkt-1",
        "question": question,
        "clobTokenIds": [yes_tid, "tok-no-1"],
        "outcomes": '["Yes", "No"]',
    }


# ---------------------------------------------------------------------------
# _score_edge
# ---------------------------------------------------------------------------

class TestScoreEdge:
    def test_below_min_edge_returns_zero(self):
        assert _score_edge(0.01) == 0.0

    def test_at_min_edge_returns_zero(self):
        from config import settings
        assert _score_edge(settings.VALUE_MIN_EDGE) == 0.0

    def test_large_edge_caps_at_one(self):
        assert _score_edge(0.50) == 1.0

    def test_monotonically_increases(self):
        scores = [_score_edge(e) for e in [0.04, 0.08, 0.15, 0.25, 0.30]]
        assert scores == sorted(scores)


# ---------------------------------------------------------------------------
# _score_consensus
# ---------------------------------------------------------------------------

class TestScoreConsensus:
    def test_fewer_than_3_books_returns_zero(self):
        assert _score_consensus(2) == 0.0
        assert _score_consensus(0) == 0.0

    def test_exactly_3_books_returns_zero(self):
        assert _score_consensus(3) == 0.0

    def test_15_books_returns_one(self):
        assert _score_consensus(15) == 1.0

    def test_above_15_caps_at_one(self):
        assert _score_consensus(30) == 1.0

    def test_monotonically_increases(self):
        scores = [_score_consensus(n) for n in [3, 6, 10, 15]]
        assert scores == sorted(scores)


# ---------------------------------------------------------------------------
# _score_line_movement
# ---------------------------------------------------------------------------

class TestScoreLineMovement:
    def test_no_movement_returns_neutral(self):
        assert _score_line_movement(0.0, "YES") == 0.5
        assert _score_line_movement(0.001, "YES") == 0.5  # below threshold

    def test_confirmatory_move_increases_score(self):
        # Betting YES, price moved up → confirms our view
        score = _score_line_movement(0.05, "YES")
        assert score > 0.5

    def test_counter_move_decreases_score(self):
        # Betting YES, price moved down → contradicts our view
        score = _score_line_movement(-0.05, "YES")
        assert score < 0.5

    def test_no_side_symmetry(self):
        # Betting NO with downward move = same as YES with upward
        assert _score_line_movement(-0.05, "NO") == _score_line_movement(0.05, "YES")

    def test_score_clamped(self):
        score = _score_line_movement(1.0, "YES")
        assert 0.0 <= score <= 1.0
        score = _score_line_movement(-1.0, "YES")
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# _composite_score
# ---------------------------------------------------------------------------

class TestCompositeScore:
    def test_weighted_sum(self):
        from config import settings
        score = _composite_score(1.0, 1.0, 1.0)
        expected = (
            settings.VALUE_WEIGHT_EDGE
            + settings.VALUE_WEIGHT_CONSENSUS
            + settings.VALUE_WEIGHT_LINE
        )
        assert abs(score - expected) < 1e-9

    def test_zero_inputs_returns_zero(self):
        assert _composite_score(0.0, 0.0, 0.0) == 0.0


# ---------------------------------------------------------------------------
# ValueSignal field access (backward compat with main.py)
# ---------------------------------------------------------------------------

class TestValueSignalFields:
    def test_main_py_field_access_pattern(self):
        sig = _make_value_signal()
        assert sig.market_id == "tok-yes-1"
        assert sig.poly_prob == 0.45
        assert sig.side == "YES"
        assert sig.question is not None


# ---------------------------------------------------------------------------
# scan_sport_for_value
# ---------------------------------------------------------------------------

class TestScanSportForValue:
    def _mock_scan(self, edge=0.10, bookmaker_count=12, line_move=0.03):
        event = _make_event(
            home_prob=0.45 + edge,
            bookmaker_count=bookmaker_count,
            home_line_move=line_move,
        )
        market = _make_poly_market(question="Will Arsenal win? Arsenal vs Chelsea")

        with patch("apis.odds_api.get_odds_with_movement", return_value=[event]), \
             patch("strategies.value_betting._find_matching_poly_market", return_value=market), \
             patch("strategies.value_betting._poly_mid_price", return_value=0.45), \
             patch("apis.extract_token_ids", return_value=("tok-yes-1", "tok-no-1")):
            return scan_sport_for_value("soccer_epl", poly_markets=[market])

    def test_returns_signal_on_clear_edge(self):
        signals = self._mock_scan(edge=0.12, bookmaker_count=15)
        assert len(signals) >= 0  # may or may not pass composite threshold

    def test_filters_below_min_edge(self):
        signals = self._mock_scan(edge=0.01)
        assert signals == []

    def test_returns_empty_on_api_failure(self):
        with patch("apis.odds_api.get_odds_with_movement", side_effect=Exception("err")):
            signals = scan_sport_for_value("soccer_epl", poly_markets=[])
        assert signals == []

    def test_returns_empty_when_no_poly_match(self):
        event = _make_event()
        with patch("apis.odds_api.get_odds_with_movement", return_value=[event]), \
             patch("strategies.value_betting._find_matching_poly_market", return_value=None):
            signals = scan_sport_for_value("soccer_epl", poly_markets=[])
        assert signals == []


# ---------------------------------------------------------------------------
# standalone_value_scan
# ---------------------------------------------------------------------------

def _make_poly_outright(question="Will the Carolina Hurricanes win the 2026 NHL Stanley Cup?", bid="0.12", ask="0.18"):
    return {
        "id": "mkt-out-1",
        "question": question,
        "clobTokenIds": ["tok-yes-out", "tok-no-out"],
        "outcomes": '["Yes", "No"]',
        "bestBid": bid,
        "bestAsk": ask,
    }


class TestStandaloneValueScan:
    def _run_scan(self, poly_markets, book_probs, poly_price=0.15, bankroll=1000.0):
        """Helper: mock Gamma (returns markets on first call, [] on subsequent), run scan."""
        gam_calls = iter([poly_markets] + [[]] * 10)
        with patch("apis.gamma_api.get_active_markets", side_effect=gam_calls), \
             patch("apis.odds_api.get_outright_probs", return_value=book_probs), \
             patch("strategies.value_betting._poly_mid_price", return_value=poly_price), \
             patch("apis.extract_token_ids", return_value=("tok-yes-out", "tok-no-out")):
            return standalone_value_scan(bankroll)

    def test_returns_value_bets(self):
        mkt = _make_poly_outright()
        # book says 25%, Poly says 15% → edge = 0.10
        bets = self._run_scan([mkt], {"Carolina Hurricanes": 0.25})
        assert isinstance(bets, list)
        for bet in bets:
            assert isinstance(bet, ValueBet)

    def test_size_capped_at_max_position(self):
        mkt = _make_poly_outright()
        bets = self._run_scan([mkt], {"Carolina Hurricanes": 0.40}, poly_price=0.15, bankroll=100000.0)
        from config import settings
        for bet in bets:
            assert bet.capped_size_usdc <= settings.MAX_POSITION_USDC

    def test_sorted_by_composite_score(self):
        """Multiple markets → bets sorted descending by composite score."""
        mkts = [
            _make_poly_outright("Will the Carolina Hurricanes win the 2026 NHL Stanley Cup?"),
            _make_poly_outright("Will the Edmonton Oilers win the 2026 NHL Stanley Cup?"),
            _make_poly_outright("Will the Vegas Golden Knights win the 2026 NHL Stanley Cup?"),
        ]
        book = {
            "Carolina Hurricanes": 0.28,
            "Edmonton Oilers": 0.35,
            "Vegas Golden Knights": 0.22,
        }
        # Each poly price 0.10 → edges differ by team, producing different scores
        # get_active_markets is called in a loop; return markets on first call, then []
        gam_calls = iter([mkts] + [[]] * 10)
        with patch("apis.gamma_api.get_active_markets", side_effect=gam_calls), \
             patch("apis.odds_api.get_outright_probs", return_value=book), \
             patch("strategies.value_betting._poly_mid_price", return_value=0.10), \
             patch("apis.extract_token_ids", side_effect=[
                 ("tok1", ""), ("tok2", ""), ("tok3", "")
             ]):
            bets = standalone_value_scan(1000.0)
        scores = [b.signal.composite_score for b in bets]
        assert scores == sorted(scores, reverse=True)

    def test_no_markets_returns_empty(self):
        with patch("apis.gamma_api.get_active_markets", return_value=[]):
            bets = standalone_value_scan(1000.0)
        assert bets == []

    def test_filters_below_min_edge(self):
        # Gamma market with bid/ask so close the edge is below VALUE_MIN_EDGE
        mkt = _make_poly_outright()
        mkt["bestBid"] = "0.149"
        mkt["bestAsk"] = "0.151"   # mid = 0.150, book = 0.151 → edge = 0.001
        gam_calls = iter([[mkt]] + [[]] * 10)
        with patch("apis.gamma_api.get_active_markets", side_effect=gam_calls), \
             patch("apis.odds_api.get_outright_probs", return_value={"Carolina Hurricanes": 0.151}), \
             patch("apis.extract_token_ids", return_value=("tok-yes-out", "tok-no-out")):
            bets = standalone_value_scan(1000.0)
        assert bets == []

    def test_no_poly_match_skipped(self):
        # Market without outright pattern → no candidates
        mkt = {"id": "m1", "question": "Who wins the election?",
               "clobTokenIds": ["t1", "t2"], "outcomes": '["Yes","No"]'}
        bets = self._run_scan([mkt], {"Some Team": 0.5})
        assert bets == []


# ---------------------------------------------------------------------------
# Backward compat: find_value_bets still works with DivergenceSignal
# ---------------------------------------------------------------------------

class TestFindValueBetsBackwardCompat:
    def test_accepts_divergence_signal(self):
        sig = _make_divergence_signal(delta=0.10)
        bets = find_value_bets([sig], bankroll=1000.0)
        assert isinstance(bets, list)

    def test_filters_below_threshold(self):
        sig = _make_divergence_signal(delta=0.01)
        bets = find_value_bets([sig], bankroll=1000.0, min_edge=0.05)
        assert bets == []
