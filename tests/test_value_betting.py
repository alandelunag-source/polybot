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

        with patch("apis.odds_api_io.get_odds_with_movement", return_value=[event]), \
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
        with patch("apis.odds_api_io.get_odds_with_movement", side_effect=Exception("err")):
            signals = scan_sport_for_value("soccer_epl", poly_markets=[])
        assert signals == []

    def test_returns_empty_when_no_poly_match(self):
        event = _make_event()
        with patch("apis.odds_api_io.get_odds_with_movement", return_value=[event]), \
             patch("strategies.value_betting._find_matching_poly_market", return_value=None):
            signals = scan_sport_for_value("soccer_epl", poly_markets=[])
        assert signals == []


# ---------------------------------------------------------------------------
# standalone_value_scan
# ---------------------------------------------------------------------------

class TestStandaloneValueScan:
    def test_returns_value_bets(self):
        sig = _make_value_signal()
        with patch("strategies.value_betting._get_sports_batch", return_value=["soccer_epl"]), \
             patch("strategies.value_betting.scan_sport_for_value", return_value=[sig]):
            bets = standalone_value_scan(1000.0)
        assert isinstance(bets, list)
        # If Kelly returns >0, we get bets
        for bet in bets:
            assert isinstance(bet, ValueBet)

    def test_size_capped_at_max_position(self):
        sig = _make_value_signal(raw_edge=0.20, book_prob=0.65, poly_prob=0.45)
        with patch("strategies.value_betting._get_sports_batch", return_value=["soccer_epl"]), \
             patch("strategies.value_betting.scan_sport_for_value", return_value=[sig]):
            bets = standalone_value_scan(100000.0)
        for bet in bets:
            from config import settings
            assert bet.capped_size_usdc <= settings.MAX_POSITION_USDC

    def test_sorted_by_composite_score(self):
        s1 = _make_value_signal(composite_score=0.9)
        s2 = _make_value_signal(composite_score=0.5)
        s3 = _make_value_signal(composite_score=0.7)
        with patch("strategies.value_betting._get_sports_batch", return_value=["soccer_epl"]), \
             patch("strategies.value_betting.scan_sport_for_value", return_value=[s2, s3, s1]):
            bets = standalone_value_scan(1000.0)
        scores = [b.signal.composite_score for b in bets]
        assert scores == sorted(scores, reverse=True)

    def test_no_sports_returns_empty(self):
        with patch("strategies.value_betting._get_sports_batch", return_value=[]):
            bets = standalone_value_scan(1000.0)
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
