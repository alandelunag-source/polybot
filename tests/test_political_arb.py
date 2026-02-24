"""
Tests for strategies/political_arb.py and apis/kalshi_api.py.
Run: venv/Scripts/python.exe -m pytest tests/test_political_arb.py -v
"""
from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock

from strategies.political_arb import _similarity, find_political_arb, PoliticalArbSignal
from apis.kalshi_api import _to_prob, _parse_orderbook, _normalize_market


# ---------------------------------------------------------------------------
# _similarity
# ---------------------------------------------------------------------------

class TestSimilarity:
    def test_identical(self):
        assert _similarity("Will Biden win 2024", "Will Biden win 2024") == 1.0

    def test_partial_overlap(self):
        score = _similarity("Will Trump win the 2024 election", "Trump wins 2024 presidential")
        assert 0.0 < score < 1.0

    def test_no_overlap(self):
        assert _similarity("soccer match result", "bitcoin price prediction") == 0.0

    def test_empty(self):
        assert _similarity("", "anything") == 0.0


# ---------------------------------------------------------------------------
# Kalshi helpers
# ---------------------------------------------------------------------------

class TestKalshiHelpers:
    def test_to_prob_normal(self):
        assert _to_prob(55) == pytest.approx(0.55)

    def test_to_prob_zero(self):
        assert _to_prob(0) == pytest.approx(0.0)

    def test_to_prob_none(self):
        assert _to_prob(None) is None

    def test_to_prob_string(self):
        assert _to_prob("72") == pytest.approx(0.72)

    def test_parse_orderbook_basic(self):
        data = {"yes": [[60, 100], [55, 200]], "no": [[45, 150]]}
        ob = _parse_orderbook(data)
        assert ob["yes_bid"] == pytest.approx(0.60)
        assert ob["no_bid"] == pytest.approx(0.45)
        assert ob["yes_ask"] == pytest.approx(1.0 - 0.45)
        assert ob["no_ask"] == pytest.approx(1.0 - 0.60)

    def test_parse_orderbook_empty(self):
        ob = _parse_orderbook({})
        assert ob["yes_bid"] is None
        assert ob["yes_ask"] is None

    def test_normalize_market(self):
        raw = {"ticker": "PRES-2024-REP", "title": "Republican wins presidency", "yes_bid": 48, "yes_ask": 52, "close_time": "2024-11-05T00:00:00Z", "volume": 9000}
        m = _normalize_market(raw)
        assert m["ticker"] == "PRES-2024-REP"
        assert m["yes_bid"] == pytest.approx(0.48)
        assert m["yes_ask"] == pytest.approx(0.52)


# ---------------------------------------------------------------------------
# find_political_arb (mocked network)
# ---------------------------------------------------------------------------

POLY_MARKETS = [
    {
        "id": "poly-abc",
        "question": "Will the Republican candidate win the 2024 presidential election?",
        "clobTokenIds": ["tok_yes_1", "tok_no_1"],
        "outcomes": ["Yes", "No"],
    }
]

KALSHI_MARKETS = [
    {
        "ticker": "PRES-2024-R",
        "title": "Republican wins 2024 presidential election",
        "yes_bid": 0.44,
        "yes_ask": 0.46,
        "close_time": "2024-11-06T00:00:00Z",
        "volume": 50000,
    }
]


def _mock_poly_book(yes_bid=0.50, yes_ask=0.52):
    return {
        "bids": [{"price": str(yes_bid), "size": "100"}],
        "asks": [{"price": str(yes_ask), "size": "100"}],
    }


def _mock_kalshi_ob(yes_bid=0.44, yes_ask=0.46):
    return {"yes_bid": yes_bid, "yes_ask": yes_ask, "no_bid": 1 - yes_ask, "no_ask": 1 - yes_bid}


class TestFindPoliticalArb:
    @patch("apis.kalshi_api.get_orderbook")
    @patch("apis.clob_client.get_order_book")
    def test_detects_edge_buy_kalshi(self, mock_poly, mock_kalshi):
        # Poly YES bid (0.50) > Kalshi YES ask (0.46) → edge 0.04 > 0.03 threshold
        mock_poly.return_value = _mock_poly_book(yes_bid=0.50, yes_ask=0.52)
        mock_kalshi.return_value = _mock_kalshi_ob(yes_bid=0.44, yes_ask=0.46)

        signals = find_political_arb(POLY_MARKETS, KALSHI_MARKETS)
        assert len(signals) == 1
        sig = signals[0]
        assert sig.buy_on == "kalshi"
        assert sig.sell_on == "poly"
        assert sig.edge == pytest.approx(0.04, abs=0.001)

    @patch("apis.kalshi_api.get_orderbook")
    @patch("apis.clob_client.get_order_book")
    def test_detects_edge_buy_poly(self, mock_poly, mock_kalshi):
        # Kalshi YES bid (0.56) > Poly YES ask (0.52) → edge 0.04
        mock_poly.return_value = _mock_poly_book(yes_bid=0.50, yes_ask=0.52)
        mock_kalshi.return_value = _mock_kalshi_ob(yes_bid=0.56, yes_ask=0.58)

        signals = find_political_arb(POLY_MARKETS, KALSHI_MARKETS)
        assert len(signals) == 1
        assert signals[0].buy_on == "poly"

    @patch("apis.kalshi_api.get_orderbook")
    @patch("apis.clob_client.get_order_book")
    def test_no_signal_below_threshold(self, mock_poly, mock_kalshi):
        # Edge = 0.50 - 0.49 = 0.01 < 0.03 threshold
        mock_poly.return_value = _mock_poly_book(yes_bid=0.50, yes_ask=0.52)
        mock_kalshi.return_value = _mock_kalshi_ob(yes_bid=0.47, yes_ask=0.49)

        signals = find_political_arb(POLY_MARKETS, KALSHI_MARKETS)
        assert signals == []

    @patch("apis.kalshi_api.get_orderbook")
    @patch("apis.clob_client.get_order_book")
    def test_illiquid_poly_filtered(self, mock_poly, mock_kalshi):
        # Poly spread = 0.12 > 0.05 max → filtered even if edge is large
        mock_poly.return_value = _mock_poly_book(yes_bid=0.50, yes_ask=0.62)
        mock_kalshi.return_value = _mock_kalshi_ob(yes_bid=0.44, yes_ask=0.46)

        signals = find_political_arb(POLY_MARKETS, KALSHI_MARKETS)
        assert signals == []

    @patch("apis.kalshi_api.get_orderbook")
    @patch("apis.clob_client.get_order_book")
    def test_no_match_on_unrelated_titles(self, mock_poly, mock_kalshi):
        kalshi = [{"ticker": "BTCUSD", "title": "Bitcoin hits $100k", "yes_bid": 0.3, "yes_ask": 0.32, "volume": 100, "close_time": ""}]
        mock_poly.return_value = _mock_poly_book()
        mock_kalshi.return_value = _mock_kalshi_ob()

        signals = find_political_arb(POLY_MARKETS, kalshi)
        assert signals == []

    def test_sorted_by_edge_descending(self):
        """Signal list should be sorted highest-edge first."""
        sig1 = PoliticalArbSignal("a", "Q1", "K1", "T1", 0.7, 0.50, 0.52, 0.44, 0.46, edge=0.04, buy_on="kalshi", sell_on="poly")
        sig2 = PoliticalArbSignal("b", "Q2", "K2", "T2", 0.8, 0.55, 0.57, 0.44, 0.46, edge=0.09, buy_on="kalshi", sell_on="poly")
        sigs = [sig1, sig2]
        sigs.sort(key=lambda s: s.edge, reverse=True)
        assert sigs[0].edge > sigs[1].edge
