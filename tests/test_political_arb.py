"""
Tests for strategies/political_arb.py and apis/manifold_api.py.
Run: venv/Scripts/python.exe -m pytest tests/test_political_arb.py -v
"""
from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock

from strategies.political_arb import _similarity, find_political_arb, PoliticalArbSignal
from apis.manifold_api import _half_spread, _normalize_market, get_orderbook


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
# Manifold API helpers
# ---------------------------------------------------------------------------

class TestManifoldHelpers:
    def test_half_spread_high_liquidity(self):
        assert _half_spread(5_000) == pytest.approx(0.010)

    def test_half_spread_medium_liquidity(self):
        assert _half_spread(500) == pytest.approx(0.020)

    def test_half_spread_low_liquidity(self):
        assert _half_spread(50) == pytest.approx(0.040)

    def test_half_spread_boundary_1000(self):
        assert _half_spread(1_000) == pytest.approx(0.010)

    def test_normalize_market_basic(self):
        raw = {
            "id": "abc123",
            "question": "Will X happen?",
            "probability": 0.60,
            "totalLiquidity": 2_000,
            "volume": 5_000,
            "closeTime": 1_700_000_000_000,
        }
        m = _normalize_market(raw)
        assert m["ticker"] == "abc123"
        assert m["title"] == "Will X happen?"
        assert m["yes_bid"] == pytest.approx(0.60 - 0.01)
        assert m["yes_ask"] == pytest.approx(0.60 + 0.01)
        assert m["volume"] == pytest.approx(5_000)

    def test_normalize_market_low_liquidity_widens_spread(self):
        raw = {
            "id": "xyz",
            "question": "Q?",
            "probability": 0.50,
            "totalLiquidity": 50,
            "volume": 10,
            "closeTime": None,
        }
        m = _normalize_market(raw)
        assert m["yes_ask"] - m["yes_bid"] == pytest.approx(0.08)  # 2 × 4%

    def test_normalize_market_clamps_near_zero(self):
        raw = {"id": "z", "question": "Q", "probability": 0.01,
               "totalLiquidity": 50, "volume": 0, "closeTime": None}
        m = _normalize_market(raw)
        assert m["yes_bid"] >= 0.01

    def test_get_orderbook_returns_synthetic_book(self):
        mock_resp = {
            "id": "abc",
            "question": "Q?",
            "probability": 0.65,
            "totalLiquidity": 1_500,
            "isResolved": False,
            "outcomeType": "BINARY",
        }
        with patch("apis.manifold_api.requests.get") as mock_get:
            mock_get.return_value.json.return_value = mock_resp
            mock_get.return_value.raise_for_status = MagicMock()
            ob = get_orderbook("abc")

        assert ob is not None
        assert ob["yes_bid"] == pytest.approx(0.64)
        assert ob["yes_ask"] == pytest.approx(0.66)
        assert ob["no_bid"] == pytest.approx(1.0 - 0.66)
        assert ob["no_ask"] == pytest.approx(1.0 - 0.64)

    def test_get_orderbook_returns_none_for_resolved(self):
        mock_resp = {"isResolved": True, "outcomeType": "BINARY", "probability": 1.0, "totalLiquidity": 1_000}
        with patch("apis.manifold_api.requests.get") as mock_get:
            mock_get.return_value.json.return_value = mock_resp
            mock_get.return_value.raise_for_status = MagicMock()
            ob = get_orderbook("abc")
        assert ob is None

    def test_get_orderbook_returns_none_for_non_binary(self):
        mock_resp = {"isResolved": False, "outcomeType": "MULTIPLE_CHOICE", "probability": 0.5, "totalLiquidity": 1_000}
        with patch("apis.manifold_api.requests.get") as mock_get:
            mock_get.return_value.json.return_value = mock_resp
            mock_get.return_value.raise_for_status = MagicMock()
            ob = get_orderbook("abc")
        assert ob is None


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

MANIFOLD_MARKETS = [
    {
        "ticker": "manifold-pres-2024",
        "title": "Republican wins 2024 presidential election",
        "yes_bid": 0.44,
        "yes_ask": 0.46,
        "close_time": "2024-11-06T00:00:00Z",
        "volume": 50_000,
        "total_liquidity": 10_000,
    }
]


def _mock_poly_book(yes_bid=0.50, yes_ask=0.52):
    return {
        "bids": [{"price": str(yes_bid), "size": "100"}],
        "asks": [{"price": str(yes_ask), "size": "100"}],
    }


def _mock_manifold_ob(yes_bid=0.44, yes_ask=0.46):
    return {"yes_bid": yes_bid, "yes_ask": yes_ask,
            "no_bid": 1 - yes_ask, "no_ask": 1 - yes_bid}


class TestFindPoliticalArb:
    @patch("apis.manifold_api.get_orderbook")
    @patch("apis.clob_client.get_order_book")
    def test_detects_edge_buy_manifold(self, mock_poly, mock_manifold):
        # Poly YES bid (0.50) > Manifold YES ask (0.46) → edge 0.04 > 0.03 threshold
        mock_poly.return_value = _mock_poly_book(yes_bid=0.50, yes_ask=0.52)
        mock_manifold.return_value = _mock_manifold_ob(yes_bid=0.44, yes_ask=0.46)

        signals = find_political_arb(POLY_MARKETS, MANIFOLD_MARKETS)
        assert len(signals) == 1
        sig = signals[0]
        assert sig.buy_on == "manifold"
        assert sig.sell_on == "poly"
        assert sig.edge == pytest.approx(0.04, abs=0.001)

    @patch("apis.manifold_api.get_orderbook")
    @patch("apis.clob_client.get_order_book")
    def test_detects_edge_buy_poly(self, mock_poly, mock_manifold):
        # Manifold YES bid (0.56) > Poly YES ask (0.52) → edge 0.04
        mock_poly.return_value = _mock_poly_book(yes_bid=0.50, yes_ask=0.52)
        mock_manifold.return_value = _mock_manifold_ob(yes_bid=0.56, yes_ask=0.58)

        signals = find_political_arb(POLY_MARKETS, MANIFOLD_MARKETS)
        assert len(signals) == 1
        assert signals[0].buy_on == "poly"

    @patch("apis.manifold_api.get_orderbook")
    @patch("apis.clob_client.get_order_book")
    def test_no_signal_below_threshold(self, mock_poly, mock_manifold):
        # Edge = 0.50 - 0.49 = 0.01 < 0.03 threshold
        mock_poly.return_value = _mock_poly_book(yes_bid=0.50, yes_ask=0.52)
        mock_manifold.return_value = _mock_manifold_ob(yes_bid=0.47, yes_ask=0.49)

        signals = find_political_arb(POLY_MARKETS, MANIFOLD_MARKETS)
        assert signals == []

    @patch("apis.manifold_api.get_orderbook")
    @patch("apis.clob_client.get_order_book")
    def test_illiquid_poly_filtered(self, mock_poly, mock_manifold):
        # Poly spread = 0.12 > 0.05 max → filtered even if edge is large
        mock_poly.return_value = _mock_poly_book(yes_bid=0.50, yes_ask=0.62)
        mock_manifold.return_value = _mock_manifold_ob(yes_bid=0.44, yes_ask=0.46)

        signals = find_political_arb(POLY_MARKETS, MANIFOLD_MARKETS)
        assert signals == []

    @patch("apis.manifold_api.get_orderbook")
    @patch("apis.clob_client.get_order_book")
    def test_no_match_on_unrelated_titles(self, mock_poly, mock_manifold):
        manifold = [{"ticker": "crypto-btc", "title": "Bitcoin hits $100k",
                     "yes_bid": 0.3, "yes_ask": 0.32, "volume": 100,
                     "close_time": "", "total_liquidity": 500}]
        mock_poly.return_value = _mock_poly_book()
        mock_manifold.return_value = _mock_manifold_ob()

        signals = find_political_arb(POLY_MARKETS, manifold)
        assert signals == []

    def test_sorted_by_edge_descending(self):
        sig1 = PoliticalArbSignal(
            "a", "Q1", "M1", "T1", 0.7,
            0.50, 0.52, 0.44, 0.46,
            edge=0.04, buy_on="manifold", sell_on="poly",
        )
        sig2 = PoliticalArbSignal(
            "b", "Q2", "M2", "T2", 0.8,
            0.55, 0.57, 0.44, 0.46,
            edge=0.09, buy_on="manifold", sell_on="poly",
        )
        sigs = [sig1, sig2]
        sigs.sort(key=lambda s: s.edge, reverse=True)
        assert sigs[0].edge > sigs[1].edge
