"""Unit tests for arbitrage scanner."""
import pytest
from unittest.mock import patch
from strategies.arbitrage import scan_market, ArbitrageOpportunity, POLYMARKET_FEE


def _make_market(yes_token_id="yes123", no_token_id="no456"):
    return {
        "id": "mkt_001",
        "question": "Will team A win?",
        "tokens": [
            {"outcome": "Yes", "token_id": yes_token_id},
            {"outcome": "No", "token_id": no_token_id},
        ],
    }


def _make_book(ask_price: float) -> dict:
    return {"asks": [{"price": str(ask_price), "size": "100"}], "bids": []}


class TestScanMarket:
    def test_detects_arbitrage_opportunity(self):
        """YES ask + NO ask < 1.0 minus fees → opportunity flagged."""
        market = _make_market()
        yes_book = _make_book(0.40)
        no_book = _make_book(0.45)

        with patch("strategies.arbitrage.clob_client.get_order_book") as mock_book:
            mock_book.side_effect = [yes_book, no_book]
            result = scan_market(market)

        assert result is not None
        assert isinstance(result, ArbitrageOpportunity)
        assert result.yes_ask == pytest.approx(0.40)
        assert result.no_ask == pytest.approx(0.45)
        assert result.raw_spread == pytest.approx(0.15)
        assert result.net_spread > 0
        assert result.expected_profit_pct > 0

    def test_no_opportunity_when_sum_near_one(self):
        """YES + NO ≈ 1.00 → no opportunity after fees."""
        market = _make_market()
        yes_book = _make_book(0.51)
        no_book = _make_book(0.51)

        with patch("strategies.arbitrage.clob_client.get_order_book") as mock_book:
            mock_book.side_effect = [yes_book, no_book]
            result = scan_market(market)

        assert result is None

    def test_no_opportunity_when_sum_over_one(self):
        """YES + NO > 1.00 → no opportunity."""
        market = _make_market()
        yes_book = _make_book(0.55)
        no_book = _make_book(0.55)

        with patch("strategies.arbitrage.clob_client.get_order_book") as mock_book:
            mock_book.side_effect = [yes_book, no_book]
            result = scan_market(market)

        assert result is None

    def test_returns_none_with_empty_books(self):
        """Empty order books → no opportunity."""
        market = _make_market()
        empty_book = {"asks": [], "bids": []}

        with patch("strategies.arbitrage.clob_client.get_order_book") as mock_book:
            mock_book.return_value = empty_book
            result = scan_market(market)

        assert result is None

    def test_returns_none_for_market_without_tokens(self):
        """Market with no token data → returns None gracefully."""
        market = {"id": "mkt_002", "question": "Q?", "tokens": []}
        result = scan_market(market)
        assert result is None

    def test_returns_none_when_api_raises(self):
        """API failure → returns None without raising."""
        market = _make_market()

        with patch(
            "strategies.arbitrage.clob_client.get_order_book",
            side_effect=Exception("network error"),
        ):
            result = scan_market(market)

        assert result is None

    def test_net_spread_accounts_for_fees(self):
        """Net spread = raw spread minus fee cost."""
        market = _make_market()
        yes_ask, no_ask = 0.40, 0.45

        with patch("strategies.arbitrage.clob_client.get_order_book") as mock_book:
            mock_book.side_effect = [_make_book(yes_ask), _make_book(no_ask)]
            result = scan_market(market)

        assert result is not None
        expected_raw = 1.0 - yes_ask - no_ask
        expected_fee = POLYMARKET_FEE * (yes_ask + no_ask)
        assert result.raw_spread == pytest.approx(expected_raw)
        assert result.net_spread == pytest.approx(expected_raw - expected_fee)
