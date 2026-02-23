"""Unit tests for sports odds divergence scanner."""
import pytest
from unittest.mock import patch
from strategies.sports_divergence import find_divergences, DivergenceSignal, _similarity


MOCK_BOOK_EVENT = {
    "event_id": "evt_001",
    "sport_key": "soccer_epl",
    "commence_time": "2026-02-28T15:00:00Z",
    "home_team": "Arsenal",
    "away_team": "Chelsea",
    "home_prob": 0.60,
    "away_prob": 0.30,
    "draw_prob": 0.10,
    "bookmaker_count": 5,
}

MOCK_POLY_MARKET = {
    "id": "poly_mkt_001",
    "question": "Will Arsenal win vs Chelsea?",
    "tokens": [
        {"outcome": "Yes", "token_id": "yes_token_001"},
        {"outcome": "No", "token_id": "no_token_001"},
    ],
}


class TestFindDivergences:
    def test_detects_underpriced_yes(self):
        """Polymarket underprices YES → signal with side=YES."""
        # Books say Arsenal wins with 60% probability
        # Polymarket shows only 45% → 15% gap → signal

        with (
            patch("strategies.sports_divergence.odds_api.get_consensus_probs") as mock_odds,
            patch("strategies.sports_divergence.gamma_api.get_sports_markets") as mock_gamma,
            patch("strategies.sports_divergence.clob_client.get_order_book") as mock_book,
        ):
            mock_odds.return_value = [MOCK_BOOK_EVENT]
            mock_gamma.return_value = [MOCK_POLY_MARKET]
            mock_book.return_value = {
                "bids": [{"price": "0.44"}],
                "asks": [{"price": "0.46"}],
            }

            signals = find_divergences("soccer_epl", poly_markets=[MOCK_POLY_MARKET])

        assert len(signals) == 1
        sig = signals[0]
        assert isinstance(sig, DivergenceSignal)
        assert sig.side == "YES"
        assert sig.poly_prob == pytest.approx(0.45)
        assert sig.book_prob == pytest.approx(0.60)
        assert sig.delta == pytest.approx(0.15)

    def test_no_signal_when_within_threshold(self):
        """Small divergence below threshold → no signal."""
        event = {**MOCK_BOOK_EVENT, "home_prob": 0.47}  # only 2% gap from poly 0.45

        with (
            patch("strategies.sports_divergence.odds_api.get_consensus_probs") as mock_odds,
            patch("strategies.sports_divergence.gamma_api.get_sports_markets") as mock_gamma,
            patch("strategies.sports_divergence.clob_client.get_order_book") as mock_book,
        ):
            mock_odds.return_value = [event]
            mock_gamma.return_value = [MOCK_POLY_MARKET]
            mock_book.return_value = {
                "bids": [{"price": "0.44"}],
                "asks": [{"price": "0.46"}],
            }

            signals = find_divergences("soccer_epl", poly_markets=[MOCK_POLY_MARKET])

        assert len(signals) == 0

    def test_returns_empty_on_odds_api_failure(self):
        """Odds API failure → empty result, no exception."""
        with patch(
            "strategies.sports_divergence.odds_api.get_consensus_probs",
            side_effect=Exception("API down"),
        ):
            signals = find_divergences("soccer_epl")

        assert signals == []

    def test_returns_empty_when_no_poly_match(self):
        """No matching Polymarket market → empty result."""
        unrelated_market = {
            "id": "poly_mkt_002",
            "question": "Will it rain tomorrow?",
            "tokens": [
                {"outcome": "Yes", "token_id": "yes_999"},
                {"outcome": "No", "token_id": "no_999"},
            ],
        }

        with (
            patch("strategies.sports_divergence.odds_api.get_consensus_probs") as mock_odds,
            patch("strategies.sports_divergence.gamma_api.get_sports_markets") as mock_gamma,
        ):
            mock_odds.return_value = [MOCK_BOOK_EVENT]
            mock_gamma.return_value = [unrelated_market]

            signals = find_divergences("soccer_epl", poly_markets=[unrelated_market])

        assert len(signals) == 0


class TestSimilarity:
    def test_identical_strings(self):
        assert _similarity("Arsenal Chelsea", "Arsenal Chelsea") == pytest.approx(1.0)

    def test_case_insensitive(self):
        assert _similarity("arsenal chelsea", "ARSENAL CHELSEA") == pytest.approx(1.0)

    def test_partial_match(self):
        score = _similarity("Arsenal", "Arsenal Chelsea")
        assert 0.5 < score < 1.0

    def test_no_match(self):
        score = _similarity("Arsenal", "Lakers vs Celtics")
        assert score < 0.5
