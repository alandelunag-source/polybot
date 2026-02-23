"""Unit tests for risk management and Kelly criterion."""
import pytest
from execution.risk import RiskManager
from strategies.value_betting import kelly_size, find_value_bets
from strategies.sports_divergence import DivergenceSignal


class TestRiskManager:
    def test_allows_order_within_limits(self):
        rm = RiskManager(max_position=100, max_total=500)
        allowed, reason = rm.check("token_A", 50)
        assert allowed is True
        assert reason == ""

    def test_blocks_order_exceeding_position_limit(self):
        rm = RiskManager(max_position=100, max_total=500)
        rm.record("token_A", 80)
        allowed, reason = rm.check("token_A", 30)
        assert allowed is False
        assert "Position limit" in reason

    def test_blocks_order_exceeding_total_exposure(self):
        rm = RiskManager(max_position=200, max_total=100)
        rm.record("token_A", 60)
        rm.record("token_B", 50)
        allowed, reason = rm.check("token_C", 20)
        assert allowed is False
        assert "Total exposure" in reason

    def test_records_and_tracks_exposure(self):
        rm = RiskManager(max_position=200, max_total=500)
        rm.record("token_A", 75)
        rm.record("token_B", 50)
        assert rm.total_exposure == pytest.approx(125)
        assert rm._positions["token_A"] == pytest.approx(75)

    def test_release_reduces_exposure(self):
        rm = RiskManager(max_position=200, max_total=500)
        rm.record("token_A", 100)
        rm.release("token_A", 40)
        assert rm._positions["token_A"] == pytest.approx(60)
        assert rm.total_exposure == pytest.approx(60)

    def test_release_does_not_go_negative(self):
        rm = RiskManager(max_position=200, max_total=500)
        rm.record("token_A", 30)
        rm.release("token_A", 100)  # More than recorded
        assert rm._positions["token_A"] == pytest.approx(0)

    def test_summary_returns_dict(self):
        rm = RiskManager(max_position=100, max_total=500)
        s = rm.summary()
        assert "total_exposure_usdc" in s
        assert "positions" in s


class TestKellySize:
    def test_positive_edge_returns_nonzero(self):
        size = kelly_size(true_prob=0.60, market_price=0.45, bankroll=1000, fraction=0.25)
        assert size > 0

    def test_no_edge_returns_zero(self):
        """Market price equals true prob → no edge → zero bet."""
        size = kelly_size(true_prob=0.50, market_price=0.50, bankroll=1000, fraction=0.25)
        assert size == pytest.approx(0, abs=1e-6)

    def test_negative_edge_returns_zero(self):
        """Market price > true prob → negative Kelly → zero bet."""
        size = kelly_size(true_prob=0.40, market_price=0.60, bankroll=1000, fraction=0.25)
        assert size == pytest.approx(0, abs=1e-6)

    def test_scales_with_bankroll(self):
        s1 = kelly_size(true_prob=0.60, market_price=0.45, bankroll=1000, fraction=0.25)
        s2 = kelly_size(true_prob=0.60, market_price=0.45, bankroll=2000, fraction=0.25)
        assert s2 == pytest.approx(s1 * 2, rel=1e-4)

    def test_scales_with_fraction(self):
        s_full = kelly_size(true_prob=0.60, market_price=0.45, bankroll=1000, fraction=1.0)
        s_half = kelly_size(true_prob=0.60, market_price=0.45, bankroll=1000, fraction=0.5)
        assert s_half == pytest.approx(s_full * 0.5, rel=1e-4)

    def test_invalid_price_returns_zero(self):
        assert kelly_size(true_prob=0.60, market_price=0, bankroll=1000) == 0
        assert kelly_size(true_prob=0.60, market_price=1.0, bankroll=1000) == 0


def _make_signal(poly_prob: float, book_prob: float, side: str = "YES") -> DivergenceSignal:
    return DivergenceSignal(
        market_id="mkt_001",
        question="Will Arsenal win?",
        team="Arsenal",
        poly_prob=poly_prob,
        book_prob=book_prob,
        delta=book_prob - poly_prob,
        side=side,
        sport="soccer_epl",
        bookmaker_count=5,
    )


class TestFindValueBets:
    def test_filters_signals_below_min_edge(self):
        """Signals with edge < min_edge are excluded."""
        signals = [_make_signal(poly_prob=0.48, book_prob=0.50)]  # 2% edge
        bets = find_value_bets(signals, bankroll=1000, min_edge=0.05)
        assert len(bets) == 0

    def test_includes_signals_above_min_edge(self):
        """Signals with edge >= min_edge are included."""
        signals = [_make_signal(poly_prob=0.45, book_prob=0.60)]  # 15% edge
        bets = find_value_bets(signals, bankroll=1000, min_edge=0.05)
        assert len(bets) == 1

    def test_sorted_by_edge_descending(self):
        """Bets are returned sorted by edge (highest first)."""
        signals = [
            _make_signal(0.45, 0.60),   # 15% edge
            _make_signal(0.40, 0.65),   # 25% edge
            _make_signal(0.48, 0.55),   # 7% edge
        ]
        bets = find_value_bets(signals, bankroll=1000, min_edge=0.05)
        edges = [b.edge for b in bets]
        assert edges == sorted(edges, reverse=True)

    def test_size_capped_at_max_position(self, monkeypatch):
        """Bet size never exceeds MAX_POSITION_USDC."""
        import config.settings as cfg
        monkeypatch.setattr(cfg, "MAX_POSITION_USDC", 50.0)

        signals = [_make_signal(poly_prob=0.30, book_prob=0.80)]  # huge edge
        bets = find_value_bets(signals, bankroll=100_000, min_edge=0.05)
        if bets:
            assert bets[0].capped_size_usdc <= 50.0

    def test_no_value_bets_for_no_edge_signals(self):
        """Signals with no positive Kelly edge return empty list."""
        signals = [_make_signal(poly_prob=0.70, book_prob=0.60)]  # poly overpriced
        bets = find_value_bets(signals, bankroll=1000, min_edge=0.05)
        assert len(bets) == 0
