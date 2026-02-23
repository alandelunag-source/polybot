"""Unit tests for BookCache and TokenRegistry."""
import time
import pytest
from apis.book_cache import BookCache, BookState
from apis.ws_client import TokenRegistry


# ---------------------------------------------------------------------------
# BookCache tests
# ---------------------------------------------------------------------------

class TestBookCache:
    def test_apply_snapshot_sets_best_ask(self):
        cache = BookCache()
        cache.apply_snapshot(
            "tok1",
            bids=[{"price": "0.44", "size": "100"}],
            asks=[{"price": "0.46", "size": "50"}, {"price": "0.48", "size": "30"}],
        )
        assert cache.best_ask("tok1") == pytest.approx(0.46)

    def test_apply_snapshot_sets_best_bid(self):
        cache = BookCache()
        cache.apply_snapshot(
            "tok1",
            bids=[{"price": "0.43", "size": "20"}, {"price": "0.44", "size": "100"}],
            asks=[{"price": "0.46", "size": "50"}],
        )
        assert cache.best_bid("tok1") == pytest.approx(0.44)

    def test_snapshot_filters_zero_size(self):
        cache = BookCache()
        cache.apply_snapshot(
            "tok1",
            bids=[{"price": "0.44", "size": "0"}],
            asks=[{"price": "0.46", "size": "50"}],
        )
        # Zero-size bid should be excluded
        assert cache.best_bid("tok1") is None
        assert cache.best_ask("tok1") == pytest.approx(0.46)

    def test_apply_delta_adds_new_level(self):
        cache = BookCache()
        cache.apply_snapshot("tok1", bids=[], asks=[{"price": "0.50", "size": "100"}])
        cache.apply_delta("tok1", [{"price": "0.48", "side": "SELL", "size": "75"}])
        assert cache.best_ask("tok1") == pytest.approx(0.48)

    def test_apply_delta_removes_level_on_zero_size(self):
        cache = BookCache()
        cache.apply_snapshot("tok1", bids=[], asks=[{"price": "0.46", "size": "50"}])
        # Remove the only ask level
        cache.apply_delta("tok1", [{"price": "0.46", "side": "SELL", "size": "0"}])
        assert cache.best_ask("tok1") is None

    def test_apply_delta_updates_existing_level(self):
        cache = BookCache()
        cache.apply_snapshot("tok1", bids=[], asks=[{"price": "0.46", "size": "50"}])
        cache.apply_delta("tok1", [{"price": "0.46", "side": "SELL", "size": "200"}])
        book = cache.get_book("tok1")
        assert book["asks"][0]["size"] == "200.0"

    def test_delta_before_snapshot_is_ignored(self):
        """Delta arriving before snapshot must not crash."""
        cache = BookCache()
        cache.apply_delta("new_token", [{"price": "0.50", "side": "SELL", "size": "10"}])
        assert cache.best_ask("new_token") is None

    def test_unknown_token_returns_none(self):
        cache = BookCache()
        assert cache.best_ask("nonexistent") is None
        assert cache.best_bid("nonexistent") is None
        assert cache.get_book("nonexistent") is None

    def test_age_seconds_after_snapshot(self):
        cache = BookCache()
        cache.apply_snapshot("tok1", bids=[], asks=[{"price": "0.50", "size": "1"}])
        age = cache.age_seconds("tok1")
        assert age is not None
        assert age < 1.0  # should be nearly instant

    def test_tracked_tokens_list(self):
        cache = BookCache()
        cache.apply_snapshot("tokA", bids=[], asks=[])
        cache.apply_snapshot("tokB", bids=[], asks=[])
        assert set(cache.tracked_tokens()) == {"tokA", "tokB"}
        assert len(cache) == 2

    def test_snapshot_replaces_previous_state(self):
        """Second snapshot should fully replace first, not merge."""
        cache = BookCache()
        cache.apply_snapshot("tok1", bids=[], asks=[{"price": "0.40", "size": "100"}])
        cache.apply_snapshot("tok1", bids=[], asks=[{"price": "0.60", "size": "50"}])
        assert cache.best_ask("tok1") == pytest.approx(0.60)


# ---------------------------------------------------------------------------
# TokenRegistry tests
# ---------------------------------------------------------------------------

def _make_market(mkt_id, yes_tid, no_tid, question="Will X happen?"):
    return {
        "id": mkt_id,
        "question": question,
        "tokens": [
            {"outcome": "Yes", "token_id": yes_tid},
            {"outcome": "No", "token_id": no_tid},
        ],
    }


class TestTokenRegistry:
    def test_builds_token_list(self):
        markets = [_make_market("m1", "yes1", "no1")]
        reg = TokenRegistry(markets)
        assert "yes1" in reg.all_token_ids
        assert "no1" in reg.all_token_ids

    def test_get_market_by_yes_token(self):
        markets = [_make_market("m1", "yes1", "no1")]
        reg = TokenRegistry(markets)
        mkt = reg.get_market("yes1")
        assert mkt is not None
        assert mkt["id"] == "m1"

    def test_get_market_by_no_token(self):
        markets = [_make_market("m1", "yes1", "no1")]
        reg = TokenRegistry(markets)
        mkt = reg.get_market("no1")
        assert mkt is not None
        assert mkt["id"] == "m1"

    def test_get_pair_yes_to_no(self):
        markets = [_make_market("m1", "yes1", "no1")]
        reg = TokenRegistry(markets)
        assert reg.get_pair("yes1") == "no1"

    def test_get_pair_no_to_yes(self):
        markets = [_make_market("m1", "yes1", "no1")]
        reg = TokenRegistry(markets)
        assert reg.get_pair("no1") == "yes1"

    def test_unknown_token_returns_none(self):
        reg = TokenRegistry([])
        assert reg.get_market("unknown") is None
        assert reg.get_pair("unknown") is None

    def test_skips_markets_missing_tokens(self):
        bad_market = {"id": "m_bad", "question": "Q?", "tokens": []}
        good_market = _make_market("m_good", "yes_g", "no_g")
        reg = TokenRegistry([bad_market, good_market])
        assert len(reg) == 1
        assert "yes_g" in reg.all_token_ids

    def test_multiple_markets(self):
        markets = [
            _make_market("m1", "yes1", "no1"),
            _make_market("m2", "yes2", "no2"),
        ]
        reg = TokenRegistry(markets)
        assert len(reg) == 2
        assert len(reg.all_token_ids) == 4
        assert reg.get_pair("yes2") == "no2"


# ---------------------------------------------------------------------------
# Integration: cache-backed arb check
# ---------------------------------------------------------------------------

class TestCacheBackedArb:
    def test_arb_detected_from_cache(self):
        from strategies.arbitrage import check_arb_from_cache

        cache = BookCache()
        cache.apply_snapshot("yes_tok", bids=[], asks=[{"price": "0.40", "size": "100"}])
        cache.apply_snapshot("no_tok", bids=[], asks=[{"price": "0.45", "size": "100"}])

        market = _make_market("m1", "yes_tok", "no_tok")
        opp = check_arb_from_cache(market, "yes_tok", "no_tok", cache)

        assert opp is not None
        assert opp.yes_ask == pytest.approx(0.40)
        assert opp.no_ask == pytest.approx(0.45)
        assert opp.net_spread > 0

    def test_no_arb_when_prices_at_parity(self):
        from strategies.arbitrage import check_arb_from_cache

        cache = BookCache()
        cache.apply_snapshot("yes_tok", bids=[], asks=[{"price": "0.51", "size": "100"}])
        cache.apply_snapshot("no_tok", bids=[], asks=[{"price": "0.51", "size": "100"}])

        market = _make_market("m1", "yes_tok", "no_tok")
        opp = check_arb_from_cache(market, "yes_tok", "no_tok", cache)
        assert opp is None

    def test_no_arb_when_book_empty(self):
        from strategies.arbitrage import check_arb_from_cache

        cache = BookCache()
        cache.apply_snapshot("yes_tok", bids=[], asks=[])
        # no_tok has no snapshot at all

        market = _make_market("m1", "yes_tok", "no_tok")
        opp = check_arb_from_cache(market, "yes_tok", "no_tok", cache)
        assert opp is None
