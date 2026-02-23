"""
Polymarket CLOB WebSocket client + token registry.

Protocol:
  - Connect to wss://clob.polymarket.com/ws/market
  - Send subscription: {"auth": {}, "type": "Market", "markets": [], "assets_ids": [...]}
  - Receive:
      {"event_type": "book",         "asset_id": "...", "bids": [...], "asks": [...]}
      {"event_type": "price_change", "asset_id": "...", "changes": [...]}
  - Messages may arrive as a JSON list or single dict
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Callable, Coroutine, Optional

import websockets

from apis import extract_token_ids
from apis.book_cache import BookCache

logger = logging.getLogger(__name__)

MARKET_WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
RECONNECT_DELAY_S = 2
SUBSCRIBE_BATCH = 500  # max tokens per subscription message


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# TokenRegistry
# ---------------------------------------------------------------------------

class TokenRegistry:
    """
    Builds lookup tables from a list of Gamma market dicts.

    Maps every YES/NO token_id to its parent market and a pair_id
    (the market's condition/id string), used by the WS callback to
    resolve which market was updated.
    """

    def __init__(self, markets: list[dict]) -> None:
        self._token_to_market: dict[str, dict] = {}
        self._token_to_pair: dict[str, str] = {}
        self.all_token_ids: list[str] = []
        self._market_count = 0

        for market in markets:
            yes_tid, no_tid = extract_token_ids(market)
            if not yes_tid or not no_tid:
                continue

            for tid, paired in ((yes_tid, no_tid), (no_tid, yes_tid)):
                self._token_to_market[tid] = market
                self._token_to_pair[tid]   = paired
                self.all_token_ids.append(tid)

            self._market_count += 1

    def __len__(self) -> int:
        return self._market_count

    def get_market(self, token_id: str) -> Optional[dict]:
        return self._token_to_market.get(token_id)

    def get_pair(self, token_id: str) -> Optional[str]:
        return self._token_to_pair.get(token_id)


# ---------------------------------------------------------------------------
# PolymarketWSClient
# ---------------------------------------------------------------------------

_Callback = Callable[[str, BookCache], Coroutine[Any, Any, None]]


class PolymarketWSClient:
    """
    Async WebSocket client for the Polymarket CLOB real-time feed.

    Subscribes to all token_ids in batches, applies snapshots and
    incremental deltas to BookCache, then fires registered callbacks.
    """

    def __init__(self, token_ids: list[str], cache: BookCache) -> None:
        self._token_ids = list(token_ids)
        self._cache     = cache
        self._callbacks: list[_Callback] = []
        self._stats = {
            "messages_received": 0,
            "snapshots":         0,
            "deltas":            0,
            "reconnects":        0,
        }

    @property
    def stats(self) -> dict:
        return dict(self._stats)

    def on_update(self, callback: _Callback) -> None:
        self._callbacks.append(callback)

    async def run(self) -> None:
        """Outer loop: reconnect indefinitely on any error."""
        first = True
        while True:
            if not first:
                self._stats["reconnects"] += 1
                logger.info("[WS] Reconnecting in %ds...", RECONNECT_DELAY_S)
                await asyncio.sleep(RECONNECT_DELAY_S)
            first = False
            try:
                await self._connect_and_listen()
            except Exception as exc:
                logger.error("[WS] Connection error: %s", exc)

    async def _connect_and_listen(self) -> None:
        async with websockets.connect(MARKET_WS_URL, max_size=4 * 1024 * 1024) as ws:
            logger.info("[WS] Connected. Subscribing %d tokens in batches of %d...",
                        len(self._token_ids), SUBSCRIBE_BATCH)
            for i in range(0, len(self._token_ids), SUBSCRIBE_BATCH):
                batch = self._token_ids[i : i + SUBSCRIBE_BATCH]
                await ws.send(json.dumps({
                    "assets_ids":             batch,
                    "type":                   "market",
                    "custom_feature_enabled": True,
                }))
            logger.info("[WS] Subscriptions sent. Listening...")

            async for raw in ws:
                self._stats["messages_received"] += 1
                await self._handle_raw(raw)

    async def _handle_raw(self, raw: str | bytes) -> None:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("[WS] Bad JSON: %s", raw[:120])
            return

        events = data if isinstance(data, list) else [data]
        for event in events:
            await self._process_event(event)

    async def _process_event(self, event: dict) -> None:
        event_type = event.get("event_type", "")
        token_id   = event.get("asset_id", "")
        if not token_id:
            return

        if event_type == "book":
            self._cache.apply_snapshot(
                token_id,
                bids=event.get("bids", []),
                asks=event.get("asks", []),
            )
            self._stats["snapshots"] += 1

        elif event_type == "price_change":
            self._cache.apply_delta(token_id, event.get("changes", []))
            self._stats["deltas"] += 1

        else:
            return  # heartbeat or unknown — ignore

        for cb in self._callbacks:
            await cb(token_id, self._cache)
