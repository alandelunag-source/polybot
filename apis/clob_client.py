"""
Thin wrapper around py-clob-client for Polymarket CLOB API.
Handles both authenticated (L2) and read-only (unauthenticated) usage.
"""
from __future__ import annotations

from typing import Any, Optional
from config import settings

try:
    from py_clob_client.client import ClobClient
    from py_clob_client.clob_types import (
        ApiCreds,
        OrderArgs,
        OrderType,
        Side,
        MarketOrderArgs,
    )
    _HAS_CLOB = True
except ImportError:
    _HAS_CLOB = False


def _require_clob() -> None:
    if not _HAS_CLOB:
        raise ImportError(
            "py-clob-client is not installed. Run: pip install py-clob-client"
        )


def get_client(authenticated: bool = True) -> Any:
    """Return a ClobClient instance. Pass authenticated=False for read-only."""
    _require_clob()
    if authenticated and settings.PRIVATE_KEY:
        creds = None
        if settings.API_KEY and settings.API_SECRET and settings.API_PASSPHRASE:
            creds = ApiCreds(
                api_key=settings.API_KEY,
                api_secret=settings.API_SECRET,
                api_passphrase=settings.API_PASSPHRASE,
            )
        client = ClobClient(
            host=settings.CLOB_ENDPOINT,
            chain_id=settings.CHAIN_ID,
            key=settings.PRIVATE_KEY,
            creds=creds,
        )
        return client
    # Unauthenticated read-only client
    return ClobClient(host=settings.CLOB_ENDPOINT, chain_id=settings.CHAIN_ID, key=None)


def get_order_book(token_id: str) -> dict:
    """Fetch the order book for a specific outcome token."""
    client = get_client(authenticated=False)
    return client.get_order_book(token_id)


def get_markets(next_cursor: str = "MA==", limit: int = 100) -> dict:
    """Page through CLOB markets. Returns raw response dict."""
    client = get_client(authenticated=False)
    return client.get_markets(next_cursor=next_cursor)


def get_market(condition_id: str) -> dict:
    """Fetch a single market by condition ID."""
    client = get_client(authenticated=False)
    return client.get_market(condition_id)


def derive_api_credentials() -> dict:
    """Derive L2 API credentials from private key (run once to populate .env)."""
    _require_clob()
    client = get_client(authenticated=False)
    # ClobClient must be instantiated with private key for key derivation
    authed = ClobClient(
        host=settings.CLOB_ENDPOINT,
        chain_id=settings.CHAIN_ID,
        key=settings.PRIVATE_KEY,
    )
    creds = authed.derive_api_key()
    return {
        "api_key": creds.api_key,
        "api_secret": creds.api_secret,
        "api_passphrase": creds.api_passphrase,
    }


def create_limit_order(
    token_id: str,
    side: str,  # "BUY" or "SELL"
    price: float,
    size: float,
) -> dict:
    """Place a limit order. Requires authenticated client."""
    if settings.DRY_RUN:
        return {
            "dry_run": True,
            "token_id": token_id,
            "side": side,
            "price": price,
            "size": size,
        }
    _require_clob()
    client = get_client(authenticated=True)
    clob_side = Side.BUY if side.upper() == "BUY" else Side.SELL
    order_args = OrderArgs(
        token_id=token_id,
        price=price,
        size=size,
        side=clob_side,
    )
    signed = client.create_order(order_args)
    return client.post_order(signed, OrderType.GTC)


def cancel_order(order_id: str) -> dict:
    """Cancel an open order by ID."""
    if settings.DRY_RUN:
        return {"dry_run": True, "cancelled": order_id}
    client = get_client(authenticated=True)
    return client.cancel(order_id)


def get_positions() -> list:
    """Return current open positions for the authenticated wallet."""
    client = get_client(authenticated=True)
    return client.get_positions()
