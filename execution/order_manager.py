"""
Order placement and management layer.
All orders respect DRY_RUN mode — no real orders placed unless BOT_MODE=live.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from apis import clob_client
from execution.risk import RiskManager
from config import settings

logger = logging.getLogger(__name__)


@dataclass
class OrderResult:
    success: bool
    order_id: Optional[str] = None
    token_id: str = ""
    side: str = ""
    price: float = 0.0
    size: float = 0.0
    dry_run: bool = False
    error: Optional[str] = None

    def __str__(self) -> str:
        status = "DRY-RUN" if self.dry_run else ("OK" if self.success else "FAILED")
        return (
            f"[{status}] {self.side} {self.size:.2f} @ {self.price:.4f} "
            f"token={self.token_id[:12]}..."
            + (f" err={self.error}" if self.error else "")
        )


class OrderManager:
    def __init__(self, risk: Optional[RiskManager] = None) -> None:
        self.risk = risk or RiskManager()
        self._open_orders: dict[str, dict] = {}  # order_id -> order info

    def place_limit_order(
        self,
        token_id: str,
        side: str,
        price: float,
        size_usdc: float,
    ) -> OrderResult:
        """
        Place a limit order after risk checks.

        Args:
            token_id:   Outcome token ID (YES or NO)
            side:       'BUY' or 'SELL'
            price:      Limit price (0.0 – 1.0)
            size_usdc:  Dollar amount to spend (shares = size_usdc / price)
        """
        if price <= 0 or price >= 1:
            return OrderResult(
                success=False,
                token_id=token_id,
                side=side,
                price=price,
                size=size_usdc,
                error="Invalid price",
            )

        shares = size_usdc / price

        # Risk check
        allowed, reason = self.risk.check(token_id, size_usdc)
        if not allowed:
            logger.warning("Risk check blocked order: %s", reason)
            return OrderResult(
                success=False,
                token_id=token_id,
                side=side,
                price=price,
                size=shares,
                error=f"Risk limit: {reason}",
            )

        if settings.DRY_RUN:
            logger.info(
                "[DRY RUN] Would place %s %.4f shares @ %.4f (token=%s)",
                side, shares, price, token_id[:16],
            )
            self.risk.record(token_id, size_usdc)
            return OrderResult(
                success=True,
                token_id=token_id,
                side=side,
                price=price,
                size=shares,
                dry_run=True,
            )

        try:
            resp = clob_client.create_limit_order(token_id, side, price, shares)
            order_id = resp.get("orderID", "")
            self._open_orders[order_id] = {
                "token_id": token_id,
                "side": side,
                "price": price,
                "size": shares,
            }
            self.risk.record(token_id, size_usdc)
            logger.info("Order placed: id=%s %s", order_id, side)
            return OrderResult(
                success=True,
                order_id=order_id,
                token_id=token_id,
                side=side,
                price=price,
                size=shares,
            )
        except Exception as exc:
            logger.error("Order placement failed: %s", exc)
            return OrderResult(
                success=False,
                token_id=token_id,
                side=side,
                price=price,
                size=shares,
                error=str(exc),
            )

    def cancel_order(self, order_id: str) -> bool:
        if settings.DRY_RUN:
            logger.info("[DRY RUN] Would cancel order %s", order_id)
            self._open_orders.pop(order_id, None)
            return True
        try:
            clob_client.cancel_order(order_id)
            self._open_orders.pop(order_id, None)
            return True
        except Exception as exc:
            logger.error("Cancel failed for %s: %s", order_id, exc)
            return False

    def cancel_all(self) -> int:
        """Cancel all tracked open orders. Returns count cancelled."""
        cancelled = 0
        for oid in list(self._open_orders.keys()):
            if self.cancel_order(oid):
                cancelled += 1
        return cancelled

    @property
    def open_orders(self) -> dict:
        return dict(self._open_orders)
