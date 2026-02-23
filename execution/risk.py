"""
Risk management: enforces position and exposure limits before order placement.
"""
from __future__ import annotations

import logging
from collections import defaultdict

from config import settings

logger = logging.getLogger(__name__)


class RiskManager:
    """
    Tracks current exposure and enforces limits:
      - MAX_POSITION_USDC: max USDC exposure per individual market token
      - MAX_TOTAL_EXPOSURE_USDC: max total USDC across all open positions
    """

    def __init__(
        self,
        max_position: float = settings.MAX_POSITION_USDC,
        max_total: float = settings.MAX_TOTAL_EXPOSURE_USDC,
    ) -> None:
        self.max_position = max_position
        self.max_total = max_total
        self._positions: dict[str, float] = defaultdict(float)  # token_id -> usdc

    @property
    def total_exposure(self) -> float:
        return sum(self._positions.values())

    def check(self, token_id: str, usdc_amount: float) -> tuple[bool, str]:
        """
        Check whether placing a position of `usdc_amount` on `token_id` is allowed.
        Returns (allowed: bool, reason: str).
        """
        current = self._positions[token_id]
        if current + usdc_amount > self.max_position:
            return (
                False,
                f"Position limit: ${current + usdc_amount:.2f} > ${self.max_position:.2f} "
                f"for token {token_id[:12]}",
            )
        if self.total_exposure + usdc_amount > self.max_total:
            return (
                False,
                f"Total exposure limit: ${self.total_exposure + usdc_amount:.2f} "
                f"> ${self.max_total:.2f}",
            )
        return True, ""

    def record(self, token_id: str, usdc_amount: float) -> None:
        """Record that a position was opened."""
        self._positions[token_id] += usdc_amount
        logger.debug(
            "Position recorded: token=%s +$%.2f  total_exposure=$%.2f",
            token_id[:12],
            usdc_amount,
            self.total_exposure,
        )

    def release(self, token_id: str, usdc_amount: float) -> None:
        """Record that a position was closed or reduced."""
        self._positions[token_id] = max(0.0, self._positions[token_id] - usdc_amount)

    def summary(self) -> dict:
        return {
            "total_exposure_usdc": self.total_exposure,
            "max_total_usdc": self.max_total,
            "positions": dict(self._positions),
        }
