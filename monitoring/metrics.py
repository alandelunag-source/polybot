"""
In-memory metrics tracker for bot performance.
Tracks signals found, orders placed, fills, and estimated P&L.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


@dataclass
class ScanRun:
    timestamp: str
    strategy: str
    markets_scanned: int
    signals_found: int


@dataclass
class Metrics:
    scan_runs: list[ScanRun] = field(default_factory=list)
    orders_attempted: int = 0
    orders_placed: int = 0
    orders_dry_run: int = 0
    orders_failed: int = 0
    estimated_pnl_usdc: float = 0.0
    _signal_counts: dict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )

    def record_scan(
        self,
        strategy: str,
        markets_scanned: int,
        signals_found: int,
    ) -> None:
        run = ScanRun(
            timestamp=datetime.now(timezone.utc).isoformat(),
            strategy=strategy,
            markets_scanned=markets_scanned,
            signals_found=signals_found,
        )
        self.scan_runs.append(run)
        self._signal_counts[strategy] += signals_found
        logger.debug(
            "Scan recorded: strategy=%s markets=%d signals=%d",
            strategy,
            markets_scanned,
            signals_found,
        )

    def record_order(self, placed: bool, dry_run: bool = False) -> None:
        self.orders_attempted += 1
        if dry_run:
            self.orders_dry_run += 1
        elif placed:
            self.orders_placed += 1
        else:
            self.orders_failed += 1

    def record_pnl(self, usdc_delta: float) -> None:
        self.estimated_pnl_usdc += usdc_delta

    @property
    def fill_rate(self) -> float:
        total = self.orders_attempted
        return self.orders_placed / total if total > 0 else 0.0

    def summary(self) -> dict:
        return {
            "total_scans": len(self.scan_runs),
            "total_signals": dict(self._signal_counts),
            "orders_attempted": self.orders_attempted,
            "orders_placed": self.orders_placed,
            "orders_dry_run": self.orders_dry_run,
            "orders_failed": self.orders_failed,
            "fill_rate": self.fill_rate,
            "estimated_pnl_usdc": self.estimated_pnl_usdc,
        }

    def print_summary(self) -> None:
        s = self.summary()
        print("\n=== Polybot Metrics ===")
        for k, v in s.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
        print("======================\n")


# Global singleton
metrics = Metrics()
