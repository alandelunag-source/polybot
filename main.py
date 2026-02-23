#!/usr/bin/env python3
"""
Polybot — Polymarket trading bot CLI.

Usage:
  python main.py scan --strategy arb
  python main.py scan --strategy divergence --sport soccer_epl
  python main.py scan --strategy value --sport soccer_epl --bankroll 1000
  python main.py run --dry-run
  python main.py run
  python main.py derive-keys
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time

from monitoring.logger import setup_logging
from monitoring.metrics import metrics
from config import settings

logger = logging.getLogger(__name__)


def cmd_scan(args: argparse.Namespace) -> None:
    strategy = args.strategy

    if strategy == "arb":
        from strategies.arbitrage import scan_markets
        from apis.gamma_api import iter_all_active_markets

        print("Fetching active markets...")
        markets = iter_all_active_markets()
        opps = scan_markets(markets)
        metrics.record_scan("arb", len(markets), len(opps))

        if not opps:
            print("No arbitrage opportunities found.")
        else:
            print(f"\nFound {len(opps)} arbitrage opportunity(ies):\n")
            for opp in opps:
                print(opp)
                print()

    elif strategy == "divergence":
        sport = args.sport or "soccer_epl"
        from strategies.sports_divergence import find_divergences

        print(f"Scanning sports divergence for: {sport}")
        signals = find_divergences(sport)
        metrics.record_scan("divergence", 0, len(signals))

        if not signals:
            print("No divergence signals found.")
        else:
            print(f"\nFound {len(signals)} divergence signal(s):\n")
            for sig in signals:
                print(sig)
                print()

    elif strategy == "value":
        sport = args.sport or "soccer_epl"
        bankroll = args.bankroll or 1000.0
        from strategies.sports_divergence import find_divergences
        from strategies.value_betting import find_value_bets

        print(f"Running value betting scan for: {sport}  (bankroll=${bankroll:.2f})")
        signals = find_divergences(sport)
        bets = find_value_bets(signals, bankroll=bankroll)
        metrics.record_scan("value", 0, len(bets))

        if not bets:
            print("No value bets found.")
        else:
            print(f"\nFound {len(bets)} value bet(s):\n")
            for bet in bets:
                print(bet)
                print()

    else:
        print(f"Unknown strategy: {strategy}")
        sys.exit(1)

    metrics.print_summary()


def cmd_run(args: argparse.Namespace) -> None:
    from strategies.arbitrage import scan_markets
    from strategies.sports_divergence import scan_all_sports
    from strategies.value_betting import find_value_bets
    from execution.order_manager import OrderManager
    from execution.risk import RiskManager
    from apis.gamma_api import iter_all_active_markets

    bankroll = args.bankroll or 1000.0
    dry = settings.DRY_RUN or args.dry_run

    if not dry and not settings.PRIVATE_KEY:
        print("ERROR: POLYMARKET_PRIVATE_KEY not set. Use --dry-run or set credentials.")
        sys.exit(1)

    print(f"Starting polybot loop (mode={'DRY RUN' if dry else 'LIVE'})...")
    print(f"  Scan interval: {settings.SCAN_INTERVAL_SECONDS}s")
    print(f"  Bankroll: ${bankroll:.2f}")
    print("Press Ctrl+C to stop.\n")

    risk = RiskManager()
    om = OrderManager(risk=risk)

    try:
        while True:
            # --- Arbitrage scan ---
            markets = iter_all_active_markets()
            arb_opps = scan_markets(markets)
            metrics.record_scan("arb", len(markets), len(arb_opps))
            for opp in arb_opps:
                print(opp)
                # Buy both sides
                size = min(settings.MAX_POSITION_USDC, bankroll * 0.01)
                r1 = om.place_limit_order(opp.yes_token_id, "BUY", opp.yes_ask, size)
                r2 = om.place_limit_order(opp.no_token_id, "BUY", opp.no_ask, size)
                metrics.record_order(r1.success, r1.dry_run)
                metrics.record_order(r2.success, r2.dry_run)
                print(r1, r2)

            # --- Sports divergence + value bets ---
            signals = scan_all_sports()
            bets = find_value_bets(signals, bankroll=bankroll)
            metrics.record_scan("value", 0, len(bets))
            for bet in bets:
                print(bet)
                token_id = bet.signal.market_id  # simplified; real use needs token_id
                r = om.place_limit_order(
                    token_id,
                    "BUY",
                    bet.signal.poly_prob if bet.signal.side == "YES" else (1 - bet.signal.poly_prob),
                    bet.capped_size_usdc,
                )
                metrics.record_order(r.success, r.dry_run)
                print(r)

            metrics.print_summary()
            print(f"Sleeping {settings.SCAN_INTERVAL_SECONDS}s...\n")
            time.sleep(settings.SCAN_INTERVAL_SECONDS)

    except KeyboardInterrupt:
        print("\nShutting down...")
        cancelled = om.cancel_all()
        print(f"Cancelled {cancelled} open orders.")
        metrics.print_summary()


async def _ws_arb_loop(args: argparse.Namespace) -> None:
    """
    Async core of the WebSocket-driven arbitrage loop.

    Steps:
      1. Fetch active markets from Gamma API (once, at startup)
      2. Build TokenRegistry — maps every token_id to its market and pair
      3. Subscribe to all token_ids via WebSocket
      4. On each price update, check arb from cache (zero REST calls)
      5. Execute qualifying opportunities immediately
    """
    from apis.gamma_api import iter_all_active_markets
    from apis.ws_client import PolymarketWSClient, TokenRegistry
    from apis.book_cache import BookCache
    from strategies.arbitrage import check_arb_from_cache
    from execution.order_manager import OrderManager
    from execution.risk import RiskManager

    bankroll = args.bankroll or 1000.0
    dry = settings.DRY_RUN or getattr(args, "dry_run", True)

    if not dry and not settings.PRIVATE_KEY:
        print("ERROR: POLYMARKET_PRIVATE_KEY not set. Use --dry-run or set credentials.")
        sys.exit(1)

    print(f"[WS] Fetching active markets from Gamma API...")
    markets = iter_all_active_markets()
    registry = TokenRegistry(markets)

    print(
        f"[WS] Loaded {len(registry)} markets  ({len(registry.all_token_ids)} tokens)"
    )
    print(f"[WS] Mode: {'DRY RUN' if dry else 'LIVE'}  Bankroll: ${bankroll:.2f}")
    print("[WS] Connecting to real-time feed... (Ctrl+C to stop)\n")

    cache = BookCache()
    risk = RiskManager()
    om = OrderManager(risk=risk)

    # Deduplicate opportunities within a short window to avoid double-firing
    # on rapid YES+NO back-to-back updates for the same market
    _recently_acted: dict[str, float] = {}  # market_id -> monotonic timestamp
    COOLDOWN_S = 10.0

    async def on_price_update(token_id: str, cache: BookCache) -> None:
        market = registry.get_market(token_id)
        if not market:
            return

        pair_id = registry.get_pair(token_id)
        if not pair_id:
            return

        # Determine which token is YES and which is NO
        tokens = market.get("tokens", [])
        yes_token = next((t for t in tokens if t.get("outcome", "").lower() == "yes"), None)
        no_token = next((t for t in tokens if t.get("outcome", "").lower() == "no"), None)
        if not yes_token or not no_token:
            return

        yes_tid = yes_token.get("token_id", "")
        no_tid = no_token.get("token_id", "")

        # Cooldown: skip if we acted on this market recently
        market_id = market.get("id", "")
        now = time.monotonic()
        if now - _recently_acted.get(market_id, 0) < COOLDOWN_S:
            return

        opp = check_arb_from_cache(market, yes_tid, no_tid, cache)
        if opp is None:
            return

        _recently_acted[market_id] = now
        logger.info("ARB SIGNAL: %s", opp)
        print(opp)

        size = min(settings.MAX_POSITION_USDC, bankroll * 0.01)

        # Place both legs concurrently
        r1, r2 = await asyncio.gather(
            asyncio.to_thread(om.place_limit_order, yes_tid, "BUY", opp.yes_ask, size),
            asyncio.to_thread(om.place_limit_order, no_tid, "BUY", opp.no_ask, size),
        )

        metrics.record_order(r1.success, r1.dry_run)
        metrics.record_order(r2.success, r2.dry_run)

        # If one leg failed, cancel the other
        if r1.success and not r2.success and r1.order_id:
            await asyncio.to_thread(om.cancel_order, r1.order_id)
            logger.warning("Leg 2 failed — cancelled leg 1 to avoid one-sided exposure")
        elif r2.success and not r1.success and r2.order_id:
            await asyncio.to_thread(om.cancel_order, r2.order_id)
            logger.warning("Leg 1 failed — cancelled leg 2 to avoid one-sided exposure")

        print(f"  Leg1: {r1}")
        print(f"  Leg2: {r2}\n")
        metrics.record_scan("arb_ws", 0, 1)

    client = PolymarketWSClient(registry.all_token_ids, cache)
    client.on_update(on_price_update)

    # Periodic stats printer (every 60s)
    async def print_stats_loop() -> None:
        while True:
            await asyncio.sleep(60)
            ws_stats = client.stats
            print(
                f"[WS STATS] msgs={ws_stats['messages_received']}  "
                f"snapshots={ws_stats['snapshots']}  "
                f"deltas={ws_stats['deltas']}  "
                f"reconnects={ws_stats['reconnects']}  "
                f"tokens_cached={len(cache)}"
            )
            metrics.print_summary()

    await asyncio.gather(
        client.run(),
        print_stats_loop(),
    )


def cmd_run_ws(args: argparse.Namespace) -> None:
    """Entry point for `python main.py run-ws`."""
    try:
        asyncio.run(_ws_arb_loop(args))
    except KeyboardInterrupt:
        print("\n[WS] Shutting down.")
        metrics.print_summary()


def cmd_derive_keys(_args: argparse.Namespace) -> None:
    if not settings.PRIVATE_KEY:
        print("ERROR: Set POLYMARKET_PRIVATE_KEY in .env first.")
        sys.exit(1)
    from apis.clob_client import derive_api_credentials
    creds = derive_api_credentials()
    print("Add these to your .env file:")
    print(f"  POLYMARKET_API_KEY={creds['api_key']}")
    print(f"  POLYMARKET_SECRET={creds['api_secret']}")
    print(f"  POLYMARKET_PASSPHRASE={creds['api_passphrase']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="polybot",
        description="Polymarket trading bot: arbitrage, sports divergence, value betting",
    )
    sub = parser.add_subparsers(dest="command")

    # scan command
    scan_p = sub.add_parser("scan", help="Run a one-shot strategy scan")
    scan_p.add_argument(
        "--strategy",
        choices=["arb", "divergence", "value"],
        default="arb",
        help="Strategy to run (default: arb)",
    )
    scan_p.add_argument("--sport", help="Sport key for divergence/value (e.g. soccer_epl)")
    scan_p.add_argument("--bankroll", type=float, help="Bankroll in USDC for value sizing")

    # run command
    run_p = sub.add_parser("run", help="Run the bot in a continuous loop")
    run_p.add_argument("--dry-run", action="store_true", help="Paper trading mode (no real orders)")
    run_p.add_argument("--bankroll", type=float, help="Bankroll in USDC")

    # run-ws command (WebSocket real-time arbitrage)
    ws_p = sub.add_parser("run-ws", help="Real-time arbitrage via WebSocket feed (recommended)")
    ws_p.add_argument("--dry-run", action="store_true", help="Paper trading mode (no real orders)")
    ws_p.add_argument("--bankroll", type=float, help="Bankroll in USDC")

    # derive-keys command
    sub.add_parser("derive-keys", help="Derive L2 API credentials from private key")

    # Logging flags (global)
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    parser.add_argument("--log-file", help="Write logs to file")

    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=level, log_file=getattr(args, "log_file", None))

    if args.command == "scan":
        cmd_scan(args)
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "run-ws":
        cmd_run_ws(args)
    elif args.command == "derive-keys":
        cmd_derive_keys(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
