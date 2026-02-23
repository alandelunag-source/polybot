import os
from dotenv import load_dotenv

load_dotenv()

# --- Polymarket credentials ---
PRIVATE_KEY: str = os.getenv("POLYMARKET_PRIVATE_KEY", "")
API_KEY: str = os.getenv("POLYMARKET_API_KEY", "")
API_SECRET: str = os.getenv("POLYMARKET_SECRET", "")
API_PASSPHRASE: str = os.getenv("POLYMARKET_PASSPHRASE", "")
CHAIN_ID: int = int(os.getenv("CHAIN_ID", "137"))

# --- Polymarket endpoints ---
CLOB_ENDPOINT = "https://clob.polymarket.com"
GAMMA_ENDPOINT = "https://gamma-api.polymarket.com"

# --- The Odds API ---
ODDS_API_KEY: str = os.getenv("ODDS_API_KEY", "")
ODDS_API_BASE = "https://api.the-odds-api.com/v4"

# --- Supported sports for divergence scanning ---
SUPPORTED_SPORTS = [
    "soccer_epl",
    "americanfootball_nfl",
    "basketball_nba",
    "baseball_mlb",
    "icehockey_nhl",
]

# --- Strategy thresholds ---
# Minimum spread (1 - YES_ask - NO_ask) to flag as arbitrage opportunity
ARB_MIN_SPREAD: float = float(os.getenv("ARB_MIN_SPREAD", "0.02"))

# Minimum probability divergence between Polymarket and bookmakers to flag
DIVERGENCE_THRESHOLD: float = float(os.getenv("DIVERGENCE_THRESHOLD", "0.05"))

# --- Risk limits ---
MAX_POSITION_USDC: float = float(os.getenv("MAX_POSITION_USDC", "100"))
MAX_TOTAL_EXPOSURE_USDC: float = float(os.getenv("MAX_TOTAL_EXPOSURE_USDC", "500"))
KELLY_FRACTION: float = float(os.getenv("KELLY_FRACTION", "0.25"))

# --- Bot mode ---
BOT_MODE: str = os.getenv("BOT_MODE", "dry_run")  # "dry_run" or "live"
DRY_RUN: bool = BOT_MODE != "live"

# --- Polling ---
SCAN_INTERVAL_SECONDS: int = 30
MAX_MARKETS_PER_SCAN: int = 200
