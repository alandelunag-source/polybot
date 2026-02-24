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
SCAN_INTERVAL_SECONDS: int = int(os.getenv("SCAN_INTERVAL_SECONDS", "300"))
MAX_MARKETS_PER_SCAN: int = int(os.getenv("MAX_MARKETS_PER_SCAN", "200"))

# --- Odds-API.io (standalone value betting) ---
# Free tier: 2,400 req/day.
# Quota math: 2 leagues × (1 events + ~5 odds) × 288 cycles/day ≈ 1,728/day ✓
ODDS_API_IO_KEY: str = os.getenv("ODDS_API_IO_KEY", "")

# --- Standalone value betting thresholds ---
VALUE_MIN_EDGE: float = float(os.getenv("VALUE_MIN_EDGE", "0.04"))
VALUE_MIN_COMPOSITE_SCORE: float = float(os.getenv("VALUE_MIN_COMPOSITE_SCORE", "0.40"))
VALUE_SPORTS_PER_CYCLE: int = int(os.getenv("VALUE_SPORTS_PER_CYCLE", "1"))
VALUE_LINE_MOVE_THRESHOLD: float = float(os.getenv("VALUE_LINE_MOVE_THRESHOLD", "0.02"))
VALUE_WEIGHT_EDGE: float = float(os.getenv("VALUE_WEIGHT_EDGE", "0.50"))
VALUE_WEIGHT_CONSENSUS: float = float(os.getenv("VALUE_WEIGHT_CONSENSUS", "0.30"))
VALUE_WEIGHT_LINE: float = float(os.getenv("VALUE_WEIGHT_LINE", "0.20"))
