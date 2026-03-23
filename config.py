# API Credentials
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY")
SECRET_KEY = os.getenv("SECRET_KEY")

if not API_KEY or not SECRET_KEY:
    raise ValueError("API_KEY and SECRET_KEY must be set in .env file.")

# Trading Parameters
SYMBOLS = ["AAPL", "MSFT", "NVDA", "SPY", "QQQ"]
LOOKBACK_DAYS = 1500

SHORT_EMA = 10
LONG_EMA = 20

TRANSACTION_COST = 0.001  # 0.1%

# Regime Detection — EMA Strategy
# These are intentionally strict. The EMA strategy benefits from only
# trading in clean, well-defined trends. Looser thresholds hurt performance.
EMA_SLOPE_THRESHOLD = 0.0007
EMA_R2_THRESHOLD = 0.6

# Regime Detection — ML Strategy
# ML uses a softer regime gate. The model already incorporates regime
# as a feature, so we don't need the regime filter to be as restrictive.
# This prevents the ML + Regime strategy from sitting in cash too often.
ML_SLOPE_THRESHOLD = 0.0003
ML_R2_THRESHOLD = 0.3

# ML Signal Threshold
# Minimum predicted probability to trigger a long position.
ML_PROB_THRESHOLD = 0.55  # Lowered from 0.6 — gives model more room to trade

# Evaluation Parameters
TRADING_DAYS_PER_YEAR = 252

# Mean Reversion Strategy
MR_WINDOW  = 20   # Bollinger Band rolling window
MR_NUM_STD = 2.0  # Number of standard deviations for band width

# Paper Trading
PAPER_TRADE_SIZE = 2000  # Fixed dollar allocation per position (USD)
