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

# Evaluation Parameters
TRADING_DAYS_PER_YEAR = 252