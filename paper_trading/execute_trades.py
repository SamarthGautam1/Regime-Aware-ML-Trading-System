import os
import math

from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus
from alpaca.common.exceptions import APIError

from utils.helper import setup_logger
from data.fetch_data import fetch_data

logger = setup_logger()
from strategy.ema_strategy import apply_ema_strategy
from features.feature_engineering import make_features, get_feature_columns
from models.save_load import load_model
from config import (
    SYMBOLS,
    LOOKBACK_DAYS,
    SHORT_EMA,
    LONG_EMA,
    ML_SLOPE_THRESHOLD,
    ML_R2_THRESHOLD,
    ML_PROB_THRESHOLD,
    PAPER_TRADE_SIZE,
)

load_dotenv()

PAPER_KEY    = os.getenv("ALPACA_PAPER_KEY")
PAPER_SECRET = os.getenv("ALPACA_PAPER_SECRET")

if not PAPER_KEY or not PAPER_SECRET:
    raise ValueError("ALPACA_PAPER_KEY and ALPACA_PAPER_SECRET must be set in .env")


# ---------------------------------------------------------------------------
# Signal generation
# ---------------------------------------------------------------------------

def get_signal(symbol: str) -> int:
    """
    Fetch data, build features, load saved model, return today's signal (0 or 1).
    Uses only the last row of the feature matrix — no future data.
    """
    data_key    = os.getenv("API_KEY")
    data_secret = os.getenv("SECRET_KEY")

    df = fetch_data(data_key, data_secret, symbol, LOOKBACK_DAYS)

    df = apply_ema_strategy(
        df,
        short_span=SHORT_EMA,
        long_span=LONG_EMA,
        slope_threshold=ML_SLOPE_THRESHOLD,
        r2_threshold=ML_R2_THRESHOLD,
    )

    df = make_features(df)

    model = load_model(symbol)

    feature_cols = get_feature_columns()
    X_latest = df[feature_cols].iloc[[-1]]  # single-row DataFrame, preserves column names

    pred_proba = model.predict_proba(X_latest)[0, 1]  # probability of up move
    regime     = int(df["regime"].iloc[-1])

    signal = 1 if (pred_proba > ML_PROB_THRESHOLD and regime == 1) else 0

    logger.info("[%s] pred_proba=%.4f  regime=%d  signal=%d", symbol, pred_proba, regime, signal)
    return signal


# ---------------------------------------------------------------------------
# Account helpers
# ---------------------------------------------------------------------------

def is_market_open(client: TradingClient) -> bool:
    """Return True only when the US equity market is currently open."""
    clock = client.get_clock()
    return clock.is_open


def get_portfolio_equity(client: TradingClient) -> float:
    """Return total portfolio equity from the live account."""
    account = client.get_account()
    return float(account.equity)


def get_current_qty(client: TradingClient, symbol: str) -> float:
    """Return held qty for symbol, or 0.0 if no position."""
    try:
        position = client.get_open_position(symbol)
        return float(position.qty)
    except APIError:
        return 0.0


def has_pending_order(client: TradingClient, symbol: str) -> bool:
    """
    Return True if there is already an open (pending/partially-filled) order
    for this symbol. Prevents submitting duplicate orders when the script
    runs more than once before the prior order fills.
    """
    open_orders = client.get_orders(
        filter=GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol])
    )
    return len(open_orders) > 0


# ---------------------------------------------------------------------------
# Order sizing
# ---------------------------------------------------------------------------

def get_latest_price(symbol: str) -> float:
    """Fetch the current ask price for a symbol via the market data API."""
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockLatestQuoteRequest

    data_key    = os.getenv("API_KEY")
    data_secret = os.getenv("SECRET_KEY")

    data_client = StockHistoricalDataClient(data_key, data_secret)
    request     = StockLatestQuoteRequest(symbol_or_symbols=symbol)
    quote       = data_client.get_stock_latest_quote(request)
    return float(quote[symbol].ask_price)


def compute_target_qty(price: float) -> int:
    """Floor PAPER_TRADE_SIZE / price to whole shares."""
    qty = math.floor(PAPER_TRADE_SIZE / price)
    return max(qty, 0)


# ---------------------------------------------------------------------------
# Per-symbol execution
# ---------------------------------------------------------------------------

def execute_symbol(client: TradingClient, symbol: str) -> None:
    logger.info("[%s] --- processing ---", symbol)

    # --- signal ---
    try:
        signal = get_signal(symbol)
    except FileNotFoundError:
        logger.warning("[%s] SKIP — no saved model found. Run main.py to train and save models first.", symbol)
        return
    except Exception as e:
        logger.error("[%s] ERROR generating signal — %s", symbol, e)
        return

    # --- current state ---
    current_qty  = get_current_qty(client, symbol)
    has_position = current_qty > 0

    # --- duplicate-order guard ---
    if has_pending_order(client, symbol):
        logger.info("[%s] SKIP — open order already exists, will not submit another", symbol)
        return

    # --- act ---
    if signal == 1 and not has_position:
        try:
            price      = get_latest_price(symbol)
            target_qty = compute_target_qty(price)
            if target_qty == 0:
                logger.warning("[%s] SKIP buy — target qty is 0 (PAPER_TRADE_SIZE=%s < price %.2f)", symbol, PAPER_TRADE_SIZE, price)
                return
            order = MarketOrderRequest(
                symbol=symbol,
                qty=target_qty,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
            )
            client.submit_order(order)
            logger.info("[%s] BUY %d shares @ ~%.2f  (allocation=$%s)", symbol, target_qty, price, PAPER_TRADE_SIZE)
        except Exception as e:
            logger.error("[%s] ERROR placing buy order — %s", symbol, e)

    elif signal == 0 and has_position:
        try:
            order = MarketOrderRequest(
                symbol=symbol,
                qty=current_qty,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
            )
            client.submit_order(order)
            logger.info("[%s] SELL %.4g shares (closing position)", symbol, current_qty)
        except Exception as e:
            logger.error("[%s] ERROR placing sell order — %s", symbol, e)

    else:
        state = "long" if has_position else "flat"
        logger.info("[%s] no action needed — signal=%d  current_state=%s", symbol, signal, state)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    logger.info("[SYSTEM] === paper trading run started ===")

    client = TradingClient(PAPER_KEY, PAPER_SECRET, paper=True)

    # Fix 2: abort immediately if market is closed — DAY orders would be rejected
    if not is_market_open(client):
        clock = client.get_clock()
        logger.info("[SYSTEM] market is closed. Next open: %s. Exiting.", clock.next_open)
        return

    try:
        equity = get_portfolio_equity(client)
        logger.info("[SYSTEM] equity=%.2f", equity)
    except Exception as e:
        logger.error("[SYSTEM] FATAL — could not fetch account info: %s", e)
        return

    for symbol in SYMBOLS:
        execute_symbol(client, symbol)

    logger.info("[SYSTEM] === run complete ===")


if __name__ == "__main__":
    main()
