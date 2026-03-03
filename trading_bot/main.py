from data.fetch_data import fetch_data
from strategy.ema_strategy import apply_ema_strategy, generate_positions
from backtesting.backtest import run_backtest
from backtesting.performance import evaluate_performance
from config import (
    API_KEY,
    SECRET_KEY,
    SYMBOLS,
    LOOKBACK_DAYS,
    SHORT_EMA,
    LONG_EMA,
    TRANSACTION_COST,
)

for symbol in SYMBOLS:
    print(f"\n==============================")
    print(f"SYMBOL: {symbol}")
    print(f"==============================")

    df = fetch_data(API_KEY, SECRET_KEY, symbol, LOOKBACK_DAYS)

    split_index = int(len(df) * 0.7)
    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()

    train_df = apply_ema_strategy(train_df, SHORT_EMA, LONG_EMA)
    train_positions = generate_positions(train_df)
    train_backtest = run_backtest(train_df, train_positions, TRANSACTION_COST)
    train_df["returns"] = train_backtest["market_returns"]
    train_df["strategy_returns"] = train_backtest["strategy_returns"]

    test_df = apply_ema_strategy(test_df, SHORT_EMA, LONG_EMA)
    test_positions = generate_positions(test_df)
    test_backtest = run_backtest(test_df, test_positions, TRANSACTION_COST)
    test_df["returns"] = test_backtest["market_returns"]
    test_df["strategy_returns"] = test_backtest["strategy_returns"]

    print("=== TRAIN PERFORMANCE ===")
    train_perf = evaluate_performance(train_df)
    for k, v in train_perf.items():
        print(f"{k}: {v:.4f}")

    print("\n=== TEST PERFORMANCE ===")
    test_perf = evaluate_performance(test_df)
    for k, v in test_perf.items():
        print(f"{k}: {v:.4f}")

    print("\n--- TEST: TRENDING REGIME ONLY ---")
    trending_test = test_df[test_df["regime"] == 1]

    if len(trending_test) > 0:
        trending_perf = evaluate_performance(trending_test)
        for k, v in trending_perf.items():
            print(f"{k}: {v:.4f}")
    else:
        print("No trending periods detected in test.")

    print("Trending ratio:", (test_df["regime"] == 1).mean())
    print("Trending count:", len(trending_test))
