from data.fetch_data import fetch_data
from strategy.ema_strategy import apply_ema_strategy, backtest_strategy
from backtesting.performance import evaluate_performance
from config import (
    API_KEY,
    SECRET_KEY,
    SYMBOLS,
    LOOKBACK_DAYS,
    SHORT_EMA,
    LONG_EMA,
    TRANSACTION_COST
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
    train_df = backtest_strategy(train_df, TRANSACTION_COST)

    test_df = apply_ema_strategy(test_df, SHORT_EMA, LONG_EMA)
    test_df = backtest_strategy(test_df, TRANSACTION_COST)
    
    print("=== TRAIN PERFORMANCE ===")
    train_perf = evaluate_performance(train_df)
    for k, v in train_perf.items():
        print(f"{k}: {v:.4f}")

    print("\n=== TEST PERFORMANCE ===")
    test_perf = evaluate_performance(test_df)
    for k, v in test_perf.items():
        print(f"{k}: {v:.4f}")

    print("\n--- TEST: TRENDING REGIME ONLY ---")
trending_test = test_df[test_df['regime'] == 1]

if len(trending_test) > 0:
    trending_perf = evaluate_performance(trending_test)
    for k, v in trending_perf.items():
        print(f"{k}: {v:.4f}")
else:
    print("No trending periods detected in test.")

print("Trending ratio:", (test_df['regime'] == 1).mean())
print("Trending count:", len(trending_test))