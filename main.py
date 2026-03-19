from data.fetch_data import fetch_data
from strategy.ema_strategy import apply_ema_strategy, generate_positions
from features.feature_engineering import make_features, add_target, get_feature_columns
from models.train_model import train_model
from models.predict import predict_proba
from strategy.ml_strategy import generate_ml_positions
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
    EMA_SLOPE_THRESHOLD,
    EMA_R2_THRESHOLD,
    ML_SLOPE_THRESHOLD,
    ML_R2_THRESHOLD,
    ML_PROB_THRESHOLD,
)


def print_performance(label: str, perf: dict):
    print(f"\n  [{label}]")
    for k, v in perf.items():
        print(f"    {k:<30} {v:.4f}")


def run_symbol(symbol: str):
    print(f"\n{'=' * 50}")
    print(f"  SYMBOL: {symbol}")
    print(f"{'=' * 50}")

    # ------------------------------------------------------------------
    # STEP 1: Fetch data
    # ------------------------------------------------------------------
    df = fetch_data(API_KEY, SECRET_KEY, symbol, LOOKBACK_DAYS)

    # ------------------------------------------------------------------
    # STEP 2: Apply EMA strategy twice with different regime thresholds.
    #
    #         EMA strategy uses strict thresholds — proven to work well
    #         for trend-following signals.
    #
    #         ML strategy uses softer thresholds — the model already has
    #         regime as a feature internally, so the external gate just
    #         needs to filter obviously choppy markets, not demand
    #         perfect trends.
    # ------------------------------------------------------------------
    df_ema = apply_ema_strategy(
        df.copy(),
        short_span=SHORT_EMA,
        long_span=LONG_EMA,
        slope_threshold=EMA_SLOPE_THRESHOLD,
        r2_threshold=EMA_R2_THRESHOLD,
    )

    df_ml = apply_ema_strategy(
        df.copy(),
        short_span=SHORT_EMA,
        long_span=LONG_EMA,
        slope_threshold=ML_SLOPE_THRESHOLD,
        r2_threshold=ML_R2_THRESHOLD,
    )

    # ------------------------------------------------------------------
    # STEP 3: Features and target on ML dataframe.
    #         Target uses shift(-1) — only correct before splitting.
    # ------------------------------------------------------------------
    df_ml = make_features(df_ml)
    df_ml = add_target(df_ml)

    # Align EMA dataframe to same index after NaN drops
    df_ema = df_ema.reindex(df_ml.index)

    # ------------------------------------------------------------------
    # STEP 4: Time-based train/test split — 70% train, 30% test.
    #         Per-symbol training: model learns this symbol's patterns.
    # ------------------------------------------------------------------
    split_index = int(len(df_ml) * 0.7)

    train_ml = df_ml.iloc[:split_index].copy()
    test_ml  = df_ml.iloc[split_index:].copy()
    test_ema = df_ema.iloc[split_index:].copy()

    feature_cols = get_feature_columns()
    X_train = train_ml[feature_cols]
    y_train = train_ml["target"]
    X_test  = test_ml[feature_cols]

    # ------------------------------------------------------------------
    # STEP 5: Train model on this symbol's training data only.
    # ------------------------------------------------------------------
    model = train_model(X_train, y_train)

    # ------------------------------------------------------------------
    # STEP 6: Generate probability predictions on test data.
    # ------------------------------------------------------------------
    test_proba = predict_proba(model, X_test)

    # ------------------------------------------------------------------
    # STEP 7: Generate positions for all 4 strategies.
    # ------------------------------------------------------------------

    # Strategy 1: Buy & Hold
    bh_positions = test_ml["close"].copy()
    bh_positions[:] = 1
    bh_positions.name = "position"

    # Strategy 2: EMA with strict regime
    ema_positions = generate_positions(test_ema)

    # Strategy 3: ML only (no regime filter)
    ml_positions = generate_ml_positions(
        test_proba,
        test_ml,
        prob_threshold=ML_PROB_THRESHOLD,
        use_regime_filter=False,
    )

    # Strategy 4: ML + soft regime filter
    ml_regime_positions = generate_ml_positions(
        test_proba,
        test_ml,
        prob_threshold=ML_PROB_THRESHOLD,
        use_regime_filter=True,
    )

    # ------------------------------------------------------------------
    # STEP 8: Backtests
    # ------------------------------------------------------------------
    bh_bt        = run_backtest(test_ml, bh_positions,        TRANSACTION_COST)
    ema_bt       = run_backtest(test_ml, ema_positions,       TRANSACTION_COST)
    ml_bt        = run_backtest(test_ml, ml_positions,        TRANSACTION_COST)
    ml_regime_bt = run_backtest(test_ml, ml_regime_positions, TRANSACTION_COST)

    # ------------------------------------------------------------------
    # STEP 9: Evaluate and print 4-strategy comparison.
    # ------------------------------------------------------------------
    def evaluate(bt_result):
        eval_df = test_ml.copy()
        eval_df["returns"]          = bt_result["market_returns"]
        eval_df["strategy_returns"] = bt_result["strategy_returns"]
        return evaluate_performance(eval_df)

    print("\n--- TEST PERIOD PERFORMANCE ---")
    print_performance("Buy & Hold",         evaluate(bh_bt))
    print_performance("EMA Strategy",       evaluate(ema_bt))
    print_performance("ML Only",            evaluate(ml_bt))
    print_performance("ML + Regime Filter", evaluate(ml_regime_bt))


def main():
    for symbol in SYMBOLS:
        run_symbol(symbol)


if __name__ == "__main__":
    main()
