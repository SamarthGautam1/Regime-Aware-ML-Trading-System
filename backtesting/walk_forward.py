import pandas as pd

from models.train_model import train_model
from models.predict import predict_proba
from strategy.ml_strategy import generate_ml_positions
from backtesting.backtest import run_backtest
from backtesting.performance import evaluate_performance
from config import ML_PROB_THRESHOLD, TRANSACTION_COST


def walk_forward_validation(df: pd.DataFrame, feature_cols: list, n_splits: int = 5) -> list:
    """
    Walk-forward validation: train on expanding window, test on next fold.

    For n_splits folds the data is divided into (n_splits + 1) equal blocks.
    Fold k trains on blocks 0..k and tests on block k+1, so the training
    window grows with each fold (expanding window).

    Parameters
    ----------
    df : pd.DataFrame
        Feature-engineered dataframe with 'target', 'close', 'regime', and
        all feature columns. Must have a DatetimeIndex.
    feature_cols : list
        Column names used as model inputs (from get_feature_columns()).
    n_splits : int
        Number of test folds. Requires len(df) >= n_splits + 1 blocks.

    Returns
    -------
    list of dict
        One performance dict per fold (keys: fold, start, end, + evaluate_performance keys).
    """
    n = len(df)
    fold_size = n // (n_splits + 1)
    results = []

    for k in range(n_splits):
        # Train: rows 0 .. (k+1)*fold_size  (expanding window)
        train_end = (k + 1) * fold_size
        # Test: next fold_size rows
        test_start = train_end
        test_end = test_start + fold_size

        train_df = df.iloc[:train_end].copy()
        test_df = df.iloc[test_start:test_end].copy()

        # Drop rows where target is NaN (last row of each slice after shift(-1))
        train_df = train_df.dropna(subset=["target"] + feature_cols)
        test_df = test_df.dropna(subset=feature_cols)

        if len(train_df) < 10 or len(test_df) < 2:
            continue

        X_train = train_df[feature_cols]
        y_train = train_df["target"]
        X_test = test_df[feature_cols]

        model = train_model(X_train, y_train)
        test_proba = predict_proba(model, X_test)

        positions = generate_ml_positions(
            test_proba,
            test_df,
            prob_threshold=ML_PROB_THRESHOLD,
            use_regime_filter=True,
        )

        bt = run_backtest(test_df, positions, TRANSACTION_COST)

        eval_df = test_df.copy()
        eval_df["returns"] = bt["market_returns"]
        eval_df["strategy_returns"] = bt["strategy_returns"]
        perf = evaluate_performance(eval_df)

        results.append({
            "fold": k + 1,
            "start": str(test_df.index[0].date()),
            "end": str(test_df.index[-1].date()),
            **perf,
        })

    return results


def print_walk_forward_results(results: list):
    """Print Sharpe ratio and total return for each walk-forward fold."""
    print("\n--- WALK-FORWARD VALIDATION (ML + Regime) ---")
    if not results:
        print("  No results to display.")
        return

    header = f"  {'Fold':<6} {'Start':<12} {'End':<12} {'Sharpe':>8} {'Total Ret':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for r in results:
        print(
            f"  {r['fold']:<6} {r['start']:<12} {r['end']:<12}"
            f" {r['annualized_sharpe']:>8.4f} {r['strategy_total_return']:>10.4f}"
        )
