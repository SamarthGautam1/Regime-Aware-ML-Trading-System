import pandas as pd
import numpy as np


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate ML features from OHLCV + strategy columns.

    All features are strictly backward-looking — no future data leaks in.
    Requires that apply_ema_strategy() has already been called on df,
    so that 'ema_200' and 'regime' columns are present.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain: 'close', 'ema_200', 'regime'

    Returns
    -------
    pd.DataFrame
        Original df with new feature columns added, NaNs dropped.
    """

    df = df.copy()

    # ------------------------------------------------------------------
    # 1. Short-term momentum features
    #    pct_change(n) gives the return over the last n days.
    #    shift(0) means we're using data up to and including today — no leakage.
    # ------------------------------------------------------------------
    df["return_1d"] = df["close"].pct_change(1)
    df["return_3d"] = df["close"].pct_change(3)
    df["return_5d"] = df["close"].pct_change(5)

    # ------------------------------------------------------------------
    # 2. Rolling volatility (10-day)
    #    Standard deviation of daily returns over the last 10 days.
    #    Tells the model how volatile the market has been recently.
    # ------------------------------------------------------------------
    df["volatility_10d"] = df["return_1d"].rolling(window=10).std()

    # ------------------------------------------------------------------
    # 3. EMA distance
    #    How far is the current price from EMA-200, as a percentage.
    #    Normalized so it's comparable across different price-level symbols.
    #    Positive = price above EMA-200 (bullish), Negative = below (bearish).
    # ------------------------------------------------------------------
    df["ema_distance"] = (df["close"] - df["ema_200"]) / df["ema_200"]

    # ------------------------------------------------------------------
    # 4. Z-score (20-day)
    #    How many standard deviations is today's close from its 20-day mean.
    #    Captures whether price is statistically stretched or compressed.
    # ------------------------------------------------------------------
    rolling_mean = df["close"].rolling(window=20).mean()
    rolling_std = df["close"].rolling(window=20).std()
    df["zscore_20d"] = (df["close"] - rolling_mean) / rolling_std

    # ------------------------------------------------------------------
    # 5. Regime flag (already computed by apply_ema_strategy)
    #    1 = trending market, 0 = choppy/sideways.
    #    Included as a feature so the ML model learns to respect regimes.
    # ------------------------------------------------------------------
    # 'regime' column is already in df — no computation needed here.

    # ------------------------------------------------------------------
    # Drop NaNs — rolling windows create NaNs at the start of the series.
    # We drop them here so the model never sees incomplete rows.
    # Index alignment is preserved by pandas automatically.
    # ------------------------------------------------------------------
    feature_columns = [
        "return_1d",
        "return_3d",
        "return_5d",
        "volatility_10d",
        "ema_distance",
        "zscore_20d",
        "regime",
    ]

    df = df.dropna(subset=feature_columns)

    return df


def get_feature_columns() -> list:
    """
    Returns the canonical list of feature column names.
    Use this everywhere you need to select features — single source of truth.
    """
    return [
        "return_1d",
        "return_3d",
        "return_5d",
        "volatility_10d",
        "ema_distance",
        "zscore_20d",
        "regime",
    ]


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add the prediction target to the dataframe.

    Target = 1 if tomorrow's return is positive, else 0.
    Uses shift(-1) which looks one day into the future —
    this is intentional and correct ONLY for the label.
    Features must never use shift(-1).

    Call this BEFORE splitting into train/test so the target
    is computed on the full series, then split cleanly.
    """
    df = df.copy()

    # Tomorrow's return, converted to binary direction
    df["target"] = (df["close"].pct_change().shift(-1) > 0).astype(int)

    # Drop the last row — it will have NaN target (no tomorrow available)
    df = df.dropna(subset=["target"])

    return df
