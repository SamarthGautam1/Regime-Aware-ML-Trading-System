import numpy as np
import pandas as pd


def apply_mean_reversion_strategy(
    df: pd.DataFrame,
    window: int = 20,
    num_std: float = 2.0,
) -> pd.DataFrame:
    """
    Add Bollinger Bands and a mean-reversion signal to df.

    Signal logic (stateful):
      Enter long (1): close crosses below lower band AND regime == 0
      Exit  flat (0): close crosses above upper band OR  regime == 1

    Exit takes priority when both conditions fire on the same bar.
    Regime == 0 targets sideways/choppy markets where mean reversion
    is more likely to succeed than trend-following.

    Parameters
    ----------
    df      : DataFrame with at least 'close' and 'regime' columns.
    window  : Rolling window for the Bollinger Band calculation.
    num_std : Number of standard deviations for the band width.

    Adds columns: bb_middle, bb_upper, bb_lower, bb_signal.
    Returns the modified DataFrame.
    """
    df = df.copy()

    rolling      = df["close"].rolling(window)
    df["bb_middle"] = rolling.mean()
    rolling_std  = rolling.std()
    df["bb_upper"]  = df["bb_middle"] + num_std * rolling_std
    df["bb_lower"]  = df["bb_middle"] - num_std * rolling_std

    close  = df["close"]
    lower  = df["bb_lower"]
    upper  = df["bb_upper"]
    regime = df["regime"]

    # A "cross below lower" happens when the previous bar was at or above
    # the lower band and the current bar closes beneath it.
    cross_below_lower = (close.shift(1) >= lower.shift(1)) & (close < lower)

    # A "cross above upper" happens when the previous bar was at or below
    # the upper band and the current bar closes above it.
    cross_above_upper = (close.shift(1) <= upper.shift(1)) & (close > upper)

    entry = cross_below_lower & (regime == 0)
    exit_ = cross_above_upper | (regime == 1)

    # Build a stateful signal using ffill:
    #   - Mark 1 at every entry, 0 at every exit, NaN elsewhere.
    #   - Setting exit_ after entry_ means exit wins on the same bar.
    #   - Forward-fill carries the last explicit state forward.
    #   - fillna(0) handles the period before the first signal.
    raw = pd.Series(np.nan, index=df.index)
    raw[entry] = 1
    raw[exit_] = 0

    df["bb_signal"] = raw.ffill().fillna(0).astype(int)

    return df


def generate_mr_positions(df: pd.DataFrame) -> pd.Series:
    """Return the bb_signal column as a position series clipped to [0, 1]."""
    position = df["bb_signal"].clip(0, 1)
    position.name = "position"
    return position
