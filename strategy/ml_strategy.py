import pandas as pd


def generate_ml_positions(
    pred_proba: pd.Series,
    df: pd.DataFrame,
    prob_threshold: float = 0.6,
    use_regime_filter: bool = True,
) -> pd.Series:
    """
    Convert ML predicted probabilities into trading positions.

    Rules:
        - If predicted probability of up move > prob_threshold → long (1)
        - Otherwise → flat (0)
        - Optionally, only trade when regime == 1 (trending market)

    Why 0.6 threshold?
        A 0.5 threshold means any slight edge triggers a trade.
        0.6 means we only act when the model has reasonable conviction.
        This reduces overtrading and keeps signal quality higher.

    Why regime filter?
        The EMA strategy already showed that regime detection adds value.
        Combining ML signal with regime filter means we only trade when:
            (a) the model is confident AND
            (b) the market is in a trending state
        This reduces false signals in choppy markets.

    Parameters
    ----------
    pred_proba : pd.Series
        Output of predict_proba(). Index must align with df.
    df : pd.DataFrame
        Must contain 'regime' column if use_regime_filter=True.
    prob_threshold : float
        Minimum probability to trigger a long position. Default 0.6.
    use_regime_filter : bool
        Whether to gate trades on regime == 1. Default True.

    Returns
    -------
    pd.Series
        Integer position series: 1 (long) or 0 (flat). Named 'position'.
    """

    # Start with probability threshold signal
    signal = (pred_proba > prob_threshold).astype(int)

    # Optionally apply regime filter — only trade in trending markets
    if use_regime_filter:
        # Align regime with prediction index (test period only)
        regime = df["regime"].reindex(pred_proba.index)
        signal = signal * (regime == 1).astype(int)

    signal.name = "position"
    return signal
