import pandas as pd


def run_backtest(df: pd.DataFrame, positions: pd.Series, transaction_cost: float):
    market_returns = df["close"].pct_change()

    aligned_positions = positions.reindex(df.index).fillna(0)
    strategy_returns = aligned_positions.shift(1) * market_returns

    position_changes = aligned_positions.diff().abs().fillna(0)
    strategy_returns = strategy_returns - (position_changes * transaction_cost)

    equity_curve = (1 + strategy_returns.fillna(0)).cumprod()

    trade_mask = position_changes > 0
    trade_log = pd.DataFrame(
        {
            "position": aligned_positions,
            "position_change": position_changes,
        },
        index=df.index,
    ).loc[trade_mask]

    return {
        "strategy_returns": strategy_returns,
        "equity_curve": equity_curve,
        "market_returns": market_returns,
        "trade_log": trade_log,
    }
