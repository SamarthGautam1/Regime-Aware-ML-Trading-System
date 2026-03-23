import os

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

ARTIFACTS_DIR = "artifacts"

STRATEGY_COLORS = {
    "Buy & Hold":     "#888888",
    "EMA Strategy":   "#1f77b4",
    "ML Only":        "#ff7f0e",
    "ML + Regime":    "#2ca02c",
    "Mean Reversion": "#9467bd",
}


def _compute_drawdown(equity_curve: pd.Series) -> pd.Series:
    peak = equity_curve.cummax()
    return (equity_curve - peak) / peak


def plot_results(symbol: str, test_df: pd.DataFrame, backtest_results: dict) -> None:
    """
    Plot equity curves and drawdowns for all strategies, save to artifacts/.

    Parameters
    ----------
    symbol           : Ticker symbol (used in titles and filename).
    test_df          : Test-period DataFrame — provides the datetime index.
    backtest_results : Dict mapping strategy label → run_backtest() result dict.
                       Each result must contain 'equity_curve' (pd.Series).
    """
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    fig.suptitle(symbol, fontsize=14, fontweight="bold")

    for label, bt in backtest_results.items():
        color  = STRATEGY_COLORS.get(label, None)
        equity = bt["equity_curve"].reindex(test_df.index)

        # --- Subplot 1: Equity curves ---
        ax1.plot(equity.index, equity.values, label=label, color=color, linewidth=1.5)

        # --- Subplot 2: Drawdown ---
        dd = _compute_drawdown(equity)
        ax2.plot(dd.index, dd.values, label=label, color=color, linewidth=1.2)
        ax2.fill_between(dd.index, dd.values, 0, color=color, alpha=0.3)

    # Formatting — ax1
    ax1.set_title(f"Equity Curves — {symbol}")
    ax1.set_ylabel("Cumulative Return (×)")
    ax1.axhline(y=1.0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Formatting — ax2
    ax2.set_title(f"Drawdown — {symbol}")
    ax2.set_ylabel("Drawdown (%)")
    ax2.axhline(y=0.0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax2.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda y, _: f"{y * 100:.0f}%")
    )
    ax2.legend(loc="lower left", fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.5)

    # X-axis: readable dates on the shared axis
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    fig.autofmt_xdate(rotation=30)

    plt.tight_layout()

    out_path = os.path.join(ARTIFACTS_DIR, f"{symbol}_performance.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Chart saved: {out_path}")

    plt.show()
    plt.close(fig)
