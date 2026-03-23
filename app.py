import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Page config — must be the first Streamlit call ───────────────────────────
st.set_page_config(
    page_title="Regime-Aware ML Trading System",
    page_icon="📈",
    layout="wide",
)

# ── All project imports wrapped so a missing .env gives a clean error ─────────
_import_error: str | None = None
try:
    from config import (
        API_KEY,
        EMA_R2_THRESHOLD,
        EMA_SLOPE_THRESHOLD,
        LONG_EMA,
        LOOKBACK_DAYS,
        ML_PROB_THRESHOLD,
        ML_R2_THRESHOLD,
        ML_SLOPE_THRESHOLD,
        MR_NUM_STD,
        MR_WINDOW,
        SECRET_KEY,
        SHORT_EMA,
        SYMBOLS,
        TRANSACTION_COST,
    )
    from backtesting.backtest import run_backtest
    from backtesting.performance import evaluate_performance
    from backtesting.walk_forward import walk_forward_validation
    from data.fetch_data import fetch_data
    from features.feature_engineering import (
        add_target,
        get_feature_columns,
        make_features,
    )
    from models.predict import predict_proba
    from models.save_load import load_model, save_model
    from models.train_model import train_model
    from strategy.ema_strategy import apply_ema_strategy, generate_positions
    from strategy.mean_reversion import (
        apply_mean_reversion_strategy,
        generate_mr_positions,
    )
    from strategy.ml_strategy import generate_ml_positions
except Exception as _e:
    _import_error = str(_e)

# ── Constants ─────────────────────────────────────────────────────────────────
STRATEGY_COLORS = {
    "Buy & Hold":     "#888888",
    "EMA Strategy":   "#1f77b4",
    "ML Only":        "#ff7f0e",
    "ML + Regime":    "#2ca02c",
    "Mean Reversion": "#9467bd",
}

METRIC_LABELS = {
    "strategy_total_return": "Total Return",
    "cagr":                  "CAGR",
    "annualized_sharpe":     "Sharpe Ratio",
    "annualized_volatility": "Annualized Volatility",
    "max_drawdown":          "Max Drawdown",
}

METRIC_KEYS = list(METRIC_LABELS.keys())


# ── Helpers ───────────────────────────────────────────────────────────────────

def _hex_to_rgba(hex_color: str, alpha: float = 0.15) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ── Cached pipeline ───────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def run_pipeline(symbol: str) -> dict:
    """Full backtest pipeline for one symbol. Cached per symbol."""
    df = fetch_data(API_KEY, SECRET_KEY, symbol, LOOKBACK_DAYS)

    df_ema = apply_ema_strategy(
        df.copy(),
        short_span=SHORT_EMA,
        long_span=LONG_EMA,
        slope_threshold=EMA_SLOPE_THRESHOLD,
        r2_threshold=EMA_R2_THRESHOLD,
    )
    df_ema = apply_mean_reversion_strategy(df_ema, window=MR_WINDOW, num_std=MR_NUM_STD)

    df_ml = apply_ema_strategy(
        df.copy(),
        short_span=SHORT_EMA,
        long_span=LONG_EMA,
        slope_threshold=ML_SLOPE_THRESHOLD,
        r2_threshold=ML_R2_THRESHOLD,
    )
    df_ml = make_features(df_ml)
    df_ml = add_target(df_ml)
    df_ema = df_ema.reindex(df_ml.index)

    split_idx = int(len(df_ml) * 0.7)
    train_ml = df_ml.iloc[:split_idx].copy()
    test_ml  = df_ml.iloc[split_idx:].copy()
    test_ema = df_ema.iloc[split_idx:].copy()

    feature_cols = get_feature_columns()
    model = train_model(train_ml[feature_cols], train_ml["target"])
    save_model(model, symbol)

    test_proba = predict_proba(model, test_ml[feature_cols])

    bh_positions = test_ml["close"].copy()
    bh_positions[:] = 1
    bh_positions.name = "position"

    positions_map = {
        "Buy & Hold":     bh_positions,
        "EMA Strategy":   generate_positions(test_ema),
        "ML Only":        generate_ml_positions(
                              test_proba, test_ml,
                              prob_threshold=ML_PROB_THRESHOLD,
                              use_regime_filter=False),
        "ML + Regime":    generate_ml_positions(
                              test_proba, test_ml,
                              prob_threshold=ML_PROB_THRESHOLD,
                              use_regime_filter=True),
        "Mean Reversion": generate_mr_positions(test_ema),
    }

    backtests: dict = {}
    performances: dict = {}
    for name, pos in positions_map.items():
        bt = run_backtest(test_ml, pos, TRANSACTION_COST)
        backtests[name] = bt

        eval_df = test_ml.copy()
        eval_df["returns"]          = bt["market_returns"]
        eval_df["strategy_returns"] = bt["strategy_returns"]
        performances[name] = evaluate_performance(eval_df)

    return {
        "test_ml":      test_ml,
        "df_ml":        df_ml,
        "feature_cols": feature_cols,
        "backtests":    backtests,
        "performances": performances,
    }


@st.cache_data(show_spinner=False)
def run_walk_forward_cached(symbol: str) -> list:
    """Walk-forward validation. Reuses the cached pipeline internally."""
    data = run_pipeline(symbol)
    return walk_forward_validation(data["df_ml"], data["feature_cols"])


# ── Signal reader (read-only, no orders) ─────────────────────────────────────

def get_signal_details(symbol: str) -> dict:
    """
    Return pred_proba, regime, and signal for the latest bar.
    Loads the saved model — does NOT place any orders.
    """
    df = fetch_data(API_KEY, SECRET_KEY, symbol, LOOKBACK_DAYS)
    df = apply_ema_strategy(
        df,
        short_span=SHORT_EMA,
        long_span=LONG_EMA,
        slope_threshold=ML_SLOPE_THRESHOLD,
        r2_threshold=ML_R2_THRESHOLD,
    )
    df = make_features(df)

    feature_cols = get_feature_columns()
    model        = load_model(symbol)
    X_latest     = df[feature_cols].iloc[[-1]]

    pred_proba_val = float(model.predict_proba(X_latest)[0, 1])
    regime         = int(df["regime"].iloc[-1])
    signal         = 1 if (pred_proba_val > ML_PROB_THRESHOLD and regime == 1) else 0

    return {
        "Symbol":    symbol,
        "Pred Proba": round(pred_proba_val, 4),
        "Regime":    regime,
        "Signal":    signal,
    }


# ── Session state init ────────────────────────────────────────────────────────
if "analyzed_symbols" not in st.session_state:
    st.session_state["analyzed_symbols"] = set()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📈 Trading System")
    st.divider()

    symbol_list = SYMBOLS if not _import_error else ["AAPL", "MSFT", "NVDA", "SPY", "QQQ"]
    symbol = st.selectbox("Symbol", symbol_list)

    run_clicked = st.button("Run Analysis", type="primary", use_container_width=True)

    st.divider()
    st.caption(
        "Phase 1 (ML Layer) — ✅ COMPLETE\n"
        "Phase 2 (Paper Trading) — 🔄 IN PROGRESS\n"
        "Phase 3 (Robustness) — TODO"
    )

# ── Title and config guard ────────────────────────────────────────────────────
st.title("Regime-Aware ML Trading System")

if _import_error:
    st.error(
        f"**Configuration error:** {_import_error}\n\n"
        "Ensure a `.env` file with `API_KEY` and `SECRET_KEY` is present in the project root."
    )
    st.stop()

# ── Mark symbol as analyzed when button is clicked ────────────────────────────
if run_clicked:
    st.session_state["analyzed_symbols"].add(symbol)

analyzed = symbol in st.session_state["analyzed_symbols"]

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 Backtest Results", "🔄 Walk-Forward", "📡 Live Positions"])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Backtest Results
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    if not analyzed:
        st.info("Select a symbol in the sidebar and click **Run Analysis** to begin.")
    else:
        with st.spinner(f"Running pipeline for {symbol}…"):
            try:
                data = run_pipeline(symbol)
            except Exception as e:
                st.error(f"Pipeline failed: {e}")
                st.stop()

        performances = data["performances"]
        backtests    = data["backtests"]
        test_ml      = data["test_ml"]

        # ── Performance metrics table ─────────────────────────────────────────
        st.subheader("Performance Metrics")

        perf_df = pd.DataFrame(
            {
                name: {METRIC_LABELS[k]: f"{perf[k]:.4f}" for k in METRIC_KEYS}
                for name, perf in performances.items()
            }
        )
        st.dataframe(perf_df, use_container_width=True)

        # ── Equity curves ─────────────────────────────────────────────────────
        st.subheader("Equity Curves")

        fig_eq = go.Figure()
        for name, bt in backtests.items():
            equity = bt["equity_curve"].reindex(test_ml.index)
            fig_eq.add_trace(go.Scatter(
                x=equity.index,
                y=equity.values,
                name=name,
                mode="lines",
                line=dict(color=STRATEGY_COLORS[name], width=1.5),
            ))
        fig_eq.update_layout(
            title=f"Equity Curves — {symbol}",
            xaxis_title="Date",
            yaxis_title="Cumulative Return (×)",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_eq, use_container_width=True)

        # ── Drawdown ──────────────────────────────────────────────────────────
        st.subheader("Drawdown")

        fig_dd = go.Figure()
        for name, bt in backtests.items():
            equity = bt["equity_curve"].reindex(test_ml.index)
            peak   = equity.cummax()
            dd     = (equity - peak) / peak
            color  = STRATEGY_COLORS[name]
            fig_dd.add_trace(go.Scatter(
                x=dd.index,
                y=dd.values,
                name=name,
                mode="lines",
                line=dict(color=color, width=1.2),
                fill="tozeroy",
                fillcolor=_hex_to_rgba(color, alpha=0.12),
            ))
        fig_dd.update_layout(
            title=f"Drawdown — {symbol}",
            xaxis_title="Date",
            yaxis_title="Drawdown",
            yaxis_tickformat=".0%",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_dd, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Walk-Forward Validation
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    if not analyzed:
        st.info("Select a symbol in the sidebar and click **Run Analysis** to begin.")
    else:
        with st.spinner(f"Running walk-forward validation for {symbol}…"):
            try:
                wf_results = run_walk_forward_cached(symbol)
            except Exception as e:
                st.error(f"Walk-forward validation failed: {e}")
                st.stop()

        if not wf_results:
            st.warning("No walk-forward results returned — dataset may be too small.")
        else:
            st.subheader("Walk-Forward Results (ML + Regime, 5 Folds)")

            # ── Results table ─────────────────────────────────────────────────
            wf_df = pd.DataFrame(wf_results)[
                ["fold", "start", "end", "annualized_sharpe", "strategy_total_return"]
            ].rename(columns={
                "fold":                   "Fold",
                "start":                  "Start",
                "end":                    "End",
                "annualized_sharpe":      "Sharpe Ratio",
                "strategy_total_return":  "Total Return",
            })
            wf_df["Sharpe Ratio"] = wf_df["Sharpe Ratio"].map("{:.4f}".format)
            wf_df["Total Return"] = wf_df["Total Return"].map("{:.4f}".format)
            st.dataframe(wf_df.set_index("Fold"), use_container_width=True)

            # ── Sharpe bar chart ──────────────────────────────────────────────
            st.subheader("Sharpe Ratio per Fold")

            sharpe_vals = [r["annualized_sharpe"] for r in wf_results]
            bar_colors  = ["#2ca02c" if s > 0 else "#d62728" for s in sharpe_vals]

            fig_wf = go.Figure(go.Bar(
                x=[f"Fold {r['fold']}<br>{r['start']}→{r['end']}" for r in wf_results],
                y=sharpe_vals,
                marker_color=bar_colors,
                text=[f"{s:.4f}" for s in sharpe_vals],
                textposition="outside",
            ))
            fig_wf.update_layout(
                title=f"Walk-Forward Sharpe Ratio — {symbol}",
                xaxis_title="Fold",
                yaxis_title="Annualized Sharpe",
                showlegend=False,
            )
            fig_wf.add_hline(y=0, line_dash="dash", line_color="black", line_width=1)
            st.plotly_chart(fig_wf, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Live Positions
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    paper_key    = os.getenv("ALPACA_PAPER_KEY")
    paper_secret = os.getenv("ALPACA_PAPER_SECRET")

    if not paper_key or not paper_secret:
        st.error(
            "Alpaca paper trading credentials not found.\n\n"
            "Add `ALPACA_PAPER_KEY` and `ALPACA_PAPER_SECRET` to your `.env` file."
        )
    else:
        try:
            from alpaca.trading.client import TradingClient
            from alpaca.trading.enums import QueryOrderStatus
            from alpaca.trading.requests import GetOrdersRequest

            client = TradingClient(paper_key, paper_secret, paper=True)

            # ── Account metrics ───────────────────────────────────────────────
            try:
                account = client.get_account()
                col1, col2, col3 = st.columns(3)
                col1.metric("Portfolio Equity",  f"${float(account.equity):,.2f}")
                col2.metric("Buying Power",       f"${float(account.buying_power):,.2f}")
                col3.metric("Cash",               f"${float(account.cash):,.2f}")
            except Exception as e:
                st.error(f"Could not fetch account info: {e}")

            st.divider()

            # ── Current positions ─────────────────────────────────────────────
            st.subheader("Current Positions")
            try:
                positions = client.get_all_positions()
                if not positions:
                    st.info("No open positions.")
                else:
                    pos_rows = [
                        {
                            "Symbol":       p.symbol,
                            "Qty":          float(p.qty),
                            "Market Value": f"${float(p.market_value):,.2f}",
                            "Avg Entry":    f"${float(p.avg_entry_price):,.2f}",
                            "Unrealized P&L": f"${float(p.unrealized_pl):,.2f}",
                            "P&L %":        f"{float(p.unrealized_plpc) * 100:.2f}%",
                        }
                        for p in positions
                    ]
                    st.dataframe(
                        pd.DataFrame(pos_rows).set_index("Symbol"),
                        use_container_width=True,
                    )
            except Exception as e:
                st.error(f"Could not fetch positions: {e}")

            st.divider()

            # ── Signal generation (read-only) ─────────────────────────────────
            st.subheader("Today's Signals")
            st.caption("Reads saved models and latest market data. No orders are placed.")

            if st.button("Generate Signals", type="secondary"):
                signal_rows = []
                progress = st.progress(0, text="Fetching signals…")

                for i, sym in enumerate(SYMBOLS):
                    progress.progress(
                        (i + 1) / len(SYMBOLS),
                        text=f"Processing {sym}…",
                    )
                    try:
                        signal_rows.append(get_signal_details(sym))
                    except FileNotFoundError:
                        signal_rows.append({
                            "Symbol":     sym,
                            "Pred Proba": "—",
                            "Regime":     "—",
                            "Signal":     "No model saved",
                        })
                    except Exception as e:
                        signal_rows.append({
                            "Symbol":     sym,
                            "Pred Proba": "—",
                            "Regime":     "—",
                            "Signal":     f"Error: {e}",
                        })

                progress.empty()

                signals_df = pd.DataFrame(signal_rows).set_index("Symbol")
                st.dataframe(signals_df, use_container_width=True)

        except Exception as e:
            st.error(f"Failed to connect to Alpaca: {e}")
