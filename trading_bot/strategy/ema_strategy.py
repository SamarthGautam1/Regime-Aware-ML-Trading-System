import numpy as np

def rolling_regression_stats(series, window=20):
    slopes = []
    r_squared = []

    for i in range(len(series)):
        if i < window:
            slopes.append(np.nan)
            r_squared.append(np.nan)
        else:
            y = series.iloc[i-window:i].values
            x = np.arange(window)

            # Fit regression
            slope, intercept = np.polyfit(x, y, 1)

            # Predicted values
            y_pred = slope * x + intercept

            # R² calculation
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)

            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

            slopes.append(slope)
            r_squared.append(r2)

    return np.array(slopes), np.array(r_squared)


def apply_ema_strategy(df,
                       short_span=10,
                       long_span=20,
                       filter_span=200,
                       slope_window=20,
                       slope_threshold=0.0007,
                       r2_threshold=0.6):

    df['ema_short'] = df['close'].ewm(span=short_span).mean()
    df['ema_long'] = df['close'].ewm(span=long_span).mean()
    df['ema_200'] = df['close'].ewm(span=filter_span).mean()

    df['log_close'] = np.log(df['close'])
    slope, r2 = rolling_regression_stats(df['log_close'], slope_window)
    df['log_slope'] = slope
    df['log_r2'] = r2

    df['regime'] = 0
    df.loc[
        (abs(df['log_slope']) > slope_threshold) &
(df['log_r2'] > r2_threshold),
        'regime'
    ] = 1

    df['signal'] = 0

    long_condition = (
        (df['ema_short'] > df['ema_long']) &
        (df['close'] > df['ema_200']) &
        (df['regime'] == 1)
    )

    df.loc[long_condition, 'signal'] = 1

    return df


def backtest_strategy(df, transaction_cost=0.001):
    # Market returns
    df['returns'] = df['close'].pct_change()

    # Strategy returns (shifted to avoid lookahead bias)
    df['strategy_returns'] = df['signal'].shift(1) * df['returns']

    # Detect position changes
    df['position_change'] = df['signal'].diff().abs()

    # Apply cost when position changes
    df['strategy_returns'] -= df['position_change'] * transaction_cost

    return df