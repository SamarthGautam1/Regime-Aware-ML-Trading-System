import numpy as np
from config import TRADING_DAYS_PER_YEAR


def calculate_total_return(returns_series):
    cumulative = (1 + returns_series).cumprod()
    return cumulative.iloc[-1] - 1


def calculate_annualized_volatility(returns_series):
    daily_vol = returns_series.std()
    return daily_vol * np.sqrt(TRADING_DAYS_PER_YEAR)


def calculate_annualized_sharpe(returns_series):
    mean_return = returns_series.mean()
    volatility = returns_series.std()

    if volatility == 0:
        return 0

    daily_sharpe = mean_return / volatility
    return daily_sharpe * np.sqrt(TRADING_DAYS_PER_YEAR)


def calculate_max_drawdown(returns_series):
    cumulative = (1 + returns_series).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    return drawdown.min()


def calculate_cagr(returns_series):
    cumulative = (1 + returns_series).cumprod()
    total_return = cumulative.iloc[-1]
    num_days = len(returns_series)

    years = num_days / TRADING_DAYS_PER_YEAR
    if years == 0:
        return 0

    return total_return ** (1 / years) - 1


def evaluate_performance(df):
    strategy_returns = df['strategy_returns'].dropna()
    market_returns = df['returns'].dropna()

    performance = {
        "strategy_total_return": calculate_total_return(strategy_returns),
        "market_total_return": calculate_total_return(market_returns),
        "cagr": calculate_cagr(strategy_returns),
        "annualized_volatility": calculate_annualized_volatility(strategy_returns),
        "annualized_sharpe": calculate_annualized_sharpe(strategy_returns),
        "max_drawdown": calculate_max_drawdown(strategy_returns)
    }

    return performance