from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime, timedelta

def fetch_data(api_key, secret_key, symbol, days=30):
    client = StockHistoricalDataClient(api_key, secret_key)

    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Day,
        start=datetime.now() - timedelta(days=days)
    )

    bars = client.get_stock_bars(request)
    df = bars.df.reset_index()
    df = df[df['symbol'] == symbol]

    return df