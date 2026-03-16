import yfinance as yf
import pandas as pd
import concurrent.futures
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestTradeRequest
from config import settings

_YF_TIMEOUT = 15  # seconds per yfinance call

def _yf(fn, **kwargs):
    """Run a yfinance call in a thread with a hard timeout."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        future = ex.submit(fn, **kwargs)
        try:
            return future.result(timeout=_YF_TIMEOUT)
        except concurrent.futures.TimeoutError:
            raise ValueError(f"yfinance timed out after {_YF_TIMEOUT}s")

_data_client: StockHistoricalDataClient | None = None


def _get_data_client() -> StockHistoricalDataClient:
    global _data_client
    if _data_client is None:
        _data_client = StockHistoricalDataClient(
            settings.ALPACA_API_KEY,
            settings.ALPACA_SECRET_KEY,
        )
    return _data_client


def get_historical(symbol: str, period: str = "5y") -> pd.DataFrame:
    """Fetches deep OHLCV history via yfinance for backtesting and permutation tests."""
    ticker = yf.Ticker(symbol)
    df = _yf(ticker.history, period=period)
    if df.empty:
        raise ValueError(f"No historical data found for {symbol}")
    return df


def get_hourly(symbol: str, period: str = "60d") -> pd.DataFrame:
    """Fetches hourly OHLCV data for chart rendering (Vision Agent)."""
    ticker = yf.Ticker(symbol)
    df = _yf(ticker.history, period=period, interval="1h")
    if df.empty:
        raise ValueError(f"No hourly data found for {symbol}")
    return df


def get_current_price(symbol: str) -> float:
    """Returns the most recent trade price via Alpaca."""
    req = StockLatestTradeRequest(symbol_or_symbols=symbol)
    trade = _get_data_client().get_stock_latest_trade(req)
    return float(trade[symbol].price)


def get_intraday_5m(symbol: str) -> pd.DataFrame:
    """Fetches today's 5-minute OHLCV bars for VWAP computation."""
    ticker = yf.Ticker(symbol)
    try:
        df = _yf(ticker.history, period="1d", interval="5m")
    except ValueError:
        return pd.DataFrame()
    return df


def get_days_to_earnings(symbol: str) -> int:
    """
    Returns the number of calendar days until the next earnings report.
    Returns 999 if the date is unknown or cannot be fetched.
    """
    from datetime import date as _date
    try:
        ed = yf.Ticker(symbol).earnings_dates
        if ed is None or ed.empty:
            return 999
        today = _date.today()
        future = [
            idx.date() for idx in ed.index
            if hasattr(idx, "date") and idx.date() >= today
        ]
        if not future:
            return 999
        return (min(future) - today).days
    except Exception:
        return 999
