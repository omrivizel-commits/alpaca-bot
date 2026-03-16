"""
Chart Renderer — generates a 4-hour candlestick PNG for the Vision Agent.
"""

import os
import mplfinance as mpf
from data.market_data import get_hourly

CHART_DIR = os.path.join(os.path.dirname(__file__), "..", "state")


def render_chart(symbol: str, bars: int = 96) -> str:
    """
    Renders the last `bars` hourly candles to a PNG file.
    Returns the absolute path to the saved image.
    """
    df = get_hourly(symbol)
    df = df.tail(bars)[["Open", "High", "Low", "Close", "Volume"]]

    os.makedirs(CHART_DIR, exist_ok=True)
    chart_path = os.path.abspath(os.path.join(CHART_DIR, f"chart_{symbol}.png"))

    mpf.plot(
        df,
        type="candle",
        style="charles",
        title=f"{symbol} — 4H Chart (last {bars} bars)",
        volume=True,
        mav=(20, 50),          # SMA overlays for S/R context
        savefig=chart_path,
    )
    return chart_path
