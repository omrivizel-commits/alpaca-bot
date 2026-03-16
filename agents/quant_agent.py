"""
Quant Agent — The "Cold Logic" Specialist.

Day-trade mode:
  - Current signal computed on HOURLY candles (more frequent triggers)
  - Statistical edge validated on 5y DAILY data (permutation test)
  - OR logic: either BB or RSI can fire (no longer requires both)
  - Relaxed thresholds: BB 1.5σ, RSI oversold<40 / overbought>60

Market filters added (pure math, no AI):
  - VWAP      : intraday institutional flow direction
  - ADX       : Average Directional Index — trending vs choppy regime
  - Rel. Vol  : today's volume vs 20-day average — confirms conviction
"""

import numpy as np
import pandas as pd
from data.market_data import get_historical, get_hourly, get_intraday_5m


# ──────────────────────────────────────────────────────────────────────────────
# Strategy 1: Bollinger Band Mean Reversion  (1.5σ — fires more often)
# ──────────────────────────────────────────────────────────────────────────────

class BollingerBandStrategy:
    def __init__(self, window: int = 20, n_std: float = 1.5):
        self.window = window
        self.n_std = n_std

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        sma   = df["Close"].rolling(self.window).mean()
        std   = df["Close"].rolling(self.window).std()
        upper = sma + self.n_std * std
        lower = sma - self.n_std * std

        signals = pd.Series(0, index=df.index)
        signals[df["Close"] < lower] =  1   # below lower band → BUY
        signals[df["Close"] > upper] = -1   # above upper band → SELL
        return signals

    def backtest(self, df: pd.DataFrame) -> float:
        signals = self.generate_signals(df)
        daily_returns = df["Close"].pct_change()
        return float((signals.shift(1) * daily_returns).sum())


# ──────────────────────────────────────────────────────────────────────────────
# Strategy 2: RSI Momentum  (40/60 thresholds — fires more often)
# ──────────────────────────────────────────────────────────────────────────────

class RSIStrategy:
    def __init__(self, period: int = 14, oversold: int = 40, overbought: int = 60):
        self.period     = period
        self.oversold   = oversold
        self.overbought = overbought

    def _compute_rsi(self, close: pd.Series) -> pd.Series:
        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(self.period).mean()
        loss  = (-delta.clip(upper=0)).rolling(self.period).mean()
        rs    = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        rsi     = self._compute_rsi(df["Close"])
        signals = pd.Series(0, index=df.index)
        signals[rsi < self.oversold]   =  1    # oversold → BUY
        signals[rsi > self.overbought] = -1    # overbought → SELL
        return signals

    def backtest(self, df: pd.DataFrame) -> float:
        signals = self.generate_signals(df)
        daily_returns = df["Close"].pct_change()
        return float((signals.shift(1) * daily_returns).sum())


# ──────────────────────────────────────────────────────────────────────────────
# Permutation Test  (Anti-Gambling Filter — runs on daily data)
# ──────────────────────────────────────────────────────────────────────────────

def permutation_test(df: pd.DataFrame, strategy, n_permutations: int = 1000) -> float:
    real_return  = strategy.backtest(df)
    daily_returns = df["Close"].pct_change().dropna().values

    better_count = 0
    for _ in range(n_permutations):
        shuffled = np.random.permutation(daily_returns)
        shuffled_prices = pd.DataFrame(
            {"Close": np.cumprod(1 + shuffled) * df["Close"].iloc[0]},
            index=df.index[-len(shuffled):],
        )
        if strategy.backtest(shuffled_prices) >= real_return:
            better_count += 1

    return better_count / n_permutations


# ──────────────────────────────────────────────────────────────────────────────
# VWAP  — Intraday Institutional Flow
# ──────────────────────────────────────────────────────────────────────────────

def _compute_vwap(df_5m: pd.DataFrame) -> float:
    """
    Volume-Weighted Average Price from today's 5-minute bars.
    Returns 0.0 if data is unavailable or volume is zero.
    """
    if df_5m.empty or "Volume" not in df_5m.columns:
        return 0.0
    vol = df_5m["Volume"]
    if vol.sum() == 0:
        return 0.0
    tp = (df_5m["High"] + df_5m["Low"] + df_5m["Close"]) / 3
    return round(float((tp * vol).sum() / vol.sum()), 4)


# ──────────────────────────────────────────────────────────────────────────────
# ADX  — Average Directional Index (Trend Strength / Market Regime)
# ──────────────────────────────────────────────────────────────────────────────

def _compute_adx(df: pd.DataFrame, period: int = 14) -> float:
    """
    Wilder-smoothed ADX.
      > 25  → TRENDING  (trade signals are reliable)
      20-25 → NEUTRAL
      < 20  → CHOPPY    (block all trades — signals are noise)
    """
    high  = df["High"]
    low   = df["Low"]
    close = df["Close"]

    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)

    up_move   = high.diff()
    down_move = -low.diff()
    plus_dm   = up_move.where((up_move > down_move) & (up_move > 0),    0.0)
    minus_dm  = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    kw = {"alpha": 1 / period, "min_periods": period, "adjust": False}
    atr      = tr.ewm(**kw).mean()
    di_plus  = 100 * (plus_dm.ewm(**kw).mean()  / atr.replace(0, np.nan))
    di_minus = 100 * (minus_dm.ewm(**kw).mean() / atr.replace(0, np.nan))

    dx  = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, np.nan)
    adx = dx.ewm(**kw).mean()

    last = adx.dropna()
    return round(float(last.iloc[-1]), 2) if not last.empty else 20.0


# ──────────────────────────────────────────────────────────────────────────────
# Relative Volume  — Conviction Filter
# ──────────────────────────────────────────────────────────────────────────────

def _compute_relative_volume(df_hourly: pd.DataFrame) -> float:
    """
    today_volume / avg_volume_of_last_20_trading_days.
      > 1.5  → strong conviction
      1.0-1.5 → normal
      < 1.0  → quiet session — block trade (likely fake move)
    """
    if df_hourly.empty or "Volume" not in df_hourly.columns:
        return 1.0

    df = df_hourly.copy()
    df.index = pd.to_datetime(df.index)
    if hasattr(df.index, "tz") and df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    daily_vol = df.groupby(df.index.date)["Volume"].sum()
    if len(daily_vol) < 2:
        return 1.0

    today_vol = float(daily_vol.iloc[-1])
    avg_vol   = float(daily_vol.iloc[:-1].tail(20).mean())
    return round(today_vol / avg_vol, 2) if avg_vol > 0 else 1.0


# ──────────────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────────────

def run_quant(symbol: str) -> dict:
    """
    Quant pipeline — day-trade mode.

    Signal source  : hourly candles (last 60 days) — fast, intraday-sensitive
    Edge validation: 5y daily permutation test — ensures signal isn't random noise
    Logic          : OR — either BB or RSI fires (no conflict allowed)

    Additional market filters:
      adx            : trend strength (14-period, Wilder EMA)
      market_regime  : TRENDING / NEUTRAL / CHOPPY
      vwap           : today's intraday VWAP (0 if market closed / no data)
      relative_volume: today's vol / 20-day avg vol
    """

    # ── Statistical edge validation (daily, 5y) ───────────────────────────────
    df_daily = get_historical(symbol, period="5y")
    bb  = BollingerBandStrategy()
    rsi = RSIStrategy()

    bb_pvalue  = permutation_test(df_daily, bb)
    rsi_pvalue = permutation_test(df_daily, rsi)
    bb_green   = bb_pvalue  < 0.01
    rsi_green  = rsi_pvalue < 0.01

    # ── Current signal (hourly, last 60 days) ─────────────────────────────────
    try:
        df_signal = get_hourly(symbol, period="60d")
    except Exception:
        df_signal = df_daily.tail(120)

    bb_sig  = int(bb.generate_signals(df_signal).iloc[-1])
    rsi_sig = int(rsi.generate_signals(df_signal).iloc[-1])

    combined = bb_sig + rsi_sig
    if combined > 0:
        signal = "BUY"
        ensemble_agreement = True
    elif combined < 0:
        signal = "SELL"
        ensemble_agreement = True
    else:
        signal = "HOLD"
        ensemble_agreement = False

    green_light = (bb_green or rsi_green) and ensemble_agreement

    # ── ADX — Market Regime ───────────────────────────────────────────────────
    adx           = 20.0
    market_regime = "NEUTRAL"
    try:
        adx = _compute_adx(df_signal)   # reuse already-fetched hourly data
        if adx >= 25:
            market_regime = "TRENDING"
        elif adx < 20:
            market_regime = "CHOPPY"
        else:
            market_regime = "NEUTRAL"
    except Exception:
        pass

    # ── VWAP — Intraday Institutional Flow ───────────────────────────────────
    vwap = 0.0
    try:
        df_5m = get_intraday_5m(symbol)
        vwap  = _compute_vwap(df_5m)
    except Exception:
        pass

    # ── Relative Volume — Conviction Check ───────────────────────────────────
    relative_volume = 1.0
    try:
        relative_volume = _compute_relative_volume(df_signal)
    except Exception:
        pass

    return {
        "symbol":             symbol,
        "signal":             signal,
        "ensemble_agreement": ensemble_agreement,
        "bb_signal":          bb_sig,
        "rsi_signal":         rsi_sig,
        "bb_pvalue":          round(bb_pvalue,  4),
        "rsi_pvalue":         round(rsi_pvalue, 4),
        "green_light":        green_light,
        # Market regime filter
        "adx":                adx,
        "market_regime":      market_regime,
        # VWAP filter
        "vwap":               vwap,
        # Volume filter
        "relative_volume":    relative_volume,
    }
