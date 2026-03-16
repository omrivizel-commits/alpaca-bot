"""
Signal Agent — Pure Mathematical Day Trading Signal Generator.

Computes RSI, MACD, Moving Averages, Bollinger Bands, and Volume indicators
from live OHLCV data, then applies a deterministic scoring system to generate
a structured trading signal — no AI calls required.

Scoring table:
  RSI < 25              → +2.5 | RSI < 30 → +2.0 | RSI < 40 → +1.0
  RSI > 80              → −2.5 | RSI > 70 → −2.0 | RSI > 60 → −1.0
  MACD hist > 0 rising  → +1.0 | hist > 0 flat → +0.5
  MACD hist < 0 falling → −1.0 | hist < 0 flat → −0.5
  Price in BB lower 10% → +2.0 | lower 25% → +1.0
  Price in BB upper 90% → −2.0 | upper 75% → −1.0
  Price > MA50 + MA20>50→ +1.0 | Price > MA50 only → +0.5
  Price < MA50 + MA20<50→ −1.0 | Price < MA50 only → −0.5
  Volume > 1.5× MA (confirming direction) → ±0.5

Signal threshold: score ≥ 3 → BUY | score ≤ −3 → SELL | else HOLD
Confidence: abs(score) / MAX_SCORE * 100, clamped [0, 95]
"""

import numpy as np
import pandas as pd
from datetime import datetime, timezone

from data.market_data import get_hourly


# ── Indicator computation ─────────────────────────────────────────────────────

def _rsi(close: pd.Series, period: int = 14) -> float:
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    val   = (100 - 100 / (1 + rs)).iloc[-1]
    return round(float(val), 2) if not np.isnan(val) else 50.0


def _macd(close: pd.Series) -> dict:
    ema12  = close.ewm(span=12, adjust=False).mean()
    ema26  = close.ewm(span=26, adjust=False).mean()
    line   = ema12 - ema26
    signal = line.ewm(span=9, adjust=False).mean()
    hist   = line - signal
    return {
        "line":      round(float(line.iloc[-1]),    4),
        "signal":    round(float(signal.iloc[-1]),  4),
        "histogram": round(float(hist.iloc[-1]),    4),
        "hist_prev": round(float(hist.iloc[-2]),    4) if len(hist) > 1 else 0.0,
    }


def _compute_indicators(df: pd.DataFrame) -> dict:
    close  = df["Close"]
    volume = df["Volume"]

    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()

    ma200_raw = close.rolling(200).mean().iloc[-1]
    ma200     = round(float(ma200_raw), 2) if not np.isnan(ma200_raw) \
                else round(float(close.rolling(50).mean().iloc[-1]), 2)

    return {
        "rsi_14": _rsi(close),
        "macd":   _macd(close),
        "moving_averages": {
            "ma_20":  round(float(bb_mid.iloc[-1]),                   2),
            "ma_50":  round(float(close.rolling(50).mean().iloc[-1]), 2),
            "ma_200": ma200,
        },
        "bollinger_bands": {
            "upper":  round(float((bb_mid + 2 * bb_std).iloc[-1]), 2),
            "middle": round(float(bb_mid.iloc[-1]),                 2),
            "lower":  round(float((bb_mid - 2 * bb_std).iloc[-1]), 2),
            "std":    round(float(bb_std.iloc[-1]),                 4),
        },
        "volume_ma_20": int(volume.rolling(20).mean().iloc[-1]),
        "volume_last":  int(volume.iloc[-1]),
    }


def _last_candles(df: pd.DataFrame, n: int = 2) -> list:
    rows = []
    for ts, row in df.tail(n).iterrows():
        rows.append({
            "time":   ts.strftime("%H:%M") if hasattr(ts, "strftime") else str(ts),
            "open":   round(float(row["Open"]),   2),
            "high":   round(float(row["High"]),   2),
            "low":    round(float(row["Low"]),    2),
            "close":  round(float(row["Close"]),  2),
            "volume": int(row["Volume"]),
        })
    return rows


def _atr_from_df(df: pd.DataFrame, period: int = 14) -> float:
    """ATR-14 (Wilder's smoothing) from OHLCV DataFrame."""
    high  = df["High"]
    low   = df["Low"]
    close = df["Close"]
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    val = float(atr.iloc[-1])
    return round(val, 4) if not np.isnan(val) else 0.0


# ── Pure-math signal engine ───────────────────────────────────────────────────

def _score_signal(price: float, indicators: dict) -> tuple:
    """
    Scores the technical setup. Returns (total_score, component_scores).
    Maximum possible score: ~7.0 (all components firing together).
    """
    score      = 0.0
    components: dict = {}

    rsi      = indicators["rsi_14"]
    macd     = indicators["macd"]
    bb       = indicators["bollinger_bands"]
    mas      = indicators["moving_averages"]
    vol_last = indicators.get("volume_last",  0)
    vol_ma   = indicators.get("volume_ma_20", 1)

    # ── RSI ±2.5 ─────────────────────────────────────────────────────────────
    if   rsi < 25: rsi_score = +2.5
    elif rsi < 30: rsi_score = +2.0
    elif rsi < 40: rsi_score = +1.0
    elif rsi > 80: rsi_score = -2.5
    elif rsi > 70: rsi_score = -2.0
    elif rsi > 60: rsi_score = -1.0
    else:          rsi_score =  0.0
    score += rsi_score
    components["rsi"] = rsi_score

    # ── MACD histogram ±1 ────────────────────────────────────────────────────
    hist      = macd["histogram"]
    hist_prev = macd.get("hist_prev", 0.0)
    if   hist > 0 and hist > hist_prev: macd_score = +1.0
    elif hist > 0:                      macd_score = +0.5
    elif hist < 0 and hist < hist_prev: macd_score = -1.0
    elif hist < 0:                      macd_score = -0.5
    else:                               macd_score =  0.0
    score += macd_score
    components["macd"] = macd_score

    # ── Bollinger Bands position ±2 ──────────────────────────────────────────
    bb_range = bb["upper"] - bb["lower"]
    if bb_range > 0:
        bb_pos = (price - bb["lower"]) / bb_range   # 0.0 = lower, 1.0 = upper
        if   bb_pos < 0.10: bb_score = +2.0
        elif bb_pos < 0.25: bb_score = +1.0
        elif bb_pos > 0.90: bb_score = -2.0
        elif bb_pos > 0.75: bb_score = -1.0
        else:               bb_score =  0.0
    else:
        bb_score = 0.0
        bb_pos   = 0.5
    score += bb_score
    components["bollinger"] = bb_score

    # ── MA trend ±1 ──────────────────────────────────────────────────────────
    ma20 = mas["ma_20"]
    ma50 = mas["ma_50"]
    if   price > ma50 and ma20 > ma50: ma_score = +1.0
    elif price > ma50:                 ma_score = +0.5
    elif price < ma50 and ma20 < ma50: ma_score = -1.0
    elif price < ma50:                 ma_score = -0.5
    else:                              ma_score =  0.0
    score += ma_score
    components["ma_trend"] = ma_score

    # ── Volume confirmation ±0.5 ─────────────────────────────────────────────
    vol_ratio = vol_last / vol_ma if vol_ma > 0 else 1.0
    if vol_ratio > 1.5:
        vol_score = +0.5 if score > 0 else -0.5   # confirms current direction
    else:
        vol_score = 0.0
    score += vol_score
    components["volume"] = vol_score

    return round(score, 2), components


# ── Public entry point ────────────────────────────────────────────────────────

def run_signal(symbol: str, timeframe: str = "1h") -> dict:
    """
    Fetches OHLCV data, computes indicators, and applies a deterministic
    scoring system to generate a structured trading signal.

    Returns a dict with signal, confidence, entry/exit prices, R:R ratio,
    and qualitative signal_strength breakdown.
    """
    _fallback = {
        "signal":            "INSUFFICIENT_DATA",
        "confidence":        0,
        "entry_price":       None,
        "stop_loss":         None,
        "take_profit_1":     None,
        "take_profit_2":     None,
        "risk_reward_ratio": None,
        "reasoning":         "",
        "signal_strength":   {
            "trend": "neutral", "momentum": "weak", "support_resistance": "midrange"
        },
    }

    try:
        df = get_hourly(symbol, period="60d")
    except Exception as exc:
        _fallback["reasoning"] = f"Data fetch failed: {exc}"
        return _fallback

    if df is None or len(df) < 26:
        _fallback["reasoning"] = "Insufficient bars for indicator computation."
        return _fallback

    current_price = round(float(df["Close"].iloc[-1]), 2)

    try:
        indicators = _compute_indicators(df)
    except Exception as exc:
        _fallback["reasoning"]   = f"Indicator computation failed: {exc}"
        _fallback["entry_price"] = current_price
        return _fallback

    # ── Score the setup ───────────────────────────────────────────────────────
    score, components = _score_signal(current_price, indicators)

    MAX_SCORE = 7.0   # sum of all component maxima

    if   score >= 3.0:  signal = "BUY"
    elif score <= -3.0: signal = "SELL"
    else:               signal = "HOLD"

    confidence = int(min(95, round(abs(score) / MAX_SCORE * 100)))
    if signal == "HOLD":
        confidence = max(0, confidence)

    # ── ATR-based price levels ────────────────────────────────────────────────
    atr = _atr_from_df(df)

    atr_stop = 1.5
    atr_tp1  = 2.0
    atr_tp2  = 3.5

    if signal == "BUY":
        stop_loss    = round(current_price - atr * atr_stop, 2)
        take_profit1 = round(current_price + atr * atr_tp1,  2)
        take_profit2 = round(current_price + atr * atr_tp2,  2)
    elif signal == "SELL":
        stop_loss    = round(current_price + atr * atr_stop, 2)
        take_profit1 = round(current_price - atr * atr_tp1,  2)
        take_profit2 = round(current_price - atr * atr_tp2,  2)
    else:
        # HOLD — still provide reference levels
        stop_loss    = round(current_price - atr * atr_stop, 2)
        take_profit1 = round(current_price + atr * atr_tp1,  2)
        take_profit2 = round(current_price + atr * atr_tp2,  2)

    risk_per_share   = abs(current_price - stop_loss)
    reward_per_share = abs(take_profit1   - current_price)
    rr_ratio = round(reward_per_share / risk_per_share, 2) if risk_per_share > 0 else 0.0

    # ── Qualitative labels ────────────────────────────────────────────────────
    mas = indicators["moving_averages"]
    rsi = indicators["rsi_14"]
    bb  = indicators["bollinger_bands"]

    if   current_price > mas["ma_50"]: trend_label = "bullish"
    elif current_price < mas["ma_50"]: trend_label = "bearish"
    else:                              trend_label = "neutral"

    abs_score = abs(score)
    if   abs_score >= 4: momentum_label = "strong"
    elif abs_score >= 2: momentum_label = "moderate"
    else:                momentum_label = "weak"

    bb_range = bb["upper"] - bb["lower"]
    if bb_range > 0:
        bb_pos = (current_price - bb["lower"]) / bb_range
        if   bb_pos < 0.25: sr_label = "at support"
        elif bb_pos > 0.75: sr_label = "at resistance"
        else:               sr_label = "midrange"
    else:
        sr_label = "midrange"

    # ── Reasoning string ──────────────────────────────────────────────────────
    parts = [f"Score {score:+.1f} → {signal}"]
    if components["rsi"] != 0:
        parts.append(f"RSI={rsi:.0f}({'+' if components['rsi']>0 else ''}{components['rsi']:.1f})")
    if components["macd"] != 0:
        direction = "rising" if components["macd"] > 0 else "falling"
        parts.append(f"MACD hist {direction}({'+' if components['macd']>0 else ''}{components['macd']:.1f})")
    if components["bollinger"] != 0:
        zone = "lower" if components["bollinger"] > 0 else "upper"
        parts.append(f"BB {zone} zone({'+' if components['bollinger']>0 else ''}{components['bollinger']:.1f})")
    if components["ma_trend"] != 0:
        parts.append(f"MA trend({'+' if components['ma_trend']>0 else ''}{components['ma_trend']:.1f})")
    if components["volume"] != 0:
        parts.append(f"Vol confirmation({'+' if components['volume']>0 else ''}{components['volume']:.1f})")

    return {
        "signal":            signal,
        "confidence":        confidence,
        "entry_price":       current_price,
        "stop_loss":         stop_loss,
        "take_profit_1":     take_profit1,
        "take_profit_2":     take_profit2,
        "risk_reward_ratio": rr_ratio,
        "reasoning":         "; ".join(parts),
        "signal_strength": {
            "trend":              trend_label,
            "momentum":           momentum_label,
            "support_resistance": sr_label,
        },
        "_score_components": components,
        "_score_total":       score,
        "_atr":               atr,
    }
