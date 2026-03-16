"""
Vision Agent — Pure Mathematical Chart Pattern Detector.

Replaces AI vision (Gemini + PIL image rendering) with deterministic
OHLCV analysis:
  - Trend detection from MA alignment (MA20 / MA50)
  - Support / resistance from recent swing highs and lows
  - Bollinger Band position for pattern classification
  - Candlestick pattern detection (Doji, Hammer, Shooting Star, Engulfing)
  - Volume confirmation
  - Vision veto logic against the quant signal

No API calls, no image rendering, no Pillow required.
"""

import numpy as np
import pandas as pd
import yfinance as yf


# ── Candlestick pattern detector ─────────────────────────────────────────────

def _detect_candlestick_patterns(df: pd.DataFrame) -> list:
    """Detects common 1- and 2-bar candlestick patterns."""
    patterns = []
    if len(df) < 2:
        return patterns

    c0 = df.iloc[-2]   # previous candle
    c1 = df.iloc[-1]   # current candle

    o1, h1, l1, c1v = float(c1["Open"]), float(c1["High"]), float(c1["Low"]), float(c1["Close"])
    o0, h0, l0, c0v = float(c0["Open"]), float(c0["High"]), float(c0["Low"]), float(c0["Close"])

    body1  = abs(c1v  - o1)
    range1 = h1 - l1

    # ── Doji: body < 10 % of range ───────────────────────────────────────────
    if range1 > 0 and body1 / range1 < 0.10:
        patterns.append("Doji")

    # ── Hammer: long lower shadow (≥ 2× body), small upper shadow ────────────
    lower1 = min(o1, c1v) - l1
    upper1 = h1 - max(o1, c1v)
    if range1 > 0 and body1 / range1 < 0.30 and lower1 > body1 * 2 and upper1 < body1:
        patterns.append("Hammer")

    # ── Shooting Star: long upper shadow, small lower shadow ─────────────────
    if range1 > 0 and body1 / range1 < 0.30 and upper1 > body1 * 2 and lower1 < body1:
        patterns.append("Shooting Star")

    # ── Bullish Engulfing: green body fully engulfs previous red body ─────────
    if c1v > o1 and c0v < o0:           # current green, prev red
        if o1 <= c0v and c1v >= o0:     # body engulfs
            patterns.append("Bullish Engulfing")

    # ── Bearish Engulfing: red body fully engulfs previous green body ─────────
    if c1v < o1 and c0v > o0:           # current red, prev green
        if o1 >= c0v and c1v <= o0:     # body engulfs
            patterns.append("Bearish Engulfing")

    return patterns


# ── Support / resistance finder ───────────────────────────────────────────────

def _find_support_resistance(df: pd.DataFrame, lookback: int = 20) -> tuple:
    """Finds nearest support and resistance from recent swing highs/lows."""
    if len(df) < 3:
        last = float(df["Close"].iloc[-1])
        return round(last * 0.98, 2), round(last * 1.02, 2)

    recent = df.tail(lookback)
    price  = float(df["Close"].iloc[-1])

    highs = recent["High"].values
    lows  = recent["Low"].values

    support_levels = sorted([l for l in lows  if l < price], reverse=True)
    resist_levels  = sorted([h for h in highs if h > price])

    support    = round(float(support_levels[0]), 2) if support_levels \
                 else round(price * 0.97, 2)
    resistance = round(float(resist_levels[0]),  2) if resist_levels  \
                 else round(price * 1.03, 2)

    return support, resistance


# ── Public entry point ────────────────────────────────────────────────────────

def analyze_chart(symbol: str, quant_signal: str = "BUY", price: float = 0.0) -> dict:
    """
    Pure OHLCV-based chart analysis.
    Fetches 30 days of daily data, detects patterns, computes trend and levels.

    Returns a dict compatible with the pipeline StateObject.
    """
    _fallback = {
        "pattern":              "INSUFFICIENT_DATA",
        "nearest_support":      "unknown",
        "nearest_resistance":   "unknown",
        "resistance_nearby":    False,
        "vision_veto":          False,
        "vision_confidence":    "LOW",
        "vision_reasoning":     "Insufficient data for analysis.",
        "candlestick_patterns": [],
        "trend":                "SIDEWAYS",
        "volume_confirmation":  "NEUTRAL",
    }

    try:
        df = yf.Ticker(symbol).history(period="30d", interval="1d", auto_adjust=True)
        if df is None or len(df) < 5:
            return _fallback
    except Exception as exc:
        _fallback["vision_reasoning"] = f"Data fetch failed: {exc}"
        return _fallback

    close  = df["Close"]
    volume = df["Volume"]
    current = price if price > 0 else float(close.iloc[-1])

    # ── Moving averages ───────────────────────────────────────────────────────
    ma20_val = close.rolling(20).mean().iloc[-1]
    ma50_val = close.rolling(50).mean().iloc[-1]
    ma20 = float(ma20_val) if len(df) >= 20 and not np.isnan(ma20_val) \
           else float(close.mean())
    ma50 = float(ma50_val) if len(df) >= 50 and not np.isnan(ma50_val) \
           else ma20

    # ── Trend from MA alignment ───────────────────────────────────────────────
    if   current > ma20 and ma20 > ma50: trend = "UPTREND"
    elif current < ma20 and ma20 < ma50: trend = "DOWNTREND"
    else:                                trend = "SIDEWAYS"

    # ── Support / resistance ──────────────────────────────────────────────────
    support, resistance = _find_support_resistance(df, lookback=min(20, len(df)))

    # Within 1.5 % above current price → resistance is nearby
    resistance_nearby = bool(0 < resistance - current < current * 0.015)

    # ── Bollinger Band position ───────────────────────────────────────────────
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_upper = float((bb_mid + 2 * bb_std).iloc[-1])
    bb_lower = float((bb_mid - 2 * bb_std).iloc[-1])
    bb_range = bb_upper - bb_lower
    bb_pos   = (current - bb_lower) / bb_range if bb_range > 0 else 0.5

    # ── Chart pattern classification ──────────────────────────────────────────
    if   trend == "UPTREND"   and bb_pos < 0.30: pattern = "BOUNCING_OFF_SUPPORT"
    elif trend == "UPTREND":                      pattern = "UPTREND"
    elif trend == "DOWNTREND" and bb_pos > 0.70: pattern = "APPROACHING_RESISTANCE"
    elif trend == "DOWNTREND":                    pattern = "DOWNTREND"
    elif bb_pos < 0.15:                           pattern = "BOUNCING_OFF_SUPPORT"
    elif bb_pos > 0.85:                           pattern = "APPROACHING_RESISTANCE"
    elif bb_pos > 0.55:                           pattern = "MID_RANGE_BULLISH_BIAS"
    else:                                         pattern = "MID_RANGE_BEARISH_BIAS"

    # ── Candlestick patterns ──────────────────────────────────────────────────
    candle_patterns = _detect_candlestick_patterns(df)

    detected = set(candle_patterns)
    bullish_reversals = {"Hammer", "Bullish Engulfing"}
    bearish_reversals = {"Shooting Star", "Bearish Engulfing"}

    if detected & bullish_reversals and trend == "DOWNTREND":
        pattern = "REVERSAL_BULLISH"
    elif detected & bearish_reversals and trend == "UPTREND":
        pattern = "REVERSAL_BEARISH"

    # ── Volume confirmation ───────────────────────────────────────────────────
    vol_last = float(volume.iloc[-1])
    vol_ma   = float(volume.rolling(20).mean().iloc[-1]) \
               if len(volume) >= 20 else vol_last
    vol_ratio = vol_last / vol_ma if vol_ma > 0 else 1.0

    if   vol_ratio > 1.30: volume_confirmation = "CONFIRMING"
    elif vol_ratio < 0.70: volume_confirmation = "DIVERGING"
    else:                  volume_confirmation = "NEUTRAL"

    # ── Vision veto logic ─────────────────────────────────────────────────────
    vision_veto  = False
    veto_reasons = []
    qs           = quant_signal.upper()

    if qs == "BUY":
        if pattern in ("APPROACHING_RESISTANCE", "REVERSAL_BEARISH"):
            vision_veto = True
            veto_reasons.append(f"Chart shows {pattern}")
        if resistance_nearby:
            vision_veto = True
            veto_reasons.append(f"Resistance at {resistance:.2f} within 1.5%")

    if qs == "SELL":
        if pattern in ("BOUNCING_OFF_SUPPORT", "REVERSAL_BULLISH"):
            vision_veto = True
            veto_reasons.append(f"Chart shows {pattern}")

    # ── Confidence from signal agreement ──────────────────────────────────────
    confirmations = sum([
        bool(trend == "UPTREND"   and qs == "BUY"),
        bool(trend == "DOWNTREND" and qs == "SELL"),
        bool(volume_confirmation == "CONFIRMING"),
        bool(pattern in ("BOUNCING_OFF_SUPPORT", "UPTREND", "REVERSAL_BULLISH")
             and qs == "BUY"),
        bool(pattern in ("APPROACHING_RESISTANCE", "DOWNTREND", "REVERSAL_BEARISH")
             and qs == "SELL"),
    ])

    if vision_veto:
        confidence = "HIGH"   # confident in the veto
    elif confirmations >= 4: confidence = "HIGH"
    elif confirmations >= 2: confidence = "MEDIUM"
    else:                    confidence = "LOW"

    # ── Reasoning ─────────────────────────────────────────────────────────────
    reasoning_parts = [
        f"Trend: {trend} (price={current:.2f}, MA20={ma20:.2f}, MA50={ma50:.2f})",
        f"BB position: {bb_pos:.0%} of range",
        f"S/R: support={support:.2f}, resistance={resistance:.2f}",
    ]
    if candle_patterns:
        reasoning_parts.append(f"Candlestick patterns: {', '.join(candle_patterns)}")
    if veto_reasons:
        reasoning_parts.append(f"Veto: {'; '.join(veto_reasons)}")

    return {
        "pattern":              pattern,
        "nearest_support":      str(support),
        "nearest_resistance":   str(resistance),
        "resistance_nearby":    resistance_nearby,
        "vision_veto":          vision_veto,
        "vision_confidence":    confidence,
        "vision_reasoning":     ". ".join(reasoning_parts),
        "candlestick_patterns": candle_patterns,
        "trend":                trend,
        "volume_confirmation":  volume_confirmation,
    }
