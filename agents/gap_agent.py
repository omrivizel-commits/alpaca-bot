"""
Gap Agent — Pure Mathematical Gap Analysis Engine.

Analyzes pre-market / overnight gaps by:
  1. Computing historical gap statistics from the last 100 trading days
     (gap-up count, gap-down count, fill rate, avg size, std-dev)
     — all derived locally from yfinance daily OHLCV.
  2. Computing the current ATR-14 (Wilder's smoothed) for gap-vs-ATR sizing.
  3. Fetching the most recent headline via Finnhub (if available).
  4. Applying deterministic rule thresholds to classify and plan the trade.

ATR-ratio classification rules:
  gap_vs_atr < 0.5  → FADE (fill rate historically > 70%)
  0.5 – 1.5         → NEUTRAL (check historical fill rate)
  > 1.5             → TREND_CONTINUATION (< 30% fill on large gaps)
  Strong catalyst keyword → adjust fill probability down, lean TREND

No AI calls required.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timezone


# ── ATR helper ────────────────────────────────────────────────────────────────

def _compute_atr(df: pd.DataFrame, period: int = 14) -> float:
    """Wilder's smoothed ATR-14 from a daily OHLCV DataFrame."""
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


# ── Historical gap statistics ─────────────────────────────────────────────────

def _compute_historical_gaps(df: pd.DataFrame) -> dict:
    """
    Analyses the last 100 (or available) daily bars to compute gap statistics.

    A gap is defined as Open deviating from the previous Close.
    A gap is considered 'filled' on the same day if:
      - Gap-up: the candle's Low trades back down to or below the prev_close.
      - Gap-down: the candle's High trades back up to or above the prev_close.
    """
    if len(df) < 5:
        return _empty_gap_stats()

    records = []
    for i in range(1, len(df)):
        prev_close = float(df["Close"].iloc[i - 1])
        op         = float(df["Open"].iloc[i])
        hi         = float(df["High"].iloc[i])
        lo         = float(df["Low"].iloc[i])

        gap_pct = (op - prev_close) / prev_close * 100 if prev_close else 0.0

        if abs(gap_pct) < 0.05:        # < 5 bps — treat as flat
            continue

        direction = "up" if gap_pct > 0 else "down"
        filled    = (lo <= prev_close) if direction == "up" else (hi >= prev_close)

        records.append({
            "direction": direction,
            "gap_pct":   round(gap_pct, 3),
            "filled":    filled,
        })

    if not records:
        return _empty_gap_stats()

    ups   = [r for r in records if r["direction"] == "up"]
    downs = [r for r in records if r["direction"] == "down"]

    def _stats(group: list) -> dict:
        if not group:
            return {"total": 0, "filled": 0, "fill_rate": 0.5,
                    "avg_size_pct": 0.0, "std_dev_pct": 0.0}
        sizes    = [abs(r["gap_pct"]) for r in group]
        n_filled = sum(1 for r in group if r["filled"])
        return {
            "total":        len(group),
            "filled":       n_filled,
            "fill_rate":    round(n_filled / len(group), 3),
            "avg_size_pct": round(float(np.mean(sizes)), 3),
            "std_dev_pct":  round(float(np.std(sizes)),  3),
        }

    return {
        "gap_up":     _stats(ups),
        "gap_down":   _stats(downs),
        "total_gaps": len(records),
    }


def _empty_gap_stats() -> dict:
    empty = {"total": 0, "filled": 0, "fill_rate": 0.5,
             "avg_size_pct": 0.0, "std_dev_pct": 0.0}
    return {"gap_up": empty, "gap_down": empty, "total_gaps": 0}


# ── Pure-math gap decision engine ─────────────────────────────────────────────

_CATALYST_KEYWORDS = [
    "earnings", "beat", "miss", "guidance", "merger", "acquisition",
    "fda", "approval", "bankruptcy", "lawsuit", "sec", "downgrade", "upgrade",
]


def _has_strong_catalyst(news_text: str) -> bool:
    h = news_text.lower()
    return any(kw in h for kw in _CATALYST_KEYWORDS)


def _make_gap_decision(
    gap_direction: str,
    gap_vs_atr: float,
    gap_class: str,
    hist_fill_rate: float,
    previous_close: float,
    premarket_price: float,
    atr: float,
    after_hours_news: str,
) -> dict:
    """
    Applies deterministic ATR-based rules to produce a complete gap trade plan.
    """
    # ── Base fill probability from gap size ───────────────────────────────────
    if   gap_vs_atr < 0.5:
        fill_prob = min(0.90, hist_fill_rate * 1.15)
        bias      = "FADE"
    elif gap_vs_atr < 1.0:
        fill_prob = hist_fill_rate
        bias      = "NEUTRAL" if hist_fill_rate >= 0.50 else "TREND_CONTINUATION"
    elif gap_vs_atr < 1.5:
        fill_prob = max(0.30, hist_fill_rate * 0.80)
        bias      = "NEUTRAL"
    else:
        fill_prob = max(0.15, hist_fill_rate * 0.50)
        bias      = "TREND_CONTINUATION"

    # ── Adjust for news catalyst ──────────────────────────────────────────────
    has_catalyst = _has_strong_catalyst(after_hours_news)
    if has_catalyst and gap_vs_atr > 0.8:
        fill_prob = max(0.10, fill_prob * 0.65)
        bias      = "TREND_CONTINUATION"
    fill_prob_adj = round(fill_prob, 3)

    # ── Entry strategy ────────────────────────────────────────────────────────
    if bias == "FADE" or (bias == "NEUTRAL" and fill_prob_adj > 0.60):
        # Fade the gap — expect price to return toward prev_close
        if gap_direction == "up":
            entry_zone = f"{premarket_price - atr * 0.2:.2f} - {premarket_price:.2f}"
            stop_loss  = round(premarket_price + atr * 0.50, 2)
            tp1        = round(premarket_price - atr * 1.00, 2)
            tp2        = round(previous_close, 2)
            sl_dist    = abs(stop_loss - premarket_price)
            tp1_dist   = abs(premarket_price - tp1)
        else:
            entry_zone = f"{premarket_price:.2f} - {premarket_price + atr * 0.2:.2f}"
            stop_loss  = round(premarket_price - atr * 0.50, 2)
            tp1        = round(premarket_price + atr * 1.00, 2)
            tp2        = round(previous_close, 2)
            sl_dist    = abs(premarket_price - stop_loss)
            tp1_dist   = abs(tp1 - premarket_price)
        entry_type  = "opening_range_breakout"
        entry_notes = "Wait 15 min after open for gap fade confirmation before entry."
    else:
        # Trend continuation — ride the gap
        if gap_direction == "up":
            entry_zone = f"{premarket_price:.2f} - {premarket_price + atr * 0.3:.2f}"
            stop_loss  = round(previous_close - atr * 0.25, 2)
            tp1        = round(premarket_price + atr * 1.0,  2)
            tp2        = round(premarket_price + atr * 2.0,  2)
            sl_dist    = abs(premarket_price - stop_loss)
            tp1_dist   = abs(tp1 - premarket_price)
        else:
            entry_zone = f"{premarket_price - atr * 0.3:.2f} - {premarket_price:.2f}"
            stop_loss  = round(previous_close + atr * 0.25, 2)
            tp1        = round(premarket_price - atr * 1.0,  2)
            tp2        = round(premarket_price - atr * 2.0,  2)
            sl_dist    = abs(stop_loss - premarket_price)
            tp1_dist   = abs(premarket_price - tp1)
        entry_type  = "breakout_continuation"
        entry_notes = "Enter on first 5-min candle close in gap direction above open."

    rr = round(tp1_dist / sl_dist, 2) if sl_dist > 0 else 1.5

    # ── Fade strategy ─────────────────────────────────────────────────────────
    fade_entry_zone = (
        f"{previous_close - atr * 0.2:.2f} - {previous_close + atr * 0.2:.2f}"
    )
    fade_stop  = round(premarket_price * 1.01, 2) \
                 if gap_direction == "up" else round(premarket_price * 0.99, 2)

    # ── Expected fill time ────────────────────────────────────────────────────
    if   fill_prob_adj > 0.70: fill_time = 30
    elif fill_prob_adj > 0.50: fill_time = 60
    else:                      fill_time = None

    # ── Risk warning ──────────────────────────────────────────────────────────
    if   gap_vs_atr > 2.0:
        risk_warning = (f"Extreme gap ({gap_vs_atr:.1f}×ATR) — "
                        "low fill probability, high volatility expected.")
    elif has_catalyst:
        risk_warning = "News catalyst detected — elevated volatility, widen stops."
    else:
        risk_warning = ""

    # ── Gap type / class ──────────────────────────────────────────────────────
    gap_type = (
        "gap_up"   if gap_direction == "up"   else
        "gap_down" if gap_direction == "down" else
        "flat"
    )

    return {
        "gap_type":                               gap_type,
        "gap_vs_atr":                             gap_vs_atr,
        "gap_classification":                     gap_class,
        "historical_fill_probability":            round(hist_fill_rate, 3),
        "fill_probability_adjusted_for_catalyst": fill_prob_adj,
        "expected_fill_time_minutes":             fill_time,
        "trading_bias":                           bias,
        "entry_strategy": {
            "type":          entry_type,
            "entry_zone":    entry_zone,
            "stop_loss":     stop_loss,
            "take_profit_1": tp1,
            "take_profit_2": tp2,
            "risk_reward":   rr,
            "notes":         entry_notes,
        },
        "fade_strategy": {
            "probability": fill_prob_adj,
            "only_if":     "Price fails to hold gap level in first 15 minutes.",
            "entry_zone":  fade_entry_zone,
            "stop_loss":   fade_stop,
            "take_profit": round(previous_close, 2),
            "notes":       "Fade only if initial gap hold fails within 15 min.",
        },
        "key_levels": {
            "previous_close":   previous_close,
            "gap_fill_target":  previous_close,
            "resistance_above": round(premarket_price + atr, 2)
                                if gap_direction == "up"   else None,
            "support_below":    round(previous_close - atr * 0.5, 2)
                                if gap_direction == "down" else None,
        },
        "risk_warning": risk_warning,
    }


# ── Public entry point ────────────────────────────────────────────────────────

def run_gap_analysis(symbol: str, premarket_price: float) -> dict:
    """
    Fetches historical OHLCV, computes gap statistics and ATR, optionally
    pulls the latest headline, then applies deterministic rules for a
    full gap trade plan.

    Parameters
    ----------
    symbol          : Ticker (e.g. 'AAPL')
    premarket_price : The current pre-market / after-hours price

    Returns
    -------
    Full gap analysis dict plus _gap_meta with raw computed values.
    """
    _fallback = {
        "gap_type":                               "flat",
        "gap_vs_atr":                             0.0,
        "gap_classification":                     "small",
        "historical_fill_probability":            0.5,
        "fill_probability_adjusted_for_catalyst": 0.5,
        "expected_fill_time_minutes":             None,
        "trading_bias":                           "NEUTRAL",
        "entry_strategy": {
            "type": "opening_range_breakout", "entry_zone": "N/A",
            "stop_loss": None, "take_profit_1": None, "take_profit_2": None,
            "risk_reward": None, "notes": "Gap analysis unavailable.",
        },
        "fade_strategy": {
            "probability": 0.5, "only_if": "N/A", "entry_zone": "N/A",
            "stop_loss": None, "take_profit": None, "notes": "",
        },
        "key_levels": {
            "previous_close": None, "gap_fill_target": None,
            "resistance_above": None, "support_below": None,
        },
        "risk_warning": "Gap scan failed — check market data connection.",
    }

    # ── Step 1: Fetch daily data (100 days) ───────────────────────────────────
    try:
        df = yf.Ticker(symbol).history(period="100d", interval="1d", auto_adjust=True)
        if df is None or len(df) < 5:
            _fallback["risk_warning"] = f"Insufficient data for {symbol}"
            return _fallback
    except Exception as exc:
        _fallback["risk_warning"] = f"Data fetch failed: {exc}"
        return _fallback

    # ── Step 2: Basic gap metrics ─────────────────────────────────────────────
    previous_close = round(float(df["Close"].iloc[-1]), 2)
    gap_pct        = round((premarket_price - previous_close) / previous_close * 100, 3) \
                     if previous_close else 0.0
    gap_direction  = "up" if gap_pct > 0 else ("down" if gap_pct < 0 else "flat")

    # ── Step 3: ATR and gap-vs-ATR ratio ──────────────────────────────────────
    atr         = _compute_atr(df)
    gap_dollars = abs(premarket_price - previous_close)
    gap_vs_atr  = round(gap_dollars / atr, 3) if atr > 0 else 0.0

    if   gap_vs_atr < 0.5: gap_class = "small"
    elif gap_vs_atr < 1.0: gap_class = "medium"
    elif gap_vs_atr < 2.0: gap_class = "large"
    else:                  gap_class = "extreme"

    # ── Step 4: Historical gap statistics ─────────────────────────────────────
    gap_stats    = _compute_historical_gaps(df)
    direction_key = "gap_up" if gap_direction == "up" else "gap_down"
    hist_stats   = gap_stats.get(direction_key, {})
    hist_fill_rate = hist_stats.get("fill_rate", 0.5)

    # ── Step 5: Latest headline (non-blocking) ────────────────────────────────
    after_hours_news = "No recent news available."
    try:
        from data.news_fetcher import fetch_latest_news_item
        item = fetch_latest_news_item(symbol)
        if item:
            after_hours_news = f"{item['headline']} ({item['source']})"
    except Exception:
        pass

    # ── Step 6: Pure-math decision ────────────────────────────────────────────
    result = _make_gap_decision(
        gap_direction, gap_vs_atr, gap_class,
        hist_fill_rate, previous_close, round(premarket_price, 2),
        atr, after_hours_news,
    )

    # Attach raw meta for dashboard
    result["_gap_meta"] = {
        "previous_close":     previous_close,
        "premarket_price":    round(premarket_price, 2),
        "gap_pct":            gap_pct,
        "gap_dollars":        round(gap_dollars, 2),
        "atr":                atr,
        "gap_vs_atr":         gap_vs_atr,
        "gap_classification": gap_class,
        "historical_stats":   hist_stats,
        "news_used":          after_hours_news,
    }

    return result
