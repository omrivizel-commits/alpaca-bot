"""
Position Agent — Pure Mathematical ATR-Adjusted Position Sizing Engine.

Given an entry price and market structure, computes:
  - ATR-14 (volatility-adjusted stop distance)
  - Swing highs/lows and nearest S/R levels from scipy local extrema
  - Exact stop-loss (1.5× ATR below entry) and take-profit targets
  - Position size (shares) based on dollar risk limit
  - Tiered exit plan with partial exits
  - Risk/reward ratio with LOW_RR / RR_VERY_LOW flags

No AI calls required — all outputs are computed deterministically.
"""

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema

from config import settings
from data.market_data import get_hourly
from execution.alpaca_broker import get_portfolio_value


# ── Indicator helpers ─────────────────────────────────────────────────────────


def _compute_atr(df: pd.DataFrame, period: int = 14) -> float:
    """Wilder's ATR."""
    high  = df["High"]
    low   = df["Low"]
    close = df["Close"]

    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)

    atr = tr.ewm(span=period, adjust=False).mean()
    return round(float(atr.iloc[-1]), 4)


def _compute_market_structure(df: pd.DataFrame, current_price: float) -> dict:
    """
    Returns swing_high, swing_low, and nearest two S/R levels above and below
    the current price using scipy local-extrema detection.
    """
    closes = df["Close"].values
    order  = 5   # bars on each side to qualify as swing

    res_idx = argrelextrema(closes, np.greater, order=order)[0]
    sup_idx = argrelextrema(closes, np.less,    order=order)[0]

    res_levels = sorted(closes[res_idx].tolist(), reverse=True) if len(res_idx) else []
    sup_levels = sorted(closes[sup_idx].tolist(), reverse=True) if len(sup_idx) else []

    swing_high = float(df["High"].tail(20).max())
    swing_low  = float(df["Low"].tail(20).min())

    # Resistances above current price
    above = sorted([r for r in res_levels if r > current_price])
    res1  = round(above[0], 2) if len(above) > 0 else round(current_price * 1.015, 2)
    res2  = round(above[1], 2) if len(above) > 1 else round(current_price * 1.030, 2)

    # Supports below current price
    below = sorted([s for s in sup_levels if s < current_price], reverse=True)
    sup1  = round(below[0], 2) if len(below) > 0 else round(current_price * 0.985, 2)
    sup2  = round(below[1], 2) if len(below) > 1 else round(current_price * 0.970, 2)

    return {
        "swing_high":         round(swing_high, 2),
        "swing_low":          round(swing_low,  2),
        "support_level_1":    sup1,
        "support_level_2":    sup2,
        "resistance_level_1": res1,
        "resistance_level_2": res2,
    }


def _infer_entry_type(state: dict) -> str:
    """Infers a human-readable entry type from signal agent context."""
    sr  = state.get("signal_sr_position", "midrange")
    trend = state.get("trend", "SIDEWAYS")
    sig = state.get("signal", "HOLD")

    if sr == "at support" and sig == "BUY":
        return "bounce_from_support"
    if sr == "at resistance" and sig == "SELL":
        return "rejection_at_resistance"
    if trend in ("UPTREND",) and sig == "BUY":
        return "breakout_continuation"
    if trend in ("DOWNTREND",) and sig == "SELL":
        return "breakdown_continuation"
    return "technical_signal"


# ── Public entry point ────────────────────────────────────────────────────────

def run_position_sizing(
    symbol: str,
    entry_price: float,
    state: dict | None = None,
    risk_per_trade_usd: float | None = None,
    account_size: float | None = None,
) -> dict:
    """
    Computes ATR, market structure, and calls Gemini for a complete trade plan.

    Parameters
    ----------
    symbol             : Ticker
    entry_price        : Confirmed entry price
    state              : Full pipeline StateObject (for entry-type inference)
    risk_per_trade_usd : Override dollar risk (default: portfolio * RISK_PCT)
    account_size       : Override account size (default: fetched from Alpaca)

    Returns a dict with position size, stop, targets, exit plan, and R:R.
    """
    # ── Fallback result (simple % sizing) ────────────────────────────────────
    simple_qty = max(1, int((100.0) / max(entry_price * settings.STOP_LOSS_PCT, 0.01)))
    _fallback = {
        "entry_price":              entry_price,
        "entry_type":               "technical_signal",
        "position_size_shares":     simple_qty,
        "stop_loss_price":          round(entry_price * (1 - settings.STOP_LOSS_PCT), 2),
        "stop_loss_atr_multiple":   1.0,
        "target_1_price":           round(entry_price * (1 + settings.STOP_LOSS_PCT * 2), 2),
        "target_1_profit_per_share": round(entry_price * settings.STOP_LOSS_PCT * 2, 2),
        "target_2_price":           round(entry_price * (1 + settings.STOP_LOSS_PCT * 3), 2),
        "target_2_profit_per_share": round(entry_price * settings.STOP_LOSS_PCT * 3, 2),
        "risk_usd":                 round(simple_qty * entry_price * settings.STOP_LOSS_PCT, 2),
        "profit_potential_1":       None,
        "profit_potential_2":       None,
        "risk_reward_ratio_target1": 2.0,
        "risk_reward_ratio_target2": 3.0,
        "exit_plan":                [],
        "confidence_adjustment":    "OK",
        "notes":                    "Fallback: simple % sizing used.",
    }

    try:
        df = get_hourly(symbol, period="60d")
    except Exception as exc:
        _fallback["notes"] = f"Data fetch failed: {exc}"
        return _fallback

    # ── Compute values ────────────────────────────────────────────────────────
    try:
        atr = _compute_atr(df)
    except Exception:
        atr = round(entry_price * 0.01, 4)

    try:
        market_structure = _compute_market_structure(df, entry_price)
    except Exception:
        market_structure = {
            "swing_high":         round(entry_price * 1.02, 2),
            "swing_low":          round(entry_price * 0.98, 2),
            "support_level_1":    round(entry_price * 0.985, 2),
            "support_level_2":    round(entry_price * 0.970, 2),
            "resistance_level_1": round(entry_price * 1.015, 2),
            "resistance_level_2": round(entry_price * 1.030, 2),
        }

    try:
        portfolio_val = get_portfolio_value()
    except Exception:
        portfolio_val = account_size or 25000.0

    account_size      = account_size or portfolio_val
    risk_per_trade    = risk_per_trade_usd or round(portfolio_val * settings.RISK_PCT, 2)
    max_position_usd  = account_size * 0.05   # hard 5% cap
    entry_type        = _infer_entry_type(state) if state else "technical_signal"

    # ── Determine signal direction from state ─────────────────────────────────
    signal = str((state or {}).get("signal", "BUY")).upper()
    is_short = signal == "SELL"

    # ── ATR-based stop and targets ────────────────────────────────────────────
    atr_stop_mult = 1.5
    atr_tp1_mult  = 2.0
    atr_tp2_mult  = 3.5

    if is_short:
        stop_loss_price = round(entry_price + atr * atr_stop_mult, 2)
        target_1_price  = round(entry_price - atr * atr_tp1_mult,  2)
        target_2_price  = round(entry_price - atr * atr_tp2_mult,  2)
    else:
        stop_loss_price = round(entry_price - atr * atr_stop_mult, 2)
        target_1_price  = round(entry_price + atr * atr_tp1_mult,  2)
        target_2_price  = round(entry_price + atr * atr_tp2_mult,  2)

    # Prefer S/R levels over pure ATR if they're within 2× ATR
    ms   = market_structure
    sup1 = ms.get("support_level_1",    stop_loss_price)
    res1 = ms.get("resistance_level_1", target_1_price)

    if not is_short and sup1 > entry_price * 0.97:   # close support below
        stop_loss_price = max(round(sup1 - atr * 0.25, 2), stop_loss_price)
    if not is_short and res1 < entry_price + atr * 3: # close resistance above
        target_1_price  = min(round(res1 - atr * 0.10, 2), target_1_price)

    risk_per_share   = round(abs(entry_price - stop_loss_price), 4)
    reward_per_share = round(abs(target_1_price - entry_price),  4)
    rr1 = round(reward_per_share / risk_per_share, 2) if risk_per_share > 0 else 0.0
    rr2 = round(abs(target_2_price - entry_price)  / risk_per_share, 2) \
          if risk_per_share > 0 else 0.0

    # ── Position size from risk budget ─────────────────────────────────────────
    qty = max(1, min(
        int(max_position_usd / entry_price),               # 5% account cap
        int(risk_per_trade / risk_per_share) if risk_per_share > 0 else 1,
    ))

    profit1 = round(qty       * reward_per_share, 2)
    profit2 = round(qty // 2  * abs(target_2_price - entry_price), 2)

    # ── Confidence adjustment ─────────────────────────────────────────────────
    if   rr1 < 1.0: ca = "RR_VERY_LOW"
    elif rr1 < 1.5: ca = "RR_LOW"
    else:           ca = "OK"

    # ── Tiered exit plan ──────────────────────────────────────────────────────
    exit_plan = [
        {"shares": qty // 2, "price": target_1_price,
         "profit": round(qty // 2 * reward_per_share, 2)},
        {"shares": qty - qty // 2, "price": target_2_price,
         "profit": round((qty - qty // 2) * abs(target_2_price - entry_price), 2)},
    ]

    notes = (
        f"ATR={atr:.4f}, stop {atr_stop_mult}×ATR, "
        f"TP1 {atr_tp1_mult}×ATR, TP2 {atr_tp2_mult}×ATR. "
        f"R:R={rr1:.1f}:1 (target1), {rr2:.1f}:1 (target2)."
    )
    if ca != "OK":
        notes += f" WARNING: {ca}."

    return {
        "entry_price":               entry_price,
        "entry_type":                entry_type,
        "position_size_shares":      qty,
        "stop_loss_price":           stop_loss_price,
        "stop_loss_atr_multiple":    atr_stop_mult,
        "target_1_price":            target_1_price,
        "target_1_profit_per_share": reward_per_share,
        "target_2_price":            target_2_price,
        "target_2_profit_per_share": round(abs(target_2_price - entry_price), 4),
        "risk_usd":                  round(qty * risk_per_share, 2),
        "profit_potential_1":        profit1,
        "profit_potential_2":        profit2,
        "risk_reward_ratio_target1": rr1,
        "risk_reward_ratio_target2": rr2,
        "exit_plan":                 exit_plan,
        "confidence_adjustment":     ca,
        "notes":                     notes,
        "_market_structure":         market_structure,
    }
