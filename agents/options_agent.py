"""
Options Agent — Pure Mathematical Options Market Sentiment Analyzer.

Fetches live options chain data via yfinance and computes:
  - Put/call ratio (volume-based, ~7-day and ~30-day expiries)
  - ATM implied volatility + OTM skew (put IV vs call IV)
  - IV Rank / IV Percentile (approximated from 1-year rolling realized HV)
  - IV implied daily move vs ATR-14
  - Max open-interest strikes → put support / call resistance

All interpretation is done with deterministic rule thresholds — no AI calls.

PCR thresholds:
  PCR > 1.5  → elevated_bearish    PCR > 1.1 → moderate_bearish
  PCR < 0.55 → elevated_bullish    PCR < 0.75 → moderate_bullish
  else       → neutral

IV rank thresholds:
  > 0.70 → elevated (favors selling premium)
  < 0.30 → depressed (favors buying premium)

Note: yfinance provides live per-contract implied volatility directly from
      the options chain. IV Rank / Percentile are approximated from
      252-day rolling realized volatility (HV), which closely tracks
      implied vol rank without a paid data feed.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timezone, timedelta


# ── Data helpers ──────────────────────────────────────────────────────────────

def _nearest_expiry(expiries: list, target_days: int) -> str:
    """Returns the expiry string closest to `target_days` from today."""
    today  = datetime.now(timezone.utc).date()
    target = today + timedelta(days=target_days)
    return min(
        expiries,
        key=lambda e: abs((datetime.strptime(e, "%Y-%m-%d").date() - target).days),
    )


def _safe_iv(df: pd.DataFrame, target_strike: float) -> float:
    """Returns IV (%) at the strike nearest to target_strike with valid IV."""
    valid = df[pd.to_numeric(df["impliedVolatility"], errors="coerce").fillna(0) > 0.001].copy()
    valid = valid.reset_index(drop=True)
    if valid.empty:
        return None
    idx = (valid["strike"] - target_strike).abs().idxmin()
    return round(float(valid.loc[idx, "impliedVolatility"]) * 100, 2)


def _compute_atr(df: pd.DataFrame, period: int = 14) -> float:
    """Wilder's smoothed ATR-14 from daily OHLCV."""
    high, low, close = df["High"], df["Low"], df["Close"]
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    val = float(atr.iloc[-1])
    return round(val, 4) if not np.isnan(val) else 0.0


def _compute_hv_stats(df: pd.DataFrame) -> tuple:
    """
    Approximates IV Rank and IV Percentile from 252-day rolling realized vol.
    Returns: (current_hv_pct, hv_30d_avg_pct, hv_rank_0_1, hv_percentile_0_1)
    """
    log_ret   = np.log(df["Close"] / df["Close"].shift(1))
    hv_series = log_ret.rolling(20).std() * np.sqrt(252) * 100
    hv_clean  = hv_series.dropna()

    if len(hv_clean) < 20:
        return 20.0, 20.0, 0.5, 0.5

    cur = round(float(hv_clean.iloc[-1]),        2)
    avg = round(float(hv_clean.tail(30).mean()),  2)
    mn  = float(hv_clean.min())
    mx  = float(hv_clean.max())

    rank = round((cur - mn) / (mx - mn), 3) if mx > mn else 0.5
    pct  = round(float((hv_clean < cur).mean()),  3)

    return cur, avg, rank, pct


def _fetch_options_snapshot(symbol: str, current_price: float) -> dict:
    """Pulls live options chain and computes PCR, IV, and skew metrics."""
    ticker   = yf.Ticker(symbol)
    expiries = list(ticker.options)
    if not expiries:
        return None

    exp_30d = _nearest_expiry(expiries, 30)
    exp_7d  = _nearest_expiry(expiries, 7)

    chain_30d = ticker.option_chain(exp_30d)
    calls_30d = chain_30d.calls.copy().reset_index(drop=True)
    puts_30d  = chain_30d.puts.copy().reset_index(drop=True)

    for col in ("volume", "openInterest", "impliedVolatility"):
        for frame in (calls_30d, puts_30d):
            frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0)

    if exp_7d != exp_30d:
        chain_7d = ticker.option_chain(exp_7d)
        calls_7d = chain_7d.calls.copy().reset_index(drop=True)
        puts_7d  = chain_7d.puts.copy().reset_index(drop=True)
        for col in ("volume", "openInterest"):
            for frame in (calls_7d, puts_7d):
                frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0)
    else:
        calls_7d, puts_7d = calls_30d, puts_30d

    def _pcr(puts: pd.DataFrame, calls: pd.DataFrame) -> float:
        pv = float(puts["volume"].sum())
        cv = float(calls["volume"].sum())
        return round(pv / cv, 3) if cv > 0 else 1.0

    pcr_30d = _pcr(puts_30d, calls_30d)
    pcr_7d  = _pcr(puts_7d,  calls_7d)

    atm_iv = _safe_iv(calls_30d, current_price) or 20.0

    otm_put_iv  = _safe_iv(puts_30d,  current_price * 0.95) or atm_iv
    otm_call_iv = _safe_iv(calls_30d, current_price * 1.05) or atm_iv

    put_spread  = otm_put_iv  - atm_iv
    call_spread = atm_iv      - otm_call_iv
    if   put_spread  > 3.0: skew_slope = "put_heavy"
    elif call_spread > 3.0: skew_slope = "call_heavy"
    else:                   skew_slope = "symmetric"

    puts_oi  = puts_30d[puts_30d["openInterest"]  > 0]
    calls_oi = calls_30d[calls_30d["openInterest"] > 0]

    put_support     = round(float(puts_oi.loc[puts_oi["openInterest"].idxmax(),
                                              "strike"]), 2) \
                      if not puts_oi.empty  else round(current_price * 0.97, 2)
    call_resistance = round(float(calls_oi.loc[calls_oi["openInterest"].idxmax(),
                                               "strike"]), 2) \
                      if not calls_oi.empty else round(current_price * 1.03, 2)

    iv_implied_move_pct = round(atm_iv / 100 / np.sqrt(252) * 100, 3)

    return {
        "put_call_ratio_30d":  pcr_30d,
        "put_call_ratio_7d":   pcr_7d,
        "atm_iv":              atm_iv,
        "otm_put_iv":          otm_put_iv,
        "otm_call_iv":         otm_call_iv,
        "skew_slope":          skew_slope,
        "put_support":         put_support,
        "call_resistance":     call_resistance,
        "iv_implied_move_pct": iv_implied_move_pct,
        "expiry_30d":          exp_30d,
        "expiry_7d":           exp_7d,
    }


# ── Pure-math interpretation ──────────────────────────────────────────────────

def _interpret_options(
    pcr_30d: float,
    pcr_7d: float,
    hv_rank: float,
    atm_iv: float,
    otm_put_iv: float,
    otm_call_iv: float,
    skew_slope: str,
    put_support: float,
    call_resistance: float,
    iv_implied_move: float,
    atr_move_pct: float,
    current_price: float,
) -> dict:
    """
    Deterministic interpretation of options metrics using rule thresholds.
    No AI calls required.
    """
    # ── Put/call ratio signal ─────────────────────────────────────────────────
    pcr_avg = (pcr_30d + pcr_7d) / 2
    if   pcr_avg > 1.50: pcr_signal = "elevated_bearish"
    elif pcr_avg > 1.10: pcr_signal = "moderate_bearish"
    elif pcr_avg < 0.55: pcr_signal = "elevated_bullish"
    elif pcr_avg < 0.75: pcr_signal = "moderate_bullish"
    else:                pcr_signal = "neutral"

    # ── Combined sentiment score ──────────────────────────────────────────────
    scores = []
    pcr_map = {
        "elevated_bearish":  -2, "moderate_bearish": -1,
        "elevated_bullish":  +2, "moderate_bullish": +1, "neutral": 0,
    }
    skew_map = {"put_heavy": -1, "call_heavy": +1, "symmetric": 0}
    scores.append(pcr_map.get(pcr_signal, 0))
    scores.append(skew_map.get(skew_slope, 0))

    sentiment_score = sum(scores)
    if   sentiment_score >=  2: options_sentiment = "bullish"
    elif sentiment_score <= -2: options_sentiment = "bearish"
    elif sentiment_score == -1: options_sentiment = "defensive"
    else:                       options_sentiment = "neutral"

    # ── IV assessment ─────────────────────────────────────────────────────────
    if   hv_rank > 0.80: iv_interp = f"IV at {hv_rank*100:.0f}% rank — elevated, premium selling conditions"
    elif hv_rank > 0.50: iv_interp = f"IV at {hv_rank*100:.0f}% rank — above average, slight premium selling edge"
    elif hv_rank > 0.20: iv_interp = f"IV at {hv_rank*100:.0f}% rank — below average, favor premium buying"
    else:                iv_interp = f"IV at {hv_rank*100:.0f}% rank — very low, premium buying environment"

    # ── Move mismatch ─────────────────────────────────────────────────────────
    if   iv_implied_move > atr_move_pct * 1.30:
        mismatch = (f"Options pricing ({iv_implied_move:.2f}%) exceeds ATR move "
                    f"({atr_move_pct:.2f}%) — IV elevated vs historical")
    elif iv_implied_move < atr_move_pct * 0.70:
        mismatch = (f"Options pricing ({iv_implied_move:.2f}%) below ATR move "
                    f"({atr_move_pct:.2f}%) — IV may spike")
    else:
        mismatch = (f"Options pricing ({iv_implied_move:.2f}%) aligned with "
                    f"ATR move ({atr_move_pct:.2f}%)")

    # ── Skew interpretation ───────────────────────────────────────────────────
    put_spread  = otm_put_iv - atm_iv
    call_spread = atm_iv     - otm_call_iv
    if skew_slope == "put_heavy":
        skew_interp = (f"Put IV {put_spread:.1f}% above ATM — "
                       "heavy put protection demand, bearish hedging activity")
    elif skew_slope == "call_heavy":
        skew_interp = (f"Call IV {call_spread:.1f}% premium — "
                       "elevated call demand, bullish speculation")
    else:
        skew_interp = "Symmetric skew — balanced options market, no directional bias"

    # ── Sentiment probabilities ───────────────────────────────────────────────
    if   sentiment_score >=  2: prob_up, prob_dn, prob_mut = 0.50, 0.20, 0.30
    elif sentiment_score <= -2: prob_up, prob_dn, prob_mut = 0.20, 0.50, 0.30
    elif sentiment_score ==  1: prob_up, prob_dn, prob_mut = 0.40, 0.25, 0.35
    elif sentiment_score == -1: prob_up, prob_dn, prob_mut = 0.25, 0.40, 0.35
    else:                       prob_up, prob_dn, prob_mut = 0.33, 0.33, 0.34

    # ── Premium bias ──────────────────────────────────────────────────────────
    if hv_rank > 0.70:
        long_bias  = "AVOID — high IV rank favors selling premium"
        short_bias = "PREFERRED — high IV rank, sell OTM spreads"
    elif hv_rank < 0.30:
        long_bias  = "PREFERRED — low IV rank, buy premium"
        short_bias = "AVOID — low IV environment, premium too cheap to sell"
    else:
        long_bias  = "CONSIDER — moderate IV environment"
        short_bias = "CONSIDER — moderate IV environment"

    dir_label = ("Bullish" if sentiment_score > 0
                 else "Bearish" if sentiment_score < 0 else "Neutral")

    # ── Alert ─────────────────────────────────────────────────────────────────
    if abs(sentiment_score) >= 2:
        alert = (f"Strong {'bullish' if sentiment_score > 0 else 'bearish'} "
                 f"options signal: PCR={pcr_30d:.2f}, skew={skew_slope}")
    elif iv_implied_move < atr_move_pct * 0.70:
        alert = (f"Vol mismatch: IV implies {iv_implied_move:.2f}% move but "
                 f"ATR suggests {atr_move_pct:.2f}% — vol spike risk")
    else:
        alert = (f"Options neutral: PCR={pcr_30d:.2f}, "
                 f"IV rank={hv_rank:.2f}, skew={skew_slope}")

    return {
        "options_sentiment":     options_sentiment,
        "put_call_ratio_signal": pcr_signal,
        "iv_assessment": {
            "current_iv":    atm_iv,
            "iv_rank":       hv_rank,
            "interpretation": iv_interp,
        },
        "skew_interpretation":   skew_interp,
        "expected_move_options": iv_implied_move,
        "expected_move_atr":     atr_move_pct,
        "move_mismatch":         mismatch,
        "sentiment_probability": {
            "big_move_up":   prob_up,
            "big_move_down": prob_dn,
            "muted_day":     prob_mut,
        },
        "trading_implications": {
            "long_premium_bias":   long_bias,
            "short_premium_bias":  short_bias,
            "direction_bias":      f"{dir_label} options flow",
            "scalp_opportunities": (
                f"Watch {put_support:.2f} support and "
                f"{call_resistance:.2f} resistance (max OI pin strikes)"
            ),
        },
        "key_levels_to_watch": {
            "put_support":     put_support,
            "call_resistance": call_resistance,
        },
        "alert": alert,
    }


# ── Public entry point ────────────────────────────────────────────────────────

def run_options_analysis(symbol: str) -> dict:
    """
    Fetches options chain + historical price data, computes all options
    sentiment metrics, and interprets them using deterministic rule thresholds.

    Returns full options sentiment analysis dict plus _options_meta with
    all raw computed values for dashboard display.
    """
    _fallback = {
        "options_sentiment":     "neutral",
        "put_call_ratio_signal": "neutral",
        "iv_assessment": {
            "current_iv":     0.0,
            "iv_rank":        0.5,
            "interpretation": "Options data unavailable.",
        },
        "skew_interpretation":   "Unable to compute skew.",
        "expected_move_options": 0.0,
        "expected_move_atr":     0.0,
        "move_mismatch":         "Options data unavailable.",
        "sentiment_probability": {"big_move_up": 0.33, "big_move_down": 0.33, "muted_day": 0.34},
        "trading_implications": {
            "long_premium_bias":   "UNKNOWN — data unavailable",
            "short_premium_bias":  "UNKNOWN — data unavailable",
            "direction_bias":      "NEUTRAL",
            "scalp_opportunities": "No options data to analyze.",
        },
        "key_levels_to_watch": {"put_support": None, "call_resistance": None},
        "alert": "Options sentiment scan failed — check data connection.",
    }

    # ── Step 1: 1-year daily OHLCV for HV stats ───────────────────────────────
    try:
        df = yf.Ticker(symbol).history(period="1y", interval="1d", auto_adjust=True)
        if df is None or len(df) < 30:
            _fallback["alert"] = f"Insufficient historical data for {symbol}"
            return _fallback
    except Exception as exc:
        _fallback["alert"] = f"Historical data fetch failed: {exc}"
        return _fallback

    current_price = round(float(df["Close"].iloc[-1]), 2)

    # ── Step 2: ATR-14 ────────────────────────────────────────────────────────
    atr          = _compute_atr(df)
    atr_move_pct = round(atr / current_price * 100, 3) if current_price else 0.0

    # ── Step 3: HV-based IV rank / percentile ─────────────────────────────────
    current_hv, hv_30d_avg, hv_rank, hv_percentile = _compute_hv_stats(df)

    # ── Step 4: Live options chain (optional — falls back to HV proxy) ────────
    opts = None
    try:
        opts = _fetch_options_snapshot(symbol, current_price)
    except Exception:
        pass

    if opts is not None:
        atm_iv          = opts["atm_iv"]
        otm_put_iv      = opts["otm_put_iv"]
        otm_call_iv     = opts["otm_call_iv"]
        pcr_30d         = opts["put_call_ratio_30d"]
        pcr_7d          = opts["put_call_ratio_7d"]
        skew_slope      = opts["skew_slope"]
        put_support     = opts["put_support"]
        call_resistance = opts["call_resistance"]
        iv_implied_move = opts["iv_implied_move_pct"]
        data_source     = "live_options_chain"
    else:
        atm_iv          = current_hv
        otm_put_iv      = round(current_hv * 1.10, 2)
        otm_call_iv     = round(current_hv * 0.95, 2)
        pcr_30d         = 1.0
        pcr_7d          = 1.0
        skew_slope      = "symmetric"
        put_support     = round(current_price * 0.97, 2)
        call_resistance = round(current_price * 1.03, 2)
        iv_implied_move = round(current_hv / 100 / np.sqrt(252) * 100, 3)
        data_source     = "hv_proxy_only"

    # ── Step 5: Pure-math interpretation ──────────────────────────────────────
    result = _interpret_options(
        pcr_30d, pcr_7d, hv_rank,
        atm_iv, otm_put_iv, otm_call_iv,
        skew_slope, put_support, call_resistance,
        iv_implied_move, atr_move_pct, current_price,
    )

    # Inject raw meta for dashboard
    result["_options_meta"] = {
        "current_price":       current_price,
        "atr":                 atr,
        "atr_move_pct":        atr_move_pct,
        "atm_iv":              atm_iv,
        "otm_put_iv":          otm_put_iv,
        "otm_call_iv":         otm_call_iv,
        "hv_rank":             hv_rank,
        "hv_percentile":       hv_percentile,
        "hv_current":          current_hv,
        "hv_30d_avg":          hv_30d_avg,
        "pcr_30d":             pcr_30d,
        "pcr_7d":              pcr_7d,
        "put_support":         put_support,
        "call_resistance":     call_resistance,
        "iv_implied_move_pct": iv_implied_move,
        "data_source":         data_source,
    }

    return result
