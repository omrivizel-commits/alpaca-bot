"""
Master Agent — Pure Mathematical Master Trading Decision Engine.

The final integration layer. Receives all available signal inputs:
  - Technical signal   (from signal_agent via pipeline state)
  - News catalyst      (from news_agent via pipeline state)
  - Sector context     (from sector_agent — optional)
  - Options market     (from options_agent — optional)
  - Quant signals      (bb_pvalue, rsi_pvalue, green_light from pipeline)
  - Vision signal      (chart_pattern, vision_veto from vision_agent)
  - Account constraints (size, max risk %, existing position)

Weighted scoring formula (all components normalized to −1 … +1):
  Technical signal : 40%
  Quant signals    : 25%
  News catalyst    : 20%
  Options market   : 10%
  Sector rotation  :  5%

Decision thresholds:
  total_score ≥ +0.25 and tech_score ≥ 0 → BUY
  total_score ≤ −0.25 and tech_score ≤ 0 → SELL
  |total_score| < 0.10                    → HOLD
  otherwise                               → HOLD
  Vision veto                             → AVOID (overrides BUY/SELL)
  R:R < 1.5 and confidence < 70           → AVOID

No AI calls required.
"""

from datetime import datetime, timezone


# ── Score helpers ─────────────────────────────────────────────────────────────

def _signal_to_score(signal: str, confidence: int) -> float:
    """Converts a BUY/SELL/HOLD signal + confidence (0-100) to a −1 … +1 score."""
    base = {
        "BUY":              +1.0,
        "SELL":             -1.0,
        "HOLD":              0.0,
        "AVOID":            -0.5,
        "INSUFFICIENT_DATA": 0.0,
    }
    return base.get(str(signal).upper(), 0.0) * (confidence / 100.0)


def _quant_score(state: dict) -> float:
    """
    Scores quant signals from the pipeline state.
    Returns a value in [0, +1] — quant always confirms or is neutral,
    never contradicts directly.
    """
    score = 0.0
    bb_p  = float(state.get("bb_pvalue",  0.5))
    rsi_p = float(state.get("rsi_pvalue", 0.5))
    gl    = bool(state.get("green_light", False))

    if   bb_p  < 0.05: score += 0.50
    elif bb_p  < 0.10: score += 0.25

    if   rsi_p < 0.05: score += 0.50
    elif rsi_p < 0.10: score += 0.25

    if gl: score += 0.50

    return min(1.0, score)


def _sentiment_to_score(sentiment: str, sentiment_score_raw) -> float:
    """Converts news sentiment to −1 … +1 score."""
    s = str(sentiment).lower()
    try:
        magnitude = min(1.0, abs(float(sentiment_score_raw)) / 100.0)
    except (TypeError, ValueError):
        magnitude = 0.3
    if   s == "bullish": return +magnitude
    elif s == "bearish": return -magnitude
    return 0.0


def _options_to_score(options_ctx: dict) -> float:
    """Converts options context to −1 … +1 score."""
    sentiment = str(options_ctx.get("sentiment",      "unknown")).lower()
    pcr       = float(options_ctx.get("put_call_ratio", 1.0))

    if sentiment == "bullish":   return +0.50
    if sentiment == "bearish":   return -0.50
    if sentiment == "defensive": return -0.30

    # Fallback to PCR
    if   pcr < 0.65: return +0.30
    elif pcr > 1.30: return -0.30
    return 0.0


def _sector_to_score(sector_ctx: dict) -> float:
    """Converts sector context to −0.5 … +0.5 score."""
    momentum = str(sector_ctx.get("sector_momentum", "unknown")).lower()
    change   = float(sector_ctx.get("sector_change_pct", 0.0))

    if momentum in ("strong", "strong_bullish", "positive"): return +0.50
    if momentum in ("negative", "weak",  "strong_bearish"):  return -0.50
    if momentum in ("bullish",  "moderate"):                  return +0.25
    if momentum in ("bearish",  "moderate_negative"):         return -0.25

    # Fallback to change_pct
    return min(0.50, max(-0.50, change / 4.0))


def _shares_estimate(
    account_size: float,
    max_risk_pct: float,
    entry_price: float,
    stop_price,
) -> int:
    """Fixed-risk position sizing, capped at 10% of account by value."""
    if not stop_price or entry_price <= 0:
        return 0
    risk_per_share = abs(entry_price - stop_price)
    if risk_per_share <= 0:
        return 0
    raw          = int((account_size * max_risk_pct) / risk_per_share)
    cap_by_value = int(account_size * 0.10 / entry_price)
    return max(0, min(raw, cap_by_value))


def _skew_from_ivs(otm_put_iv: float, atm_iv: float, otm_call_iv: float) -> str:
    if otm_put_iv - atm_iv > 3.0:  return "put_heavy"
    if atm_iv - otm_call_iv > 3.0: return "call_heavy"
    return "symmetric"


# ── Public entry point ────────────────────────────────────────────────────────

def run_master_decision(
    symbol: str,
    state: dict,
    account_size: float = None,
    max_risk_pct: float = 0.02,
    options_summary: dict = None,
    sector_summary: dict = None,
) -> dict:
    """
    Integrates all available signals from the pipeline state plus optional
    enrichment (options, sector) and produces the final execution decision
    using a deterministic weighted scoring system.

    Parameters
    ----------
    symbol          : Ticker (e.g. 'NVDA')
    state           : Full pipeline state dict from build_state()
    account_size    : Override; if None, fetched from Alpaca equity
    max_risk_pct    : Fraction of account to risk per trade (default 0.02)
    options_summary : Output from run_options_analysis() — optional
    sector_summary  : Output from run_sector_rotation()  — optional

    Returns
    -------
    Full decision dict plus _master_meta with raw input values for
    logging and dashboard display.
    """
    _fallback = {
        "final_decision":    "HOLD",
        "confidence":        0,
        "decision_summary":  "Master decision engine unavailable — defaulting to HOLD.",
        "execution": {
            "action": "HOLD", "ticker": symbol,
            "entry_price": 0.0, "shares": 0,
            "position_value": 0.0, "position_size_pct": 0.0,
        },
        "risk_management": {
            "stop_loss": None, "risk_per_share": 0.0,
            "total_risk_usd": 0.0, "risk_pct": 0.0,
        },
        "exits": {
            "target_1": {"price": None, "shares": 0, "profit": 0},
            "target_2": {"price": None, "shares": 0, "profit": 0},
        },
        "signal_alignment": {
            "technicals": "neutral", "catalyst":  "unknown",
            "sector":     "neutral", "options":   "unknown",
            "overall_alignment": "0%",
        },
        "time_horizon":      "unknown",
        "risk_level":        "HIGH",
        "execution_urgency": "NO_TRADE — data unavailable",
        "exit_triggers":     [],
    }

    # ── Account size ──────────────────────────────────────────────────────────
    if account_size is None:
        try:
            from execution.alpaca_broker import get_portfolio_value
            account_size = float(get_portfolio_value())
        except Exception:
            account_size = 10_000.0

    current_price = float(state.get("price", 0.0))
    if current_price <= 0:
        _fallback["decision_summary"] = "No price data available."
        return _fallback

    # ── Technical signal ──────────────────────────────────────────────────────
    entry_price = float(state.get("signal_entry") or current_price)
    stop_price  = state.get("signal_stop_loss")
    target1     = state.get("signal_take_profit_1")
    target2     = state.get("signal_take_profit_2")
    rr_ratio    = state.get("signal_risk_reward")

    tech_signal     = state.get("signal_gemini") or state.get("signal", "HOLD")
    tech_confidence = int(state.get("signal_confidence", 50))

    # ── Score all components ──────────────────────────────────────────────────

    # 1. Technical (40%)
    tech_score = _signal_to_score(tech_signal, tech_confidence)

    # 2. Quant (25%)
    quant_raw   = _quant_score(state)
    # Quant confirms direction of technical signal
    if tech_score < 0:
        quant_score = -quant_raw
    else:
        quant_score = +quant_raw

    # 3. News/catalyst (20%)
    sentiment_str   = str(state.get("sentiment_direction", "neutral"))
    sentiment_raw   = state.get("sentiment_score", 0)
    news_score      = _sentiment_to_score(sentiment_str, sentiment_raw)

    # 4. Options (10%)
    if options_summary and isinstance(options_summary, dict):
        meta = options_summary.get("_options_meta", {})
        options_ctx = {
            "put_call_ratio": float(meta.get("pcr_30d", 1.0)),
            "sentiment":      options_summary.get("options_sentiment", "neutral"),
        }
    else:
        options_ctx = {"put_call_ratio": 1.0, "sentiment": "unknown"}
    options_score = _options_to_score(options_ctx)

    # 5. Sector (5%)
    if sector_summary and isinstance(sector_summary, dict):
        ranking = sector_summary.get("sector_ranking", [])
        top     = ranking[0] if ranking else {}
        sector_ctx = {
            "sector_momentum":   top.get("momentum", "unknown"),
            "sector_change_pct": float(top.get("change_pct", 0.0)),
        }
    else:
        sector_ctx = {"sector_momentum": "unknown", "sector_change_pct": 0.0}
    sector_score_val = _sector_to_score(sector_ctx)

    # ── Weighted total ────────────────────────────────────────────────────────
    total_score = (
        tech_score       * 0.40 +
        quant_score      * 0.25 +
        news_score       * 0.20 +
        options_score    * 0.10 +
        sector_score_val * 0.05
    )
    total_score = round(total_score, 4)

    # ── Decision ──────────────────────────────────────────────────────────────
    if   total_score >= +0.25 and tech_score >= 0: final_decision = "BUY"
    elif total_score <= -0.25 and tech_score <= 0: final_decision = "SELL"
    else:                                          final_decision = "HOLD"

    # R:R guard — avoid trades with poor reward-to-risk unless high confidence
    if rr_ratio is not None:
        rr = float(rr_ratio)
        if rr < 1.5 and abs(total_score) * 100 < 70:
            final_decision = "AVOID"

    # Vision veto override (hard rule)
    vision_vetoed = False
    if bool(state.get("vision_veto", False)) and final_decision in ("BUY", "SELL"):
        final_decision = "AVOID"
        vision_vetoed  = True

    # ── Confidence (0-100) ────────────────────────────────────────────────────
    confidence = max(0, min(100, int(abs(total_score) * 100)))

    # ── Baseline shares ───────────────────────────────────────────────────────
    baseline_shares = _shares_estimate(account_size, max_risk_pct,
                                       entry_price, stop_price)
    shares = baseline_shares if final_decision in ("BUY", "SELL") else 0

    position_value   = round(shares * entry_price, 2)
    position_size_pct = round(position_value / account_size * 100, 2) \
                        if account_size > 0 else 0.0

    # ── Risk management ───────────────────────────────────────────────────────
    stop_f  = float(stop_price)  if stop_price  else None
    risk_ps = round(abs(entry_price - stop_f), 2) if stop_f else 0.0
    total_risk_usd = round(shares * risk_ps, 2)
    risk_pct_val   = round(total_risk_usd / account_size * 100, 2) \
                     if account_size > 0 else 0.0

    # ── Exit targets ──────────────────────────────────────────────────────────
    def _profit(tgt):
        if tgt and shares:
            return round(abs(float(tgt) - entry_price) * shares, 2)
        return 0

    exits = {
        "target_1": {"price": float(target1) if target1 else None,
                     "shares": shares,     "profit": _profit(target1)},
        "target_2": {"price": float(target2) if target2 else None,
                     "shares": shares // 2, "profit": _profit(target2) // 2},
    }

    # ── Signal alignment labels ───────────────────────────────────────────────
    def _label(score: float) -> str:
        if   score >  0.60: return "strong_bullish"
        elif score >  0.20: return "bullish"
        elif score < -0.60: return "strong_bearish"
        elif score < -0.20: return "bearish"
        return "neutral"

    signal_alignment = {
        "technicals": _label(tech_score),
        "catalyst":   _label(news_score) if news_score != 0 else "unknown",
        "sector":     _label(sector_score_val),
        "options":    _label(options_score) if options_score != 0 else "unknown",
        "overall_alignment": f"{int(abs(total_score) * 100)}%",
    }

    # ── Time horizon and risk level ───────────────────────────────────────────
    if   abs(total_score) >= 0.60: time_horizon = "15-30 minutes until first target"
    elif abs(total_score) >= 0.40: time_horizon = "30-60 minutes until first target"
    elif abs(total_score) >= 0.25: time_horizon = "1-2 hours"
    else:                          time_horizon = "Day trade not advised"

    if   abs(total_score) >= 0.60 and confidence >= 70: risk_level = "LOW"
    elif abs(total_score) >= 0.40:                       risk_level = "MODERATE"
    else:                                                risk_level = "HIGH"

    # ── Execution urgency ─────────────────────────────────────────────────────
    if final_decision in ("BUY", "SELL") and abs(total_score) >= 0.40:
        urgency = f"IMMEDIATE — score {total_score:+.3f}, all signals aligned"
    elif final_decision in ("BUY", "SELL"):
        urgency = f"WAIT — score {total_score:+.3f}, confirm with price action"
    else:
        urgency = f"NO_TRADE — score {total_score:+.3f} below threshold"

    # ── Exit triggers ─────────────────────────────────────────────────────────
    triggers = []
    if stop_f:
        triggers.append(f"Stop hit at {stop_f:.2f}")
    if target1:
        triggers.append(f"Take profit 1 at {float(target1):.2f} — exit 50%")
    if target2:
        triggers.append(f"Take profit 2 at {float(target2):.2f} — exit remainder")
    if bool(state.get("resistance_nearby", False)):
        triggers.append("Resistance nearby — scale out early")
    triggers.append("Time stop: exit all positions 30 min before market close")

    # ── Decision summary ──────────────────────────────────────────────────────
    summary = (
        f"{final_decision} {symbol} — score {total_score:+.3f} | "
        f"tech={tech_score:+.2f} quant={quant_score:+.2f} "
        f"news={news_score:+.2f} opts={options_score:+.2f} "
        f"sector={sector_score_val:+.2f}"
    )
    if vision_vetoed:
        summary += " [Vision veto overrides — no entry.]"
    if final_decision == "AVOID" and not vision_vetoed:
        summary += " [R:R below threshold.]"

    result = {
        "final_decision":   final_decision,
        "confidence":       confidence,
        "decision_summary": summary,
        "execution": {
            "action":            final_decision,
            "ticker":            symbol,
            "entry_price":       round(entry_price, 2),
            "shares":            shares,
            "position_value":    position_value,
            "position_size_pct": position_size_pct,
        },
        "risk_management": {
            "stop_loss":      stop_f,
            "risk_per_share": risk_ps,
            "total_risk_usd": total_risk_usd,
            "risk_pct":       risk_pct_val,
        },
        "exits":            exits,
        "signal_alignment": signal_alignment,
        "time_horizon":     time_horizon,
        "risk_level":       risk_level,
        "execution_urgency": urgency,
        "exit_triggers":    triggers,
        "_master_meta": {
            "account_size":    round(account_size, 2),
            "max_risk_pct":    max_risk_pct,
            "baseline_shares": baseline_shares,
            "entry_price":     round(entry_price, 2),
            "stop_price":      round(stop_f, 2) if stop_f else None,
            "had_options":     options_summary is not None,
            "had_sector":      sector_summary  is not None,
            "total_score":     total_score,
            "component_scores": {
                "technical": round(tech_score,       4),
                "quant":     round(quant_score,      4),
                "news":      round(news_score,       4),
                "options":   round(options_score,    4),
                "sector":    round(sector_score_val, 4),
            },
        },
    }

    return result
