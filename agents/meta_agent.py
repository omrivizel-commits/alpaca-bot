"""
Meta Agent — Pure Mathematical Kelly Criterion Composite Confidence & Position Sizing.

Aggregates all signal sources (quant, sentiment, vision, signal agent)
into a single composite confidence score, then applies the Kelly Criterion
and hard risk limits to recommend an exact share count.

Signal conflict detection: bullish technicals + bearish news → penalty.
Hard cap: never recommend > 5% of account.
Sizing formula: (Confidence% × 2% of account) / Risk%

No AI calls required — all outputs are computed deterministically.
"""

from config import settings
from execution.alpaca_broker import get_portfolio_value


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pvalue_to_confidence(bb_p: float, rsi_p: float) -> int:
    """Converts quant p-values to a 0–100 confidence integer."""
    avg_p = (bb_p + rsi_p) / 2
    raw   = max(0.0, (0.05 - avg_p) / 0.05)
    return min(100, int(raw * 100))


def _vision_confidence_to_int(vision_conf: str) -> int:
    return {"HIGH": 80, "MEDIUM": 60, "LOW": 30}.get(str(vision_conf).upper(), 50)


def _get_historical_stats() -> tuple:
    """
    Returns (win_rate, avg_win_loss_ratio) from closed trades in the log.
    Defaults to (0.55, 2.0) if fewer than 5 closed trades exist.
    """
    try:
        from utils.post_mortem import _load_log
        records = _load_log()
        closed  = [r for r in records if r.get("result") in ("WIN", "LOSS")]

        if len(closed) < 5:
            return 0.55, 2.0

        wins   = [r for r in closed if r["result"] == "WIN"]
        losses = [r for r in closed if r["result"] == "LOSS"]
        win_rate = len(wins) / len(closed)

        win_pcts  = [abs(r["pnl_pct"]) for r in wins   if r.get("pnl_pct")]
        loss_pcts = [abs(r["pnl_pct"]) for r in losses if r.get("pnl_pct")]

        avg_win  = sum(win_pcts)  / len(win_pcts)  if win_pcts  else 0.02
        avg_loss = sum(loss_pcts) / len(loss_pcts) if loss_pcts else 0.01
        ratio    = round(avg_win / avg_loss, 2) if avg_loss > 0 else 2.0

        return win_rate, ratio
    except Exception:
        return 0.55, 2.0


def _build_signals(state: dict) -> dict:
    """Converts the pipeline StateObject into component signal scores."""
    tech_conf = _pvalue_to_confidence(
        state.get("bb_pvalue",  0.05),
        state.get("rsi_pvalue", 0.05),
    )
    sent_score = state.get("sentiment_score", 0.0)
    news_conf  = min(100, int(abs(float(sent_score)) * 100))
    news_type  = state.get("sentiment_direction", "NEUTRAL").lower()

    vision_conf_raw = state.get("vision_confidence", "MEDIUM")
    vision_int  = _vision_confidence_to_int(vision_conf_raw)
    vision_type = "bullish" if not state.get("vision_veto") else "bearish"

    gem_sig  = state.get("signal_gemini", "HOLD")
    gem_conf = state.get("signal_confidence", 0)

    return {
        "technical_signal": {"type": state.get("signal", "HOLD"), "confidence": tech_conf},
        "news_sentiment":   {"type": news_type,                    "confidence": news_conf},
        "vision_signal":    {"type": vision_type,                  "confidence": vision_int},
        "gemini_signal":    {"type": gem_sig.lower() if gem_sig else "hold", "confidence": gem_conf},
    }


# ── Pure-math composite scoring ───────────────────────────────────────────────

def _compute_composite(signals: dict, vision_veto: bool) -> tuple:
    """
    Computes a composite confidence (0-100) and alignment score.
    Returns (composite_confidence, alignment_score, confidence_breakdown).
    """
    tech  = signals["technical_signal"]
    news  = signals["news_sentiment"]
    vision = signals["vision_signal"]
    gem   = signals["gemini_signal"]

    # Directional agreement
    def _is_bullish(sig: dict) -> bool:
        return sig["type"].lower() in ("buy", "bullish", "bullish_bias")

    def _is_bearish(sig: dict) -> bool:
        return sig["type"].lower() in ("sell", "bearish", "bearish_bias", "bearish")

    bullish_count = sum([_is_bullish(tech), _is_bullish(news),
                         _is_bullish(vision), _is_bullish(gem)])
    bearish_count = sum([_is_bearish(tech), _is_bearish(news),
                         _is_bearish(vision), _is_bearish(gem)])

    # Conflict penalty
    conflict = abs(bullish_count - bearish_count)
    alignment_score = round(conflict / 4, 2)

    # Weighted confidence
    weights = {"technical": 0.35, "news": 0.20, "vision": 0.20, "signal": 0.25}
    raw_conf = (
        tech["confidence"]   * weights["technical"] +
        news["confidence"]   * weights["news"]      +
        vision["confidence"] * weights["vision"]    +
        gem["confidence"]    * weights["signal"]
    )

    # Conflict penalty reduces confidence by 20% per conflicting signal pair
    conflict_pairs = min(bullish_count, bearish_count)
    conflict_penalty = conflict_pairs * 0.20
    adj_conf = int(raw_conf * (1 - conflict_penalty))

    # Vision veto hard-caps confidence at 40
    if vision_veto:
        adj_conf = min(adj_conf, 40)

    composite = max(0, min(100, adj_conf))
    breakdown = {
        "technical":      tech["confidence"],
        "news":           news["confidence"],
        "vision":         vision["confidence"],
        "gemini_signal":  gem["confidence"],
        "alignment_score": alignment_score,
    }
    return composite, alignment_score, breakdown


# ── Public entry point ────────────────────────────────────────────────────────

def run_meta_sizing(
    symbol: str,
    state: dict,
    entry_price: float,
    stop_loss: float,
    account_size: float = None,
) -> dict:
    """
    Aggregates all signals, computes Kelly-adjusted position size, and returns
    a final share recommendation using pure mathematical rules.

    Parameters
    ----------
    symbol       : Ticker
    state        : Full pipeline StateObject
    entry_price  : Confirmed entry price
    stop_loss    : ATR/S&R-derived stop-loss from position_agent
    account_size : Override; defaults to live portfolio equity from Alpaca

    Returns a full sizing dict.
    """
    risk_per_share = round(abs(entry_price - stop_loss), 4) \
                     if stop_loss else entry_price * settings.STOP_LOSS_PCT

    try:
        portfolio_val = account_size or get_portfolio_value()
    except Exception:
        portfolio_val = 25000.0

    max_risk_usd      = round(portfolio_val * settings.RISK_PCT, 2)
    max_position_usd  = portfolio_val * 0.05

    win_rate, avg_wl  = _get_historical_stats()
    signals           = _build_signals(state)
    vision_veto       = bool(state.get("vision_veto", False))

    # ── Composite confidence ──────────────────────────────────────────────────
    composite, alignment_score, breakdown = _compute_composite(signals, vision_veto)

    # ── Kelly Criterion ───────────────────────────────────────────────────────
    p = win_rate
    q = 1 - p
    b = avg_wl

    kelly_raw   = (p * b - q) / b if b > 0 else 0.0
    kelly_frac  = max(0.0, min(0.25, round(kelly_raw, 4)))   # cap at 25%
    kelly_adj   = kelly_frac * (composite / 100.0)           # scale by confidence

    # ── Position size ─────────────────────────────────────────────────────────
    # Primary formula: (Confidence% × 2% of account) / risk_per_share
    conf_risk_usd = round(portfolio_val * 0.02 * (composite / 100.0), 2)
    qty_by_conf   = int(conf_risk_usd / risk_per_share) if risk_per_share > 0 else 1

    qty_by_risk = int(max_risk_usd / risk_per_share) if risk_per_share > 0 else 1
    qty_by_cap  = int(max_position_usd / entry_price)

    shares = max(1, min(qty_by_conf, qty_by_risk, qty_by_cap))

    position_value   = round(shares * entry_price, 2)
    position_size_pct = round(position_value / portfolio_val, 4) if portfolio_val > 0 else 0.0

    # ── Risk assessment ───────────────────────────────────────────────────────
    ev = round(p * avg_wl * max_risk_usd - q * max_risk_usd, 2)
    prob_ruin_10 = round(q ** 10, 4)

    sizing_rec = f"{shares} shares ({position_size_pct*100:.1f}% of account, {composite}% confidence)"
    sizing_rule = (
        f"Kelly={kelly_frac:.2%}, adjusted={kelly_adj:.2%} → "
        f"({composite}% conf × 2% account) / ${risk_per_share:.2f} risk/share"
    )
    summary = (
        f"Composite confidence {composite}% | "
        f"{'Low conflict' if alignment_score >= 0.75 else 'Signal conflict detected'} | "
        f"Kelly-adjusted {shares} shares"
    )

    return {
        "composite_confidence": composite,
        "confidence_breakdown": breakdown,
        "position_size_calculation": {
            "risk_per_trade_usd":  max_risk_usd,
            "entry_price":         entry_price,
            "stop_loss":           stop_loss,
            "risk_per_share":      risk_per_share,
            "shares_recommended":  shares,
            "position_value":      position_value,
        },
        "position_size_pct":       position_size_pct,
        "kelly_criterion":         kelly_frac,
        "kelly_adjusted_position": kelly_adj,
        "sizing_recommendation":   sizing_rec,
        "risk_assessment": {
            "probability_win":            win_rate,
            "expected_value":             ev,
            "probability_ruin_10trades":  prob_ruin_10,
        },
        "confidence_summary": summary,
        "sizing_rule":        sizing_rule,
    }
