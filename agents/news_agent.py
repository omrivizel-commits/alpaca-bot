"""
News Agent — Pure Mathematical News Sentiment Analyzer.

Classifies headlines using keyword rules and maps them to deterministic
sentiment scores and position recommendations — no AI calls required.

Scoring map:
  earnings_beat    → +85   earnings_miss    → −85
  guidance_raised  → +65   guidance_lowered → −65
  m_and_a          → +40   legal            → −55
  analyst_upgrade  → +35   analyst_downgrade → −35
  product          → +25   macro            → ±20
  other            → 0

Position recommendation from (score × position_direction):
  score ≥ 65          → ADD (if long) / EXIT_HALF (if short)
  20 ≤ score < 65     → HOLD
  −20 < score < 20    → HOLD
  −50 ≤ score < −20   → EXIT_HALF (if long)
  score < −50         → EXIT_ALL  (if long)
  No position         → always HOLD
"""

import numpy as np
from datetime import datetime, timezone

from data.market_data import get_hourly


# ── Sentiment base scores ─────────────────────────────────────────────────────

_SENTIMENT_SCORES: dict = {
    "earnings_beat":    +85,
    "earnings":          +15,   # unclassified earnings — mild positive
    "earnings_miss":    -85,
    "guidance_raised":  +65,
    "guidance":           0,
    "guidance_lowered": -65,
    "m_and_a":          +40,
    "legal":            -55,
    "restructure":      -55,    # layoffs, job cuts, restructuring
    "negative_event":   -40,    # losses, deficit, subscribers lost
    "positive_catalyst": +35,   # rally, surge, profit, approval, growth
    "product":          +25,
    "macro":              0,    # refined by keyword below
    "analyst":            0,    # refined by keyword below
    "other":              0,
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _classify_headline(headline: str) -> str:
    """Keyword-based headline classifier — no API call needed."""
    h = headline.lower()
    # ── Earnings ──────────────────────────────────────────────────────────────
    if any(w in h for w in ["beat", "exceed", "surpass", "record", "q1", "q2",
                             "q3", "q4", "eps", "earnings", "miss"]):
        if any(w in h for w in ["beat", "exceed", "surpass", "record", "above"]):
            return "earnings_beat"
        if any(w in h for w in ["miss", "below", "disappoint", "short"]):
            return "earnings_miss"
        return "earnings"
    # ── Guidance ──────────────────────────────────────────────────────────────
    if any(w in h for w in ["guidance", "outlook", "forecast"]):
        if any(w in h for w in ["raise", "higher", "increase", "boost"]):
            return "guidance_raised"
        if any(w in h for w in ["cut", "lower", "reduce", "slash", "warn"]):
            return "guidance_lowered"
        return "guidance"
    # ── M&A ───────────────────────────────────────────────────────────────────
    if any(w in h for w in ["acqui", "merger", "buyout", "takeover", "deal", "bid"]):
        return "m_and_a"
    # ── Legal / regulatory (includes product recalls) ─────────────────────────
    if any(w in h for w in ["lawsuit", "sec", "probe", "investig",
                             "fine", "penalty", "settlement", "recall", "recalls"]):
        return "legal"
    # ── Analyst ───────────────────────────────────────────────────────────────
    if any(w in h for w in ["upgrade", "downgrade", "price target", "analyst",
                             "rating", "outperform", "underperform"]):
        return "analyst"
    # ── Restructuring / layoffs (check BEFORE product to avoid false product match) ──
    if any(w in h for w in ["layoff", "lay off", "laid off", "job cut", "job cuts",
                             "redundan", "workforce reduc"]):
        return "restructure"
    # ── Negative business events ──────────────────────────────────────────────
    if any(w in h for w in ["losses", "loses ", "deficit", " lose "]):
        return "negative_event"
    # ── Positive catalysts (check BEFORE generic product) ─────────────────────
    if any(w in h for w in ["approv", "rally", "surge", "surges", "profit",
                             "profits", "jump", "jumps"]):
        return "positive_catalyst"
    # ── Product / announcement ────────────────────────────────────────────────
    if any(w in h for w in ["launch", "product", "unveil", "announce", "release", "debut"]):
        return "product"
    # ── Macro / economic ──────────────────────────────────────────────────────
    if any(w in h for w in ["rate", "inflation", "fed", "gdp", "macro",
                             "economy", "tariff", "trade war",
                             "payroll", "unemployment", "jobs report"]):
        return "macro"
    return "other"


def _score_headline(headline: str, classification: str) -> int:
    """Computes a −100…+100 sentiment score using classification + keywords."""
    h    = headline.lower()
    base = _SENTIMENT_SCORES.get(classification, 0)

    if classification == "analyst":
        if any(w in h for w in ["upgrade", "outperform", "overweight",
                                 "buy", "strong buy"]):
            base = +35
        elif any(w in h for w in ["downgrade", "underperform", "underweight",
                                   "sell", "reduce"]):
            base = -35
        elif any(w in h for w in ["neutral", "hold", "equal"]):
            base =   0
        else:
            base = +10   # unspecified analyst comment — mild positive

    if classification == "macro":
        if any(w in h for w in ["cut", "lower", "dovish", "stimulus",
                                 "payroll", "unemployment", "jobs report"]):
            base = +20
        elif any(w in h for w in ["hike", "raise", "hawkish",
                                   "tariff", "trade war", "inflation"]):
            base = -20
        else:
            base =   0

    return max(-100, min(100, base))


def _recent_volatility(symbol: str) -> str:
    """Classifies recent price volatility from last 24 hourly bars."""
    try:
        df  = get_hourly(symbol, period="5d")
        std = df["Close"].pct_change().dropna().tail(24).std() * 100
        if   std > 2.0: return "elevated"
        elif std > 1.0: return "moderate"
        return "low"
    except Exception:
        return "moderate"


def _format_position_string(position: dict) -> str:
    qty   = float(position.get("qty",             0))
    side  = "LONG" if qty > 0 else "SHORT"
    entry = float(position.get("avg_entry_price", 0))
    return f"{side} {int(abs(qty))} shares @ {entry:.2f}"


# ── Public entry point ────────────────────────────────────────────────────────

def analyze_news(
    symbol: str,
    headline: str,
    source: str = "Unknown",
    news_timestamp: str = None,
    position: dict = None,
    stop_loss: float = None,
    take_profit: float = None,
    sector: str = "unknown",
) -> dict:
    """
    Analyses breaking news against an open position using pure keyword scoring.

    Parameters
    ----------
    symbol        : Ticker symbol
    headline      : News headline text
    source        : News source (Reuters, Bloomberg, etc.)
    news_timestamp: ISO-8601 string of when the news was published
    position      : Dict from alpaca_broker.get_position()
    stop_loss     : Active stop-loss price
    take_profit   : Active take-profit price
    sector        : Sector label for market context

    Returns a dict with sentiment, score, and position recommendation.
    """
    _fallback = {
        "sentiment":                  "neutral",
        "sentiment_score":            0,
        "expected_volatility_change": "moderate",
        "volatility_multiplier":      1.0,
        "probability_gap_up_30min":   0.5,
        "probability_spike_move":     0.5,
        "position_recommendation":    "HOLD",
        "suggested_exit_price":       None,
        "new_stop_loss":              stop_loss,
        "reasoning":                  "Analysis unavailable — defaulting to HOLD.",
        "catalysts":                  [],
        "risk_level":                 "moderate",
        "catalyst_type":              "other",
        "already_priced_in":          True,
        "sentiment_direction":        "neutral",
        "top_headline":               headline or "",
    }

    if not headline:
        _fallback["reasoning"] = "No headline provided."
        return _fallback

    # ── Derive prices ─────────────────────────────────────────────────────────
    if position:
        current_price = float(position.get("current_price",     0))
        entry_price   = float(position.get("avg_entry_price",   0))
        qty           = float(position.get("qty",               0))
    else:
        current_price = 0.0
        entry_price   = 0.0
        qty           = 0.0

    if stop_loss is None and entry_price:
        try:
            from config import settings
            stop_loss = round(entry_price * (1 - settings.STOP_LOSS_PCT), 2)
        except Exception:
            stop_loss = round(entry_price * 0.98, 2)

    if take_profit is None and entry_price and stop_loss:
        stop_dist   = entry_price - stop_loss
        take_profit = round(entry_price + stop_dist * 2, 2)

    # ── Classify and score headline ───────────────────────────────────────────
    classification = _classify_headline(headline)
    score          = _score_headline(headline, classification)

    # ── Sentiment direction ───────────────────────────────────────────────────
    if   score >= 20: sentiment = "bullish"
    elif score <= -20: sentiment = "bearish"
    else:              sentiment = "neutral"

    # ── Volatility from score magnitude ──────────────────────────────────────
    abs_score = abs(score)
    if   abs_score >= 65: vol_change = "high";     vol_mult = 2.5
    elif abs_score >= 40: vol_change = "moderate"; vol_mult = 1.6
    else:                 vol_change = "low";      vol_mult = 1.1

    # ── Spike / gap probabilities ─────────────────────────────────────────────
    spike_prob  = round(min(0.95, abs_score / 100.0 * 1.1),          2)
    gap_up_prob = round(max(0.0, min(1.0, 0.5 + score / 200.0)),     2)

    # ── Position recommendation ───────────────────────────────────────────────
    is_long = qty > 0

    if   score >= 65:  recommendation = "ADD"        if is_long else "EXIT_HALF"
    elif score >= 20:  recommendation = "HOLD"
    elif score >= -20: recommendation = "HOLD"
    elif score >= -50: recommendation = "EXIT_HALF" if is_long else "HOLD"
    else:              recommendation = "EXIT_ALL"  if is_long else "ADD"

    # No open position → always HOLD
    if not position or qty == 0:
        recommendation = "HOLD"

    # ── Risk level ────────────────────────────────────────────────────────────
    if   abs_score >= 65: risk_level = "extreme"
    elif abs_score >= 40: risk_level = "high"
    elif abs_score >= 20: risk_level = "moderate"
    else:                 risk_level = "low"

    # ── Tighten stop on extreme adverse news ─────────────────────────────────
    new_stop = stop_loss
    if abs_score >= 65 and entry_price and current_price:
        try:
            from config import settings
            tighter_pct = settings.STOP_LOSS_PCT * 0.5
        except Exception:
            tighter_pct = 0.01
        new_stop = round(entry_price * (1 - tighter_pct), 2) if is_long \
                   else round(entry_price * (1 + tighter_pct), 2)

    # ── Reasoning ─────────────────────────────────────────────────────────────
    reasoning = (
        f"Headline classified as '{classification}' → score {score:+d}. "
        f"Sentiment: {sentiment}. "
        f"{'Recommend ' + recommendation + ' on open position.' if qty != 0 else 'No position open.'}"
    )

    return {
        "sentiment":                  sentiment,
        "sentiment_score":            score,
        "expected_volatility_change": vol_change,
        "volatility_multiplier":      vol_mult,
        "probability_gap_up_30min":   gap_up_prob,
        "probability_spike_move":     spike_prob,
        "position_recommendation":    recommendation,
        "suggested_exit_price":       current_price if current_price else None,
        "new_stop_loss":              new_stop,
        "reasoning":                  reasoning,
        "catalysts":                  [classification] if classification != "other" else [],
        "risk_level":                 risk_level,
        "catalyst_type":              classification,
        "already_priced_in":          abs_score < 20,
        "sentiment_direction":        sentiment,
        "top_headline":               headline,
    }
