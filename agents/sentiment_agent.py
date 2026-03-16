"""
Sentiment Agent — Pure Mathematical News Sentiment Analyzer.

Fetches the latest headlines via Finnhub, classifies each using keyword
rules, computes per-headline scores, and aggregates them into a single
composite sentiment — no AI calls required.

Aggregate logic:
  - Score each headline using the same keyword map as news_agent
  - Weight the most recent headlines more heavily (decay by index)
  - Most market-moving headline = highest |score|
  - impact_category: MAJOR if |composite| > 0.50 | MODERATE > 0.20 | NOISE

Falls back to NEUTRAL on any error.
"""

import numpy as np

from data.news_fetcher import fetch_headlines


# ── Sentiment keyword map (shared with news_agent) ───────────────────────────

_SENTIMENT_SCORES: dict = {
    "earnings_beat":    +85,
    "earnings":         +15,
    "earnings_miss":    -85,
    "guidance_raised":  +65,
    "guidance":           0,
    "guidance_lowered": -65,
    "m_and_a":          +40,
    "legal":            -55,
    "product":          +25,
    "analyst":            0,    # refined below
    "macro":              0,    # refined below
    "other":              0,
}

_CATALYST_LABEL_MAP: dict = {
    "earnings_beat":    "EARNINGS",
    "earnings_miss":    "EARNINGS",
    "earnings":         "EARNINGS",
    "guidance_raised":  "GUIDANCE",
    "guidance_lowered": "GUIDANCE",
    "guidance":         "GUIDANCE",
    "m_and_a":          "M_AND_A",
    "legal":            "LEGAL",
    "product":          "PRODUCT",
    "analyst":          "ANALYST",
    "macro":            "MACRO",
    "other":            "OTHER",
}


def _classify(headline: str) -> str:
    h = headline.lower()
    if any(w in h for w in ["beat", "exceed", "surpass", "record", "q1", "q2",
                              "q3", "q4", "eps", "earnings"]):
        if any(w in h for w in ["beat", "exceed", "surpass", "record", "above"]): return "earnings_beat"
        if any(w in h for w in ["miss", "below", "disappoint", "short"]):         return "earnings_miss"
        return "earnings"
    if any(w in h for w in ["guidance", "outlook", "forecast"]):
        if any(w in h for w in ["raise", "higher", "increase", "boost"]): return "guidance_raised"
        if any(w in h for w in ["cut", "lower", "reduce", "slash", "warn"]):  return "guidance_lowered"
        return "guidance"
    if any(w in h for w in ["acqui", "merger", "buyout", "takeover", "deal", "bid"]): return "m_and_a"
    if any(w in h for w in ["lawsuit", "sec", "probe", "investig",
                              "fine", "penalty", "settlement"]):                          return "legal"
    if any(w in h for w in ["upgrade", "downgrade", "price target", "analyst",
                              "rating", "outperform", "underperform"]):                   return "analyst"
    if any(w in h for w in ["launch", "product", "unveil", "announce",
                              "release", "debut"]):                                       return "product"
    if any(w in h for w in ["rate", "inflation", "fed", "gdp", "macro",
                              "economy", "tariff", "trade war"]):                         return "macro"
    return "other"


def _score_one(headline: str, cls: str) -> float:
    """Returns a raw −100 … +100 integer score, converted to −1 … +1 float."""
    h    = headline.lower()
    base = _SENTIMENT_SCORES.get(cls, 0)

    if cls == "analyst":
        if any(w in h for w in ["upgrade", "outperform", "overweight", "buy", "strong buy"]):
            base = +35
        elif any(w in h for w in ["downgrade", "underperform", "underweight", "sell", "reduce"]):
            base = -35
        elif any(w in h for w in ["neutral", "hold", "equal"]):
            base = 0
        else:
            base = +10

    if cls == "macro":
        if any(w in h for w in ["cut", "lower", "dovish", "stimulus"]): base = +20
        elif any(w in h for w in ["hike", "raise", "hawkish",
                                   "tariff", "trade war", "inflation"]): base = -20

    return max(-100, min(100, base)) / 100.0


# ── Public entry point ────────────────────────────────────────────────────────

def analyze_sentiment(symbol: str, market_cap_billions: float = None) -> dict:
    """
    Fetches up to 20 headlines for `symbol` and produces a composite
    sentiment score using deterministic keyword-based rules.

    Returns a dict compatible with the pipeline StateObject.
    """
    _neutral = {
        "sentiment_score":    0.0,
        "direction":          "NEUTRAL",
        "relative_impact":    "No news found",
        "top_headline":       "",
        "impact_category":    "NOISE",
        "catalyst_type":      "OTHER",
        "already_priced_in":  True,
        "sentiment_reasoning": "No headlines available.",
    }

    try:
        headlines = fetch_headlines(symbol, n=20)
    except Exception:
        headlines = []

    if not headlines:
        return _neutral

    # ── Score each headline with recency decay ────────────────────────────────
    n = len(headlines)
    scores     = []
    top_score  = 0.0
    top_idx    = 0
    top_cls    = "other"

    for i, h in enumerate(headlines):
        cls   = _classify(h)
        score = _score_one(h, cls)
        # Recency weight: most recent (index 0) = 1.0, oldest = 0.5
        weight = 1.0 - (i / max(n - 1, 1)) * 0.5
        scores.append(score * weight)

        if abs(score) > abs(top_score):
            top_score = score
            top_idx   = i
            top_cls   = cls

    # Weighted average
    composite = float(np.mean(scores)) if scores else 0.0
    composite = round(max(-1.0, min(1.0, composite)), 4)

    # ── Direction & impact category ───────────────────────────────────────────
    if   composite >  0.20: direction = "BULLISH"
    elif composite < -0.20: direction = "BEARISH"
    else:                   direction = "NEUTRAL"

    if   abs(composite) > 0.50: impact = "MAJOR"
    elif abs(composite) > 0.20: impact = "MODERATE"
    else:                       impact = "NOISE"

    # ── Already priced in? ────────────────────────────────────────────────────
    # Heuristic: if all top 5 headlines are of the same type, market likely
    # already knows; only major unexpected events are fresh catalysts
    already_in = abs(composite) < 0.25

    catalyst_label = _CATALYST_LABEL_MAP.get(top_cls, "OTHER")

    relative_impact = (
        f"Composite from {n} headlines: {direction.lower()} "
        f"({abs(composite):.2f} magnitude). "
        f"Top driver: {top_cls.replace('_', ' ')}."
    )

    reasoning = (
        f"Scored {n} headlines; composite = {composite:+.4f}. "
        f"Most impactful: '{headlines[top_idx][:80]}' "
        f"(class={top_cls}, score={top_score:+.2f}). "
        f"Direction: {direction}."
    )

    return {
        "sentiment_score":    composite,
        "direction":          direction,
        "relative_impact":    relative_impact,
        "top_headline":       headlines[top_idx] if headlines else "",
        "impact_category":    impact,
        "catalyst_type":      catalyst_label,
        "already_priced_in":  already_in,
        "sentiment_reasoning": reasoning,
    }
