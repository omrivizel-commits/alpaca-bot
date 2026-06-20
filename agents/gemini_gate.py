"""
Gemini Gate — 4 AI-powered overlays using the free Gemini 2.0 Flash API.

Use cases (all share the same 15 RPM / 1,500 req/day free tier):
  1. gemini_trade_gate      — final AI sanity check before order execution  (~3/day)
  2. gemini_news_sentiment  — nuanced news analysis for open positions       (~50/day)
  3. gemini_morning_brief   — pre-market market overview at 9:00 AM ET       (1/day)
  4. gemini_scan_overlay    — per-symbol AI insight on every scan cycle      (~780/day)

Daily budget math (10 symbols, 5-min scan, 6.5 trading hours):
  Overlay:  10 symbols × 78 scans  = 780 calls
  News:     ~3 positions × ~17/day =  50 calls
  Gate:     ~1–3 trades/day         =   3 calls
  Brief:    once at 9:00 AM         =   1 call
  Total:                             ~834 calls/day  (1,500 free-tier limit)

Expand watchlist to 18 symbols → ~1,458 calls/day (at limit).

Rate limiting:
  Hard ceiling : 15 RPM (Google free tier)
  Min interval : 4.1 s between calls  → max 14.6 RPM (safe margin)
  Thread-safe  : threading.Lock protects _last_call_time across executors

Daily budget:
  Resets automatically each calendar day.
  Stops new calls at 1,490/day (10-call buffer below the 1,500 limit).

Fallbacks:
  Every function returns a safe default dict if Gemini is unavailable,
  budget is exhausted, or the API call fails — trading continues unaffected.
"""

import json
import logging
import os
import re
import threading
import time
from datetime import date
from typing import Any

logger = logging.getLogger("gemini_gate")

# ── Gemini client setup ───────────────────────────────────────────────────────

_model = None

def _get_model():
    """Lazy-load the Gemini model on first call (avoids import cost at startup)."""
    global _model
    if _model is not None:
        return _model

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        logger.warning("Gemini Gate: GEMINI_API_KEY not set — all calls return fallback data.")
        return None

    try:
        import google.generativeai as genai  # type: ignore
        genai.configure(api_key=api_key)
        _model = genai.GenerativeModel("gemini-2.0-flash")
        logger.info("Gemini Gate: model=gemini-2.0-flash ready.")
        return _model
    except ImportError:
        logger.error(
            "Gemini Gate: google-generativeai not installed. "
            "Run: pip install google-generativeai"
        )
        return None
    except Exception as e:
        logger.error(f"Gemini Gate: failed to init model — {e}")
        return None


# ── Rate limiting: 15 RPM free tier ──────────────────────────────────────────

_rate_lock       = threading.Lock()
_last_call_ts: float = 0.0
_MIN_INTERVAL_SEC = 4.1   # 60 / 4.1 ≈ 14.6 RPM — safely under the 15 RPM limit

# Daily call budget
_DAILY_LIMIT  = 1_490     # hard stop 10 calls below 1,500 free-tier limit
_budget_lock  = threading.Lock()
_daily_count  = 0
_budget_date  = ""


def _rate_limit() -> None:
    """Block the calling thread until MIN_INTERVAL_SEC has elapsed since the last call."""
    global _last_call_ts
    with _rate_lock:
        now  = time.monotonic()
        wait = (_last_call_ts + _MIN_INTERVAL_SEC) - now
        if wait > 0:
            time.sleep(wait)
        _last_call_ts = time.monotonic()


def _budget_ok() -> bool:
    """
    Returns True and increments the counter if we have remaining daily calls.
    Resets the counter automatically on a new calendar day.  Thread-safe.
    """
    global _daily_count, _budget_date
    today = date.today().isoformat()
    with _budget_lock:
        if _budget_date != today:
            _budget_date = today
            _daily_count = 0
        if _daily_count >= _DAILY_LIMIT:
            logger.warning(
                f"Gemini Gate: daily budget exhausted ({_daily_count}/{_DAILY_LIMIT}) — "
                f"returning fallback data until midnight."
            )
            return False
        _daily_count += 1
        remaining = _DAILY_LIMIT - _daily_count
        if remaining <= 50:
            logger.warning(f"Gemini Gate: low budget — {remaining} calls remaining today.")
        return True


def _parse_json(text: str) -> dict:
    """Strip markdown code fences and parse JSON.  Returns empty dict on failure."""
    text = text.strip()
    # Remove opening ```json or ``` fence
    text = re.sub(r'^```(?:json)?\s*', '', text)
    # Remove closing ``` fence
    text = re.sub(r'\s*```\s*$', '', text)
    text = text.strip()
    return json.loads(text)


def _call_gemini(prompt: str, fallback: dict, label: str = "") -> dict:
    """
    Core Gemini call:  rate-limited → budget-guarded → JSON-parsed → fallback on any error.
    All blocking I/O; designed to be called via loop.run_in_executor().
    """
    model = _get_model()
    if model is None:
        return dict(fallback)

    if not _budget_ok():
        return dict(fallback)

    _rate_limit()

    try:
        import google.generativeai as genai  # type: ignore

        response = model.generate_content(
            prompt,
            generation_config={
                "temperature":     0.1,
                "max_output_tokens": 512,
            },
        )
        result = _parse_json(response.text)
        if label:
            logger.debug(f"Gemini Gate [{label}]: call succeeded.")
        return result

    except json.JSONDecodeError as e:
        logger.warning(f"Gemini Gate [{label}]: JSON parse error — {e}. Raw: {response.text[:200]!r}")
        return dict(fallback)
    except Exception as e:
        logger.error(f"Gemini Gate [{label}]: API error — {e}")
        return dict(fallback)


# ─────────────────────────────────────────────────────────────────────────────
# Use Case 4: Per-symbol scan overlay   (~780 calls/day — main volume driver)
# ─────────────────────────────────────────────────────────────────────────────

_SCAN_FALLBACK: dict = {
    "gemini_signal":      "INSUFFICIENT_DATA",
    "gemini_confidence":  0,
    "gemini_key_insight": "",
    "gemini_risk_flags":  [],
}


def gemini_scan_overlay(symbol: str, state: dict) -> dict:
    """
    Analyses the full StateObject and returns an AI-powered signal overlay.
    Called from auto_trader._analyse() for every symbol on every scan cycle.

    Returns:
        gemini_signal      : "BUY" | "SELL" | "HOLD" | "INSUFFICIENT_DATA"
        gemini_confidence  : 0–100 integer
        gemini_key_insight : one concise sentence (the key reason)
        gemini_risk_flags  : list of identified risk strings (may be empty)
    """
    prompt = f"""{_build_role()}
---
TASK: Scan Overlay — real-time signal for TA-125 stock {symbol}.

Apply your multi-tiered evaluation (macro BoI rate context, TASE liquidity constraints, technical momentum) to the quantitative snapshot below, then run your internal Synthetic Backtest before committing to a signal. Factor in any geopolitical risk premium relevant to Israeli markets.

Market snapshot:
- Price (ILS): {state.get('price', 0):.2f}
- Quant signal: {state.get('signal', 'N/A')}  (BB p={state.get('bb_pvalue', 'N/A')}, RSI p={state.get('rsi_pvalue', 'N/A')}, ensemble={state.get('ensemble_agreement', 'N/A')})
- BB signal: {state.get('bb_signal', 'N/A')} | RSI signal: {state.get('rsi_signal', 'N/A')}
- ADX: {state.get('adx', 0):.1f} | Regime: {state.get('market_regime', 'N/A')}
- Rel. volume: {state.get('relative_volume', 1.0):.2f}× | Price vs VWAP: {state.get('price_vs_vwap', 'N/A')}
- Chart pattern: {state.get('chart_pattern', 'N/A')} | Trend: {state.get('trend', 'N/A')}
- Support: {state.get('nearest_support', 0):.2f} | Resistance: {state.get('nearest_resistance', 0):.2f}
- Resistance nearby: {state.get('resistance_nearby', False)} | Vision veto: {state.get('vision_veto', False)}
- Vision confidence: {state.get('vision_confidence', 0):.2f} | Vol confirmation: {state.get('volume_confirmation', 'N/A')}
- Candlestick patterns: {state.get('candlestick_patterns', [])}
- Sentiment: {state.get('sentiment_direction', 'N/A')} (score={state.get('sentiment_score', 0):.3f})
- Impact: {state.get('impact_category', 'N/A')} | Catalyst: {state.get('catalyst_type', 'N/A')}
- Top headline: {str(state.get('top_headline', 'None'))[:100]}
- Days to earnings: {state.get('days_to_earnings', 999)}
- Signal agent: {state.get('signal_gemini', 'N/A')} (conf={state.get('signal_confidence', 0)}, RR={state.get('signal_risk_reward', 'N/A')})
- Momentum: {state.get('signal_momentum', 'N/A')} | Trend strength: {state.get('signal_trend', 'N/A')}

Return ONLY this JSON with no markdown fences, no explanation:
{{
  "gemini_signal": "BUY" or "SELL" or "HOLD" or "INSUFFICIENT_DATA",
  "gemini_confidence": <integer 0-100>,
  "gemini_key_insight": "<single concise sentence — the one most important reason for your signal, grounded in your TA-125 expertise>",
  "gemini_risk_flags": ["<risk1>", "<risk2>"]
}}
If data is insufficient to form a view, use "INSUFFICIENT_DATA" and confidence 0."""

    return _call_gemini(prompt, _SCAN_FALLBACK, label=f"scan/{symbol}")


# ─────────────────────────────────────────────────────────────────────────────
# Use Case 2: News sentiment for open positions   (~50 calls/day)
# ─────────────────────────────────────────────────────────────────────────────

_NEWS_FALLBACK: dict = {
    "recommendation": "HOLD",
    "risk_level":     "low",
    "reasoning":      "Gemini unavailable — holding position by default.",
}


def gemini_news_sentiment(
    symbol: str,
    headline: str,
    position_side: str = "LONG",
) -> dict:
    """
    Deep-reads a news headline in the context of an open position.
    Replaces the keyword-only news_agent with nuanced AI understanding.
    Called from auto_trader._run_news_check() when a NEW headline is detected.

    Args:
        symbol        : ticker (e.g. "NVDA")
        headline      : the new headline string
        position_side : "LONG" or "SHORT" (direction of the open position)

    Returns:
        recommendation : "HOLD" | "ADD" | "EXIT_HALF" | "EXIT_ALL"
        risk_level     : "low" | "medium" | "high" | "extreme"
        reasoning      : 1–2 sentence explanation
    """
    prompt = f"""{_build_role()}
---
TASK: News Sentiment — you are monitoring an open {position_side} position in TASE-listed {symbol}.

A new headline just appeared:
"{headline}"

Apply your TA-125 expertise: weigh Israeli regulatory context (ISA actions, gas royalties, Bank of Israel statements), geopolitical risk premiums, and TASE liquidity dynamics before assessing the threat level to this {position_side} position. Run your internal Synthetic Backtest — recall how similar headlines moved TA-125 stocks historically and calibrate your recommendation accordingly.

Return ONLY this JSON with no markdown fences, no explanation:
{{
  "recommendation": "HOLD" or "ADD" or "EXIT_HALF" or "EXIT_ALL",
  "risk_level": "low" or "medium" or "high" or "extreme",
  "reasoning": "<one to two sentences — cite the specific TASE or macro factor driving your recommendation>"
}}

Decision rules:
- EXIT_ALL  → catastrophic immediate risk (ISA trading halt, fraud confirmed, severe earnings miss + guidance cut, geopolitical escalation directly targeting the company)
- EXIT_HALF → clearly negative catalyst directly affecting {symbol}'s fundamentals or Israeli-market-specific regulatory action
- ADD       → strongly positive surprise not yet priced in (BoI rate cut, M&A, major contract win)
- HOLD      → ambiguous, already-known, minor, or broad-market noise
- extreme risk → imminent insolvency, ISA enforcement, or security event that could halt TASE trading"""

    return _call_gemini(prompt, _NEWS_FALLBACK, label=f"news/{symbol}")


# ─────────────────────────────────────────────────────────────────────────────
# Use Case 1: Final trade gate before execution   (~3 calls/day)
# ─────────────────────────────────────────────────────────────────────────────

_GATE_FALLBACK: dict = {
    "approved":              True,
    "confidence_adjustment": 0,
    "reasoning":             "Gemini gate unavailable — proceeding with math-only approval.",
    "risk_flags":            [],
}


def gemini_trade_gate(
    symbol: str,
    state: dict,
    direction: str,
    master_confidence: int,
) -> dict:
    """
    Final AI sanity check before an order is submitted.
    Called from auto_trader._analyse() AFTER the master decision passes threshold
    and BEFORE run_consensus executes the order.

    Gemini reviews the full state and can VETO a trade if it sees obvious
    contradictions that the deterministic math agents cannot catch
    (e.g., earnings in 1 day, strong counter-trend sentiment, chop regime).

    Args:
        symbol            : ticker
        state             : full StateObject (including gemini_scan_overlay fields)
        direction         : "BUY" or "SELL"
        master_confidence : 0–100 master agent confidence score

    Returns:
        approved              : True to proceed, False to veto
        confidence_adjustment : integer −15 to +15 applied to master_confidence
        reasoning             : 1–2 sentence explanation
        risk_flags            : list of identified risk strings (may be empty)
    """
    prompt = f"""{_build_role()}
---
TASK: Final Trade Gate — review a proposed {direction} trade on TASE-listed {symbol} before order submission.

As the Distinguished Chair in algorithmic trading on the TASE, apply your full three-step pipeline:
1. Research: assess macro (BoI policy, USD/ILS, geopolitical premium), micro (company fundamentals), and technical confluence.
2. Synthetic Backtest: recall a comparable historical TASE setup — did a similar signal succeed or fail? What was the key differentiator?
3. Evolved Thesis: integrate the backtest lesson into your VETO/APPROVE decision below.

Proposed trade summary:
- Direction          : {direction}
- Price (ILS)        : {state.get('price', 0):.2f}
- Master confidence  : {master_confidence}%
- Quant signal       : {state.get('signal', 'N/A')}
- Ensemble agreement : {state.get('ensemble_agreement', 'N/A')}
- Market regime      : {state.get('market_regime', 'N/A')} | ADX: {state.get('adx', 0):.1f}
- Vision veto        : {state.get('vision_veto', False)} | Pattern: {state.get('chart_pattern', 'N/A')}
- Resistance nearby  : {state.get('resistance_nearby', False)}
- Sentiment          : {state.get('sentiment_direction', 'N/A')} (score={state.get('sentiment_score', 0):.3f})
- Impact category    : {state.get('impact_category', 'N/A')} | Catalyst: {state.get('catalyst_type', 'N/A')}
- Days to earnings   : {state.get('days_to_earnings', 999)}
- Top headline       : {str(state.get('top_headline', 'None'))[:80]}
- Relative volume    : {state.get('relative_volume', 1.0):.2f}×
- Price vs VWAP      : {state.get('price_vs_vwap', 'N/A')}
- Scan overlay       : signal={state.get('gemini_signal', 'N/A')} conf={state.get('gemini_confidence', 0)}%
- Key insight        : {state.get('gemini_key_insight', 'N/A')}
- Risk flags         : {state.get('gemini_risk_flags', [])}

VETO criteria (set approved=false if ANY apply):
1. Earnings within 2 days — binary surprise risk, especially acute on TASE due to thin post-earnings liquidity
2. vision_veto=True AND chart pattern is strongly counter-directional to {direction}
3. Market regime is VOLATILE AND ADX < 20 — no trend edge; TASE thin-book amplifies whipsaws
4. Sentiment strongly counter to {direction} (BEARISH + BUY or BULLISH + SELL) AND impact=MAJOR
5. resistance_nearby=True for BUY with low volume — TASE resistance levels are stickier due to lower float
6. Geopolitical escalation risk (conflict, sanctions, credit-rating downgrade) visible in headline or flags

Confidence adjustment rules:
- Strong confirming scan overlay + TASE macro tailwind → +5 to +10
- Minor contradictions or thin TASE liquidity concern → −5 to −10
- Earnings 3–5 days out or elevated geopolitical risk → −5
- All signals aligned, healthy volume, benign macro → 0

Return ONLY this JSON with no markdown fences, no explanation:
{{
  "approved": true or false,
  "confidence_adjustment": <integer from -15 to +15>,
  "reasoning": "<one to two sentences — reference the specific TASE factor or backtest lesson that drove your decision>",
  "risk_flags": ["<flag1>", "<flag2>"]
}}"""

    return _call_gemini(prompt, _GATE_FALLBACK, label=f"gate/{symbol}")


# ─────────────────────────────────────────────────────────────────────────────
# Use Case 3: Daily 9:00 AM morning brief   (1 call/day)
# ─────────────────────────────────────────────────────────────────────────────

_BRIEF_FALLBACK: dict = {
    "market_outlook": "NEUTRAL",
    "key_macro_risks": [],
    "sector_notes":    "Gemini morning brief unavailable.",
    "symbol_notes":    {},
}


def gemini_morning_brief(watchlist: list[str]) -> dict:
    """
    Pre-market AI overview called once at 9:00 AM ET each trading day.
    Results are logged and emailed via auto_trader._run_gap_scan().

    Args:
        watchlist : list of tickers currently being monitored

    Returns:
        market_outlook  : "BULLISH" | "BEARISH" | "NEUTRAL" | "VOLATILE"
        key_macro_risks : list of up to 3 macro risk strings
        sector_notes    : 1–2 sentence sector dynamics summary
        symbol_notes    : dict mapping ticker → 1-sentence pre-market note
    """
    from datetime import date as _date
    today       = _date.today().strftime("%A, %B %d, %Y")
    syms_str    = ", ".join(watchlist)
    first_sym   = watchlist[0] if watchlist else "AAPL"

    prompt = f"""{_build_role()}
---
TASK: Morning Brief — pre-market analysis for the TA-125 (TASE). Today is {today}.

Active watchlist: {syms_str}

Execute your full three-step pipeline for this morning brief:

## Step 1 — Academic Research & Market Context
Assess the current TA-125 macro environment: Bank of Israel stance, USD/ILS and EUR/ILS trend, geopolitical risk premium, and sector dynamics (tech, real estate, energy/gas, financials) within the Israeli market. Note any scheduled announcements (BoI decision, US Fed spillover, earnings on TASE) that could move the index today.

## Step 2 — Internal Backtest
Identify the most analogous past TASE trading session to today's setup. State what happened then, and the single most important lesson it teaches about today's likely price action.

## Step 3 — Evolved Morning Thesis
Synthesize Steps 1 & 2 into a single directional outlook for the TA-125 today, and provide one actionable pre-market note per watchlist symbol informed by that thesis.

Return ONLY this JSON with no markdown fences, no explanation:
{{
  "market_outlook": "BULLISH" or "BEARISH" or "NEUTRAL" or "VOLATILE",
  "key_macro_risks": ["<TA-125 macro risk 1>", "<TA-125 macro risk 2>", "<TA-125 macro risk 3>"],
  "sector_notes": "<1–2 sentences on TASE sector dynamics — cite BoI, USD/ILS, geopolitical factors, or index-level technical levels>",
  "symbol_notes": {{
    "{first_sym}": "<1-sentence note — tie to your evolved morning thesis>",
    "<symbol2>": "<1-sentence note>",
    "<add all watchlist symbols>": "..."
  }}
}}

Rules:
- Each symbol note must be ONE actionable sentence referencing today's TA-125 context
- Include ALL {len(watchlist)} symbols in symbol_notes
- key_macro_risks must be the TOP 3 factors specific to the TASE and Israeli economy today
- If a symbol has no company-specific news, note how it correlates to BoI policy or USD/ILS moves"""

    return _call_gemini(prompt, _BRIEF_FALLBACK, label="morning_brief")


# ─────────────────────────────────────────────────────────────────────────────
# Budget status utility  (used by main.py /status endpoint)
# ─────────────────────────────────────────────────────────────────────────────

def _build_role() -> str:
    """Return the shared elite TA-125 professor system role prepended to every prompt."""
    return """You are an elite Professor of Quantitative Finance and Investment Management, holding the equivalent of a "5th-degree" (Distinguished Chair) university mastery in algorithmic trading and market microstructure. Your specific domain of expertise is the Tel Aviv Stock Exchange, with a hyper-focus on the TA-125 Index (TLV:TA125).

Your ultimate directive is to obsessively chase perfection in stock market prediction, recognizing that while 100% accuracy is impossible, continuous evolution through self-correction is mandatory.

SYSTEM OBJECTIVES & OPERATIONAL PIPELINE:

1. DEEP DIVE & RESEARCH
When analyzing a TA-125 stock or the index as a whole, perform a multi-tiered evaluation:
  * Macro: Bank of Israel interest rate decisions, USD/ILS or EUR/ILS currency impacts, geopolitical risk premiums.
  * Micro/Fundamental: P/E ratios, debt-to-equity, cash flow, and regulatory shifts specific to Israeli markets (e.g., gas royalties, tech sector trends, real estate mandates).
  * Technical/Quantitative: Momentum indicators, volume profiles, and liquidity constraints unique to the TASE.

2. MANDATORY SELF-BACKTESTING & SIMULATION LOOP
Before finalizing any live prediction or strategy, perform an internal "Synthetic Backtest":
  * Select a historical date or a hypothetical past market regime relevant to the current setup.
  * Formulate a "blind" prediction for Day T+1 based strictly on data available up to Day T.
  * Simulate or retrieve the actual outcome of Day T+1.
  * Self-Evaluate: Calculate your simulated error metric (Directional Accuracy, MSE, Max Drawdown).
  * Critique: Write a brutally honest post-mortem explaining why the prediction succeeded or failed, identifying any blind spots (e.g., "overemphasized volume, ignored macro sentiment").

3. ADAPTIVE LEARNING & EVOLUTION
Based on the self-backtest critique, adjust your weights, underlying assumptions, and analytical lens.
Document this evolution as a "Learned Insight" that will strictly govern your final prediction.

You must embody this professor-level rigor in every signal, gate decision, and market brief you produce.
"""


def get_budget_status() -> dict:
    """Returns today's Gemini API call count and remaining budget. Thread-safe."""
    today = date.today().isoformat()
    with _budget_lock:
        if _budget_date != today:
            return {"date": today, "used": 0, "remaining": _DAILY_LIMIT, "limit": _DAILY_LIMIT}
        return {
            "date":      _budget_date,
            "used":      _daily_count,
            "remaining": max(0, _DAILY_LIMIT - _daily_count),
            "limit":     _DAILY_LIMIT,
        }
