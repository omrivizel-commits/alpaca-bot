"""
Data Pipeline — The "Nervous System."

Builds the central StateObject by running all four agents concurrently
via asyncio. The StateObject is the single source of truth passed to the Supervisor.
"""

import asyncio
from datetime import datetime, timezone

from agents.quant_agent import run_quant
from agents.sentiment_agent import analyze_sentiment
from agents.vision_agent import analyze_chart
from agents.signal_agent import run_signal
from data.market_data import get_current_price, get_days_to_earnings


async def _run_quant_async(symbol: str) -> dict:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, run_quant, symbol)


async def _run_sentiment_async(symbol: str, market_cap_billions: float) -> dict:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None, analyze_sentiment, symbol, market_cap_billions
    )


async def _run_vision_async(symbol: str, quant_signal: str, price: float) -> dict:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, analyze_chart, symbol, quant_signal, price)


async def _run_signal_async(symbol: str) -> dict:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, run_signal, symbol)


async def build_state(symbol: str, market_cap_billions: float = None) -> dict:
    """
    Assembles the full StateObject by running Quant, Sentiment, Vision,
    and Signal agents in parallel. Returns a single dict the Supervisor reads.
    """
    # Step 1: Get current price (fast, blocking is fine here)
    loop = asyncio.get_running_loop()
    price = await loop.run_in_executor(None, get_current_price, symbol)

    # Step 2: Quant runs first (vision needs its signal for the veto check)
    quant_task = asyncio.create_task(_run_quant_async(symbol))
    quant_report = await quant_task

    # Step 3: Sentiment, Vision, Signal, and Earnings run in parallel
    sentiment_task = asyncio.create_task(
        _run_sentiment_async(symbol, market_cap_billions)
    )
    vision_task = asyncio.create_task(
        _run_vision_async(symbol, quant_report["signal"], price)
    )
    signal_task = asyncio.create_task(_run_signal_async(symbol))
    # run_in_executor returns a Future (not a coroutine), so pass it directly to
    # gather() — do NOT wrap in create_task(), which only accepts coroutines.
    earnings_future = loop.run_in_executor(None, get_days_to_earnings, symbol)

    sentiment_report, vision_report, signal_report, days_to_earnings = await asyncio.gather(
        sentiment_task, vision_task, signal_task, earnings_future
    )

    # ── Assemble the StateObject ──────────────────────────────────────────────
    state = {
        "symbol": symbol,
        "price": price,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        # Quant fields
        "signal":             quant_report["signal"],
        "green_light":        quant_report["green_light"],
        "bb_pvalue":          quant_report["bb_pvalue"],
        "rsi_pvalue":         quant_report["rsi_pvalue"],
        "ensemble_agreement": quant_report["ensemble_agreement"],
        "bb_signal":          quant_report["bb_signal"],
        "rsi_signal":         quant_report["rsi_signal"],
        # Market regime / ADX
        "adx":                quant_report.get("adx", 20.0),
        "market_regime":      quant_report.get("market_regime", "NEUTRAL"),
        # VWAP — compare live price to today's VWAP for institutional flow direction
        "vwap":               quant_report.get("vwap", 0.0),
        "price_vs_vwap": (
            "ABOVE" if quant_report.get("vwap", 0) and price > quant_report["vwap"]
            else "BELOW" if quant_report.get("vwap", 0) and price < quant_report["vwap"]
            else "NEUTRAL"
        ),
        # Volume filter
        "relative_volume":    quant_report.get("relative_volume", 1.0),
        # Earnings calendar (pre-earnings blackout gate)
        "days_to_earnings":   days_to_earnings if isinstance(days_to_earnings, int) else 999,
        # Sentiment fields
        "sentiment_score": sentiment_report["sentiment_score"],
        "sentiment_direction": sentiment_report["direction"],
        "relative_impact": sentiment_report["relative_impact"],
        "top_headline": sentiment_report["top_headline"],
        "impact_category": sentiment_report["impact_category"],
        "catalyst_type": sentiment_report.get("catalyst_type", "OTHER"),
        "already_priced_in": sentiment_report.get("already_priced_in", True),
        # Vision fields
        "chart_pattern": vision_report["pattern"],
        "nearest_support": vision_report["nearest_support"],
        "nearest_resistance": vision_report["nearest_resistance"],
        "resistance_nearby": vision_report["resistance_nearby"],
        "vision_veto": vision_report["vision_veto"],
        "vision_confidence": vision_report["vision_confidence"],
        "vision_reasoning": vision_report.get("vision_reasoning", ""),
        "candlestick_patterns": vision_report.get("candlestick_patterns", []),
        "trend": vision_report.get("trend", "SIDEWAYS"),
        "volume_confirmation": vision_report.get("volume_confirmation", "NEUTRAL"),
        # Signal agent fields
        "signal_gemini":       signal_report.get("signal", "INSUFFICIENT_DATA"),
        "signal_confidence":   signal_report.get("confidence", 0),
        "signal_entry":        signal_report.get("entry_price"),
        "signal_stop_loss":    signal_report.get("stop_loss"),
        "signal_take_profit_1": signal_report.get("take_profit_1"),
        "signal_take_profit_2": signal_report.get("take_profit_2"),
        "signal_risk_reward":  signal_report.get("risk_reward_ratio"),
        "signal_reasoning":    signal_report.get("reasoning", ""),
        "signal_trend":        signal_report.get("signal_strength", {}).get("trend", "neutral"),
        "signal_momentum":     signal_report.get("signal_strength", {}).get("momentum", "weak"),
        "signal_sr_position":  signal_report.get("signal_strength", {}).get("support_resistance", "midrange"),
    }

    return state
