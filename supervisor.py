"""
Supervisor Agent — The "High Command."

Collects reports from all four specialists, applies weighted consensus logic,
and only executes a trade when ALL conditions are green. After every trade,
it writes a post-mortem. On startup it reads self-evolved instructions from
system_instructions.json and applies them as dynamic thresholds.
"""

import logging
from execution.alpaca_broker import submit_order, get_position
from execution.risk_manager import calculate_position_size
from agents.position_agent import run_position_sizing
from agents.meta_agent import run_meta_sizing
from utils.post_mortem import write_post_mortem, load_system_instructions

logger = logging.getLogger("supervisor")

# Minimum composite confidence required to execute a trade.
# Trades with confidence below this value are blocked regardless of gate results.
CONFIDENCE_THRESHOLD = 0.65


def _compute_confidence(state: dict, instructions: dict) -> float:
    """
    Calculates a weighted confidence score [0.0 – 1.0].

    Weights:
    - Quant green_light + low p-values:      30%
    - Positive sentiment:                    25%
    - Vision pattern (no veto):              25%
    - Signal agent confidence (0-100 → 0-1): 20%
    """
    quant_score = 0.0
    if state["green_light"]:
        avg_p = (state["bb_pvalue"] + state["rsi_pvalue"]) / 2
        quant_score = max(0.0, 1.0 - avg_p * 100)

    sentiment_score = max(0.0, state["sentiment_score"])

    vision_score = 0.0
    if not state["vision_veto"]:
        vision_score = 1.0
        if state["resistance_nearby"]:
            vision_score = 0.5

    # Signal agent confidence: normalize 0-100 → 0-1
    # Only count it when signal agrees with quant direction
    raw_signal_conf = state.get("signal_confidence", 0) / 100.0
    quant_dir  = state.get("signal", "HOLD")
    gemini_dir = state.get("signal_gemini", "HOLD")
    signal_score = raw_signal_conf if quant_dir == gemini_dir else raw_signal_conf * 0.3

    return 0.30 * quant_score + 0.25 * sentiment_score + 0.25 * vision_score + 0.20 * signal_score


def run_consensus(state: dict, check_only: bool = False) -> dict:
    """
    Evaluates the Triad of Certainty.

    Parameters
    ----------
    state      : Full pipeline state from build_state()
    check_only : If True, run all gates and compute confidence but do NOT
                 submit an order or run position/meta agents. Useful as a
                 fast pre-check before running the master decision engine.

    Returns a decision dict:
    - approved: bool
    - confidence: float
    - reason: str
    - action_taken: str  ("BUY_EXECUTED" | "SELL_EXECUTED" | "NO_ACTION" | "GATE_PASSED")
    - order: dict | None
    """
    instructions = load_system_instructions()
    sentiment_threshold = instructions.get("sentiment_threshold", 0.3)
    avoid_patterns      = instructions.get("avoid_patterns", [])
    avoid_symbols       = instructions.get("avoid_symbols",  [])

    # ── Gate 1: Quant must give green light ──────────────────────────────────
    if not state["green_light"]:
        return _no_action(state, "Quant: No statistical edge (p-value too high)", instructions)

    # ── Gate 2: Ensemble must agree ───────────────────────────────────────────
    if not state.get("ensemble_agreement"):
        return _no_action(state, "Quant: BB and RSI strategies disagree", instructions)

    # ── Gate 3: Sentiment must be above threshold ─────────────────────────────
    if state["sentiment_score"] < sentiment_threshold:
        return _no_action(
            state,
            f"Sentiment: score {state['sentiment_score']:.2f} below threshold {sentiment_threshold}",
            instructions,
        )

    # ── Gate 4: Vision must not veto ──────────────────────────────────────────
    if state["vision_veto"]:
        return _no_action(state, "Vision: chart pattern vetoes this trade", instructions)

    # ── Gate 5: Signal agent hard conflict ───────────────────────────────────
    # If Gemini's signal strongly disagrees AND confidence is high, block the trade
    gemini_signal = state.get("signal_gemini", "HOLD")
    gemini_conf   = state.get("signal_confidence", 0)
    quant_signal  = state["signal"]
    if (
        gemini_signal not in ("HOLD", "INSUFFICIENT_DATA")
        and gemini_signal != quant_signal
        and gemini_conf >= 70
    ):
        return _no_action(
            state,
            f"Signal conflict: Quant={quant_signal} vs Gemini={gemini_signal} "
            f"(confidence={gemini_conf})",
            instructions,
        )

    # ── Gate 6: Self-evolved pattern avoidance ────────────────────────────────
    if state.get("chart_pattern") in avoid_patterns:
        return _no_action(
            state,
            f"Self-evolved rule: avoiding pattern '{state['chart_pattern']}'",
            instructions,
        )

    # ── Gate 7: Self-evolved symbol avoidance ─────────────────────────────────
    if state.get("symbol") in avoid_symbols:
        return _no_action(
            state,
            f"Self-evolved rule: avoiding symbol '{state['symbol']}' (poor historical win-rate)",
            instructions,
        )

    # ── Gate 8: Market Regime — block choppy/trendless markets ───────────────
    if state.get("market_regime") == "CHOPPY":
        return _no_action(
            state,
            f"Market Regime: ADX={state.get('adx', 0):.1f} < 20 — "
            f"choppy market, signals are noise",
            instructions,
        )

    # ── Gate 9: Volume Gate — require above-average volume for conviction ─────
    rel_vol = state.get("relative_volume", 1.0)
    if rel_vol < 1.0:
        return _no_action(
            state,
            f"Volume Gate: relative_volume={rel_vol:.2f} — "
            f"below-average volume, move likely a fake-out",
            instructions,
        )

    # ── Gate 10: VWAP Gate — price must be above VWAP for BUY entries ────────
    if state.get("signal") == "BUY" and state.get("price_vs_vwap") == "BELOW":
        return _no_action(
            state,
            f"VWAP Gate: price ${state.get('price', 0):.2f} is below "
            f"VWAP ${state.get('vwap', 0):.2f} — against institutional flow",
            instructions,
        )

    # ── Gate 11: Pre-Earnings Blackout — skip within 3 days of earnings ───────
    days_to_earnings = state.get("days_to_earnings", 999)
    if 0 <= days_to_earnings <= 3:
        return _no_action(
            state,
            f"Pre-Earnings Blackout: {days_to_earnings} day(s) to earnings — "
            f"unpredictable gap risk bypasses all stops",
            instructions,
        )

    # ── All gates passed — compute confidence ────────────────────────────────
    confidence = _compute_confidence(state, instructions)
    signal = state["signal"]

    if signal not in ("BUY", "SELL"):
        return _no_action(state, "Signal is HOLD — no action.", instructions)

    # ── Confidence threshold gate (dynamic — self-tuned) ──────────────────────
    # instructions stores threshold as integer 0-100 (e.g. 65); convert to 0-1 float
    dyn_conf_threshold = instructions.get("confidence_threshold", int(CONFIDENCE_THRESHOLD * 100)) / 100.0
    if confidence < dyn_conf_threshold:
        return _no_action(
            state,
            f"Confidence {confidence:.2f} below threshold {dyn_conf_threshold:.2f}",
            instructions,
        )

    # ── check_only: return gate result without executing ─────────────────────
    if check_only:
        logger.info(
            f"GATE CHECK PASSED [{state['symbol']}]: {signal} "
            f"| confidence={confidence:.2f} — awaiting master decision."
        )
        return {
            "approved":    True,
            "confidence":  round(confidence, 3),
            "reason":      "All gates passed (check_only — no order submitted)",
            "action_taken": "GATE_PASSED",
            "order":       None,
            "signal":      signal,
        }

    # Check if already holding this symbol to avoid double-entry
    existing = get_position(state["symbol"])
    side = signal.lower()
    if existing and side == "buy":
        return _no_action(
            state,
            f"Already holding {existing['qty']} shares of {state['symbol']}",
            instructions,
        )

    # ── Step 1: ATR + S/R levels (stop, targets, R:R) ────────────────────────
    try:
        position_plan = run_position_sizing(
            symbol=state["symbol"],
            entry_price=state["price"],
            state=state,
        )
    except Exception as exc:
        logger.warning(f"Position agent failed, using defaults: {exc}")
        position_plan = {}

    stop_loss_price = position_plan.get("stop_loss_price")

    # ── Step 2: Kelly/composite meta-sizing (final qty) ───────────────────────
    try:
        meta_plan = run_meta_sizing(
            symbol=state["symbol"],
            state=state,
            entry_price=state["price"],
            stop_loss=stop_loss_price,
        )
        qty = meta_plan["position_size_calculation"]["position_value"]

        # Apply LOW_RR penalty from position_plan on top of Kelly sizing
        if position_plan.get("confidence_adjustment") in ("RR_LOW", "RR_VERY_LOW"):
            qty = max(1.0, qty / 2)
            logger.warning(
                f"[{state['symbol']}] {position_plan['confidence_adjustment']} — "
                f"halving Kelly notional to ${qty:.2f}. {position_plan.get('notes', '')}"
            )
    except Exception as exc:
        logger.warning(f"Meta agent failed, falling back to simple sizing: {exc}")
        meta_plan = {}
        qty = calculate_position_size(state["price"])

    logger.info(
        f"CONSENSUS REACHED [{state['symbol']}]: {signal} ${qty:.2f} notional "
        f"@ ${state['price']:.2f} | confidence={confidence:.2f} | "
        f"SL={position_plan.get('stop_loss_price')} "
        f"TP1={position_plan.get('target_1_price')}"
    )

    order = submit_order(state["symbol"], qty, side)

    # Send email notification
    try:
        from utils.notifier import notify_sync, trade_email
        subj, body = trade_email(state["symbol"], signal, qty, state["price"], confidence)
        notify_sync(subj, body)
    except Exception:
        pass

    meta_calc = meta_plan.get("position_size_calculation", {})
    result = {
        "approved":               True,
        "confidence":             round(confidence, 3),
        "reason":                 "Quad Certainty: all gates passed",
        "action_taken":           f"{signal}_EXECUTED",
        "order":                  order,
        "qty":                    qty,
        # Position plan (ATR-adjusted levels)
        "entry_price":            position_plan.get("entry_price",       state["price"]),
        "stop_loss":              position_plan.get("stop_loss_price"),
        "target_1":               position_plan.get("target_1_price"),
        "target_2":               position_plan.get("target_2_price"),
        "risk_reward_t1":         position_plan.get("risk_reward_ratio_target1"),
        "risk_reward_t2":         position_plan.get("risk_reward_ratio_target2"),
        "exit_plan":              position_plan.get("exit_plan", []),
        "rr_flag":                position_plan.get("confidence_adjustment", "OK"),
        "position_notes":         position_plan.get("notes", ""),
        # Meta / Kelly fields
        "composite_confidence":   meta_plan.get("composite_confidence"),
        "kelly_criterion":        meta_plan.get("kelly_criterion"),
        "kelly_adjusted_pct":     meta_plan.get("kelly_adjusted_position"),
        "risk_usd":               meta_calc.get("risk_per_trade_usd"),
        "position_value":         meta_calc.get("position_value"),
        "position_size_pct":      meta_plan.get("position_size_pct"),
        "sizing_recommendation":  meta_plan.get("sizing_recommendation"),
        "confidence_summary":     meta_plan.get("confidence_summary"),
        "signal_reasoning":       state.get("signal_reasoning", ""),
    }

    # Write post-mortem entry at trade open (outcome to be updated on close)
    write_post_mortem(state, {"result": "OPEN", "exit_price": None, "pnl_dollars": None, "pnl_pct": None})

    return result


def _no_action(state: dict, reason: str, instructions: dict) -> dict:
    logger.info(f"NO ACTION [{state['symbol']}]: {reason}")
    return {
        "approved": False,
        "confidence": _compute_confidence(state, instructions),
        "reason": reason,
        "action_taken": "NO_ACTION",
        "order": None,
    }
