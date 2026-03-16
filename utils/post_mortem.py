"""
Post-Mortem Engine — The "Feedback Loop."

Writes a structured record after every trade. The self-review uses
pure Python statistics — no AI API required.

Self-learning cycle (fires automatically every 10 closed trades):
  - write_post_mortem()       → called at trade OPEN (by supervisor)
  - update_trade_result()     → called at trade CLOSE (WIN / LOSS)
  - auto_review_if_ready(10)  → checks if 10+ new trades since last review
  - weekly_self_review()      → adjusts all thresholds when triggered

Thresholds adjusted by the review:
  sentiment_threshold   — tighten when high-sentiment trades lose
  confidence_threshold  — raise if win-rate < 45%, lower if > 65% (range 55–85)
  avoid_patterns        — chart patterns dominant in losses (≥2 occurrences)
  avoid_symbols         — symbols with ≥2 losses AND win-rate < 40%
  best_symbols          — symbols with win-rate ≥ 65% (≥3 closed trades)
"""

import json
import logging
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("post_mortem")

STATE_DIR = Path(__file__).parent.parent / "state"
TRADE_LOG = STATE_DIR / "trade_log.json"
SYSTEM_INSTRUCTIONS = STATE_DIR / "system_instructions.json"


# ── I/O helpers ───────────────────────────────────────────────────────────────

def _load_log() -> list[dict]:
    if not TRADE_LOG.exists():
        return []
    with open(TRADE_LOG, "r") as f:
        return json.load(f)


def _save_log(records: list[dict]) -> None:
    STATE_DIR.mkdir(exist_ok=True)
    with open(TRADE_LOG, "w") as f:
        json.dump(records, f, indent=2)


# ── Trade entry ───────────────────────────────────────────────────────────────

def write_post_mortem(state: dict, outcome: dict) -> None:
    """Writes an OPEN record at trade entry. Closed via update_trade_result()."""
    record = {
        "id":                  len(_load_log()) + 1,
        "timestamp":           datetime.now(timezone.utc).isoformat(),
        "symbol":              state["symbol"],
        "entry_price":         state["price"],
        "signal":              state["signal"],
        "bb_pvalue":           state.get("bb_pvalue"),
        "rsi_pvalue":          state.get("rsi_pvalue"),
        "ensemble_agreement":  state.get("ensemble_agreement"),
        "sentiment_score":     state.get("sentiment_score"),
        "sentiment_direction": state.get("sentiment_direction"),
        "impact_category":     state.get("impact_category"),
        "chart_pattern":       state.get("chart_pattern"),
        "vision_veto":         state.get("vision_veto"),
        "signal_confidence":   state.get("signal_confidence", 0),
        "exit_price":          outcome.get("exit_price"),
        "pnl_dollars":         outcome.get("pnl_dollars"),
        "pnl_pct":             outcome.get("pnl_pct"),
        "result":              outcome.get("result", "UNKNOWN"),
    }
    records = _load_log()
    records.append(record)
    _save_log(records)
    logger.info(f"Post-mortem written: {state['symbol']} OPEN @ ${state['price']:.2f}")


# ── Trade close ───────────────────────────────────────────────────────────────

def update_trade_result(symbol: str, exit_price: float, result: str) -> None:
    """
    Updates the most recent OPEN record for a symbol with the closing outcome.

    Parameters
    ----------
    symbol     : Ticker symbol (e.g. "AAPL")
    exit_price : Actual price the position was exited at
    result     : "WIN" or "LOSS"
    """
    records = _load_log()
    updated = False
    for record in reversed(records):
        if record.get("symbol") == symbol and record.get("result") == "OPEN":
            entry_price      = float(record.get("entry_price") or exit_price)
            pnl_pct          = round((exit_price - entry_price) / entry_price, 4) if entry_price else 0.0
            pnl_dollars      = round(exit_price - entry_price, 4)
            record["exit_price"]  = round(exit_price, 4)
            record["pnl_pct"]     = pnl_pct
            record["pnl_dollars"] = pnl_dollars
            record["result"]      = result
            record["closed_at"]   = datetime.now(timezone.utc).isoformat()
            updated = True
            logger.info(
                f"Trade result updated: {symbol} → {result} | "
                f"exit=${exit_price:.2f} | P&L={pnl_pct:.2%}"
            )
            break

    if not updated:
        logger.warning(
            f"update_trade_result: no OPEN record found for {symbol}. "
            f"Writing stand-alone {result} record."
        )
        records.append({
            "id":         len(records) + 1,
            "timestamp":  datetime.now(timezone.utc).isoformat(),
            "symbol":     symbol,
            "exit_price": round(exit_price, 4),
            "result":     result,
            "closed_at":  datetime.now(timezone.utc).isoformat(),
        })

    _save_log(records)


# ── Auto-trigger ──────────────────────────────────────────────────────────────

def auto_review_if_ready(min_trades: int = 10) -> str | None:
    """
    Triggers weekly_self_review() when min_trades or more new closed trades
    have accumulated since the last review.

    Returns the review summary string if triggered, else None.
    """
    records      = _load_log()
    closed       = [r for r in records if r.get("result") in ("WIN", "LOSS")]
    instructions = load_system_instructions()
    last_count   = instructions.get("last_review_trade_count", 0)
    new_trades   = len(closed) - last_count

    if new_trades >= min_trades:
        logger.info(
            f"Auto-review triggered: {new_trades} new closed trades "
            f"(total={len(closed)}, last_review_at={last_count})."
        )
        return weekly_self_review()
    return None


# ── Core review ───────────────────────────────────────────────────────────────

def weekly_self_review() -> str:
    """
    Pure Python statistical review — no AI API needed.

    Analyses the full closed-trade history and writes updated thresholds
    to state/system_instructions.json.

    Adjustments made:
      sentiment_threshold  : +0.05 if losses had higher avg sentiment,
                             -0.05 if win-rate > 65%  (range 0.20–0.70)
      confidence_threshold : +5    if win-rate < 45%,
                             -3    if win-rate > 65%  (range 55–85)
      avoid_patterns       : patterns appearing ≥2× in losses
      avoid_symbols        : symbols with wr < 40% AND losses ≥ 2
      best_symbols         : symbols with wr ≥ 65% AND ≥3 closed trades
    """
    records = _load_log()
    closed  = [r for r in records if r.get("result") in ("WIN", "LOSS")]

    if len(closed) < 10:
        msg = f"Not enough closed trades for review (need 10, have {len(closed)})."
        logger.info(msg)
        return msg

    wins     = [r for r in closed if r["result"] == "WIN"]
    losses   = [r for r in closed if r["result"] == "LOSS"]
    win_rate = len(wins) / len(closed)

    current = load_system_instructions()

    # ── Sentiment threshold ───────────────────────────────────────────────────
    avg_sent_win  = sum(r.get("sentiment_score", 0) or 0 for r in wins)  / max(len(wins),   1)
    avg_sent_loss = sum(r.get("sentiment_score", 0) or 0 for r in losses) / max(len(losses), 1)
    cur_sent = current.get("sentiment_threshold", 0.30)

    if win_rate < 0.50 and avg_sent_loss > avg_sent_win:
        new_sent = min(cur_sent + 0.05, 0.70)
        sent_action = (
            f"Raised sentiment_threshold {cur_sent:.2f} → {new_sent:.2f} "
            f"(losses had higher avg sentiment: {avg_sent_loss:.3f} vs {avg_sent_win:.3f})"
        )
    elif win_rate > 0.65 and cur_sent > 0.20:
        new_sent = max(cur_sent - 0.05, 0.20)
        sent_action = (
            f"Lowered sentiment_threshold {cur_sent:.2f} → {new_sent:.2f} "
            f"(system performing well at {win_rate:.1%})"
        )
    else:
        new_sent = cur_sent
        sent_action = f"sentiment_threshold unchanged at {cur_sent:.2f}"

    # ── Confidence threshold ──────────────────────────────────────────────────
    cur_conf = current.get("confidence_threshold", 65)

    if win_rate < 0.45:
        new_conf = min(cur_conf + 5, 85)
        conf_action = (
            f"Raised confidence_threshold {cur_conf} → {new_conf} "
            f"(win-rate {win_rate:.1%} < 45%)"
        )
    elif win_rate > 0.65:
        new_conf = max(cur_conf - 3, 55)
        conf_action = (
            f"Lowered confidence_threshold {cur_conf} → {new_conf} "
            f"(win-rate {win_rate:.1%} > 65%)"
        )
    else:
        new_conf = cur_conf
        conf_action = f"confidence_threshold unchanged at {cur_conf}"

    # ── Pattern analysis ──────────────────────────────────────────────────────
    loss_patterns = Counter(
        r.get("chart_pattern", "UNKNOWN") for r in losses
    )
    avoid_patterns = [
        p for p, count in loss_patterns.items()
        if count >= 2 and p and p != "UNKNOWN"
    ]

    # ── Per-symbol win-rate analysis ──────────────────────────────────────────
    all_symbols   = {r.get("symbol") for r in closed if r.get("symbol")}
    avoid_symbols = []
    best_symbols  = []

    for sym in sorted(all_symbols):
        sym_trades = [r for r in closed if r.get("symbol") == sym]
        sym_wins   = [r for r in sym_trades if r["result"] == "WIN"]
        sym_losses = [r for r in sym_trades if r["result"] == "LOSS"]
        sym_wr     = len(sym_wins) / len(sym_trades)

        if sym_wr < 0.40 and len(sym_losses) >= 2:
            avoid_symbols.append(sym)
        if sym_wr >= 0.65 and len(sym_trades) >= 3:
            best_symbols.append(sym)

    # ── Avg P&L stats ─────────────────────────────────────────────────────────
    avg_win_pnl  = sum(r.get("pnl_pct", 0) or 0 for r in wins)  / max(len(wins),   1)
    avg_loss_pnl = sum(r.get("pnl_pct", 0) or 0 for r in losses) / max(len(losses), 1)

    # ── Write updated instructions ────────────────────────────────────────────
    new_instructions = {
        "sentiment_threshold":      round(new_sent, 2),
        "confidence_threshold":     new_conf,
        "min_ensemble_agreement":   True,
        "avoid_patterns":           avoid_patterns,
        "avoid_symbols":            avoid_symbols,
        "best_symbols":             best_symbols,
        "last_review_trade_count":  len(closed),
        "additional_rules": [
            f"Win rate over last {len(closed)} trades: {win_rate:.1%}",
            f"Avg P&L — wins: {avg_win_pnl:.2%} | losses: {avg_loss_pnl:.2%}",
            sent_action,
            conf_action,
        ],
    }

    STATE_DIR.mkdir(exist_ok=True)
    with open(SYSTEM_INSTRUCTIONS, "w") as f:
        json.dump(new_instructions, f, indent=2)

    summary = (
        f"Self-Review Complete — {len(closed)} closed trades\n"
        f"Win rate: {win_rate:.1%} ({len(wins)}W / {len(losses)}L)\n"
        f"Avg P&L — wins: {avg_win_pnl:.2%} | losses: {avg_loss_pnl:.2%}\n"
        f"Patterns to avoid: {avoid_patterns or 'none'}\n"
        f"Symbols to avoid:  {avoid_symbols  or 'none'}\n"
        f"Best symbols:      {best_symbols   or 'none'}\n"
        f"Action 1: {sent_action}\n"
        f"Action 2: {conf_action}"
    )
    logger.info(f"Self-review complete:\n{summary}")
    return summary


# ── Instructions loader ───────────────────────────────────────────────────────

def load_system_instructions() -> dict:
    """
    Loads runtime-adjustable parameters from state/system_instructions.json.
    Back-fills any missing keys for backward compatibility.
    """
    defaults = {
        "sentiment_threshold":      0.30,
        "confidence_threshold":     65,
        "min_ensemble_agreement":   True,
        "avoid_patterns":           [],
        "avoid_symbols":            [],
        "best_symbols":             [],
        "last_review_trade_count":  0,
        "additional_rules":         [],
    }
    if not SYSTEM_INSTRUCTIONS.exists():
        return defaults

    with open(SYSTEM_INSTRUCTIONS, "r") as f:
        data = json.load(f)

    # Back-fill any keys added in newer versions
    for key, val in defaults.items():
        data.setdefault(key, val)

    return data
