"""
Auto-Trader — Autonomous trading loop.

Integration flow (per usage guide):
  Real-time feed → Technical (Prompt 1) → News (Prompt 2) →
  Confidence (Prompt 4) → Master Decision (Prompt 8) →
  Entry/Exit rules (Prompt 3) → Execute → Position management

Loop schedule:
  _scan_loop              every 5 min   — technical scan on full watchlist
  _news_loop              every 30 sec  — news check on open positions
  _sector_loop            every 60 min  — sector rotation + watchlist expansion
  _gap_loop               09:00 AM ET   — pre-market gap analysis on movers
  _position_mgmt_loop     every 60 sec  — target-1 partial exit + breakeven stop

Key constraints enforced:
  - Only execute if master confidence >= CONFIDENCE_THRESHOLD (65)
  - Every trade has a hard stop-loss (non-negotiable)
  - Position sizing via Kelly Criterion (Prompt 4 via supervisor)
  - News latency budget: 30-60 sec built into NEWS_LATENCY_SLEEP
  - All decisions are timestamped for backtesting

Controls:
  enable()  → start trading autonomously
  disable() → pause trading (loops keep running)
  status    → dict with current state and last scan results
"""

import asyncio
import logging
from datetime import time as dtime
from datetime import datetime
from zoneinfo import ZoneInfo

logger = logging.getLogger("auto_trader")

ET = ZoneInfo("America/New_York")
MARKET_OPEN  = dtime(9, 30)
MARKET_CLOSE = dtime(16, 0)
PRE_MARKET_GAP_START = dtime(9, 0)   # start gap scan at 9:00 AM
PRE_MARKET_GAP_END   = dtime(9, 29)  # finish before open

MAX_POSITIONS            = 3     # Step 4: reduced from 5 → 3 for paper trading (concentrate in best setups)
SCAN_INTERVAL_MINUTES    = 5     # technical scan every 5 minutes
NEWS_CHECK_INTERVAL_SEC  = 30    # news check on open positions every 30 seconds
SECTOR_SCAN_INTERVAL_MIN = 60    # sector rotation every hour
POS_MGMT_INTERVAL_SEC    = 60    # position management check every 60 seconds
NEWS_LATENCY_SLEEP_SEC   = 45    # wait ~45s after news fetch before acting (latency budget)

CONFIDENCE_THRESHOLD     = 65    # master confidence must exceed this to execute
SECTOR_MIN_SCORE         = 80    # min day_trade_score for sector expansion
NEWS_EXIT_RISK_LEVELS    = {"high", "extreme"}
GAP_MIN_PCT              = 0.5   # minimum gap % to warrant a gap analysis call

# Step 4 — Daily loss limit: halt ALL new entries if portfolio drops this % intraday
DAILY_LOSS_LIMIT_PCT     = 0.03  # -3% drawdown from today's opening equity → pause entries

# Step 4 — Sector correlation: max open positions allowed per sector group simultaneously
MAX_SAME_SECTOR          = 2

# Sector groupings for correlation check (symbols not listed → treated as their own sector)
_SECTOR_MAP: dict[str, str] = {
    # High-volatility tech momentum
    "NVDA": "tech_hv", "AMD": "tech_hv", "TSLA": "tech_hv", "PLTR": "tech_hv",
    "COIN": "tech_hv", "SQ": "tech_hv", "SMCI": "tech_hv",
    # Mega-cap tech
    "AAPL": "tech_mega", "GOOGL": "tech_mega", "META": "tech_mega",
    "AMZN": "tech_mega", "MSFT": "tech_mega",
    "NFLX": "tech_mega", "CRM":  "tech_mega",   # Lever A additions
    # Semiconductors
    "AVGO": "semis", "QCOM": "semis", "MU": "semis", "INTC": "semis", "TXN": "semis",
    # Finance
    "JPM": "finance", "BAC": "finance", "GS": "finance", "MS": "finance",
    "WFC": "finance", "C": "finance", "V": "finance", "MA": "finance",  # V = Lever A
    # ETFs / passive
    "SPY": "etf", "QQQ": "etf", "DIA": "etf", "IWM": "etf",
    "GLD": "etf", "SLV": "etf", "TLT": "etf", "HYG": "etf",
    # Energy
    "XOM": "energy", "CVX": "energy", "COP": "energy", "SLB": "energy", "OXY": "energy",
    # Healthcare
    "UNH": "healthcare", "JNJ": "healthcare", "PFE": "healthcare",
    "MRNA": "healthcare", "ABBV": "healthcare",
}

# Target-1 partial exit: sell this fraction when TP1 is hit
TARGET1_PARTIAL_SELL_FRAC = 0.50  # close 50% at target 1, run rest to target 2

# Opening Range Blackout: skip all scans for first 30 min (9:30–10:00 AM ET)
# Market makers push prices both ways in this window to trigger retail stops.
OPENING_RANGE_END = dtime(10, 0)

# Trailing stop: close if price falls this % below the high-water mark.
# Activates only after position is at least 0.5% in profit (HardKillSwitch
# covers initial downside via the fixed stop-loss).
TRAIL_STOP_PCT = 0.015   # 1.5% below the highest price seen


class AutoTrader:
    def __init__(self, watchlist: list[str]):
        self.watchlist = watchlist
        self._enabled  = False
        self._running  = False
        self._last_scan_time: str | None  = None
        self._last_scan_results: list[dict] = []
        self._trades_today = 0
        self._gap_scan_done_today: str | None = None  # "YYYY-MM-DD" of last gap scan

        # In-memory position targets for partial-exit management.
        # Populated when a trade is executed; cleared when position is closed.
        # {symbol: {target_1, target_2, entry_price, original_qty, stop_moved}}
        self._position_targets: dict[str, dict] = {}

        # Headline deduplication: track the last headline seen per symbol so
        # news_loop only calls Gemini when there is genuinely new information.
        # {symbol: last_headline_string}
        self._last_headlines: dict[str, str] = {}

        # Step 4 — Daily loss limit tracking
        self._daily_start_equity: float | None = None  # equity at market open today
        self._daily_loss_halt: bool = False             # True = entries paused for today
        self._daily_date: str | None = None             # "YYYY-MM-DD" of current session

        # Gemini rate-limit guard: free tier allows 15 RPM.
        # Each build_state fires 3 concurrent Gemini calls (signal + sentiment + vision)
        # plus 1 for master_decision = 4 per symbol.
        # Cap to 3 parallel symbol analyses → max ~12 Gemini calls/min in flight.
        self._gemini_semaphore = asyncio.Semaphore(3)

    # ── Public controls ───────────────────────────────────────────────────────

    def enable(self):
        self._enabled = True
        logger.info("Auto-Trader ENABLED — will trade on next scan cycle.")

    def disable(self):
        self._enabled = False
        logger.info("Auto-Trader DISABLED — monitoring paused.")

    def stop(self):
        self._running = False
        logger.info("Auto-Trader background loops stopped.")

    @property
    def status(self) -> dict:
        return {
            "enabled":                self._enabled,
            "market_open":            self._is_market_open(),
            "last_scan_time":         self._last_scan_time,
            "trades_today":           self._trades_today,
            "scan_interval_minutes":  SCAN_INTERVAL_MINUTES,
            "confidence_threshold":   CONFIDENCE_THRESHOLD,
            "max_positions":          MAX_POSITIONS,
            "last_scan_results":      self._last_scan_results,
            "tracked_positions":      list(self._position_targets.keys()),
            # Step 4 risk controls
            "daily_loss_halt":        self._daily_loss_halt,
            "daily_start_equity":     self._daily_start_equity,
            "daily_loss_limit_pct":   DAILY_LOSS_LIMIT_PCT,
            "max_same_sector":        MAX_SAME_SECTOR,
        }

    # ── Background loops ──────────────────────────────────────────────────────

    async def start(self):
        self._running = True
        logger.info("Auto-Trader background loops started (5 loops).")
        await asyncio.gather(
            self._scan_loop(),
            self._news_loop(),
            self._sector_loop(),
            self._gap_loop(),
            self._position_mgmt_loop(),
        )

    async def _scan_loop(self):
        """Technical signal scan on all watchlist stocks every 5 minutes."""
        while self._running:
            if self._enabled and self._is_market_open():
                now = datetime.now(ET)
                # Opening Range Blackout: skip 9:30–10:00 AM ET
                # Institutions deliberately spike prices both ways in this window
                # to trigger retail stops before the real move begins.
                if now.time() < OPENING_RANGE_END:
                    logger.info(
                        f"Opening Range Blackout active — no scans until 10:00 AM ET "
                        f"(now {now.strftime('%H:%M')} ET)."
                    )
                else:
                    # Step 4 — Daily loss limit check before each scan
                    halted = await self._check_daily_loss_limit()
                    if halted:
                        logger.warning("Daily loss limit active — scan skipped, no new entries.")
                    else:
                        await self._run_scan()
            await asyncio.sleep(SCAN_INTERVAL_MINUTES * 60)

    async def _news_loop(self):
        """News check on every open position every 30 seconds.
        Incorporates 30-60s news latency budget before acting."""
        while self._running:
            if self._enabled and self._is_market_open():
                await self._run_news_check()
            await asyncio.sleep(NEWS_CHECK_INTERVAL_SEC)

    async def _sector_loop(self):
        """Sector rotation scan every hour; expands watchlist with hot stocks."""
        while self._running:
            if self._enabled and self._is_market_open():
                await self._run_sector_expansion()
            await asyncio.sleep(SECTOR_SCAN_INTERVAL_MIN * 60)

    async def _gap_loop(self):
        """Pre-market gap analysis at 9:00-9:29 AM ET, once per trading day."""
        while self._running:
            now     = datetime.now(ET)
            today   = now.strftime("%Y-%m-%d")
            in_window = (
                now.weekday() < 5
                and PRE_MARKET_GAP_START <= now.time() <= PRE_MARKET_GAP_END
            )
            if in_window and self._gap_scan_done_today != today:
                self._gap_scan_done_today = today
                await self._run_gap_scan()
            # Poll every 60 seconds to catch the window
            await asyncio.sleep(60)

    async def _position_mgmt_loop(self):
        """
        Monitors open positions every 60 seconds.
        - When price >= target_1: sells 50% and moves stop to entry (breakeven).
        - End-of-day at 15:50 ET: closes all positions.
        """
        while self._running:
            if self._enabled:
                await self._manage_open_positions()
            await asyncio.sleep(POS_MGMT_INTERVAL_SEC)

    # ── Core scan ─────────────────────────────────────────────────────────────

    async def _run_scan(self):
        """
        For each watchlist symbol:
          1. Build state (Prompts 1+2+Vision+Quant in parallel)
          2. Run supervisor gates (check_only=True) — fast gate check
          3. If gates pass, run master decision (Prompt 8) with live options
          4. If master confidence >= 65 and BUY/SELL, execute via supervisor
          5. Store target levels for position management
        """
        from data.pipeline import build_state
        from agents.supervisor import run_consensus
        from execution.alpaca_broker import list_positions

        self._last_scan_time    = datetime.now(ET).strftime("%Y-%m-%d %H:%M:%S ET")
        self._last_scan_results = []
        logger.info(f"Auto-Trader scan starting: {len(self.watchlist)} symbols...")

        loop = asyncio.get_running_loop()
        try:
            positions   = await asyncio.wait_for(loop.run_in_executor(None, list_positions), timeout=10.0)
        except Exception:
            positions   = []
        open_symbols = {p["symbol"] for p in positions}
        slots_free   = MAX_POSITIONS - len(open_symbols)

        # Step 4 — Sector correlation pre-filter
        # Count how many open positions belong to each sector group.
        # Candidates whose sector already has MAX_SAME_SECTOR open positions are skipped
        # entirely (no API calls wasted, no execution attempted).
        from collections import Counter
        open_sector_counts = Counter(_SECTOR_MAP.get(s, s) for s in open_symbols)
        corr_skipped = []
        watchlist_to_scan = []
        for sym in self.watchlist:
            sector = _SECTOR_MAP.get(sym, sym)
            if sym in open_symbols:
                continue  # already have this position
            if open_sector_counts.get(sector, 0) >= MAX_SAME_SECTOR:
                corr_skipped.append(f"{sym}({sector})")
            else:
                watchlist_to_scan.append(sym)
        if corr_skipped:
            logger.info(f"Correlation filter: skipping {corr_skipped} — sector limit ({MAX_SAME_SECTOR}) reached.")

        # Read dynamic confidence threshold from self-learning instructions
        from utils.post_mortem import load_system_instructions as _load_si
        _dyn_threshold = _load_si().get("confidence_threshold", CONFIDENCE_THRESHOLD)

        async def _analyse(symbol: str) -> dict:
            ts = datetime.now(ET).strftime("%Y-%m-%dT%H:%M:%S")
            # Concurrency guard: analyse max 3 symbols in parallel.
            # Gemini rate-limiting is handled inside gemini_gate.py (4.1 s min interval,
            # threading.Lock). The semaphore prevents thread-pool saturation when all
            # 18 watchlist symbols are dispatched simultaneously.
            async with self._gemini_semaphore:
                try:
                    # Step 1: full pipeline state
                    state = await asyncio.wait_for(build_state(symbol), timeout=25.0)

                    # Step 1a: Gemini scan overlay — AI insight for every symbol (Option 4)
                    from agents.gemini_gate import gemini_scan_overlay
                    try:
                        overlay = await asyncio.wait_for(
                            loop.run_in_executor(None, gemini_scan_overlay, symbol, state),
                            timeout=15.0,
                        )
                        state.update(overlay)
                        if overlay.get("gemini_signal") not in ("INSUFFICIENT_DATA", None, ""):
                            logger.info(
                                f"Gemini overlay [{symbol}]: {overlay.get('gemini_signal')} "
                                f"({overlay.get('gemini_confidence')}%) — "
                                f"{str(overlay.get('gemini_key_insight', ''))[:80]}"
                            )
                    except asyncio.TimeoutError:
                        logger.warning(f"Gemini scan overlay timeout [{symbol}] — continuing.")
                    except Exception as _ov_err:
                        logger.warning(f"Gemini scan overlay error [{symbol}]: {_ov_err}")

                    # Step 2: fast gate check (no order, no position sizing)
                    gate = await asyncio.wait_for(
                        loop.run_in_executor(None, run_consensus, state, True),
                        timeout=10.0,
                    )
                    if not gate["approved"]:
                        return {
                            "symbol": symbol, "timestamp": ts,
                            "action": "NO_ACTION", "confidence": round(gate["confidence"], 3),
                            "reason": gate["reason"], "price": state.get("price"),
                            "_executed": False,
                        }

                    # Step 3: master decision (Prompt 8) with live options
                    from agents.master_agent import run_master_decision
                    from agents.options_agent import run_options_analysis

                    opts = None
                    try:
                        opts = await asyncio.wait_for(
                            loop.run_in_executor(None, run_options_analysis, symbol),
                            timeout=30.0,
                        )
                    except Exception:
                        pass  # non-fatal — master handles missing options

                    master = await asyncio.wait_for(
                        loop.run_in_executor(
                            None, run_master_decision, symbol, state, None, 0.02, opts, None
                        ),
                        timeout=20.0,
                    )

                    m_decision   = master.get("final_decision", "HOLD")
                    m_confidence = int(master.get("confidence", 0))

                    if m_decision not in ("BUY", "SELL") or m_confidence < _dyn_threshold:
                        return {
                            "symbol": symbol, "timestamp": ts,
                            "action": f"MASTER_{m_decision}",
                            "confidence": m_confidence,
                            "reason": master.get("decision_summary", "Master decision: no trade"),
                            "price": state.get("price"),
                            "_executed": False,
                        }

                    # Step 4: Gemini Trade Gate — final AI sanity check before execution (Option 1)
                    from agents.gemini_gate import gemini_trade_gate
                    try:
                        g_gate = await asyncio.wait_for(
                            loop.run_in_executor(
                                None, gemini_trade_gate, symbol, state, m_decision, m_confidence
                            ),
                            timeout=15.0,
                        )
                        if not g_gate.get("approved", True):
                            logger.info(
                                f"Gemini VETO [{symbol}]: {g_gate.get('reasoning', '')} | "
                                f"flags={g_gate.get('risk_flags', [])}"
                            )
                            return {
                                "symbol": symbol, "timestamp": ts,
                                "action": "GEMINI_VETO",
                                "confidence": m_confidence,
                                "reason": f"Gemini veto: {g_gate.get('reasoning', '')}",
                                "price": state.get("price"),
                                "_executed": False,
                            }
                        adj = g_gate.get("confidence_adjustment", 0)
                        if adj:
                            m_confidence = max(0, min(100, m_confidence + adj))
                            logger.debug(
                                f"Gemini gate adj [{symbol}]: {adj:+d} → {m_confidence}%"
                            )
                        if m_confidence < _dyn_threshold:
                            return {
                                "symbol": symbol, "timestamp": ts,
                                "action": "GEMINI_LOW_CONF",
                                "confidence": m_confidence,
                                "reason": (
                                    f"Gemini gate lowered confidence below threshold: "
                                    f"{g_gate.get('reasoning', '')}"
                                ),
                                "price": state.get("price"),
                                "_executed": False,
                            }
                    except asyncio.TimeoutError:
                        logger.warning(
                            f"Gemini trade gate timeout [{symbol}] — proceeding without AI gate."
                        )

                    # Step 5: full execution via supervisor (gates + position sizing + Kelly + order)
                    decision = await asyncio.wait_for(
                        loop.run_in_executor(None, run_consensus, state, False),
                        timeout=20.0,
                    )

                    executed = "EXECUTED" in decision.get("action_taken", "")
                    if executed:
                        # Step 5: store target levels for position management loop
                        _entry = decision.get("entry_price", state.get("price"))
                        self._position_targets[symbol] = {
                            "target_1":     decision.get("target_1"),
                            "target_2":     decision.get("target_2"),
                            "entry_price":  _entry,
                            "original_qty": decision.get("qty", 0),
                            "stop_moved":   False,
                            "partial_done": False,
                            "timestamp":    ts,
                            "trail_high":   _entry,  # high-water mark for trailing stop
                        }

                    return {
                        "symbol":            symbol,
                        "timestamp":         ts,
                        "action":            decision.get("action_taken", "NO_ACTION"),
                        "confidence":        round(max(gate["confidence"], m_confidence / 100), 3),
                        "master_confidence": m_confidence,
                        "master_summary":    master.get("decision_summary", ""),
                        "reason":            decision.get("reason", ""),
                        "price":             state.get("price"),
                        "stop_loss":         decision.get("stop_loss"),
                        "target_1":          decision.get("target_1"),
                        "target_2":          decision.get("target_2"),
                        "execution_urgency": master.get("execution_urgency", ""),
                        "_executed":         executed,
                    }

                except asyncio.TimeoutError:
                    logger.warning(f"Auto-Trader: {symbol} timed out.")
                    return {"symbol": symbol, "timestamp": ts, "action": "TIMEOUT",
                            "reason": "Analysis timed out", "_executed": False}
                except Exception as e:
                    logger.error(f"Auto-Trader error [{symbol}]: {e}")
                    return {"symbol": symbol, "timestamp": ts, "action": "ERROR",
                            "reason": str(e), "_executed": False}

        tasks = [asyncio.create_task(_analyse(s)) for s in watchlist_to_scan]
        for coro in asyncio.as_completed(tasks):
            r = await coro
            if r["_executed"]:
                if slots_free > 0:
                    slots_free     -= 1
                    self._trades_today += 1
                    logger.info(
                        f"TRADE EXECUTED [{r['symbol']}]: {r['action']} "
                        f"@ ${r.get('price', 0):.2f} | "
                        f"master_conf={r.get('master_confidence', 0)}% | "
                        f"urgency={r.get('execution_urgency', '')}"
                    )
                else:
                    r["action"] = "SKIPPED"
                    r["reason"] = f"max {MAX_POSITIONS} positions reached"
            self._last_scan_results.append({k: v for k, v in r.items() if k != "_executed"})

        executed = [r for r in self._last_scan_results if "EXECUTED" in r.get("action", "")]
        logger.info(f"Scan complete — {len(executed)} trade(s) executed.")
        try:
            from utils.notifier import notify, scan_summary_email
            subj, body = scan_summary_email(self._last_scan_results)
            await notify(subj, body)
        except Exception:
            pass

    # ── Gap analysis pre-market ───────────────────────────────────────────────

    async def _run_gap_scan(self):
        """
        Runs at 9:00-9:29 AM ET before market opens.
        For each watchlist stock with a gap >= GAP_MIN_PCT, runs Prompt 6
        (gap analysis) and logs trading bias + entry/fade strategy.
        """
        from agents.gap_agent import run_gap_analysis
        from data.market_data import get_current_price

        loop = asyncio.get_running_loop()
        logger.info(f"Gap scan starting on {len(self.watchlist)} symbols...")

        # ── Gemini Morning Brief (Option 3 — pre-market AI overview, once/day) ──────
        from agents.gemini_gate import gemini_morning_brief
        try:
            brief = await asyncio.wait_for(
                loop.run_in_executor(None, gemini_morning_brief, self.watchlist),
                timeout=20.0,
            )
            logger.info(
                f"🌅 MORNING BRIEF | Outlook: {brief.get('market_outlook', 'N/A')} | "
                f"Risks: {brief.get('key_macro_risks', [])} | "
                f"{brief.get('sector_notes', '')}"
            )
            for sym_note, note_txt in brief.get("symbol_notes", {}).items():
                logger.info(f"  📊 {sym_note}: {note_txt}")
            try:
                from utils.notifier import notify
                await notify(
                    f"🌅 Morning Brief — {brief.get('market_outlook', 'N/A')}",
                    "<b>AI Pre-Market Brief</b><br>"
                    + f"<b>Outlook:</b> {brief.get('market_outlook', 'N/A')}<br>"
                    + f"<b>Key Risks:</b> {', '.join(brief.get('key_macro_risks', []))}<br>"
                    + f"<b>Sectors:</b> {brief.get('sector_notes', '')}<br><br>"
                    + "<b>Watchlist Notes:</b><br>"
                    + "".join(
                        f"• <b>{s}:</b> {n}<br>"
                        for s, n in brief.get("symbol_notes", {}).items()
                    ),
                )
            except Exception:
                pass  # email failure is non-fatal
        except Exception as e:
            logger.warning(f"Morning brief error: {e}")

        async def _gap_one(symbol: str):
            try:
                price = await asyncio.wait_for(
                    loop.run_in_executor(None, get_current_price, symbol),
                    timeout=8.0,
                )
                if not price:
                    return
                result = await asyncio.wait_for(
                    loop.run_in_executor(None, run_gap_analysis, symbol, price),
                    timeout=40.0,
                )
                gap_pct = abs(result.get("_gap_meta", {}).get("gap_pct", 0))
                if gap_pct < GAP_MIN_PCT:
                    return
                bias  = result.get("trading_bias", "NEUTRAL")
                gtype = result.get("gap_type", "?")
                gclass = result.get("gap_classification", "?")
                fill_p = result.get("fill_probability_adjusted_for_catalyst", 0)
                logger.info(
                    f"GAP [{symbol}]: {gtype} {gclass} {gap_pct:+.2f}% | "
                    f"bias={bias} | fill_prob={fill_p:.0%} | "
                    f"alert={result.get('risk_warning', '')[:80]}"
                )
            except Exception as e:
                logger.warning(f"Gap scan error [{symbol}]: {e}")

        await asyncio.gather(*[_gap_one(s) for s in self.watchlist])
        logger.info("Gap scan complete.")

    # ── Position management ───────────────────────────────────────────────────

    async def _manage_open_positions(self):
        """
        Per usage guide position rules:
          - When price >= target_1: sell 50% (TARGET1_PARTIAL_SELL_FRAC), move stop to entry (breakeven)
          - When price >= target_2: close remainder
          - End-of-day at 15:50 ET: close all positions (no overnight holds)
        """
        from execution.alpaca_broker import list_positions, submit_order, close_position
        from data.market_data import get_current_price

        loop = asyncio.get_running_loop()
        now  = datetime.now(ET)

        # ── End-of-day flat-all ───────────────────────────────────────────────
        if now.weekday() < 5 and now.time() >= dtime(15, 50):
            try:
                from utils.post_mortem import update_trade_result, auto_review_if_ready
                positions = await loop.run_in_executor(None, list_positions)
                for pos in positions:
                    sym         = pos["symbol"]
                    curr_price  = float(pos.get("current_price",   0) or 0)
                    entry_price = float(pos.get("avg_entry_price", 0) or 0)
                    trade_result = "WIN" if curr_price >= entry_price else "LOSS"
                    # Record outcome before closing
                    if curr_price:
                        update_trade_result(sym, curr_price, trade_result)
                    await loop.run_in_executor(None, close_position, sym)
                    self._position_targets.pop(sym, None)
                    logger.info(f"EOD flat: closed {sym} → {trade_result}")
                # Trigger self-review if enough trades have accumulated
                review = auto_review_if_ready(10)
                if review:
                    logger.info(f"Auto-review triggered after EOD flat:\n{review}")
            except Exception as e:
                logger.error(f"EOD flat error: {e}")
            return

        # ── Target management ─────────────────────────────────────────────────
        if not self._position_targets:
            return

        for symbol, targets in list(self._position_targets.items()):
            if targets.get("partial_done") and targets.get("stop_moved"):
                continue  # all management done for this position

            try:
                price = await asyncio.wait_for(
                    loop.run_in_executor(None, get_current_price, symbol),
                    timeout=5.0,
                )
                if not price:
                    continue

                t1 = targets.get("target_1")
                t2 = targets.get("target_2")

                # ── Target 2 hit → close remainder ───────────────────────────
                if t2 and price >= t2 and not targets.get("t2_done"):
                    await loop.run_in_executor(None, close_position, symbol)
                    logger.info(f"TARGET 2 HIT [{symbol}]: ${price:.2f} >= ${t2:.2f} — closed remainder.")
                    # Record WIN post-mortem and trigger self-review if ready
                    try:
                        from utils.post_mortem import update_trade_result, auto_review_if_ready
                        update_trade_result(symbol, price, "WIN")
                        review = auto_review_if_ready(10)
                        if review:
                            logger.info(f"Auto-review triggered after TP2:\n{review}")
                    except Exception as _pm_err:
                        logger.warning(f"Post-mortem error [{symbol}]: {_pm_err}")
                    self._position_targets.pop(symbol, None)
                    _notify_exit(symbol, "TARGET 2", price, t2)
                    continue

                # ── Trailing stop — lock in profits / protect gains ──────────
                # Only activates after price is at least 0.5% above entry price.
                # Updates a high-water mark every tick; closes if price falls
                # TRAIL_STOP_PCT below that mark.
                trail_high  = targets.get("trail_high") or targets.get("entry_price", 0)
                entry_price = targets.get("entry_price", 0)

                # Update high-water mark
                if price and price > trail_high:
                    targets["trail_high"] = price
                    trail_high = price

                trail_stop_price = trail_high * (1 - TRAIL_STOP_PCT)

                if (
                    trail_high
                    and entry_price
                    and trail_high > entry_price * 1.005   # only once 0.5% in profit
                    and price <= trail_stop_price
                ):
                    await loop.run_in_executor(None, close_position, symbol)
                    trade_result = "WIN" if price > entry_price else "LOSS"
                    logger.info(
                        f"TRAILING STOP [{symbol}]: ${price:.2f} <= "
                        f"trail_stop=${trail_stop_price:.2f} "
                        f"(high=${trail_high:.2f}) — closed ({trade_result})."
                    )
                    try:
                        from utils.post_mortem import update_trade_result, auto_review_if_ready
                        update_trade_result(symbol, price, trade_result)
                        review = auto_review_if_ready(10)
                        if review:
                            logger.info(f"Auto-review triggered after trailing stop:\n{review}")
                    except Exception as _pm_err:
                        logger.warning(f"Post-mortem error [{symbol}]: {_pm_err}")
                    self._position_targets.pop(symbol, None)
                    _notify_exit(symbol, "TRAILING STOP", price, trail_stop_price)
                    continue

                # ── Target 1 hit → partial exit + move stop to breakeven ─────
                if t1 and price >= t1 and not targets.get("partial_done"):
                    orig_qty = targets.get("original_qty", 0)
                    sell_qty = max(1.0, round(orig_qty * TARGET1_PARTIAL_SELL_FRAC, 2))
                    await asyncio.wait_for(
                        loop.run_in_executor(None, submit_order, symbol, sell_qty, "sell"),
                        timeout=10.0,
                    )
                    targets["partial_done"] = True
                    targets["stop_moved"]   = True   # note: stop is advisory; update in broker separately
                    logger.info(
                        f"TARGET 1 HIT [{symbol}]: ${price:.2f} >= ${t1:.2f} — "
                        f"sold ${sell_qty:.2f}/${orig_qty:.2f} notional. Stop moved to entry "
                        f"${targets.get('entry_price', '?'):.2f} (breakeven)."
                    )
                    _notify_exit(symbol, "TARGET 1 PARTIAL", price, t1)

            except Exception as e:
                logger.error(f"Position mgmt error [{symbol}]: {e}")

    # ── Sector rotation expansion ─────────────────────────────────────────────

    async def _run_sector_expansion(self):
        """Hourly sector rotation scan; adds high-score stocks to watchlist."""
        from agents.sector_agent import run_sector_rotation
        from data.watchlist import add_symbol

        loop = asyncio.get_running_loop()
        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(None, run_sector_rotation, 2),
                timeout=60.0,
            )
        except Exception as e:
            logger.error(f"Sector scan failed: {e}")
            return

        hottest   = result.get("hottest_sector", "?")
        top_stocks = result.get("top_stocks_in_rotation", [])
        added     = []

        for stock in top_stocks:
            score  = stock.get("day_trade_score", 0)
            ticker = stock.get("ticker", "")
            if ticker and score >= SECTOR_MIN_SCORE and ticker not in self.watchlist:
                self.watchlist = add_symbol(ticker)
                added.append(ticker)
                logger.info(f"Sector expansion: added {ticker} (score={score}, sector={hottest})")

        logger.info(
            f"Sector scan done. Hottest: {hottest}. "
            f"Added: {added if added else 'none'}."
        )

    # ── News check ────────────────────────────────────────────────────────────

    async def _run_news_check(self):
        """
        Every 30 seconds: fetches latest headline for each open position and
        runs Prompt 2 (news_agent). Incorporates NEWS_LATENCY_SLEEP_SEC delay.
        Auto-exits on EXIT_ALL + high/extreme risk level.
        """
        from agents.gemini_gate import gemini_news_sentiment   # Option 2 — AI news
        from execution.alpaca_broker import list_positions, close_position
        from data.news_fetcher import fetch_latest_news_item

        loop = asyncio.get_running_loop()
        try:
            positions = await loop.run_in_executor(None, list_positions)
        except Exception:
            return

        for pos in positions:
            symbol = pos["symbol"]
            try:
                news_item = await asyncio.wait_for(
                    loop.run_in_executor(None, fetch_latest_news_item, symbol),
                    timeout=10.0,
                )
                if not news_item or not news_item.get("headline"):
                    continue

                headline = news_item["headline"]

                # Headline deduplication: skip Gemini call if same headline as last check.
                # Prevents burning 30-sec RPM quota on unchanged news.
                if self._last_headlines.get(symbol) == headline:
                    continue
                self._last_headlines[symbol] = headline

                # News latency budget: wait before acting on headline
                await asyncio.sleep(NEWS_LATENCY_SLEEP_SEC)

                position_side = "LONG" if float(pos.get("qty", 0) or 0) > 0 else "SHORT"
                result = await asyncio.wait_for(
                    loop.run_in_executor(
                        None, gemini_news_sentiment,
                        symbol, headline, position_side,
                    ),
                    timeout=20.0,
                )

                rec   = result.get("recommendation", "HOLD")
                risk  = result.get("risk_level", "low")
                ts    = datetime.now(ET).strftime("%H:%M:%S ET")

                logger.info(
                    f"[{ts}] News [{symbol}]: {rec} | risk={risk} | "
                    f"'{news_item['headline'][:70]}...'"
                )

                # Auto-exit on EXIT_ALL + high/extreme risk
                if rec == "EXIT_ALL" and risk in NEWS_EXIT_RISK_LEVELS:
                    logger.warning(f"NEWS EXIT [{symbol}]: {result.get('reasoning', '')}")
                    curr_price  = float(pos.get("current_price",   0) or 0)
                    entry_price = float(pos.get("avg_entry_price", 0) or 0)
                    trade_result = "WIN" if curr_price >= entry_price else "LOSS"
                    await loop.run_in_executor(None, close_position, symbol)
                    # Record outcome and trigger self-review if ready
                    try:
                        from utils.post_mortem import update_trade_result, auto_review_if_ready
                        if curr_price:
                            update_trade_result(symbol, curr_price, trade_result)
                        review = auto_review_if_ready(10)
                        if review:
                            logger.info(f"Auto-review triggered after news exit:\n{review}")
                    except Exception as _pm_err:
                        logger.warning(f"Post-mortem error [{symbol}]: {_pm_err}")
                    self._position_targets.pop(symbol, None)
                    _notify_exit(symbol, "NEWS EXIT", None, None, result.get("reasoning", ""))

            except Exception as e:
                logger.error(f"News check error [{symbol}]: {e}")

    # ── Helpers ───────────────────────────────────────────────────────────────

    async def _check_daily_loss_limit(self) -> bool:
        """
        Step 4 — Daily loss guard.
        Returns True (halt) if today's portfolio drawdown exceeds DAILY_LOSS_LIMIT_PCT.
        Resets automatically at the start of each new trading day.
        """
        from execution.alpaca_broker import get_portfolio_value
        loop = asyncio.get_running_loop()
        today = datetime.now(ET).strftime("%Y-%m-%d")

        # New trading day → reset baseline
        if self._daily_date != today:
            self._daily_date = today
            self._daily_loss_halt = False
            try:
                equity = await asyncio.wait_for(
                    loop.run_in_executor(None, get_portfolio_value), timeout=10.0
                )
                self._daily_start_equity = equity
                logger.info(f"Daily P&L tracker reset for {today}. Opening equity: ${equity:,.2f}")
            except Exception as e:
                logger.warning(f"Could not fetch opening equity: {e}")
                self._daily_start_equity = None
            return False

        # Already halted today — stay halted
        if self._daily_loss_halt:
            return True

        # No baseline captured yet — safe to trade
        if not self._daily_start_equity:
            return False

        # Check current equity against daily baseline
        try:
            equity = await asyncio.wait_for(
                loop.run_in_executor(None, get_portfolio_value), timeout=10.0
            )
            loss_pct = (equity - self._daily_start_equity) / self._daily_start_equity
            if loss_pct <= -DAILY_LOSS_LIMIT_PCT:
                self._daily_loss_halt = True
                logger.warning(
                    f"⛔ DAILY LOSS LIMIT HIT: {loss_pct:.2%} drawdown "
                    f"(limit: -{DAILY_LOSS_LIMIT_PCT:.0%}, "
                    f"start=${self._daily_start_equity:,.2f}, now=${equity:,.2f}). "
                    f"All new entries PAUSED for the rest of today."
                )
                try:
                    from utils.notifier import notify
                    await notify(
                        f"⛔ Daily Loss Limit: {loss_pct:.2%}",
                        f"Portfolio down <b>{loss_pct:.2%}</b> today "
                        f"(start=${self._daily_start_equity:,.2f} → now=${equity:,.2f}).<br>"
                        f"Auto-Trader entry PAUSED — existing positions still managed.",
                    )
                except Exception:
                    pass
                return True
        except Exception as e:
            logger.warning(f"Daily loss check error: {e}")

        return False

    def _is_market_open(self) -> bool:
        now = datetime.now(ET)
        if now.weekday() >= 5:
            return False
        return MARKET_OPEN <= now.time() <= MARKET_CLOSE


def _notify_exit(symbol: str, trigger: str, price, target, reason: str = ""):
    """Fire-and-forget email notification for exits."""
    try:
        from utils.notifier import notify_sync
        subj = f"EXIT {trigger}: {symbol}"
        body = (
            f"<b>{trigger}</b> triggered for <b>{symbol}</b><br>"
            + (f"<b>Price:</b> ${price:.2f} | <b>Target:</b> ${target:.2f}<br>" if price else "")
            + (f"<b>Reason:</b> {reason}<br>" if reason else "")
        )
        notify_sync(subj, body)
    except Exception:
        pass
