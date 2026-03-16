"""
Omni-Agent System — FastAPI Entrypoint.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel

from data.pipeline import build_state
from agents.supervisor import run_consensus
from execution.alpaca_broker import list_positions, close_position, get_portfolio_value
from execution.risk_manager import HardKillSwitch
from execution.auto_trader import AutoTrader
from data.watchlist import load as load_watchlist, add_symbol, remove_symbol, UNIVERSE
from utils.post_mortem import load_system_instructions, weekly_self_review

logging.basicConfig(level=logging.INFO, format="%(name)s | %(levelname)s | %(message)s")
logger = logging.getLogger("omni-agent")

kill_switch = HardKillSwitch(poll_interval=30)
auto_trader = AutoTrader(watchlist=load_watchlist())


@asynccontextmanager
async def lifespan(app: FastAPI):
    ks_task = asyncio.create_task(kill_switch.start())
    at_task = asyncio.create_task(auto_trader.start())
    logger.info("Omni-Agent System online. HardKillSwitch armed. Auto-Trader standing by.")
    yield
    kill_switch.stop()
    auto_trader.stop()
    ks_task.cancel()
    at_task.cancel()
    logger.info("Omni-Agent System shutting down.")


app = FastAPI(
    title="Omni-Agent Trading System",
    description="Triad of Certainty — Quant + Sentiment + Vision consensus trading.",
    version="1.0.0",
    lifespan=lifespan,
)


class TradeRequest(BaseModel):
    market_cap_billions: float | None = None


# ──────────────────────────────────────────────────────────────────────────────
# Dashboard
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def dashboard():
    html = (Path(__file__).parent / "dashboard" / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(content=html)


# ──────────────────────────────────────────────────────────────────────────────
# Watchlist (Option 5)
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/watchlist")
def get_watchlist():
    wl = load_watchlist()
    auto_trader.watchlist = wl
    return JSONResponse(content={"symbols": wl, "universe": UNIVERSE})


@app.post("/watchlist/{symbol}")
def watchlist_add(symbol: str):
    wl = add_symbol(symbol.upper())
    auto_trader.watchlist = wl
    return JSONResponse(content={"symbols": wl})


@app.delete("/watchlist/{symbol}")
def watchlist_remove(symbol: str):
    wl = remove_symbol(symbol.upper())
    auto_trader.watchlist = wl
    return JSONResponse(content={"symbols": wl})


# ──────────────────────────────────────────────────────────────────────────────
# Auto-Trade Mode
# ──────────────────────────────────────────────────────────────────────────────

@app.post("/auto-trade/enable")
def auto_trade_enable():
    auto_trader.enable()
    return JSONResponse(content={"auto_trade": True, "message": "Auto-Trade enabled."})


@app.post("/auto-trade/disable")
def auto_trade_disable():
    auto_trader.disable()
    return JSONResponse(content={"auto_trade": False, "message": "Auto-Trade disabled."})


@app.get("/auto-trade/status")
def auto_trade_status():
    return JSONResponse(content=auto_trader.status)


@app.post("/auto-trade/scan-now")
async def auto_trade_scan_now():
    asyncio.create_task(auto_trader._run_scan())
    return JSONResponse(content={"message": "Scan started — check status in a moment."})


# ──────────────────────────────────────────────────────────────────────────────
# Scanner
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/scanner")
async def scanner():
    watchlist = load_watchlist()

    async def _scan(symbol: str):
        try:
            state = await build_state(symbol)
            return {
                "symbol": symbol,
                "price": state["price"],
                "signal": state["signal"],
                "green_light": state["green_light"],
                "bb_pvalue": state["bb_pvalue"],
                "rsi_pvalue": state["rsi_pvalue"],
                "sentiment_score": state["sentiment_score"],
                "sentiment_direction": state["sentiment_direction"],
                "chart_pattern": state["chart_pattern"],
                "vision_veto": state["vision_veto"],
                "top_headline": state.get("top_headline", ""),
                "error": None,
            }
        except Exception as e:
            return {"symbol": symbol, "error": str(e), "green_light": False}

    results = await asyncio.gather(*[_scan(s) for s in watchlist])
    return JSONResponse(content=list(results))


# ──────────────────────────────────────────────────────────────────────────────
# Backtest (Option 4)
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/backtest/{symbol}")
async def backtest(symbol: str, period: str = "2y", capital: float = 5000.0):
    symbol = symbol.upper()
    try:
        from backtest.engine import run_backtest
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, run_backtest, symbol, period, capital)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Backtest failed for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ──────────────────────────────────────────────────────────────────────────────
# Portfolio summary
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/portfolio")
def portfolio():
    try:
        value = get_portfolio_value()
        positions = list_positions()
        total_pnl = sum(p["unrealized_pnl"] or 0 for p in positions)
        return JSONResponse(content={
            "equity": value,
            "open_positions": len(positions),
            "total_unrealized_pnl": round(total_pnl, 2),
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ──────────────────────────────────────────────────────────────────────────────
# Core routes
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "online", "system": "Omni-Agent v1.0"}


@app.get("/account-info")
def account_info():
    from execution.alpaca_broker import _get_client
    from config import settings
    client = _get_client()
    account = client.get_account()
    return JSONResponse(content={
        "account_id": str(account.id),
        "status": str(account.status),
        "equity": str(account.equity),
        "cash": str(account.cash),
        "paper": True,
        "api_key_last4": settings.ALPACA_API_KEY[-4:],
    })


@app.get("/analyze/{symbol}")
async def analyze(symbol: str, market_cap_billions: float | None = None):
    symbol = symbol.upper()
    try:
        state = await build_state(symbol, market_cap_billions)
        return JSONResponse(content=state)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/trade/{symbol}")
async def trade(symbol: str, body: TradeRequest = TradeRequest()):
    symbol = symbol.upper()
    try:
        state = await build_state(symbol, body.market_cap_billions)
        decision = run_consensus(state)
        return JSONResponse(content={"state": state, "decision": decision})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/positions")
def positions():
    try:
        return JSONResponse(content=list_positions())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/close/{symbol}")
def close(symbol: str):
    result = close_position(symbol.upper())
    if result is None:
        raise HTTPException(status_code=404, detail=f"No open position for {symbol.upper()}")
    return JSONResponse(content=result)


@app.get("/post-mortems")
def post_mortems():
    from utils.post_mortem import _load_log
    return JSONResponse(content=_load_log())


# ──────────────────────────────────────────────────────────────────────────────
# ATR Position Sizing
# ──────────────────────────────────────────────────────────────────────────────

class PositionSizeRequest(BaseModel):
    entry_price: float | None = None       # override; defaults to current price
    risk_per_trade_usd: float | None = None
    account_size: float | None = None


@app.post("/position-size/{symbol}")
async def position_size(symbol: str, body: PositionSizeRequest = PositionSizeRequest()):
    """
    Computes ATR-adjusted stop, targets, position size, and exit plan for a symbol.
    Uses current price unless entry_price is provided in the request body.
    """
    symbol = symbol.upper()
    loop   = asyncio.get_running_loop()
    try:
        from agents.position_agent import run_position_sizing
        from data.market_data import get_current_price

        price = body.entry_price or await loop.run_in_executor(None, get_current_price, symbol)
        plan  = await loop.run_in_executor(
            None, run_position_sizing,
            symbol, price, None,
            body.risk_per_trade_usd, body.account_size,
        )
        return JSONResponse(content={"symbol": symbol, "plan": plan})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ──────────────────────────────────────────────────────────────────────────────
# Meta-signal aggregator (Kelly Criterion sizing)
# ──────────────────────────────────────────────────────────────────────────────

class MetaSignalRequest(BaseModel):
    entry_price: float | None = None
    stop_loss: float | None = None
    account_size: float | None = None


@app.post("/meta-signal/{symbol}")
async def meta_signal(symbol: str, body: MetaSignalRequest = MetaSignalRequest()):
    """
    Runs the full pipeline for a symbol and passes all signals to the
    meta-agent for Kelly Criterion composite confidence and position sizing.
    Returns composite_confidence, kelly_criterion, shares_recommended, and more.
    """
    symbol = symbol.upper()
    loop   = asyncio.get_running_loop()
    try:
        from agents.meta_agent import run_meta_sizing
        from data.pipeline import build_state

        state = await build_state(symbol)
        price = body.entry_price or state["price"]
        sl    = body.stop_loss   or state.get("signal_stop_loss")

        meta = await loop.run_in_executor(
            None, run_meta_sizing, symbol, state, price, sl, body.account_size
        )
        return JSONResponse(content={"symbol": symbol, "price": price, "meta": meta})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ──────────────────────────────────────────────────────────────────────────────
# News-to-trade position check
# ──────────────────────────────────────────────────────────────────────────────

class NewsCheckRequest(BaseModel):
    headline: str | None = None      # override: use this headline instead of fetching
    source: str | None = None        # override: news source
    stop_loss: float | None = None   # override: current stop-loss
    take_profit: float | None = None # override: current take-profit
    sector: str = "unknown"


@app.post("/news-check/{symbol}")
async def news_check(symbol: str, body: NewsCheckRequest = NewsCheckRequest()):
    """
    Fetches the latest news for a symbol, evaluates impact on the open
    position, and returns a position management recommendation.

    If no position is open, still returns the news sentiment assessment.
    Override the headline by passing { "headline": "..." } in the body.
    """
    symbol = symbol.upper()
    loop   = asyncio.get_running_loop()

    try:
        from agents.news_agent import analyze_news
        from execution.alpaca_broker import get_position
        from data.news_fetcher import fetch_latest_news_item

        # Fetch position and news concurrently
        position_task = loop.run_in_executor(None, get_position, symbol)
        news_task     = loop.run_in_executor(None, fetch_latest_news_item, symbol)
        position, news_item = await asyncio.gather(position_task, news_task)

        # Caller can override headline
        if body.headline:
            headline  = body.headline
            source    = body.source or "Manual"
            news_ts   = None
        elif news_item:
            headline  = news_item["headline"]
            source    = news_item["source"]
            news_ts   = news_item["timestamp"]
        else:
            raise HTTPException(status_code=404, detail=f"No recent news found for {symbol}")

        result = await loop.run_in_executor(
            None,
            analyze_news,
            symbol, headline, source, news_ts,
            position, body.stop_loss, body.take_profit, body.sector,
        )

        return JSONResponse(content={
            "symbol":    symbol,
            "headline":  headline,
            "source":    source,
            "position":  position,
            "analysis":  result,
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ──────────────────────────────────────────────────────────────────────────────
# Notifications
# ──────────────────────────────────────────────────────────────────────────────

@app.post("/notifications/test")
async def notifications_test():
    from utils.notifier import notify, is_configured
    if not is_configured():
        raise HTTPException(status_code=400, detail="Email not configured in .env")
    await notify(
        "Test Alert",
        "<div style='font-family:monospace;background:#0d1117;color:#c9d1d9;padding:24px'>"
        "<h2 style='color:#58a6ff'>✅ Omni-Agent notifications working!</h2></div>"
    )
    return JSONResponse(content={"message": "Test email sent."})


@app.get("/notifications/log")
def notifications_log():
    from utils.notifier import get_log, is_configured
    return JSONResponse(content={"configured": is_configured(), "log": get_log()})


@app.post("/weekly-review")
async def weekly_review():
    loop = asyncio.get_running_loop()
    summary = await loop.run_in_executor(None, weekly_self_review)
    instructions = load_system_instructions()
    return JSONResponse(content={"summary": summary, "new_instructions": instructions})


# ──────────────────────────────────────────────────────────────────────────────
# Master trading decision engine
# ──────────────────────────────────────────────────────────────────────────────

class MasterDecisionRequest(BaseModel):
    account_size:  float | None = None   # override; defaults to Alpaca equity
    max_risk_pct:  float        = 0.02   # fraction of account to risk per trade
    run_options:   bool         = True   # fetch live options chain (adds ~10-20s)
    run_sector:    bool         = False  # run full sector scan (adds ~30-40s)


@app.post("/master-decision/{symbol}")
async def master_decision(symbol: str, body: MasterDecisionRequest = MasterDecisionRequest()):
    """
    Runs the full pipeline (technical + news + vision + quant), optionally
    fetches options sentiment and/or sector context, then asks the master
    decision engine for a single BUY/SELL/HOLD/AVOID with a complete
    execution plan (shares, entry, stops, targets, exit triggers).

    Body params (all optional):
      account_size  — override Alpaca equity (USD)
      max_risk_pct  — risk per trade, default 0.02 (2%)
      run_options   — fetch live options chain, default true
      run_sector    — run sector rotation scan, default false (slow)

    Example: POST /master-decision/NVDA
    """
    symbol = symbol.upper()
    loop   = asyncio.get_running_loop()
    try:
        from agents.master_agent   import run_master_decision
        from agents.options_agent  import run_options_analysis
        from agents.sector_agent   import run_sector_rotation
        from data.pipeline         import build_state

        # 1. Core pipeline (always)
        state = await build_state(symbol)

        # 2. Options (optional, parallel-friendly)
        options_summary = None
        if body.run_options:
            try:
                options_summary = await asyncio.wait_for(
                    loop.run_in_executor(None, run_options_analysis, symbol),
                    timeout=40.0,
                )
            except Exception:
                pass   # non-fatal — master agent handles missing options

        # 3. Sector (optional, slow)
        sector_summary = None
        if body.run_sector:
            try:
                sector_summary = await asyncio.wait_for(
                    loop.run_in_executor(None, run_sector_rotation, 3),
                    timeout=55.0,
                )
            except Exception:
                pass

        # 4. Master decision
        result = await asyncio.wait_for(
            loop.run_in_executor(
                None, run_master_decision,
                symbol, state, body.account_size, body.max_risk_pct,
                options_summary, sector_summary,
            ),
            timeout=30.0,
        )

        return JSONResponse(content={"symbol": symbol, **result})

    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Master decision timed out")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ──────────────────────────────────────────────────────────────────────────────
# Options market sentiment
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/options-sentiment/{symbol}")
async def options_sentiment(symbol: str):
    """
    Fetches live options chain (put/call ratio, ATM IV, OTM skew, max-OI
    strikes) + 1-year daily OHLCV for IV Rank/ATR, then asks Gemini to
    interpret the combined options market sentiment.

    Returns: options_sentiment, put_call_ratio_signal, iv_assessment,
    skew_interpretation, expected_move (options vs ATR), sentiment_probability,
    trading_implications, key_levels_to_watch, alert.

    Example: GET /options-sentiment/AAPL
    """
    symbol = symbol.upper()
    loop   = asyncio.get_running_loop()
    try:
        from agents.options_agent import run_options_analysis
        result = await asyncio.wait_for(
            loop.run_in_executor(None, run_options_analysis, symbol),
            timeout=60.0,
        )
        return JSONResponse(content={"symbol": symbol, **result})
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Options sentiment scan timed out (>60s)")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ──────────────────────────────────────────────────────────────────────────────
# Gap analysis engine
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/gap-analysis/{symbol}")
async def gap_analysis(symbol: str, premarket_price: float):
    """
    Analyses an overnight / pre-market gap for a symbol.

    Query params:
      premarket_price (required) — the current pre-market or after-hours price.

    Returns gap classification, fill probability (raw + catalyst-adjusted),
    trading bias, and both entry (continuation) and fade trade plans.

    Example: GET /gap-analysis/AAPL?premarket_price=202.85
    """
    symbol = symbol.upper()
    loop   = asyncio.get_running_loop()
    try:
        from agents.gap_agent import run_gap_analysis
        result = await asyncio.wait_for(
            loop.run_in_executor(None, run_gap_analysis, symbol, premarket_price),
            timeout=45.0,
        )
        return JSONResponse(content={"symbol": symbol, "premarket_price": premarket_price, **result})
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Gap analysis timed out (>45s)")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ──────────────────────────────────────────────────────────────────────────────
# Sector rotation scan
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/sector-scan")
async def sector_scan(top_n: int = 3):
    """
    Fetches all 10 sector ETFs + constituent stock data, then asks Gemini
    to identify the hottest sectors and best intraday setups.

    Query param:
      top_n (default 3) — number of hot sectors to analyse in depth.
    """
    loop = asyncio.get_running_loop()
    try:
        from agents.sector_agent import run_sector_rotation
        result = await asyncio.wait_for(
            loop.run_in_executor(None, run_sector_rotation, top_n),
            timeout=60.0,
        )
        return JSONResponse(content=result)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Sector scan timed out (>60s)")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
