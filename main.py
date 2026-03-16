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

_ET_CLOCK = """
<style>
#countdown{visibility:hidden;width:0;overflow:hidden}
#_etclock{position:fixed;top:14px;right:28px;z-index:200;display:flex;flex-direction:column;align-items:flex-end;gap:2px;pointer-events:none}
#_ct{font-family:'JetBrains Mono',monospace;font-size:17px;font-weight:400;color:rgba(255,255,255,.82);letter-spacing:2px;line-height:1}
#_cl{font-family:'JetBrains Mono',monospace;font-size:9px;color:rgba(255,255,255,.25);letter-spacing:3px;text-transform:uppercase}
</style>
<div id="_etclock"><span id="_ct">00:00:00</span><span id="_cl">EASTERN TIME</span></div>
<script>
(function(){
  var fmt=new Intl.DateTimeFormat('en-US',{timeZone:'America/New_York',hour:'2-digit',minute:'2-digit',second:'2-digit',hour12:false});
  function tick(){var el=document.getElementById('_ct');if(el)el.textContent=fmt.format(new Date());}
  tick();
  setInterval(tick,1000);
})();
</script>
"""

_LOGIN_GATE = """
<style>
#_lg{position:fixed;inset:0;z-index:99999;display:flex;align-items:center;justify-content:center;background:#05050f;font-family:'Inter',sans-serif}
#_lg::before{content:'';position:fixed;inset:0;pointer-events:none;background:radial-gradient(ellipse 60% 50% at -10% 0%,rgba(139,92,246,.12) 0%,transparent 70%),radial-gradient(ellipse 50% 50% at 110% 100%,rgba(0,212,255,.10) 0%,transparent 70%)}
#_lc{background:rgba(8,12,35,.9);border:1px solid rgba(0,212,255,.18);border-radius:20px;padding:48px 40px;width:320px;display:flex;flex-direction:column;align-items:center;gap:20px;box-shadow:0 0 80px rgba(0,212,255,.06),0 0 140px rgba(139,92,246,.06);position:relative;overflow:hidden;backdrop-filter:blur(24px)}
#_lc::before{content:'';position:absolute;top:0;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,rgba(0,212,255,.6),rgba(139,92,246,.6),transparent)}
#_ll{font-family:monospace;font-size:10px;letter-spacing:5px;color:rgba(255,255,255,.35);text-transform:uppercase}
#_lt{font-size:19px;font-weight:300;color:#fff;letter-spacing:3px;text-align:center;line-height:1.4}
#_lt span{color:#00d4ff;font-weight:500}
#_li{width:100%;background:rgba(0,0,0,.5);border:1px solid rgba(0,212,255,.18);border-radius:10px;padding:14px 18px;color:#fff;font-family:monospace;font-size:17px;letter-spacing:4px;outline:none;text-align:center;box-sizing:border-box}
#_li:focus{border-color:rgba(0,212,255,.45);box-shadow:0 0 24px rgba(0,212,255,.1)}
#_lb{width:100%;padding:14px;background:linear-gradient(135deg,rgba(0,212,255,.14),rgba(139,92,246,.14));border:1px solid rgba(0,212,255,.3);border-radius:10px;color:#00d4ff;font-family:monospace;font-size:12px;letter-spacing:4px;font-weight:500;cursor:pointer}
#_lb:hover{background:linear-gradient(135deg,rgba(0,212,255,.24),rgba(139,92,246,.24));box-shadow:0 0 30px rgba(0,212,255,.18)}
#_le{font-family:monospace;font-size:11px;color:#ff4a3d;letter-spacing:2px;text-align:center;min-height:14px;opacity:0;transition:opacity .2s}
#_lh{font-size:10px;color:rgba(255,255,255,.2);letter-spacing:2px;font-family:monospace}
</style>
<div id="_lg">
  <div id="_lc">
    <div id="_ll">&#9672; &nbsp; O M N I - A G E N T</div>
    <div id="_lt">TRADING<br><span>DASHBOARD</span></div>
    <input type="password" id="_li" placeholder="&bull;&bull;&bull;&bull;&bull;&bull;&bull;&bull;" autocomplete="off"/>
    <div id="_le">INCORRECT PASSWORD</div>
    <button id="_lb" onclick="_cl()">ENTER &rarr;</button>
    <div id="_lh">AUTHORIZED ACCESS ONLY</div>
  </div>
</div>
<script>
function _cl(){
  var v=document.getElementById('_li').value;
  var e=document.getElementById('_le');
  if(v==='Pitter2018'){
    var g=document.getElementById('_lg');
    g.style.transition='opacity .7s ease';
    g.style.opacity='0';
    setTimeout(function(){g.style.display='none'},700);
  } else {
    e.style.opacity='1';
    document.getElementById('_li').value='';
    document.getElementById('_li').focus();
    setTimeout(function(){e.style.opacity='0'},2200);
  }
}
document.getElementById('_li').addEventListener('keydown',function(e){if(e.key==='Enter')_cl();});
</script>
"""

_MOBILE_CSS = """
<style>
@media (max-width: 768px) {
  /* Stack all grid layouts to single column */
  .grid, [class*="grid-cols"], [style*="grid-template-columns"] {
    grid-template-columns: 1fr !important;
  }
  /* Reduce outer padding so content fills screen */
  body { padding: 8px !important; }
  .container, main, section, .dashboard, #app {
    padding: 8px !important;
    margin: 0 !important;
    width: 100% !important;
    max-width: 100% !important;
    box-sizing: border-box !important;
  }
  /* Cards: full width, smaller padding */
  .card, [class*="card"], .panel, [class*="panel"] {
    width: 100% !important;
    box-sizing: border-box !important;
    padding: 14px !important;
    margin-bottom: 12px !important;
  }
  /* Tables: scroll horizontally rather than overflow off-screen */
  table { width: 100% !important; display: block !important; overflow-x: auto !important; }
  /* Move ET clock so it doesn't overlap header on mobile */
  #_etclock { top: 8px !important; right: 10px !important; }
  #_ct { font-size: 13px !important; }
  /* Shrink large headings */
  h1 { font-size: 1.3rem !important; }
  h2 { font-size: 1.1rem !important; }
  /* Inputs and buttons: full width */
  input[type="text"], input[type="number"], select, button {
    width: 100% !important;
    box-sizing: border-box !important;
  }
}
</style>
"""

@app.get("/", response_class=HTMLResponse)
def dashboard():
    try:
        html = (Path(__file__).parent / "dashboard" / "index.html").read_text(encoding="utf-8", errors="replace")
    except Exception:
        html = "<html><head></head><body></body></html>"
    # inject before closing body tag (last occurrence only)
    tag = "</body>"
    idx = html.rfind(tag)
    if idx != -1:
        html = html[:idx] + _LOGIN_GATE + _ET_CLOCK + _MOBILE_CSS + html[idx:]
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
    sem = asyncio.Semaphore(3)  # max 3 concurrent yfinance/API calls

    async def _scan(symbol: str):
        async with sem:
            try:
                state = await asyncio.wait_for(build_state(symbol), timeout=60)
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
            except asyncio.TimeoutError:
                return {"symbol": symbol, "error": "timeout", "price": None, "signal": "HOLD", "green_light": False}
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
        result = await asyncio.wait_for(
            loop.run_in_executor(None, run_backtest, symbol, period, capital),
            timeout=50
        )
        return JSONResponse(content=result)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail=f"Backtest timed out — try a shorter period (1y or 6mo)")
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
